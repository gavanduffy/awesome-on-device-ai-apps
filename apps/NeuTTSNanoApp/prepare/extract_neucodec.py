#!/usr/bin/env python3
"""
Script to extract and export neuphonic/neucodec models for Melange (PT2 / TorchScript).

Exports:
1. neucodec_decoder: FSQ codes [1, 1, 50] (int64) -> 24kHz audio [1, 1, 24000] (float32)
2. neucodec_encoder: 16kHz audio [1, 1, 16000] (float32) -> acoustic embeddings
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from neucodec.codec_decoder_vocos import CodecDecoderVocos
from neucodec.codec_encoder import CodecEncoder


def main():
    model_zoo_root = os.environ.get("MODEL_ZOO_DIR", "/root/model_zoo")
    ckpt_path = "/root/.cache/huggingface/hub/models--neuphonic--neucodec/snapshots/30c1fdd19e68aee65d542cf043750d4c0165893e/pytorch_model.bin"

    print(f"[INFO] Using model zoo root: {model_zoo_root}", flush=True)
    print(f"[INFO] Loading NeuCodec checkpoint from {ckpt_path}...", flush=True)

    state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
    print(f"[INFO] ✓ State dict loaded ({len(state_dict)} keys).", flush=True)

    # ---------------------------------------------------------
    # 1. NeuCodec Decoder
    # ---------------------------------------------------------
    print("\n[INFO] --- 1. Exporting NeuCodec Decoder ---", flush=True)
    decoder_base = os.path.join(model_zoo_root, "neucodec_decoder")
    decoder_model_dir = os.path.join(decoder_base, "model")
    decoder_inputs_dir = os.path.join(decoder_base, "inputs")
    os.makedirs(decoder_model_dir, exist_ok=True)
    os.makedirs(decoder_inputs_dir, exist_ok=True)

    generator = CodecDecoderVocos(hop_length=480)
    fc_post_a = nn.Linear(2048, 1024)

    gen_weights = {k.replace("generator.", ""): v for k, v in state_dict.items() if k.startswith("generator.")}
    fc_weights = {k.replace("fc_post_a.", ""): v for k, v in state_dict.items() if k.startswith("fc_post_a.")}

    generator.load_state_dict(gen_weights, strict=False)
    fc_post_a.load_state_dict(fc_weights, strict=True)

    # Replace head with real-valued ISTFT for seamless ONNX / cross-backend export
    class RealValuedISTFTHead(nn.Module):
        def __init__(self, original_head):
            super().__init__()
            self.out = original_head.out
            self.n_fft = original_head.istft.n_fft
            self.hop_length = original_head.istft.hop_length
            self.win_length = original_head.istft.win_length
            self.padding = original_head.istft.padding
            self.register_buffer("window", original_head.istft.window)

            N = self.n_fft // 2 + 1
            n = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(1)
            k = torch.arange(N, dtype=torch.float32).unsqueeze(0)
            angles = 2 * np.pi * n * k / self.n_fft

            weights = torch.full((1, N), 2.0, dtype=torch.float32)
            weights[0, 0] = 1.0
            weights[0, -1] = 1.0

            cos_kernel = (torch.cos(angles) * weights) / self.n_fft
            sin_kernel = (-torch.sin(angles) * weights) / self.n_fft
            self.register_buffer("cos_kernel", cos_kernel)
            self.register_buffer("sin_kernel", sin_kernel)

        def forward(self, x: torch.Tensor):
            x_pred = self.out(x)
            x_pred = x_pred.transpose(1, 2)
            mag, p = x_pred.chunk(2, dim=1)
            mag = torch.exp(mag)
            mag = torch.clip(mag, max=1e2)
            cos_p = torch.cos(p)
            sin_p = torch.sin(p)
            real = mag * cos_p
            imag = mag * sin_p

            pad = (self.win_length - self.hop_length) // 2
            B, N, T = real.shape

            real_t = real.permute(0, 2, 1)
            imag_t = imag.permute(0, 2, 1)

            ifft = torch.matmul(real_t, self.cos_kernel.t()) + torch.matmul(imag_t, self.sin_kernel.t())
            ifft = ifft.permute(0, 2, 1) * self.window[None, :, None]

            output_size = (T - 1) * self.hop_length + self.win_length
            y = torch.nn.functional.fold(
                ifft,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length)
            )[:, 0, 0, pad:-pad]

            window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
            window_envelope = torch.nn.functional.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length)
            ).squeeze()[pad:-pad]

            y = y / window_envelope
            return y.unsqueeze(1), x_pred

    generator.head = RealValuedISTFTHead(generator.head)

    class StandaloneNeuCodecDecoder(nn.Module):
        def __init__(self, gen, fc):
            super().__init__()
            self.quantizer = gen.quantizer
            self.fc_post_a = fc
            self.generator = gen

        def forward(self, fsq_codes):
            fsq_codes = fsq_codes.long()
            fsq_post_emb = self.quantizer.get_output_from_indices(fsq_codes.transpose(1, 2))
            fsq_post_emb = fsq_post_emb.transpose(1, 2)
            fsq_post_emb = self.fc_post_a(fsq_post_emb.transpose(1, 2)).transpose(1, 2)
            recon = self.generator(fsq_post_emb.transpose(1, 2), vq=False)[0]
            return recon

    decoder = StandaloneNeuCodecDecoder(generator, fc_post_a).eval()
    sample_codes = torch.randint(0, 1024, (1, 1, 50), dtype=torch.int64)

    # Save inputs
    codes_npy_path = os.path.join(decoder_inputs_dir, "fsq_codes.npy")
    np.save(codes_npy_path, sample_codes.numpy().astype(np.int64))
    print(f"[INFO] ✓ Decoder input saved to {codes_npy_path} (shape: {sample_codes.shape})", flush=True)

    # Export PT2
    decoder_pt2_path = os.path.join(decoder_model_dir, "neucodec_decoder.pt2")
    print(f"[INFO] Exporting decoder to PT2 (.pt2)...", flush=True)
    try:
        with torch.no_grad():
            exp_dec = torch.export.export(decoder, (sample_codes,))
            torch.export.save(exp_dec, decoder_pt2_path)
            size_mb = os.path.getsize(decoder_pt2_path) / (1024 * 1024)
            print(f"[INFO] ✓ PT2 decoder saved to {decoder_pt2_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[WARNING] Decoder PT2 export error: {e}", flush=True)

    # Export TorchScript (.pt)
    decoder_pt_path = os.path.join(decoder_model_dir, "neucodec_decoder.pt")
    print(f"[INFO] Tracing decoder to TorchScript (.pt)...", flush=True)
    try:
        with torch.no_grad():
            traced_dec = torch.jit.trace(decoder, sample_codes, strict=False)
            torch.jit.save(traced_dec, decoder_pt_path)
            size_mb = os.path.getsize(decoder_pt_path) / (1024 * 1024)
            print(f"[INFO] ✓ TorchScript decoder saved to {decoder_pt_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[ERROR] Decoder TorchScript error: {e}", flush=True)

    # Export ONNX (.onnx)
    decoder_onnx_path = os.path.join(decoder_model_dir, "neucodec_decoder.onnx")
    print(f"[INFO] Exporting decoder to ONNX (.onnx)...", flush=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                (sample_codes,),
                decoder_onnx_path,
                input_names=["fsq_codes"],
                output_names=["audio"],
                dynamo=True
            )
            size_mb = os.path.getsize(decoder_onnx_path) / (1024 * 1024)
            print(f"[INFO] ✓ ONNX decoder saved to {decoder_onnx_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[ERROR] Decoder ONNX error: {e}", flush=True)

    # ---------------------------------------------------------
    # 2. NeuCodec Encoder
    # ---------------------------------------------------------
    print("\n[INFO] --- 2. Exporting NeuCodec Encoder ---", flush=True)
    encoder_base = os.path.join(model_zoo_root, "neucodec_encoder")
    encoder_model_dir = os.path.join(encoder_base, "model")
    encoder_inputs_dir = os.path.join(encoder_base, "inputs")
    os.makedirs(encoder_model_dir, exist_ok=True)
    os.makedirs(encoder_inputs_dir, exist_ok=True)

    codec_enc = CodecEncoder()
    enc_weights = {k.replace("CodecEnc.", ""): v for k, v in state_dict.items() if k.startswith("CodecEnc.")}
    codec_enc.load_state_dict(enc_weights, strict=True)

    class StandaloneNeuCodecEncoder(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.CodecEnc = enc

        def forward(self, audio):
            return self.CodecEnc(audio)

    encoder = StandaloneNeuCodecEncoder(codec_enc).eval()
    sample_audio = torch.randn(1, 1, 16000, dtype=torch.float32)

    # Save inputs
    audio_npy_path = os.path.join(encoder_inputs_dir, "audio.npy")
    np.save(audio_npy_path, sample_audio.numpy().astype(np.float32))
    print(f"[INFO] ✓ Encoder input saved to {audio_npy_path} (shape: {sample_audio.shape})", flush=True)

    # Export PT2
    encoder_pt2_path = os.path.join(encoder_model_dir, "neucodec_encoder.pt2")
    print(f"[INFO] Exporting encoder to PT2 (.pt2)...", flush=True)
    try:
        with torch.no_grad():
            exp_enc = torch.export.export(encoder, (sample_audio,))
            torch.export.save(exp_enc, encoder_pt2_path)
            size_mb = os.path.getsize(encoder_pt2_path) / (1024 * 1024)
            print(f"[INFO] ✓ PT2 encoder saved to {encoder_pt2_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[WARNING] Encoder PT2 export error: {e}", flush=True)

    # Export TorchScript (.pt)
    encoder_pt_path = os.path.join(encoder_model_dir, "neucodec_encoder.pt")
    print(f"[INFO] Tracing encoder to TorchScript (.pt)...", flush=True)
    try:
        with torch.no_grad():
            traced_enc = torch.jit.trace(encoder, sample_audio, strict=False)
            torch.jit.save(traced_enc, encoder_pt_path)
            size_mb = os.path.getsize(encoder_pt_path) / (1024 * 1024)
            print(f"[INFO] ✓ TorchScript encoder saved to {encoder_pt_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[ERROR] Encoder TorchScript error: {e}", flush=True)

    # Export ONNX (.onnx)
    encoder_onnx_path = os.path.join(encoder_model_dir, "neucodec_encoder.onnx")
    print(f"[INFO] Exporting encoder to ONNX (.onnx)...", flush=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                encoder,
                (sample_audio,),
                encoder_onnx_path,
                input_names=["audio"],
                output_names=["codes"],
                dynamo=True
            )
            size_mb = os.path.getsize(encoder_onnx_path) / (1024 * 1024)
            print(f"[INFO] ✓ ONNX encoder saved to {encoder_onnx_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[ERROR] Encoder ONNX error: {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("ALL NEUCODEC MODELS EXPORTED FOR MELANGE", flush=True)
    print(f"Decoder PT2 (.pt2):   {decoder_pt2_path}", flush=True)
    print(f"Decoder PT (.pt):     {decoder_pt_path}", flush=True)
    print(f"Decoder ONNX (.onnx): {decoder_onnx_path}", flush=True)
    print(f"Encoder PT2 (.pt2):   {encoder_pt2_path}", flush=True)
    print(f"Encoder PT (.pt):     {encoder_pt_path}", flush=True)
    print(f"Encoder ONNX (.onnx): {encoder_onnx_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
