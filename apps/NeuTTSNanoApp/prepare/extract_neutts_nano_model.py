#!/usr/bin/env python3
"""
Script to export neuphonic/neutts-nano model for Melange (PT2 / TorchScript / ONNX).
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
from tokenizers import Tokenizer


class NeuttsNanoWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        return outputs.logits


def main():
    project_name = "neutts_nano"
    model_zoo_root = os.environ.get("MODEL_ZOO_DIR", "/root/model_zoo")
    base_dir = os.path.join(model_zoo_root, project_name)
    model_dir = os.path.join(base_dir, "model")
    inputs_dir = os.path.join(base_dir, "inputs")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)

    print(f"[INFO] Base directory: {base_dir}", flush=True)
    print(f"[INFO] Model directory: {model_dir}", flush=True)
    print(f"[INFO] Inputs directory: {inputs_dir}", flush=True)

    # 1. Load Local Snapshot or HF Hub
    snapshot_dir = "/root/.cache/huggingface/hub/models--neuphonic--neutts-nano/snapshots/94c32e783cb1d00097a85fd3e5b12db90f9f3fb0"
    config_path = os.path.join(snapshot_dir, "config.json")
    weights_path = os.path.join(snapshot_dir, "model.safetensors")
    tokenizer_path = os.path.join(snapshot_dir, "tokenizer.json")

    print(f"[INFO] Loading configuration from {config_path}...", flush=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg_dict["_attn_implementation"] = "eager"

    config = LlamaConfig.from_dict(cfg_dict)
    print("[INFO] ✓ Config loaded.", flush=True)

    print("[INFO] Instantiating LlamaForCausalLM...", flush=True)
    model = LlamaForCausalLM(config)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[INFO] ✓ Model weights loaded from safetensors.", flush=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("[INFO] ✓ Tokenizer loaded.", flush=True)

    # 2. Prepare Sample Inputs (Static Shapes: [1, 128])
    print("\n[INFO] Preparing input tokens...", flush=True)
    sample_text = "The quick brown fox jumps over the lazy dog."
    FIXED_SEQ_LEN = 128

    enc = tokenizer.encode(sample_text)
    ids = list(enc.ids)[:FIXED_SEQ_LEN]
    mask = [1] * len(ids)
    while len(ids) < FIXED_SEQ_LEN:
        ids.append(128001)
        mask.append(0)

    input_ids = torch.tensor([ids], dtype=torch.int64)
    attention_mask = torch.tensor([mask], dtype=torch.int64)

    print(f"[INFO] Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}", flush=True)

    # Save inputs as .npy (Int32 & Int64)
    input_ids_path = os.path.join(inputs_dir, "input_ids.npy")
    attention_mask_path = os.path.join(inputs_dir, "attention_mask.npy")

    np.save(input_ids_path, input_ids.numpy().astype(np.int32))
    np.save(attention_mask_path, attention_mask.numpy().astype(np.int32))
    print(f"[INFO] ✓ Inputs saved to {inputs_dir}", flush=True)

    wrapped_model = NeuttsNanoWrapper(model).eval()

    # 3. Forward Pass Verification
    print("\n[INFO] Verifying forward pass...", flush=True)
    with torch.no_grad():
        out = wrapped_model(input_ids, attention_mask)
        print(f"[INFO] ✓ Forward pass logits shape: {out.shape}", flush=True)

    # 4. Export to PT2 (torch.export)
    pt2_path = os.path.join(model_dir, f"{project_name}.pt2")
    print(f"\n[INFO] Exporting to PyTorch 2.x (.pt2) via torch.export...", flush=True)
    try:
        with torch.no_grad():
            exported_program = torch.export.export(wrapped_model, (input_ids, attention_mask))
            torch.export.save(exported_program, pt2_path)
            size_mb = os.path.getsize(pt2_path) / (1024 * 1024)
            print(f"[INFO] ✓ PT2 model saved to {pt2_path} ({size_mb:.1f} MB)", flush=True)

            # Reload check
            loaded_prog = torch.export.load(pt2_path)
            test_out = loaded_prog.module()(input_ids, attention_mask)
            print(f"[INFO] ✓ PT2 verification successful. Output shape: {test_out.shape}", flush=True)
    except Exception as e:
        print(f"[WARNING] torch.export error: {e}", flush=True)

    # 5. Export to TorchScript (.pt)
    pt_path = os.path.join(model_dir, f"{project_name}.pt")
    print(f"\n[INFO] Tracing model to TorchScript (.pt)...", flush=True)
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask), strict=False)
            torch.jit.save(traced_model, pt_path)
            size_mb = os.path.getsize(pt_path) / (1024 * 1024)
            print(f"[INFO] ✓ TorchScript model saved to {pt_path} ({size_mb:.1f} MB)", flush=True)

            # Reload check
            test_model = torch.jit.load(pt_path)
            test_output = test_model(input_ids, attention_mask)
            print(f"[INFO] ✓ TorchScript verification successful. Output shape: {test_output.shape}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to trace model: {e}", flush=True)

    # 6. Export to ONNX (.onnx)
    onnx_path = os.path.join(model_dir, f"{project_name}.onnx")
    print(f"\n[INFO] Exporting model to ONNX (.onnx)...", flush=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                (input_ids, attention_mask),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamo=True
            )
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"[INFO] ✓ ONNX model saved to {onnx_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to export ONNX model: {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("CONVERSION COMPLETE: READY FOR MELANGE UPLOAD", flush=True)
    print(f"PT2 Model (.pt2):   {pt2_path}", flush=True)
    print(f"TorchScript (.pt):  {pt_path}", flush=True)
    print(f"ONNX Model (.onnx): {onnx_path}", flush=True)
    print(f"Sample Inputs:      {input_ids_path}, {attention_mask_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
