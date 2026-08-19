//
//  ContentView.swift
//  NeuTTSNanoApp
//
//  Created by Assistant
//

import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var ttsManager = NeuTTSManager()

    @State private var inputText = "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all."
    @State private var referenceText = "Hello, this is a reference voice sample for voice cloning."
    @State private var showAudioPicker = false
    @State private var referenceAudioURL: URL?
    @State private var generatedAudioData: Data?

    // Token editing state
    @State private var customTokenInput: String = ""
    @State private var isTokenSectionExpanded: Bool = false
    @State private var isLogsExpanded: Bool = false
    @State private var showTokenSavedAlert: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // Status & Download Banner
                    statusHeaderCard

                    // Engine Selector Mode
                    engineSelectorCard

                    // Error & 403 Callout
                    if let error = ttsManager.errorMessage {
                        errorCalloutView(error: error)
                    }

                    // ZETIC API Token Settings Card
                    tokenConfigCard

                    // Text to Synthesize Card
                    textSynthesisCard

                    // Voice Cloning Card (Optional)
                    voiceCloningCard

                    // Action Controls
                    actionButtonsCard

                    // Model Logs Card (Collapsible)
                    logsCard

                    // Footer Info
                    aboutSectionCard
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 32)
            }
            .background(Color(uiColor: .systemGroupedBackground).ignoresSafeArea())
            .navigationTitle("NeuTTS Nano")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        Task {
                            await ttsManager.retryInitialization()
                        }
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(ttsManager.isProcessing)
                }
            }
            .sheet(isPresented: $showAudioPicker) {
                AudioPickerView(selectedURL: $referenceAudioURL)
            }
            .onAppear {
                customTokenInput = ttsManager.tokenKey
                if ttsManager.errorMessage != nil {
                    isTokenSectionExpanded = true
                }
            }
        }
    }

    // MARK: - Subviews

    private var statusHeaderCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Circle()
                    .fill(ttsManager.isInitialized ? Color.green : (ttsManager.errorMessage != nil ? Color.red : Color.orange))
                    .frame(width: 12, height: 12)

                Text(ttsManager.isInitialized ? "Models Ready" : ttsManager.statusMessage)
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .foregroundColor(.primary)
                    .lineLimit(2)

                Spacer()

                if ttsManager.isProcessing {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }

            if !ttsManager.isInitialized && ttsManager.errorMessage == nil {
                ProgressView()
                    .progressViewStyle(.linear)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(14)
        .shadow(color: Color.black.opacity(0.04), radius: 4, x: 0, y: 2)
    }

    private var engineSelectorCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Speech Engine", systemImage: "gearshape.2.fill")
                .font(.subheadline.weight(.semibold))

            Picker("Engine Mode", selection: $ttsManager.useNativeFallbackEngine) {
                Text("Apple Native (Offline)").tag(true)
                Text("ZETIC NPU Neural").tag(false)
            }
            .pickerStyle(.segmented)

            if ttsManager.useNativeFallbackEngine {
                Text("⚡ Using local system speech synthesis with full audio buffer generation. Works 100% offline with zero server dependencies.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } else {
                Text("🧠 Using NeuTTS NPU pipeline with voice cloning via ZETIC MLange.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(14)
        .shadow(color: Color.black.opacity(0.04), radius: 4, x: 0, y: 2)
    }

    private func errorCalloutView(error: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                    .font(.title3)

                VStack(alignment: .leading, spacing: 4) {
                    Text("Download Error")
                        .font(.headline)
                        .foregroundColor(.red)

                    Text(error)
                        .font(.subheadline)
                        .foregroundColor(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Divider()

            HStack {
                Button(action: {
                    isTokenSectionExpanded = true
                }) {
                    Label("Configure ZETIC Token", systemImage: "key.fill")
                        .font(.footnote.weight(.semibold))
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)

                Spacer()

                Button(action: {
                    Task {
                        await ttsManager.retryInitialization()
                    }
                }) {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.footnote)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(14)
        .background(Color.red.opacity(0.08))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.red.opacity(0.25), lineWidth: 1)
        )
        .cornerRadius(14)
    }

    private var tokenConfigCard: some View {
        DisclosureGroup(isExpanded: $isTokenSectionExpanded) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Enter your Personal Access Token from mlange.zetic.ai to authenticate on-device model downloads.")
                    .font(.caption)
                    .foregroundColor(.secondary)

                HStack {
                    Image(systemName: "key")
                        .foregroundColor(.secondary)

                    TextField("ztp_...", text: $customTokenInput)
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)
                        .font(.system(.body, design: .monospaced))

                    if !customTokenInput.isEmpty {
                        Button(action: { customTokenInput = "" }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(10)
                .background(Color(uiColor: .tertiarySystemGroupedBackground))
                .cornerRadius(10)

                HStack(spacing: 10) {
                    Button(action: {
                        Task {
                            await ttsManager.updateTokenKey(customTokenInput)
                            showTokenSavedAlert = true
                        }
                    }) {
                        Label("Save & Download", systemImage: "arrow.down.circle.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(customTokenInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button(action: {
                        if let url = URL(string: "https://mlange.zetic.ai") {
                            UIApplication.shared.open(url)
                        }
                    }) {
                        Label("Get Key", systemImage: "safari")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(.top, 8)
        } label: {
            HStack {
                Label("ZETIC MLange Token", systemImage: "key.fill")
                    .font(.headline)
                    .foregroundColor(.primary)

                Spacer()

                if ttsManager.isInitialized {
                    Text("Configured")
                        .font(.caption2.weight(.bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.15))
                        .foregroundColor(.green)
                        .cornerRadius(6)
                }
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(14)
        .shadow(color: Color.black.opacity(0.04), radius: 4, x: 0, y: 2)
    }

    private var textSynthesisCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Text to Synthesize", systemImage: "text.quote")
                    .font(.headline)

                Spacer()

                Text("\(inputText.count) chars")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            TextEditor(text: $inputText)
                .frame(minHeight: 90, maxHeight: 130)
                .padding(8)
                .background(Color(uiColor: .tertiarySystemGroupedBackground))
                .cornerRadius(10)

            // Preset buttons
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    presetButton(title: "Andy Intro", text: "My name is Andy. I'm 25 and I just moved to London. The underground is pretty confusing, but it gets me around in no time at all.")
                    presetButton(title: "Tech News", text: "NeuTTS Nano runs entirely on-device with zero cloud latency and complete user privacy.")
                    presetButton(title: "Short Quote", text: "The quick brown fox jumps over the lazy dog.")
                }
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(14)
        .shadow(color: Color.black.opacity(0.04), radius: 4, x: 0, y: 2)
    }

    private func presetButton(title: String, text: String) -> some View {
        Button(action: {
            inputText = text
        }) {
            Text(title)
                .font(.caption)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color(uiColor: .tertiarySystemGroupedBackground))
                .foregroundColor(.primary)
                .cornerRadius(8)
        }
    }

    private var voiceCloningCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Voice Cloning (Optional)", systemImage: "person.wave.2.fill")
                .font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                Text("Reference Audio Transcript")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                TextField("Transcript matching the reference audio", text: $referenceText)
                    .padding(8)
                    .background(Color(uiColor: .tertiarySystemGroupedBackground))
                    .cornerRadius(8)
            }

            HStack {
                Button(action: {
                    showAudioPicker = true
                }) {
                    HStack {
                        Image(systemName: referenceAudioURL != nil ? "checkmark.circle.fill" : "waveform.badge.plus")
                        Text(referenceAudioURL != nil ? "Change Audio" : "Select Sample WAV")
                    }
                    .font(.subheadline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.bordered)
                .tint(referenceAudioURL != nil ? .green : .blue)

                if referenceAudioURL != nil {
                    Button(action: {
                        referenceAudioURL = nil
                    }) {
                        Image(systemName: "trash")
                            .foregroundColor(.red)
                    }
                    .buttonStyle(.bordered)
                }
            }

            if let url = referenceAudioURL {
                HStack {
                    Image(systemName: "doc.badge.waveform")
                    Text(url.lastPathComponent)
                        .lineLimit(1)
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(14)
        .shadow(color: Color.black.opacity(0.04), radius: 4, x: 0, y: 2)
    }

    private var actionButtonsCard: some View {
        VStack(spacing: 10) {
            Button(action: {
                Task {
                    await synthesizeSpeech()
                }
            }) {
                HStack(spacing: 8) {
                    if ttsManager.isProcessing {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Image(systemName: "waveform.and.mic")
                    }
                    Text(ttsManager.isProcessing ? "Synthesizing..." : "Generate Speech")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(ttsManager.isInitialized && !ttsManager.isProcessing ? Color.blue : Color.gray.opacity(0.5))
                .foregroundColor(.white)
                .cornerRadius(12)
            }
            .disabled(!ttsManager.isInitialized || ttsManager.isProcessing)

            if let data = generatedAudioData {
                Button(action: {
                    ttsManager.playAudio(data: data)
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: "speaker.wave.3.fill")
                        Text("Play Generated Audio (\(data.count / 1024) KB)")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
            }
        }
    }

    private var logsCard: some View {
        DisclosureGroup(isExpanded: $isLogsExpanded) {
            VStack(alignment: .leading, spacing: 8) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        if ttsManager.logLines.isEmpty {
                            Text("No log messages yet.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        } else {
                            ForEach(Array(ttsManager.logLines.enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                    .padding(8)
                }
                .frame(maxHeight: 180)
                .background(Color(uiColor: .tertiarySystemGroupedBackground))
                .cornerRadius(10)

                HStack {
                    Button(action: {
                        Task {
                            await ttsManager.resetModelCache()
                        }
                    }) {
                        Label("Clear Cache & Redownload", systemImage: "arrow.counterclockwise")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Button(action: {
                        UIPasteboard.general.string = ttsManager.logLines.joined(separator: "\n")
                    }) {
                        Label("Copy Logs", systemImage: "doc.on.doc")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(.top, 8)
        } label: {
            Label("Model & Inference Logs", systemImage: "terminal.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.primary)
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .cornerRadius(14)
        .shadow(color: Color.black.opacity(0.04), radius: 4, x: 0, y: 2)
    }

    private var aboutSectionCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("About NeuTTS Nano", systemImage: "info.circle")
                .font(.footnote.weight(.semibold))
                .foregroundColor(.secondary)

            Text("On-device Voice AI with instant zero-shot voice cloning. Models are compiled to run locally on the Neural Processing Unit (NPU) via ZETIC MLange.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 4)
    }

    // MARK: - Speech Synthesis

    private func synthesizeSpeech() async {
        do {
            let referenceAudioData = try await loadReferenceAudioData()
            let audioData = try await ttsManager.synthesizeSpeech(
                text: inputText,
                referenceAudioData: referenceAudioData,
                referenceText: referenceText.isEmpty ? nil : referenceText
            )

            await MainActor.run {
                generatedAudioData = audioData
                ttsManager.playAudio(data: audioData)
            }
        } catch {
            await MainActor.run {
                ttsManager.errorMessage = "Synthesis failed: \(error.localizedDescription)"
            }
        }
    }

    private func loadReferenceAudioData() async throws -> Data? {
        guard let url = referenceAudioURL else { return nil }

        guard url.startAccessingSecurityScopedResource() else {
            throw NSError(domain: "Audio", code: -2, userInfo: [NSLocalizedDescriptionKey: "Permission denied for selected audio file"])
        }
        defer { url.stopAccessingSecurityScopedResource() }

        let data = try Data(contentsOf: url)
        guard data.count > 44 else {
            throw NSError(domain: "Audio", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid WAV audio file"])
        }
        return data
    }
}

struct AudioPickerView: UIViewControllerRepresentable {
    @Binding var selectedURL: URL?
    @Environment(\.presentationMode) var presentationMode

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.audio])
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: AudioPickerView

        init(_ parent: AudioPickerView) {
            self.parent = parent
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            if let url = urls.first {
                parent.selectedURL = url
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
