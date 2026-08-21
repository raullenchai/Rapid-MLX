import Foundation
import Observation

@MainActor
@Observable
final class AudioViewModel {
    enum Mode: String, CaseIterable, Identifiable {
        case dictation
        case speech
        case transcription

        var id: String { rawValue }
        var label: String {
            switch self {
            case .dictation: return "Dictation"
            case .transcription: return "Transcription"
            case .speech: return "Speech"
            }
        }

        /// The AX identifier suffix, kept separate from ``label`` so the
        /// harness never has to quote a control name: the golden flows address
        /// controls by identifier as bare shell words, and the labels now carry
        /// spaces ("Speech to Text").
        var axName: String {
            switch self {
            case .dictation: return "Dictation"
            case .speech: return "Speech"
            case .transcription: return "Transcription"
            }
        }
    }

    var mode: Mode = .dictation
    var audioModels: [ModelEntry] = []
    var catalogLoaded = false
    var selectedTranscriptionAlias = ""
    var selectedSpeechAlias = ""

    var selectedFileURL: URL?
    var transcription: AudioTranscriptionResult?
    var speechText = ""
    var voices: [String] = []
    var selectedVoice = ""
    var speed = 1.0
    var synthesizedAudio: SynthesizedAudio?

    var isTranscribing = false
    var isLoadingVoices = false
    var isSynthesizing = false
    var previewingVoice: String?
    var errorMessage: String?

    private let server: ServerManager
    private let client: AudioClient

    init(server: ServerManager, client: AudioClient = AudioClient()) {
        self.server = server
        self.client = client
    }

    var transcriptionModels: [ModelEntry] {
        audioModels.filter {
            $0.audioCapability?.supportsTranscription == true
                && ModelCatalog.isDesktopAudioAliasVisible($0.alias)
        }
    }

    var speechModels: [ModelEntry] {
        // The signed desktop sidecar intentionally bundles the compact MLX
        // audio core, not Kokoro's spaCy/espeak language stack or F5 cloning.
        // Qwen3 CustomVoice is reference-free, has real named speakers, and
        // runs entirely on dependencies in the desktop audio extra.
        audioModels.filter {
            $0.audioCapability?.supportsPresetSpeech == true
                && $0.audioFamily == "qwen3_tts"
        }
    }

    var isBusy: Bool {
        isTranscribing || isLoadingVoices || isSynthesizing || previewingVoice != nil
    }

    func refreshCatalog() async {
        guard let binary = server.binaryPath else {
            audioModels = []
            catalogLoaded = true
            return
        }
        audioModels = await ModelCatalog.audioEntries(binary: binary)
        catalogLoaded = true
        resolveSelections()
    }

    func selectFile(_ url: URL) {
        selectedFileURL = url
        transcription = nil
        errorMessage = nil
    }

    func selectSpeechModel(_ alias: String) {
        guard alias != selectedSpeechAlias else { return }
        selectedSpeechAlias = alias
        voices = []
        selectedVoice = ""
        synthesizedAudio = nil
        errorMessage = nil
    }

    func transcribe() async {
        guard !isBusy,
              let fileURL = selectedFileURL,
              let entry = transcriptionModels.first(where: {
                  $0.alias == selectedTranscriptionAlias
              }) else { return }
        isTranscribing = true
        errorMessage = nil
        transcription = nil
        defer { isTranscribing = false }
        guard await ensureVoiceLane(
            alias: entry.alias,
            hfPath: entry.hfRepo
        ) else {
            errorMessage = "The audio model couldn't start. Audio support may be unavailable in this app build."
            return
        }
        do {
            transcription = try await client.transcribe(
                fileURL: fileURL,
                model: entry.alias,
                port: server.activePort,
                bearer: server.activeBearer
            )
        } catch {
            errorMessage = Self.message(for: error)
        }
    }

    func loadVoices() async -> Bool {
        guard !isBusy,
              let entry = speechModels.first(where: { $0.alias == selectedSpeechAlias }) else {
            return false
        }
        isLoadingVoices = true
        errorMessage = nil
        defer { isLoadingVoices = false }
        guard await ensureVoiceLane(
            alias: entry.alias,
            hfPath: entry.hfRepo
        ) else {
            errorMessage = "The speech model couldn't start. Audio support may be unavailable in this app build."
            return false
        }
        do {
            let loaded = try await client.voices(
                model: entry.alias,
                port: server.activePort,
                bearer: server.activeBearer
            )
            guard !loaded.isEmpty else {
                errorMessage = "This model did not report any available voices."
                return false
            }
            voices = loaded
            if !loaded.contains(selectedVoice) { selectedVoice = loaded[0] }
            return true
        } catch {
            errorMessage = Self.message(for: error)
            return false
        }
    }

    func synthesize() async {
        let trimmed = speechText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isBusy, !trimmed.isEmpty,
              let entry = speechModels.first(where: { $0.alias == selectedSpeechAlias }) else {
            return
        }
        if voices.isEmpty {
            guard await loadVoices() else { return }
        }
        guard !selectedVoice.isEmpty else { return }

        isSynthesizing = true
        errorMessage = nil
        synthesizedAudio = nil
        defer { isSynthesizing = false }
        guard await ensureVoiceLane(
            alias: entry.alias,
            hfPath: entry.hfRepo
        ) else {
            errorMessage = "The speech model is no longer running. Load its voices and try again."
            return
        }
        do {
            synthesizedAudio = try await client.synthesize(
                text: trimmed,
                model: entry.alias,
                voice: selectedVoice,
                speed: speed,
                port: server.activePort,
                bearer: server.activeBearer
            )
        } catch {
            errorMessage = Self.message(for: error)
        }
    }

    func previewVoice(_ voice: String) async -> SynthesizedAudio? {
        guard !isBusy, voices.contains(voice),
              let entry = speechModels.first(where: { $0.alias == selectedSpeechAlias }) else {
            return nil
        }

        previewingVoice = voice
        errorMessage = nil
        defer { previewingVoice = nil }

        guard await ensureVoiceLane(
                  alias: entry.alias,
                  hfPath: entry.hfRepo
              ),
              !Task.isCancelled else {
            if !Task.isCancelled {
                errorMessage = "The speech model is no longer running. Load its voices and try again."
            }
            return nil
        }

        do {
            return try await client.synthesize(
                text: Self.previewText(for: voice),
                model: entry.alias,
                voice: voice,
                speed: speed,
                port: server.activePort,
                bearer: server.activeBearer
            )
        } catch {
            if !Task.isCancelled {
                errorMessage = Self.message(for: error)
            }
            return nil
        }
    }

    static func voiceDetails(for voice: String) -> String {
        switch voice.lowercased() {
        case "vivian", "serena": return "Chinese · Female"
        case "uncle_fu": return "Chinese · Male"
        case "dylan": return "Chinese · Beijing · Male"
        case "eric": return "Chinese · Sichuan · Male"
        case "ryan", "aiden": return "English · Male"
        case "ono_anna": return "Japanese · Female"
        case "sohee": return "Korean · Female"
        default: return "Multilingual"
        }
    }

    static func previewText(for voice: String) -> String {
        switch voice.lowercased() {
        case "vivian", "serena", "uncle_fu", "dylan", "eric":
            return "你好，这是我的声音，很高兴认识你。"
        case "ono_anna":
            return "こんにちは、私の声を聞いてください。"
        case "sohee":
            return "안녕하세요, 제 목소리를 들어 보세요."
        default:
            return "Hello, this is a preview of my voice."
        }
    }

    /// Ensure a server is reachable for a voice (STT/TTS) request, reusing the
    /// app's primary LLM/VLM process so speech co-loads alongside the chat
    /// model instead of replacing it.
    ///
    /// See ``ServerManager.ensureVoiceLane`` for the co-load-or-fallback
    /// contract. Returns false only when no server can be brought up at all
    /// (e.g. the voice model's weights couldn't start), mirroring the old swap
    /// contract so call sites can keep their error messaging.
    func ensureVoiceLane(alias: String, hfPath: String?) async -> Bool {
        await server.ensureVoiceLane(alias: alias, hfPath: hfPath)
    }

    private func resolveSelections() {
        if !transcriptionModels.contains(where: { $0.alias == selectedTranscriptionAlias }) {
            selectedTranscriptionAlias = preferredAlias(
                from: transcriptionModels,
                preferred: ["whisper-small", "whisper-large-v3-turbo", "whisper-large-v3"]
            )
        }
        if !speechModels.contains(where: { $0.alias == selectedSpeechAlias }) {
            selectedSpeechAlias = preferredAlias(
                from: speechModels,
                preferred: ["qwen3-tts-4bit", "qwen3-tts-6bit", "qwen3-tts"]
            )
        }
    }

    private func preferredAlias(from entries: [ModelEntry], preferred: [String]) -> String {
        if let cached = entries.first(where: { $0.cached }) { return cached.alias }
        for alias in preferred where entries.contains(where: { $0.alias == alias }) { return alias }
        return entries.first?.alias ?? ""
    }

    private static func message(for error: Error) -> String {
        if let localized = error as? LocalizedError, let message = localized.errorDescription {
            return message
        }
        return error.localizedDescription
    }
}
