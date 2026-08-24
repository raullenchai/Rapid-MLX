import Foundation
import Observation

@MainActor
@Observable
final class AudioViewModel {
    struct TranscriptionModelDetails: Equatable, Sendable {
        let displayName: String
        let badge: String
        let summary: String
        let isRecommended: Bool
    }

    enum Mode: String, CaseIterable, Identifiable {
        case dictation
        case speech

        var id: String { rawValue }
        var label: String {
            switch self {
            case .dictation: return "Speech to Text"
            case .speech: return "Text to Speech"
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
            }
        }
    }

    var mode: Mode = .dictation
    var audioModels: [ModelEntry] = []
    var catalogLoaded = false
    var selectedTranscriptionAlias = ""
    var selectedSpeechAlias = ""

    var speechText = ""
    var voices: [String] = []
    var selectedVoice = ""
    var speed = 1.0
    var synthesizedAudio: SynthesizedAudio?

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
        let candidates = audioModels.filter {
            $0.audioCapability?.supportsTranscription == true
                && ModelCatalog.isDesktopAudioAliasVisible($0.alias)
        }
        return Self.deduplicatedTranscriptionModels(candidates)
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
        isLoadingVoices || isSynthesizing || previewingVoice != nil
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

    func selectSpeechModel(_ alias: String) {
        guard alias != selectedSpeechAlias else { return }
        selectedSpeechAlias = alias
        voices = []
        selectedVoice = ""
        synthesizedAudio = nil
        errorMessage = nil
    }

    func loadVoices() async -> Bool {
        guard !isBusy,
              let entry = speechModels.first(where: { $0.alias == selectedSpeechAlias }) else {
            return false
        }
        isLoadingVoices = true
        errorMessage = nil
        defer { isLoadingVoices = false }
        guard await server.ensureServing(
            alias: entry.alias,
            hfPath: entry.hfRepo,
            residencyEligible: false
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
        guard await server.ensureServing(
            alias: entry.alias,
            hfPath: entry.hfRepo,
            residencyEligible: false
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

        guard await server.ensureServing(
                  alias: entry.alias,
                  hfPath: entry.hfRepo,
                  residencyEligible: false
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

    /// Product-facing guidance for the Speech to Text picker. The engine's
    /// alias catalog is deliberately technical; this layer answers the user's
    /// actual question: which model fits my language and speed/quality needs?
    /// Keep the fallback useful so a newly added engine alias never lands in
    /// the UI as an unexplained name.
    static func transcriptionDetails(
        alias: String,
        family: String?
    ) -> TranscriptionModelDetails {
        let normalized = alias.lowercased()
        switch normalized {
        case "whisper", "whisper-1", "whisper-large-v3":
            return .init(
                displayName: "Whisper Large v3",
                badge: "best quality",
                summary: "Highest-accuracy Whisper model. Supports 99+ languages and difficult accents.",
                isRecommended: false
            )
        case "whisper-large-v3-turbo":
            return .init(
                displayName: "Whisper Large v3 Turbo",
                badge: "balanced",
                summary: "Near Large v3 accuracy with much faster transcription. Supports 99+ languages.",
                isRecommended: true
            )
        case "whisper-medium":
            return .init(
                displayName: "Whisper Medium",
                badge: "multilingual",
                summary: "Good multilingual accuracy with lower memory use than the Large models.",
                isRecommended: false
            )
        case "whisper-small":
            return .init(
                displayName: "Whisper Small",
                badge: "fast",
                summary: "Fast, lightweight multilingual transcription for everyday dictation.",
                isRecommended: false
            )
        case "whisper-base":
            return .init(
                displayName: "Whisper Base",
                badge: "low memory",
                summary: "A compact multilingual model for older Macs. Faster, with lower accuracy.",
                isRecommended: false
            )
        case "parakeet-v3", "parakeet-tdt-0.6b-v3":
            return .init(
                displayName: "Parakeet TDT v3",
                badge: "English",
                summary: "Fast English transcription with improved accuracy over v2 and automatic punctuation.",
                isRecommended: false
            )
        case "parakeet", "parakeet-tdt-0.6b", "parakeet-tdt-0.6b-v2":
            return .init(
                displayName: "Parakeet TDT v2",
                badge: "English",
                summary: "English-only transcription. Very fast with strong punctuation and capitalization.",
                isRecommended: false
            )
        case "qwen3-asr", "qwen3-asr-1.7b":
            return .init(
                displayName: "Qwen3 ASR 1.7B",
                badge: "code-switching",
                summary: "Strong Chinese and English code-switching, punctuation, and custom vocabulary hints.",
                isRecommended: false
            )
        case "qwen3-asr-0.6b":
            return .init(
                displayName: "Qwen3 ASR 0.6B",
                badge: "fast",
                summary: "A smaller, faster Chinese and English model with a modest accuracy tradeoff.",
                isRecommended: false
            )
        case "sensevoice", "sensevoice-small":
            return .init(
                displayName: "SenseVoice Small",
                badge: "Asian languages",
                summary: "Fast Chinese, Cantonese, Japanese, Korean, and English recognition with sound-event tags.",
                isRecommended: false
            )
        default:
            let familyName = family?
                .replacingOccurrences(of: "_", with: " ")
                .capitalized ?? "Speech"
            return .init(
                displayName: alias,
                badge: familyName,
                summary: "Local speech-to-text model. Runs offline after its first download.",
                isRecommended: false
            )
        }
    }

    /// The engine exposes compatibility aliases for API and CLI callers, but
    /// a visual picker should not show the same checkpoint three times. Group
    /// by HF repo and keep the explicit product alias where one exists.
    static func deduplicatedTranscriptionModels(_ entries: [ModelEntry]) -> [ModelEntry] {
        var order: [String] = []
        var representative: [String: ModelEntry] = [:]

        for entry in entries {
            let key = entry.hfRepo?.lowercased() ?? "alias:\(entry.alias.lowercased())"
            guard let current = representative[key] else {
                order.append(key)
                representative[key] = entry
                continue
            }
            if transcriptionAliasPriority(entry.alias)
                < transcriptionAliasPriority(current.alias) {
                representative[key] = entry
            }
        }
        let position = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
        return order.compactMap { representative[$0] }.sorted { lhs, rhs in
            let lhsRecommended = transcriptionDetails(
                alias: lhs.alias, family: lhs.audioFamily
            ).isRecommended
            let rhsRecommended = transcriptionDetails(
                alias: rhs.alias, family: rhs.audioFamily
            ).isRecommended
            if lhsRecommended != rhsRecommended { return lhsRecommended }
            if lhs.cached != rhs.cached { return lhs.cached }
            let lhsKey = lhs.hfRepo?.lowercased() ?? "alias:\(lhs.alias.lowercased())"
            let rhsKey = rhs.hfRepo?.lowercased() ?? "alias:\(rhs.alias.lowercased())"
            return position[lhsKey, default: .max] < position[rhsKey, default: .max]
        }
    }

    private static func transcriptionAliasPriority(_ alias: String) -> Int {
        switch alias.lowercased() {
        case "whisper-large-v3", "parakeet", "parakeet-v3", "qwen3-asr", "sensevoice":
            return 0
        case "whisper", "whisper-1", "parakeet-tdt-0.6b",
             "parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v3",
             "qwen3-asr-1.7b", "sensevoice-small":
            return 2
        default:
            return 1
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

    func wouldReplaceServingModel(alias: String) -> String? {
        guard let serving = server.servingAlias, serving != alias else { return nil }
        return serving
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
