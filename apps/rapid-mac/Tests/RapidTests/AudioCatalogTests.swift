import Foundation
import Testing
@testable import Rapid

@Suite("Audio model catalog")
struct AudioCatalogTests {
    static let sample = """
      Available models (1 aliases)
      Alias                 Size       Tools
      qwen3.6-27b-4bit      15.0 GiB   hermes

      Audio models (6 aliases)
      Alias                        Size       Kind       Family        HF id
      kokoro                       338.9 MiB  [audio:tts] kokoro        mlx-community/Kokoro-82M-bf16
      whisper-tiny                 71.0 MiB   [audio:stt] whisper       mlx-community/whisper-tiny-mlx
      qwen3-aligner                1.2 GiB    [audio:stt] qwen3_aligner mlx-community/Qwen3-ForcedAligner-0.6B-8bit
      qwen3-tts-clone              4.2 GiB    [audio:tts] qwen3_tts     mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
      qwen3-tts-4bit               1.1 GiB    [audio:tts] qwen3_tts     mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit
      qwen3-tts-voicedesign-4bit   2.2 GiB    [audio:tts] qwen3_tts     mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit

      Image models (1 aliases)
      flux2-klein-4b               4.3 GiB    [image:gen] Runpod/FLUX.2-klein-4B-mflux-4bit
    """

    @Test("audio adds Dictation without removing either existing workbench")
    @MainActor
    func modeOrder() {
        #expect(AudioViewModel.Mode.allCases == [.dictation, .speech, .transcription])
        let viewModel = AudioViewModel(server: ServerManager(testingState: .idle))
        #expect(viewModel.mode == .dictation)
    }

    @Test("Qwen voice labels and previews follow each speaker's primary language")
    @MainActor
    func voiceLanguageProfiles() {
        #expect(AudioViewModel.voiceDetails(for: "Vivian") == "Chinese · Female")
        #expect(AudioViewModel.voiceDetails(for: "Uncle_Fu") == "Chinese · Male")
        #expect(AudioViewModel.voiceDetails(for: "Dylan") == "Chinese · Beijing · Male")
        #expect(AudioViewModel.voiceDetails(for: "Eric") == "Chinese · Sichuan · Male")
        #expect(AudioViewModel.voiceDetails(for: "Ryan") == "English · Male")
        #expect(AudioViewModel.voiceDetails(for: "Ono_Anna") == "Japanese · Female")
        #expect(AudioViewModel.voiceDetails(for: "Sohee") == "Korean · Female")
        #expect(AudioViewModel.previewText(for: "Ryan").hasPrefix("Hello"))
        #expect(AudioViewModel.previewText(for: "Ono_Anna").hasPrefix("こんにちは"))
        #expect(AudioViewModel.previewText(for: "Sohee").hasPrefix("안녕하세요"))
    }

    @Test("parser extracts audio rows and preserves subtype, family, size, and repo")
    func parsesRows() {
        let rows = ModelCatalog.parseAudioRows(Self.sample)
        #expect(rows.count == 6)
        let kokoro = rows.first { $0.alias == "kokoro" }
        #expect(kokoro?.subtype == "tts")
        #expect(kokoro?.family == "kokoro")
        #expect(kokoro?.size == "338.9 MiB")
        #expect(kokoro?.hfRepo == "mlx-community/Kokoro-82M-bf16")
        #expect(!rows.contains { $0.alias == "qwen3.6-27b-4bit" })
        #expect(!rows.contains { $0.alias == "flux2-klein-4b" })
    }

    @Test("operation classification keeps unsupported audio shapes out of basic pickers")
    func classifiesOperations() {
        #expect(ModelCatalog.audioCapability(
            alias: "whisper-tiny", subtype: "stt", family: "whisper"
        ) == .transcription)
        #expect(ModelCatalog.audioCapability(
            alias: "qwen3-aligner", subtype: "stt", family: "qwen3_aligner"
        ) == .alignment)
        #expect(ModelCatalog.audioCapability(
            alias: "kokoro", subtype: "tts", family: "kokoro"
        ) == .speech)
        #expect(ModelCatalog.audioCapability(
            alias: "qwen3-tts-clone", subtype: "tts", family: "qwen3_tts"
        ) == .voiceCloning)
        #expect(ModelCatalog.audioCapability(
            alias: "qwen3-tts-voicedesign-4bit", subtype: "tts", family: "qwen3_tts"
        ) == .voiceDesign)
    }

    @Test("audio rows remain excluded from the chat catalog")
    func staysOutOfChat() {
        let available = ModelCatalog.parseAvailable(Self.sample).map(\.0)
        #expect(available.contains("qwen3.6-27b-4bit"))
        #expect(!available.contains("kokoro"))
        #expect(!available.contains("whisper-tiny"))
        let excluded = ModelCatalog.parseExcludedAliases(Self.sample)
        #expect(excluded.contains("kokoro"))
        #expect(excluded.contains("whisper-tiny"))
    }

    @Test("speech picker exposes only Qwen3 preset-voice models")
    @MainActor
    func filtersSpeechPickerModels() {
        let viewModel = AudioViewModel(server: ServerManager(testingState: .idle))
        viewModel.audioModels = [
            audioEntry(alias: "qwen3-tts-4bit", capability: .speech, family: "qwen3_tts"),
            audioEntry(alias: "qwen3-tts-clone", capability: .voiceCloning, family: "qwen3_tts"),
            audioEntry(alias: "qwen3-tts-voicedesign-4bit", capability: .voiceDesign, family: "qwen3_tts"),
            audioEntry(alias: "kokoro", capability: .speech, family: "kokoro"),
            audioEntry(alias: "whisper-tiny", capability: .transcription, family: "whisper"),
            audioEntry(alias: "whisper-small", capability: .transcription, family: "whisper"),
        ]

        #expect(viewModel.speechModels.map(\.alias) == ["qwen3-tts-4bit"])
        #expect(viewModel.transcriptionModels.map(\.alias) == ["whisper-small"])
        #expect(!ModelCatalog.isDesktopAudioAliasVisible("whisper-tiny"))
        #expect(ModelCatalog.isDesktopAudioAliasVisible("whisper-small"))
    }

    @Test("audio surface refreshes after a background model download")
    func refreshesOnCacheGeneration() throws {
        let source = try String(contentsOf: Self.audioViewURL, encoding: .utf8)

        #expect(source.contains("@Environment(DownloadManager.self) private var downloads"))
        #expect(source.contains(".task(id: downloads.cacheGeneration)"),
                "The Audio view may stay mounted while Settings downloads a model.")
        #expect(source.contains("WAV, MP3, M4A, AAC, FLAC, MP4, AIFF, or CAF - up to 25 MB"))
        #expect(!source.contains("OGG, Opus, WebM"),
                "The desktop sidecar does not bundle ffmpeg for these formats.")
    }

    @Test("audio model rows use the shared cache icons instead of status suffixes")
    func pickerUsesCacheIcons() throws {
        let source = try String(contentsOf: Self.audioViewURL, encoding: .utf8)

        #expect(source.contains("ModelPickerBar.cacheGlyph(cached: entry.cached)"))
        #expect(source.contains(".map(\\.alias) ?? \"Choose a model\""))
        #expect(!source.contains("private func modelTitle(_ entry: ModelEntry)"),
                "Keep visible audio model labels to the alias; the icon owns cache state.")
    }

    @Test("uncached audio ignores the lazy server's early ready signal")
    @MainActor
    func uncachedAudioNeedsARealDownload() {
        let readiness = AudioView.audioDownloadReadiness(
            alias: "qwen3-tts-4bit",
            cached: false,
            sizeText: "1.1 GiB",
            job: nil,
            activationInFlight: false
        )

        #expect(readiness == .needsDownload(alias: "qwen3-tts-4bit", sizeText: "1.1 GiB"))
        #expect(readiness?.sendAllowed == false)
    }

    @Test("audio download job drives progress and failure instead of false ready")
    @MainActor
    func audioDownloadJobDrivesReadiness() throws {
        let downloads = DownloadManager()
        let alias = "whisper-medium"
        let job = downloads._testingSeedJob(alias: alias)
        downloads._testingIngestStderr(
            alias: alias,
            line: "Fetching 10 files:  30%|███       | 3/10 [00:03<00:07, 1.00it/s]"
        )

        let downloading = AudioView.audioDownloadReadiness(
            alias: alias,
            cached: false,
            sizeText: "1.5 GiB",
            job: job,
            activationInFlight: true
        )
        guard case .downloading(let model, _, let fraction) = downloading else {
            Issue.record("Expected the live pull to own audio readiness")
            return
        }
        #expect(model == alias)
        #expect(fraction == 0.3)

        downloads._testingFinish(alias: alias, status: 1, reason: .exit)
        let failed = AudioView.audioDownloadReadiness(
            alias: alias,
            cached: false,
            sizeText: "1.5 GiB",
            job: job,
            activationInFlight: false
        )
        guard case .failed(let model, _, let action) = failed else {
            Issue.record("Expected a failed pull to offer retry")
            return
        }
        #expect(model == alias)
        #expect(action == .retry(alias: alias))
    }

    @Test("audio activation pulls and verifies the cache before serving")
    func activationOrdersDownloadBeforeServe() throws {
        let source = try String(contentsOf: Self.audioViewURL, encoding: .utf8)
        let pull = try #require(source.range(of: "downloads.startDownload("))
        let wait = try #require(source.range(of: "downloads.awaitDownloadSettlement(alias: alias)"))
        let cacheProof = try #require(source.range(
            of: "guard viewModel.audioModels.first(where: { $0.alias == alias })?.cached == true"
        ))
        let serve = try #require(source.range(of: "_ = await server.ensureServing(", options: .backwards))

        #expect(pull.lowerBound < wait.lowerBound)
        #expect(wait.lowerBound < cacheProof.lowerBound)
        #expect(cacheProof.lowerBound < serve.lowerBound)
    }

    @Test("unmapped audio cache rows match their Hugging Face repo")
    func matchesUnmappedAudioCacheByRepo() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-audio-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("rapid-mlx")
        let script = """
        #!/bin/sh
        if [ "$1" = "models" ]; then
          cat <<'EOF'
          Audio models (1 aliases)
          Alias               Size       Kind        Family      HF id
          qwen3-tts-4bit      2.2 GiB    [audio:tts] qwen3_tts   mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
        EOF
        else
          cat <<'EOF'
          Cached models (1 on disk)
          Alias        HF repo                                              Size
          (unmapped)             mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit 2.2 GiB   29m ago
        EOF
        fi
        """
        try script.write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)

        let entries = await ModelCatalog.audioEntries(binary: binary, hubCacheOverride: nil)
        let qwen = try #require(entries.first { $0.alias == "qwen3-tts-4bit" })

        #expect(qwen.cached)
        #expect(qwen.sizeOnDisk == "2.2 GiB")
    }

    @Test("incomplete audio cache rows remain downloadable")
    @MainActor
    func incompleteAudioCacheIsNotStartable() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-partial-audio-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let binary = directory.appendingPathComponent("rapid-mlx")
        let script = """
        #!/bin/sh
        if [ "$1" = "models" ]; then
          cat <<'EOF'
          Audio models (1 aliases)
          Alias               Size       Kind        Family      HF id
          qwen3-tts-4bit      2.2 GiB    [audio:tts] qwen3_tts   mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit
        EOF
        else
          cat <<'EOF'
          Cached models (1 on disk)
          Alias          HF repo                                              Size
          (incomplete)   mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit   611 MiB
        EOF
        fi
        """
        try script.write(to: binary, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)

        let entries = await ModelCatalog.audioEntries(binary: binary, hubCacheOverride: nil)
        let qwen = try #require(entries.first { $0.alias == "qwen3-tts-4bit" })

        #expect(!qwen.cached)
        #expect(qwen.sizeOnDisk == "2.2 GiB")
        #expect(ModelPickerBar.cacheGlyph(cached: qwen.cached) == "icloud.and.arrow.down")
    }

    private func audioEntry(
        alias: String,
        capability: AudioModelCapability,
        family: String
    ) -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "mlx-community/\(alias)",
            sizeOnDisk: nil,
            cached: false,
            kind: .audio,
            audioCapability: capability,
            audioFamily: family
        )
    }

    private static var audioViewURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // RapidTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
            .appendingPathComponent("Sources/Rapid/UI/AudioView.swift")
    }
}
