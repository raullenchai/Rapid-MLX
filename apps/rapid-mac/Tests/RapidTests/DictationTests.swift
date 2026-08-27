import Foundation
import Testing

@testable import Rapid

/// Unit cover for the parts of dictation that are pure logic. The hotkey tap,
/// the audio engine and the paste all need a real session to exercise, but the
/// text handling, the vocabulary budget and the on-disk formats do not — and
/// those are where the bugs that survive a manual smoke test live.
@Suite("Dictation")
struct DictationTests {

    @MainActor
    private final class CatalogState {
        var cached = true
    }

    // MARK: - Transcript tidying

    /// The trailing period is stripped because a dictated fragment usually
    /// lands mid-sentence. Doing it with a byte-wise character class would
    /// truncate the 3-byte `。` to a lone `E3 80` and emit mojibake, which is
    /// exactly the bug this pins.
    @Test("tidy strips one trailing stop without corrupting multi-byte text")
    func tidyTrailingStop() {
        #expect(DictationController.tidy("先跑起来再说。") == "先跑起来再说")
        #expect(DictationController.tidy("ship it.") == "ship it")
        #expect(DictationController.tidy("  spaced out  ") == "spaced out")
        // Every scalar must survive; a truncated UTF-8 sequence would not
        // round-trip through String.
        let chinese = DictationController.tidy("你好世界。")
        #expect(chinese == "你好世界")
        #expect(chinese.unicodeScalars.allSatisfy { $0.value != 0xFFFD })
    }

    @Test("tidy leaves an ellipsis alone")
    func tidyEllipsis() {
        #expect(DictationController.tidy("wait for it...") == "wait for it...")
    }

    @Test("tidy on an empty or punctuation-only string stays safe")
    func tidyDegenerate() {
        #expect(DictationController.tidy("") == "")
        #expect(DictationController.tidy("   ") == "")
        #expect(DictationController.tidy("。") == "")
    }

    // MARK: - Warm-up probe

    /// The prewarm probe is a hand-assembled WAV; a malformed header would
    /// turn every prewarm into a silent 400 and the lazy weight-load would
    /// quietly return to the user's first real dictation. Pins the exact
    /// layout ``DictationRecorder`` produces: PCM16, mono, 16 kHz, with RIFF
    /// and data sizes that match the payload.
    @MainActor
    @Test("the silent prewarm probe is a valid 16 kHz mono PCM16 WAV")
    func silentProbeShape() {
        let wav = DictationController.silentProbeWAV
        func u32(_ offset: Int) -> UInt32 {
            wav.subdata(in: offset..<(offset + 4)).withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
            }
        }
        func u16(_ offset: Int) -> UInt16 {
            wav.subdata(in: offset..<(offset + 2)).withUnsafeBytes {
                UInt16(littleEndian: $0.loadUnaligned(as: UInt16.self))
            }
        }
        #expect(String(data: wav.prefix(4), encoding: .ascii) == "RIFF")
        #expect(String(data: wav.subdata(in: 8..<16), encoding: .ascii) == "WAVEfmt ")
        #expect(String(data: wav.subdata(in: 36..<40), encoding: .ascii) == "data")
        #expect(u16(20) == 1)              // PCM
        #expect(u16(22) == 1)              // mono
        #expect(u32(24) == 16_000)         // sample rate
        #expect(u32(28) == 32_000)         // byte rate
        #expect(u16(32) == 2)              // block align
        #expect(u16(34) == 16)             // bits per sample
        let payload = u32(40)
        #expect(u32(4) == 36 + payload)    // RIFF size ties to data size
        #expect(Int(payload) == wav.count - 44)
        // A beat of silence: long enough to force the weight load, short
        // enough to be free.
        let seconds = Double(payload) / 32_000
        #expect(seconds > 0.05 && seconds < 1.0)
        #expect(wav.suffix(Int(payload)).allSatisfy { $0 == 0 })
    }

    // MARK: - Vocabulary

    /// The cap is the whole design constraint: measured accuracy falls off past
    /// roughly twenty terms, so the prompt must never carry more than that no
    /// matter how many the user has saved.
    @MainActor
    @Test("vocabulary never sends more than the active limit")
    func vocabularyBudget() throws {
        let vocabulary = DictationVocabulary(storeURL: Self.tempStore())
        for index in 0..<(DictationVocabulary.activeLimit + 8) {
            vocabulary.add("term\(index)")
        }
        #expect(vocabulary.terms.count == DictationVocabulary.activeLimit + 8)
        #expect(vocabulary.activeTerms.count == DictationVocabulary.activeLimit)
        #expect(vocabulary.isOverBudget)
    }

    @MainActor
    @Test("an empty vocabulary sends no hint at all")
    func vocabularyEmptyPrompt() {
        let vocabulary = DictationVocabulary(storeURL: Self.tempStore())
        // An empty prompt is not free — it still occupies decoder attention —
        // so the field has to be omitted rather than sent blank.
        #expect(vocabulary.contextPrompt.isEmpty)
    }

    @MainActor
    @Test("parked terms are kept but not sent")
    func vocabularyParking() {
        let vocabulary = DictationVocabulary(storeURL: Self.tempStore())
        vocabulary.add("herdr")
        vocabulary.add("spark1")
        vocabulary.setActive("spark1", false)
        #expect(vocabulary.terms.count == 2)
        #expect(vocabulary.activeTerms.map(\.text) == ["herdr"])
        #expect(vocabulary.contextPrompt.contains("herdr"))
        #expect(!vocabulary.contextPrompt.contains("spark1"))
    }

    /// A correction is the strongest signal a term matters, so it must both
    /// reactivate the term and move it inside the budget.
    @MainActor
    @Test("correcting to a parked term reactivates and promotes it")
    func vocabularyCorrectionPromotes() {
        let vocabulary = DictationVocabulary(storeURL: Self.tempStore())
        vocabulary.add("zzz")
        vocabulary.add("herdr")
        vocabulary.setActive("herdr", false)
        vocabulary.noteCorrection(to: "herdr")

        let promoted = vocabulary.terms.first
        #expect(promoted?.text == "herdr")
        #expect(promoted?.isActive == true)
        #expect(promoted?.corrections == 1)
    }

    @MainActor
    @Test("adding a term twice does not duplicate it")
    func vocabularyDeduplicates() {
        let vocabulary = DictationVocabulary(storeURL: Self.tempStore())
        vocabulary.add("herdr")
        vocabulary.add("herdr")
        #expect(vocabulary.terms.count == 1)
    }

    @MainActor
    @Test("a quick add then remove cannot resurrect the older snapshot")
    func vocabularyPersistencePreservesMutationOrder() async {
        let store = Self.tempStore()
        let vocabulary = DictationVocabulary(storeURL: store)
        vocabulary.add("GoldenTerm2049")
        vocabulary.remove("GoldenTerm2049")
        await vocabulary.waitForPersistence()

        let reloaded = DictationVocabulary(storeURL: store)
        #expect(reloaded.terms.isEmpty)
    }

    /// Suggestions exist because nobody maintains a word list by hand. The
    /// filter has to keep invented names — the ones ASR actually gets wrong —
    /// and drop ordinary words a model already knows, since every kept term
    /// spends part of a hard budget.
    @Test(
        "proper-noun heuristic keeps invented names and drops dictionary words",
        arguments: [
            ("herdr", true),          // not a word
            ("spark1", true),         // digit
            ("ds-0731", true),        // digits + hyphen
            ("vLLM", true),           // interior capitals
            ("BambuStudio", true),    // camel case
            ("Documents", false),     // common folder
            ("downloads", false),
            ("build", false),
            ("no", false),            // too short
        ]
    )
    func properNounHeuristic(name: String, expected: Bool) {
        #expect(DictationVocabulary.isLikelyProperNoun(name) == expected)
    }

    // MARK: - History

    /// `save()` encodes dates as ISO-8601. A decoder left on the default
    /// strategy silently fails the whole array, which reads as "history is
    /// empty" on every launch rather than as an error.
    @MainActor
    @Test("history survives a save/load round-trip")
    func historyRoundTrip() throws {
        let directory = Self.tempDirectory()
        let history = DictationHistory(directory: directory)
        history.record(
            text: "帮我 review 这个 pull request",
            audio: nil,
            duration: 4.2,
            latency: 0.31,
            appName: "Claude Code",
            archiveAudio: false
        )
        #expect(history.entries.count == 1)

        // The store writes asynchronously; poll rather than sleep a fixed time.
        let indexURL = directory.appendingPathComponent("history.json")
        var waited = 0
        while !FileManager.default.fileExists(atPath: indexURL.path), waited < 200 {
            usleep(10_000)
            waited += 1
        }

        let reloaded = DictationHistory(directory: directory)
        #expect(reloaded.entries.count == 1)
        #expect(reloaded.entries.first?.text == "帮我 review 这个 pull request")
        #expect(reloaded.entries.first?.appName == "Claude Code")
    }

    @MainActor
    @Test("editing a transcript keeps its identity")
    func historyEdit() throws {
        let history = DictationHistory(directory: Self.tempDirectory())
        let entry = history.record(
            text: "让 Header 跑在 spark1 上",
            audio: nil, duration: 3, latency: 0.2,
            appName: nil, archiveAudio: false
        )
        history.updateText("让 herdr 跑在 spark1 上", for: entry.id)
        #expect(history.entries.first?.id == entry.id)
        #expect(history.entries.first?.text == "让 herdr 跑在 spark1 上")
    }

    @MainActor
    @Test("archived audio is immediately available before its disk write finishes")
    func historyPendingAudioIsReadable() {
        let history = DictationHistory(directory: Self.tempDirectory())
        let audio = Data(repeating: 0x5A, count: 4096)
        let entry = history.record(
            text: "rapid mlx", audio: audio,
            duration: 1, latency: 0.1, appName: nil, archiveAudio: true
        )
        #expect(history.audioData(for: entry) == audio)
    }

    @MainActor
    @Test("clearing immediately after recording cannot leave archived audio behind")
    func historyImmediateClearRemovesAudio() async throws {
        let directory = Self.tempDirectory()
        let history = DictationHistory(directory: directory)
        let entry = history.record(
            text: "discard me", audio: Data(repeating: 0x2A, count: 4096),
            duration: 1, latency: 0.1, appName: nil, archiveAudio: true
        )
        let audioURL = try #require(history.audioURL(for: entry))
        history.clear()
        #expect(history.entries.isEmpty)
        #expect(history.audioData(for: entry) == nil)
        await history.waitForPersistence()
        #expect(!FileManager.default.fileExists(atPath: audioURL.path))
    }

    // MARK: - Hotkey

    @MainActor
    @Test("turning dictation off while catalog refresh waits cannot finish enabling")
    func staleEnableCannotRearmAfterDisable() async {
        var continuation: CheckedContinuation<[ModelEntry], Never>?
        let binary = Self.tempDirectory().appendingPathComponent("rapid-mlx")
        let server = ServerManager(testingState: .idle, binaryPath: binary)
        let controller = DictationController(
            server: server,
            testingEnabled: true,
            audioCatalogLoader: { _ in
                await withCheckedContinuation { continuation = $0 }
            }
        )

        let enabling = Task { await controller.enable() }
        while continuation == nil { await Task.yield() }
        controller.isEnabled = false
        continuation?.resume(returning: [])
        await enabling.value

        #expect(controller.isEnabled == false)
        #expect(controller.phase == .off)
    }

    @MainActor
    @Test("a hotkey-boundary refresh observes a model deleted after enabling")
    func recordingBoundaryRefreshObservesDeletion() async {
        let state = CatalogState()
        let entry: (Bool) -> ModelEntry = { cached in
            ModelEntry(
                alias: "whisper-small",
                hfRepo: "mlx-community/whisper-small",
                sizeOnDisk: cached ? "461 MiB" : nil,
                cached: cached,
                kind: .audio,
                audioCapability: .transcription,
                audioFamily: "whisper"
            )
        }
        let binary = Self.tempDirectory().appendingPathComponent("rapid-mlx")
        let controller = DictationController(
            server: ServerManager(testingState: .idle, binaryPath: binary),
            audioCatalogLoader: { _ in [entry(state.cached)] }
        )

        #expect(await controller.modelIsOnDiskAfterRefresh("whisper-small"))
        state.cached = false
        #expect(await !controller.modelIsOnDiskAfterRefresh("whisper-small"))
    }

    @MainActor
    @Test("a second hotkey tap cancels a pending disk check")
    func secondTapCancelsPendingRecordingRequest() async {
        var continuation: CheckedContinuation<[ModelEntry]?, Never>?
        let controller = DictationController(
            server: ServerManager(
                testingState: .ready(alias: "whisper-small"),
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingPhase: .idle,
            audioCatalogLoader: { _ in
                await withCheckedContinuation { continuation = $0 }
            }
        )

        controller.toggleFromUI()
        while continuation == nil { await Task.yield() }
        controller.toggleFromUI()
        continuation?.resume(returning: [])
        await Task.yield()

        #expect(controller.phase == .idle)
    }

    @MainActor
    @Test("the hotkey is registered only after model warmup finishes")
    func hotkeyWaitsForModelWarmup() async {
        var warmupContinuation: CheckedContinuation<Bool, Never>?
        var hotkeyStartCount = 0
        let entry = ModelEntry(
            alias: "whisper-small",
            hfRepo: "mlx-community/whisper-small",
            sizeOnDisk: "461 MiB",
            cached: true,
            kind: .audio,
            audioCapability: .transcription,
            audioFamily: "whisper"
        )
        let controller = DictationController(
            server: ServerManager(
                testingState: .ready(alias: "whisper-small"),
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingReadiness: .init(
                microphone: true,
                accessibility: true,
                modelSelected: true,
                modelOnDisk: true
            ),
            testingPrewarm: {
                await withCheckedContinuation { warmupContinuation = $0 }
            },
            testingHotkeyStart: {
                hotkeyStartCount += 1
                return true
            },
            audioCatalogLoader: { _ in [entry] }
        )

        let enabling = Task { await controller.enable() }
        while warmupContinuation == nil { await Task.yield() }

        #expect(controller.phase == .preparingModel)
        #expect(hotkeyStartCount == 0)

        warmupContinuation?.resume(returning: true)
        await enabling.value

        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
    }

    @MainActor
    @Test("completed chat-model switch re-prepares enabled dictation before Ready")
    func chatModelSwitchRepreparesEnabledDictation() async {
        var warmupContinuation: CheckedContinuation<Bool, Never>?
        var prewarmCount = 0
        var hotkeyStartCount = 0
        let controller = readinessController(
            phase: .idle,
            prewarm: {
                prewarmCount += 1
                return await withCheckedContinuation { warmupContinuation = $0 }
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        controller.serverStateDidChange(.starting(alias: "lfm2.5-1b-4bit"))
        #expect(controller.phase == .preparingModel)
        #expect(prewarmCount == 0)
        #expect(hotkeyStartCount == 0)

        controller.serverStateDidChange(.ready(alias: "lfm2.5-1b-4bit"))
        while warmupContinuation == nil { await Task.yield() }
        #expect(controller.phase == .preparingModel)
        #expect(prewarmCount == 1)
        #expect(hotkeyStartCount == 0)

        warmupContinuation?.resume(returning: true)
        for _ in 0..<20 where controller.phase != .idle { await Task.yield() }
        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
    }

    @MainActor
    @Test("chat-model switches keep the enabled feature's event tap armed")
    func chatModelSwitchKeepsHotkeyRegistration() async {
        var prewarmCount = 0
        var hotkeyStartCount = 0
        var hotkeyStopCount = 0
        let controller = readinessController(
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            },
            hotkeyStop: { hotkeyStopCount += 1 }
        )

        await controller.enable()
        #expect(controller.phase == .idle)
        #expect(controller.isHotkeyArmed)
        #expect(hotkeyStartCount == 1)

        // Whole-process replacement publishes the old child stopping before
        // the new alias starts. That terminal-looking intermediate state was
        // the exact RC1 vector that destroyed the tap and exposed Arm now.
        controller.serverStateDidChange(.stopped)
        #expect(controller.phase == .off)
        #expect(controller.isHotkeyArmed)
        #expect(hotkeyStopCount == 0)

        controller.serverStateDidChange(.starting(alias: "qwen3.5-4b-4bit"))
        #expect(controller.phase == .preparingModel)
        #expect(controller.isHotkeyArmed)
        #expect(hotkeyStopCount == 0)

        controller.serverStateDidChange(.ready(alias: "qwen3.5-4b-4bit"))
        for _ in 0..<40 where controller.phase != .idle { await Task.yield() }

        #expect(controller.phase == .idle)
        #expect(controller.isHotkeyArmed)
        #expect(prewarmCount == 2)
        #expect(hotkeyStartCount == 1)
        #expect(hotkeyStopCount == 0)

        controller.disable()
        #expect(!controller.isHotkeyArmed)
        #expect(hotkeyStopCount == 1)
    }

    @MainActor
    @Test("failed chat-model switch leaves enabled dictation retryable")
    func failedChatModelSwitchIsRetryable() async {
        var warmupContinuation: CheckedContinuation<Bool, Never>?
        var hotkeyStartCount = 0
        let controller = readinessController(
            phase: .idle,
            prewarm: {
                await withCheckedContinuation { warmupContinuation = $0 }
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        controller.serverStateDidChange(.starting(alias: "lfm2.5-1b-4bit"))
        controller.serverStateDidChange(.crashed(
            alias: "lfm2.5-1b-4bit",
            message: "fixture failure"
        ))
        #expect(controller.isEnabled)
        #expect(controller.phase == .off)
        #expect(hotkeyStartCount == 0)

        controller.revalidate()
        while warmupContinuation == nil { await Task.yield() }
        #expect(controller.phase == .preparingModel)
        warmupContinuation?.resume(returning: true)
        for _ in 0..<20 where controller.phase != .idle { await Task.yield() }
        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
    }

    @MainActor
    @Test("audio-only fallback transitions do not cancel their owning preparation flight")
    func audioFallbackDoesNotSelfCancel() async {
        var warmupContinuation: CheckedContinuation<Bool, Never>?
        var prewarmCount = 0
        let controller = readinessController(
            prewarm: {
                prewarmCount += 1
                return await withCheckedContinuation { warmupContinuation = $0 }
            },
            hotkeyStart: { true }
        )

        let enabling = Task { await controller.enable() }
        while warmupContinuation == nil { await Task.yield() }
        controller.serverStateDidChange(.starting(alias: "whisper-small"))
        controller.serverStateDidChange(.ready(alias: "whisper-small"))

        #expect(controller.phase == .preparingModel)
        #expect(prewarmCount == 1)
        warmupContinuation?.resume(returning: true)
        await enabling.value
        #expect(controller.phase == .idle)
    }

    @MainActor
    @Test("audio-only auto-respawn re-arms enabled dictation")
    func audioFallbackAutoRespawnRearms() async {
        var warmupContinuation: CheckedContinuation<Bool, Never>?
        var prewarmCount = 0
        var hotkeyStartCount = 0
        let controller = readinessController(
            phase: .idle,
            prewarm: {
                prewarmCount += 1
                return await withCheckedContinuation { warmupContinuation = $0 }
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        controller.serverStateDidChange(.crashed(
            alias: "whisper-small",
            message: "fixture crash"
        ))
        controller.serverStateDidChange(.starting(alias: "whisper-small"))
        #expect(controller.phase == .preparingModel)
        controller.serverStateDidChange(.ready(alias: "whisper-small"))
        while warmupContinuation == nil { await Task.yield() }
        #expect(prewarmCount == 1)
        #expect(hotkeyStartCount == 0)

        warmupContinuation?.resume(returning: true)
        for _ in 0..<20 where controller.phase != .idle { await Task.yield() }
        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
    }

    @MainActor
    @Test("launch bootstrap arms before deferred model preparation")
    func launchBootstrapArmsWithoutWaitingForPrimaryHealth() async {
        var prewarmCount = 0
        var hotkeyStartCount = 0
        let controller = readinessController(
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        await controller.bootstrap(deferModelPreparation: true)

        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
        #expect(prewarmCount == 0, "audio preparation must not race the primary launch")

        controller.toggleFromUI()
        await Task.yield()
        #expect(controller.lastError?.contains("chat model finishes starting") == true)
        #expect(prewarmCount == 0)

        await controller.finishDeferredBootstrap()

        #expect(controller.phase == .idle)
        #expect(prewarmCount == 1)
        #expect(hotkeyStartCount == 1, "finishing launch restore reuses the feature-owned event tap")
    }

    @MainActor
    @Test("a cancelled session restore releases its audio barrier without prewarming")
    func cancelledSessionRestoreReleasesBarrier() async {
        var prewarmCount = 0
        let controller = readinessController(
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: { true }
        )
        await controller.bootstrap(deferModelPreparation: true)

        let cancelledFinish = Task { @MainActor in
            await controller.finishDeferredBootstrap()
        }
        cancelledFinish.cancel()
        await cancelledFinish.value
        #expect(prewarmCount == 0, "a superseded restore cannot start the audio lane")

        await controller.enable()
        #expect(prewarmCount == 1, "the replacement flow must not inherit a stranded barrier")
    }

    @MainActor
    @Test("cold-start revalidation inherits the synchronous launch barrier")
    func coldStartRevalidationCannotPrewarmBeforeChatRestore() async {
        var prewarmCount = 0
        var hotkeyStartCount = 0
        let controller = readinessController(
            initiallyDeferred: true,
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        controller.revalidate()
        while hotkeyStartCount == 0 { await Task.yield() }
        #expect(prewarmCount == 0, "Audio-view revalidation cannot start the audio lane")

        await controller.finishDeferredBootstrap()
        #expect(prewarmCount == 1)
    }

    @MainActor
    @Test("model changes inherit the launch audio-preparation barrier")
    func modelChangeDuringDeferredBootstrapCannotPrewarm() async {
        var prewarmCount = 0
        var hotkeyStartCount = 0
        let controller = readinessController(
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        await controller.bootstrap(deferModelPreparation: true)
        controller.modelAlias = "another-speech-input"
        for _ in 0..<40 where controller.phase != .idle { await Task.yield() }

        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1, "the enabled feature keeps one event-tap registration")
        #expect(prewarmCount == 0, "a model change must not steal the starting chat process")
    }

    @MainActor
    @Test("launch barrier exists before catalog refresh suspends")
    func modelChangeDuringDeferredCatalogRefreshCannotPrewarm() async {
        var firstCatalogContinuation: CheckedContinuation<[ModelEntry]?, Never>?
        var catalogLoadCount = 0
        var prewarmCount = 0
        let entry = cachedAudioEntry(alias: "whisper-small")
        let controller = DictationController(
            server: ServerManager(
                testingState: .ready(alias: "whisper-small"),
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingReadiness: .init(
                microphone: true,
                accessibility: true,
                modelSelected: true,
                modelOnDisk: true
            ),
            testingPrewarm: {
                prewarmCount += 1
                return true
            },
            testingHotkeyStart: { true },
            audioCatalogLoader: { _ in
                catalogLoadCount += 1
                if catalogLoadCount == 1 {
                    return await withCheckedContinuation { firstCatalogContinuation = $0 }
                }
                return [entry]
            }
        )

        let bootstrap = Task { await controller.bootstrap(deferModelPreparation: true) }
        while firstCatalogContinuation == nil { await Task.yield() }
        controller.modelAlias = "another-speech-input"
        for _ in 0..<30 { await Task.yield() }

        #expect(prewarmCount == 0)
        firstCatalogContinuation?.resume(returning: [entry])
        await bootstrap.value
        #expect(prewarmCount == 0)
    }

    @MainActor
    @Test("hotkey failure cannot release the launch barrier")
    func failedDeferredHotkeyRegistrationCannotEnablePrewarm() async {
        var hotkeyAttempts = 0
        var prewarmCount = 0
        let controller = readinessController(
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: {
                hotkeyAttempts += 1
                return hotkeyAttempts > 1
            }
        )

        await controller.bootstrap(deferModelPreparation: true)
        #expect(controller.phase == .off)
        #expect(prewarmCount == 0)

        await controller.enable()
        #expect(controller.phase == .idle)
        #expect(hotkeyAttempts == 2)
        #expect(prewarmCount == 0)
    }

    @MainActor
    @Test("failed model warmup leaves dictation visibly unarmed")
    func failedWarmupDoesNotArmHotkey() async {
        var hotkeyStartCount = 0
        let controller = readinessController(
            prewarm: { false },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        await controller.enable()

        #expect(controller.phase == .off)
        #expect(hotkeyStartCount == 0)
        #expect(controller.lastError?.contains("couldn't load") == true)
    }

    @MainActor
    @Test("co-loaded dictation arms and records while the conversation alias stays primary")
    func coLoadedConversationLaneArmsAndRecords() async {
        var hotkeyStartCount = 0
        var recorderStartCount = 0
        let entry = cachedAudioEntry(alias: "whisper-small")
        let server = ServerManager(
            testingState: .ready(alias: "qwen3-0.6b-4bit"),
            binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx"),
            activeBearer: "test-bearer"
        )
        let controller = DictationController(
            server: server,
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingReadiness: .init(
                microphone: true,
                accessibility: true,
                modelSelected: true,
                modelOnDisk: true
            ),
            testingPrewarm: { true },
            testingHotkeyStart: {
                hotkeyStartCount += 1
                return true
            },
            testingRecorderStart: { recorderStartCount += 1 },
            audioCatalogLoader: { _ in [entry] }
        )

        await controller.enable()

        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
        #expect(server.servingAlias == "qwen3-0.6b-4bit")

        controller.toggleFromUI()
        while recorderStartCount == 0 { await Task.yield() }

        #expect(controller.phase == .starting)
        #expect(recorderStartCount == 1)
        #expect(server.servingAlias == "qwen3-0.6b-4bit")
        #expect(controller.testingHasActiveTicker)
        #expect(!controller.testingHasRecorder)

        controller.disable()
        #expect(!controller.testingHasActiveTicker)
        #expect(!controller.testingHasRecorder)
    }

    @MainActor
    @Test("disabling during model warmup cannot register a stale hotkey")
    func disableDuringWarmupDoesNotArmHotkey() async {
        var continuation: CheckedContinuation<Bool, Never>?
        var hotkeyStartCount = 0
        let controller = readinessController(
            phase: .idle,
            prewarm: {
                await withCheckedContinuation { continuation = $0 }
            },
            hotkeyStart: {
                hotkeyStartCount += 1
                return true
            }
        )

        let enabling = Task { await controller.enable() }
        while continuation == nil { await Task.yield() }
        controller.isEnabled = false
        continuation?.resume(returning: true)
        await enabling.value

        #expect(controller.phase == .off)
        #expect(hotkeyStartCount == 0)
    }

    @MainActor
    @Test("changing models cancels an active capture before preparation")
    func modelChangeCancelsActiveCapture() {
        var cancellationCount = 0
        let controller = DictationController(
            server: ServerManager(testingState: .ready(alias: "whisper-small")),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingPhase: .recording,
            testingRecorderCancel: { cancellationCount += 1 },
            audioCatalogLoader: { _ in [] }
        )

        controller.modelAlias = "qwen3-asr"

        #expect(cancellationCount == 1)
        #expect(controller.phase == .preparingModel)
    }

    @MainActor
    @Test("changing models invalidates an active transcription before preparation")
    func modelChangeCancelsActiveTranscription() {
        var cancellationCount = 0
        let controller = DictationController(
            server: ServerManager(testingState: .ready(alias: "whisper-small")),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingPhase: .transcribing,
            testingTranscribeCancel: { cancellationCount += 1 },
            audioCatalogLoader: { _ in [] }
        )

        controller.modelAlias = "qwen3-asr"

        #expect(cancellationCount == 1)
        #expect(controller.phase == .preparingModel)
    }

    @MainActor
    @Test("replaying a retained completed download keeps a ready hotkey armed")
    func completedDownloadReplayIsIdempotent() async {
        var prewarmCount = 0
        let controller = readinessController(
            phase: .idle,
            prewarm: {
                prewarmCount += 1
                return true
            },
            hotkeyStart: { true }
        )

        await controller.modelDownloadDidFinish()

        #expect(controller.phase == .idle)
        #expect(prewarmCount == 0)
    }

    @MainActor
    @Test("a failed optional weight probe still arms a serving model")
    func failedWeightProbeStillArmsServingModel() async {
        var hotkeyStartCount = 0
        let entry = cachedAudioEntry(alias: "whisper-small")
        let controller = DictationController(
            server: ServerManager(
                testingState: .ready(alias: "whisper-small"),
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingReadiness: .init(
                microphone: true,
                accessibility: true,
                modelSelected: true,
                modelOnDisk: true
            ),
            testingWarmup: { false },
            testingHotkeyStart: {
                hotkeyStartCount += 1
                return true
            },
            audioCatalogLoader: { _ in [entry] }
        )

        await controller.enable()

        #expect(controller.phase == .idle)
        #expect(hotkeyStartCount == 1)
        #expect(controller.lastWarmupWarning?.contains("first dictation may be slower") == true)
    }

    @MainActor
    @Test("replacement preparation waits for the old server mutation")
    func replacementPrewarmIsSerialized() async {
        var starts = 0
        var firstContinuation: CheckedContinuation<Bool, Never>?
        var secondContinuation: CheckedContinuation<Bool, Never>?
        let entries = [cachedAudioEntry(alias: "whisper-small"), cachedAudioEntry(alias: "qwen3-asr")]
        let controller = DictationController(
            server: ServerManager(
                testingState: .ready(alias: "whisper-small"),
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingReadiness: .init(
                microphone: true,
                accessibility: true,
                modelSelected: true,
                modelOnDisk: true
            ),
            testingPrewarm: {
                starts += 1
                if starts == 1 {
                    return await withCheckedContinuation { firstContinuation = $0 }
                }
                return await withCheckedContinuation { secondContinuation = $0 }
            },
            testingHotkeyStart: { true },
            audioCatalogLoader: { _ in entries }
        )

        let firstEnable = Task { await controller.enable() }
        while firstContinuation == nil { await Task.yield() }
        controller.modelAlias = "qwen3-asr"
        for _ in 0..<20 { await Task.yield() }
        #expect(starts == 1)

        firstContinuation?.resume(returning: false)
        while secondContinuation == nil { await Task.yield() }
        #expect(starts == 2)
        secondContinuation?.resume(returning: true)
        await firstEnable.value
    }

    @MainActor
    private func readinessController(
        phase: DictationController.Phase = .off,
        initiallyDeferred: Bool = false,
        prewarm: @escaping @MainActor () async -> Bool,
        hotkeyStart: @escaping @MainActor () -> Bool,
        hotkeyStop: @escaping @MainActor () -> Void = {}
    ) -> DictationController {
        let entry = ModelEntry(
            alias: "whisper-small",
            hfRepo: "mlx-community/whisper-small",
            sizeOnDisk: "461 MiB",
            cached: true,
            kind: .audio,
            audioCapability: .transcription,
            audioFamily: "whisper"
        )
        return DictationController(
            server: ServerManager(
                testingState: .ready(alias: "whisper-small"),
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            testingEnabled: true,
            testingModelAlias: "whisper-small",
            testingPhase: phase,
            testingReadiness: .init(
                microphone: true,
                accessibility: true,
                modelSelected: true,
                modelOnDisk: true
            ),
            testingPrewarm: prewarm,
            testingHotkeyStart: hotkeyStart,
            testingHotkeyStop: hotkeyStop,
            testingInitialModelPreparationDeferred: initiallyDeferred,
            audioCatalogLoader: { _ in [entry] }
        )
    }

    private func cachedAudioEntry(alias: String) -> ModelEntry {
        ModelEntry(
            alias: alias,
            hfRepo: "mlx-community/\(alias)",
            sizeOnDisk: "461 MiB",
            cached: true,
            kind: .audio,
            audioCapability: .transcription,
            audioFamily: "whisper"
        )
    }

    @MainActor
    @Test("a failed catalog probe preserves the last successful cache snapshot")
    func failedCatalogProbePreservesCacheSnapshot() async {
        let state = CatalogState()
        let entry = ModelEntry(
            alias: "whisper-small",
            hfRepo: "mlx-community/whisper-small",
            sizeOnDisk: "461 MiB",
            cached: true,
            kind: .audio,
            audioCapability: .transcription,
            audioFamily: "whisper"
        )
        let controller = DictationController(
            server: ServerManager(
                testingState: .idle,
                binaryPath: Self.tempDirectory().appendingPathComponent("rapid-mlx")
            ),
            audioCatalogLoader: { _ in state.cached ? [entry] : nil }
        )

        #expect(await controller.modelIsOnDiskAfterRefresh("whisper-small"))
        state.cached = false
        #expect(await controller.modelIsOnDiskAfterRefresh("whisper-small"))
    }


    /// Only right-hand modifiers are offered. Left ⌘ rides along with ⌘C/⌘V/
    /// ⌘Tab dozens of times an hour, so "tapped on its own" cannot be detected
    /// reliably enough to arm a microphone with.
    @MainActor
    @Test("only right-hand modifiers are offered, with side-specific keycodes")
    func hotkeyTriggers() {
        let codes = DictationHotkey.Trigger.allCases.map(\.keyCode)
        #expect(codes == [54, 61])   // kVK_RightCommand, kVK_RightOption
        #expect(!DictationHotkey.Trigger.allCases.contains { $0.keyCode == 55 })
        #expect(DictationHotkey.Trigger.rightCommand.label == "Right ⌘")
    }

    // MARK: - Audio mode

    /// The Audio surface is a two-lane product: Speech to Text (dictation)
    /// and Text to Speech. The old file-transcription workbench was removed
    /// deliberately — a third tab reappearing here is a regression, not a
    /// feature. Speech to Text stays the landing mode.
    @MainActor
    @Test("Audio opens on Speech to Text and offers exactly the two lanes")
    func audioModeDefault() {
        #expect(AudioViewModel.Mode.allCases == [.dictation, .speech])
        #expect(AudioViewModel.Mode.dictation.label == "Speech to Text")
        #expect(AudioViewModel.Mode.speech.label == "Text to Speech")
        let viewModel = AudioViewModel(server: ServerManager(testingState: .idle))
        #expect(viewModel.mode == .dictation)
    }

    /// The AX identifier is derived from `axName`, not from `label`, precisely
    /// so it survives labels that contain spaces: the golden-flow harness
    /// addresses controls as bare shell words (`press … Audio.Mode.SpeechToText`),
    /// and a space would split the argument and make the control unreachable.
    @MainActor
    @Test("audio mode AX names are single shell words")
    func audioModeIdentifiersAreShellSafe() {
        for mode in AudioViewModel.Mode.allCases {
            #expect(!mode.axName.contains(" "))
            #expect(mode.axName.allSatisfy { $0.isLetter || $0.isNumber })
        }
        #expect(AudioViewModel.Mode.dictation.axName == "Dictation")
        #expect(AudioViewModel.Mode.speech.axName == "Speech")
    }

    // MARK: - Helpers

    private static func tempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("dictation-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func tempStore() -> URL {
        tempDirectory().appendingPathComponent("vocabulary.json")
    }
}
