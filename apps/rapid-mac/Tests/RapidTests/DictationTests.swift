import Foundation
import Testing

@testable import Rapid

/// Unit cover for the parts of dictation that are pure logic. The hotkey tap,
/// the audio engine and the paste all need a real session to exercise, but the
/// text handling, the vocabulary budget and the on-disk formats do not — and
/// those are where the bugs that survive a manual smoke test live.
@Suite("Dictation")
struct DictationTests {

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

    /// Dictation is additive: neither existing file transcription nor speech
    /// synthesis disappears when the global hotkey workflow is introduced.
    @MainActor
    @Test("Audio opens on Dictation without removing either workbench")
    func audioModeDefault() {
        #expect(AudioViewModel.Mode.allCases == [.dictation, .speech, .transcription])
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
        #expect(AudioViewModel.Mode.transcription.axName == "Transcription")
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
