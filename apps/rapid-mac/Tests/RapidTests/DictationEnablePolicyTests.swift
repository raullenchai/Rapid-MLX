import Testing

@testable import Rapid

@Suite("Dictation enable policy")
struct DictationEnablePolicyTests {
    @Test(
        "every missing prerequisite rejects before model preparation",
        arguments: [
            (false, true, true, true, true, "Microphone"),
            (true, true, false, true, true, "Choose a transcription model"),
            (true, true, true, false, true, "isn't downloaded yet"),
            (true, false, true, true, false, "Accessibility"),
        ]
    )
    func prerequisiteFailures(
        microphone: Bool,
        accessibility: Bool,
        selected: Bool,
        onDisk: Bool,
        disablesIntent: Bool,
        messageFragment: String
    ) {
        let decision = DictationEnablePolicy.evaluate(.init(
            microphone: microphone,
            accessibility: accessibility,
            modelSelected: selected,
            modelOnDisk: onDisk,
            modelAlias: "whisper-small"
        ))

        guard case .reject(let message, let disableIntent) = decision else {
            Issue.record("missing prerequisite was allowed to prepare the model")
            return
        }
        #expect(message.contains(messageFragment))
        #expect(disableIntent == disablesIntent)
    }

    @Test("missing model errors are hidden only while the download row is visible")
    func errorBannerPolicy() {
        let alias = "whisper-small"
        let messages = [DictationEnablePolicy.modelNotDownloadedMessage(alias: alias),
                        DictationEnablePolicy.modelNoLongerOnDiskMessage(alias: alias)]
        for message in messages {
            #expect(!DictationEnablePolicy.showsErrorBanner(lastError: message, modelAlias: alias, modelDownloadRowVisible: true))
            #expect(DictationEnablePolicy.showsErrorBanner(lastError: message, modelAlias: alias, modelDownloadRowVisible: false))
        }
        for message in ["Microphone access denied", "Accessibility access denied"] {
            #expect(DictationEnablePolicy.showsErrorBanner(lastError: message, modelAlias: alias, modelDownloadRowVisible: true))
        }
        #expect(!DictationEnablePolicy.showsErrorBanner(lastError: nil, modelAlias: alias, modelDownloadRowVisible: true))
        #expect(!DictationEnablePolicy.showsErrorBanner(lastError: "", modelAlias: alias, modelDownloadRowVisible: false))
        #expect(DictationEnablePolicy.evaluate(.init(microphone: true, accessibility: true, modelSelected: true,
            modelOnDisk: false, modelAlias: alias)) == .reject(
                message: "Whisper Small isn't downloaded yet. Download it in the Model row, then turn dictation on.",
                disableIntent: true))
    }

    @Test("complete prerequisites advance to model preparation")
    func completePrerequisitesPrepare() {
        #expect(DictationEnablePolicy.evaluate(.init(
            microphone: true,
            accessibility: true,
            modelSelected: true,
            modelOnDisk: true,
            modelAlias: "whisper-small"
        )) == .prepareModel)
    }

    @Test(
        "hotkey registration requires every post-warmup identity boundary",
        arguments: [
            (false, true, true, "whisper-small", "whisper-small", true, true),
            (true, false, true, "whisper-small", "whisper-small", true, true),
            (true, true, false, "whisper-small", "whisper-small", true, true),
            (true, true, true, "qwen-asr", "whisper-small", true, true),
            (true, true, true, "whisper-small", "whisper-small", false, true),
            (true, true, true, "whisper-small", "whisper-small", true, false),
        ]
    )
    func stalePreparedStateCannotArm(
        prewarm: Bool,
        enabled: Bool,
        currentRequest: Bool,
        selectedAlias: String,
        preparingAlias: String,
        preparing: Bool,
        voiceLaneReady: Bool
    ) {
        #expect(!DictationEnablePolicy.mayRegisterHotkey(after: .init(
            prewarmSucceeded: prewarm,
            isEnabled: enabled,
            requestIsCurrent: currentRequest,
            selectedAlias: selectedAlias,
            preparingAlias: preparingAlias,
            isPreparing: preparing,
            voiceLaneReady: voiceLaneReady
        )))
    }

    @Test("the current prepared voice lane may register the hotkey")
    func currentPreparedVoiceLaneMayArm() {
        #expect(DictationEnablePolicy.mayRegisterHotkey(after: .init(
            prewarmSucceeded: true,
            isEnabled: true,
            requestIsCurrent: true,
            selectedAlias: "whisper-small",
            preparingAlias: "whisper-small",
            isPreparing: true,
            voiceLaneReady: true
        )))
    }
}
