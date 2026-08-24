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
            (false, true, true, "whisper-small", "whisper-small", true, "whisper-small"),
            (true, false, true, "whisper-small", "whisper-small", true, "whisper-small"),
            (true, true, false, "whisper-small", "whisper-small", true, "whisper-small"),
            (true, true, true, "qwen-asr", "whisper-small", true, "whisper-small"),
            (true, true, true, "whisper-small", "whisper-small", false, "whisper-small"),
            (true, true, true, "whisper-small", "whisper-small", true, "other-model"),
        ]
    )
    func stalePreparedStateCannotArm(
        prewarm: Bool,
        enabled: Bool,
        currentRequest: Bool,
        selectedAlias: String,
        preparingAlias: String,
        preparing: Bool,
        servingAlias: String
    ) {
        #expect(!DictationEnablePolicy.mayRegisterHotkey(after: .init(
            prewarmSucceeded: prewarm,
            isEnabled: enabled,
            requestIsCurrent: currentRequest,
            selectedAlias: selectedAlias,
            preparingAlias: preparingAlias,
            isPreparing: preparing,
            servingAlias: servingAlias
        )))
    }

    @Test("only the current ready model may register the hotkey")
    func currentPreparedModelMayArm() {
        #expect(DictationEnablePolicy.mayRegisterHotkey(after: .init(
            prewarmSucceeded: true,
            isEnabled: true,
            requestIsCurrent: true,
            selectedAlias: "whisper-small",
            preparingAlias: "whisper-small",
            isPreparing: true,
            servingAlias: "whisper-small"
        )))
    }
}
