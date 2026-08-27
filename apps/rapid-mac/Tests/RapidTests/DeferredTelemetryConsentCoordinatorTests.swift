import Testing
@testable import Rapid

@MainActor
@Suite("Deferred telemetry consent")
struct DeferredTelemetryConsentCoordinatorTests {
    @MainActor
    final class Recorder {
        var decisions: [Bool] = []
        var sessionStarts = 0
    }

    private func makeCoordinator(
        needsDecision: Bool = true,
        recorder: Recorder = Recorder()
    ) -> DeferredTelemetryConsentCoordinator {
        DeferredTelemetryConsentCoordinator(
            needsDecision: { needsDecision },
            recordDecision: { recorder.decisions.append($0) },
            startTelemetrySession: { recorder.sessionStarts += 1 }
        )
    }

    @Test("An undecided install is not interrupted before product value")
    func waitsForValue() {
        let coordinator = makeCoordinator()
        #expect(!coordinator.isPresented)
        #expect(coordinator.triggeringValue == nil)
    }

    @Test(arguments: [
        ProductValueKind.chatReply,
        .dictationTranscript,
        .generatedImage,
    ])
    func everyRealSuccessCanTrigger(_ kind: ProductValueKind) {
        let coordinator = makeCoordinator()
        coordinator.productValueDelivered(kind)
        #expect(coordinator.isPresented)
        #expect(coordinator.triggeringValue == kind)
    }

    @Test("Concurrent feature successes converge on the first invitation")
    func onlyOneInvitation() {
        let coordinator = makeCoordinator()
        coordinator.productValueDelivered(.chatReply)
        coordinator.productValueDelivered(.generatedImage)
        #expect(coordinator.triggeringValue == .chatReply)
    }

    @Test("An existing durable decision suppresses the invitation")
    func existingDecisionWins() {
        let coordinator = makeCoordinator(needsDecision: false)
        coordinator.productValueDelivered(.chatReply)
        #expect(!coordinator.isPresented)
    }

    @Test("Decline is durable and never asks again")
    func declineIsFinal() {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.dictationTranscript)
        coordinator.decline()
        coordinator.productValueDelivered(.chatReply)
        #expect(recorder.decisions == [false])
        #expect(recorder.sessionStarts == 0)
        #expect(!coordinator.isPresented)
    }

    @Test("Close is a quiet durable decline")
    func closeIsFinal() {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.generatedImage)
        coordinator.close()
        #expect(recorder.decisions == [false])
        #expect(recorder.sessionStarts == 0)
        #expect(!coordinator.isPresented)
    }

    @Test("Share records consent before starting the current session")
    func shareStartsSession() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.chatReply)
        coordinator.share()
        coordinator.share()
        await Task.yield()
        #expect(recorder.decisions == [true])
        #expect(recorder.sessionStarts == 1)
        #expect(!coordinator.isPresented)
    }

    @Test("Settings stays authoritative while an invitation is visible")
    func settingsDecisionDismissesInvitation() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.chatReply)
        coordinator.settingsChanged(enabled: true)
        await Task.yield()
        #expect(recorder.decisions == [true])
        #expect(recorder.sessionStarts == 1)
        #expect(!coordinator.isPresented)
    }

    @Test("Settings can opt in later after the one-time invitation was declined")
    func settingsRemainsTheDurableControl() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.chatReply)
        coordinator.decline()
        coordinator.settingsChanged(enabled: true)
        await Task.yield()

        #expect(recorder.decisions == [false, true])
        #expect(recorder.sessionStarts == 1)
        #expect(!coordinator.isPresented)
    }
}
