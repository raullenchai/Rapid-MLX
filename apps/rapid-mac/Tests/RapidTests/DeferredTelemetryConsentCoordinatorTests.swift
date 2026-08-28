import Testing
@testable import Rapid

@MainActor
@Suite("Deferred telemetry consent")
struct DeferredTelemetryConsentCoordinatorTests {
    @MainActor
    final class Recorder {
        var decisions: [Bool] = []
        var sessionStarts = 0
        var activations: [ProductValueKind] = []
    }

    private func makeCoordinator(
        needsDecision: Bool = true,
        recorder: Recorder = Recorder()
    ) -> DeferredTelemetryConsentCoordinator {
        DeferredTelemetryConsentCoordinator(
            needsDecision: { needsDecision },
            recordDecision: { recorder.decisions.append($0) },
            startTelemetrySession: { recorder.sessionStarts += 1 },
            reportActivation: { recorder.activations.append($0) }
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

    @Test("Every typed success before the answer is emitted only after Share")
    func pendingKindsWaitForShare() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.chatReply)
        coordinator.productValueDelivered(.generatedImage)
        coordinator.productValueDelivered(.chatReply)
        #expect(recorder.activations.isEmpty)

        coordinator.share()
        await Task.yield()

        #expect(recorder.activations == [.chatReply, .generatedImage])
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
        #expect(recorder.activations.isEmpty)
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
        #expect(recorder.activations.isEmpty)
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
        #expect(recorder.activations == [.chatReply])
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
        #expect(recorder.activations == [.chatReply])
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
        #expect(recorder.activations.isEmpty)
        #expect(!coordinator.isPresented)
    }

    @Test("An already-consented install reports future product value without showing a banner")
    func existingOptInReportsActivation() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(needsDecision: false, recorder: recorder)

        coordinator.productValueDelivered(.dictationTranscript)
        await Task.yield()

        #expect(recorder.decisions.isEmpty)
        #expect(recorder.activations == [.dictationTranscript])
        #expect(!coordinator.isPresented)
    }

    @Test("Product-value kinds map to the deployed closed activation vocabulary")
    func productValueWireMapping() {
        #expect(ProductValueKind.chatReply.telemetryActivationKind == .firstChatReply)
        #expect(ProductValueKind.dictationTranscript.telemetryActivationKind == .firstDictation)
        #expect(ProductValueKind.generatedImage.telemetryActivationKind == .firstImage)
    }
}
