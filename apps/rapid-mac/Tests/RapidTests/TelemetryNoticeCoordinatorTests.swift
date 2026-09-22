import Testing
@testable import Rapid

@MainActor
@Suite("Telemetry launch notice")
struct TelemetryNoticeCoordinatorTests {
    @MainActor
    final class Recorder {
        var presentations = 0
        var sessionStarts = 0
        var activations: [ProductValueKind] = []
    }

    private func makeCoordinator(
        needsNotice: Bool = true,
        result: TelemetryConsent.NoticePresentationResult = .init(
            persisted: true, uploadAllowedThisRun: true
        ),
        recorder: Recorder = Recorder()
    ) -> TelemetryNoticeCoordinator {
        TelemetryNoticeCoordinator(
            needsNotice: { needsNotice },
            recordPresentation: {
                recorder.presentations += 1
                return result
            },
            startTelemetrySession: { recorder.sessionStarts += 1 },
            reportActivation: { recorder.activations.append($0) }
        )
    }

    @Test("Launch scheduling alone never records the disclosure")
    func schedulingDoesNotWrite() {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        #expect(coordinator.isPresented)
        #expect(recorder.presentations == 0)
    }

    @Test("The marker callback runs once only after the notice appears")
    func appearanceRecordsOnce() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.noticeDidAppear()
        coordinator.noticeDidAppear()
        await Task.yield()
        #expect(recorder.presentations == 1)
        #expect(recorder.sessionStarts == 1)
    }

    @Test("A failed marker write never starts telemetry")
    func failedWriteStaysDark() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(
            result: .init(persisted: false, uploadAllowedThisRun: false),
            recorder: recorder
        )
        coordinator.noticeDidAppear()
        await Task.yield()
        #expect(recorder.presentations == 1)
        #expect(recorder.sessionStarts == 0)
    }

    @Test("Legacy migration writes but does not upload in that run")
    func migrationRunStaysDark() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(
            result: .init(persisted: true, uploadAllowedThisRun: false),
            recorder: recorder
        )
        coordinator.noticeDidAppear()
        await Task.yield()
        #expect(recorder.presentations == 1)
        #expect(recorder.sessionStarts == 0)
    }

    @Test("Acknowledgement only dismisses the already-delivered notice")
    func acknowledgeDismisses() {
        let coordinator = makeCoordinator()
        coordinator.acknowledge()
        #expect(!coordinator.isPresented)
    }

    @Test("Kill switch or existing marker suppresses presentation entirely")
    func noNoticeStaysHidden() {
        let coordinator = makeCoordinator(needsNotice: false)
        coordinator.noticeDidAppear()
        #expect(!coordinator.isPresented)
    }

    @Test("Product-value signals still reach the deployed activation reporter seam")
    func productValueReportingRemainsWired() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        coordinator.productValueDelivered(.dictationTranscript)
        await Task.yield()
        #expect(recorder.activations == [.dictationTranscript])
        #expect(ProductValueKind.chatReply.telemetryActivationKind == .firstChatReply)
        #expect(ProductValueKind.generatedImage.telemetryActivationKind == .firstImage)
    }
}
