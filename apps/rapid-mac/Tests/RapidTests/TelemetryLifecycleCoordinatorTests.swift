import Testing
@testable import Rapid

@MainActor
@Suite("Telemetry lifecycle")
struct TelemetryLifecycleCoordinatorTests {
    @MainActor
    final class Recorder {
        var policyApplications = 0
        var sessionStarts = 0
        var activations: [ProductValueKind] = []
    }

    private func makeCoordinator(
        result: TelemetryConsent.DefaultPolicyResult = .init(
            persisted: true, uploadAllowedThisRun: true
        ),
        recorder: Recorder = Recorder()
    ) -> TelemetryLifecycleCoordinator {
        TelemetryLifecycleCoordinator(
            applyPolicy: {
                recorder.policyApplications += 1
                return result
            },
            startTelemetrySession: { recorder.sessionStarts += 1 },
            reportActivation: { recorder.activations.append($0) }
        )
    }

    @Test("Startup applies the default policy once")
    func startupAppliesPolicyOnce() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)

        await coordinator.start()
        await coordinator.start()

        #expect(recorder.policyApplications == 1)
        #expect(recorder.sessionStarts == 1)
    }

    @Test("Failed or deferred policy application does not start telemetry")
    func failedPolicyStaysDark() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(
            result: .init(persisted: false, uploadAllowedThisRun: false),
            recorder: recorder
        )

        await coordinator.start()

        #expect(recorder.policyApplications == 1)
        #expect(recorder.sessionStarts == 0)
    }

    @Test("A persisted migration policy that defers upload stays dark")
    func deferredPolicyStaysDark() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(
            result: .init(persisted: true, uploadAllowedThisRun: false),
            recorder: recorder
        )

        await coordinator.start()

        #expect(recorder.policyApplications == 1)
        #expect(recorder.sessionStarts == 0)
    }

    @Test("Product-value signals still reach the activation reporter seam")
    func productValueReportingRemainsWired() async {
        let recorder = Recorder()
        let coordinator = makeCoordinator(recorder: recorder)
        await coordinator.reportProductValue(.dictationTranscript)
        #expect(recorder.activations == [.dictationTranscript])
        #expect(ProductValueKind.chatReply.telemetryActivationKind == .firstChatReply)
        #expect(ProductValueKind.generatedImage.telemetryActivationKind == .firstImage)
    }
}
