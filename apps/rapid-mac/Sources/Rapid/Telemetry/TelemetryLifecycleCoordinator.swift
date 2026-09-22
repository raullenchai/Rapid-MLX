import Foundation
import Observation

enum ProductValueKind: Equatable, Sendable {
    case chatReply
    case dictationTranscript
    case generatedImage

    var telemetryActivationKind: TelemetryEvent.Activation.Kind {
        switch self {
        case .chatReply: .firstChatReply
        case .dictationTranscript: .firstDictation
        case .generatedImage: .firstImage
        }
    }
}

/// Owns Desktop telemetry startup and activation reporting without mounting a
/// launch-time UI surface. The shared policy writer remains fail-closed: a
/// failed or suppressed write leaves the sender disabled for this launch.
@MainActor
@Observable
final class TelemetryLifecycleCoordinator {
    typealias PolicyApplier = @MainActor () async -> TelemetryConsent.DefaultPolicyResult
    typealias ActivationReporter = @MainActor (ProductValueKind) async -> Void

    @ObservationIgnored private let applyPolicy: PolicyApplier
    @ObservationIgnored private let startTelemetrySession: @MainActor () async -> Void
    @ObservationIgnored private let reportActivation: ActivationReporter
    @ObservationIgnored private var policyAttempted = false

    init(
        applyPolicy: @escaping PolicyApplier = { await TelemetryConsent.applyDefaultPolicy() },
        startTelemetrySession: @escaping @MainActor () async -> Void = {
            await TelemetrySession.sendStartIfNeeded()
        },
        reportActivation: ActivationReporter? = nil
    ) {
        self.applyPolicy = applyPolicy
        self.startTelemetrySession = startTelemetrySession
        self.reportActivation = reportActivation ?? { kind in
            await DesktopActivationReporter.shared.report(kind.telemetryActivationKind)
        }
    }

    func start() async {
        guard !policyAttempted else { return }
        policyAttempted = true
        let result = await applyPolicy()
        if result.persisted, result.uploadAllowedThisRun {
            await startTelemetrySession()
        }
    }

    func productValueDelivered(_ kind: ProductValueKind) {
        Task { await reportActivation(kind) }
    }
}
