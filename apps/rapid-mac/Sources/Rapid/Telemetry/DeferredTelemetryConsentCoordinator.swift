import Foundation
import Observation

/// The product-approved, closed trigger set for the one-time telemetry
/// invitation. These are consent-timing policy, not an inventory of every
/// successful feature: Audio-tab file transcription and speech synthesis are
/// deliberately excluded, as are edits of an existing image. Expanding this
/// enum requires an explicit product decision rather than wiring every new
/// feature into consent automatically.
enum ProductValueKind: Equatable, Sendable {
    case chatReply
    case dictationTranscript
    case generatedImage
}

@MainActor
@Observable
final class DeferredTelemetryConsentCoordinator {

    private(set) var isPresented = false
    private(set) var triggeringValue: ProductValueKind?

    @ObservationIgnored private let needsDecision: () -> Bool
    @ObservationIgnored private let recordDecision: (Bool) -> Void
    @ObservationIgnored private let startTelemetrySession: () async -> Void
    @ObservationIgnored private var resolvedThisProcess: Bool

    init(
        needsDecision: @escaping () -> Bool = { TelemetryConsent.needsDecision() },
        recordDecision: @escaping (Bool) -> Void = { TelemetryConsent.record(enabled: $0) },
        startTelemetrySession: @escaping () async -> Void = {
            await TelemetrySession.sendStartIfNeeded()
        }
    ) {
        self.needsDecision = needsDecision
        self.recordDecision = recordDecision
        self.startTelemetrySession = startTelemetrySession
        self.resolvedThisProcess = !needsDecision()
    }

    /// Records a real delivered outcome. Concurrent feature completions
    /// converge on one prompt and an existing durable decision always wins.
    func productValueDelivered(_ kind: ProductValueKind) {
        let decisionIsOwed = needsDecision()
        guard !resolvedThisProcess, !isPresented, decisionIsOwed else {
            if !decisionIsOwed { resolvedThisProcess = true }
            return
        }
        triggeringValue = kind
        isPresented = true
    }

    func share() {
        resolve(enabled: true)
    }

    func decline() {
        resolve(enabled: false)
    }

    /// Closing the invitation is an explicit quiet decline. It is persisted
    /// so a user who dismissed it is never interrupted again on a later run.
    func close() {
        resolve(enabled: false)
    }

    /// Settings remains the durable control after the one-time invitation.
    /// Unlike `resolve`, this may be called repeatedly as the user changes
    /// their preference and also dismisses any invitation currently visible.
    func settingsChanged(enabled: Bool) {
        resolvedThisProcess = true
        isPresented = false
        recordDecision(enabled)
        guard enabled else { return }
        Task { await startTelemetrySession() }
    }

    private func resolve(enabled: Bool) {
        guard !resolvedThisProcess else { return }
        resolvedThisProcess = true
        isPresented = false
        recordDecision(enabled)
        guard enabled else { return }
        Task { await startTelemetrySession() }
    }
}
