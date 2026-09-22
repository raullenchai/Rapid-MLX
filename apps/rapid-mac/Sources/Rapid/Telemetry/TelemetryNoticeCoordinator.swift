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

/// Owns the one-time launch disclosure. Scheduling does not mutate consent;
/// the shared marker is attempted only after SwiftUI confirms the banner
/// actually appeared.
@MainActor
@Observable
final class TelemetryNoticeCoordinator {
    typealias Presentation = @MainActor () async -> TelemetryConsent.NoticePresentationResult
    typealias ActivationReporter = @MainActor (ProductValueKind) async -> Void

    private(set) var isPresented: Bool

    @ObservationIgnored private let recordPresentation: Presentation
    @ObservationIgnored private let startTelemetrySession: () async -> Void
    @ObservationIgnored private let reportActivation: ActivationReporter
    @ObservationIgnored private var presentationAttempted = false

    init(
        needsNotice: () -> Bool = { TelemetryConsent.needsNotice() },
        recordPresentation: @escaping Presentation = { await TelemetryConsent.noticePresented() },
        startTelemetrySession: @escaping () async -> Void = {
            await TelemetrySession.sendStartIfNeeded()
        },
        reportActivation: ActivationReporter? = nil
    ) {
        isPresented = needsNotice()
        self.recordPresentation = recordPresentation
        self.startTelemetrySession = startTelemetrySession
        self.reportActivation = reportActivation ?? { kind in
            await DesktopActivationReporter.shared.report(kind.telemetryActivationKind)
        }
    }

    func noticeDidAppear() {
        guard isPresented, !presentationAttempted else { return }
        presentationAttempted = true
        Task {
            let result = await recordPresentation()
            guard result.persisted, result.uploadAllowedThisRun else { return }
            await startTelemetrySession()
        }
    }

    func acknowledge() {
        isPresented = false
    }

    func productValueDelivered(_ kind: ProductValueKind) {
        Task { await reportActivation(kind) }
    }
}
