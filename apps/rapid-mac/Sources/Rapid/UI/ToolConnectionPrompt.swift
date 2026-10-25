import Foundation
import Observation

/// Local presentation preferences are independent of telemetry consent.
@MainActor @Observable
final class ToolConnectionPrompt {
    enum Action: String, CaseIterable {
        case copyURL, copyCommand, launch

        var milestone: TelemetryEvent.Activation.Kind {
            switch self {
            case .copyURL: .toolCardCopyURL
            case .copyCommand: .toolCardCopyCommand
            case .launch: .toolCardLaunch
            }
        }
    }

    static let shared = ToolConnectionPrompt()
    private let defaults: UserDefaults
    private let report: (TelemetryEvent.Activation.Kind) -> Void
    private(set) var eligible: Bool
    private(set) var retired: Bool
    private(set) var shown: Bool
    private var visibleThisSession = false

    init(defaults: UserDefaults = .standard,
         report: @escaping (TelemetryEvent.Activation.Kind) -> Void = { kind in
             Task { await DesktopActivationReporter.shared.report(kind) }
         }) {
        self.defaults = defaults
        self.report = report
        // Preserve the earlier tip's eligibility and dismissal on upgrade.
        eligible = defaults.bool(forKey: "chatAgentTipEligible")
        retired = defaults.bool(forKey: "chatAgentTipDismissed")
            || defaults.bool(forKey: "chatAgentTipCompleted")
        shown = defaults.bool(forKey: "chatAgentTipShown")
    }

    func recordSuccess() {
        eligible = true
        defaults.set(true, forKey: "chatAgentTipEligible")
    }

    func shouldPresent(isStreaming: Bool) -> Bool {
        eligible && !retired && !isStreaming && (!shown || visibleThisSession)
    }

    func didShow() {
        guard eligible, !retired, !shown else { return }
        shown = true
        visibleThisSession = true
        defaults.set(true, forKey: "chatAgentTipShown")
        report(.toolCardShown)
    }

    func complete(_ action: Action) {
        guard !retired else { return }
        retired = true
        defaults.set(true, forKey: "chatAgentTipCompleted")
        report(action.milestone)
    }

    func dismiss() {
        guard !retired else { return }
        retired = true
        defaults.set(true, forKey: "chatAgentTipDismissed")
        report(.toolCardDismissed)
    }
}
