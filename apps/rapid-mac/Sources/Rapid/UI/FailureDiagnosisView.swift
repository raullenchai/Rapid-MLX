import SwiftUI

/// Shared inline failure surface used by chat, model lifecycle, tools, and
/// downloads. The owner supplies the action handler because it owns the
/// relevant server, settings router, or download job.
struct FailureDiagnosisView: View {
    let diagnosis: FailureDiagnosis
    var onAction: ((FailureDiagnosis.Action) -> Void)? = nil
    var isActionDisabled = false
    var actionAccessibilityIdentifier: String? = nil

    var body: some View {
        ViewThatFits(in: .horizontal) {
            wideLayout
            narrowLayout
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
    }

    private var wideLayout: some View {
        HStack(alignment: .center, spacing: 8) {
            diagnosisIcon
            diagnosisText
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            Spacer(minLength: 8)
            if let action = diagnosis.action, onAction != nil {
                actionButton(action)
                    .fixedSize()
            }
        }
    }

    private var narrowLayout: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                diagnosisIcon
                diagnosisText
                Spacer(minLength: 0)
            }
            if let action = diagnosis.action, onAction != nil {
                actionButton(action)
                    .padding(.leading, 24)
            }
        }
    }

    private var diagnosisIcon: some View {
        Image(systemName: iconName)
            .foregroundStyle(diagnosis.severity == .notice ? Color.secondary : Color.orange)
            .accessibilityHidden(true)
    }

    private var diagnosisText: some View {
        Text(diagnosis.message)
            .font(.callout)
            .foregroundStyle(.primary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func actionButton(_ action: FailureDiagnosis.Action) -> some View {
        Button {
            onAction?(action)
        } label: {
            Label(action.title, systemImage: action.systemImage)
                .fixedSize(horizontal: false, vertical: true)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        .disabled(isActionDisabled)
        .accessibilityIdentifier(actionAccessibilityIdentifier ?? "")
    }

    private var accessibilityLabel: String {
        guard let action = diagnosis.action, onAction != nil else {
            return diagnosis.message
        }
        return "\(diagnosis.message) Action: \(action.title)."
    }

    private var iconName: String {
        switch diagnosis.kind {
        case .modelOutOfMemory: return "memorychip"
        case .engineNotRunning, .modelLoadFailed: return "bolt.slash"
        case .webSearchOffline, .webSearchUnavailable: return "wifi.exclamationmark"
        case .webSearchRateLimited: return "hourglass"
        case .browsePageTooLarge: return "doc.text.magnifyingglass"
        case .commandPermissionDenied, .filePermissionDenied: return "hand.raised.fill"
        // Unfilled, and tinted secondary above: the user turned this down on
        // purpose, so it reads as a note rather than a stop sign.
        case .userDeclined: return "hand.raised"
        case .fileNotFound: return "doc.questionmark"
        case .downloadFailed, .downloadSourceUnavailable: return "arrow.down.circle"
        // Unfilled and tinted secondary by ``diagnosisIcon``, for the same
        // reason ``userDeclined`` is: the user stopped this on purpose.
        case .downloadCancelled: return "stop.circle"
        case .commandFailed, .toolFailed, .requestFailed: return "exclamationmark.triangle.fill"
        }
    }
}
