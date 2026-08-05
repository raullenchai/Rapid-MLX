import SwiftUI

/// The canonical rendering of ``ServerState``: a tinted pill with a
/// status dot and one word.
///
/// The mapping from lifecycle to colour lives here and only here, so
/// "Ready is green, Starting is amber, Crashed is red, Idle is neutral"
/// is a fact about the app rather than a convention each view
/// re-implements. ``ModelPickerBar`` previously owned a private
/// `stateColor` / `stateLabel` pair; this replaces it.
///
/// Pure inputs — it takes a ``ServerState``, not a ``ServerManager`` —
/// so it renders in a snapshot harness with no live subprocess.
struct ServerStatusPill: View {
    let state: ServerState
    /// Pulse the dot while work is in flight. Suppressed automatically
    /// under Reduce Motion by ``PulsingStateDot``.
    var animatesWhenWorking: Bool = true

    var body: some View {
        HStack(spacing: RapidTheme.Space.sm - 2) {
            PulsingStateDot(color: tint, isAnimating: animatesWhenWorking && isWorking)
            Text(label)
                .font(RapidFont.secondary)
                .foregroundStyle(.primary)
                .lineLimit(1)
        }
        .padding(.horizontal, RapidTheme.Space.sm + 2)
        .padding(.vertical, RapidTheme.Space.xs + 1)
        .background(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .fill(tint.opacity(0.10))
        )
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.button, style: .continuous)
                .strokeBorder(tint.opacity(0.22), lineWidth: 0.5)
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Server status")
        .accessibilityValue(label)
    }

    /// Lifecycle → status token. The single source of truth.
    var tint: Color {
        switch state {
        case .idle, .stopped:  return RapidTheme.statusIdle
        case .missing:         return RapidTheme.statusError
        case .starting:        return RapidTheme.statusWorking
        case .ready:           return RapidTheme.statusReady
        case .crashed:         return RapidTheme.statusError
        }
    }

    /// One word. Detail (alias, progress, ETA) belongs to whatever
    /// surface owns the pill, not to the pill.
    var label: String {
        switch state {
        case .idle, .stopped: return "Idle"
        case .missing:        return "Setup needed"
        case .starting:       return "Starting"
        case .ready:          return "Ready"
        case .crashed:        return "Crashed"
        }
    }

    private var isWorking: Bool {
        if case .starting = state { return true }
        return false
    }
}
