import SwiftUI

/// A small live-telemetry read-out: status dot + monospaced value.
///
/// The three system pills (CPU / GPU / memory), the tok-s pill, and
/// the desktop-version pill each hand-rolled this same shape with
/// their own dot size, font literal, and `.green` / `.yellow` / `.red`
/// picks. ``MetricChip`` is the one implementation, and — importantly —
/// it routes colour through ``RapidTheme`` status tokens so a "warning"
/// dot is the same amber the rest of the product uses for "working".
///
/// Monospaced is correct here (it's one of the four sanctioned uses:
/// code, endpoints, keys, metrics) and specifically stops a
/// live-updating number from re-laying-out its own row every tick.
struct MetricChip: View {
    /// Deliberately NOT spelled `none` — a case with that name shadows
    /// `Optional.none` at every `.none` call site and produces
    /// genuinely confusing inference errors.
    enum Level {
        /// No data yet. Reads as pending, not broken.
        case noData
        /// Normal / healthy.
        case ok
        /// Elevated — amber, matching the product's "working" state.
        case warning
        /// Bad — red.
        case critical

        var tint: Color {
            switch self {
            case .noData:   return RapidTheme.statusIdle.opacity(0.6)
            case .ok:       return RapidTheme.statusReady
            case .warning:  return RapidTheme.statusWorking
            case .critical: return RapidTheme.statusError
            }
        }
    }

    let label: String
    var level: Level = .noData
    /// Draw the leading status dot. Off for chips that are purely
    /// informational (a version string), on for live measurements.
    var showsDot: Bool = true

    var body: some View {
        HStack(spacing: RapidTheme.Space.xs + 1) {
            if showsDot {
                Circle()
                    .fill(level.tint)
                    .frame(width: 6, height: 6)
            }
            Text(label)
                .font(RapidFont.metric)
                .foregroundStyle(
                    level == .noData
                        ? AnyShapeStyle(HierarchicalShapeStyle.tertiary)
                        : AnyShapeStyle(HierarchicalShapeStyle.secondary)
                )
                .lineLimit(1)
        }
    }
}
