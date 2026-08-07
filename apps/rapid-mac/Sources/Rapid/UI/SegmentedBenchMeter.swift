import SwiftUI

/// One labelled 5-segment benchmark bar (e.g. `Accuracy  ▓▓▓▓░  86`).
/// Issue #507. Renders a ``ResolvedMeter`` from ``ModelMeter``: a fixed
/// label, ``ModelMeter/segmentCount`` blocks with `filledSegments`
/// painted in the rating colour, and a right-aligned numeric readout.
///
/// A meter with `level == nil` (the author published no score for the
/// axis) says "Untested" — never a blank-looking track or fabricated fill.
///
/// Segmented blocks (not a continuous bar) are the deliberate style
/// choice from the #507 design review: five discrete blocks read faster
/// at a glance than a gradient fill (validated against Superwhisper).
struct SegmentedBenchMeter: View {
    let meter: ResolvedMeter
    /// Width of the label column. The card + table pin this so the bars
    /// line up vertically down a column.
    var labelWidth: CGFloat = 56
    var showValue: Bool = true

    private let segmentHeight: CGFloat = 7
    private let segmentSpacing: CGFloat = 3

    // #546: this meter renders in every table row and card, so pinned
    // sizes locked all benchmark numbers out of Dynamic Type. Scale the
    // label + value with `.caption2` while keeping the exact 10.5/10pt
    // look at the default size. The value keeps its Font-level
    // `.monospacedDigit()` (so digits stay column-aligned), which is why
    // it uses a local `@ScaledMetric` rather than `.scaledSystemFont`.
    @ScaledMetric(relativeTo: .caption2) private var labelSize: CGFloat = 10.5
    @ScaledMetric(relativeTo: .caption2) private var valueSize: CGFloat = 10

    var body: some View {
        HStack(spacing: 8) {
            Text(meter.label)
                .font(.system(size: labelSize))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(width: labelWidth, alignment: .trailing)

            if meter.level == nil {
                untestedLabel
            } else {
                segments
                if showValue {
                    Text(meter.formattedValue)
                        .font(.system(size: valueSize, weight: .medium).monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 30, alignment: .trailing)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Self.accessibilityLabel(for: meter))
    }

    private var segments: some View {
        HStack(spacing: segmentSpacing) {
            ForEach(0..<ModelMeter.segmentCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: 2.5, style: .continuous)
                    .fill(index < meter.filledSegments ? fillColor : Self.emptyTrack)
                    .frame(height: segmentHeight)
            }
        }
    }

    private var untestedLabel: some View {
        Text("Untested")
            .font(.system(size: valueSize, weight: .medium))
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var fillColor: Color {
        switch meter.level {
        case .great: return RapidTheme.brand
        case .good:  return RapidTheme.amber
        case .low, .none: return Self.lowFill
        }
    }

    private static let emptyTrack = Color.secondary.opacity(0.15)
    private static let lowFill = Color.secondary.opacity(0.55)

    /// VoiceOver phrasing so a screen-reader user gets the same signal
    /// sighted users read off the blocks. Pure + static for testability.
    static func accessibilityLabel(for meter: ResolvedMeter) -> String {
        guard let level = meter.level else {
            return "\(meter.label): untested"
        }
        let rating: String
        switch level {
        case .great: rating = "great"
        case .good:  rating = "good"
        case .low:   rating = "below average"
        }
        return "\(meter.label): \(meter.formattedValue), \(rating)"
    }
}
