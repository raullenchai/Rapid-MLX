import Foundation

/// Pure logic behind the two compact benchmark meters (Accuracy · Speed)
/// shown on every "All models" row. Issue #507.
///
/// The management surface only has room for two meters per row,
/// so this collapses the four quality axes into a single **primary
/// quality** axis + the **speed** axis, and turns each into a
/// 5-segment fill + a great/good/low rating.
///
/// Kept free of SwiftUI so every branch is unit-testable without a view
/// host; ``SegmentedBenchMeter`` maps the ``MeterLevel`` this returns to
/// the ``RapidTheme`` colours.
enum MeterLevel: String, Equatable, Sendable {
    /// At or above the axis's `great` threshold — brand-accent fill.
    case great
    /// At or above `good`, below `great` — amber fill.
    case good
    /// Below `good` — muted grey fill.
    case low
}

/// One resolved meter: which axis it represents, the label to show, the
/// number of filled segments (0…5), the rating, and the formatted value.
/// ``value``/``level`` are ``nil`` when the model author didn't publish
/// the axis — the view renders an explicit "Untested" state (never a
/// fabricated number, per the standing benchmark-honesty policy).
struct ResolvedMeter: Equatable, Sendable {
    let axis: BenchScores.Axis
    let label: String
    let filledSegments: Int
    let level: MeterLevel?
    let formattedValue: String
}

enum ModelMeter {
    /// Total segments in a meter. Five matches Superwhisper's density
    /// and the mockup; kept as one constant so a future 3- or 7-block
    /// restyle is a one-line change + a test update.
    static let segmentCount = 5

    /// Filled-segment count for a value on an axis, 0…``segmentCount``.
    /// Uses the axis's own normalizer (Speed = 300 t/s; the four quality
    /// axes = 100), so the fill is honest to each axis's scale.
    static func segments(value: Double, axis: BenchScores.Axis) -> Int {
        let normalizer = axis.thresholds.normalizer
        guard normalizer > 0 else { return 0 }
        let pct = max(0.0, min(1.0, value / normalizer))
        return Int((pct * Double(segmentCount)).rounded())
    }

    /// great / good / low classification for a value on an axis, using
    /// the axis's own ``thresholds`` so every surface that reads them
    /// (the "All models" meters) agrees on a model's rating.
    static func level(value: Double, axis: BenchScores.Axis) -> MeterLevel {
        let t = axis.thresholds
        if value >= t.great { return .great }
        if value >= t.good { return .good }
        return .low
    }

    /// The short label for the compact meter — shorter than
    /// ``BenchScores.Axis.label`` ("General & Reasoning" → "Accuracy")
    /// because the row column is narrow.
    static func shortLabel(for axis: BenchScores.Axis) -> String {
        switch axis {
        case .generalReasoning: return "Accuracy"
        case .code:             return "Code"
        case .tool:             return "Tool"
        case .ifeval:           return "Instructions"
        case .speed:            return "Speed"
        }
    }

    /// Right-aligned numeric readout: Speed as an integer ("158"),
    /// the quality axes rounded to a whole number ("86"). Kept
    /// suffix-free — the header column already says "Accuracy · Speed"
    /// and "t/s" lives in the legend, so the cell stays uncluttered.
    static func formatted(value: Double, axis: BenchScores.Axis) -> String {
        "\(Int(value.rounded()))"
    }

    /// Pick the single quality axis to headline. Prefers General &
    /// Reasoning (the all-around signal); for a coder alias whose
    /// author publishes only a code score (`qwen3-coder`, `devstral`),
    /// General & Reasoning is `nil` so we fall through to Code, then
    /// Tool, then Instruction Following. Returns `nil` only when the
    /// alias has no quality score at all.
    static func primaryQualityAxis(_ scores: BenchScores) -> BenchScores.Axis? {
        if scores.generalReasoning != nil { return .generalReasoning }
        if scores.code != nil { return .code }
        if scores.tool != nil { return .tool }
        if scores.ifeval != nil { return .ifeval }
        return nil
    }

    /// Resolve the quality meter for an alias. Returns a meter whose
    /// `value`/`level` are `nil` when no quality axis is
    /// published, so the caller always has a labelled row to render.
    static func qualityMeter(for alias: String) -> ResolvedMeter {
        guard let scores = BenchScoresCatalog.lookup(alias: alias),
              let axis = primaryQualityAxis(scores),
              let value = scores.value(for: axis) else {
            // Default the label to "Accuracy" so an unscored row still
            // reads as a quality meter rather than a blank.
            return ResolvedMeter(
                axis: .generalReasoning,
                label: shortLabel(for: .generalReasoning),
                filledSegments: 0,
                level: nil,
                formattedValue: "Untested"
            )
        }
        return ResolvedMeter(
            axis: axis,
            // Label the meter with the axis that actually drives the value.
            // Most models resolve to General & Reasoning → "Accuracy", but a
            // coder alias that publishes only a code score (generalReasoning
            // == nil) resolves to Code → "Code". Labelling the true axis
            // stops a coding pick from reading as "Accuracy 29" — which looks
            // like a failing general-correctness grade — when 29 is really its
            // honest coding-bench number. The meters column header is the
            // axis-agnostic "Quality · Speed"; the full per-axis breakdown
            // lives in the picker's benchmark tooltip.
            label: shortLabel(for: axis),
            filledSegments: segments(value: value, axis: axis),
            level: level(value: value, axis: axis),
            formattedValue: formatted(value: value, axis: axis)
        )
    }

    /// Resolve the speed meter for an alias. `nil` / "Untested" when the
    /// alias has no measured decode tok/s (many vision + newest models).
    static func speedMeter(for alias: String) -> ResolvedMeter {
        guard let scores = BenchScoresCatalog.lookup(alias: alias),
              let value = scores.speedTps else {
            return ResolvedMeter(
                axis: .speed,
                label: shortLabel(for: .speed),
                filledSegments: 0,
                level: nil,
                formattedValue: "Untested"
            )
        }
        return ResolvedMeter(
            axis: .speed,
            label: shortLabel(for: .speed),
            filledSegments: segments(value: value, axis: .speed),
            level: level(value: value, axis: .speed),
            formattedValue: formatted(value: value, axis: .speed)
        )
    }
}
