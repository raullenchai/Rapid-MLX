import SwiftUI

/// The graphite band a surface opens above its content while the app is
/// doing lifecycle work the user is waiting on — pulling weights, or
/// loading them into Metal.
///
/// ## Why this exists
///
/// A multi-gigabyte download used to be a two-line notice in a ~44pt
/// strip wedged between the transcript and the compose field: the same
/// visual weight as "this model doesn't support images". The single
/// longest wait in the product was also its quietest moment on screen,
/// and users read the silence as "nothing is happening".
///
/// The band gives that wait the priority it earns, and takes it back the
/// instant the work ends. That is the whole of the rule it implements:
/// the resting composition is for deciding, reading, and working; this
/// one is for active AI work that deserves the room. It is not a theme
/// and not a mode — nothing the user can turn on, nothing that outlives
/// the task, and nothing the sidebar or the status strip ever adopt.
///
/// ## Why horizontal
///
/// The obvious shape for "a priority area" is a column down one side.
/// That fails here for a measurable reason: chat has a 720pt reading
/// measure, and at the 720pt window floor a graphite column would be
/// taking width directly out of the conversation. Turned 90° the band
/// contests nothing — only its own height flexes (see ``height(for:)``),
/// and the transcript keeps every point of its measure at every width.
///
/// ## What it says
///
/// Nothing of its own. Every string is read off ``ModelReadiness`` — the
/// same value the composer's placeholder, the Send tooltip and the
/// empty-state subtitle read — so the band cannot describe a moment
/// differently from the surfaces around it. The percentage is the only
/// thing it formats, and only when a real fraction exists: an
/// indeterminate bar here would imply a precision the download monitor
/// does not have.
struct LifecycleBand: View {
    let readiness: ModelReadiness
    /// Bumped by the composer each time a send is attempted while gated,
    /// exactly as ``ReadinessBanner`` uses it. The band takes over the
    /// banner's slot during working states, so it has to take over the
    /// acknowledgement too — otherwise a blocked Return would go silent
    /// precisely during the longest wait in the product.
    var attentionToken: Int = 0
    /// Width of the surface the band spans, supplied by the parent.
    ///
    /// An input rather than something the band measures itself, and the
    /// distinction is load-bearing. The first version read its own width
    /// through a background ``GeometryReader`` into `@State`, which needs
    /// a SECOND layout pass to take effect — and a single-pass render
    /// (the snapshot harness, and by extension every screenshot the
    /// design gets reviewed from) never gives it one. The compact capture
    /// came out wearing the middle layout, which is exactly the kind of
    /// defect a screenshot review is supposed to catch and instead
    /// silently produced.
    ///
    /// The parent already has a definite width; passing it down makes the
    /// band's geometry a pure function of its inputs, correct on the
    /// first pass, and reproducible in a capture.
    var width: CGFloat

    @State private var attentionActive = false

    var body: some View {
        content
            .frame(maxWidth: RapidTheme.Layout.contentMaxWidth)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, RapidTheme.Space.xl)
            .frame(height: Self.height(for: width))
            .background(RapidTheme.surfaceBand)
            .overlay(alignment: .bottom) {
                // A blocked send flashes the band's leading edge rather
                // than moving anything. Reduce Motion is handled by
                // ``rapidAnimation`` — the change snaps instead of
                // fading. The cue still HAPPENS, because it is feedback
                // the user is actively waiting on, and suppressing it
                // would answer a keypress with nothing at all.
                Rectangle()
                    .fill(attentionActive ? RapidTheme.brandPrimary : RapidTheme.bandTrack)
                    .frame(height: attentionActive ? 2 : 1)
            }
            .rapidAnimation(RapidMotion.quick, value: attentionActive)
            .accessibilityElement(children: .contain)
            .accessibilityLabel(readiness.accessibilityLabel)
            .accessibilityIdentifier("Readiness.Band")
            .task(id: attentionToken) {
                // Token 0 is the initial value — only a real bump should
                // flash, or the band would emphasise itself the moment it
                // opens.
                guard attentionToken > 0 else { return }
                attentionActive = true
                try? await Task.sleep(nanoseconds: 1_400_000_000)
                guard !Task.isCancelled else { return }
                attentionActive = false
            }
    }

    @ViewBuilder
    private var content: some View {
        if isCompact {
            // At the 720pt window floor (a 520pt detail pane after the
            // sidebar) the band collapses to one line plus its
            // rule. The detail line is the first thing dropped, because
            // it is the one part the composer placeholder already
            // paraphrases — and dropping the headline instead would take
            // the model's name with it.
            VStack(alignment: .leading, spacing: RapidTheme.Space.xs + 1) {
                HStack(spacing: RapidTheme.Space.md) {
                    Text(readiness.headline)
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(RapidTheme.bandInk)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let percent = percentText {
                        Text(percent)
                            .font(RapidFont.bandEyebrow)
                            .foregroundStyle(RapidTheme.brandPrimary)
                            .monospacedDigit()
                            .fixedSize()
                    }
                }
                progressTrack
            }
        } else {
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                HStack(alignment: .lastTextBaseline, spacing: RapidTheme.Space.lg) {
                    Text(readiness.headline)
                        .font(RapidFont.bandTitle)
                        .foregroundStyle(RapidTheme.bandInk)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let percent = percentText {
                        Text(percent)
                            .font(RapidFont.bandMetric)
                            .foregroundStyle(RapidTheme.brandPrimary)
                            .monospacedDigit()
                            .fixedSize()
                    }
                }
                progressTrack
                if let detail = readiness.detail {
                    Text(detail)
                        // Byte counts and ETAs tick every 500 ms;
                        // monospaced digits stop the line rewriting its
                        // own width as they do.
                        .font(RapidFont.metric)
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .monospacedDigit()
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    /// The 4pt rule. Determinate when the download monitor has a real
    /// fraction; a bare track otherwise, which reads honestly as "work is
    /// happening and we cannot say how far in" rather than as a bar stuck
    /// at zero.
    private var progressTrack: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule(style: .continuous)
                    .fill(RapidTheme.bandTrack)
                if let fraction = readiness.progressFraction {
                    Capsule(style: .continuous)
                        .fill(RapidTheme.brandPrimary)
                        .frame(width: proxy.size.width * min(max(fraction, 0), 1))
                }
            }
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }

    /// Whole percent, or `nil` when no real fraction exists.
    ///
    /// Static and pure so the rounding and the clamp can be pinned
    /// directly — a fraction slightly over 1.0 (the byte monitor can
    /// overshoot on the last chunk) must read "100%", never "101%".
    static func percentText(for fraction: Double?) -> String? {
        guard let fraction else { return nil }
        return "\(Int((min(max(fraction, 0), 1) * 100).rounded()))%"
    }

    private var percentText: String? { Self.percentText(for: readiness.progressFraction) }

    // MARK: - Geometry

    /// Only the band's HEIGHT responds to window width — never its width,
    /// which is the property that makes a horizontal band safe beside a
    /// fixed reading measure.
    ///
    /// Static and pure so the three steps can be pinned by a test without
    /// standing up a window at three sizes.
    static func height(for width: CGFloat) -> CGFloat {
        if width >= RapidTheme.Layout.Breakpoint.wide { return 132 }
        if width >= RapidTheme.Layout.Breakpoint.mid { return 112 }
        return 44
    }

    /// True at the compact step, where the band is one line plus its rule.
    static func isCompact(width: CGFloat) -> Bool {
        height(for: width) <= 44
    }

    private var isCompact: Bool { Self.isCompact(width: width) }
}
