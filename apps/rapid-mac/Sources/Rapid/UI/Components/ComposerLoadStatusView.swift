import SwiftUI

/// The status rows the composer shows above its text field while the
/// selected model is being prepared, or has failed to start.
///
/// It sits INSIDE the existing composer card, above the same
/// ``ComposeField`` the ready state uses — never in place of it. The field
/// stays mounted and editable throughout, so a draft typed during a
/// download is exactly where the user left it when Send unlocks, and focus
/// is never moved on their behalf. Only the composer's Send action is gated.
///
/// Every word comes off ``ComposerLoadStatus``, which in turn reads
/// ``ModelReadiness``; the only thing this view decides is layout.
struct ComposerLoadStatusView: View {
    let status: ComposerLoadStatus
    var downloadActionTitle: String? = nil
    var downloadActionEnabled: Bool = true
    var onDownloadAction: () -> Void = {}
    var onAction: (ModelReadiness.Action) -> Void = { _ in }
    var onExportDiagnostics: () -> Void = {}

    var body: some View {
        HStack(alignment: .center, spacing: RapidTheme.Space.sm) {
            rows
                .frame(maxWidth: .infinity, alignment: .leading)
            if let downloadActionTitle {
                Button(downloadActionTitle, action: onDownloadAction)
                    .disabled(!downloadActionEnabled)
                    .buttonStyle(RapidSecondaryButtonStyle(height: RapidTheme.ControlHeight.small))
                    .fixedSize()
                    .help(status.phase == .stopped
                        ? String(localized: "Resumes the download using available cached data.")
                        : String(localized: "Stops the download without deleting cached files. Continuing reuses available cached data."))
                    .accessibilityIdentifier("Composer.DownloadAction")
            }
            if status.isFailure { failureActions }
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("Composer.LoadStatus")
    }

    private var rows: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
            HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
                if status.isFailure {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(RapidTheme.statusError)
                        .accessibilityHidden(true)
                }
                Text(status.title)
                    .font(status.isFailure ? RapidFont.bodyEmphasis : RapidFont.body)
                    .foregroundStyle(status.isFailure ? Color.primary : Color.secondary)
                    .lineLimit(1)
                    // A failure title carries the alias ("Couldn't start
                    // <alias>"); middle truncation keeps its quant suffix.
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if let percent = status.percentText {
                    Text(percent)
                        .font(RapidFont.bodyEmphasis)
                        .foregroundStyle(.primary)
                        // Ticks every 500 ms; monospaced digits keep the
                        // title from re-flowing as it does.
                        .monospacedDigit()
                        .fixedSize()
                }
            }
            HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.sm) {
                if let detail = status.detail {
                    Text(detail)
                        .font(RapidFont.secondary)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        // A failure message may need two lines; a model name
                        // plus byte count never should. Middle truncation
                        // keeps both ends of a long alias recognisable.
                        .lineLimit(status.isFailure ? 2 : 1)
                        .truncationMode(status.isFailure ? .tail : .middle)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    Spacer(minLength: 0)
                }
                if let eta = status.etaText {
                    Text(eta)
                        .font(RapidFont.secondary)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        .fixedSize()
                }
            }
        }
        // One readable element. Not marked `.updatesFrequently`: phase
        // changes are announced once by the composer, and byte ticks must
        // not turn into a running commentary.
        // The full, untruncated text for a pointer user; VoiceOver gets the
        // same string as the label.
        .help(status.accessibilityLabel)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(status.accessibilityLabel)
        // Without a role an ignore-children element surfaces as AXUnknown,
        // which VoiceOver reads but does not navigate as text.
        .accessibilityAddTraits(.isStaticText)
    }

    /// The same recovery the readiness notice offered for a failure —
    /// Retry / Restart / Open Startup Log, plus diagnostics — moved into
    /// the slot the failure now occupies.
    @ViewBuilder
    private var failureActions: some View {
        HStack(spacing: RapidTheme.Space.xs) {
            if let action = status.action, action.isRenderable {
                Button {
                    onAction(action)
                } label: {
                    Label(action.title, systemImage: action.systemImage)
                }
                // Secondary, not primary: inside the composer the brand
                // accent is reserved for progress, and the failure is
                // already marked by its icon and title.
                .buttonStyle(RapidSecondaryButtonStyle(height: RapidTheme.ControlHeight.small))
                .fixedSize()
                .accessibilityIdentifier("Readiness.Action")
            }
            QuietIconButton(
                symbol: "stethoscope",
                label: String(localized: "Export diagnostics…"),
                help: String(localized: "Export a privacy-scrubbed diagnostics bundle for support"),
                action: onExportDiagnostics
            )
            .accessibilityIdentifier("Readiness.ExportDiagnostics")
        }
    }
}

/// The thin progress rule along the composer card's top edge.
///
/// Brand amber is spent here and nowhere else in the slot. Determinate only
/// when the download monitor has a real fraction; otherwise a short segment
/// travels the track, which says "working" without implying a value. Under
/// Reduce Motion the segment holds still in the middle and the title carries
/// the state.
struct ComposerProgressRail: View {
    let progress: ComposerLoadStatus.Progress
    /// Spoken name of the indicator — the slot's title.
    let label: String

    static let height: CGFloat = 4
    /// Share of the track the indeterminate segment covers.
    static let segmentShare: CGFloat = 0.3
    /// Seconds for one indeterminate pass.
    static let sweepPeriod: Double = 1.6

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Rectangle().fill(RapidTheme.hairline)
                switch progress {
                case .determinate(let fraction):
                    Rectangle()
                        .fill(RapidTheme.brandPrimary)
                        .frame(width: proxy.size.width * fraction)
                        .rapidAnimation(RapidMotion.standard, value: fraction)
                case .indeterminate:
                    indeterminateSegment(trackWidth: proxy.size.width, reduceMotion: reduceMotion)
                case .none:
                    EmptyView()
                }
            }
        }
        .frame(height: Self.height)
        .clipped()
        .accessibilityRepresentation {
            // The real control, so assistive technology gets progress-
            // indicator semantics: a value only when one exists.
            switch progress {
            case .determinate(let fraction):
                ProgressView(value: fraction, total: 1) { Text(label) }
            case .indeterminate:
                ProgressView { Text(label) }
            case .none:
                EmptyView()
            }
        }
        .accessibilityIdentifier("Composer.LoadProgress")
    }

    @ViewBuilder
    func indeterminateSegment(trackWidth: CGFloat, reduceMotion: Bool) -> some View {
        let segment = max(1, trackWidth * Self.segmentShare)
        if RapidMotion.shouldPulse(isAnimating: true, reduceMotion: reduceMotion) {
            // The clock exists only while this branch does: the moment the
            // fraction arrives, the load finishes, or Reduce Motion turns
            // on, the TimelineView leaves the hierarchy and stops ticking.
            TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { context in
                Capsule()
                    .fill(RapidTheme.brandPrimary)
                    .frame(width: segment)
                    .offset(x: Self.segmentOffset(
                        at: context.date.timeIntervalSinceReferenceDate,
                        trackWidth: trackWidth,
                        segmentWidth: segment
                    ))
            }
        } else {
            Capsule()
                .fill(RapidTheme.brandPrimary)
                .frame(width: segment)
                .offset(x: (trackWidth - segment) / 2)
        }
    }

    /// Leading offset of the travelling segment. It enters from beyond the
    /// leading edge and leaves past the trailing one, so it never pops into
    /// existence at an end. Pure so the sweep is testable without a clock.
    static func segmentOffset(at time: TimeInterval, trackWidth: CGFloat, segmentWidth: CGFloat) -> CGFloat {
        let phase = time.truncatingRemainder(dividingBy: sweepPeriod) / sweepPeriod
        return (trackWidth + segmentWidth) * CGFloat(phase) - segmentWidth
    }
}
