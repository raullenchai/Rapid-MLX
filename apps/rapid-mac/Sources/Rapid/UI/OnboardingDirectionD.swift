import SwiftUI

/// The Direction D visual system for first-run setup (Paper 05.1 / 05.2).
///
/// ## What this file is
///
/// Paper 05.1.A defines setup as a *full-window application experience*
/// composed of two planes, and every one of the twenty registered states is a
/// variation on that pair:
///
/// * **The rail owns where you are, and what this Mac is.** SET UP, the four
///   steps with current / completed states, and the machine facts that bear on
///   the decision. During the blocking lifecycle states it owns the lifecycle
///   name, the model identity, real progress and a measured ETA instead.
/// * **The canvas owns the decision, and what the action will do.** The active
///   choice, its explanation, warnings, whether leaving is safe, and the
///   recovery actions. Never a second copy of what the rail already says.
///
/// D1 asks; D2 waits. D1 is the mineral rail — Welcome, detection, the
/// chooser, warnings and recoverable failures. D2 is the full-height graphite
/// subject rail — Download, Preparing, Starting: the states where the user
/// cannot act and the honest thing to show is the work itself.
///
/// ## Why a separate file
///
/// ``QuickstartView`` is the behaviour: a coordinator, a download observer, a
/// serve observer, a pre-flight, a diagnosis table and a completion
/// transaction. Mixing a full visual language into it would make both harder
/// to review, and the visual layer has to be replaceable without anyone
/// re-reading the state machine. Everything here is a host-free ``View`` over
/// ``RapidTheme``, so a screen can be composed — and snapshot-rendered —
/// without a running app.
///
/// ## Tokens
///
/// Nothing here invents a colour. ``RapidTheme`` already carries the whole
/// Direction D palette at the exact values Paper names, with Light and Dark
/// resolved per token, so this file maps roles onto existing names and adds
/// only the geometry Paper specifies.
enum OnboardingD {

    // MARK: - Geometry

    /// D1 rail width at full size (Paper 05.1.B, 1440).
    static let railWidth: CGFloat = 300
    /// D1 rail width once the window can no longer afford 300 (Paper 05.1.D,
    /// 1000×700). The rail keeps every element; it just gives width back.
    static let railNarrowWidth: CGFloat = 240

    /// At or above this window width the Step 2 heading sits BESIDE the list
    /// (Paper 05.1.B). Below it, Paper stacks them — see 05.1.D, where the
    /// 1000pt chooser runs kicker, title, subtitle and list down one column.
    ///
    /// The number is derived, not guessed. Side-by-side costs the rail (300),
    /// the canvas gutters (96 + 56), the heading column (360) and the column
    /// gap (56) — 868pt before a single model row is drawn. Below
    /// 1290 the rows fall under ``rowMinWidth`` and the alias, repo, badge and
    /// size lanes start colliding, so the heading stacks above the list
    /// instead (Paper 05.1.D draws exactly that at 1000).
    static let columnsMinWidth: CGFloat = 1290

    /// The narrowest a catalogue row can be and still hold its four lanes.
    static let rowMinWidth: CGFloat = 420
    /// D2 subject rail width (Paper: 340). Wider because it carries a 64pt
    /// percentage and a byte line that must not wrap.
    static let subjectRailWidth: CGFloat = 340

    /// Top inset before rail content begins.
    ///
    /// Paper's frames mock a whole window, so they draw a 52pt titlebar with
    /// traffic lights inside the rail. Setup is the window's CONTENT view, and
    /// macOS has already drawn the real title bar above it — so the 52pt is
    /// spoken for and painting a second set of dots would be decoration that
    /// cannot be clicked. What is left is the breathing room Paper puts
    /// between the titlebar and the brand mark.
    static let railTopInset: CGFloat = 24

    /// Canvas top inset, measured the same way: Paper's 56 runs from the top
    /// of a mocked window, and roughly 28 of that is the title bar the system
    /// now provides.
    static let canvasTop: CGFloat = 32

    /// Canvas insets. Asymmetric by design: Direction D leads from a deep left
    /// margin and lets content run out to a tighter right edge.
    static let canvasBottom: CGFloat = 44
    static let canvasLeading: CGFloat = 96
    static let canvasTrailing: CGFloat = 56

    /// Content width caps Paper uses per composition.
    static let decisionWidth: CGFloat = 660
    static let proseWidth: CGFloat = 640
    /// The chooser's fixed heading column.
    static let headingColumnWidth: CGFloat = 360
    /// Gap between the heading column and the choice list.
    static let columnGap: CGFloat = 56

    /// Below this window width the vertical rail would leave the canvas around
    /// 400pt — unusable for the model cards — so it rotates into a horizontal
    /// strip and the canvas takes the full width (Paper 05.1.E).
    static let railCollapseWidth: CGFloat = 820

    /// The collapsed rail's height, and the D2 band's.
    static let compactRailHeight: CGFloat = 46
    static let compactBandHeight: CGFloat = 56
    /// The canvas gutter once the rail is horizontal.
    static let compactGutter: CGFloat = 28

    /// Standard control height for the action lane.
    static let actionHeight: CGFloat = 44
    static let actionRadius: CGFloat = 10
    static let cardRadius: CGFloat = 12

    /// Catalogue row geometry — fixed slots so badges and sizes form vertical
    /// lanes across rows rather than drifting with name length.
    static let rowHeight: CGFloat = 56
    static let rowBadgeSlot: CGFloat = 172
    static let rowSizeSlot: CGFloat = 76
    static let selectionGlyph: CGFloat = 18

    // MARK: - Type

    /// Tracking values Paper specifies in em, converted to points at the size
    /// they are used. SwiftUI's `.tracking` is absolute, so these cannot be
    /// shared across sizes.
    enum Tracking {
        /// 0.14em label tracking on an 11pt kicker.
        static let kicker: CGFloat = 1.54
        /// 0.12em on a 10pt group label.
        static let groupLabel: CGFloat = 1.2
        /// 0.08em on a 9pt badge.
        static let badge: CGFloat = 0.72
        /// 0.16em on the D2 lifecycle name.
        static let lifecycle: CGFloat = 1.76
        /// Display tracking, applied per display size below.
        static func display(_ size: CGFloat) -> CGFloat { size * -0.026 }
    }
}

// MARK: - Step titles

extension QuickstartCoordinator.Step {
    /// The rail's name for this step (Paper 05.1.A — "SET UP · Welcome ·
    /// Choose a model · Download · Start").
    ///
    /// Presentation only, and deliberately an extension rather than a stored
    /// property on the coordinator: the four-step MODEL is behaviour and is
    /// owned there; what the rail calls each step is this file's business.
    var railTitle: String {
        switch self {
        case .welcome:     return "Welcome"
        case .chooseModel: return "Choose a model"
        case .download:    return "Download"
        case .start:       return "Start"
        }
    }
}

// MARK: - Rail

/// One step in the setup rail, and how it is drawn.
enum OnboardingStepState: Equatable {
    case complete
    case current
    case upcoming
}

/// The D1 setup rail (Paper 05.1.A, and every D1 frame in 05.1.B).
///
/// Replaces the production sidebar for the duration of setup. It never shows
/// chat, a composer or a status strip: setup owns the window until it ends.
struct OnboardingSetupRail: View {
    @Environment(\.onboardingLayout) private var layout
    /// The macro step the user is on. Steps before it read complete, steps
    /// after it read upcoming.
    let step: QuickstartCoordinator.Step
    /// This Mac, for the footer facts. `nil` omits the footer entirely rather
    /// than printing placeholders for numbers we do not have.
    var hardware: MacHardware?
    /// Free space on the volume that holds the model cache, once measured.
    ///
    /// Paper adds this row from Step 2 onward (05.1 frames 03 and 04): once
    /// the user is choosing what to download, how much room there is becomes
    /// one of the machine facts the decision rests on. `nil` omits the row.
    var freeSpace: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Color.clear.frame(height: OnboardingD.railTopInset)

            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 9) {
                    OnboardingRailMark()
                    Text("Rapid-MLX")
                        .scaledSystemFont(13, weight: .semibold)
                        .foregroundStyle(RapidTheme.textPrimary)
                }
                Text("SET UP")
                    .scaledSystemFont(11, relativeTo: .caption, weight: .semibold, design: .monospaced)
                    .tracking(OnboardingD.Tracking.kicker)
                    .foregroundStyle(RapidTheme.textTertiary)
            }
            .padding(.horizontal, 24)
            .padding(.top, 12)
            .padding(.bottom, 22)

            VStack(alignment: .leading, spacing: 2) {
                ForEach(QuickstartCoordinator.Step.allCases, id: \.rawValue) { entry in
                    OnboardingRailStepRow(step: entry, state: state(for: entry))
                }
            }
            .padding(.horizontal, 16)
            // The rail is a progress report, not a control: reading it as one
            // element keeps VoiceOver from offering four inert rows.
            .accessibilityElement(children: .ignore)
            .accessibilityIdentifier("Quickstart.Progress")
            .accessibilityLabel(Self.progressAccessibilityLabel(current: step))
            .accessibilityValue(Self.progressAccessibilityValue(current: step))

            Spacer(minLength: 0)

            if let hardware {
                OnboardingRailFacts(
                    hardware: hardware,
                    freeSpace: freeSpace,
                    isNarrow: layout != .wide
                )
            }
        }
        .frame(width: layout.railWidth)
        .frame(maxHeight: .infinity)
        .background(RapidTheme.surfaceSidebar)
        .overlay(alignment: .trailing) {
            Rectangle().fill(RapidTheme.hairline).frame(width: 1)
        }
    }

    private func state(for entry: QuickstartCoordinator.Step) -> OnboardingStepState {
        if entry.rawValue < step.rawValue { return .complete }
        if entry == step { return .current }
        return .upcoming
    }

    /// Spoken name of the rail.
    ///
    /// Kept EXACTLY at `Setup progress, step N of M`. This string is a
    /// contract, not styling: `gui-golden-flows.sh` matches it verbatim on
    /// three screens to prove the rail reports honest progress, and PR #1917
    /// put it there. Anything richer belongs in the value below, where it can
    /// grow without breaking the pin.
    static func progressAccessibilityLabel(current: QuickstartCoordinator.Step) -> String {
        "Setup progress, step \(current.displayNumber) of \(QuickstartCoordinator.Step.total)"
    }

    /// What the rail says beyond its position: the step's name, and which
    /// steps are already behind the user. VoiceOver reads a value after a
    /// label, so this arrives in the right order without disturbing the pin.
    static func progressAccessibilityValue(current: QuickstartCoordinator.Step) -> String {
        let done = QuickstartCoordinator.Step.allCases
            .prefix(current.rawValue)
            .map(\.railTitle)
        var value = current.railTitle
        if !done.isEmpty {
            value += ". Completed: \(done.joined(separator: ", "))"
        }
        return value
    }
}

/// The brand mark in the rail head: an ink tile with an amber R.
private struct OnboardingRailMark: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 5, style: .continuous)
            .fill(RapidTheme.textPrimary)
            .frame(width: 22, height: 22)
            .overlay {
                Text("R")
                    .scaledSystemFont(13, relativeTo: .caption, weight: .bold)
                    .foregroundStyle(RapidTheme.brandPrimary)
            }
            .accessibilityHidden(true)
    }
}

private struct OnboardingRailStepRow: View {
    let step: QuickstartCoordinator.Step
    let state: OnboardingStepState

    var body: some View {
        HStack(spacing: 12) {
            marker.frame(width: 16)
            Text(step.railTitle)
                .scaledSystemFont(13, weight: state == .current ? .semibold : .regular)
                .foregroundStyle(state == .current ? RapidTheme.textPrimary : RapidTheme.textTertiary)
            Spacer(minLength: 0)
        }
        .padding(.leading, state == .current ? 9 : 12)
        .frame(height: 34)
        .background(alignment: .leading) {
            if state == .current {
                UnevenRoundedRectangle(
                    topLeadingRadius: 0,
                    bottomLeadingRadius: 0,
                    bottomTrailingRadius: 6,
                    topTrailingRadius: 6,
                    style: .continuous
                )
                .fill(RapidTheme.textPrimary.opacity(0.043))
                .overlay(alignment: .leading) {
                    Rectangle().fill(RapidTheme.textPrimary).frame(width: 3)
                }
            }
        }
    }

    @ViewBuilder
    private var marker: some View {
        switch state {
        case .complete:
            Text("✓")
                .scaledSystemFont(11, relativeTo: .caption, weight: .semibold, design: .monospaced)
                .foregroundStyle(RapidTheme.statusReady)
        case .current:
            Text("\(step.displayNumber)")
                .scaledSystemFont(11, relativeTo: .caption, weight: .semibold, design: .monospaced)
                .foregroundStyle(RapidTheme.textPrimary)
        case .upcoming:
            Text("\(step.displayNumber)")
                .scaledSystemFont(11, relativeTo: .caption, weight: .medium, design: .monospaced)
                .foregroundStyle(RapidTheme.textTertiary)
        }
    }
}

/// THIS MAC — the machine facts that bear on the model decision.
private struct OnboardingRailFacts: View {
    let hardware: MacHardware
    var freeSpace: String?
    /// At 240pt the two-column `label … value` rows start truncating the chip
    /// name, so Paper collapses them to a single line (05.1.D).
    var isNarrow: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("THIS MAC")
                .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                .tracking(OnboardingD.Tracking.groupLabel)
                .foregroundStyle(RapidTheme.textTertiary)
            if isNarrow {
                // Paper 05.1.D: at 240pt the rail states the machine as one
                // full-width line and drops the free-space row. A label/value
                // pair at this width truncates the chip name, which is the one
                // thing the line exists to say.
                Text("\(Self.memoryText(hardware)) · \(hardware.brandString)")
                    .scaledSystemFont(12, design: .monospaced)
                    .foregroundStyle(RapidTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let freeSpace {
                // From Step 2 onward the decision rests on room as well as
                // memory, so the chip joins the memory line to make space for
                // it (Paper 05.1 frame 04).
                row("Memory", "\(Self.memoryText(hardware)) · \(hardware.brandString)")
                row("Free space", freeSpace)
            } else {
                row("Chip", hardware.brandString)
                row("Memory", Self.memoryText(hardware))
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 18)
        .padding(.bottom, 22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .top) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("Quickstart.Rail.ThisMac")
        .accessibilityLabel(Self.spokenFacts(hardware: hardware, freeSpace: freeSpace))
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .scaledSystemFont(12)
                .foregroundStyle(RapidTheme.textSecondary)
            Spacer(minLength: 8)
            Text(value)
                .scaledSystemFont(12, design: .monospaced)
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    /// Whole gibibytes, matching how macOS About reports memory. Pure so the
    /// rounding is pinned rather than re-derived at the call site.
    static func memoryText(_ hardware: MacHardware) -> String {
        "\(Int(hardware.physicalRAMGB.rounded())) GB"
    }

    /// The footer read aloud as one sentence, so VoiceOver gets the machine
    /// facts in one stop rather than three unlabelled fragments.
    static func spokenFacts(hardware: MacHardware, freeSpace: String?) -> String {
        var text = "This Mac. Chip \(hardware.brandString). Memory \(memoryText(hardware))."
        if let freeSpace { text += " Free space \(freeSpace)." }
        return text
    }
}

// MARK: - D2 subject rail

/// The graphite lifecycle rail (Paper 05.1.A "D2 waits", state 09/14/15).
///
/// Everything on it is measured. The percentage, the byte line and the rate
/// are rendered only when the caller has real values to pass; a `nil` becomes
/// an absent row, never a zero or a guess.
struct OnboardingSubjectRail: View {
    /// The lifecycle name — DOWNLOADING, PREPARING, STARTING.
    let lifecycle: String
    /// Model identity, as an alias rather than a display name: this rail is
    /// the technical plane.
    let identity: String
    /// Measured fraction 0...1, or `nil` for an indeterminate stage.
    var fraction: Double?
    /// "271 MB / 633 MB", or `nil` before bytes are observed.
    var bytesLine: String?
    /// "4.4 MB/s · 1 min left", or `nil` until the rate stabilises.
    var rateLine: String?
    /// Footer left — "STEP 3 OF 4".
    let stepLabel: String
    /// Footer right — the macro step name.
    let stepName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Color.clear.frame(height: OnboardingD.railTopInset)

            VStack(alignment: .leading, spacing: 0) {
                Text(lifecycle)
                    .scaledSystemFont(11, relativeTo: .caption, weight: .semibold, design: .monospaced)
                    .tracking(OnboardingD.Tracking.lifecycle)
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .padding(.bottom, 20)

                Text(identity)
                    .scaledSystemFont(19, relativeTo: .title3, weight: .medium, design: .monospaced)
                    .foregroundStyle(RapidTheme.bandInk)
                    .fixedSize(horizontal: false, vertical: true)

                if let fraction {
                    Text(Self.percentText(fraction))
                        .scaledSystemFont(64, relativeTo: .largeTitle, weight: .semibold)
                        .monospacedDigit()
                        .foregroundStyle(RapidTheme.brandPrimary)
                        .padding(.top, 26)
                        .accessibilityIdentifier("Quickstart.Subject.Percent")

                    OnboardingSubjectTrack(fraction: fraction)
                        .padding(.top, 16)
                } else {
                    OnboardingIndeterminateTrack()
                        .padding(.top, 26)
                }

                if let bytesLine {
                    Text(bytesLine)
                        .scaledSystemFont(13, design: .monospaced)
                        .foregroundStyle(RapidTheme.bandInk)
                        .padding(.top, 16)
                        .accessibilityIdentifier("Quickstart.Subject.Bytes")
                }
                if let rateLine {
                    Text(rateLine)
                        .scaledSystemFont(12, design: .monospaced)
                        .foregroundStyle(RapidTheme.bandInkSecondary)
                        .padding(.top, 7)
                        .accessibilityIdentifier("Quickstart.Subject.Rate")
                }

                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 28)
            .padding(.top, 22)

            HStack {
                Text(stepLabel)
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                Spacer(minLength: 8)
                Text(stepName)
                    .foregroundStyle(RapidTheme.bandInk)
            }
            .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
            .tracking(OnboardingD.Tracking.groupLabel)
            .padding(.horizontal, 28)
            .padding(.top, 20)
            .padding(.bottom, 24)
            .overlay(alignment: .top) {
                Rectangle().fill(Color.white.opacity(0.09)).frame(height: 1)
            }
        }
        .frame(width: OnboardingD.subjectRailWidth)
        .frame(maxHeight: .infinity)
        .background(RapidTheme.surfaceBand)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("Quickstart.Subject.Rail")
    }

    /// Whole percent, floored, so the rail never rounds 99.6% up to a "100%"
    /// that sits there while files are still being written.
    static func percentText(_ fraction: Double) -> String {
        let clamped = min(max(fraction, 0), 1)
        return "\(Int(clamped * 100))%"
    }
}

private struct OnboardingSubjectTrack: View {
    let fraction: Double

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(RapidTheme.bandTrack)
                Capsule()
                    .fill(RapidTheme.brandPrimary)
                    .frame(width: max(0, min(1, fraction)) * proxy.size.width)
            }
        }
        .frame(height: 5)
        .accessibilityHidden(true)
    }
}

/// Indeterminate stages get a travelling segment, never a static partial bar
/// and never 0% (Paper state 14: "No number at all"). Reduced Motion pulses
/// the track instead of moving anything.
private struct OnboardingIndeterminateTrack: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase: CGFloat = -0.4

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(RapidTheme.bandTrack)
                if reduceMotion {
                    Capsule()
                        .fill(RapidTheme.brandPrimary.opacity(0.55))
                } else {
                    Capsule()
                        .fill(RapidTheme.brandPrimary)
                        .frame(width: proxy.size.width * 0.4)
                        .offset(x: phase * proxy.size.width)
                }
            }
        }
        .frame(height: 5)
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 1.4).repeatForever(autoreverses: false)) {
                phase = 1.0
            }
        }
        .accessibilityHidden(true)
    }
}

// MARK: - Compact rail

/// The rail, rotated (Paper 05.1.E).
///
/// Below ``OnboardingD/railCollapseWidth`` the vertical rail would squeeze the
/// canvas to roughly 400pt, which the model cards cannot use. It becomes a
/// 46pt strip instead: the same three facts — that this is setup, which step,
/// and how far along — in a horizontal reading order. Nothing is dropped; the
/// machine facts move into the screens that actually need them.
struct OnboardingCompactRail: View {
    let step: QuickstartCoordinator.Step

    var body: some View {
        HStack(spacing: 12) {
            Text("SET UP")
                .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                .tracking(OnboardingD.Tracking.kicker)
                .foregroundStyle(RapidTheme.textTertiary)
            Text(step.railTitle)
                .scaledSystemFont(13, weight: .semibold)
                .foregroundStyle(RapidTheme.textPrimary)
                .lineLimit(1)
            Spacer(minLength: 8)
            Text("STEP \(step.displayNumber) OF \(QuickstartCoordinator.Step.total)")
                .scaledSystemFont(10, relativeTo: .caption2, design: .monospaced)
                .foregroundStyle(RapidTheme.textTertiary)
                .fixedSize()
            HStack(spacing: 4) {
                ForEach(QuickstartCoordinator.Step.allCases, id: \.rawValue) { entry in
                    Capsule()
                        .fill(entry.rawValue <= step.rawValue
                              ? RapidTheme.textPrimary
                              : RapidTheme.hairlineStrong)
                        .frame(width: 26, height: 4)
                }
            }
            .fixedSize()
        }
        .padding(.horizontal, 20)
        .frame(height: OnboardingD.compactRailHeight)
        .frame(maxWidth: .infinity)
        .background(RapidTheme.surfaceSidebar)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityIdentifier("Quickstart.Progress")
        .accessibilityLabel(OnboardingSetupRail.progressAccessibilityLabel(current: step))
        .accessibilityValue(OnboardingSetupRail.progressAccessibilityValue(current: step))
    }
}

/// The D2 rail, rotated: a 56pt graphite band in the same slot.
struct OnboardingCompactSubjectBand: View {
    let lifecycle: String
    let identity: String
    var fraction: Double?
    var bytesLine: String?

    var body: some View {
        HStack(spacing: 14) {
            Text(lifecycle)
                .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                .tracking(OnboardingD.Tracking.groupLabel)
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .fixedSize()
            Text(identity)
                .scaledSystemFont(13, design: .monospaced)
                .foregroundStyle(RapidTheme.bandInk)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 8)
            if let bytesLine {
                Text(bytesLine)
                    .scaledSystemFont(11, design: .monospaced)
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .fixedSize()
            }
            if let fraction {
                Text(OnboardingSubjectRail.percentText(fraction))
                    .scaledSystemFont(15, weight: .semibold)
                    .monospacedDigit()
                    .foregroundStyle(RapidTheme.brandPrimary)
                    .fixedSize()
                    .accessibilityIdentifier("Quickstart.Subject.Percent")
            }
        }
        .padding(.horizontal, 20)
        .frame(height: OnboardingD.compactBandHeight)
        .frame(maxWidth: .infinity)
        .background(RapidTheme.surfaceBand)
        .overlay(alignment: .bottom) {
            if let fraction {
                GeometryReader { proxy in
                    Rectangle()
                        .fill(RapidTheme.brandPrimary)
                        .frame(width: max(0, min(1, fraction)) * proxy.size.width)
                }
                .frame(height: 3)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("Quickstart.Subject.Rail")
    }
}

// MARK: - Canvas scaffold

/// How much room setup has, in the three shapes Paper draws it at.
///
/// Decided once by the shell from the real window width, then read from the
/// environment, so no screen re-measures and none can disagree with another
/// about which side of a breakpoint it is on.
enum OnboardingLayout: Equatable {
    /// 1440-class. Full 300pt rail; Step 2's heading sits beside the list.
    case wide
    /// 1000-class. Rail narrows to 240 and the canvas stacks (Paper 05.1.D).
    case medium
    /// Below 820. The rail rotates into a strip (Paper 05.1.E).
    case compact

    static func resolve(width: CGFloat) -> OnboardingLayout {
        if width < OnboardingD.railCollapseWidth { return .compact }
        if width < OnboardingD.columnsMinWidth { return .medium }
        return .wide
    }

    /// True once the rail is horizontal.
    var isCompact: Bool { self == .compact }

    /// Only the widest tier can afford the side-by-side heading.
    var usesColumns: Bool { self == .wide }

    var railWidth: CGFloat {
        self == .wide ? OnboardingD.railWidth : OnboardingD.railNarrowWidth
    }
}

private struct OnboardingLayoutKey: EnvironmentKey {
    static let defaultValue = OnboardingLayout.wide
}

extension EnvironmentValues {
    var onboardingLayout: OnboardingLayout {
        get { self[OnboardingLayoutKey.self] }
        set { self[OnboardingLayoutKey.self] = newValue }
    }
}

/// The right-hand plane. Applies Paper's asymmetric insets once so no screen
/// re-states them, and lets a screen override the trailing inset when its
/// content runs wider (the catalogue does). Once the rail has rotated, both
/// insets collapse to the compact gutter and the asymmetry goes with them —
/// there is no rail to lead away from any more.
struct OnboardingCanvas<Content: View>: View {
    @Environment(\.onboardingLayout) private var layout
    var trailing: CGFloat = OnboardingD.canvasTrailing
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(.top, layout.isCompact ? OnboardingD.compactGutter : OnboardingD.canvasTop)
            .padding(.bottom, layout.isCompact ? OnboardingD.compactGutter : OnboardingD.canvasBottom)
            .padding(.leading, layout.isCompact ? OnboardingD.compactGutter : OnboardingD.canvasLeading)
            .padding(.trailing, layout.isCompact ? OnboardingD.compactGutter : trailing)
            .background(RapidTheme.surfaceCanvas)
    }
}

/// The canvas's vertical composition, in one place (Paper 05.1.B).
///
/// Every Paper canvas is the same two-part shape: a **principal group** that
/// takes the remaining height and centres its content in it, and an optional
/// **action lane** anchored beneath. In the frames this is `flexGrow: 1` with
/// `alignItems: center` on the content, then the footer with a `marginTop`.
///
/// It lives here rather than in each screen because the alternative — a
/// per-state top padding — is what produced the defect it replaces: the states
/// drifted apart, Welcome sat near centre and Step 2 sat pinned to the top,
/// and there was no single number to correct because each screen had its own.
struct OnboardingCanvasLayout<Principal: View, Footer: View>: View {
    @ViewBuilder var principal: () -> Principal
    @ViewBuilder var footer: () -> Footer

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                Spacer(minLength: 24)
                principal()
                Spacer(minLength: 24)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)

            footer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

extension OnboardingCanvasLayout where Footer == EmptyView {
    init(@ViewBuilder principal: @escaping () -> Principal) {
        self.init(principal: principal, footer: { EmptyView() })
    }
}

/// A canvas whose content is vertically centred with no action lane — the
/// hero and every outcome screen.
struct OnboardingCenteredCanvas<Content: View>: View {
    var trailing: CGFloat = OnboardingD.canvasTrailing
    @ViewBuilder var content: () -> Content

    var body: some View {
        OnboardingCanvas(trailing: trailing) {
            OnboardingCanvasLayout(principal: content)
        }
    }
}

// MARK: - Text roles

/// `STEP 2 OF 4 · CHOOSE A MODEL` — the one element that names the branch.
struct OnboardingKicker: View {
    let text: String

    var body: some View {
        Text(text)
            .scaledSystemFont(11, relativeTo: .caption, weight: .semibold, design: .monospaced)
            .tracking(OnboardingD.Tracking.kicker)
            .foregroundStyle(RapidTheme.textTertiary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// The one display line on a setup screen.
///
/// One size on the desktop, one step at the floor — never a curve.
///
/// The headline holds its full size through BOTH the wide and the medium
/// layouts and changes exactly once, at the compact breakpoint where the rail
/// rotates. An earlier version scaled it proportionally off the layout tier,
/// which made the title read as shrinking continuously as the window narrowed
/// and left it small long before the window was actually small.
///
/// **Deliberate Paper deviation.** Paper 05.1.D redraws this at 28/34 in its
/// 1000×700 frame, i.e. it steps at medium as well. The approved instruction
/// for this pass is to preserve the desktop size through wide and medium and
/// change only at compact, so that is what ships; the medium step is the one
/// Paper value not carried over.
struct OnboardingDisplayTitle: View {
    @Environment(\.onboardingLayout) private var layout
    let text: String
    var size: CGFloat = 38
    /// The single compact step. Defaults to Paper's floor relationship.
    var compactSize: CGFloat?

    private var resolvedSize: CGFloat {
        guard layout.isCompact else { return size }
        return compactSize ?? max(24, (size * 0.68).rounded())
    }

    var body: some View {
        // The hard line break is authored for the desktop measure. It is only
        // dropped where the column is genuinely too narrow to honour it.
        Text(layout.isCompact ? text.replacingOccurrences(of: "\n", with: " ") : text)
            .scaledSystemFont(resolvedSize, relativeTo: .largeTitle, weight: .semibold)
            .tracking(OnboardingD.Tracking.display(resolvedSize))
            .foregroundStyle(RapidTheme.textPrimary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// Paper's Step 2 body: a fixed heading column beside the live list, which
/// stacks once the rail has rotated.
///
/// The asymmetry is load-bearing at full width. The heading column is where
/// the branch names itself and — on Review download — where the cost is
/// stated, while the list stays in exactly the same place across all three
/// micro-stages so switching between them never moves the rows under the
/// pointer.
///
/// A struct rather than a `@ViewBuilder` method on the screen, because it has
/// to READ ``EnvironmentValues/onboardingLayout``, and a value set inside a
/// view's own body is not visible to that same body.
struct OnboardingStepColumns<Aside: View, Content: View>: View {
    @Environment(\.onboardingLayout) private var layout
    let kicker: String
    let title: String
    var subtitle: String?
    @ViewBuilder var aside: () -> Aside
    @ViewBuilder var content: () -> Content

    var body: some View {
        if layout.usesColumns {
            // Paper's `alignItems: center` on the Columns row: the heading and
            // the list are centred against each other, and the row as a whole
            // is centred in the canvas by ``OnboardingCanvasLayout``.
            HStack(alignment: .center, spacing: OnboardingD.columnGap) {
                heading.frame(width: OnboardingD.headingColumnWidth, alignment: .leading)
                content().frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            VStack(alignment: .leading, spacing: 20) {
                heading
                content().frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var heading: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Micro progress. The step number is repeated deliberately: this is
            // the only element that names the branch, and it must not be
            // mistaken for a step of its own. Never sub-numbered.
            OnboardingKicker(text: kicker)
                .accessibilityIdentifier("Quickstart.Step2.Kicker")
                .padding(.bottom, 16)

            OnboardingDisplayTitle(text: title)

            if let subtitle {
                Text(subtitle)
                    .scaledSystemFont(15, relativeTo: .callout)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 14)
            }

            aside()
        }
    }
}

/// A small caps group label above a run of rows.
struct OnboardingGroupLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
            .tracking(OnboardingD.Tracking.groupLabel)
            .foregroundStyle(RapidTheme.textTertiary)
    }
}

// MARK: - Intrinsic column

/// A column that keeps its natural height while it fits, and only becomes a
/// scroller when it genuinely cannot.
///
/// ## The defect this exists to prevent
///
/// Paper frame 04 centres the Step 2 heading and the model list against each
/// other, and centres the pair in the canvas. Wrapping the list in a bare
/// `ScrollView` breaks that silently: a `ScrollView` is vertically GREEDY, so
/// it takes the whole offered height. The enclosing `HStack(alignment:
/// .center)` then dutifully centres a child that is already full-height, and
/// the rows inside it stack from ITS top — so the heading reads centred while
/// the list reads pinned to the top of the canvas, which is exactly the
/// mismatch it looks like.
///
/// `ViewThatFits` is the fix rather than a tweak: it offers the content the
/// available height first and takes the intrinsic layout when that fits, so
/// the column has a real height for the centring to act on. Only when the
/// list is genuinely taller than the canvas does the scrolling variant win,
/// and at that point filling the height is the correct behaviour anyway.
///
/// The content closure is evaluated in both branches; it must therefore stay a
/// pure function of state, which every Step 2 list already is.
struct OnboardingIntrinsicColumn<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        ViewThatFits(in: .vertical) {
            // Preferred: natural height, so the parent can centre it.
            content()
            // Fallback: taller than the canvas, so scrolling is the honest
            // shape and filling the height is what it should do.
            ScrollView {
                content()
            }
            .scrollBounceBehavior(.basedOnSize)
        }
    }
}

// MARK: - Selection-driven heading content

/// The trade-up comparison Paper draws when a bigger model is picked
/// (05.1 state 05 — "Bigger, and what it costs").
///
/// Two columns of the same three facts, with the picked one in full ink and
/// the alternative stepped back. Every value is read from ``ModelSizing``
/// against ``MacHardware`` — the same estimate the footer primary is already
/// gated on — so this table compares; it does not appraise.
struct OnboardingComparisonTable: View {
    struct Column: Identifiable {
        let id = UUID()
        /// Short header, e.g. "9B".
        let title: String
        let isPicked: Bool
        let download: String
        let memory: String
        let fit: String
        /// True when the fit reads as a caution rather than a reassurance.
        let fitIsWarning: Bool
    }

    let columns: [Column]
    /// Row label for the fit line, e.g. "Fit on 32 GB".
    let fitLabel: String

    private let labelWidth: CGFloat = 158
    private let valueWidth: CGFloat = 100

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .bottom, spacing: 0) {
                Color.clear.frame(width: labelWidth, height: 1)
                ForEach(columns) { column in
                    Text(column.isPicked ? "\(column.title) · PICKED" : column.title)
                        .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                        .tracking(OnboardingD.Tracking.groupLabel)
                        .foregroundStyle(column.isPicked ? RapidTheme.textPrimary : RapidTheme.textTertiary)
                        .frame(width: valueWidth, alignment: .trailing)
                }
            }
            .frame(height: 32, alignment: .bottom)
            .overlay(alignment: .bottom) {
                Rectangle().fill(RapidTheme.hairlineStrong).frame(height: 1)
            }

            row("Download") { $0.download }
            row("Memory when loaded") { $0.memory }
            row(fitLabel, isFit: true) { $0.fit }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("Quickstart.Compare")
    }

    @ViewBuilder
    private func row(
        _ label: String,
        isFit: Bool = false,
        value: @escaping (Column) -> String
    ) -> some View {
        HStack(spacing: 0) {
            Text(label)
                .scaledSystemFont(13)
                .foregroundStyle(RapidTheme.textSecondary)
                .frame(width: labelWidth, alignment: .leading)
            ForEach(columns) { column in
                Text(value(column))
                    .scaledSystemFont(13, design: isFit ? .default : .monospaced)
                    .foregroundStyle(tint(for: column, isFit: isFit))
                    .frame(width: valueWidth, alignment: .trailing)
            }
        }
        .frame(height: 44)
        .overlay(alignment: .bottom) {
            Rectangle().fill(RapidTheme.hairline).frame(height: 1)
        }
    }

    private func tint(for column: Column, isFit: Bool) -> Color {
        if isFit {
            return column.fitIsWarning ? RapidTheme.statusWarning : RapidTheme.statusReady
        }
        return column.isPicked ? RapidTheme.textPrimary : RapidTheme.textSecondary
    }
}

/// The "what the primary will do" summary Paper puts under the heading when
/// nothing is on this Mac yet (05.1 state 07).
struct OnboardingSelectionSummary: View {
    let title: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                .tracking(OnboardingD.Tracking.groupLabel)
                .foregroundStyle(RapidTheme.textTertiary)
            Text(detail)
                .scaledSystemFont(13)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("Quickstart.SelectionSummary")
        .accessibilityLabel("\(title). \(detail)")
    }
}

/// A placeholder for a value that is genuinely still being read.
///
/// Deliberately static. Paper 05.1 state 02 is explicit that the hardware read
/// "must never be dressed as a scanning animation with fake duration" — this is
/// a shape standing in for a number that has not arrived, and nothing more.
struct OnboardingSkeleton: View {
    var width: CGFloat?
    var height: CGFloat = 12

    var body: some View {
        RoundedRectangle(cornerRadius: height / 3, style: .continuous)
            .fill(RapidTheme.hairline)
            .frame(width: width, height: height)
            .accessibilityHidden(true)
    }
}

// MARK: - Glyph tile

/// The 52pt tinted tile above a recovery or completion headline. Its tint is
/// the state's meaning: amber for something recoverable, red for something
/// that stopped setup, green for done.
struct OnboardingGlyphTile: View {
    enum Tone {
        case amber
        case error
        case ready

        var tint: Color {
            switch self {
            case .amber: return RapidTheme.brandPrimaryTint
            case .error: return RapidTheme.statusErrorTint
            case .ready: return RapidTheme.statusReadyTint
            }
        }

        var ink: Color {
            switch self {
            case .amber: return RapidTheme.brandPrimaryDeep
            case .error: return RapidTheme.statusError
            case .ready: return RapidTheme.statusReady
            }
        }
    }

    let systemName: String
    let tone: Tone

    var body: some View {
        RoundedRectangle(cornerRadius: 13, style: .continuous)
            .fill(tone.tint)
            .frame(width: 52, height: 52)
            .overlay {
                Image(systemName: systemName)
                    .font(.system(size: 24, weight: .regular))
                    .foregroundStyle(tone.ink)
            }
            .accessibilityHidden(true)
    }
}

// MARK: - Badges

/// A row badge. Fixed vocabulary so a row cannot invent a claim: each case
/// maps to a fact the catalogue or ``ModelSizing`` already reports.
struct OnboardingBadge: View {
    enum Tone {
        case ink
        case ready
        case amber
        case error
        case neutral
    }

    let text: String
    let tone: Tone

    var body: some View {
        Text(text)
            .scaledSystemFont(9, relativeTo: .caption2, weight: .semibold, design: .monospaced)
            .tracking(OnboardingD.Tracking.badge)
            .foregroundStyle(foreground)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(RoundedRectangle(cornerRadius: 4, style: .continuous).fill(background))
            .fixedSize()
    }

    private var foreground: Color {
        switch tone {
        case .ink: return RapidTheme.surfaceCanvas
        case .ready: return RapidTheme.statusReady
        case .amber: return RapidTheme.brandPrimaryDeep
        case .error: return RapidTheme.statusError
        case .neutral: return RapidTheme.textSecondary
        }
    }

    private var background: Color {
        switch tone {
        case .ink: return RapidTheme.textPrimary
        case .ready: return RapidTheme.statusReadyTint
        case .amber: return RapidTheme.brandPrimaryTint
        case .error: return RapidTheme.statusErrorTint
        case .neutral: return RapidTheme.surfaceCode
        }
    }
}

/// The qualitative pill under the starter card ("Instant", "Runs on any Mac").
struct OnboardingAttributePill: View {
    let text: String

    var body: some View {
        Text(text)
            .scaledSystemFont(11, relativeTo: .caption, weight: .medium)
            .foregroundStyle(RapidTheme.textSecondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().fill(RapidTheme.surfaceCode))
    }
}

// MARK: - Selection glyph

/// The 18pt selection dot. Filled amber when picked, a hairline ring when not,
/// and a dimmer ring when the row cannot be chosen at all.
struct OnboardingSelectionGlyph: View {
    let isSelected: Bool
    var isEnabled: Bool = true

    var body: some View {
        Group {
            if isSelected {
                Circle()
                    .fill(RapidTheme.brandPrimary)
                    .overlay {
                        Image(systemName: "checkmark")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(RapidTheme.brandOnAccent)
                    }
            } else {
                Circle()
                    .strokeBorder(
                        isEnabled ? RapidTheme.hairlineStrong : RapidTheme.hairline,
                        lineWidth: 1
                    )
            }
        }
        .frame(width: OnboardingD.selectionGlyph, height: OnboardingD.selectionGlyph)
        .accessibilityHidden(true)
    }
}

// MARK: - Buttons

/// The one strong amber moment on a screen.
struct OnboardingPrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaledSystemFont(15, weight: .semibold)
            .foregroundStyle(RapidTheme.brandOnAccent)
            .padding(.horizontal, 30)
            .frame(height: OnboardingD.actionHeight)
            .background(
                RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                    .fill(RapidTheme.brandPrimary)
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.86 : 1) : 0.4)
            .contentShape(RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous))
    }
}

/// A bordered secondary — used where the alternative is a real, weighty action
/// (Quit) rather than a quiet way back.
struct OnboardingOutlineButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaledSystemFont(14, weight: .medium)
            .foregroundStyle(RapidTheme.textPrimary)
            .padding(.horizontal, 20)
            .frame(height: OnboardingD.actionHeight)
            .background(
                RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .overlay(
                RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                    .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.86 : 1) : 0.4)
            .contentShape(RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous))
    }
}

/// A quiet text action in the footer lane — Back, Skip, Cancel.
struct OnboardingQuietButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var tone: Color = RapidTheme.textSecondary

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaledSystemFont(13, weight: .medium)
            .foregroundStyle(tone)
            .frame(height: OnboardingD.actionHeight)
            .opacity(isEnabled ? (configuration.isPressed ? 0.6 : 1) : 0.4)
            .contentShape(Rectangle())
    }
}

extension ButtonStyle where Self == OnboardingPrimaryButtonStyle {
    static var onboardingPrimary: OnboardingPrimaryButtonStyle { .init() }
}

extension ButtonStyle where Self == OnboardingOutlineButtonStyle {
    static var onboardingOutline: OnboardingOutlineButtonStyle { .init() }
}

extension ButtonStyle where Self == OnboardingQuietButtonStyle {
    /// Neutral quiet action.
    static var onboardingQuiet: OnboardingQuietButtonStyle { .init() }
    /// Quiet action that navigates somewhere — steel, the link colour.
    static var onboardingLink: OnboardingQuietButtonStyle {
        .init(tone: RapidTheme.brandSecondary)
    }
}

// MARK: - Fact table

/// One `label · value` line in a Review-style fact table.
struct OnboardingFactRow: Identifiable {
    let id = UUID()
    let label: String
    let value: String
    /// Soft values (a path, "Not downloaded yet") sit back a step so the
    /// numbers that drive the decision stay the strongest thing in the column.
    var isStrong: Bool = true
    /// The one row that is not merely a fact but the REASON — Paper 05.2.D
    /// colours both the label and the value of "Memory when loaded" when the
    /// model will not fit, so the table itself points at the offending number
    /// instead of leaving the callout above to carry it alone.
    var isAlert: Bool = false
    var identifier: String?

    init(
        _ label: String,
        _ value: String,
        isStrong: Bool = true,
        isAlert: Bool = false,
        identifier: String? = nil
    ) {
        self.label = label
        self.value = value
        self.isStrong = isStrong
        self.isAlert = isAlert
        self.identifier = identifier
    }
}

/// Hairline-separated facts. No card, no fill: Paper puts information directly
/// on the surface and reserves boxes for choices.
struct OnboardingFactTable: View {
    let rows: [OnboardingFactRow]

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                if index > 0 {
                    Rectangle().fill(RapidTheme.hairline).frame(height: 1)
                }
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(row.label)
                        .scaledSystemFont(13)
                        .foregroundStyle(row.isAlert ? RapidTheme.statusError : RapidTheme.textSecondary)
                    Spacer(minLength: 12)
                    Text(row.value)
                        .scaledSystemFont(13, design: .monospaced)
                        .foregroundStyle(
                            row.isAlert
                                ? RapidTheme.statusError
                                : (row.isStrong ? RapidTheme.textPrimary : RapidTheme.textSecondary)
                        )
                        .multilineTextAlignment(.trailing)
                        .textSelection(.enabled)
                }
                .padding(.top, index == 0 ? 0 : 11)
                .padding(.bottom, index == rows.count - 1 ? 0 : 11)
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier(row.identifier ?? "")
                .accessibilityLabel("\(row.label): \(row.value)")
            }
        }
    }
}

/// A tinted note that sits INSIDE a decision, above the facts that justify it
/// (Paper 05.2.D — the incompatible-memory callout).
///
/// Deliberately not an ``OnboardingOutcomeBlock``: that template owns whole
/// screens whose subject is the outcome, and it centres a glyph tile over a
/// title. This is a paragraph with a mark beside it, subordinate to the heading
/// it explains — the screen is still Review download, and the note is one of
/// the things Review has to say rather than a replacement for it.
struct OnboardingInlineNote: View {
    let text: String
    var glyph: String = "exclamationmark.triangle"
    var tone: Color = RapidTheme.statusError
    var tint: Color = RapidTheme.statusErrorTint
    var identifier: String?
    /// Spoken form. Defaults to ``text``; a caller that has already said the
    /// same thing in a heading can hand VoiceOver a shorter line instead.
    var accessibilityText: String?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: glyph)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(tone)
                .accessibilityHidden(true)
            Text(text)
                .scaledSystemFont(12.5)
                .foregroundStyle(tone)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 14)
        .background(
            RoundedRectangle(cornerRadius: OnboardingD.actionRadius, style: .continuous)
                .fill(tint)
        )
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(identifier ?? "")
        .accessibilityLabel(accessibilityText ?? text)
    }
}

// MARK: - Recovery / completion composition

/// The shared template behind every state that reports an outcome: the
/// recoverable failures, the disk warning, the memory guard, missing engine
/// and Ready (Paper 05.1 states 10–13, 16, 19, 20).
///
/// One shape, four slots. What changes between them is the tone of the glyph,
/// the words, and which actions are offered — never the composition, so a user
/// who has seen one recognises the next.
struct OnboardingOutcomeBlock<Actions: View, Detail: View>: View {
    let glyph: String
    let tone: OnboardingGlyphTile.Tone
    let kicker: String
    let title: String
    let message: String
    @ViewBuilder var detail: () -> Detail
    @ViewBuilder var actions: () -> Actions

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingGlyphTile(systemName: glyph, tone: tone)
                .padding(.bottom, 26)

            OnboardingKicker(text: kicker)
                .padding(.bottom, 16)

            OnboardingDisplayTitle(text: title)

            Text(message)
                .scaledSystemFont(16, relativeTo: .title3)
                .foregroundStyle(RapidTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 16)

            detail()

            actions()
                .padding(.top, 34)
        }
        .frame(maxWidth: OnboardingD.decisionWidth, alignment: .leading)
    }
}

extension OnboardingOutcomeBlock where Detail == EmptyView {
    init(
        glyph: String,
        tone: OnboardingGlyphTile.Tone,
        kicker: String,
        title: String,
        message: String,
        @ViewBuilder actions: @escaping () -> Actions
    ) {
        self.init(
            glyph: glyph,
            tone: tone,
            kicker: kicker,
            title: title,
            message: message,
            detail: { EmptyView() },
            actions: actions
        )
    }
}

/// The horizontal action lane under an outcome: one amber primary, then quiet
/// alternatives. 22pt apart so the secondary reads as a different weight of
/// choice rather than a second button.
struct OnboardingActionLane<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        HStack(spacing: 22) {
            content()
        }
    }
}

/// The one footer lane a Step 2 micro-stage may hold: at most one Back on the
/// left, exactly one primary on the right.
///
/// Replaces `OnboardingWizardFooter`. The keyboard contract is carried over
/// verbatim, because it is behaviour and PR #1931 pinned it: Return is the
/// primary's `.defaultAction`, so a disabled primary swallows it and no input
/// can reach an action the user cannot see; Escape is Back's `.cancelAction`,
/// so it resolves to the visible control at each depth.
struct OnboardingStepFooter: View {
    let primaryTitle: String
    var primaryEnabled: Bool = true
    /// Back's label, named for its destination ("← Back to all models") so the
    /// control says where it goes rather than only that it goes back.
    var backTitle: String = "Back"
    /// Spoken form of ``backTitle``, without the arrow glyph.
    var backAccessibilityLabel: String?
    /// Spoken form of the primary. Defaults to ``primaryTitle`` so the AX
    /// baselines that pin `desc="Review download"` are unaffected.
    var primaryAccessibilityLabel: String?
    /// Why the primary is unavailable, when it is. macOS announces a disabled
    /// control as "dimmed", which says that it cannot be pressed but not why —
    /// and on Review download for a model this Mac cannot run, the why is the
    /// entire point of the screen.
    var primaryAccessibilityHint: String?
    var onBack: (() -> Void)?
    let onPrimary: () -> Void

    var body: some View {
        HStack(spacing: 22) {
            if let onBack {
                Button(action: onBack) {
                    Text(backTitle)
                }
                .buttonStyle(.onboardingLink)
                .keyboardShortcut(.cancelAction)
                .accessibilityIdentifier("Quickstart.Footer.Back")
                .accessibilityLabel(backAccessibilityLabel ?? backTitle)
            }
            Spacer(minLength: 12)
            Button(action: onPrimary) {
                Text(primaryTitle)
            }
            .buttonStyle(.onboardingPrimary)
            .disabled(!primaryEnabled)
            .keyboardShortcut(.defaultAction)
            .accessibilityIdentifier("Quickstart.Footer.Primary")
            .accessibilityLabel(primaryAccessibilityLabel ?? primaryTitle)
            .accessibilityHint(primaryAccessibilityHint ?? "")
        }
        .frame(maxWidth: .infinity)
    }
}

/// A statistic pair under a warning — "FREE NOW 1.4 GB".
struct OnboardingStat: View {
    let label: String
    let value: String
    var tone: Color = RapidTheme.textPrimary

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label)
                .scaledSystemFont(10, relativeTo: .caption2, weight: .semibold, design: .monospaced)
                .tracking(OnboardingD.Tracking.groupLabel)
                .foregroundStyle(RapidTheme.textTertiary)
            Text(value)
                .scaledSystemFont(22, relativeTo: .title2, design: .monospaced)
                .foregroundStyle(tone)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(value)")
    }
}
