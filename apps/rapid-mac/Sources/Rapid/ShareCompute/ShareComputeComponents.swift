import SwiftUI

// MARK: - Tabs

/// The three top-level surfaces, each answering a different user question.
enum ShareComputeTab: String, CaseIterable, Identifiable, Hashable {
    case share
    /// Was `myContribution`. Renamed with the API-aligned redesign: the tab
    /// answers "what do I earn?", and what QuickSilver pays is API credits.
    case credits
    /// Was `pool`. "Live" is load-bearing — the tab shows what is online right
    /// now, not a catalog of what the pool contains.
    case livePool

    var id: String { rawValue }

    var title: String {
        switch self {
        case .share: return String(localized: "Share")
        case .credits: return String(localized: "Credits")
        case .livePool: return String(localized: "Live Pool")
        }
    }

    /// Announced by VoiceOver after the tab name, so the purpose of each
    /// surface is available without reading the whole page.
    var accessibilityHint: String {
        switch self {
        case .share: return String(localized: "What can I serve right now?")
        case .credits: return String(localized: "What have I earned?")
        // NOT "what does the pool need" — the summary endpoint publishes
        // availability, never demand. See ``ShareComputePoolSummary``.
        case .livePool: return String(localized: "What is online right now?")
        }
    }

    /// The tab a GUI harness asked the module to open on.
    ///
    /// Same contract and same gate as
    /// ``SidebarSection/harnessRequested(environment:)``: inert unless
    /// ``RAPID_GUI_GOLDEN_MODE`` is set, so a normal launch always opens on
    /// Share.
    static func harnessRequested(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ShareComputeTab? {
        guard environment["RAPID_GUI_GOLDEN_MODE"] == "1",
              let name = environment["RAPID_GUI_SHARE_COMPUTE_TAB"] else { return nil }
        return ShareComputeTab(rawValue: name)
    }
}

/// The underlined tab strip.
///
/// Deliberately NOT ``.pickerStyle(.segmented)``, which is what Community
/// Benchmark uses: Paper draws a 36pt row of three equal-width labels over a
/// single hairline, with a 2pt amber rule under the selected one. A segmented
/// control's filled track reads as a heavier control than the page needs and
/// changes width with its longest label, which would let the strip shift
/// between tabs — the handoff forbids exactly that.
///
/// The fixed 360pt width is Paper's, and it is what keeps the strip identical
/// on all three surfaces.
struct ShareComputeTabBar: View {
    @Binding var selection: ShareComputeTab

    /// Paper's width. Below it the bar fills whatever it is given so the
    /// narrow layout keeps all three tabs rather than truncating one.
    static let preferredWidth: CGFloat = 360
    private static let height: CGFloat = 36

    @Namespace private var underline

    var body: some View {
        HStack(spacing: 0) {
            ForEach(ShareComputeTab.allCases) { tab in
                segment(tab)
            }
        }
        .frame(maxWidth: Self.preferredWidth)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(RapidTheme.hairline)
                .frame(height: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("ShareCompute.Tabs")
    }

    private func segment(_ tab: ShareComputeTab) -> some View {
        let isSelected = selection == tab
        return Button {
            selection = tab
        } label: {
            Text(tab.title)
                .font(.system(size: 12, weight: isSelected ? .bold : .medium))
                .foregroundStyle(
                    isSelected ? RapidTheme.textPrimary : RapidTheme.textSecondary
                )
                .frame(maxWidth: .infinity)
                .frame(height: Self.height)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .overlay(alignment: .bottom) {
            // Drawn above the strip's own hairline so the selected rule wins
            // where they overlap.
            if isSelected {
                Rectangle()
                    .fill(RapidTheme.brandPrimary)
                    .frame(height: 2)
                    .matchedGeometryEffect(id: "underline", in: underline)
            }
        }
        .accessibilityAddTraits(isSelected ? [.isSelected, .isButton] : .isButton)
        .accessibilityLabel(tab.title)
        .accessibilityHint(tab.accessibilityHint)
        .accessibilityIdentifier("ShareCompute.Tab.\(tab.rawValue)")
    }
}

// MARK: - Workbench chrome

/// The rounded container every Share Compute composition sits in.
///
/// One shape, one border, one clip — so the Share, My Contribution, and Pool
/// surfaces cannot drift apart, and so dark mode gets its explicit edge in a
/// single place. The border is ``hairlineStrong`` rather than ``hairline``
/// because in Dark the band ground and the window canvas are only a few
/// percent apart; without a stated edge the workbench dissolves into the page.
struct ShareComputeWorkbenchFrame<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .clipShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.card))
            .overlay {
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card)
                    .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
            }
    }
}

/// An all-caps eyebrow. Tracked open, because small caps set solid read as a
/// single word.
struct ShareComputeEyebrow: View {
    let text: String
    var tone: Color = RapidTheme.brandPrimary
    var size: CGFloat = 10

    var body: some View {
        Text(text.uppercased())
            .font(.system(size: size, weight: .bold))
            .tracking(0.06 * size)
            .foregroundStyle(tone)
            .accessibilityLabel(text)
    }
}

// MARK: - Status tags

/// The visual weight of a status tag.
///
/// Every tone is defined ON THE BAND and is appearance-independent, because
/// the band's graphite ground is. Reaching for a general-purpose token here is
/// the specific mistake that made `SELECTED` invisible in Dark: the tone read
/// its fill from ``RapidTheme/brandPrimaryTint``, which flips to a near-black
/// amber in Dark, while its ink stayed the near-black ``onBrandPrimary``. The
/// band-scoped tokens below cannot produce that pairing in either appearance.
enum ShareComputeTagTone: Equatable {
    /// Green, for a settled positive state.
    case ready
    /// Amber, for the selected row.
    case selected
    /// Outlined neutral, for in-flight or absent states.
    case neutral

    var foreground: Color {
        switch self {
        case .ready: return RapidTheme.bandReady
        case .selected: return RapidTheme.onBandTagAmber
        case .neutral: return RapidTheme.bandTagNeutralInk
        }
    }

    var background: Color? {
        switch self {
        case .ready: return RapidTheme.bandReadyTint
        case .selected: return RapidTheme.bandTagAmberFill
        case .neutral: return nil
        }
    }

    var stroke: Color? {
        switch self {
        case .ready: return RapidTheme.statusReady
        case .selected: return nil
        case .neutral: return RapidTheme.bandTagNeutralStroke
        }
    }
}

/// A compact status tag that sizes to its own content.
///
/// The handoff calls this out explicitly: `SELECTED`, `AVAILABLE`, and
/// `PROCESSING` must not be forced to a shared width. A fixed-width tag column
/// would pad `47m`-scale words out to the longest label and turn a scannable
/// column into a row of identical boxes, which is the opposite of what a
/// status is for. The tag therefore hugs its text and the COLUMN around it is
/// what holds the lane.
struct ShareComputeStatusTag: View {
    let title: String
    let tone: ShareComputeTagTone

    var body: some View {
        Text(title.uppercased())
            .font(.system(size: 9, weight: .bold))
            .tracking(0.4)
            .foregroundStyle(tone.foreground)
            .padding(.horizontal, 10)
            .frame(height: 22)
            .background {
                if let background = tone.background {
                    RoundedRectangle(cornerRadius: 5).fill(background)
                }
            }
            .overlay {
                if let stroke = tone.stroke {
                    RoundedRectangle(cornerRadius: 5).strokeBorder(stroke, lineWidth: 1)
                }
            }
            .fixedSize()
            .accessibilityLabel(title)
    }
}

// MARK: - Pagination

/// Previous / page-of / next, as Paper draws it.
///
/// Both arrows stay mounted at their fixed size even when disabled so the
/// control cannot change width between pages, and each carries a screen-reader
/// label because an arrow glyph alone announces nothing useful.
struct ShareComputePager: View {
    let page: Int
    let pageCount: Int
    let onPrevious: () -> Void
    let onNext: () -> Void
    var controlSize: CGFloat = 26

    private var canGoBack: Bool { page > 0 }
    private var canGoForward: Bool { page + 1 < pageCount }

    var body: some View {
        HStack(spacing: 8) {
            arrow(
                systemImage: "chevron.left",
                label: String(localized: "Previous page"),
                enabled: canGoBack,
                action: onPrevious
            )
            Text("\(page + 1) / \(pageCount)")
                .font(.system(size: 10, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(RapidTheme.bandInkSecondary)
                .accessibilityLabel(
                    String(
                        format: String(localized: "Page %1$d of %2$d"),
                        page + 1,
                        pageCount
                    )
                )
            arrow(
                systemImage: "chevron.right",
                label: String(localized: "Next page"),
                enabled: canGoForward,
                action: onNext
            )
        }
    }

    private func arrow(
        systemImage: String,
        label: String,
        enabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 10, weight: .medium))
                .frame(width: controlSize, height: controlSize)
                .foregroundStyle(
                    enabled ? RapidTheme.bandInk : RapidTheme.bandInkTertiary
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 5)
                        .strokeBorder(
                            enabled
                                ? RapidTheme.bandHairlineStrong
                                : RapidTheme.bandHairline,
                            lineWidth: 1
                        )
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityLabel(label)
    }
}

// MARK: - Band actions

/// The single amber call to action on a workbench.
struct ShareComputePrimaryBandButton: View {
    let title: String
    var height: CGFloat = 72
    var isEnabled = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Text(title)
                    .font(.system(size: 15, weight: .bold))
                Image(systemName: "arrow.right")
                    .font(.system(size: 13, weight: .bold))
                    .accessibilityHidden(true)
            }
            .foregroundStyle(RapidTheme.onBrandPrimary)
            .frame(maxWidth: .infinity)
            .frame(height: height)
            .background(
                RapidTheme.brandPrimary,
                in: RoundedRectangle(cornerRadius: RapidTheme.Radius.button)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : RapidTheme.disabledOpacity)
        .accessibilityLabel(title)
    }
}

/// Stop Sharing — quiet, red, and always in the same corner of the band so it
/// is findable without hunting during a session the user wants to end.
struct ShareComputeStopButton: View {
    var isEnabled = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text("Stop Sharing")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(RapidTheme.bandDestructive)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : RapidTheme.disabledOpacity)
        .accessibilityLabel(String(localized: "Stop sharing this Mac"))
        .accessibilityIdentifier("ShareCompute.Stop")
    }
}

// MARK: - Three-step value path

/// One step of the Share tab's value path.
struct ShareComputeValueStep: Identifiable {
    let number: Int
    let title: String
    let detail: String
    /// The narrow layout's wording, transcribed from Paper's narrow artboard.
    ///
    /// Paper does use different copy at 720pt — the steps stay on one line
    /// there — so these strings are a spec, not an abbreviation the app is
    /// free to choose. An earlier pass shortened them further ("Model
    /// pauses", "History grows") to buy room, because the app was still
    /// rendering its full 200pt sidebar at 720pt and the steps had ~145pt
    /// less to work with than the artboard assumes. The shell now collapses
    /// to Paper's 64pt rail at that width (see ``SidebarView/isCompact``), so
    /// the copy no longer has to absorb a layout difference and is Paper's
    /// verbatim.
    let narrowTitle: String
    let narrowDetail: String
    let tint: Color
    let tintBackground: Color

    var id: Int { number }

    func title(isNarrow: Bool) -> String { isNarrow ? narrowTitle : title }
    func detail(isNarrow: Bool) -> String { isNarrow ? narrowDetail : detail }

    /// The approved path. Numbered circles are tinted by WHO owns the step:
    /// steel for the local Mac, green for the record Rapid keeps, amber for
    /// the provider's reward — the same ownership split the whole module is
    /// built around.
    ///
    /// All three backings are BAND-scoped tints. The general-purpose
    /// ``brandSecondaryTint`` / ``brandPrimaryTint`` were used here and are
    /// appearance-dependent: in Light they resolve to a near-white and a
    /// cream, so on the graphite band steps 1 and 3 rendered as pale discs
    /// carrying pale numerals while step 2 — which already used a band token
    /// — looked correct. Only step 2 was right by accident.
    static let path: [ShareComputeValueStep] = [
        // Copy is the API-aligned wording, and each line is chosen to kill a
        // specific wrong mental model. There is no queue to poll and no task to
        // claim: the relay pushes a live request down a socket this Mac dialled
        // out, and the answer streams straight back up it.
        .init(
            number: 1,
            title: String(localized: "Connect securely"),
            detail: String(localized: "Register this Mac and open the relay"),
            narrowTitle: String(localized: "Connect securely"),
            narrowDetail: String(localized: "Register and open the relay"),
            tint: RapidTheme.bandLink,
            tintBackground: RapidTheme.bandLinkTint
        ),
        .init(
            number: 2,
            title: String(localized: "Serve live requests"),
            detail: String(localized: "Responses return through the tunnel"),
            narrowTitle: String(localized: "Serve live requests"),
            narrowDetail: String(localized: "Answers stream back"),
            tint: RapidTheme.bandReady,
            tintBackground: RapidTheme.bandReadyTint
        ),
        .init(
            number: 3,
            // API credits, not cash, not tokens on a chain. The wording is the
            // product fact and must not be softened into "rewards".
            title: String(localized: "Receive monthly credits"),
            detail: String(localized: "Based on metered input and output tokens"),
            narrowTitle: String(localized: "Receive credits"),
            narrowDetail: String(localized: "From metered tokens"),
            tint: RapidTheme.brandPrimary,
            tintBackground: RapidTheme.bandBrandTint
        ),
    ]
}

/// The three-step path with fixed connector slots.
///
/// The layout rule the handoff insists on: connectors occupy their OWN
/// fixed-width slot in the row, so the arrow never moves when a step's copy
/// wraps — it is not laid out relative to any text.
///
/// ## Where the arrows sit
///
/// Paper centres the whole row (`items-center`) and centres each step's
/// circle against its own text block. This view previously pinned everything
/// with `.top`, which is subtly but visibly wrong: a `.top`-aligned arrow
/// hangs off the first line of the tallest label rather than sitting on the
/// circles' shared centreline, and it drifts further the moment any step
/// wraps. Centre alignment is what makes the arrow read as "between two
/// steps" instead of "attached to one".
///
/// Copy is allowed to wrap to a second line rather than being scaled down.
/// Shrinking the type to force one line is exactly the move the responsive
/// brief rules out, and with centre alignment a wrapped step grows the row
/// symmetrically and leaves the arrows where they were.
struct ShareComputeValuePath: View {
    var isNarrow = false

    /// Paper: 28pt circles on the desktop board, 24pt on the narrow one.
    private var circleSize: CGFloat { isNarrow ? 24 : 28 }
    /// Paper: a 56pt connector slot on desktop, 20pt narrow.
    private var connectorWidth: CGFloat { isNarrow ? 20 : 56 }
    /// Paper: 12pt between circle and copy on desktop, 9pt narrow.
    private var stepSpacing: CGFloat { isNarrow ? 9 : 12 }

    var body: some View {
        HStack(alignment: .center, spacing: 0) {
            ForEach(Array(ShareComputeValueStep.path.enumerated()), id: \.element.id) { index, step in
                stepView(step)
                if index < ShareComputeValueStep.path.count - 1 {
                    connector
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(String(localized: "How sharing works, in three steps"))
    }

    private func stepView(_ step: ShareComputeValueStep) -> some View {
        HStack(alignment: .center, spacing: stepSpacing) {
            // Paper sets the numeral semibold on the narrow board and bold on
            // the desktop one; both at 12pt.
            Text("\(step.number)")
                .font(.system(size: 12, weight: isNarrow ? .semibold : .bold))
                .foregroundStyle(step.tint)
                .frame(width: circleSize, height: circleSize)
                .background(step.tintBackground, in: Circle())
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: isNarrow ? 1 : 3) {
                // Paper's hierarchy exactly: 14/bold desktop, 13/semibold
                // narrow for the title; regular — never medium or heavier —
                // for the supporting line in both.
                Text(step.title(isNarrow: isNarrow))
                    .font(.system(size: isNarrow ? 13 : 14, weight: isNarrow ? .semibold : .bold))
                    .foregroundStyle(RapidTheme.bandInk)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Text(step.detail(isNarrow: isNarrow))
                    .font(.system(size: isNarrow ? 11 : 12, weight: .regular))
                    .foregroundStyle(RapidTheme.bandInkSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        // VoiceOver always gets the full wording, even where the narrow
        // layout shows the short form.
        .accessibilityLabel("\(step.number). \(step.title). \(step.detail)")
    }

    /// The connector's own slot — a fixed width lane the copy cannot reach
    /// into, centred vertically with the steps beside it.
    private var connector: some View {
        Image(systemName: "arrow.right")
            .font(.system(size: isNarrow ? 10 : 11, weight: .medium))
            .foregroundStyle(RapidTheme.bandInkTertiary)
            .frame(width: connectorWidth, height: circleSize)
            .accessibilityHidden(true)
    }
}
