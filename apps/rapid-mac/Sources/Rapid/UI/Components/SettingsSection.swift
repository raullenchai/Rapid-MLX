import SwiftUI

/// The grouped-section primitives every Settings panel is built from.
///
/// ## Why these exist
///
/// Before this, four panels each carried a private copy of the same two
/// helpers — `SettingsView.settingsCard`/`sectionHeader`,
/// `SettingsToolsPanel.card`/`header`,
/// `SettingsConnectorsPanel.card`/`header`, plus six inline repetitions
/// of the same `RoundedRectangle` recipe in
/// `SettingsModelManagementPanel`. Four copies meant four radii to keep
/// in sync, four heading sizes, and four sets of padding, and they had
/// already drifted: cards were 12pt while the rest of the app was 8pt,
/// headings ranged over `.title2` / `.title3` / `.callout`, and two
/// panels drew no card at all.
///
/// These types are the single owner of that treatment. A panel now says
/// what a group IS; it does not describe how a group looks.
///
/// ## The rules they encode
///
/// * One card per section, and **never a card inside a card** — content
///   handed to ``SettingsSection`` draws no background of its own.
/// * Section headings live OUTSIDE the card, so the card holds controls
///   and only controls.
/// * Rows are separated by ``SettingsRowDivider``, inset to the content's
///   leading edge so the divider starts where the text does.

// MARK: - Responsive context

private struct SettingsContentIsCompactKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    /// True when the Settings detail column is too narrow to carry every
    /// optional column a panel would like to draw.
    ///
    /// Published by the detail canvas, which measures itself, so panels
    /// respond to the width they actually got rather than inferring it
    /// from the window. A panel should use this to DROP something
    /// optional (a meters column, a legend), never to hide a control —
    /// an unreachable control at a supported window size is the defect
    /// this exists to prevent.
    var settingsContentIsCompact: Bool {
        get { self[SettingsContentIsCompactKey.self] }
        set { self[SettingsContentIsCompactKey.self] = newValue }
    }
}

// MARK: - Section

/// A titled group of settings rows on one card.
struct SettingsSection<Content: View, Accessory: View>: View {
    let title: String?
    var subtitle: String?
    @ViewBuilder var accessory: Accessory
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.sm) {
            if let title {
                if Accessory.self == EmptyView.self {
                    SectionHeader(title, subtitle: subtitle, emphasis: .section)
                } else {
                    SectionHeader(title, subtitle: subtitle, emphasis: .section) {
                        accessory
                    }
                }
            }
            VStack(alignment: .leading, spacing: 0) {
                content
            }
            .settingsGroupedCard()
        }
    }
}

extension SettingsSection where Accessory == EmptyView {
    /// A section with no trailing header control.
    init(
        _ title: String? = nil,
        subtitle: String? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.accessory = EmptyView()
        self.content = content()
    }
}

extension SettingsSection {
    /// A section whose heading carries a trailing control (an "Add…").
    init(
        _ title: String,
        subtitle: String? = nil,
        @ViewBuilder accessory: () -> Accessory,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.accessory = accessory()
        self.content = content()
    }
}

// MARK: - Grouped card

/// Inset inside a grouped Settings card.
///
/// The UI-1 review found content sitting too close to the card edge at
/// ``Space/lg`` (16). 24 at regular widths gives the title, description
/// and control room to read as a group rather than as text jammed into a
/// box; 20 at compact widths keeps that feeling without stealing the
/// width a three-line description needs at the 720pt window floor.
enum SettingsCardMetrics {
    static let regularInset: CGFloat = RapidTheme.Space.xl   // 24
    static let compactInset: CGFloat = 20

    static func inset(isCompact: Bool) -> CGFloat {
        isCompact ? compactInset : regularInset
    }
}

/// Applies the shared grouped-card treatment, reading the compact flag
/// from the environment so every card in the window breathes the same
/// amount at the same window size.
private struct SettingsGroupedCard: ViewModifier {
    @Environment(\.settingsContentIsCompact) private var isCompact
    /// Explicit override for the rare caller that manages its own inset.
    var override: CGFloat?

    func body(content: Content) -> some View {
        let inset = override ?? SettingsCardMetrics.inset(isCompact: isCompact)
        return content
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(inset)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .fill(RapidTheme.surfaceRaised)
            )
            .clipShape(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                    .strokeBorder(RapidTheme.hairline, lineWidth: 1)
            )
    }
}

extension View {
    /// The one grouped-surface treatment in Settings: raised fill,
    /// structural hairline, ``Radius/card`` corners, and the shared
    /// responsive inset.
    ///
    /// Matches the main window's cards exactly (8pt, not the legacy
    /// 12pt ``RapidTheme/cardRadius``), which is what stops the Settings
    /// window reading as a different app from the one behind it.
    func settingsGroupedCard(padding: CGFloat? = nil) -> some View {
        modifier(SettingsGroupedCard(override: padding))
    }
}

// MARK: - Row label

/// The text half of a settings row: a label and an optional explanation.
///
/// Extracted so a `Toggle`'s label, a button row's label, and a value
/// row's label are the same two typographic roles rather than three
/// hand-picked font pairs. The description is deliberately allowed to
/// wrap to as many lines as it needs — at the 720pt window floor several
/// of these run to three lines, and truncating a sentence that explains
/// what a switch does is the wrong trade.
struct SettingsRowLabel: View {
    let title: String
    var description: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xxs) {
            Text(title)
                .font(RapidFont.bodyEmphasis)
                .foregroundStyle(RapidTheme.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            if let description {
                Text(description)
                    .font(RapidFont.caption)
                    .foregroundStyle(RapidTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Row

/// One settings row: a label (plus optional explanation) on the leading
/// edge and a control on the trailing edge.
///
/// The control keeps its intrinsic width and the label takes the rest,
/// which is what makes the row survive a narrow window: text rewraps,
/// the control never compresses, and nothing overlaps. Use this for
/// buttons, pickers, and read-only values. Toggles keep
/// ``TrailingSettingsToggleStyle`` with a ``SettingsRowLabel`` inside —
/// the native switch stays native.
struct SettingsRow<Control: View>: View {
    let title: String
    var description: String? = nil
    @ViewBuilder var control: Control

    var body: some View {
        HStack(alignment: .top, spacing: TrailingSettingsToggleStyle.gutter) {
            SettingsRowLabel(title: title, description: description)
            control
                .fixedSize(horizontal: true, vertical: false)
                // Align the control with the label's first line rather
                // than the centre of a three-line description. The
                // gutter matches ``TrailingSettingsToggleStyle`` so a
                // button row and a switch row put their controls in the
                // same column.
                .frame(minHeight: RapidTheme.ControlHeight.small, alignment: .topTrailing)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Divider

/// The separator between rows inside one grouped card.
///
/// Inset to the content's leading edge so it starts under the text
/// rather than under the card's padding — the macOS grouped-list
/// convention, and the thing that makes a stack of rows read as one
/// table instead of several stripes.
struct SettingsRowDivider: View {
    var body: some View {
        Rectangle()
            .fill(RapidTheme.hairline)
            .frame(height: 1)
            // Symmetric, and larger than the old 12: with 24pt card
            // insets a 12pt gap made rows look pinched relative to the
            // space around the group.
            .padding(.vertical, RapidTheme.Space.lg)
            .accessibilityHidden(true)
    }
}
