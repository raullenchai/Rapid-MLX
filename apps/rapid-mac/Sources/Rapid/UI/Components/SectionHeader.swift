import SwiftUI

/// A page or group heading: title, optional supporting line, optional
/// trailing accessory.
///
/// Exists so headings stop being re-invented per view. Before v1.0 the
/// same conceptual heading appeared as `.title3.weight(.semibold)` in
/// Connect Tools, `.title2.weight(.semibold)` in the missing-sidecar
/// overlay, and a hand-tracked uppercase `scaledSystemFont(11)` in
/// onboarding — three sizes for one role.
///
/// Three emphases:
///   * ``page`` — the one title on a page. 20pt semibold.
///   * ``section`` — a titled division within a page ("Models folder",
///     "Web search"). 15pt semibold, primary.
///   * ``group`` — a quiet label over a group of rows. 11pt semibold,
///     secondary. Deliberately understated: it organises content, it
///     doesn't announce it.
///
/// ``group`` is the default because it is what every pre-existing call
/// site meant when it took the default. The ``section`` tier was added
/// later, for Settings; making IT the default would have silently
/// promoted the sidebar's date headings and Connect Tools' "Endpoint"
/// from 11pt to 15pt.
struct SectionHeader: View {
    enum Emphasis {
        case page
        case section
        case group
    }

    let title: String
    var subtitle: String? = nil
    var emphasis: Emphasis = .group
    /// Trailing control (a "See all", a count, a toggle).
    var accessory: AnyView? = nil

    init(
        _ title: String,
        subtitle: String? = nil,
        emphasis: Emphasis = .group
    ) {
        self.title = title
        self.subtitle = subtitle
        self.emphasis = emphasis
        self.accessory = nil
    }

    init<Accessory: View>(
        _ title: String,
        subtitle: String? = nil,
        emphasis: Emphasis = .group,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.title = title
        self.subtitle = subtitle
        self.emphasis = emphasis
        self.accessory = AnyView(accessory())
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: RapidTheme.Space.md) {
            VStack(alignment: .leading, spacing: subtitle == nil ? 0 : RapidTheme.Space.xs) {
                titleText
                if let subtitle {
                    Text(subtitle)
                        .font(RapidFont.secondary)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            // Keep the heading text as one rotor destination, but do not fold
            // trailing buttons/toggles into it. Combining the whole HStack
            // made visually separate accessories (for example Copy and Save)
            // one unpressable AXHeading.
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            if accessory != nil {
                Spacer(minLength: RapidTheme.Space.sm)
                accessory
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var titleText: some View {
        switch emphasis {
        case .page:
            Text(title)
                .font(RapidFont.pageTitle)
                .foregroundStyle(RapidTheme.textPrimary)
        case .section:
            Text(title)
                .font(RapidFont.sectionTitle)
                .foregroundStyle(RapidTheme.textPrimary)
        case .group:
            // v1.0.1: Title Case, no tracking. ALL-CAPS + letter-spacing
            // gave a purely organisational label more presence than the
            // content under it — "ENDPOINT" was shouting at the values
            // it labels. A quiet 11pt semibold in secondary does the
            // same structural job without competing.
            Text(title)
                .font(RapidFont.groupLabel)
                .foregroundStyle(RapidTheme.textSecondary)
        }
    }
}
