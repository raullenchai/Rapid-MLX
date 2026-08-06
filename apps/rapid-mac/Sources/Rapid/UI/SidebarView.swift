import SwiftUI

/// Which surface the detail pane shows. Ollama-style: a chat surface and
/// a "Launch" page of connect-your-tools cards. Conversation history
/// ("Older" list) is a later milestone.
enum SidebarSection: Hashable {
    case chat
    case launch
}

/// The left sidebar — Ollama/ChatGPT layout: a "New Chat" action at the
/// top, a "Launch" page entry, then (later) the conversation history. It
/// is the primary column of ``ContentView``'s ``NavigationSplitView``, so
/// macOS gives us the collapse toggle in the toolbar for free.
struct SidebarView: View {
    /// Column metrics. v1.0: narrowed from 190/230/300.
    ///
    /// At the old ideal the rail took ~35% of a 640pt window and ~26% of
    /// a 900pt one for two nav rows and a usually-empty history list —
    /// which is what made the mostly-blank column read as unfinished.
    /// A 200pt rail still fits the longest conversation titles at this
    /// row density while giving the detail pane the width it needs at
    /// the 640pt floor.
    ///
    /// Exposed (rather than inlined at the ``ContentView`` call site) so
    /// the snapshot harness composes the same column it ships.
    static let columnMinWidth: CGFloat = 176
    static let columnIdealWidth: CGFloat = 200
    static let columnMaxWidth: CGFloat = 260

    @Binding var selection: SidebarSection
    /// The chat model — source of the conversation history list + the
    /// active conversation id (for highlighting).
    @Bindable var chat: ChatViewModel
    /// Start a fresh conversation and show the chat surface.
    var onNewChat: () -> Void
    /// Open a saved conversation (switches the detail pane back to chat).
    var onSelectConversation: (UUID) -> Void

    /// The "now" the date buckets are computed against. Rolled forward by
    /// ``dayBoundaryTicker`` at each midnight so an open, untouched sidebar
    /// re-labels yesterday's conversations instead of freezing on the day it
    /// was first rendered. Injected (rather than reading ``Date()`` inside the
    /// section builder) so the roll-over is an observable state change.
    @State private var referenceDate = Date()

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            row(
                title: "New Chat",
                systemImage: "square.and.pencil",
                isSelected: false,
                action: onNewChat
            )
            row(
                title: "Launch",
                systemImage: "paperplane",
                isSelected: selection == .launch,
                action: { selection = .launch }
            )

            if !chat.conversations.isEmpty {
                // Date-grouped history (#1470), titled with the shared
                // SectionHeader so the groups match the refreshed visual
                // system (#1460) instead of the PR's original inline caption.
                ScrollView {
                    VStack(alignment: .leading, spacing: 1) {
                        ForEach(historySections, id: \.title) { section in
                            SectionHeader(section.title)
                                .padding(.horizontal, RapidTheme.Space.sm)
                                .padding(.top, RapidTheme.Space.lg)
                                .padding(.bottom, RapidTheme.Space.xs)
                            ForEach(section.conversations) { conv in
                                conversationRow(conv)
                            }
                        }
                    }
                }
                .scrollIndicators(.never)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, RapidTheme.Space.sm)
        .padding(.vertical, RapidTheme.Space.md)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .task { await dayBoundaryTicker() }
    }

    /// The history list split into dated sections, newest first.
    ///
    /// Everything used to sit under a hard-coded "Older" heading, so a
    /// conversation five seconds old was filed as ancient history.
    private var historySections: [HistorySection] {
        SidebarView.sections(for: chat.conversations, now: referenceDate)
    }

    /// Advance ``referenceDate`` at each calendar-day boundary for as long as
    /// the sidebar is on screen. Sleeps until the next midnight, bumps the
    /// state (which re-buckets the list), then loops. Cancels with the view.
    private func dayBoundaryTicker() async {
        let calendar = Calendar.current
        while !Task.isCancelled {
            let next =
                calendar.nextDate(
                    after: Date(),
                    matching: DateComponents(hour: 0, minute: 0, second: 0),
                    matchingPolicy: .nextTime
                ) ?? Date().addingTimeInterval(24 * 60 * 60)
            let delay = next.timeIntervalSinceNow
            if delay > 0 {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            if Task.isCancelled { break }
            referenceDate = Date()
        }
    }

    struct HistorySection {
        let title: String
        let conversations: [ChatConversation]
    }

    /// Bucket conversations by recency. ``now`` is injected rather than read
    /// inside, matching ``RelativeTimestamp`` — it keeps the function pure so
    /// the day boundaries can be exercised without waiting for midnight.
    ///
    /// Uses `Calendar` (not a fixed 86 400s divisor) because the buckets are
    /// *calendar* days: something sent at 23:55 belongs to "Yesterday" once
    /// the clock passes midnight, even though barely any time has elapsed.
    /// Empty buckets produce no section, so no stray headings appear.
    static func sections(
        for conversations: [ChatConversation],
        now: Date,
        calendar: Calendar = .current
    ) -> [HistorySection] {
        var today: [ChatConversation] = []
        var yesterday: [ChatConversation] = []
        var week: [ChatConversation] = []
        var older: [ChatConversation] = []

        // The 7-day cutoff is anchored to the START of today, not to `now` —
        // otherwise the boundary would drift through the day and a
        // conversation could slide between sections while the user watches.
        let startOfToday = calendar.startOfDay(for: now)
        let weekCutoff = calendar.date(byAdding: .day, value: -7, to: startOfToday)

        for conv in conversations {
            if calendar.isDate(conv.updatedAt, inSameDayAs: now) {
                today.append(conv)
            } else if calendar.isDateInYesterday(conv.updatedAt) {
                yesterday.append(conv)
            } else if let weekCutoff, conv.updatedAt >= weekCutoff {
                week.append(conv)
            } else {
                older.append(conv)
            }
        }

        return [
            ("Today", today),
            ("Yesterday", yesterday),
            ("Previous 7 Days", week),
            ("Older", older),
        ]
        .filter { !$0.1.isEmpty }
        .map { HistorySection(title: $0.0, conversations: $0.1) }
    }

    /// One history row — the conversation's derived title, amber-selected
    /// when it's the open one, with a right-click Delete.
    private func conversationRow(_ conv: ChatConversation) -> some View {
        let isActive = selection == .chat && conv.id == chat.activeConversationID
        return SidebarRow(isSelected: isActive) {
            onSelectConversation(conv.id)
        } content: {
            // History rows carry no icon but keep the same leading inset
            // as the nav rows above, so titles and nav labels align down
            // one column instead of stepping in and out.
            Text(conv.title)
                .font(RapidFont.body)
                .lineLimit(1)
                .truncationMode(.tail)
                .padding(.leading, RapidTheme.Layout.iconSlot + RapidTheme.Space.sm)
        }
        .contextMenu {
            Button("Delete", role: .destructive) {
                chat.deleteConversation(conv.id)
            }
        }
    }

    /// One nav row — icon in a fixed-width slot so every label starts on
    /// the same x, whatever the glyph's natural width.
    private func row(
        title: String,
        systemImage: String,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        SidebarRow(isSelected: isSelected, action: action) {
            HStack(spacing: RapidTheme.Space.sm) {
                Image(systemName: systemImage)
                    .font(.system(size: 13, weight: .medium))
                    .frame(width: RapidTheme.Layout.iconSlot, alignment: .center)
                Text(title)
                    .font(RapidFont.body)
                    .lineLimit(1)
            }
        }
    }
}

/// Shared chrome for every sidebar row: fixed height, one row radius,
/// amber selection, neutral hover.
///
/// The selected treatment is the product's canonical "this is chosen"
/// signal — amber tint fill plus the deep-amber label. Deep amber (not
/// raw ``brandPrimary``) because a 13pt label in #EFA23A on the light
/// tint is under 3:1; the deeper shade of the same hue clears AA while
/// reading as the same colour.
private struct SidebarRow<Content: View>: View {
    let isSelected: Bool
    let action: () -> Void
    @ViewBuilder let content: Content

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            content
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, RapidTheme.Space.sm)
                .frame(height: RapidTheme.ControlHeight.row)
                .background(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                        .fill(fill)
                )
                .contentShape(
                    RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                )
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? RapidTheme.brandPrimaryDeep : Color.primary)
        .onHover { hovering = $0 }
        .rapidAnimation(RapidMotion.quick, value: hovering)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var fill: Color {
        if isSelected { return RapidTheme.brandPrimaryTint }
        return hovering ? RapidTheme.hoverFill : .clear
    }
}

/// The "Launch" detail page — reuses the connect-your-tools cards. Until
/// the engine ships `rapid-mlx launch <tool>` (issue #1405) these are the
/// copy-the-config cards; that CLI upgrade will turn each into a one-line
/// `rapid-mlx launch …` command without changing this surface's shape.
struct LaunchView: View {
    @Bindable var server: ServerManager
    let alias: String
    /// The window's shared readiness value. Optional so the dev snapshot
    /// harness can render the page standalone; when supplied, the page
    /// describes the model lifecycle in exactly the same words the chat
    /// composer does, and offers the same next step.
    var readiness: ModelReadiness? = nil
    var onReadinessAction: (ModelReadiness.Action) -> Void = { _ in }

    var body: some View {
        ConnectToolsView(
            host: "127.0.0.1",
            port: server.activePort,
            bearer: server.activeBearer ?? "",
            alias: alias,
            // Page context: the sidebar owns navigation, so there is no sheet
            // to dismiss. Hide the close ✕ (it used to render as a dead
            // no-op button). A dedicated page-mode header lands with the
            // #1405 Launch redesign.
            onClose: {},
            showsCloseButton: false,
            readiness: readiness,
            onReadinessAction: onReadinessAction
        )
    }
}
