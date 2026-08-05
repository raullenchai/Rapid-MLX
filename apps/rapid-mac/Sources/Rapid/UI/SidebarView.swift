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
    @Binding var selection: SidebarSection
    /// The chat model — source of the conversation history list + the
    /// active conversation id (for highlighting).
    @Bindable var chat: ChatViewModel
    /// Start a fresh conversation and show the chat surface.
    var onNewChat: () -> Void
    /// Open a saved conversation (switches the detail pane back to chat).
    var onSelectConversation: (UUID) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
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
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(historySections, id: \.title) { section in
                            Text(section.title)
                                .font(.caption.weight(.medium))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 10)
                                .padding(.top, 12)
                                .padding(.bottom, 2)
                            ForEach(section.conversations) { conv in
                                conversationRow(conv)
                            }
                        }
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(8)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    /// The history list split into dated sections, newest first.
    ///
    /// Everything used to sit under a hard-coded "Older" heading, so a
    /// conversation five seconds old was filed as ancient history.
    private var historySections: [HistorySection] {
        SidebarView.sections(for: chat.conversations, now: Date())
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
        return Button {
            onSelectConversation(conv.id)
        } label: {
            Text(conv.title)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(isActive ? RapidTheme.brandAmberTint : Color.clear)
                )
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundStyle(isActive ? RapidTheme.brandAmber : Color.primary)
        .contextMenu {
            Button("Delete", role: .destructive) {
                chat.deleteConversation(conv.id)
            }
        }
    }

    /// One sidebar row — a borderless button that fills the column and
    /// paints an amber-tinted rounded highlight when selected (matching
    /// the design system's selection = amber role).
    private func row(
        title: String,
        systemImage: String,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.body)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(isSelected ? RapidTheme.brandAmberTint : Color.clear)
                )
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? RapidTheme.brandAmber : Color.primary)
    }
}

/// The "Launch" detail page — reuses the connect-your-tools cards. Until
/// the engine ships `rapid-mlx launch <tool>` (issue #1405) these are the
/// copy-the-config cards; that CLI upgrade will turn each into a one-line
/// `rapid-mlx launch …` command without changing this surface's shape.
struct LaunchView: View {
    @Bindable var server: ServerManager
    let alias: String

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
            showsCloseButton: false
        )
    }
}
