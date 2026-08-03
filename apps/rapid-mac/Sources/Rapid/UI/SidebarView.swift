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
                Text("Older")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.top, 12)
                    .padding(.bottom, 2)
                ScrollView {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(chat.conversations) { conv in
                            conversationRow(conv)
                        }
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(8)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
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
