import SwiftUI

/// Window-level search panel opened from the toolbar. It searches the local
/// conversation snapshot as the user types and opens a result in place.
struct ConversationSearchView: View {
    let conversations: [ChatConversation]
    let now: Date
    let onNewChat: () -> Void
    let onSelectConversation: (UUID) -> Void
    let onDismiss: () -> Void

    @State private var query = ""
    @State private var hoveredConversationID: UUID?
    @State private var selectedConversationID: UUID?
    @FocusState private var searchFieldFocused: Bool

    private var results: [ChatConversation] {
        ConversationSearch.results(in: conversations, matching: query)
    }

    private var sections: [ConversationSearch.Section] {
        ConversationSearch.sections(for: results, now: now)
    }

    var body: some View {
        VStack(spacing: 0) {
            searchHeader
            Divider()
                .overlay(RapidTheme.hairlineStrong)
            resultsList
        }
        .background {
            RapidTheme.surfaceOverlay
                .accessibilityElement()
                .accessibilityIdentifier("ConversationSearch.Panel")
        }
        .clipShape(RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: RapidTheme.Radius.card, style: .continuous)
                .strokeBorder(RapidTheme.hairlineStrong, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.22), radius: 28, x: 0, y: 12)
        .onExitCommand(perform: onDismiss)
        .onMoveCommand(perform: moveSelection)
        .onChange(of: query) { _, _ in
            selectedConversationID = results.first?.id
        }
        .task {
            await Task.yield()
            guard !Task.isCancelled else { return }
            selectedConversationID = results.first?.id
            searchFieldFocused = true
        }
    }

    private var searchHeader: some View {
        HStack(spacing: RapidTheme.Space.sm) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)

            TextField("Search chats", text: $query)
                .textFieldStyle(.plain)
                .font(RapidFont.body)
                .focused($searchFieldFocused)
                .onSubmit { openSelectedResult() }
                .accessibilityIdentifier("ConversationSearch.Field")

            if !query.isEmpty {
                QuietIconButton(
                    symbol: "xmark.circle.fill",
                    label: "Clear search",
                    size: RapidTheme.ControlHeight.small,
                    symbolSize: 12
                ) {
                    query = ""
                    searchFieldFocused = true
                }
                .accessibilityIdentifier("ConversationSearch.Clear")
            }

            SheetCloseButton(action: onDismiss)
                .accessibilityIdentifier("ConversationSearch.Close")
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .frame(height: 56)
    }

    private var resultsList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 1) {
                newChatRow

                if conversations.isEmpty {
                    emptyState(title: "No chats yet", symbol: "bubble.left")
                } else if results.isEmpty {
                    emptyState(title: "No chats match", symbol: "magnifyingglass")
                } else {
                    ForEach(sections, id: \.bucket) { section in
                        sectionHeader(section.bucket)
                        ForEach(section.conversations) { conversation in
                            resultRow(conversation)
                        }
                    }
                }
            }
            .padding(RapidTheme.Space.md)
        }
        .scrollIndicators(.never)
    }

    private var newChatRow: some View {
        Button {
            onNewChat()
        } label: {
            HStack(spacing: RapidTheme.Space.md) {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: RapidTheme.Layout.iconSlot)
                Text("New chat")
                    .font(RapidFont.bodyEmphasis)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, RapidTheme.Space.md)
            .frame(height: 42)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("ConversationSearch.NewChat")
    }

    private func sectionHeader(_ bucket: ConversationSearch.Section.Bucket) -> some View {
        Text(localizedTitle(for: bucket))
            .font(RapidFont.groupLabel)
            .foregroundStyle(.secondary)
            .padding(.horizontal, RapidTheme.Space.md)
            .padding(.top, RapidTheme.Space.lg)
            .padding(.bottom, RapidTheme.Space.xs)
            .accessibilityAddTraits(.isHeader)
    }

    private func resultRow(_ conversation: ChatConversation) -> some View {
        let hovering = hoveredConversationID == conversation.id
        let selected = selectedConversationID == conversation.id
        return Button {
            onSelectConversation(conversation.id)
        } label: {
            HStack(spacing: RapidTheme.Space.md) {
                Image(systemName: conversation.isArchived ? "archivebox" : "bubble.left")
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(.secondary)
                    .frame(width: RapidTheme.Layout.iconSlot)
                Text(conversation.title)
                    .font(selected ? RapidFont.bodyEmphasis : RapidFont.body)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: RapidTheme.Space.sm)
                if conversation.isPinned {
                    Image(systemName: "pin.fill")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .accessibilityLabel("Pinned")
                }
            }
            .padding(.horizontal, RapidTheme.Space.md)
            .frame(height: 40)
            .background(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
                    .fill(selected ? RapidTheme.selectionFill : (hovering ? RapidTheme.hoverFill : .clear))
            )
            // Same selected treatment as the sidebar rows this panel is a
            // fast path to — amber bar, neutral fill, semibold label. The
            // search results ARE the conversation list, so answering
            // "which row am I on?" differently in the two places would be
            // two conventions for one question.
            .overlay(alignment: .leading) {
                if selected {
                    Capsule(style: .continuous)
                        .fill(RapidTheme.selectionBar)
                        .frame(
                            width: RapidTheme.Layout.selectionBarWidth,
                            height: RapidTheme.Layout.selectionBarHeight
                        )
                }
            }
            .contentShape(
                RoundedRectangle(cornerRadius: RapidTheme.Radius.row, style: .continuous)
            )
        }
        .buttonStyle(.plain)
        .foregroundStyle(Color.primary)
        .onHover {
            hoveredConversationID = $0 ? conversation.id : nil
            if $0 { selectedConversationID = conversation.id }
        }
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier("ConversationSearch.Result.\(conversation.id.uuidString)")
    }

    private func emptyState(title: LocalizedStringKey, symbol: String) -> some View {
        VStack(spacing: RapidTheme.Space.sm) {
            Image(systemName: symbol)
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(.secondary)
            Text(title)
                .font(RapidFont.body)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, RapidTheme.Space.xxl)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("ConversationSearch.Empty")
    }

    private func openSelectedResult() {
        guard let id = selectedConversationID ?? results.first?.id else { return }
        onSelectConversation(id)
    }

    private func moveSelection(_ direction: MoveCommandDirection) {
        guard direction == .up || direction == .down, !results.isEmpty else { return }
        let currentIndex = selectedConversationID.flatMap { id in
            results.firstIndex(where: { $0.id == id })
        }
        let nextIndex: Int
        switch direction {
        case .up:
            nextIndex = max(0, (currentIndex ?? 1) - 1)
        case .down:
            nextIndex = min(results.count - 1, (currentIndex ?? -1) + 1)
        default:
            return
        }
        selectedConversationID = results[nextIndex].id
    }

    private func localizedTitle(for bucket: ConversationSearch.Section.Bucket) -> LocalizedStringKey {
        switch bucket {
        case .today: return "Today"
        case .yesterday: return "Yesterday"
        case .previous7Days: return "Previous 7 Days"
        case .previous30Days: return "Previous 30 Days"
        case .older: return "Older"
        }
    }
}
