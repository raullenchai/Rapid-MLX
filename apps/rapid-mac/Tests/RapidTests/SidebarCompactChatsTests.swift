import SwiftUI
import Testing
@testable import Rapid

@MainActor
@Suite("Compact sidebar chats")
struct SidebarCompactChatsTests {
    @Test("Compact history exposes saved, pinned, and folder chats and dismisses on selection")
    func openSavedConversation() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-compact-chats-\(UUID().uuidString)")
        defer {
            ConversationStore.flush()
            ConversationFolderStore.flush()
            try? FileManager.default.removeItem(at: root)
        }
        let store = root.appendingPathComponent("conversations.json")
        let now = Date()
        let conversations = ["Recent chat", "Pinned chat", "Folder chat"].map { title in
            ChatConversation(
                id: UUID(), title: title,
                messages: [ChatMessage(role: .user, content: title)],
                createdAt: now, updatedAt: now
            )
        }
        ConversationStore.save(conversations, to: store)
        ConversationStore.flush()
        let chat = ChatViewModel(conversationStoreURL: store)
        chat.setConversationPinned(conversations[1].id, true)
        let folder = try #require(chat.createFolder(named: "Work"))
        chat.moveConversation(conversations[2].id, toFolder: folder.id)
        var selectedID: UUID?
        let stage = GoldenStage(
            SidebarView(
                isCompact: true,
                selection: .constant(.images),
                chat: chat,
                onNewChat: {},
                onSelectConversation: { id in
                    selectedID = id
                    chat.selectConversation(id)
                }
            ),
            size: CGSize(width: SidebarView.compactWidth, height: 700)
        )

        let anchor = "Sidebar.Compact.Chats"
        let popover = "Sidebar.Compact.Chats.Popover"
        try await stage.waitForIdentifier(anchor)
        #expect(stage.label(of: anchor) == String(localized: "Chats"))
        try stage.press(anchor)
        try await stage.waitForIdentifier(popover)
        for conversation in conversations {
            try await stage.waitForIdentifier("Sidebar.Conversation.\(conversation.id.uuidString)")
        }
        let target = conversations[2]
        try stage.press("Sidebar.Conversation.\(target.id.uuidString)")
        try await stage.waitForIdentifierGone(popover)
        #expect(selectedID == target.id)
        #expect(chat.activeConversationID == target.id)
        #expect(chat.messages.map(\.content) == [target.title])

        // The history remains available after opening a conversation.
        try stage.press(anchor)
        try await stage.waitForIdentifier(popover)
        stage.pressEscape()
        try await stage.waitForIdentifierGone(popover)
    }

    @Test("An empty compact history still opens and explains that no chats are saved")
    func emptyHistory() async throws {
        let stage = GoldenStage(
            SidebarView(
                isCompact: true,
                selection: .constant(.chat),
                chat: ChatViewModel(persistsConversations: false),
                onNewChat: {},
                onSelectConversation: { _ in }
            ),
            size: CGSize(width: SidebarView.compactWidth, height: 700)
        )
        try await stage.waitForIdentifier("Sidebar.Compact.Chats")
        try stage.press("Sidebar.Compact.Chats")
        try await stage.waitForText(String(localized: "No conversations yet"))
        stage.pressEscape()
        try await stage.waitForIdentifierGone("Sidebar.Compact.Chats.Popover")
    }
}
