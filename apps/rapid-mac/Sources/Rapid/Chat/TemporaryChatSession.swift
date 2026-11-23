import Foundation

/// Where temporary chats keep the few things that must live on disk while one
/// is open. Everything under here is deleted when it ends, on quit, and at
/// the next launch.
enum TemporaryChatStorage {
    static let folderName = "temp-chat"

    static func root(
        applicationSupport: URL = ApplicationSupportLocator.applicationSupportRoot()
    ) -> URL {
        applicationSupport.appendingPathComponent(folderName, isDirectory: true)
    }

    static func directory(for windowID: UUID, in root: URL = root()) -> URL {
        root.appendingPathComponent(windowID.uuidString, isDirectory: true)
    }

    static func remove(windowID: UUID, in root: URL = root()) {
        try? FileManager.default.removeItem(at: directory(for: windowID, in: root))
    }

    /// Crash recovery and quit: no window survives either, so nothing under
    /// the root is still wanted.
    static func removeAll(in root: URL = root()) {
        try? FileManager.default.removeItem(at: root)
    }
}

/// One temporary chat's state. Owns a view model that persists
/// nothing, and the private stores its tools and attachments write to.
@MainActor
final class TemporaryChatSession {
    let id: UUID
    let chat: ChatViewModel
    let documentCache: DocumentContentCache

    private let storageRoot: URL
    private var isClosed = false

    /// - Parameter makeTools: builds the window's tool registry around its own
    ///   document and page caches, so no extract reaches the shared ones.
    init(
        id: UUID = UUID(),
        storageRoot: URL = TemporaryChatStorage.root(),
        server: ServerManager?,
        sampling: SamplingConfig?,
        client: ChatStreamClient = ChatStreamClient(),
        makeTools: (DocumentContentCache, BrowseContentCache) -> any ToolRegistry
    ) {
        self.id = id
        self.storageRoot = storageRoot
        let directory = TemporaryChatStorage.directory(for: id, in: storageRoot)
        let documents = DocumentContentCache(
            diskDirectory: directory.appendingPathComponent("documents", isDirectory: true)
        )
        let pages = BrowseContentCache(
            diskDirectory: directory.appendingPathComponent("pages", isDirectory: true)
        )
        self.documentCache = documents
        self.chat = ChatViewModel(
            client: client,
            chatBackendConfig: ChatBackendConfig(pinnedTo: .legacy),
            tools: makeTools(documents, pages),
            sampling: sampling,
            customInstructions: .ephemeral(),
            memoryStore: nil,
            server: server,
            persistsConversations: false,
            documentCache: documents,
            generatesTitles: false
        )
    }

    /// The visible transcript as an exportable conversation.
    func exportSnapshot(now: Date = Date()) -> ChatConversation {
        ChatConversation(
            id: chat.activeConversationID,
            title: ConversationStore.title(from: chat.messages),
            messages: chat.messages,
            createdAt: now,
            updatedAt: now
        )
    }

    /// Idempotent. Stops in-flight work, then deletes this session's directory.
    func close() {
        guard !isClosed else { return }
        isClosed = true
        chat.stop()
        TemporaryChatStorage.remove(windowID: id, in: storageRoot)
    }
}
