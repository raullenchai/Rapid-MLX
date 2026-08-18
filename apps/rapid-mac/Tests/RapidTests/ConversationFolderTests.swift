import Foundation
import Testing
@testable import Rapid

/// User-created folders: the model, the store, and the sidebar grouping.
@MainActor
@Suite("Conversation folders")
struct ConversationFolderTests {
    private func isolatedStore() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-folders-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("conversations.json")
    }

    private func conversation(
        title: String = "Chat",
        updatedAt: Date = Date(),
        isPinned: Bool = false,
        isArchived: Bool = false,
        folderID: UUID? = nil
    ) -> ChatConversation {
        ChatConversation(
            id: UUID(),
            title: title,
            messages: [],
            createdAt: updatedAt,
            updatedAt: updatedAt,
            isPinned: isPinned,
            isArchived: isArchived,
            folderID: folderID
        )
    }

    // MARK: - Model + CRUD

    @Test("Creating a folder rejects a blank name")
    func createRejectsBlankName() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        #expect(model.createFolder(named: "   ") == nil)
        #expect(model.createFolder(named: "") == nil)
        #expect(model.folders.isEmpty)
    }

    @Test("Creating a folder trims the name")
    func createTrimsName() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        let folder = model.createFolder(named: "  Work  ")
        #expect(folder?.name == "Work")
        #expect(model.folders.count == 1)
    }

    @Test("Folder names are unique regardless of case")
    func duplicateNamesAreRejected() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        let work = try #require(model.createFolder(named: "Work"))
        #expect(model.createFolder(named: "work") == nil)
        let personal = try #require(model.createFolder(named: "Personal"))
        #expect(model.renameFolder(personal.id, to: "WORK") == false)
        #expect(model.renameFolder(work.id, to: "work") == true)
        #expect(model.folders.map(\.name) == ["work", "Personal"])
    }

    @Test("Renaming rejects blank and keeps the old name")
    func renameRejectsBlank() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        let folder = try #require(model.createFolder(named: "Work"))
        #expect(model.renameFolder(folder.id, to: "  ") == false)
        #expect(model.folders.first?.name == "Work")
        #expect(model.renameFolder(folder.id, to: "Personal") == true)
        #expect(model.folders.first?.name == "Personal")
    }

    @Test("Deleting a folder keeps its conversations and unfiles them")
    func deleteFolderKeepsConversations() throws {
        let store = try isolatedStore()
        let model = ChatViewModel(conversationStoreURL: store)
        let folder = try #require(model.createFolder(named: "Work"))
        model.send("hello", alias: "test-model")
        model.stopAndPersist()
        let conversationID = try #require(model.conversations.first).id
        model.moveConversation(conversationID, toFolder: folder.id)
        #expect(model.conversations.first?.folderID == folder.id)

        model.deleteFolder(folder.id)

        // The transcript is the valuable thing — deleting the folder must not
        // take it with them.
        #expect(model.conversations.count == 1)
        #expect(model.conversations.first?.folderID == nil)
        #expect(model.folders.isEmpty)
    }

    @Test("A conversation can't be filed into a folder that doesn't exist")
    func moveRejectsUnknownFolder() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        model.send("hello", alias: "test-model")
        model.stopAndPersist()
        let conversationID = try #require(model.conversations.first).id

        model.moveConversation(conversationID, toFolder: UUID())

        // Filing into a nonexistent folder would put the row in a section
        // that never renders, which looks exactly like deletion.
        #expect(model.conversations.first?.folderID == nil)
    }

    @Test("Filing an archived conversation surfaces it into the folder")
    func filingUnarchives() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        let folder = try #require(model.createFolder(named: "Work"))
        model.send("hello", alias: "test-model")
        model.stopAndPersist()
        let id = try #require(model.conversations.first).id
        model.setConversationArchived(id, true)
        #expect(model.conversations.first?.isArchived == true)

        model.moveConversation(id, toFolder: folder.id)

        // Archived rows render only in the Archived disclosure, ahead of any
        // folder — filing without surfacing would record the folder and
        // change nothing visible, i.e. look like the drop did nothing.
        #expect(model.conversations.first?.isArchived == false)
        #expect(model.conversations.first?.folderID == folder.id)
        let sections = SidebarView.folderSections(
            for: model.conversations,
            folders: model.folders
        )
        #expect(sections.first?.conversations.count == 1)
    }

    @Test("Unfiling does not archive — there is no earlier state to restore")
    func unfilingDoesNotArchive() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        let folder = try #require(model.createFolder(named: "Work"))
        model.send("hello", alias: "test-model")
        model.stopAndPersist()
        let id = try #require(model.conversations.first).id
        model.moveConversation(id, toFolder: folder.id)

        model.moveConversation(id, toFolder: nil)

        #expect(model.conversations.first?.folderID == nil)
        #expect(model.conversations.first?.isArchived == false)
    }

    @Test("The drag payload carries only the id, so it can't go stale")
    func transferCarriesIdentityOnly() throws {
        let id = UUID()
        let transfer = ConversationTransfer(id)
        #expect(transfer.conversationID == id)
        // Round-trips through the same Codable representation the drag uses.
        let data = try JSONEncoder().encode(transfer)
        let restored = try JSONDecoder().decode(ConversationTransfer.self, from: data)
        #expect(restored == transfer)
    }

    @Test("Filing a conversation is not activity: order and timestamp hold")
    func filingIsNotActivity() throws {
        let model = ChatViewModel(conversationStoreURL: try isolatedStore())
        let folder = try #require(model.createFolder(named: "Work"))
        model.send("hello", alias: "test-model")
        model.stopAndPersist()
        let before = try #require(model.conversations.first)

        model.moveConversation(before.id, toFolder: folder.id)

        let after = try #require(model.conversations.first)
        #expect(after.id == before.id)
        #expect(after.updatedAt == before.updatedAt)
    }

    // MARK: - Persistence

    @Test("Folders and filing survive a relaunch")
    func foldersPersist() throws {
        let store = try isolatedStore()
        let folderID: UUID
        do {
            let model = ChatViewModel(conversationStoreURL: store)
            let folder = try #require(model.createFolder(named: "Work"))
            folderID = folder.id
            model.send("hello", alias: "test-model")
            model.stopAndPersist()
            model.moveConversation(try #require(model.conversations.first).id, toFolder: folder.id)
            ConversationStore.flush()
            ConversationFolderStore.flush()
        }

        let relaunched = ChatViewModel(conversationStoreURL: store)
        #expect(relaunched.folders.map(\.name) == ["Work"])
        #expect(relaunched.conversations.first?.folderID == folderID)
    }

    @Test("History written before folders shipped still loads")
    func legacyHistoryWithoutFolderIDLoads() throws {
        let store = try isolatedStore()
        // A conversations.json exactly as a pre-folders build wrote it: no
        // folderID key anywhere, and no folders.json beside it. Decoding this
        // as corrupt would side the file and present as a wiped sidebar.
        let legacy = """
        [{
          "id": "\(UUID().uuidString)",
          "title": "Old chat",
          "messages": [],
          "createdAt": 760000000,
          "updatedAt": 760000000
        }]
        """
        try Data(legacy.utf8).write(to: store)

        let model = ChatViewModel(conversationStoreURL: store)
        #expect(model.conversations.count == 1)
        #expect(model.conversations.first?.title == "Old chat")
        #expect(model.conversations.first?.folderID == nil)
        #expect(model.folders.isEmpty)
    }

    @Test("A corrupt folder file is sided, not fatal to the history")
    func corruptFolderFileIsSided() throws {
        let store = try isolatedStore()
        let folderFile = try #require(
            ConversationFolderStore.companionURL(forConversationStore: store)
        )
        try Data("{ not json".utf8).write(to: folderFile)

        let loaded = ConversationFolderStore.load(from: folderFile)
        #expect(loaded.isEmpty)
        // Sided rather than overwritten in place, so the bad file is
        // recoverable and the next save can't clobber it.
        let siblings = try FileManager.default.contentsOfDirectory(
            atPath: folderFile.deletingLastPathComponent().path
        )
        #expect(siblings.contains { $0.hasPrefix("folders.corrupt-") })
    }

    // MARK: - Ordering + identifiers

    @Test("Folders sort case-insensitively by name")
    func displayOrderIsByName() {
        let folders = [
            ChatFolder(name: "zebra"),
            ChatFolder(name: "Alpha"),
            ChatFolder(name: "beta"),
        ]
        #expect(ChatFolder.displayOrder(folders).map(\.name) == ["Alpha", "beta", "zebra"])
    }

    @Test("AX slugs are name-derived and predictable for the golden harness")
    func axSlugIsNameDerived() {
        #expect(ChatFolder(name: "Work").axSlug == "Work")
        #expect(ChatFolder(name: "Q3 / budget").axSlug == "Q3-budget")
        #expect(ChatFolder(name: "···").axSlug == "Folder")
    }

    // MARK: - Sidebar grouping

    @Test("A filed conversation appears in its folder and nowhere else")
    func filedRowsLeaveTheDateBuckets() {
        let folder = ChatFolder(name: "Work")
        let now = Date()
        let filed = conversation(title: "Filed", updatedAt: now, folderID: folder.id)
        let loose = conversation(title: "Loose", updatedAt: now)

        let buckets = SidebarView.sections(
            for: [filed, loose],
            folders: [folder],
            now: now
        )
        let folders = SidebarView.folderSections(for: [filed, loose], folders: [folder])

        #expect(buckets.flatMap(\.conversations).map(\.title) == ["Loose"])
        #expect(folders.first?.conversations.map(\.title) == ["Filed"])
    }

    @Test("Filing outranks pinning, so a row is never listed twice")
    func filingOutranksPinning() {
        let folder = ChatFolder(name: "Work")
        let now = Date()
        let pinnedAndFiled = conversation(
            title: "Both",
            updatedAt: now,
            isPinned: true,
            folderID: folder.id
        )

        let buckets = SidebarView.sections(for: [pinnedAndFiled], folders: [folder], now: now)
        let folders = SidebarView.folderSections(for: [pinnedAndFiled], folders: [folder])

        #expect(buckets.isEmpty)
        #expect(folders.first?.conversations.map(\.title) == ["Both"])
    }

    @Test("Archiving outranks filing")
    func archivingOutranksFiling() {
        let folder = ChatFolder(name: "Work")
        let now = Date()
        let archived = conversation(
            title: "Archived",
            updatedAt: now,
            isArchived: true,
            folderID: folder.id
        )

        #expect(SidebarView.folderSections(for: [archived], folders: [folder]).first?.conversations.isEmpty == true)
        #expect(SidebarView.sections(for: [archived], folders: [folder], now: now).isEmpty)
        #expect(SidebarView.archived(for: [archived]).map(\.title) == ["Archived"])
    }

    @Test("An orphaned folderID falls back to the date buckets instead of vanishing")
    func orphanedFolderIDDegrades() {
        let now = Date()
        let orphan = conversation(title: "Orphan", updatedAt: now, folderID: UUID())

        // No folders at all — the id names one that has been deleted.
        let buckets = SidebarView.sections(for: [orphan], folders: [], now: now)
        #expect(buckets.flatMap(\.conversations).map(\.title) == ["Orphan"])

        // And with unrelated folders present, it still shows up exactly once.
        let other = ChatFolder(name: "Work")
        let withFolders = SidebarView.sections(for: [orphan], folders: [other], now: now)
        let grouped = SidebarView.folderSections(for: [orphan], folders: [other])
        #expect(withFolders.flatMap(\.conversations).map(\.title) == ["Orphan"])
        #expect(grouped.flatMap(\.conversations).isEmpty)
    }

    @Test("An empty folder still renders, so creating one doesn't look broken")
    func emptyFolderStillRenders() {
        let folder = ChatFolder(name: "Work")
        let sections = SidebarView.folderSections(for: [], folders: [folder])
        #expect(sections.count == 1)
        #expect(sections.first?.conversations.isEmpty == true)
    }

    @Test("Inside a folder, pinned rows lead and the rest are newest-first")
    func withinFolderOrdering() {
        let folder = ChatFolder(name: "Work")
        let now = Date()
        let old = conversation(
            title: "Old",
            updatedAt: now.addingTimeInterval(-10_000),
            folderID: folder.id
        )
        let recent = conversation(title: "Recent", updatedAt: now, folderID: folder.id)
        let pinned = conversation(
            title: "Pinned",
            updatedAt: now.addingTimeInterval(-50_000),
            isPinned: true,
            folderID: folder.id
        )

        let section = SidebarView.folderSections(
            for: [old, recent, pinned],
            folders: [folder]
        ).first
        #expect(section?.conversations.map(\.title) == ["Pinned", "Recent", "Old"])
    }

    @Test("No folders means no folder sections at all")
    func noFoldersNoSections() {
        let now = Date()
        #expect(SidebarView.folderSections(for: [conversation()], folders: []).isEmpty)
        // And the buckets behave exactly as they did before folders existed.
        #expect(SidebarView.sections(for: [conversation()], now: now).count == 1)
    }
}
