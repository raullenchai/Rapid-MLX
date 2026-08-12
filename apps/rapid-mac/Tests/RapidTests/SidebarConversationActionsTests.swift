import Foundation
import Testing
@testable import Rapid

/// The sidebar's per-row actions: rename, pin, archive.
///
/// The interesting behaviour is all in the model + the pure section builder,
/// so these are behavioural tests rather than the source-grep shape used by
/// ``SidebarConversationDeleteConfirmationTests`` (ViewInspector is not in
/// this target — #1492).
@MainActor
@Suite("Sidebar conversation actions")
struct SidebarConversationActionsTests {
    private func isolatedStoreURL() throws -> (root: URL, file: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("rapid-sidebar-actions-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return (root, root.appendingPathComponent("conversations.json"))
    }

    /// A model with exactly one conversation on disk.
    ///
    /// Goes through ``send`` + ``stopAndPersist`` rather than
    /// ``devSeedMessages`` alone: seeding only fills the message buffer, and
    /// the snapshot into ``conversations`` hangs off the end of a turn.
    /// ``stopAndPersist`` cancels the (unroutable) request before it can
    /// stream anything, leaving a persisted single-turn conversation.
    private func seededModel(_ store: URL) -> ChatViewModel {
        let model = ChatViewModel(conversationStoreURL: store)
        model.send("how do I pin a chat", alias: "test-model")
        model.stopAndPersist()
        return model
    }

    private func conversation(
        title: String = "Chat",
        updatedAt: Date = Date(),
        isPinned: Bool = false,
        isArchived: Bool = false
    ) -> ChatConversation {
        ChatConversation(
            id: UUID(),
            title: title,
            messages: [],
            createdAt: updatedAt,
            updatedAt: updatedAt,
            isPinned: isPinned,
            isArchived: isArchived
        )
    }

    // MARK: - Rename

    @Test("Rename trims, persists, and survives a reload")
    func renamePersists() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        #expect(model.renameConversation(id, to: "  Pinning notes  "))
        ConversationStore.flush()

        let reloaded = ChatViewModel(conversationStoreURL: store.file)
        let conv = reloaded.conversations.first(where: { $0.id == id })
        #expect(conv?.title == "Pinning notes")
        #expect(conv?.hasCustomTitle == true)
    }

    @Test("Rename rejects blank input rather than clearing the row label")
    func renameRejectsBlank() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID
        let before = model.conversations.first?.title

        #expect(!model.renameConversation(id, to: "   "))
        #expect(model.conversations.first?.title == before)
    }

    /// The regression this flag exists for: ``persistActive`` re-derives the
    /// title from the first user turn on every save, so without the
    /// ``hasCustomTitle`` guard the very next persisted turn reverted the
    /// user's rename back to the opening words of their prompt.
    @Test("A renamed conversation keeps its name across later saves")
    func renameSurvivesLaterPersists() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID
        #expect(model.renameConversation(id, to: "Pinning notes"))

        // A later turn on the same conversation re-runs the title derivation.
        model.send("and archive?", alias: "test-model")
        model.stopAndPersist()

        #expect(model.conversations.first(where: { $0.id == id })?.title == "Pinning notes")
    }

    // MARK: - Pin / archive

    @Test("Pinning and archiving persist and are mutually exclusive")
    func pinAndArchiveArePersistedAndExclusive() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        model.setConversationPinned(id, true)
        // Archiving a pinned row clears the pin: a row can't be pinned to the
        // top of a list it has been filed out of.
        model.setConversationArchived(id, true)
        #expect(model.conversations.first?.isPinned == false)
        #expect(model.conversations.first?.isArchived == true)

        // ...and pinning an archived row brings it back into the main list.
        model.setConversationPinned(id, true)
        #expect(model.conversations.first?.isArchived == false)
        ConversationStore.flush()

        let reloaded = ChatViewModel(conversationStoreURL: store.file)
        #expect(reloaded.conversations.first?.isPinned == true)
    }

    @Test("Archiving the open conversation leaves it open")
    func archivingOpenConversationDoesNotNavigate() throws {
        let store = try isolatedStoreURL()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let model = seededModel(store.file)
        let id = model.activeConversationID

        model.setConversationArchived(id, true)
        #expect(model.activeConversationID == id)
        #expect(!model.messages.isEmpty)
    }

    // MARK: - Sectioning

    @Test("Pinned rows lift into their own leading section, exempt from date buckets")
    func pinnedRowsGetTheirOwnSection() {
        let old = Date().addingTimeInterval(-60 * 60 * 24 * 30)
        let pinned = conversation(title: "Pinned but stale", updatedAt: old, isPinned: true)
        let today = conversation(title: "Today's chat")
        let sections = SidebarView.sections(for: [pinned, today], now: Date())

        #expect(sections.first?.title == "Pinned")
        #expect(sections.first?.conversations.map(\.id) == [pinned.id])
        // The pinned row must NOT also appear under "Older".
        #expect(!sections.contains { $0.title == "Older" })
    }

    @Test("Archived rows are excluded from the main sections and listed separately")
    func archivedRowsAreSplitOut() {
        let archived = conversation(title: "Filed away", isArchived: true)
        let active = conversation(title: "Live chat")
        let sections = SidebarView.sections(for: [archived, active], now: Date())

        let listed = sections.flatMap(\.conversations).map(\.id)
        #expect(listed == [active.id])
        #expect(SidebarView.archived(for: [archived, active]).map(\.id) == [archived.id])
    }

    /// ``archived(for:)`` sorts rather than inheriting the caller's order.
    /// Archiving does not touch ``updatedAt`` or a row's slot in
    /// ``ChatViewModel.conversations``, so the input can hand the archived
    /// rows over in any order — the group still has to read newest-first.
    @Test("Archived rows are ordered newest-updated first")
    func archivedRowsAreSortedByRecency() {
        let now = Date()
        let stale = conversation(
            title: "Filed long ago",
            updatedAt: now.addingTimeInterval(-60 * 60 * 24 * 30),
            isArchived: true
        )
        let recent = conversation(title: "Filed today", updatedAt: now, isArchived: true)

        #expect(SidebarView.archived(for: [stale, recent]).map(\.id) == [recent.id, stale.id])
    }

    // MARK: - Backward compatibility

    /// A history file written before pin/archive shipped must still decode.
    /// The synthesised ``Codable`` conformance treats every stored property as
    /// required, so a missing `isPinned` key would THROW — and
    /// ``ConversationStore.load`` reads one throw as "the whole file is
    /// corrupt", sides it off to `.corrupt-*`, and returns empty. That is an
    /// apparently wiped sidebar on upgrade, which is why the hand-written
    /// ``init(from:)`` defaults the new keys instead.
    @Test("A pre-pin/archive history file still decodes, defaulting the new flags")
    func legacyHistoryDecodesWithDefaults() throws {
        let legacy = """
        [{
          "id": "6F3F2B2F-6F94-4B49-9C1E-6B4B0A5E1D77",
          "title": "Legacy chat",
          "messages": [],
          "createdAt": 700000000,
          "updatedAt": 700000000
        }]
        """
        let decoded = try JSONDecoder().decode(
            [ChatConversation].self,
            from: Data(legacy.utf8)
        )
        #expect(decoded.count == 1)
        #expect(decoded.first?.title == "Legacy chat")
        #expect(decoded.first?.isPinned == false)
        #expect(decoded.first?.isArchived == false)
        #expect(decoded.first?.hasCustomTitle == false)
        #expect(decoded.first?.customInstructions == nil)
    }
}
