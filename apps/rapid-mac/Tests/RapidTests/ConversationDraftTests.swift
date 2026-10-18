import Foundation
import XCTest
@testable import Rapid

final class ConversationDraftTests: XCTestCase {
    private func isolatedStore() throws -> URL {
        let root = URL(fileURLWithPath: "/private/tmp/harbor-4862-draft-tests")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock {
            ConversationStore.flush()
            ConversationDraftStore.flush()
            try FileManager.default.removeItem(at: root)
        }
        return root.appendingPathComponent("conversations.json")
    }

    private func conversation(_ title: String) -> ChatConversation {
        ChatConversation(id: UUID(), title: title, messages: [], createdAt: Date(), updatedAt: Date())
    }

    @MainActor
    private func seededModel(at url: URL) -> (ChatViewModel, UUID, UUID) {
        let a = conversation("A"), b = conversation("B")
        ConversationStore.save([a, b], to: url)
        ConversationStore.flush()
        return (ChatViewModel(persistsConversations: true, conversationStoreURL: url), a.id, b.id)
    }

    @MainActor
    func testIsolationAndNewChat() throws {
        let (model, a, b) = seededModel(at: try isolatedStore())
        model.selectConversation(a)
        model.composerDraft = "  A's exact text\n"
        model.selectConversation(b)
        XCTAssertEqual(model.composerDraft, "")
        model.composerDraft = "B's text"
        model.selectConversation(a)
        XCTAssertEqual(model.composerDraft, "  A's exact text\n")
        model.newConversation()
        XCTAssertEqual(model.composerDraft, "")
        model.composerDraft = "New Chat text"
        model.newConversation()
        XCTAssertEqual(model.composerDraft, "New Chat text")
        model.selectConversation(a)
        XCTAssertEqual(model.composerDraft, "  A's exact text\n")
        model.selectConversation(b)
        XCTAssertEqual(model.composerDraft, "B's text")
        model.stopAndPersist()
    }

    @MainActor
    func testPersistenceAndIdleTermination() throws {
        let url = try isolatedStore()
        let (model, a, b) = seededModel(at: url)
        model.selectConversation(a)
        model.composerDraft = "A"
        model.selectConversation(b)
        model.composerDraft = "B"
        model.newConversation()
        model.composerDraft = "New"
        model.stopAndPersist() // Must enqueue even without an active stream.
        ConversationDraftStore.flush()
        let restored = ChatViewModel(conversationStoreURL: url)
        XCTAssertEqual(restored.composerDraft, "New")
        restored.selectConversation(a)
        XCTAssertEqual(restored.composerDraft, "A")
        restored.selectConversation(b)
        XCTAssertEqual(restored.composerDraft, "B")
    }

    @MainActor
    func testClearOnlySendingConversationAndFirstSendTransition() throws {
        let url = try isolatedStore()
        let (model, a, b) = seededModel(at: url)
        model.selectConversation(a)
        model.composerDraft = "A"
        model.selectConversation(b)
        model.composerDraft = "  B\n"
        var submitted: [String] = []
        XCTAssertTrue(model.submitComposerDraft(hasAttachments: false, canSubmit: {
            XCTAssertEqual(model.composerDraft, "  B\n")
            return true
        }, submit: { text in
            XCTAssertEqual(model.composerDraft, "")
            submitted.append(text)
        }))
        XCTAssertEqual(submitted, ["  B\n"])
        XCTAssertEqual(model.composerDraft, "")
        model.selectConversation(a)
        XCTAssertEqual(model.composerDraft, "A")
        model.newConversation()
        model.composerDraft = "first send"
        let newID = model.activeConversationID
        XCTAssertTrue(model.submitComposerDraft(hasAttachments: false, canSubmit: { true }, submit: { text in
            XCTAssertEqual(model.composerDraft, "")
            XCTAssertEqual(text, "first send")
            model.send(text, alias: "test-model")
        }))
        model.stopAndPersist()
        XCTAssertTrue(model.conversations.contains { $0.id == newID })
        XCTAssertEqual(model.composerDraft, "")
        model.newConversation()
        XCTAssertEqual(model.composerDraft, "")
        model.selectConversation(a)
        XCTAssertEqual(model.composerDraft, "A")
        ConversationDraftStore.flush()
        let saved = ConversationDraftStore.load(from: ConversationDraftStore.companionURL(forConversationStore: url))
        XCTAssertEqual(saved.text(for: ConversationDrafts.newChatKey), "")
        XCTAssertEqual(saved.text(for: a.uuidString), "A")
        XCTAssertEqual(saved.text(for: b.uuidString), "")
    }

    @MainActor
    func testGateBlockedSubmissionPreservesDraft() throws {
        let model = ChatViewModel(persistsConversations: false, conversationStoreURL: try isolatedStore())
        model.composerDraft = "  waiting\n"
        var gateCalls = 0
        XCTAssertFalse(model.submitComposerDraft(hasAttachments: false, canSubmit: {
            gateCalls += 1
            XCTAssertEqual(model.composerDraft, "  waiting\n")
            return false
        }, submit: { _ in
            XCTFail("Blocked submission must not send")
        }))
        XCTAssertEqual(gateCalls, 1)
        XCTAssertEqual(model.composerDraft, "  waiting\n")
    }

    @MainActor
    func testBlankSubmissionWithoutAttachmentsIsNoOp() throws {
        let model = ChatViewModel(persistsConversations: false, conversationStoreURL: try isolatedStore())
        for text in ["", "  \n\t"] {
            model.composerDraft = text
            XCTAssertFalse(model.submitComposerDraft(hasAttachments: false, canSubmit: {
                XCTFail("Empty submission must not reach the gate")
                return true
            }, submit: { _ in
                XCTFail("Empty submission must not send")
            }))
            XCTAssertEqual(model.composerDraft, text)
        }
    }

    @MainActor
    func testBlankSubmissionWithAttachmentsSends() throws {
        let model = ChatViewModel(persistsConversations: false, conversationStoreURL: try isolatedStore())
        for text in ["", "  \n\t"] {
            model.composerDraft = text
            var gateCalls = 0
            var submitted: [String] = []
            XCTAssertTrue(model.submitComposerDraft(hasAttachments: true, canSubmit: {
                gateCalls += 1
                XCTAssertEqual(model.composerDraft, text)
                return true
            }, submit: { text in
                XCTAssertEqual(model.composerDraft, "")
                submitted.append(text)
            }))
            XCTAssertEqual(gateCalls, 1)
            XCTAssertEqual(submitted, [text])
            XCTAssertEqual(model.composerDraft, "")
        }
    }

    @MainActor
    func testSuggestionKeepsDraftThroughFirstSave() throws {
        let url = try isolatedStore()
        let (model, a, _) = seededModel(at: url)
        model.selectConversation(a)
        model.composerDraft = "A"
        model.newConversation()
        model.composerDraft = "still composing"
        let id = model.activeConversationID
        // Suggestion chips call send directly, without clearComposerDraft.
        model.send("suggestion", alias: "test-model")
        model.stopAndPersist()
        XCTAssertEqual(model.composerDraft, "still composing")
        model.newConversation()
        XCTAssertEqual(model.composerDraft, "")
        model.selectConversation(id)
        XCTAssertEqual(model.composerDraft, "still composing")
        model.send("another suggestion", alias: "test-model")
        model.stopAndPersist()
        XCTAssertEqual(model.composerDraft, "still composing")
        model.selectConversation(a)
        XCTAssertEqual(model.composerDraft, "A")
    }

    @MainActor
    func testDeletingActiveAndInactiveDrafts() throws {
        let url = try isolatedStore()
        let (model, a, b) = seededModel(at: url)
        model.composerDraft = "New"
        model.selectConversation(a)
        model.composerDraft = "A"
        model.selectConversation(b)
        model.composerDraft = "B"
        model.deleteConversation(a)
        XCTAssertEqual(model.composerDraft, "B")
        ConversationDraftStore.flush()
        let draftURL = ConversationDraftStore.companionURL(forConversationStore: url)
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL).text(for: a.uuidString), "")
        model.deleteConversation(b)
        XCTAssertEqual(model.composerDraft, "New")
        ConversationDraftStore.flush()
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL).text(for: b.uuidString), "")
        ConversationStore.flush()
        let restored = ChatViewModel(conversationStoreURL: url)
        XCTAssertTrue(restored.conversations.isEmpty)
        XCTAssertEqual(restored.composerDraft, "New")
    }

    @MainActor
    func testMemoryOnlyNeverWrites() async throws {
        let url = try isolatedStore()
        let model = ChatViewModel(persistsConversations: false, conversationStoreURL: url)
        model.composerDraft = "Memory"
        model.newConversation()
        XCTAssertEqual(model.composerDraft, "Memory")
        model.clearComposerDraft()
        XCTAssertEqual(model.composerDraft, "")
        model.composerDraft = "Again"
        model.stopAndPersist()
        try await Task.sleep(for: .milliseconds(650))
        ConversationDraftStore.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("drafts.json").path))
    }

    @MainActor
    func testDebounceSavesLatestTextAndCancellation() async throws {
        let url = try isolatedStore()
        let model = ChatViewModel(conversationStoreURL: url)
        model.composerDraft = "old"
        await Task.yield() // Start the first sleeper so cancellation exercises its catch.
        model.composerDraft = "latest"
        try await Task.sleep(for: .milliseconds(750))
        ConversationDraftStore.flush()
        let draftURL = ConversationDraftStore.companionURL(forConversationStore: url)
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL).text(for: ConversationDrafts.newChatKey), "latest")
        model.composerDraft = "  \n\t"
        model.stopAndPersist()
        ConversationDraftStore.flush()
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL), ConversationDrafts())
    }

    func testValuePruningPreservesWhitespaceUntilEncoding() throws {
        let a = UUID(), stale = UUID(), missing = UUID()
        var drafts = ConversationDrafts()
        drafts.set(" A \n", for: a.uuidString)
        drafts.set("stale", for: stale.uuidString)
        drafts.set("New", for: ConversationDrafts.newChatKey)
        let pruned = drafts.pruned(keeping: [a, missing])
        XCTAssertEqual(pruned.text(for: a.uuidString), " A \n")
        XCTAssertEqual(pruned.text(for: stale.uuidString), "")
        XCTAssertEqual(pruned.text(for: ConversationDrafts.newChatKey), "New")
        drafts.remove(a)
        drafts.remove(missing)
        drafts.set("", for: stale.uuidString)
        drafts.set(" \n\t", for: ConversationDrafts.newChatKey)
        XCTAssertEqual(drafts.text(for: ConversationDrafts.newChatKey), " \n\t")
        XCTAssertEqual(drafts.pruned(keeping: []).text(for: ConversationDrafts.newChatKey), " \n\t")
        XCTAssertEqual(try JSONDecoder().decode(ConversationDrafts.self, from: JSONEncoder().encode(drafts)), ConversationDrafts())
        drafts.set("", for: ConversationDrafts.newChatKey)
        XCTAssertEqual(drafts, ConversationDrafts())
    }

    @MainActor
    func testWhitespaceComposerInputSurvivesSavingButIsNotPersisted() throws {
        let url = try isolatedStore()
        let (model, a, _) = seededModel(at: url)
        let draftURL = try XCTUnwrap(ConversationDraftStore.companionURL(forConversationStore: url))
        for text in ["  ", "\n"] {
            model.selectConversation(a)
            model.composerDraft = text
            XCTAssertEqual(model.composerDraft, text)
            model.stopAndPersist()
            ConversationDraftStore.flush()
            XCTAssertEqual(model.composerDraft, text)
            model.newConversation()
            model.selectConversation(a)
            XCTAssertEqual(model.composerDraft, text)
            let json = try JSONSerialization.jsonObject(with: Data(contentsOf: draftURL)) as? [String: [String: String]]
            XCTAssertEqual(try XCTUnwrap(json?["entries"]), [:])
        }
    }

    func testStorePathsMissingRoundTripAndPermissions() throws {
        let url = try isolatedStore()
        let draftURL = try XCTUnwrap(ConversationDraftStore.companionURL(forConversationStore: url))
        XCTAssertEqual(draftURL.lastPathComponent, "drafts.json")
        XCTAssertEqual(ConversationDraftStore.fileURL(override: draftURL), draftURL)
        XCTAssertNil(ConversationDraftStore.companionURL(forConversationStore: nil))
        XCTAssertEqual(ConversationDraftStore.fileURL().lastPathComponent, "drafts.json")
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL), ConversationDrafts())
        XCTAssertFalse(FileManager.default.fileExists(atPath: draftURL.path))
        var drafts = ConversationDrafts()
        drafts.set("private", for: UUID().uuidString)
        drafts.set(" \n", for: "empty")
        ConversationDraftStore.save(drafts, to: draftURL)
        ConversationDraftStore.flush()
        drafts.set("", for: "empty")
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL), drafts)
        let attributes = try FileManager.default.attributesOfItem(atPath: draftURL.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        let json = try String(contentsOf: draftURL, encoding: .utf8)
        XCTAssertFalse(json.contains("empty"))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: draftURL.deletingLastPathComponent().path), ["drafts.json"])

        // Replace even a previously permissive file with a fresh owner-only inode.
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: draftURL.path)
        drafts.set("updated", for: ConversationDrafts.newChatKey)
        ConversationDraftStore.save(drafts, to: draftURL)
        ConversationDraftStore.flush()
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL), drafts)
        let overwrittenAttributes = try FileManager.default.attributesOfItem(atPath: draftURL.path)
        XCTAssertEqual((overwrittenAttributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: draftURL.deletingLastPathComponent().path), ["drafts.json"])
    }

    func testSaveFailureWhenParentIsAFile() throws {
        let blocker = try isolatedStore()
        let original = Data("keep this file".utf8)
        try original.write(to: blocker)
        ConversationDraftStore.save(ConversationDrafts(), to: blocker.appendingPathComponent("drafts.json"))
        ConversationDraftStore.flush()
        XCTAssertEqual(try Data(contentsOf: blocker), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: blocker.deletingLastPathComponent().path), ["conversations.json"])
    }

    func testSaveFailureWhenTemporaryFileCannotBeCreated() throws {
        let directory = try isolatedStore().deletingLastPathComponent()
        let fm = FileManager.default
        try fm.setAttributes([.posixPermissions: 0o500], ofItemAtPath: directory.path)
        defer { try? fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path) }
        ConversationDraftStore.save(ConversationDrafts(), to: directory.appendingPathComponent("drafts.json"))
        ConversationDraftStore.flush()
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: directory.path), [])
    }

    func testRenameFailureRemovesTemporaryFile() throws {
        let directory = try isolatedStore().deletingLastPathComponent()
        let destination = directory.appendingPathComponent("drafts.json")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let blocker = destination.appendingPathComponent("keep")
        let original = Data("keep this file".utf8)
        try original.write(to: blocker)
        ConversationDraftStore.save(ConversationDrafts(), to: destination)
        ConversationDraftStore.flush()
        XCTAssertEqual(try Data(contentsOf: blocker), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["drafts.json"])
    }

    func testReadFailureLeavesOriginalPathInPlace() throws {
        let directory = try isolatedStore().deletingLastPathComponent()
        let url = directory.appendingPathComponent("drafts.json")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        XCTAssertEqual(ConversationDraftStore.load(from: url), ConversationDrafts())
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["drafts.json"])
    }

    func testCorruptFileBackup() throws {
        let url = try isolatedStore().deletingLastPathComponent().appendingPathComponent("drafts.json")
        let corrupt = Data("not JSON".utf8)
        try corrupt.write(to: url)
        XCTAssertEqual(ConversationDraftStore.load(from: url), ConversationDrafts())
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        let backups = try FileManager.default.contentsOfDirectory(at: url.deletingLastPathComponent(), includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("drafts.corrupt-") }
        XCTAssertEqual(backups.count, 1)
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(backups.first)), corrupt)
    }

    @MainActor
    func testLoadPrunesStaleAndBlankEntries() throws {
        let url = try isolatedStore()
        let a = conversation("A"), stale = UUID()
        ConversationStore.save([a], to: url)
        ConversationStore.flush()
        let draftURL = try XCTUnwrap(ConversationDraftStore.companionURL(forConversationStore: url))
        let json = ["entries": [a.id.uuidString: "A", stale.uuidString: "stale", "invalid": "bad", ConversationDrafts.newChatKey: " \n"]]
        try JSONEncoder().encode(json).write(to: draftURL)
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL).text(for: ConversationDrafts.newChatKey), "")
        let model = ChatViewModel(conversationStoreURL: url)
        XCTAssertEqual(model.composerDraft, "")
        model.selectConversation(a.id)
        XCTAssertEqual(model.composerDraft, "A")
        model.stopAndPersist()
        ConversationDraftStore.flush()
        var expected = ConversationDrafts()
        expected.set("A", for: a.id.uuidString)
        XCTAssertEqual(ConversationDraftStore.load(from: draftURL), expected)
    }
}
