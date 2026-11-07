import Foundation
import Testing
@testable import Rapid

@Suite("Owner conversation deletion real transport") @MainActor
struct OwnerDeletionRealTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_DELETE_OWNER_URL"] != nil))
    func nativeDeletionReachesOwner() async throws {
        let endpoint = URL(string: ProcessInfo.processInfo.environment["RAPID_DELETE_OWNER_URL"]!)!
        let client = try URLSessionChatTurnClient(endpoint: endpoint, bearer: "owned-delete-test")
        guard case .object(let created) = try await client.createConversation(workspace: "/private/tmp"),
              case .string(let ownerID) = created["id"] else { Issue.record("Missing owned conversation"); return }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("native-delete-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = directory.appendingPathComponent("conversations.json")
        let row = ChatConversation(id: UUID(), title: "Owned deletion fixture", messages: [ChatMessage(role: .user, content: "Fixture")], createdAt: Date(), updatedAt: Date(), backend: .ownerV1, ownerConversationID: ownerID, ownerStoreUUID: try await client.ownerStoreUUID())
        try JSONEncoder().encode([row]).write(to: store)
        let model = ChatViewModel(ownerClientFactory: { client }, conversationStoreURL: store)
        model.deleteConversation(row.id)
        #expect(model.conversations.isEmpty)
        // This is the actual native delete function and actual authenticated owner HTTP,
        // without inference or user history. Before the fix, the owner remains readable.
        var removed = false
        for _ in 0..<50 {
            do { _ = try await client.conversation(id: ownerID) }
            catch ChatTurnClientError.http(status: 404) { removed = true; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        #expect(removed, "Native deletion must revoke owner conversation access")
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["RAPID_DELETE_OWNER_URL"] != nil))
    func offlineNativeDeletionResumesAfterRelaunch() async throws {
        let endpoint = URL(string: ProcessInfo.processInfo.environment["RAPID_DELETE_OWNER_URL"]!)!
        let client = try URLSessionChatTurnClient(endpoint: endpoint, bearer: "owned-delete-test")
        guard case .object(let created) = try await client.createConversation(workspace: "/private/tmp"), case .string(let ownerID) = created["id"] else { throw ChatTurnClientError.invalidResponse }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("native-delete-offline-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("conversations.json")
        let row = ChatConversation(id: UUID(), title: "Offline fixture", messages: [ChatMessage(role: .user, content: "Offline")], createdAt: Date(), updatedAt: Date(), backend: .ownerV1, ownerConversationID: ownerID, ownerStoreUUID: try await client.ownerStoreUUID())
        try JSONEncoder().encode([row]).write(to: file)
        let offline = ChatViewModel(ownerClientFactory: { throw URLError(.notConnectedToInternet) }, conversationStoreURL: file)
        offline.deleteConversation(row.id)
        await offline.ownerDeletions.retry()
        #expect(offline.conversations.isEmpty)
        #expect(offline.ownerDeletions.intents.count == 1)
        _ = try await client.conversation(id: ownerID)
        let relaunched = ChatViewModel(ownerClientFactory: { client }, conversationStoreURL: file)
        for _ in 0..<100 { if relaunched.ownerDeletions.intents.isEmpty { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(relaunched.ownerDeletions.intents.isEmpty)
        #expect(relaunched.conversations.isEmpty)
        await #expect(throws: ChatTurnClientError.http(status: 404)) { _ = try await client.conversation(id: ownerID) }
    }

}

private actor DeletionGate {
    var entered = false
    var continuation: CheckedContinuation<Void, Never>?
    func wait() async { entered = true; await withCheckedContinuation { continuation = $0 } }
    func release() { continuation?.resume(); continuation = nil }
}
private actor AuthorityRecorder {
    var ids: [(String, UUID?, String?)] = []
    func record(_ id: String, _ store: UUID?, _ turn: String?) { ids.append((id, store, turn)) }
}
private actor DeletionFixtureClient: ChatTurnClient {
    let store: UUID
    var replacement: UUID?
    var calls: [String] = []
    var failure: Int?
    var running = true
    var cancelledStillRunning = false
    var foreignTurn = false
    var missingTurn = false
    var createGate: DeletionGate?
    var deleteGate: DeletionGate?
    var identityGate: DeletionGate?
    var deletedIDs: [String] = []
    init(store: UUID = UUID()) { self.store = store }
    func setMissingTurn() { missingTurn = true }
    func gates(create: DeletionGate? = nil, delete: DeletionGate? = nil, identity: DeletionGate? = nil) { createGate = create; deleteGate = delete; identityGate = identity }
    func createConversation(workspace: String) async throws -> ChatWireJSON {
        calls.append("create")
        await createGate?.wait()
        return .object(["id": .string("owned"), "branch_id": .string("branch"), "branches": .array([.object(["id": .string("branch"), "revision": .integer(0)])])])
    }
    func configure(failure: Int? = nil, replacement: UUID? = nil, cancelledStillRunning: Bool = false, foreignTurn: Bool = false) {
        self.failure = failure; self.replacement = replacement
        self.cancelledStillRunning = cancelledStillRunning; self.foreignTurn = foreignTurn
    }
    func ownerStoreUUID() async throws -> UUID? {
        calls.append("identity")
        if let gate = identityGate { identityGate = nil; await gate.wait() }
        if calls.filter({ $0 == "identity" }).count > 1, let replacement { return replacement }
        return store
    }
    func conversation(id: String) async throws -> ChatWireJSON { calls.append("inspect"); return .object(["id": .string(id), "title": .string("Owned fixture")]) }
    func deleteConversation(id: String) async throws { calls.append("delete"); deletedIDs.append(id); await deleteGate?.wait(); if let failure { throw ChatTurnClientError.http(status: failure) } }
    func turn(turnID: String) async throws -> ChatWireJSON { calls.append("turn"); if missingTurn { throw ChatTurnClientError.http(status: 404) }; return receipt() }
    func cancel(turnID: String) async throws -> ChatWireJSON { calls.append("cancel"); running = cancelledStillRunning; return receipt() }
    private func receipt() -> ChatWireJSON { .object(["conversation_id": .string(foreignTurn ? "foreign" : "owned"), "status": .string(running ? "running" : "cancelled")]) }
    func submit(_ request: ChatTurnRequest, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws { calls.append("submit"); throw ChatTurnClientError.invalidResponse }
    func stream(turnID: String, after: Int, onEvent: @escaping ChatTurnEventHandler) async throws { throw ChatTurnClientError.invalidResponse }
    func uploadAttachment(conversationID: String, name: String, data: Data) async throws -> String { throw ChatTurnClientError.invalidResponse }
    func respond(_ response: ChatPermissionResponse) async throws -> ChatWireJSON { throw ChatTurnClientError.invalidResponse }
    func operation(opID: String) async throws -> ChatWireJSON { throw ChatTurnClientError.invalidResponse }
    func context(_ request: ChatTurnRequest) async throws -> ChatWireContextEvent { throw ChatTurnClientError.invalidResponse }
    func compact(_ request: ChatTurnRequest, instructions: String) async throws -> ChatCompactResponse { throw ChatTurnClientError.invalidResponse }
    func undo(conversationID: String, operationID: String) async throws -> ChatWireJSON { throw ChatTurnClientError.invalidResponse }
}

@Suite("Durable owner deletion") @MainActor
struct OwnerDeletionTests {
    func intent(_ store: UUID?, turn: String? = "turn") -> OwnerDeletionIntent { OwnerDeletionIntent(id: UUID(), conversationID: "owned", storeUUID: store, turnID: turn) }

    @Test func cancelTerminalThenDeleteAndNoCredentialsPersisted() async throws {
        let client = DeletionFixtureClient()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("pending.json")
        let queue = OwnerConversationDeletion(url: file, clientProvider: { client })
        try queue.enqueue(intent(client.store))
        let data = try Data(contentsOf: file)
        #expect(String(decoding: data, as: UTF8.self).contains("storeUUID"))
        #expect(!String(decoding: data, as: UTF8.self).contains("bearer"))
        let permissions = try FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? Int
        #expect(permissions == 0o600)
        await queue.retry()
        #expect(await client.calls == ["identity", "turn", "cancel", "identity", "delete"])
        #expect(queue.intents.isEmpty)
        #expect(try JSONDecoder().decode([OwnerDeletionIntent].self, from: Data(contentsOf: file)).isEmpty)
    }

    @Test(arguments: [401, 403, 409, 500]) func ownerFailureRemainsDurable(status: Int) async throws {
        let client = DeletionFixtureClient(); await client.configure(failure: status)
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(client.store, turn: nil))
        await queue.retry()
        #expect(queue.intents.count == 1)
        #expect(queue.notice != nil)
    }

    @Test func alreadyAbsentCompletesOnlyAfterIdentityValidation() async throws {
        let client = DeletionFixtureClient(); await client.configure(failure: 404)
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(client.store, turn: nil)); await queue.retry()
        #expect(queue.intents.isEmpty)
        #expect(await client.calls == ["identity", "identity", "delete"])
    }

    @Test(arguments: [true, false]) func foreignStoreCannotCancelOrDelete(midflight: Bool) async throws {
        let client = DeletionFixtureClient()
        if midflight { await client.configure(replacement: UUID()) }
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(midflight ? client.store : UUID())); await queue.retry()
        #expect(queue.intents.count == 1)
        #expect(!(await client.calls).contains("delete"))
        if !midflight { #expect(!(await client.calls).contains("cancel")) }
    }

    @Test(arguments: [true, false]) func unprovenCancellationCannotDelete(foreign: Bool) async throws {
        let client = DeletionFixtureClient(); await client.configure(cancelledStillRunning: !foreign, foreignTurn: foreign)
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(client.store)); await queue.retry()
        #expect(queue.intents.count == 1)
        #expect(!(await client.calls).contains("delete"))
        if foreign { #expect(!(await client.calls).contains("cancel")) }
    }

    @Test func offlineIntentSurvivesRelaunchAndEpochIndependentStoreBinding() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("pending.json")
        let client = DeletionFixtureClient()
        let offline = OwnerConversationDeletion(url: file, clientProvider: { throw URLError(.notConnectedToInternet) })
        let value = intent(client.store, turn: nil)
        try offline.enqueue(value); await offline.retry()
        #expect(offline.intents == [value])
        // A new owner client (new process/epoch/bearer) with the same stable store can finish.
        let restored = OwnerConversationDeletion(url: file, clientProvider: { client })
        #expect(restored.intents == [value]); await restored.retry()
        #expect(restored.intents.isEmpty)
    }

    @Test func legacyIdentityRequiresInspectedExplicitConfirmation() async throws {
        let client = DeletionFixtureClient()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(nil, turn: nil)); await queue.retry()
        #expect(await client.calls == [])
        await queue.inspectLegacy()
        let confirmation = try #require(queue.confirmation)
        #expect(confirmation.title == "Owned fixture")
        #expect(confirmation.storeUUID == client.store)
        #expect(queue.intents.first?.storeUUID == nil)
        await queue.confirmLegacy(confirmation)
        #expect(queue.intents.isEmpty)
    }

    @Test func unknownCreationStaysPendingAndLateCaptureDoesNotRestoreHistory() async throws {
        let client = DeletionFixtureClient()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        let value = OwnerDeletionIntent(id: UUID(), conversationID: nil, storeUUID: nil, turnID: nil)
        try queue.enqueue(value); await queue.retry()
        #expect(queue.intents == [value]); #expect(await client.calls == [])
        try queue.capture(localID: value.id, conversationID: "owned", storeUUID: client.store, turnID: "turn")
        await queue.retry(); #expect(queue.intents.isEmpty)
    }

    @Test func unreadableIntentCannotBeOverwrittenOrLoseNativeRow() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = directory.appendingPathComponent("conversations.json")
        let pending = OwnerConversationDeletion.companionURL(for: store)
        try Data("broken".utf8).write(to: pending)
        let row = ChatConversation(id: UUID(), title: "Keep", messages: [ChatMessage(role: .user, content: "Keep")], createdAt: Date(), updatedAt: Date(), backend: .ownerV1, ownerConversationID: "owned")
        try JSONEncoder().encode([row]).write(to: store)
        let model = ChatViewModel(conversationStoreURL: store)
        model.deleteConversation(row.id)
        #expect(model.conversations.count == 1)
        #expect(model.lastError != nil)
        #expect(try Data(contentsOf: pending) == Data("broken".utf8))
    }

    @Test func localCommitFailureCannotCompleteAuthorityDeletion() async throws {
        let client = DeletionFixtureClient()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        queue.commitLocalRemoval = { throw URLError(.cannotWriteToFile) }
        try queue.enqueue(intent(client.store)); await queue.retry()
        #expect(queue.intents.count == 1)
        #expect(await client.calls == [])
    }

    @Test func crashBetweenIntentAndLocalRemovalCannotResurrectRow() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = directory.appendingPathComponent("conversations.json")
        let client = DeletionFixtureClient()
        let row = ChatConversation(id: UUID(), title: "Deleted", messages: [ChatMessage(role: .user, content: "Old")], createdAt: Date(), updatedAt: Date(), backend: .ownerV1, ownerConversationID: "owned", ownerStoreUUID: client.store)
        try JSONEncoder().encode([row]).write(to: store)
        let pending = OwnerConversationDeletion(url: OwnerConversationDeletion.companionURL(for: store), clientProvider: { client })
        try pending.enqueue(OwnerDeletionIntent(id: row.id, conversationID: "owned", storeUUID: client.store, turnID: nil))
        let model = ChatViewModel(ownerClientFactory: { client }, conversationStoreURL: store)
        #expect(model.conversations.isEmpty)
        for _ in 0..<50 { if model.ownerDeletions.intents.isEmpty { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(model.ownerDeletions.intents.isEmpty)
        #expect(ConversationStore.load(from: store).isEmpty)
    }

    @Test func lateCaptureDuringDeleteCannotEraseNewerIntent() async throws {
        let client = DeletionFixtureClient(), gate = DeletionGate()
        await client.gates(delete: gate)
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        let value = intent(client.store, turn: nil)
        try queue.enqueue(value)
        let retry = Task { await queue.retry() }
        while !(await gate.entered) { await Task.yield() }
        try queue.capture(localID: value.id, conversationID: "owned", storeUUID: client.store, turnID: "late-turn")
        await gate.release(); await retry.value
        #expect(queue.intents.first?.turnID == "late-turn")
        #expect(queue.intents.count == 1)
    }

    @Test func stopDuringCreateStillCapturesAuthorityWithoutSubmitting() async throws {
        let client = DeletionFixtureClient(), gate = DeletionGate(), recorder = AuthorityRecorder()
        await client.gates(create: gate)
        let session = OwnerChatSession(client: client)
        let submission = Task {
            try await session.submit(text: "Question", target: .init(url: URL(string: "http://127.0.0.1:1234")!, bearer: nil, model: "fixture"), onAuthority: { id, store, turn in await recorder.record(id, store, turn) })
        }
        while !(await gate.entered) { await Task.yield() }
        try await session.stop()
        await gate.release()
        await #expect(throws: CancellationError.self) { try await submission.value }
        #expect(await recorder.ids.count == 1)
        #expect(await recorder.ids.first?.0 == "owned")
        #expect(await recorder.ids.first?.1 == client.store)
        #expect(await recorder.ids.first?.2 == nil)
        #expect(!(await client.calls).contains("submit"))
        await session.close()
    }

    @Test func newIntentDuringCommitMustWaitForItsOwnBarrier() async throws {
        let client = DeletionFixtureClient(), gate = DeletionGate()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        let first = intent(client.store, turn: nil)
        try queue.enqueue(first)
        queue.commitLocalRemoval = { await gate.wait() }
        let retry = Task { await queue.retry() }
        while !(await gate.entered) { await Task.yield() }
        let second = OwnerDeletionIntent(id: UUID(), conversationID: "second", storeUUID: client.store, turnID: nil)
        try queue.enqueue(second)
        await gate.release(); await retry.value
        #expect(await client.deletedIDs == ["owned"])
        #expect(queue.intents == [second])
    }

    @Test func legacyConfirmationCannotBindAnotherIntentDuringSuspension() async throws {
        let client = DeletionFixtureClient(), gate = DeletionGate()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(client.store, turn: nil))
        let legacy = OwnerDeletionIntent(id: UUID(), conversationID: "legacy", storeUUID: nil, turnID: nil)
        try queue.enqueue(legacy)
        await queue.inspectLegacy()
        let confirmation = try #require(queue.confirmation)
        await client.gates(identity: gate)
        let confirm = Task { await queue.confirmLegacy(confirmation) }
        while !(await gate.entered) { await Task.yield() }
        await queue.retry()
        let other = OwnerDeletionIntent(id: UUID(), conversationID: "unconfirmed", storeUUID: nil, turnID: nil)
        try queue.enqueue(other)
        await gate.release(); await confirm.value
        #expect(!(await client.deletedIDs).contains("unconfirmed"))
        #expect(queue.intents.contains(other))
    }

    @Test func blockedDeletionMustNotDelayUnrelatedChatAdmission() async throws {
        let client = DeletionFixtureClient(), gate = DeletionGate()
        await client.gates(delete: gate)
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.harnessEnabled = true
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        await Task.yield() // Let the empty startup pass finish before adding work.
        try model.ownerDeletions.enqueue(intent(client.store, turn: nil))
        model.send("New unrelated question", alias: "fixture")
        while !(await gate.entered) { await Task.yield() }
        for _ in 0..<50 { if (await client.calls).contains("create") { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect((await client.calls).contains("create"), "Deletion network delay must not block chat admission")
        await gate.release()
        await model._testingWaitForCurrentTurn()
    }

    @Test func ordinarySendMustNotPromoteLegacyDeletionAuthority() async throws {
        let client = DeletionFixtureClient()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("conversations.json")
        let row = ChatConversation(id: UUID(), title: "Legacy", messages: [ChatMessage(role: .user, content: "Old")], createdAt: Date(), updatedAt: Date(), backend: .ownerV1, ownerConversationID: "legacy")
        try JSONEncoder().encode([row]).write(to: file)
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.harnessEnabled = true
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, conversationStoreURL: file)
        model.selectConversation(row.id)
        model.send("New question", alias: "fixture")
        await model._testingWaitForCurrentTurn()
        #expect((await client.calls).contains("identity"))
        #expect(model.conversations.first?.ownerStoreUUID == nil)
        #expect(ConversationStore.load(from: file).first?.ownerStoreUUID == nil)
    }

    @Test func missingKnownTurnStillDefersToAuthoritativeDelete() async throws {
        let client = DeletionFixtureClient(); await client.setMissingTurn()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        try queue.enqueue(intent(client.store)); await queue.retry()
        #expect(queue.intents.isEmpty)
        #expect(await client.calls == ["identity", "turn", "identity", "delete"])
    }

    @Test func scheduledNewIntentGetsSeparateCommitWithoutUnboundedRetry() async throws {
        let client = DeletionFixtureClient(), gate = DeletionGate()
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { client })
        var commits = 0
        queue.commitLocalRemoval = { commits += 1; if commits == 1 { await gate.wait() } }
        try queue.enqueue(intent(client.store, turn: nil)); queue.scheduleRetry()
        while !(await gate.entered) { await Task.yield() }
        let second = OwnerDeletionIntent(id: UUID(), conversationID: "second", storeUUID: client.store, turnID: nil)
        try queue.enqueue(second); queue.scheduleRetry()
        await gate.release()
        for _ in 0..<100 { if queue.intents.isEmpty { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(queue.intents.isEmpty)
        #expect(commits == 2)
        #expect(await client.deletedIDs == ["owned", "second"])
    }

    @Test func deletionRetryCannotLaunchMissingOwner() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        await #expect(throws: ChatOwnerConnectionError.ownerUnavailable) {
            _ = try await ChatOwnerConnection.existingDeletionClient(environment: ["RAPID_MLX_CHAT_HOME": root.path])
        }
        #expect(!FileManager.default.fileExists(atPath: root.path))
    }

    @Test func lateLegacyCaptureDoesNotPromoteDeletionAuthority() throws {
        let queue = OwnerConversationDeletion(url: nil, clientProvider: { throw URLError(.notConnectedToInternet) })
        let value = intent(nil, turn: nil)
        try queue.enqueue(value)
        try queue.capture(localID: value.id, conversationID: "owned", storeUUID: UUID(), turnID: "new-turn")
        #expect(queue.intents.first?.storeUUID == nil)
        #expect(queue.intents.first?.turnID == "new-turn")
    }

    @Test func oldHistoryDecodesWithoutAuthorityIdentity() throws {
        let row = ChatConversation(id: UUID(), title: "Old", messages: [], createdAt: Date(), updatedAt: Date())
        let data = try JSONEncoder().encode(row)
        let restored = try JSONDecoder().decode(ChatConversation.self, from: data)
        #expect(restored.ownerStoreUUID == nil && restored.ownerTurnID == nil)
    }
}


@Suite("Owner deletion native status") @MainActor
struct OwnerDeletionInteractionTests {
    @Test(arguments: [520.0, 720.0]) func pendingDeletionCanInspectAndCancelConfirmation(width: Double) async throws {
        let client = DeletionFixtureClient()
        let model = ChatViewModel(ownerClientFactory: { client }, persistsConversations: false)
        try model.ownerDeletions.enqueue(OwnerDeletionIntent(id: UUID(), conversationID: "owned", storeUUID: nil, turnID: nil))
        let server = ServerManager(testingState: .idle)
        let stage = GoldenStage(
            ChatView(viewModel: model, server: server, alias: .constant("fixture"), readiness: .noModel)
                .environment(DownloadManager()).environment(QuickstartCoordinator()).environment(server),
            size: CGSize(width: width, height: 620)
        )
        try await stage.waitForIdentifier("Chat.OwnerDeletion.Pending")
        #expect(stage.treeText().contains("Server deletion is pending"))
        if let path = ProcessInfo.processInfo.environment["RAPID_DELETE_REVIEW_DIR"] {
            try stage.writeScreenshot(to: URL(fileURLWithPath: path).appendingPathComponent("pending-\(Int(width)).png"))
        }
        try await stage.waitForIdentifier("Chat.OwnerDeletion.Review")
        try stage.press("Chat.OwnerDeletion.Review")
        try await stage.waitForText("Delete from this chat server?")
        #expect(stage.treeText().contains("Owned fixture"))
        #expect(stage.treeText().contains(client.store.uuidString))
        stage.pressEscape()
        #expect(model.ownerDeletions.intents.count == 1)
        #expect(!(await client.calls).contains("delete"))
    }
}
