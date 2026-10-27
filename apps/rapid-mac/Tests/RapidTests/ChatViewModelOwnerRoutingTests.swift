import Foundation
import Testing
@testable import Rapid

@Suite("Chat view model owner routing") @MainActor
struct ChatViewModelOwnerRoutingTests {
    func config() -> ChatBackendConfig {
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        return ChatBackendConfig(defaults: defaults)
    }
    func fake() async throws -> FakeChatTurnClient {
        let context = try JSONDecoder().decode(ChatWireContextEvent.self, from: Data(#"{"used_tokens":0,"limit_tokens":512,"system_tools_tokens":0,"conversation_tokens":0,"tool_output_tokens":0,"free_tokens":512,"percent":0}"#.utf8))
        let frame = Data("id: turn-1:1\nevent: turn.terminal\ndata: {\"turn_id\":\"turn-1\",\"seq\":1,\"event\":{\"type\":\"turn.terminal\",\"outcome\":\"completed\"}}\n\n".utf8)
        let client = FakeChatTurnClient(frames: [frame], context: context)
        await client.enableTurnSubstitution()
        return client
    }
    @Test func ownerNewConversation() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("hello", alias: "model")
        await model._testingWaitForCurrentTurn()
        #expect(model.conversations.first?.backend == .ownerV1)
        #expect(model.conversations.first?.ownerConversationID == "c")
        #expect(await client.calls.contains { if case .submit = $0 { true } else { false } })
        #expect(!model.isStreaming)
    }
    @Test func unavailableIsVisible() async {
        let config = config(); config.chatBackend = .ownerV1
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { throw ChatOwnerConnectionError.ownerUnavailable }, persistsConversations: false)
        model.send("hello", alias: "model")
        await model._testingWaitForCurrentTurn()
        #expect(model.messages.last?.errorMessage == ChatOwnerConnectionError.ownerUnavailable.localizedDescription)
        #expect(!model.isStreaming)
    }
    @Test func existingLegacyRemainsLegacyAfterToggle() async throws {
        let config = config()
        let client = try await fake()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("first", alias: "model")
        await model._testingWaitForCurrentTurn()
        #expect(model.conversations.first?.backend == nil)
        config.chatBackend = .ownerV1
        model.send("second", alias: "model")
        await model._testingWaitForCurrentTurn()
        #expect(model.conversations.first?.backend == nil)
        #expect(await client.calls.isEmpty)
    }
    @Test func discardsSessionsOnSwitchAndToggle() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("one", alias: "model"); await model._testingWaitForCurrentTurn()
        #expect(model._testingOwnerSessionCount == 1)
        model.newConversation()
        #expect(model._testingOwnerSessionCount == 0)
        model.send("two", alias: "model"); await model._testingWaitForCurrentTurn()
        #expect(model._testingOwnerSessionCount == 1)
        config.harnessEnabled = false
        #expect(model._testingOwnerSessionCount == 0)
    }
    @Test func failedStopRemainsAttached() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake(); await client.setFrames([]); await client.failCancel()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("hello", alias: "model")
        for _ in 0..<100 {
            if await client.calls.contains(where: { if case .submit = $0 { true } else { false } }) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        model.stop()
        for _ in 0..<100 {
            if model.lastError != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(model.lastError == "Couldn't confirm the stop — the harness may still be running this turn.")
        #expect(await client.calls.filter { if case .cancel = $0 { true } else { false } }.count == 3)
        #expect(model.isStreaming)
    }
    @Test func ownerBranchNavigationDisabled() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("hello", alias: "model"); await model._testingWaitForCurrentTurn()
        let user = try #require(model.messages.first)
        let count = model.messages.count
        #expect(!model.editUserMessage(id: user.id, newContent: "changed", alias: "model"))
        #expect(!model.selectBranch(at: 0, forSiblingOf: user.id))
        model.regenerateLast(alias: "model")
        #expect(model.messages.count == count)
    }
    @Test func stopSendsCancel() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake(); await client.setFrames([])
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("hello", alias: "model")
        for _ in 0..<100 {
            if await client.calls.contains(where: { if case .submit = $0 { true } else { false } }) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        model.stop()
        for _ in 0..<100 {
            if await client.calls.contains(where: { if case .cancel = $0 { true } else { false } }) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(await client.calls.contains { if case .cancel = $0 { true } else { false } })
    }

    @Test func terminationCancelsOwnerTurn() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake(); await client.setFrames([])
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("hello", alias: "model")
        for _ in 0..<100 {
            if await client.calls.contains(where: { if case .submit = $0 { true } else { false } }) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        model.stopAndPersist()
        for _ in 0..<100 {
            if await client.calls.contains(where: { if case .cancel = $0 { true } else { false } }) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(await client.calls.contains { if case .cancel = $0 { true } else { false } })
    }
    @Test func navigationAndDisableCancelOwnerTurn() async throws {
        for disable in [false, true] {
            let config = config(); config.chatBackend = .ownerV1
            let client = try await fake(); await client.setFrames([])
            let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
            model.send("hello", alias: "model")
            for _ in 0..<100 {
                if await client.calls.contains(where: { if case .submit = $0 { true } else { false } }) { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            if disable { config.chatBackend = .legacy } else { model.newConversation() }
            for _ in 0..<100 {
                if await client.calls.contains(where: { if case .cancel = $0 { true } else { false } }) { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(await client.calls.contains { if case .cancel = $0 { true } else { false } })
        }
    }
    @Test func stopDuringConnectionPreparationDoesNotSubmit() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake()
        let gate = OwnerPreparationGate()
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { await gate.wait(); return client }, persistsConversations: false)
        model.send("hello", alias: "model")
        for _ in 0..<100 {
            if await gate.started { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        model.stop()
        await gate.release()
        try await Task.sleep(for: .milliseconds(100))
        #expect(!(await client.calls.contains { if case .submit = $0 { true } else { false } }))
    }
    @Test func approvalErrorsRemainRetryable() async throws {
        let config = config(); config.chatBackend = .ownerV1
        let client = try await fake()
        let frame = Data("id: turn-1:2\nevent: permission.request\ndata: {\"turn_id\":\"turn-1\",\"seq\":2,\"event\":{\"type\":\"permission.request\",\"op_id\":\"op\",\"revision\":1,\"tool\":\"write_file\",\"summary\":\"Edit\",\"preview\":\"x\",\"risk\":\"file\",\"reason\":\"Approval\",\"options\":[\"once\",\"deny\"],\"expires_at\":2000000000}}\n\n".utf8)
        let tool = Data("id: turn-1:1\nevent: tool.started\ndata: {\"turn_id\":\"turn-1\",\"seq\":1,\"event\":{\"type\":\"tool.started\",\"call_id\":\"op\",\"name\":\"write_file\",\"args\":{\"op_id\":\"op\"}}}\n\n".utf8)
        await client.setFrames([tool, frame])
        await client.setReceipt(.object(["id": .string("op"), "state": .string("intent"), "revision": .integer(1), "expires_at": .integer(2000000000)]))
        let model = ChatViewModel(chatBackendConfig: config, ownerClientFactory: { client }, persistsConversations: false)
        model.send("hello", alias: "model")
        for _ in 0..<100 {
            if model.ownerApprovals["op"] != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        await client.setOperationError(.beforeHeaders)
        model.resolveOwnerPermission(opID: "op", allow: true)
        for _ in 0..<100 { if model.lastError != nil { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(model.lastError == "Couldn't send your approval — try again.")
        #expect(model.ownerApprovals["op"]?.unavailable == false)
        await client.setOperationError(nil); await client.setResponseError(.beforeHeaders)
        model.resolveOwnerPermission(opID: "op", allow: false)
        for _ in 0..<100 { if model.lastError != nil { break }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(model.lastError == "Couldn't send your approval — try again.")
        #expect(model.ownerApprovals["op"]?.unavailable == false)
    }

}

private actor OwnerPreparationGate {
    var started = false
    var continuation: CheckedContinuation<Void, Never>?
    func wait() async { started = true; await withCheckedContinuation { continuation = $0 } }
    func release() { continuation?.resume(); continuation = nil }
}
