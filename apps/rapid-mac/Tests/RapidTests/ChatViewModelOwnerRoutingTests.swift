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

}
