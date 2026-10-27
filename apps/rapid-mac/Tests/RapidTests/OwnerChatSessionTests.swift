import Foundation
import Testing
@testable import Rapid

@Suite("Owner chat session")
struct OwnerChatSessionTests {
    func fake(_ frames: [Data]) async throws -> FakeChatTurnClient {
        let context = try JSONDecoder().decode(ChatWireContextEvent.self, from: Data(#"{"used_tokens":0,"limit_tokens":512,"system_tools_tokens":0,"conversation_tokens":0,"tool_output_tokens":0,"free_tokens":512,"percent":0}"#.utf8))
        let fake = FakeChatTurnClient(frames: frames, context: context)
        await fake.enableTurnSubstitution()
        return fake
    }
    func frame(_ event: String, _ seq: Int) -> Data {
        let type = (try! JSONSerialization.jsonObject(with: Data(event.utf8)) as! [String: Any])["type"] as! String
        return Data("id: turn-1:\(seq)\nevent: \(type)\ndata: {\"turn_id\":\"turn-1\",\"seq\":\(seq),\"event\":\(event)}\n\n".utf8)
    }
    func start(_ fake: FakeChatTurnClient) async throws -> OwnerChatSession {
        let session = OwnerChatSession(client: fake)
        try await session.submit(text: "Hello", target: .init(url: URL(string: "http://localhost:8000/v1")!, bearer: "key", model: "m"))
        return session
    }
    func wait(_ session: OwnerChatSession, _ predicate: (OwnerChatSession.Snapshot) -> Bool) async throws {
        for _ in 0..<100 { if predicate(await session.snapshot()) { return }; try await Task.sleep(for: .milliseconds(20)) }
        Issue.record("Timed out waiting for session")
    }
    @Test func textAndReplay() async throws {
        let frames = [frame(#"{"type":"text.delta","part_id":"p","text":"Hi"}"#, 1), frame(#"{"type":"tool.started","call_id":"x","name":"search","args":{"q":"hello"}}"#, 2), frame(#"{"type":"tool.result","call_id":"x","result":"found"}"#, 3), frame(#"{"type":"turn.terminal","outcome":"completed"}"#, 4)]
        let client = try await fake([frames[0], frames[1], frames[1], frames[2], frames[3]])
        let session = try await start(client)
        try await wait(session) { $0.state == .finished }
        let projection = try #require(await session.snapshot().projection)
        let rows = ownerProjectionMessages(projection, assistantID: UUID(), createdAt: Date())
        #expect(rows[0].content == "Hi")
        #expect(rows[0].toolCalls?.count == 1)
        #expect(rows.filter { $0.toolCallID == "x" }.count == 1)
    }
    @Test func disconnectAndStop() async throws {
        let client = try await fake([frame(#"{"type":"text.delta","part_id":"p","text":"Hi"}"#, 1)])
        let session = try await start(client)
        try await wait(session) { $0.state == .reconnecting }
        try await Task.sleep(for: .milliseconds(400))
        let calls = await client.calls
        #expect(calls.contains { if case .stream(_, after: 1) = $0 { true } else { false } })
        #expect(!calls.contains { if case .cancel = $0 { true } else { false } })
        try await session.stop()
        #expect(await client.calls.contains { if case .cancel = $0 { true } else { false } })
    }
    @Test func interrupted() async throws {
        let client = try await fake([frame(#"{"type":"turn.terminal","outcome":"error","error_code":"owner_restarted"}"#, 1)])
        let session = try await start(client)
        try await wait(session) { $0.state == .finished }
        let projection = try #require(await session.snapshot().projection)
        let row = ownerProjectionMessages(projection, assistantID: UUID(), createdAt: Date())[0]
        #expect(row.status == .failed)
        #expect(row.errorMessage?.contains("Interrupted") == true)
    }
    @Test func stalePermission() async throws {
        let client = try await fake([frame(#"{"type":"permission.request","op_id":"op","revision":1,"tool":"write_file","summary":"Edit","preview":"x","risk":"file","reason":"Approval","options":["once","deny"],"expires_at":2000000000}"#, 1)])
        await client.setReceipt(.object(["id": .string("op"), "state": .string("intent"), "revision": .integer(2), "expires_at": .integer(2000000000)]))
        let session = try await start(client)
        try await wait(session) { $0.projection?.pendingPermission != nil }
        do { try await session.resolve(opID: "op", decision: .allow); Issue.record("stale approval accepted") } catch {}
        #expect(await session.snapshot().unavailablePermissions.contains("op"))
        #expect(!(await client.calls).contains { if case .respond = $0 { true } else { false } })
    }
}
