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
    @Test func stopDuringSessionPreparationDoesNotSubmit() async throws {
        actor Gate {
            var started = false
            var continuation: CheckedContinuation<Void, Never>?
            func wait() async { started = true; await withCheckedContinuation { continuation = $0 } }
            func release() { continuation?.resume(); continuation = nil }
        }
        let gate = Gate()
        let client = try await fake([])
        let session = OwnerChatSession(clientProvider: { await gate.wait(); return client })
        let submission = Task { try await session.submit(text: "x", target: .init(url: URL(string: "http://localhost")!, bearer: nil, model: "m")) }
        for _ in 0..<100 { if await gate.started { break }; try await Task.sleep(for: .milliseconds(10)) }
        try await session.stop()
        await gate.release()
        do { try await submission.value; Issue.record("Expected cancellation") } catch is CancellationError {}
        #expect(!(await client.calls.contains { if case .submit = $0 { true } else { false } }))
    }
    @Test func retriesUnadmittedPost() async throws {
        let client = try await fake([frame(#"{"type":"turn.terminal","outcome":"completed"}"#, 1)])
        await client.failSubmissions(1)
        let session = try await start(client)
        try await wait(session) { $0.state == .finished }
        let calls = await client.calls
        guard case .submit(let first, _) = calls[0], case .submit(let second, _) = calls[1] else { Issue.record("Expected two POSTs"); return }
        #expect(first == second)
    }
    @Test func persistsIDBeforeBranchParsing() async throws {
        let client = try await fake([]); await client.setMalformedConversation()
        let session = OwnerChatSession(client: client)
        actor Capture { var id: String?; func set(_ value: String) { id = value } }
        let capture = Capture()
        do { try await session.submit(text: "x", target: .init(url: URL(string: "http://localhost")!, bearer: nil, model: "m"), onConversation: { await capture.set($0) }); Issue.record("Expected invalid response") } catch {}
        #expect(await capture.id == "c")
        #expect(await session.ownerConversationID() == "c")
    }
    @Test func textAndReplay() async throws {
        let frames = [frame(#"{"type":"text.delta","part_id":"p","text":"Hi"}"#, 1), frame(#"{"type":"tool.started","call_id":"x","name":"search","args":{"q":"hello"}}"#, 2), frame(#"{"type":"tool.result","call_id":"x","result":"found"}"#, 3), frame(#"{"type":"turn.terminal","outcome":"completed"}"#, 4)]
        let client = try await fake([frames[0], frames[1], frames[1], frames[2], frames[3]])
        let session = try await start(client)
        try await wait(session) { $0.state == .finished }
        let projection = try #require(await session.snapshot().projection)
        let rows = ownerProjectionMessages(projection, assistantID: UUID(), createdAt: Date())
        #expect(rows[0].content == "Hi")
        #expect(rows[1].toolCalls?.count == 1)
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
    @Test func permanentFailureAndReconnectLimit() async throws {
        let rejected = try await fake([])
        await rejected.setDeliveryError(.http(status: 409))
        let first = try await start(rejected)
        try await wait(first) { if case .failed = $0.state { true } else { false } }
        #expect(await rejected.calls.count == 2)
        #expect(await rejected.calls.contains { if case .cancel = $0 { true } else { false } })
        let disconnected = try await fake([])
        await disconnected.setDeliveryError(.disconnected(lastEventID: ""))
        let second = try await start(disconnected)
        try await wait(second) { if case .failed = $0.state { true } else { false } }
        #expect(await disconnected.calls.count == 5)
        #expect(await disconnected.calls.contains { if case .cancel = $0 { true } else { false } })
    }
    @Test func reconnectReacquiresClient() async throws {
        let client = try await fake([])
        await client.setDeliveryError(.http(status: 401))
        actor Provider {
            var count = 0
            let client: FakeChatTurnClient
            init(_ client: FakeChatTurnClient) { self.client = client }
            func get() -> any ChatTurnClient { count += 1; return client }
        }
        let provider = Provider(client)
        let session = OwnerChatSession(clientProvider: { await provider.get() })
        try await session.submit(text: "Hello", target: .init(url: URL(string: "http://localhost:8000/v1")!, bearer: nil, model: "m"))
        try await wait(session) { if case .failed = $0.state { true } else { false } }
        #expect(await provider.count == 4)
    }
    @Test func approvalTransportFailureRemainsActionable() async throws {
        let client = try await fake([frame(#"{"type":"permission.request","op_id":"op","revision":1,"tool":"write_file","summary":"Edit","preview":"x","risk":"file","reason":"Approval","options":["once","deny"],"expires_at":2000000000}"#, 1)])
        await client.setOperationError(.disconnected(lastEventID: ""))
        let session = try await start(client)
        try await wait(session) { $0.projection?.pendingPermission != nil }
        do { try await session.resolve(opID: "op", decision: .allow); Issue.record("expected failure") } catch {}
        #expect(!(await session.snapshot().unavailablePermissions.contains("op")))
    }
    @Test func orderedTextAndTools() async throws {
        let client = try await fake([frame(#"{"type":"text.delta","part_id":"a","text":"before"}"#, 1), frame(#"{"type":"tool.started","call_id":"x","name":"search","args":{}}"#, 2), frame(#"{"type":"tool.result","call_id":"x","result":"found"}"#, 3), frame(#"{"type":"text.delta","part_id":"b","text":"after"}"#, 4), frame(#"{"type":"turn.terminal","outcome":"completed"}"#, 5)])
        let session = try await start(client)
        try await wait(session) { $0.state == .finished }
        let projection = try #require(await session.snapshot().projection)
        let rows = ownerProjectionMessages(projection, assistantID: UUID(), createdAt: Date())
        #expect(rows.map(\.content) == ["before", "", "found", "after"])
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
    @Test func newerPermissionRevisionIsActionable() async throws {
        let first = frame(#"{"type":"permission.request","op_id":"op","revision":1,"tool":"write_file","summary":"Edit","preview":"x","risk":"file","reason":"Approval","options":["once","deny"],"expires_at":2000000000}"#, 1)
        let second = frame(#"{"type":"permission.request","op_id":"op","revision":2,"tool":"write_file","summary":"Edit","preview":"x","risk":"file","reason":"Approval","options":["once","deny"],"expires_at":2000000000}"#, 2)
        let client = try await fake([first])
        await client.setReceipt(.object(["id": .string("op"), "state": .string("intent"), "revision": .integer(2), "expires_at": .integer(2000000000)]))
        let session = try await start(client)
        try await wait(session) { $0.projection?.pendingPermission != nil }
        do { try await session.resolve(opID: "op", decision: .allow); Issue.record("expected stale permission") } catch {}
        #expect(await session.snapshot().unavailablePermissions.contains("op"))
        await client.setFrames([second])
        try await wait(session) { $0.projection?.pendingPermission?.revision == 2 }
        #expect(!(await session.snapshot().unavailablePermissions.contains("op")))
        try await session.resolve(opID: "op", decision: .allow)
        #expect(await client.calls.contains { if case .respond = $0 { true } else { false } })
    }
    @Test func approvalCopyIsProjected() async throws {
        let client = try await fake([
            frame(#"{"type":"tool.started","call_id":"call","name":"write_file","args":{"op_id":"secret-op"}}"#, 1),
            frame(#"{"type":"permission.request","op_id":"secret-op","revision":1,"tool":"write_file","title":"Update config","summary":"Change the setting","preview":"+enabled = true","risk":"file","reason":"Approval","options":["once","deny"],"expires_at":2000000000}"#, 2)
        ])
        let session = try await start(client)
        try await wait(session) { $0.projection?.pendingPermission != nil }
        let projection = try #require(await session.snapshot().projection)
        let approval = try #require(ownerProjectionApprovals(projection, unavailable: [])["call"])
        #expect(approval.title == "Update config")
        #expect(approval.summary == "Change the setting")
        #expect(approval.preview == "+enabled = true")
        let rows = ownerProjectionMessages(projection, assistantID: UUID(), createdAt: Date())
        #expect(rows[0].toolCalls?.first?.function.arguments.contains("secret-op") == false)
    }
}
