import Foundation
import Testing
@testable import Rapid

private actor WireCollector {
    private(set) var events: [ChatWireEnvelope] = []
    func append(_ event: ChatWireEnvelope) { events.append(event) }
}

private struct OwnerStubReply: Sendable {
    var status = 200
    var contentType = "application/json"
    var data = Data("{}".utf8)
    var error: URLError.Code?
}

private final class OwnerStubState: @unchecked Sendable {
    private let lock = NSLock()
    private var replies: [OwnerStubReply] = []
    private var recorded: [URLRequest] = []
    var requests: [URLRequest] { lock.withLock { recorded } }
    func reset(_ replies: [OwnerStubReply]) { lock.withLock { self.replies = replies; recorded = [] } }
    func next(_ request: URLRequest) -> OwnerStubReply {
        lock.withLock {
            recorded.append(request)
            return replies.isEmpty ? OwnerStubReply(status: 500) : replies.removeFirst()
        }
    }
}

private final class OwnerStubProtocol: URLProtocol, @unchecked Sendable {
    static let state = OwnerStubState()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        var recorded = request
        if recorded.httpBody == nil, let stream = recorded.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var data = Data()
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                data.append(contentsOf: buffer.prefix(count))
            }
            recorded.httpBody = data
        }
        let reply = Self.state.next(recorded)
        if let error = reply.error {
            client?.urlProtocol(self, didFailWithError: URLError(error))
            return
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: reply.status,
                                       httpVersion: "HTTP/1.1", headerFields: ["Content-Type": reply.contentType])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Exercise URLSession's AsyncBytes with several callback boundaries.
        for offset in stride(from: 0, to: reply.data.count, by: 7) {
            client?.urlProtocol(self, didLoad: reply.data.subdata(in: offset..<min(offset + 7, reply.data.count)))
        }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@Suite("Owner turn URLSession contract", .serialized)
struct ChatTurnClientTests {
    private var intent: ChatTurnRequest {
        ChatTurnRequest(turnID: "turn-1", conversationID: "c", branchID: "b", expectedRevision: 7,
                        text: "hello", inferenceURL: "http://127.0.0.1:8000", model: "local",
                        inferenceAPIKey: "inference-test-key")
    }

    private func makeClient(_ replies: [OwnerStubReply]) throws -> URLSessionChatTurnClient {
        OwnerStubProtocol.state.reset(replies)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [OwnerStubProtocol.self]
        return try URLSessionChatTurnClient(endpoint: URL(string: "http://127.0.0.1:4567")!,
                                            bearer: "owner-test-key", session: URLSession(configuration: config))
    }

    @Test("Submit is SSE, authenticated, and preserves the immutable turn intent")
    func submit() async throws {
        let frames = try ChatWireFixtures.frames("completed").reduce(Data(), +)
        let client = try makeClient([OwnerStubReply(contentType: "text/event-stream; charset=utf-8", data: frames)])
        let collector = WireCollector()
        try await client.submit(intent, after: 0) { await collector.append($0) }
        #expect(await collector.events.count == 4)
        let request = try #require(OwnerStubProtocol.state.requests.first)
        #expect(request.url?.path == "/v1/rapid/chat/turns")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer owner-test-key")
        #expect(request.value(forHTTPHeaderField: "Accept") == "text/event-stream")
        #expect(request.value(forHTTPHeaderField: "Last-Event-ID") == "turn-1:0")
        let body = try #require(request.httpBody)
        #expect(try JSONDecoder().decode(ChatTurnRequest.self, from: body) == intent)
        let json = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        #expect(json["inference_api_key"] as? String == "inference-test-key")
        #expect(json["surface"] as? String == "desktop")
    }

    @Test("Resume uses GET and the last applied cursor")
    func resume() async throws {
        let frames = try ChatWireFixtures.frames("completed").dropFirst(2).reduce(Data(), +)
        let client = try makeClient([OwnerStubReply(contentType: "text/event-stream", data: frames)])
        let collector = WireCollector()
        try await client.stream(turnID: "turn-1", after: 2) { await collector.append($0) }
        #expect(await collector.events.map(\.seq) == [3, 4])
        let request = try #require(OwnerStubProtocol.state.requests.first)
        #expect(request.httpMethod == "GET")
        #expect(request.url?.path == "/v1/rapid/chat/turns/turn-1/events")
        #expect(request.value(forHTTPHeaderField: "Last-Event-ID") == "turn-1:2")
        #expect(request.httpBody == nil)
    }

    @Test("401, cursor-ahead 409, and validation 422 are preserved", arguments: [401, 409, 422])
    func httpErrors(_ status: Int) async throws {
        let client = try makeClient([OwnerStubReply(status: status)])
        await #expect(throws: ChatTurnClientError.http(status: status)) {
            try await client.stream(turnID: "turn-1", after: 999) { _ in Issue.record("HTTP error delivered an event") }
        }
        #expect(OwnerStubProtocol.state.requests.count == 1)
    }

    @Test("Content type and turn identity are checked before delivery")
    func malformedResponse() async throws {
        let frames = try ChatWireFixtures.frames("completed").reduce(Data(), +)
        let client = try makeClient([OwnerStubReply(data: frames)])
        await #expect(throws: ChatTurnClientError.invalidResponse) {
            try await client.stream(turnID: "turn-1", after: 0) { _ in Issue.record("Invalid MIME delivered an event") }
        }
        let other = try makeClient([OwnerStubReply(contentType: "text/event-stream", data: frames)])
        await #expect(throws: ChatWireError.wrongTurn) {
            try await other.stream(turnID: "other", after: 0) { _ in Issue.record("Wrong turn delivered an event") }
        }
    }

    @Test("Abrupt EOF leaves a reconnect cursor and never invents completion")
    func eof() async throws {
        let frames = try ChatWireFixtures.frames("completed").prefix(2).reduce(Data(), +)
        let client = try makeClient([OwnerStubReply(contentType: "text/event-stream", data: frames)])
        let collector = WireCollector()
        await #expect(throws: ChatTurnClientError.disconnected(lastEventID: "turn-1:2")) {
            try await client.stream(turnID: "turn-1", after: 0) { await collector.append($0) }
        }
        #expect(await collector.events.count == 2)
        #expect(OwnerStubProtocol.state.requests.count == 1)
    }

    @Test("Lost POST response reconciles admission, then reuses identical bytes and ID", arguments: [200, 404])
    func lostSubmit(_ lookupStatus: Int) async throws {
        let frames = try ChatWireFixtures.frames("completed").reduce(Data(), +)
        let client = try makeClient([
            OwnerStubReply(error: .networkConnectionLost),
            OwnerStubReply(status: lookupStatus),
            OwnerStubReply(contentType: "text/event-stream", data: frames)
        ])
        try await client.submit(intent, after: 0) { _ in }
        let requests = OwnerStubProtocol.state.requests
        #expect(requests.map(\.httpMethod) == ["POST", "GET", "POST"])
        #expect(requests[1].url?.path == "/v1/rapid/chat/turns/turn-1")
        #expect(requests[0].httpBody == requests[2].httpBody)
        #expect(requests.allSatisfy { $0.value(forHTTPHeaderField: "Authorization") == "Bearer owner-test-key" })
    }

    @Test("Consumer cancellation detaches without posting cancel")
    func detach() async throws {
        let frames = try ChatWireFixtures.frames("completed").reduce(Data(), +)
        let client = try makeClient([OwnerStubReply(contentType: "text/event-stream", data: frames)])
        await #expect(throws: CancellationError.self) {
            try await client.stream(turnID: "turn-1", after: 0) { _ in throw CancellationError() }
        }
        #expect(OwnerStubProtocol.state.requests.count == 1)
        #expect(OwnerStubProtocol.state.requests[0].httpMethod == "GET")
    }

    @Test("Control endpoints use exact request shapes and bearer auth")
    func controls() async throws {
        let context = #"{"type":"context","used_tokens":10,"limit_tokens":512,"system_tools_tokens":2,"conversation_tokens":8,"tool_output_tokens":0,"free_tokens":502,"percent":2}"#
        let client = try makeClient([
            OwnerStubReply(), OwnerStubReply(), OwnerStubReply(),
            OwnerStubReply(data: Data(context.utf8)),
            OwnerStubReply(data: Data(("{\"context\":" + context + ",\"notice\":\"Done\"}").utf8)),
            OwnerStubReply()
        ])
        _ = try await client.cancel(turnID: "turn-1")
        let permission = ChatPermissionResponse(opID: "op", revision: 3, decision: .deny, scope: .once, reason: "No")
        _ = try await client.respond(permission)
        _ = try await client.operation(opID: "op")
        #expect(try await client.context(intent).usedTokens == 10)
        #expect(try await client.compact(intent, instructions: "Preserve names").notice == "Done")
        _ = try await client.undo(conversationID: "c", operationID: "undo-1")
        let requests = OwnerStubProtocol.state.requests
        #expect(requests.map { $0.url!.path } == [
            "/v1/rapid/chat/turns/turn-1/cancel", "/v1/rapid/chat/permission.response",
            "/v1/rapid/chat/operations/op", "/v1/rapid/chat/conversations/c/context",
            "/v1/rapid/chat/conversations/c/compact", "/v1/rapid/chat/conversations/c/undo"
        ])
        #expect(requests.map(\.httpMethod) == ["POST", "POST", "GET", "POST", "POST", "POST"])
        #expect(requests.allSatisfy { $0.value(forHTTPHeaderField: "Authorization") == "Bearer owner-test-key" })
        #expect(try JSONDecoder().decode(ChatPermissionResponse.self, from: requests[1].httpBody!) == permission)
        #expect(try JSONDecoder().decode(ChatTurnRequest.self, from: requests[3].httpBody!) == intent)
        let compact = try JSONSerialization.jsonObject(with: requests[4].httpBody!) as! [String: Any]
        #expect(compact["instructions"] as? String == "Preserve names")
        #expect((compact["request"] as? [String: Any])?["turn_id"] as? String == "turn-1")
        #expect(try JSONSerialization.jsonObject(with: requests[5].httpBody!) as? [String: String] == ["operation_id": "undo-1"])
    }

    @Test("Fake client supports the same protocol and replay cursor")
    func fake() async throws {
        let context = try JSONDecoder().decode(ChatWireContextEvent.self, from: Data(
            #"{"used_tokens":0,"limit_tokens":512,"system_tools_tokens":0,"conversation_tokens":0,"tool_output_tokens":0,"free_tokens":512,"percent":0}"#.utf8))
        let fake = FakeChatTurnClient(frames: try ChatWireFixtures.frames("completed"), context: context)
        let client: any ChatTurnClient = fake
        let collector = WireCollector()
        try await client.stream(turnID: "turn-1", after: 2) { await collector.append($0) }
        #expect(await collector.events.map(\.seq) == [3, 4])
        _ = try await client.cancel(turnID: "turn-1")
        #expect(await fake.calls == [.stream("turn-1", after: 2), .cancel("turn-1")])
    }
}
