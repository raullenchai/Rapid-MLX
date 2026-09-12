import Foundation
import Testing
@testable import Rapid

/// The chat request body is not just transport. The engine renders `tools` and
/// the message list into the prompt TEXT through the model's chat template, and
/// its prefix cache reuses a stored request only when the new one is a
/// byte-exact token prefix of it. So the body's serialization is part of the
/// prompt, and any reordering between two turns re-prefills the conversation.
///
/// 0.14.1 dogfood, captured from two consecutive turns of one conversation:
///
///     turn 1  "tools":[{"function":{"name":"web_search","description":…
///     turn 2  "tools":[{"type":"function","function":{"name":"web_search","parameters":…
///
/// Same four tools in the same order, different bytes — `parameters` is a
/// ``CodableJSON`` blob whose `.object` case is a Swift dictionary. The engine
/// reported `shared=96 entry_len=1460 requested_len=1511` and re-prefilled all
/// 1511 tokens: 4.6 s to first token on a follow-up, 15–17 s with an 8-page PDF
/// in the conversation.
///
/// These tests pin the two properties that fix it: the body is byte-stable
/// across encodes, and its keys are sorted (which is what makes it stable
/// across app launches too, so a reloaded conversation can still hit the
/// engine's on-disk prefix cache).
@Suite("Chat request body is byte-stable")
struct WireBodyDeterminismTests {

    /// A tool whose schema has enough sibling keys at two levels that an
    /// unordered encode would show up.
    private static func tool(_ name: String) -> ToolDefinition {
        ToolDefinition(
            name: name,
            description: "Tool \(name) for the determinism fixture.",
            parameters: .object([
                "type": .string("object"),
                "required": .array([.string("query")]),
                "properties": .object([
                    "query": .object([
                        "type": .string("string"),
                        "description": .string("Search query in natural language."),
                    ]),
                    "limit": .object([
                        "type": .string("number"),
                        "description": .string("How many results."),
                    ]),
                    "authority": .object([
                        "type": .string("string"),
                        "description": .string("Restrict to one site."),
                    ]),
                ]),
            ])
        )
    }

    private static func request() -> ChatStreamClient.Request {
        ChatStreamClient.Request(
            alias: "test-model",
            messages: [
                ChatMessage(role: .system, content: "[CURRENT DATE]\nToday is Friday."),
                ChatMessage(role: .user, content: "Name three colors."),
                ChatMessage(role: .assistant, content: "Red, blue, and green."),
                ChatMessage(role: .user, content: "Now name three animals."),
            ],
            tools: [tool("web_search"), tool("browse"), tool("weather"), tool("read_document")],
            supportsImageInput: false
        )
    }

    @Test("Encoding the same request twice yields identical bytes")
    func encodingIsByteStable() async throws {
        let first = try #require(await WireBodyCaptureProtocol.capture(Self.request()))
        let second = try #require(await WireBodyCaptureProtocol.capture(Self.request()))
        #expect(first == second,
                "Two sends of the same logical request must serialize identically — the engine's prefix cache compares bytes, not meaning.")
    }

    @Test("Body keys are sorted at every level, including inside a tool schema")
    func keysAreSorted() async throws {
        let body = try #require(await WireBodyCaptureProtocol.capture(Self.request()))
        let text = try #require(String(data: body, encoding: .utf8))

        // Sorted order is what makes the bytes stable across app launches as
        // well as across requests, so assert the property directly rather than
        // just observing stability within one process.
        func index(_ needle: String) throws -> String.Index {
            try #require(text.range(of: needle)?.lowerBound)
        }
        // Top level: "max_tokens" < "messages" < "model" < "stream".
        #expect(try index("\"max_tokens\"") < index("\"messages\""))
        #expect(try index("\"messages\"") < index("\"model\""))
        #expect(try index("\"model\"") < index("\"stream\""))
        // Inside a tool: "description" < "name" < "parameters", and the schema
        // object's own keys sort too.
        let toolsText = String(text[try index("\"tools\"")...])
        func toolIndex(_ needle: String) throws -> String.Index {
            try #require(toolsText.range(of: needle)?.lowerBound)
        }
        #expect(try toolIndex("\"description\"") < toolIndex("\"name\""))
        #expect(try toolIndex("\"name\"") < toolIndex("\"parameters\""))
        #expect(try toolIndex("\"properties\"") < toolIndex("\"required\""))
        // The tool object itself: "function" sorts before "type".
        #expect(toolsText.hasPrefix("\"tools\":[{\"function\""))
    }

    @Test("A follow-up turn's body extends the previous turn's byte-for-byte")
    func followUpExtendsThePreviousBody() async throws {
        // The property the prefix cache actually needs: turn two's prompt
        // region must START with turn one's. Tool order and key order are part
        // of that prefix, which is why they have to be stable.
        let turnOne = ChatStreamClient.Request(
            alias: "test-model",
            messages: [
                ChatMessage(role: .system, content: "[CURRENT DATE]\nToday is Friday."),
                ChatMessage(role: .user, content: "Name three colors."),
            ],
            tools: [Self.tool("web_search"), Self.tool("browse")],
            supportsImageInput: false
        )
        let turnTwo = ChatStreamClient.Request(
            alias: "test-model",
            messages: [
                ChatMessage(role: .system, content: "[CURRENT DATE]\nToday is Friday."),
                ChatMessage(role: .user, content: "Name three colors."),
                ChatMessage(role: .assistant, content: "Red, blue, and green."),
                ChatMessage(role: .user, content: "Now name three animals."),
            ],
            tools: [Self.tool("web_search"), Self.tool("browse")],
            supportsImageInput: false
        )
        let oneData = try #require(await WireBodyCaptureProtocol.capture(turnOne))
        let twoData = try #require(await WireBodyCaptureProtocol.capture(turnTwo))
        let one = try #require(String(data: oneData, encoding: .utf8))
        let two = try #require(String(data: twoData, encoding: .utf8))
        // Sorted keys put "messages" before "model"/"tools", so the messages
        // array is the one region that legitimately grows; everything encoded
        // before it must match, and the earlier turns inside it must match too.
        let head = try #require(one.range(of: "\"messages\":["))
        #expect(one[..<head.upperBound] == two[..<head.upperBound])
        let firstTurns = try #require(
            one.range(of: "{\"content\":\"Name three colors.\",\"role\":\"user\"}")
        )
        #expect(one[..<firstTurns.upperBound] == two[..<firstTurns.upperBound],
                "Everything up to and including the first user turn must be byte-identical between the two requests.")
    }
}

/// Captures the encoded HTTP body of one ``ChatStreamClient/send`` call.
///
/// The body is written on the `URLProtocol` loading thread and read from the
/// caller afterwards, so it is boxed behind a lock (the pattern the other wire
/// captures in this suite use).
private final class WireBodyCaptureProtocol: URLProtocol, @unchecked Sendable {
    private final class Store: @unchecked Sendable {
        private let lock = NSLock()
        private var body: Data?
        func get() -> Data? { lock.lock(); defer { lock.unlock() }; return body }
        func set(_ value: Data?) { lock.lock(); defer { lock.unlock() }; body = value }
    }

    private static let store = Store()

    static func capture(_ request: ChatStreamClient.Request) async -> Data? {
        store.set(nil)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [WireBodyCaptureProtocol.self]
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://wire-determinism")!,
            session: URLSession(configuration: config)
        )
        try? await client.send(request) { _ in }
        return store.get()
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.store.set(Self.bodyData(from: request))
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func bodyData(from request: URLRequest) -> Data? {
        guard let stream = request.httpBodyStream else { return request.httpBody }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.read(pointer.baseAddress!, maxLength: pointer.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return nil }
        }
    }
}
