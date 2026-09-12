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

    private static func request(lastUser: String = "Now name three animals.") -> ChatStreamClient.Request {
        ChatStreamClient.Request(
            alias: "test-model",
            messages: [
                ChatMessage(role: .system, content: "[CURRENT DATE]\nToday is Friday."),
                ChatMessage(role: .user, content: "Name three colors."),
                ChatMessage(role: .assistant, content: "Red, blue, and green."),
                ChatMessage(role: .user, content: lastUser),
            ],
            tools: [tool("web_search"), tool("browse"), tool("weather"), tool("read_document")],
            supportsImageInput: false
        )
    }

    /// Index of the `]` that closes the top-level `"messages":[...]` array,
    /// found by bracket-depth counting (string- and escape-aware) rather
    /// than by searching for a literal. The follow-up test compares
    /// everything up to that point, so the window must be the real end of
    /// the array even if a fixture later contains a bracket or a quote.
    private static func messagesArrayClose(in text: String) -> String.Index? {
        guard let open = text.range(of: "\"messages\":[") else { return nil }
        var depth = 0
        var inString = false
        var escaped = false
        var index = text.index(before: open.upperBound)  // the `[` itself
        while index < text.endIndex {
            let character = text[index]
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                inString.toggle()
            } else if !inString {
                if character == "[" || character == "{" {
                    depth += 1
                } else if character == "]" || character == "}" {
                    depth -= 1
                    if depth == 0 { return index }
                }
            }
            index = text.index(after: index)
        }
        return nil
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
        // array is the one region that legitimately grows. Three things have
        // to hold, and all three are the property the prefix cache needs --
        // comparing only through the FIRST user message (an earlier version
        // of this test) would have stayed green while a later message, or the
        // whole tools array, changed underneath.
        let oneClose = try #require(Self.messagesArrayClose(in: one))
        let twoClose = try #require(Self.messagesArrayClose(in: two))

        // (1) Turn one's ENTIRE encoding up to the end of its last message
        //     object -- the request head, every earlier message, and that
        //     message's closing brace -- is a byte prefix of turn two.
        let onePrefix = String(one[..<oneClose])
        let sharedLength = zip(onePrefix, two).prefix(while: { $0 == $1 }).count
        #expect(two.hasPrefix(onePrefix),
                "Turn two must begin with every byte of turn one's message array, closing brace included; the two diverged after \(sharedLength) of \(onePrefix.count) characters.")

        // (2) It EXTENDS that prefix rather than merely equalling it: the very
        //     next bytes are the new assistant turn, appended.
        #expect(two.dropFirst(onePrefix.count).hasPrefix(",{\"content\":\"Red, blue, and green.\""),
                "Turn two must append the new turns after turn one's bytes, not re-serialize the array.")

        // (3) Everything encoded AFTER the array -- `model`, `stream`, and the
        //     whole `tools` array -- is byte-identical. The tools array is the
        //     one this PR exists for: the engine renders it into the prompt
        //     text, so a reordered schema is a reordered prompt.
        #expect(one[oneClose...] == two[twoClose...],
                "Every field after `messages` (`model`, `stream`, `tools`) must be byte-identical between turns.")
    }

    @Test("Two captures running concurrently do not read each other's bodies")
    func concurrentCapturesStayIsolated() async throws {
        // The capture harness is process-global, so without a per-capture key
        // two overlapping sends would overwrite (or consume) one another and
        // this suite would go flaky -- or, worse, falsely green, comparing one
        // request against itself. The desktop suite runs `--no-parallel`
        // today; correctness must not depend on a runner flag.
        async let alpha = WireBodyCaptureProtocol.capture(Self.request(lastUser: "Alpha marker question."))
        async let beta = WireBodyCaptureProtocol.capture(Self.request(lastUser: "Beta marker question."))
        let (alphaData, betaData) = await (alpha, beta)
        let alphaBody = try #require(alphaData)
        let betaBody = try #require(betaData)
        let alphaText = try #require(String(data: alphaBody, encoding: .utf8))
        let betaText = try #require(String(data: betaBody, encoding: .utf8))
        #expect(alphaText.contains("Alpha marker question."))
        #expect(!alphaText.contains("Beta marker question."))
        #expect(betaText.contains("Beta marker question."))
        #expect(!betaText.contains("Alpha marker question."))
    }
}

/// Captures the encoded HTTP body of one ``ChatStreamClient/send`` call.
///
/// `URLProtocol` subclasses are instantiated by the loading system, so the
/// captured body has to travel through process-global state. It is keyed by a
/// token minted per capture and carried in the fake base URL's host, which the
/// protocol reads back off the outgoing request: two captures in flight at once
/// therefore write to (and consume) different slots instead of clobbering each
/// other. The store is boxed behind a lock because the write happens on the
/// loading thread and the read on the caller's.
private final class WireBodyCaptureProtocol: URLProtocol, @unchecked Sendable {
    private final class Store: @unchecked Sendable {
        private let lock = NSLock()
        private var bodies: [String: Data] = [:]
        func put(_ value: Data?, for token: String) {
            lock.lock()
            defer { lock.unlock() }
            if let value { bodies[token] = value }
        }
        /// Read-and-remove, so a capture can never see a stale body left
        /// behind by an earlier one.
        func take(_ token: String) -> Data? {
            lock.lock()
            defer { lock.unlock() }
            return bodies.removeValue(forKey: token)
        }
    }

    private static let store = Store()

    static func capture(_ request: ChatStreamClient.Request) async -> Data? {
        // Lowercased because `URL.host` normalises case; hyphens and hex are
        // valid host characters, so a UUID round-trips unchanged.
        let token = UUID().uuidString.lowercased()
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [WireBodyCaptureProtocol.self]
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://\(token)")!,
            session: URLSession(configuration: config)
        )
        try? await client.send(request) { _ in }
        return store.take(token)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        if let token = request.url?.host {
            Self.store.put(Self.bodyData(from: request), for: token)
        }
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
