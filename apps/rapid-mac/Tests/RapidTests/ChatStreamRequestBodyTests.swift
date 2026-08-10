import Foundation
import Testing
@testable import Rapid

/// Contract: the on-the-wire body that ``ChatStreamClient`` ships to
/// ``/v1/chat/completions`` must include the BCG-recipe sampling
/// defaults (``repetition_penalty=1.1`` etc.) so small fine-tuned
/// hybrid models (Qwopus 9B/27B) don't degenerate into endless
/// list-cycle loops without ever emitting EOS.
///
/// We assert the JSON body via direct encode of
/// ``Wire.ChatCompletionRequest`` rather than spinning up a fake
/// HTTP server — the wire shape is the contract that matters; the
/// transport is exercised by every integration smoke run.
@Suite("ChatStreamClient wire body")
struct ChatStreamRequestBodyTests {
    @Test("Default request includes repetition_penalty 1.1")
    func defaultsIncludeRepetitionPenalty() throws {
        let body = encode(temperature: 0.7, topP: 0.95)
        #expect(body["repetition_penalty"] as? Double == 1.1)
    }

    @Test("Default request sends frequency_penalty 0.0 (OFF — see v0.4.1 emoji-scatter fix)")
    func defaultsIncludeFrequencyPenalty() throws {
        let body = encode(temperature: 0.7, topP: 0.95)
        // v0.4.1: dropped fp/pp defaults to neutral after live
        // testing showed the previous ``fp=1.0, pp=0.5`` recipe
        // pushed small 4-bit models (qwen3.5-4b) into emoji /
        // non-Latin Unicode space. ``repetition_penalty=1.1``
        // catches the same loop pathology without the scatter.
        #expect(body["frequency_penalty"] as? Double == 0.0)
    }

    @Test("Default request sends presence_penalty 0.0 (OFF — see v0.4.1 emoji-scatter fix)")
    func defaultsIncludePresencePenalty() throws {
        let body = encode(temperature: 0.7, topP: 0.95)
        #expect(body["presence_penalty"] as? Double == 0.0)
    }

    @Test("Default temperature + top_p preserved")
    func defaultTempPreserved() throws {
        let body = encode(temperature: 0.7, topP: 0.95)
        #expect(body["temperature"] as? Double == 0.7)
        #expect(body["top_p"] as? Double == 0.95)
    }

    @Test("Caller can override penalties for advanced users")
    func canOverridePenalties() throws {
        // A future "advanced sampling" sheet might let the user
        // disable penalties entirely; the request struct must allow
        // a zero / 1.0 pass-through (rapid-mlx treats 1.0 / 0.0 as
        // disabled).
        let req = ChatStreamClient.Request(
            alias: "test",
            messages: [],
            repetitionPenalty: 1.0,
            frequencyPenalty: 0.0,
            presencePenalty: 0.0
        )
        let body = try encode(request: req)
        #expect(body["repetition_penalty"] as? Double == 1.0)
        #expect(body["frequency_penalty"] as? Double == 0.0)
        #expect(body["presence_penalty"] as? Double == 0.0)
    }

    @Test("Tools field omitted when registry is empty")
    func toolsOmittedWhenEmpty() throws {
        let body = encode(temperature: 0.7, topP: 0.95)
        // ``tools`` and ``tool_choice`` must be absent when nil so
        // the server doesn't see ``"tools": null`` and reject the
        // request (some validators are strict on null vs absent).
        #expect(body["tools"] == nil)
        #expect(body["tool_choice"] == nil)
    }

    // MARK: - helpers

    /// Build the wire body the same way ``ChatStreamClient.send``
    /// would, then return it as a parsed JSON dict for inspection.
    private func encode(temperature: Double, topP: Double) -> [String: Any] {
        let req = ChatStreamClient.Request(
            alias: "qwopus-9b-4bit",
            messages: [
                ChatMessage(role: .user, content: "hi", status: .complete)
            ],
            temperature: temperature,
            topP: topP
        )
        // ``encode(request:)`` throws but the only failure path is
        // ``JSONEncoder`` choking on Encodable, which can't happen
        // here. ``try!`` for test brevity.
        return try! encode(request: req)
    }

    private func encode(request: ChatStreamClient.Request) throws -> [String: Any] {
        let body = Wire.ChatCompletionRequest(
            model: request.alias,
            messages: request.messages,
            stream: true,
            temperature: request.temperature,
            top_p: request.topP,
            max_tokens: request.maxTokens,
            repetition_penalty: request.repetitionPenalty,
            frequency_penalty: request.frequencyPenalty,
            presence_penalty: request.presencePenalty,
            tools: (request.tools?.isEmpty == false) ? request.tools : nil,
            tool_choice: Wire.ToolChoice.resolve(
                hasTools: request.tools?.isEmpty == false,
                forcedTool: request.forcedTool
            ),
            stream_options: .init(include_usage: true),
            // #161: mirror production's enableThinking → kwargs mapping so
            // the encoded body in tests matches what ``send()`` actually
            // ships on the wire.
            chat_template_kwargs: request.enableThinking
                ? nil
                : .init(enable_thinking: false)
        )
        let data = try JSONEncoder().encode(body)
        let parsed = try JSONSerialization.jsonObject(with: data)
        return parsed as! [String: Any]
    }
}

/// #161: pin the ``chat_template_kwargs`` wire-body contract used to
/// suppress the ``<think>...</think>`` block on hybrid models (Qwen 3 /
/// 3.5 / 3.6, GLM 4.7, Qwopus). Without this, 4 B / 9 B-class hybrids
/// burn the entire ``max_tokens`` budget inside their reasoning trace
/// and emit zero answer tokens — the "prompts don't work" repro from
/// the 2026-06-14 cliclick triage.
@Suite("ChatStreamClient #161 enableThinking wire contract", .serialized)
struct ChatStream161ThinkingBodyTests {
    @Test("Default request (thinking OFF) ships chat_template_kwargs: {enable_thinking: false}")
    func defaultEmitsKwargFalse() throws {
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        let body = try encode(request: req)
        guard let kwargs = body["chat_template_kwargs"] as? [String: Any] else {
            Issue.record("expected chat_template_kwargs block, got \(body["chat_template_kwargs"] ?? "nil")")
            return
        }
        #expect(kwargs["enable_thinking"] as? Bool == false)
    }

    @Test("Thinking ON omits the kwarg entirely — hybrid chat template keeps its default ON behaviour")
    func thinkingOnOmitsKwarg() throws {
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)],
            enableThinking: true
        )
        let body = try encode(request: req)
        // Strict absence (not ``null``) — a strict server would treat
        // ``chat_template_kwargs: null`` differently from omission.
        #expect(body["chat_template_kwargs"] == nil)
    }

    @Test("codex r1 NIT: raw JSON body MUST NOT contain the literal \"chat_template_kwargs\":null")
    func rawJSONHasNoNullKwargs() throws {
        // Defends against a future encode(to:) refactor that swaps
        // out the default Encodable behaviour for ``encodeIfPresent``
        // and accidentally serialises a nil Optional as ``null``
        // instead of omitting the key. ``JSONSerialization`` round-
        // trips ``null`` keys to absent in the parsed dict, so the
        // ``body["chat_template_kwargs"] == nil`` check above isn't
        // strict enough on its own.
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)],
            enableThinking: true
        )
        let body = Wire.ChatCompletionRequest(
            model: req.alias,
            messages: req.messages,
            stream: true,
            temperature: req.temperature,
            top_p: req.topP,
            max_tokens: req.maxTokens,
            repetition_penalty: req.repetitionPenalty,
            frequency_penalty: req.frequencyPenalty,
            presence_penalty: req.presencePenalty,
            tools: nil,
            tool_choice: nil,
            stream_options: .init(include_usage: true),
            chat_template_kwargs: req.enableThinking
                ? nil
                : .init(enable_thinking: false)
        )
        let data = try JSONEncoder().encode(body)
        guard let raw = String(data: data, encoding: .utf8) else {
            Issue.record("body wasn't UTF-8 decodable")
            return
        }
        #expect(!raw.contains("\"chat_template_kwargs\":null"),
                "wire body must OMIT the kwarg, never emit \"chat_template_kwargs\":null")
        #expect(!raw.contains("\"chat_template_kwargs\": null"),
                "ditto — guard against an outputFormatting change that adds a space after the colon")
    }

    @Test("End-to-end: send() ships enable_thinking: false on the wire by default")
    @MainActor
    func endToEndDefault() async throws {
        Thinking161Protocol.reset()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: Thinking161Protocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        try await client.send(req) { _ in }
        guard let body = Thinking161Protocol.lastRequestBody else {
            Issue.record("no request body captured")
            return
        }
        let parsed = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        guard let kwargs = parsed["chat_template_kwargs"] as? [String: Any] else {
            Issue.record("send() must ship chat_template_kwargs by default — got \(parsed["chat_template_kwargs"] ?? "nil")")
            return
        }
        #expect(kwargs["enable_thinking"] as? Bool == false)
    }

    @Test("End-to-end: send() with enableThinking: true omits the kwarg")
    @MainActor
    func endToEndThinkingOn() async throws {
        Thinking161Protocol.reset()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: Thinking161Protocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)],
            enableThinking: true
        )
        try await client.send(req) { _ in }
        guard let body = Thinking161Protocol.lastRequestBody else {
            Issue.record("no request body captured")
            return
        }
        let parsed = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        #expect(parsed["chat_template_kwargs"] == nil)
    }

    // MARK: - helpers

    private func encode(request: ChatStreamClient.Request) throws -> [String: Any] {
        let body = Wire.ChatCompletionRequest(
            model: request.alias,
            messages: request.messages,
            stream: true,
            temperature: request.temperature,
            top_p: request.topP,
            max_tokens: request.maxTokens,
            repetition_penalty: request.repetitionPenalty,
            frequency_penalty: request.frequencyPenalty,
            presence_penalty: request.presencePenalty,
            tools: (request.tools?.isEmpty == false) ? request.tools : nil,
            tool_choice: Wire.ToolChoice.resolve(
                hasTools: request.tools?.isEmpty == false,
                forcedTool: request.forcedTool
            ),
            stream_options: .init(include_usage: true),
            chat_template_kwargs: request.enableThinking
                ? nil
                : .init(enable_thinking: false)
        )
        let data = try JSONEncoder().encode(body)
        let parsed = try JSONSerialization.jsonObject(with: data)
        return parsed as! [String: Any]
    }
}

/// Standalone URLProtocol for the #161 suite. Avoids racing the
/// shared ``FakeChatProtocol.lastRequestBody`` static with the v0.4.1
/// / v0.4.13 suites under swift-testing's parallel scheduler — those
/// other suites reset the global between their own tests and clobber
/// the request body our suite is trying to read back.
///
/// codex r1 NIT — single-suite-owned. The ``lastRequestBody`` static
/// is meant to be read ONLY from ``ChatStream161ThinkingBodyTests``,
/// which uses ``.serialized`` to keep its own four tests from
/// racing each other. If you find yourself instantiating
/// ``Thinking161Protocol.session()`` from a different suite, mint a
/// new URLProtocol class instead — sharing this static across
/// suites would reintroduce exactly the race this type was created
/// to fix.
final class Thinking161Protocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var lastRequestBody: Data?

    static func reset() { lastRequestBody = nil }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [Thinking161Protocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        if let stream = request.httpBodyStream {
            stream.open()
            var data = Data()
            let bufSize = 4096
            var buf = [UInt8](repeating: 0, count: bufSize)
            while stream.hasBytesAvailable {
                let n = buf.withUnsafeMutableBufferPointer { ptr in
                    stream.read(ptr.baseAddress!, maxLength: bufSize)
                }
                if n > 0 { data.append(buf, count: n) }
                if n <= 0 { break }
            }
            stream.close()
            Thinking161Protocol.lastRequestBody = data
        } else {
            Thinking161Protocol.lastRequestBody = request.httpBody
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
}

/// Wire-body contract via a faked URLProtocol so we exercise the
/// real ``ChatStreamClient.send`` code path (codex round-1 NIT).
/// The fake captures the outgoing body bytes, returns a minimal
/// ``[DONE]`` SSE stream, and lets the test assert the on-the-wire
/// JSON the production path actually shipped.
@Suite("ChatStreamClient end-to-end body via URLProtocol", .serialized)
struct ChatStreamClientBodyE2ETests {
    @Test("send() actually ships repetition_penalty=1.1 by default")
    @MainActor
    func endToEndDefaultPenalty() async throws {
        FakeChatProtocol.reset()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: FakeChatProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwopus-9b-4bit",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        // ``ChatStreamClient.Event`` callbacks land on the MainActor;
        // hosting the test on the MainActor lets the closure mutate
        // local state without tripping Swift 6 region-isolation.
        var finished = false
        try await client.send(req) { event in
            if case .finished = event { finished = true }
        }
        #expect(finished)
        guard let body = FakeChatProtocol.lastRequestBody else {
            Issue.record("no request body captured")
            return
        }
        let parsed = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        #expect(parsed["repetition_penalty"] as? Double == 1.1)
        // v0.4.1: fp/pp default to 0.0 (see Suite above).
        #expect(parsed["frequency_penalty"] as? Double == 0.0)
        #expect(parsed["presence_penalty"] as? Double == 0.0)
        #expect(parsed["model"] as? String == "qwopus-9b-4bit")
        #expect(parsed["stream"] as? Bool == true)
    }

    @Test("v0.4.13: wire body opts into stream_options.include_usage")
    @MainActor
    func sendsIncludeUsage() async throws {
        FakeChatProtocol.reset()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: FakeChatProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        try await client.send(req) { _ in }
        guard let body = FakeChatProtocol.lastRequestBody else {
            Issue.record("no request body captured")
            return
        }
        let parsed = try JSONSerialization.jsonObject(with: body) as! [String: Any]
        guard let opts = parsed["stream_options"] as? [String: Any] else {
            Issue.record("stream_options block missing from wire body")
            return
        }
        // The whole point of v0.4.13: an OpenAI-spec server reading
        // this body MUST emit a terminal ``usage`` chunk. Pin the
        // exact field name + value so a future refactor can't
        // silently flip it to ``includeUsage`` (camelCase) which
        // OpenAI would silently ignore.
        #expect(opts["include_usage"] as? Bool == true)
    }
}

/// v0.4.13: separate suite for the ``include_usage`` → ``.usage``
/// event end-to-end path. Pin the parser semantics so a future
/// refactor of the SSE decoder can't silently drop the usage chunk.
///
/// ``.serialized`` because we share ``FakeChatProtocol.lastRequestBody``
/// with ``ChatStreamClientBodyE2ETests`` via a class static — Swift
/// Testing's default parallel scheduler would race two tests writing
/// to the same static and the read-back assertion in the older suite
/// would pick up the wrong body. ``.serialized`` only enforces order
/// WITHIN the suite, but in practice swift-testing schedules
/// serialized suites in sequence with each other too, which is what
/// closes the race.
@Suite("ChatStreamClient stream_options.include_usage (v0.4.13)", .serialized)
struct ChatStreamClientUsageTests {
    @Test("Usage chunk arriving before [DONE] surfaces as a .usage event with prompt + completion tokens")
    @MainActor
    func usageEventEmitted() async throws {
        UsageEmittingProtocol.reset()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: UsageEmittingProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        var captured: (prompt: Int, completion: Int)?
        var finished = false
        try await client.send(req) { event in
            switch event {
            case .usage(let p, let c):
                captured = (p, c)
            case .finished:
                finished = true
            default:
                break
            }
        }
        #expect(finished)
        #expect(captured?.prompt == 42)
        #expect(captured?.completion == 17)
    }

    @Test("Non-conforming server (no usage chunk) still completes cleanly — fallback path stays alive")
    @MainActor
    func noUsageChunkStillCompletes() async throws {
        // Use a dedicated protocol class — sharing
        // ``FakeChatProtocol.lastRequestBody`` with the v0.4.1 suite
        // races their parallel scheduler. ``NoUsageProtocol`` has no
        // static state at all, so it's safe to run alongside anything.
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: NoUsageProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        var sawUsage = false
        var sawFinish = false
        try await client.send(req) { event in
            switch event {
            case .usage: sawUsage = true
            case .finished: sawFinish = true
            default: break
            }
        }
        // The vanilla FakeChatProtocol's stream has no usage chunk.
        // Stream MUST finish cleanly (so the v0.4.12 char-estimate
        // path still runs); .usage MUST stay false (so we don't
        // accidentally regress and inherit a stale usage value from
        // a previous turn or default-construct one).
        #expect(sawFinish)
        #expect(!sawUsage)
    }
}

/// URLProtocol that emits a minimal SSE stream WITHOUT the v0.4.13
/// usage chunk — same shape a non-OpenAI / older rapid-mlx server
/// would produce. No static state so this is safe to run alongside
/// the v0.4.1 ``FakeChatProtocol`` tests.
final class NoUsageProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [NoUsageProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        // ``ChatStreamClient`` uploads through an HTTP body stream. Drain it
        // before replying: Foundation can otherwise wait forever for the
        // upload to finish on CI even though this stub ignores the payload.
        _ = requestBodyData(from: request)
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
}

/// URLProtocol that emits an OpenAI-spec compliant terminal usage
/// chunk before [DONE]. Mirrors the wire shape rapid-mlx (and any
/// other OpenAI-compatible server with the streaming-usage
/// extension) produces.
final class UsageEmittingProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var lastRequestBody: Data?

    static func reset() {
        lastRequestBody = nil
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [UsageEmittingProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        _ = requestBodyData(from: request)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Per OpenAI spec: the usage chunk has ``choices: []`` and
        // arrives AFTER finish_reason on the previous chunk but
        // BEFORE [DONE]. We replicate that ordering exactly.
        let body = """
        data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n
        data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":17,"total_tokens":59}}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// In-process URLProtocol that captures the outgoing request body
/// and returns a canned SSE stream. Avoids spinning up a real HTTP
/// server while still exercising ``ChatStreamClient.send``.
final class FakeChatProtocol: URLProtocol, @unchecked Sendable {
    // ``nonisolated(unsafe)`` is acceptable here — these are only
    // touched on the URLSession's protocol queue between
    // ``reset()`` and the test's await point.
    nonisolated(unsafe) static var lastRequestBody: Data?

    static func reset() {
        lastRequestBody = nil
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FakeChatProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        FakeChatProtocol.lastRequestBody = requestBodyData(from: request)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Minimum-viable stream that satisfies ChatStreamClient:
        // one chunk + [DONE]. ``finish_reason: "stop"`` so the loop
        // terminates cleanly.
        let body = """
        data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private func requestBodyData(from request: URLRequest) -> Data? {
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

/// #896: rapid-mlx that crashes mid-response closes the TCP socket
/// gracefully (uvicorn catches the SIGTERM and shuts down its SSE
/// streams cleanly) so the URLSession iterator simply EOFs without
/// throwing. v0.4.4 silently treated this as a successful
/// completion and marked the assistant message ``.complete``, which
/// (a) lied about the response state, and (b) gave the crash banner
/// nothing to anchor against because the chat surface looked fine.
///
/// Contract under v0.4.5+: an SSE stream that EOFs before either a
/// ``[DONE]`` sentinel OR a chunk with ``finish_reason`` is treated
/// as a transport failure (``ChatStreamError.streamTruncated``), so
/// the catch path in ``ChatViewModel.runOneStream`` marks the
/// in-flight message ``.failed`` and the crash banner appears next
/// to a visibly-broken response rather than a green checkmark.
@Suite("ChatStreamClient mid-response crash recovery (#896)")
struct ChatStreamClientCrashRecoveryTests {
    @Test("Stream EOFing without [DONE] or finish_reason throws streamTruncated")
    @MainActor
    func truncatedStreamThrows() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: TruncatedChatProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        var sawContent = false
        var sawFinished = false
        do {
            try await client.send(req) { event in
                switch event {
                case .content: sawContent = true
                case .finished: sawFinished = true
                default: break
                }
            }
            Issue.record("expected ChatStreamError.streamTruncated, got clean return")
        } catch ChatStreamError.streamTruncated {
            // Expected: partial content delivered, but no terminal
            // event followed before EOF → throw.
            #expect(sawContent, "content delta should have landed before the truncated EOF")
            #expect(!sawFinished, "no terminal .finished event should have been emitted on a truncated stream")
        } catch {
            Issue.record("expected ChatStreamError.streamTruncated, got \(error)")
        }
    }

    @Test("Stream with finish_reason but no [DONE] still completes cleanly (rapid-mlx contract tolerance)")
    @MainActor
    func finishReasonWithoutDoneStillCompletes() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: FinishReasonNoDoneProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        var finished = false
        try await client.send(req) { event in
            if case .finished(let reason) = event {
                finished = true
                #expect(reason == "stop")
            }
        }
        #expect(finished, "finish_reason: stop should still produce a .finished event even without [DONE]")
    }
}

/// URLProtocol that emits one content delta and then closes the
/// stream without [DONE] or finish_reason — simulates rapid-mlx
/// being SIGKILL'd mid-response.
final class TruncatedChatProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [TruncatedChatProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Two content chunks, no finish_reason, no [DONE]. The
        // server's TCP socket then closes cleanly — exactly what
        // uvicorn does when it catches SIGTERM mid-response and
        // shuts down the SSE generator.
        let body = """
        data: {"choices":[{"delta":{"content":"par"}}]}\n
        data: {"choices":[{"delta":{"content":"tial"}}]}\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// URLProtocol that emits a finish_reason chunk but omits the
/// trailing ``[DONE]`` sentinel. Some OpenAI-compatible servers
/// (including older rapid-mlx) skip the ``[DONE]`` when a
/// finish_reason already terminated the round; we tolerate this
/// because the contract really is "any terminal event ends the
/// stream", not "specifically [DONE]".
final class FinishReasonNoDoneProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FinishReasonNoDoneProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

// MARK: - Codex audit r1 SSE wire-quirks

/// URLProtocol that delivers an OpenAI-style error envelope
/// mid-stream: ``data: {"error":{"message":"out of memory"}}``
/// followed by a connection close. Pre-audit, the JSON decoder
/// would treat this as a malformed StreamChunk and silently skip,
/// then EOF would surface as ``.streamTruncated`` with no useful
/// reason. Post-fix the envelope is recognised and the message
/// reaches the UI as a transport error.
final class ErrorEnvelopeProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ErrorEnvelopeProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"thinking"}}]}\n
        data: {"error":{"message":"CUDA out of memory","type":"server_error","code":"oom"}}\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// URLProtocol that emits a terminal chunk carrying ONLY
/// ``finish_reason`` without a ``delta`` object — observed in the
/// wild from some OpenAI-compatible proxies. Pre-audit, the
/// ``Choice`` struct required ``delta`` so the whole chunk was
/// silently dropped and the stream EOF'd as ``streamTruncated``.
/// Post-fix the optional ``delta`` lets the finish_reason path
/// fire independently.
final class FinishReasonNoDeltaProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FinishReasonNoDeltaProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"hello"}}]}\n
        data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Contracts added by the codex audit r1 sweep of ChatStreamClient.
@Suite("ChatStreamClient codex-r1 SSE wire quirks")
struct ChatStreamCodexR1Tests {

    @Test("Mid-stream error envelope surfaces as ChatStreamError.transport")
    @MainActor
    func errorEnvelopeSurfacesAsTransport() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: ErrorEnvelopeProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        do {
            try await client.send(req) { _ in }
            Issue.record("expected ChatStreamError.transport, got clean return")
        } catch ChatStreamError.transport(let message) {
            #expect(message.contains("CUDA out of memory"))
        } catch {
            Issue.record("expected ChatStreamError.transport, got \(error)")
        }
    }

    @Test("Terminal chunk without a delta still produces a .finished event")
    @MainActor
    func finishWithoutDeltaCompletes() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: FinishReasonNoDeltaProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        var sawContent = false
        var finishedReason: String? = nil
        try await client.send(req) { event in
            switch event {
            case .content: sawContent = true
            case .finished(let r): finishedReason = r
            default: break
            }
        }
        #expect(sawContent, "the delta-bearing chunk should have surfaced its content")
        #expect(finishedReason == "stop", "delta-less terminal chunk must still emit finish_reason")
    }
}
