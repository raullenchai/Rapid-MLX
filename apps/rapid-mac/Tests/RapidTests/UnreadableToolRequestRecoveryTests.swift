import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Unreadable tool request recovery", .serialized)
struct UnreadableToolRequestRecoveryTests {
    @Test("a malformed call gets one more chance to reach the tool")
    func resamplesBeforeGivingUpOnTools() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: true)
        let registry = RecoveryToolRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("Look up the current result", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        #expect(registry.runCount == 1)
        #expect(model.messages.last?.content == "The verified result is ready.")
        #expect(model.messages.last?.toolCallArtifactSuppressed == false)
        #expect(model.messages.last?.status == .complete)
        #expect(model.lastError == nil)
        for body in UnreadableToolProtocol.requestBodies.prefix(2) {
            let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
            #expect(json["tools"] != nil)
        }
    }

    @Test("an early empty length stop gets one fresh sample")
    func retriesEarlyEmptyLength() async throws {
        UnreadableToolProtocol.reset(
            recoverOnSecondRequest: true, emptyLengthOnFirst: true
        )
        let registry = RecoveryToolRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )
        model.send("Look up the current result", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        #expect(registry.runCount == 1)
        #expect(model.messages.last?.content == "The verified result is ready.")
        let retryBody = try #require(UnreadableToolProtocol.requestBodies.dropFirst().first)
        let retryJSON = try #require(JSONSerialization.jsonObject(with: retryBody) as? [String: Any])
        #expect(retryJSON["max_tokens"] as? Int == 2048)
    }

    @Test("repeated early empty stops end with an honest answer")
    func repeatedEarlyEmptyLength() async throws {
        UnreadableToolProtocol.reset(
            recoverOnSecondRequest: false,
            emptyLengthOnFirst: true,
            emptyLengthOnSecond: true
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )
        model.send("东京今天天气怎么样", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.content == ChatViewModel.unreadableToolFallbackChineseCopy)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("two malformed calls end in a deterministic honest answer")
    func fallsBackWithoutPrintingTheArtifact() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false)
        let registry = RecoveryToolRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("What happened today?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(registry.runCount == 0)
        #expect(model.messages.last?.content == ChatViewModel.unreadableToolFallbackCopy)
        #expect(model.messages.last?.toolCallArtifactSuppressed == false)
        #expect(model.messages.last?.status == .complete)
        #expect(model.lastError == nil)

        for body in UnreadableToolProtocol.requestBodies {
            let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
            #expect(json["tools"] != nil)
        }
    }

    @Test("a model that keeps printing call syntax gets a plain answer")
    func repeatedArtifactDoesNotLeak() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false)
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )

        model.send("What happened today?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.content ==
            "I couldn't retrieve or verify the information needed for a reliable answer. Try again or choose another model.")
        #expect(model.messages.last?.toolCallArtifactSuppressed == false)
        #expect(model.messages.last?.status == .complete)
        #expect(ChatViewModel.unreadableToolFallbackCopy(for: "东京今天天气怎么样") ==
            ChatViewModel.unreadableToolFallbackChineseCopy)
    }

    @Test("an empty retry completion gives a clear fallback")
    func emptyRetryCompletion() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false, secondContent: "")
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )
        model.send("What happened today?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.content == ChatViewModel.unreadableToolFallbackCopy)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("prose above a retry raw call does not bypass the honest fallback")
    func retryProseCannotBypassFallback() async throws {
        UnreadableToolProtocol.reset(
            recoverOnSecondRequest: false,
            secondContent: "I couldn't verify today's result.\n<｜tool▁calls▁begin｜>"
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )
        model.send("What happened today?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.content == ChatViewModel.unreadableToolFallbackCopy)
        #expect(model.messages.last?.toolCallArtifactSuppressed == false)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("ungrounded weather claims are replaced in both languages", arguments: [
        ("东京今天天气怎么样", "东京今天晴，23°C。", ChatViewModel.unreadableToolFallbackChineseCopy),
        ("What's Tokyo's weather today?", "Tokyo is sunny and 23°C today.", ChatViewModel.unreadableToolFallbackCopy)
    ])
    func fabricatedWeatherIsNeverShown(
        prompt: String, fabricated: String, expected: String
    ) async throws {
        UnreadableToolProtocol.reset(
            recoverOnSecondRequest: false,
            secondContent: fabricated,
            secondReasoning: "Weather is definitely 23°C and sunny."
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )
        model.send(prompt, alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.content == expected)
        #expect(model.messages.last?.reasoning == "")
        #expect(model.messages.last?.reasoningTruncated == false)
        #expect(model.messages.last?.toolNotCalledFlagged == false)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("stopping the first retry preserves cancellation")
    func stopPendingRetry() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false, holdSecond: true)
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )
        model.send("东京今天天气怎么样", alias: "test-model")
        for _ in 0..<300 where UnreadableToolProtocol.requestBodies.count < 2 {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        model.stop()
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.errorMessage == "Stopped.")
    }

}

@MainActor
private final class RecoveryToolRegistry: ToolRegistry {
    private(set) var runCount = 0

    var definitions: [ToolDefinition] {
        [ToolDefinition(
            name: "weather",
            description: "Look up a current result",
            parameters: .object(["type": .string("object")])
        )]
    }

    func run(_ call: ToolCall) async -> ToolCallResult {
        runCount += 1
        return ToolCallResult(toolCallID: call.id, content: "Verified evidence")
    }
}

private final class UnreadableToolProtocol: URLProtocol, @unchecked Sendable {
    private struct Configuration: Sendable {
        var recoverOnSecondRequest = false
        var secondContent: String?
        var secondReasoning: String?
        var holdSecond = false
        var emptyLengthOnFirst = false
        var emptyLengthOnSecond = false
    }

    private static let stateLock = NSLock()
    nonisolated(unsafe) private static var bodies: [Data] = []
    nonisolated(unsafe) private static var configuration = Configuration()

    static var requestBodies: [Data] {
        stateLock.withLock { bodies }
    }

    static func reset(
        recoverOnSecondRequest: Bool,
        secondContent: String? = nil,
        secondReasoning: String? = nil,
        holdSecond: Bool = false,
        emptyLengthOnFirst: Bool = false,
        emptyLengthOnSecond: Bool = false
    ) {
        stateLock.withLock {
            bodies = []
            configuration = Configuration(
                recoverOnSecondRequest: recoverOnSecondRequest,
                secondContent: secondContent,
                secondReasoning: secondReasoning,
                holdSecond: holdSecond,
                emptyLengthOnFirst: emptyLengthOnFirst,
                emptyLengthOnSecond: emptyLengthOnSecond
            )
        }
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [UnreadableToolProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let body = Self.readBody(from: request)
        let (number, config) = Self.stateLock.withLock { () -> (Int, Configuration) in
            Self.bodies.append(body)
            return (Self.bodies.count, Self.configuration)
        }
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if number == 2 && config.holdSecond { return }

        let choice: [String: Any]
        if (number == 1 && config.emptyLengthOnFirst)
            || (number == 2 && config.emptyLengthOnSecond)
        {
            choice = ["delta": ["content": ""], "finish_reason": "length"]
        } else if number == 1 {
            choice = [
                "delta": ["content": "<tool_call>{\"name\":\"weather\",\"arguments\":{"],
                "finish_reason": "stop"
            ]
        } else if number == 2 && !config.recoverOnSecondRequest {
            choice = [
                "delta": [
                    "content": config.secondContent
                        ?? "<tool_call>{\"name\":\"weather\",\"arguments\":{",
                    "reasoning_content": config.secondReasoning ?? ""
                ],
                "finish_reason": "stop"
            ]
        } else if number == 2 {
            choice = [
                "delta": ["tool_calls": [[
                    "index": 0, "id": "call_recovered", "type": "function",
                    "function": ["name": "weather", "arguments": "{}"]
                ]]],
                "finish_reason": "tool_calls"
            ]
        } else {
            choice = [
                "delta": ["content": "The verified result is ready."],
                "finish_reason": "stop"
            ]
        }
        let chunk = try! JSONSerialization.data(withJSONObject: [
            "choices": [choice],
            "usage": ["prompt_tokens": 40, "completion_tokens": 336, "total_tokens": 376]
        ])
        var stream = Data("data: ".utf8)
        stream.append(chunk)
        stream.append(Data("\n\ndata: [DONE]\n\n".utf8))
        client?.urlProtocol(self, didLoad: stream)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func readBody(from request: URLRequest) -> Data {
        guard let input = request.httpBodyStream else { return request.httpBody ?? Data() }
        input.open()
        defer { input.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while input.hasBytesAvailable {
            let count = input.read(&buffer, maxLength: buffer.count)
            guard count > 0 else { break }
            data.append(buffer, count: count)
        }
        return data
    }
}
