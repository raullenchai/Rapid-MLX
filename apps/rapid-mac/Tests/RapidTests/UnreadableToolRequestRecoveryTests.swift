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

    @Test("two malformed calls end in a tools-disabled honest answer")
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
        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        #expect(registry.runCount == 0)
        #expect(model.messages.last?.content ==
            "I couldn't fetch current information, so I can't verify the answer.")
        #expect(model.messages.last?.toolCallArtifactSuppressed == false)
        #expect(model.messages.last?.status == .complete)
        #expect(model.lastError == nil)

        let final = try #require(UnreadableToolProtocol.requestBodies.last)
        let json = try #require(JSONSerialization.jsonObject(with: final) as? [String: Any])
        #expect(json["tools"] == nil)
        let messages = try #require(json["messages"] as? [[String: Any]])
        let lastUser = try #require(messages.last)
        #expect(lastUser["role"] as? String == "user")
        #expect((lastUser["content"] as? String)?.contains("Do not guess about live or external facts") == true)
    }

    @Test("a model that keeps printing call syntax gets a plain final answer")
    func repeatedArtifactDoesNotLeak() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false, unreadableOnFinal: true)
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

        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        #expect(model.messages.last?.content ==
            "I couldn't retrieve or verify the information needed for a reliable answer. Try again or choose another model.")
        #expect(model.messages.last?.toolCallArtifactSuppressed == false)
        #expect(model.messages.last?.status == .complete)
        #expect(ChatViewModel.unreadableToolFallbackCopy(for: "东京今天天气怎么样") ==
            ChatViewModel.unreadableToolFallbackChineseCopy)
    }

    @Test("an empty final completion gives a clear fallback")
    func emptyFinalCompletion() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false, finalContent: "")
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
        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        #expect(model.messages.last?.content == ChatViewModel.unreadableToolFallbackCopy)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("prose above a final raw call survives")
    func finalProseSurvivesArtifact() async throws {
        UnreadableToolProtocol.reset(
            recoverOnSecondRequest: false,
            finalContent: "I couldn't verify today's result.\n<｜tool▁calls▁begin｜>"
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
        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        #expect(model.messages.last?.content.contains("I couldn't verify today's result.") == true)
        #expect(model.messages.last?.toolCallArtifactSuppressed == true)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("stopping a pending fallback preserves cancellation")
    func stopPendingFallback() async throws {
        UnreadableToolProtocol.reset(recoverOnSecondRequest: false, holdFinal: true)
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://unreadable-tool")!,
                session: UnreadableToolProtocol.session()
            ),
            tools: RecoveryToolRegistry(),
            persistsConversations: false
        )
        model.send("What happened today?", alias: "test-model")
        for _ in 0..<300 where UnreadableToolProtocol.requestBodies.count < 3 {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(UnreadableToolProtocol.requestBodies.count == 3)
        model.stop()
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(model.messages.last?.errorMessage == "Stopped.")
        #expect(model.messages.last?.content != ChatViewModel.unreadableToolFallbackCopy)
    }
}

@MainActor
private final class RecoveryToolRegistry: ToolRegistry {
    private(set) var runCount = 0

    var definitions: [ToolDefinition] {
        [ToolDefinition(
            name: "lookup",
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
    nonisolated(unsafe) static var requestBodies: [Data] = []
    nonisolated(unsafe) static var recoverOnSecondRequest = false
    nonisolated(unsafe) static var unreadableOnFinal = false
    nonisolated(unsafe) static var finalContent: String?
    nonisolated(unsafe) static var holdFinal = false

    static func reset(
        recoverOnSecondRequest: Bool,
        unreadableOnFinal: Bool = false,
        finalContent: String? = nil,
        holdFinal: Bool = false
    ) {
        requestBodies = []
        self.recoverOnSecondRequest = recoverOnSecondRequest
        self.unreadableOnFinal = unreadableOnFinal
        self.finalContent = finalContent
        self.holdFinal = holdFinal
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
        Self.requestBodies.append(body)
        let number = Self.requestBodies.count
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if number == 3 && Self.holdFinal { return }

        let choice: [String: Any]
        if number == 1 || (number == 2 && !Self.recoverOnSecondRequest) {
            choice = [
                "delta": ["content": "<tool_call>{\"name\":\"lookup\",\"arguments\":{"],
                "finish_reason": "stop"
            ]
        } else if number == 2 {
            choice = [
                "delta": ["tool_calls": [[
                    "index": 0, "id": "call_recovered", "type": "function",
                    "function": ["name": "lookup", "arguments": "{}"]
                ]]],
                "finish_reason": "tool_calls"
            ]
        } else {
            choice = [
                "delta": ["content": Self.finalContent ?? (Self.unreadableOnFinal
                    ? "<tool_call>{\"name\":\"lookup\",\"arguments\":{"
                    : Self.recoverOnSecondRequest
                        ? "The verified result is ready."
                        : "I couldn't fetch current information, so I can't verify the answer.")],
                "finish_reason": "stop"
            ]
        }
        let chunk = try! JSONSerialization.data(withJSONObject: ["choices": [choice]])
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
