import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Tool-loop budget integration", .serialized)
struct ToolLoopBudgetIntegrationTests {
    @Test("three tool rounds end in a tools-disabled synthesis answer")
    func cappedLoopSynthesizesInsteadOfFailing() async throws {
        ToolLoopBudgetProtocol.reset()
        let registry = CountingToolRegistry()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://tool-loop")!,
            session: ToolLoopBudgetProtocol.session()
        )
        let model = ChatViewModel(
            client: client,
            tools: registry,
            persistsConversations: false
        )

        model.send("Research this topic thoroughly", alias: "test-model")
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        #expect(registry.runCount == 3)
        #expect(ToolLoopBudgetProtocol.requestBodies.count == 4)
        #expect(model.messages.last?.content == "Here is the answer from the evidence.")
        #expect(model.messages.last?.status == .complete)
        #expect(model.lastError == nil)

        let finalBody = try #require(ToolLoopBudgetProtocol.requestBodies.last)
        let json = try #require(
            JSONSerialization.jsonObject(with: finalBody) as? [String: Any]
        )
        #expect(json["tools"] == nil)
        let messages = try #require(json["messages"] as? [[String: Any]])
        let system = try #require(messages.first)
        #expect((system["content"] as? String)?.contains("tool-use budget") == true)
    }

    @Test("a batched response cannot execute past the three-call budget")
    func batchedCallsAreCappedIndividually() async throws {
        ToolLoopBudgetProtocol.reset(batched: true)
        let registry = CountingToolRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://tool-loop")!,
                session: ToolLoopBudgetProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("Research several sources", alias: "test-model")
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        #expect(registry.runCount == 3)
        #expect(ToolLoopBudgetProtocol.requestBodies.count == 2)
        #expect(model.messages.last?.content == "Here is the answer from the evidence.")
        let toolRows = model.messages.filter { $0.role == .tool }
        #expect(toolRows.count == 5)
        #expect(toolRows.filter { $0.content.contains("budget exhausted") }.count == 2)
    }
}

@MainActor
private final class CountingToolRegistry: ToolRegistry {
    private(set) var runCount = 0

    let definitions = [
        ToolDefinition(
            name: "lookup",
            description: "Look up evidence",
            parameters: .object(["type": .string("object")])
        )
    ]

    func run(_ call: ToolCall) async -> ToolCallResult {
        runCount += 1
        return ToolCallResult(
            toolCallID: call.id,
            content: "Evidence item \(runCount)"
        )
    }
}

private final class ToolLoopBudgetProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requestBodies: [Data] = []
    nonisolated(unsafe) static var sendsBatchedCalls = false

    static func reset(batched: Bool = false) {
        requestBodies = []
        sendsBatchedCalls = batched
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ToolLoopBudgetProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let body = readBody(from: request)
        Self.requestBodies.append(body)
        let requestNumber = Self.requestBodies.count
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        let stream: String
        if Self.sendsBatchedCalls, requestNumber == 1 {
            let calls = (1...5).map { index in
                "{\"index\":\(index - 1),\"id\":\"call_\(index)\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}"
            }.joined(separator: ",")
            stream = """
            data: {"choices":[{"delta":{"tool_calls":[\(calls)]},"finish_reason":"tool_calls"}]}

            data: [DONE]

            """
        } else if !Self.sendsBatchedCalls, requestNumber <= 3 {
            stream = """
            data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_\(requestNumber)","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

            data: [DONE]

            """
        } else {
            stream = """
            data: {"choices":[{"delta":{"content":"Here is the answer from the evidence."},"finish_reason":"stop"}]}

            data: [DONE]

            """
        }
        client?.urlProtocol(self, didLoad: Data(stream.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private func readBody(from request: URLRequest) -> Data {
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
