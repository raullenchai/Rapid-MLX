import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Tool-loop budget integration", .serialized)
struct ToolLoopBudgetIntegrationTests {
    private static let synthesisInstruction = """
    The tool-use budget for this turn is exhausted. Do not request or describe any more tool calls. Answer the user's question now using the evidence already present in the conversation. If that evidence is insufficient, say what remains uncertain.
    """

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
        #expect(system["role"] as? String == "system")
        #expect((system["content"] as? String)?.contains("tool-use budget") == false)
        let synthesis = try #require(messages.last)
        #expect(synthesis["role"] as? String == "user")
        #expect(synthesis["content"] as? String == Self.synthesisInstruction)
        #expect((synthesis["content"] as? String)?.contains("(Local time:") == false)
        #expect(messages[messages.count - 2]["role"] as? String == "tool")
        // Clock rows precede the real prompt; the existing final synthesis
        // control row stays last and receives no clock or tool guidance.
        let promptIndex = try #require(messages.firstIndex { ($0["content"] as? String) == "Research this topic thoroughly" })
        #expect(promptIndex > 0)
        #expect((messages[promptIndex - 1]["content"] as? String)?.hasPrefix("(Local time:") == true)
        #expect(model.messages.filter { $0.role == .user }.map(\.content) == ["Research this topic thoroughly"])
        #expect(!model.messages.contains { $0.content.contains(Self.synthesisInstruction) })
    }

    nonisolated private static let artifact = """
    <tool_call>
    <function=lookup>
    <parameter=q>
    x
    </parameter>
    </function>
    </tool_call>
    """

    @Test(
        "Pure final-round markup fails without rendering markup, sources, or triggering correction",
        arguments: ["lookup", "web_search"], ["x", "I can't access real-time information."]
    )
    func finalArtifactFails(toolName: String, query: String) async throws {
        let artifact = Self.artifact.replacingOccurrences(of: "\nx\n", with: "\n" + query + "\n")
        ToolLoopBudgetProtocol.reset(finalAnswers: [artifact], toolName: toolName)
        let model = makeModel(toolName: toolName)
        try await sendAndWait(model)

        let final = try #require(model.messages.last)
        #expect(final.role == .assistant)
        #expect(final.content.isEmpty)
        #expect(final.status == .failed)
        #expect(final.failureKind == .toolFailed)
        #expect(final.errorMessage == ChatViewModel.toolRoundCapMessage(cap: 3))
        #expect(!final.content.contains("<tool_call>"))
        #expect(!final.content.contains("Sources:"))
        #expect(model.lastError == final.errorMessage)
        #expect(ToolLoopBudgetProtocol.requestBodies.count == 4)
        if toolName == "web_search" {
            #expect(model.messages.contains { $0.role == .tool && $0.content.contains("https://example.com/evidence") })
        }
    }

    // Retain coverage for single-line parameter values as well.
    nonisolated private static let trailingArtifact = """
    <tool_call>
    <function=lookup>
    <parameter=q>x</parameter>
    </function>
    </tool_call>
    """

    // Matches both the whole-content and the trailing detector; the prose
    // above it must still win.
    nonisolated private static let trailingDeepSeekMarker = "<\u{FF5C}tool\u{2581}calls\u{2581}begin\u{FF5C}>"

    @Test(
        "Final-round prose survives a trailing call with the suppression caption",
        arguments: [artifact, trailingArtifact, trailingDeepSeekMarker]
    )
    func finalArtifactKeepsProse(tail: String) async throws {
        ToolLoopBudgetProtocol.reset(finalAnswers: ["Here is the list.\n\n" + tail])
        let model = makeModel()
        try await sendAndWait(model)

        let final = try #require(model.messages.last)
        #expect(final.content == "Here is the list.\n\n" + tail)
        #expect(ChatMessage.proseAboveSuppressedToolCallArtifact(content: final.content) == "Here is the list.")
        #expect(final.toolCallArtifactSuppressed)
        #expect(final.status == .complete)
        #expect(model.lastError == nil)
    }

    @Test(
        "Grounding correction restores a draft on raw markup and keeps corrected prose",
        arguments: [(false, artifact), (true, trailingArtifact), (true, trailingDeepSeekMarker)]
    )
    func groundingCorrectionHandlesArtifacts(hasProse: Bool, tail: String) async throws {
        let draft = "I can't access real-time information."
        let correction = hasProse ? "Here is the list.\n\n" + tail : tail
        ToolLoopBudgetProtocol.reset(finalAnswers: [draft, correction], toolName: "web_search")
        let model = makeModel(toolName: "web_search")
        try await sendAndWait(model)

        #expect(ToolLoopBudgetProtocol.requestBodies.count == 5)
        let final = try #require(model.messages.last)
        #expect(final.content.hasPrefix(hasProse ? "Here is the list." : draft))
        #expect(final.content.contains("Sources:"))
        let visible = hasProse
            ? ChatMessage.proseAboveSuppressedToolCallArtifact(content: final.content)
            : final.content
        #expect(visible?.contains("Here is the list.") == hasProse)
        #expect(visible?.contains("Sources:") == true)
        #expect(visible?.contains("<tool_call>") == false)
        #expect(final.toolCallArtifactSuppressed == hasProse)
        #expect(final.status == .complete)
        #expect(final.errorMessage == nil)
        let body = try #require(ToolLoopBudgetProtocol.requestBodies.last)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(json["tools"] == nil)
        let rows = try #require(json["messages"] as? [[String: Any]])
        #expect((rows.first?["content"] as? String)?.contains(ChatViewModel.groundingCorrectionPreamble) == true)
        #expect((rows.first?["content"] as? String)?.contains(Self.synthesisInstruction) == false)
        #expect(rows.last?["role"] as? String == "user")
        #expect(rows.last?["content"] as? String == Self.synthesisInstruction)
        #expect(model.messages.filter { $0.role == .user }.count == 1)
    }

    @Test("A restored draft that ended in a call keeps that call suppressed")
    func restoredDraftKeepsItsSuppression() async throws {
        let draft = "I can't access real-time information.\n\n" + Self.trailingArtifact
        ToolLoopBudgetProtocol.reset(finalAnswers: [draft, Self.artifact], toolName: "web_search")
        let model = makeModel(toolName: "web_search")
        try await sendAndWait(model)

        #expect(ToolLoopBudgetProtocol.requestBodies.count == 5)
        let final = try #require(model.messages.last)
        #expect(final.toolCallArtifactSuppressed)
        #expect(final.status == .complete)
        let visible = try #require(ChatMessage.proseAboveSuppressedToolCallArtifact(content: final.content))
        #expect(visible.hasPrefix("I can't access real-time information."))
        #expect(visible.contains("Sources:"))
        #expect(!visible.contains("<tool_call>"))
    }

    @Test("Synthesis helper appends a bare user row without changing existing rows", arguments: [false, true])
    func synthesisHelperIsAppendOnly(hasSystem: Bool) {
        let history = (hasSystem ? [ChatMessage(role: .system, content: "System instructions")] : []) + [
            ChatMessage(role: .user, content: "Question"),
            ChatMessage(role: .tool, content: "Evidence", toolCallID: "call_1")
        ]
        let result = ChatViewModel.addingToolBudgetSynthesisPreamble(to: history)
        #expect(Array(result.dropLast()) == history)
        #expect(result.count == history.count + 1)
        #expect(result.last?.role == .user)
        #expect(result.last?.content == Self.synthesisInstruction)
        #expect(result.last?.status == .complete)
    }

    @Test("A tool-free chat keeps a tool-call-shaped answer and gets no synthesis row")
    func toolFreeChatIsNotASynthesisRound() async throws {
        ToolLoopBudgetProtocol.reset(finalAnswers: [Self.artifact])
        let model = ChatViewModel(
            client: ChatStreamClient(baseURL: URL(string: "fake://tool-loop")!, session: ToolLoopBudgetProtocol.session()),
            tools: EmptyToolRegistry(),
            persistsConversations: false
        )
        try await sendAndWait(model)

        #expect(ToolLoopBudgetProtocol.requestBodies.count == 1)
        let final = try #require(model.messages.last)
        #expect(final.content == Self.artifact)
        #expect(final.status == .complete)
        let body = try #require(ToolLoopBudgetProtocol.requestBodies.first)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let rows = try #require(json["messages"] as? [[String: Any]])
        #expect(!rows.contains { ($0["content"] as? String) == Self.synthesisInstruction })
    }

    @Test("A request that does not end on a tool result gets no synthesis row")
    func synthesisHelperSkipsToolFreeRequests() {
        let history = [
            ChatMessage(role: .system, content: "System instructions"),
            ChatMessage(role: .user, content: "Describe this")
        ]
        #expect(ChatViewModel.addingToolBudgetSynthesisPreamble(to: history) == history)
        #expect(ChatViewModel.addingToolBudgetSynthesisPreamble(to: []) == [])
    }

    @Test("Synthetic user row preserves image selection and document focus", arguments: [0, 1, 2])
    func synthesisPreservesImageSelection(focus: Int) throws {
        let image = try ChatImageAttachment(
            filename: "photo.png", mimeType: "image/png", data: Data([0x89, 0x50, 0x4e, 0x47])
        )
        var history = [ChatMessage(role: .user, content: "Describe", imageAttachments: [image])]
        if focus == 1 {
            history.append(ChatMessage(role: .user, content: "Explain further"))
        } else if focus == 2 {
            let file = try ChatFileAttachment(
                filename: "report.txt", kind: .txt, extractedText: "Report", sourceByteCount: 6
            )
            history.append(ChatMessage(role: .user, content: "Read this", fileAttachments: [file]))
        }
        history.append(ChatMessage(role: .tool, content: "Evidence", toolCallID: "call_1"))
        let original = ChatStreamClient.Request(alias: "test-model", messages: history, supportsImageInput: true)
        let synthesis = ChatStreamClient.Request(
            alias: "test-model",
            messages: ChatViewModel.addingToolBudgetSynthesisPreamble(to: history),
            supportsImageInput: true
        )
        #expect(original.imageMessageID == (focus == 2 ? nil : history[0].id))
        #expect(synthesis.imageMessageID == original.imageMessageID)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        #expect(try encoder.encode(Array(synthesis.messages.dropLast())) == encoder.encode(original.messages))
    }

    private func makeModel(toolName: String = "lookup") -> ChatViewModel {
        ChatViewModel(
            client: ChatStreamClient(baseURL: URL(string: "fake://tool-loop")!, session: ToolLoopBudgetProtocol.session()),
            tools: CountingToolRegistry(toolName: toolName),
            persistsConversations: false
        )
    }

    private func sendAndWait(_ model: ChatViewModel) async throws {
        model.send("Research this topic thoroughly", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!model.isStreaming)
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
    @Test("A plain request omits the reader and unsolicited document calls are refused")
    func noFileOmitsAndRefusesDocumentReader() async throws {
        // Keep another tool offered so the loop reaches normal dispatch. The
        // fake emits read_document on round one regardless of its omission.
        ToolLoopBudgetProtocol.reset(documentReads: 1)
        let registry = CountingToolRegistry(
            toolName: "read_document", additionalToolNames: ["lookup"]
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://tool-loop")!,
                session: ToolLoopBudgetProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )
        model.send("Write a story", alias: "test-model")
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!model.isStreaming)
        #expect(ToolLoopBudgetProtocol.requestBodies.count == 2)
        let body = try #require(ToolLoopBudgetProtocol.requestBodies.first)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let tools = try #require(json["tools"] as? [[String: Any]])
        let toolNames = tools.compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
        #expect(toolNames.contains("lookup"))
        #expect(!toolNames.contains("read_document"))
        #expect(registry.runCount == 0)
        let toolRows = model.messages.filter { $0.role == .tool }
        #expect(toolRows.count == 1)
        let refusal = try #require(toolRows.first)
        #expect(refusal.toolCallID == "call_1")
        #expect(refusal.status == .failed)
        #expect(refusal.content.contains("read_document"))
        #expect(refusal.content.contains("isn't available in this conversation"))
        #expect(model.messages.last?.content == "Here is the answer from the evidence.")
        #expect(model.messages.last?.status == .complete)
        #expect(model.lastError == nil)
    }

    @Test("read_document draws on its own budget instead of the general three")
    func documentReadsDoNotSpendTheGeneralBudget() async throws {
        // Paging a long attachment is inherently multi-call. If those calls
        // were charged to maxToolExecutions, four pages would exhaust the
        // budget the search-and-verify loop needs — the exact starvation the
        // separate document quota exists to prevent.
        ToolLoopBudgetProtocol.reset(documentReads: 6)
        let registry = CountingToolRegistry(toolName: "read_document")
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://tool-loop")!,
                session: ToolLoopBudgetProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        let attachment = try ChatFileAttachment(
            filename: "report.txt", kind: .txt,
            extractedText: "Report contents", sourceByteCount: 15
        )
        model.send("Summarize the attached report", alias: "test-model", fileAttachments: [attachment])
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        // All six ran: past three, and none refused for want of budget.
        #expect(registry.runCount == 6)
        #expect(model.messages.last?.content == "Here is the answer from the evidence.")
        #expect(model.messages.last?.status == .complete)
        let toolRows = model.messages.filter { $0.role == .tool }
        #expect(toolRows.allSatisfy { !$0.content.contains("budget exhausted") })
    }

    @Test("The document budget removes the tool and forces synthesis")
    func documentBudgetIsItselfCapped() async throws {
        // The stub requests read_document whenever it is offered. It has no
        // voluntary round cap, so only removing the exhausted tool from the
        // next request can terminate this loop.
        ToolLoopBudgetProtocol.reset(documentReads: .max)
        let registry = CountingToolRegistry(toolName: "read_document")
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://tool-loop")!,
                session: ToolLoopBudgetProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        let attachment = try ChatFileAttachment(
            filename: "report.txt", kind: .txt,
            extractedText: "Report contents", sourceByteCount: 15
        )
        model.send("Read the whole thing", alias: "test-model", fileAttachments: [attachment])
        for _ in 0..<400 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        #expect(registry.runCount == 12)
        #expect(ToolLoopBudgetProtocol.requestBodies.count == 13)
        #expect(model.messages.last?.content == "Here is the answer from the evidence.")
        #expect(model.messages.last?.status == .complete)
        #expect(model.lastError == nil)
        let toolRows = model.messages.filter { $0.role == .tool }
        #expect(toolRows.count == 12)
        #expect(toolRows.allSatisfy { !$0.content.contains("budget exhausted") })

        let finalBody = try #require(ToolLoopBudgetProtocol.requestBodies.last)
        let json = try #require(
            JSONSerialization.jsonObject(with: finalBody) as? [String: Any]
        )
        #expect(json["tools"] == nil)
    }
}

@MainActor
private final class CountingToolRegistry: ToolRegistry {
    private(set) var runCount = 0
    private let toolNames: [String]

    init(toolName: String = "lookup", additionalToolNames: [String] = []) {
        self.toolNames = [toolName] + additionalToolNames
    }

    var definitions: [ToolDefinition] {
        toolNames.map { toolName in
            ToolDefinition(
                name: toolName,
                description: "Look up evidence",
                parameters: .object(["type": .string("object")])
            )
        }
    }

    func run(_ call: ToolCall) async -> ToolCallResult {
        runCount += 1
        return ToolCallResult(
            toolCallID: call.id,
            content: call.function.name == "web_search"
                ? "1. Evidence item \(runCount)\nhttps://example.com/evidence"
                : "Evidence item \(runCount)"
        )
    }
}

private final class ToolLoopBudgetProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requestBodies: [Data] = []
    nonisolated(unsafe) static var sendsBatchedCalls = false
    /// Name the stub emits calls for, and how many rounds it keeps asking.
    nonisolated(unsafe) static var toolName = "lookup"
    nonisolated(unsafe) static var toolRounds = 3

    nonisolated(unsafe) static var finalAnswers: [String] = []
    nonisolated(unsafe) static var finalRoundCount = 0

    static func reset(
        batched: Bool = false, documentReads: Int? = nil,
        finalAnswers: [String] = ["Here is the answer from the evidence."],
        toolName: String = "lookup"
    ) {
        precondition(!finalAnswers.isEmpty)
        self.finalAnswers = finalAnswers
        finalRoundCount = 0
        requestBodies = []
        sendsBatchedCalls = batched
        if let documentReads {
            self.toolName = "read_document"
            toolRounds = documentReads
        } else {
            self.toolName = toolName
            toolRounds = 3
        }
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
        let requestJSON = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        let offersTools = requestJSON?["tools"] != nil
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        let stream: String
        if !offersTools {
            let answer = Self.finalAnswers[min(Self.finalRoundCount, Self.finalAnswers.count - 1)]
            Self.finalRoundCount += 1
            let encoded = try! JSONEncoder().encode(answer)
            let jsonAnswer = String(decoding: encoded, as: UTF8.self)
            stream = """
            data: {"choices":[{"delta":{"content":\(jsonAnswer)},"finish_reason":"stop"}]}

            data: [DONE]

            """
        } else if Self.sendsBatchedCalls, requestNumber == 1 {
            let calls = (1...5).map { index in
                "{\"index\":\(index - 1),\"id\":\"call_\(index)\",\"type\":\"function\",\"function\":{\"name\":\"\(Self.toolName)\",\"arguments\":\"{}\"}}"
            }.joined(separator: ",")
            stream = """
            data: {"choices":[{"delta":{"tool_calls":[\(calls)]},"finish_reason":"tool_calls"}]}

            data: [DONE]

            """
        } else if !Self.sendsBatchedCalls, requestNumber <= Self.toolRounds {
            stream = """
            data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_\(requestNumber)","type":"function","function":{"name":"\(Self.toolName)","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

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
