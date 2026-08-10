import Foundation
import Testing
@testable import Rapid

// File-scope (nonisolated) so the URLProtocol stub can read them without
// crossing the @MainActor boundary of the test suite.
private let kGroundedAnswer =
    "According to the results, the biggest story is the Lumen-Nexus merger."
private let kConfabAnswer =
    "I can't access real-time information, and my knowledge is only up to 2024."

/// Grounded-answer confabulation guard (dogfood-0810 BUG C): when a tool
/// result for THIS turn is on the wire but the model's synthesis still denies
/// having real-time / current data ("my knowledge is only up to 2024"), the
/// loop forces exactly one tools-disabled correction round instead of shipping
/// the confabulated refusal.
@MainActor
@Suite("Grounding confabulation retry", .serialized)
struct GroundingConfabulationRetryTests {
    static let caveat =
        "I can't browse directly, but the search result shows the Lumen-Nexus merger closed."

    @Test("A tool-grounded answer that denies real-time access is re-synthesized once")
    func confabulatedDenialTriggersOneCorrection() async throws {
        ConfabRetryProtocol.reset(firstAnswer: kConfabAnswer)
        let registry = WebSearchStubRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://confab")!,
                session: ConfabRetryProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("What's a major news story from the last week?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        // The final answer is the grounded retry, not the confabulated draft.
        #expect(model.messages.last?.content == kGroundedAnswer)
        #expect(model.messages.last?.status == .complete)
        // Exactly two synthesis requests: the confabulated one, then one
        // correction. (The forced web_search itself is dispatched app-side and
        // is not an HTTP request.)
        #expect(ConfabRetryProtocol.requestBodies.count == 2)
        // The correction round is tools-disabled and carries a failure-specific
        // system instruction.
        let correction = try #require(ConfabRetryProtocol.requestBodies.last)
        let json = try #require(
            JSONSerialization.jsonObject(with: correction) as? [String: Any]
        )
        #expect(json["tools"] == nil)
        let messages = try #require(json["messages"] as? [[String: Any]])
        let systemBlob = messages
            .filter { ($0["role"] as? String) == "system" }
            .compactMap { $0["content"] as? String }
            .joined(separator: "\n")
            .lowercased()
        // Assert a distinctive clause from ``groundingCorrectionPreamble`` so
        // this cannot pass on an unrelated pre-existing system prompt.
        #expect(systemBlob.contains("that is false for this turn"))
    }

    @Test("A grounded answer with no denial is shipped as-is, never retried")
    func groundedAnswerIsNotRetried() async throws {
        ConfabRetryProtocol.reset(firstAnswer: kGroundedAnswer)
        let registry = WebSearchStubRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://confab")!,
                session: ConfabRetryProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("What's a major news story from the last week?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        #expect(model.messages.last?.content == kGroundedAnswer)
        // A clean grounded answer must not spend a second synthesis round.
        #expect(ConfabRetryProtocol.requestBodies.count == 1)
    }

    @Test("A failed/empty correction round restores the original draft, not a blank")
    func failedCorrectionRestoresOriginalDraft() async throws {
        // Request 1 confabulates → correction is forced; request 2 (the
        // correction) comes back empty → the loop must restore the original
        // answer rather than leave the blanked placeholder empty.
        ConfabRetryProtocol.reset(firstAnswer: kConfabAnswer)
        ConfabRetryProtocol.emptySecondResponse = true
        defer { ConfabRetryProtocol.emptySecondResponse = false }
        let registry = WebSearchStubRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://confab")!,
                session: ConfabRetryProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("What's a major news story from the last week?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        // The correction was attempted (two requests) but produced nothing…
        #expect(ConfabRetryProtocol.requestBodies.count == 2)
        // …so the original draft is preserved, and the message is not blank.
        #expect(model.messages.last?.content == kConfabAnswer)
        #expect(model.messages.last?.status == .complete)
    }

    @Test("A disclaimer in front of a grounded answer is preserved end-to-end")
    func caveatedGroundedAnswerIsPreservedThroughTheLoop() async throws {
        ConfabRetryProtocol.reset(firstAnswer: Self.caveat)
        let registry = WebSearchStubRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://confab")!,
                session: ConfabRetryProtocol.session()
            ),
            tools: registry,
            persistsConversations: false
        )

        model.send("What's a major news story from the last week?", alias: "test-model")
        for _ in 0..<300 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        // The caveat carries a refusal phrase ("I can't browse") but also draws
        // on the evidence ("the search result shows …"), so the combined guard
        // must NOT retry it — the answer is preserved verbatim.
        #expect(model.messages.last?.content == Self.caveat)
        #expect(ConfabRetryProtocol.requestBodies.count == 1)
    }
}

@Suite("Ungrounded-refusal detection")
struct UngroundedRefusalDetectionTests {
    @Test(
        "First-person temporal-denial refusals are detected",
        arguments: [
            "I can't access real-time information.",
            // Typographic (curly) apostrophe — normalized before matching.
            "I can\u{2019}t access real-time information.",
            "I cannot provide current data, my knowledge is only up to 2024.",
            "As of my last update, I don't have real-time access.",
            "My knowledge cutoff is 2024, so I can't help with this week's news.",
            "Sorry, I'm not able to browse the internet.",
            "I don't have access to current events.",
            "抱歉，我无法访问实时信息。",
            "我的知识截止到2024年。"
        ]
    )
    func detectsDenials(_ text: String) {
        #expect(ChatViewModel.looksLikeUngroundedRefusal(text))
    }

    @Test(
        "Grounded answers, including third-party cutoff reports and quotes, are not flagged",
        arguments: [
            "According to the results, the biggest story is the Lumen-Nexus merger.",
            "The article reports real-time traffic data updated this morning.",
            "Revenue grew steadily up to 2024 before the merger closed.",
            "The live scoreboard shows the current score is 2-1.",
            "Here are the latest headlines from the search results, with sources.",
            // Third-party cutoff report — must NOT trigger (no first-person).
            "The article explains that the model's knowledge cutoff is 2023.",
            "The vendor says its assistant is unable to browse the web.",
            // Quoted tool-result text — must NOT trigger.
            "The outage report notes users were 'unable to browse' the site.",
            "根据搜索结果，本周最大的新闻是这次合并。",
            "文章说该模型无法联网，只能用训练数据。"
        ]
    )
    func ignoresGroundedAnswers(_ text: String) {
        #expect(!ChatViewModel.looksLikeUngroundedRefusal(text))
    }

    @Test("Empty text is not a refusal")
    func emptyIsNotRefusal() {
        #expect(!ChatViewModel.looksLikeUngroundedRefusal(""))
    }

    @Test(
        "Answers that draw on the tool result are recognized as evidence-backed",
        arguments: [
            "According to the results, the merger closed on Aug 7.",
            "Based on the search result, the score is 2-1.",
            "The full story is here: [report](https://example.com/story)",
            "The article says the deal is done. [source](https://x.com)",
            "根据搜索结果，本周最大的新闻是这次合并。"
        ]
    )
    func recognizesEvidenceBackedAnswers(_ text: String) {
        #expect(ChatViewModel.answerReliesOnEvidence(text))
    }

    @Test("A bare link is not, by itself, proof the model answered from the tool")
    func bareLinkAloneIsNotEvidence() {
        // A refusal that tacks on a raw URL must still be corrected — a bare
        // link is not a substantive, result-referencing answer.
        let text = "I can't access current data; see https://example.com"
        #expect(ChatViewModel.looksLikeUngroundedRefusal(text))
        #expect(!ChatViewModel.answerReliesOnEvidence(text))
    }

    @Test(
        "A disclaimer in front of a grounded answer is NOT treated as a refusal",
        arguments: [
            "I can't browse directly, but the tool results show the merger closed Aug 7.",
            "I don't have real-time data myself, but the search result says it is 2-1.",
            "我无法联网，但根据搜索结果，合并已完成。"
        ]
    )
    func caveatBeforeGroundedAnswerIsSpared(_ text: String) {
        // The refusal phrase is present, but the answer visibly relies on the
        // evidence — so the loop's combined guard must NOT force a correction.
        #expect(ChatViewModel.looksLikeUngroundedRefusal(text))
        #expect(ChatViewModel.answerReliesOnEvidence(text))
    }

    @Test(
        "A refusal wrapped in a vague connector is still corrected, not spared",
        arguments: [
            "According to my knowledge cutoff, I cannot provide current data.",
            "Based on my training, I don't have access to real-time information.",
            "Per the usual disclaimer, I can't access current events."
        ]
    )
    func vagueConnectorRefusalIsNotEvidence(_ text: String) {
        // "according to" / "based on" / "per the" attach to a refusal just as
        // readily as to a citation, so they must NOT count as grounding — the
        // combined guard must still force a correction here.
        #expect(ChatViewModel.looksLikeUngroundedRefusal(text))
        #expect(!ChatViewModel.answerReliesOnEvidence(text))
    }
}

@MainActor
@Suite("Successful-tool-result scoping")
struct SuccessfulToolResultScopingTests {
    private func history(toolStatus: ChatMessage.Status) -> [ChatMessage] {
        [
            ChatMessage(role: .user, content: "What's the latest news?"),
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "s1", name: "web_search", arguments: "{}")]
            ),
            ChatMessage(
                role: .tool,
                content: "web search results: ...",
                status: toolStatus,
                toolCallID: "s1"
            )
        ]
    }

    @Test("A failed tool result this turn does not count as a successful one")
    func failedResultIsNotSuccessful() {
        let h = history(toolStatus: .failed)
        // A tool row IS present…
        #expect(ChatViewModel.carriesToolResultForThisTurn(h))
        // …but it failed, so the correction must not treat it as fetched data.
        #expect(!ChatViewModel.carriesSuccessfulToolResultForThisTurn(h))
    }

    @Test("A completed tool result this turn counts as successful")
    func completedResultIsSuccessful() {
        let h = history(toolStatus: .complete)
        #expect(ChatViewModel.carriesSuccessfulToolResultForThisTurn(h))
    }

    @Test("A completed-but-empty tool result does not count as successful")
    func completedButEmptyResultIsNotSuccessful() {
        let h = [
            ChatMessage(role: .user, content: "What's the latest news?"),
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "s1", name: "web_search", arguments: "{}")]
            ),
            ChatMessage(role: .tool, content: "   ", status: .complete, toolCallID: "s1")
        ]
        #expect(ChatViewModel.carriesToolResultForThisTurn(h))
        #expect(!ChatViewModel.carriesSuccessfulToolResultForThisTurn(h))
    }

    @Test("A successful non-live tool result does not arm the current-data correction")
    func nonLiveResultIsNotCurrentData() {
        let h = [
            ChatMessage(role: .user, content: "Calculate 17 times 23."),
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "c1", name: "calculator", arguments: "{}")]
            ),
            ChatMessage(role: .tool, content: "391", status: .complete, toolCallID: "c1")
        ]
        #expect(!ChatViewModel.carriesSuccessfulToolResultForThisTurn(h))
    }

    @Test("A successful search with no results does not claim current data exists")
    func noResultsIsNotCurrentData() {
        let h = [
            ChatMessage(role: .user, content: "Find the latest frobnicator news."),
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "s1", name: "web_search", arguments: "{}")]
            ),
            ChatMessage(
                role: .tool,
                content: "web_search: no results found for frobnicator",
                status: .complete,
                toolCallID: "s1"
            )
        ]
        #expect(!ChatViewModel.carriesSuccessfulToolResultForThisTurn(h))
    }

    @Test("A live result must match the originating live tool call ID")
    func resultMustMatchLiveCallID() {
        let h = [
            ChatMessage(role: .user, content: "What's the weather?"),
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "w1", name: "weather", arguments: "{}")]
            ),
            ChatMessage(
                role: .tool,
                content: "72 F and sunny",
                status: .complete,
                toolCallID: "other"
            )
        ]
        #expect(!ChatViewModel.carriesSuccessfulToolResultForThisTurn(h))
    }
}

@MainActor
private final class WebSearchStubRegistry: ToolRegistry {
    let definitions = [
        ToolDefinition(
            name: "web_search",
            description: "Search the web",
            parameters: .object(["type": .string("object")])
        )
    ]

    func run(_ call: ToolCall) async -> ToolCallResult {
        ToolCallResult(
            toolCallID: call.id,
            content: "Web search results: the Lumen-Nexus merger closed on Aug 7, 2026."
        )
    }
}

private final class ConfabRetryProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requestBodies: [Data] = []
    nonisolated(unsafe) static var firstAnswer = ""
    nonisolated(unsafe) static var emptySecondResponse = false

    static func reset(firstAnswer: String) {
        requestBodies = []
        self.firstAnswer = firstAnswer
        emptySecondResponse = false
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ConfabRetryProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requestBodies.append(readBody(from: request))
        let requestNumber = Self.requestBodies.count
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        let answer =
            requestNumber == 1
            ? Self.firstAnswer
            : (Self.emptySecondResponse ? "" : kGroundedAnswer)
        let stream = """
        data: {"choices":[{"delta":{"content":\(Self.jsonString(answer))},"finish_reason":"stop"}]}

        data: [DONE]

        """
        client?.urlProtocol(self, didLoad: Data(stream.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    private static func jsonString(_ value: String) -> String {
        let data = try? JSONSerialization.data(
            withJSONObject: [value],
            options: [.withoutEscapingSlashes]
        )
        // Serialize as a 1-element array then strip the brackets to get a
        // correctly-escaped JSON string literal.
        guard let data, let array = String(data: data, encoding: .utf8) else {
            return "\"\""
        }
        return String(array.dropFirst().dropLast())
    }

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
