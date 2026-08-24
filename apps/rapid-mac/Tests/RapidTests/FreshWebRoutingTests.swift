import Foundation
import Testing
@testable import Rapid

@Suite("Fresh web routing")
@MainActor
struct FreshWebRoutingTests {
    @Test("Free-typed weather stays in the native schema-driven tool loop")
    func freeTypedWeatherUsesNativeAutoRouting() async throws {
        NativeRoutingCaptureProtocol.reset()
        let suite = "NativeRoutingTests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let registry = NativeRoutingRegistry()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://native-routing")!,
                session: NativeRoutingCaptureProtocol.session()
            ),
            tools: registry,
            toolDefaults: defaults,
            persistsConversations: false
        )

        model.send(
            "What is the current weather in Tokyo? Use the Weather tool.",
            alias: "qwen3.5-9b-4bit"
        )
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        let body = try #require(NativeRoutingCaptureProtocol.lastRequestBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(json["tool_choice"] as? String == "auto")
        let tools = try #require(json["tools"] as? [[String: Any]])
        let names = tools.compactMap {
            ($0["function"] as? [String: Any])?["name"] as? String
        }
        #expect(Set(names) == ["weather", "web_search"])
        #expect(registry.runCount == 0, "the app must not parse or execute free-typed weather itself")
    }

    @Test("Tool schemas, not prompt keywords, define weather ownership")
    func schemasDefineWeatherOwnership() {
        #expect(WeatherTool.definition.function.description.contains("not web_search"))
        #expect(WebSearchTool.definition.function.description.contains("Do not use it for current weather"))
    }

    @Test("Last-week query uses the previous complete calendar week")
    func concreteLastWeekDates() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 8, hour: 19
        )))
        let query = WebSearchTool.preparedQuery(
            "What's a major news story from the last week?",
            now: now,
            calendar: calendar
        )
        #expect(query.contains("2026-07-26 through 2026-08-01"))
    }

    @Test("Chinese current World Cup query is expanded to completed-event terms")
    func currentWorldCupQueryExpansion() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(calendar.date(from: DateComponents(
            year: 2026, month: 8, day: 8
        )))
        let query = WebSearchTool.preparedQuery(
            "今年世界杯为什么 Spain 夺冠了",
            now: now,
            calendar: calendar
        )
        #expect(query.contains("2026 FIFA World Cup Spain completed final result"))
        #expect(query.contains("winner champion why tactical analysis"))
        let prediction = WebSearchTool.preparedQuery(
            "今年世界杯谁会夺冠？",
            now: now,
            calendar: calendar
        )
        #expect(prediction.contains("2026 FIFA World Cup"))
        #expect(!prediction.contains("completed final result"))
    }

    @Test("Historical and weekend phrases are not rewritten")
    func relativeDateFalsePositives() {
        #expect(WebSearchTool.preparedQuery("the last week of July 2020") == "the last week of July 2020")
        #expect(WebSearchTool.preparedQuery("the last week in July") == "the last week in July")
        #expect(WebSearchTool.preparedQuery("what happened last week in 2020?") == "what happened last week in 2020?")
        #expect(WebSearchTool.preparedQuery("what happened last week compared with 2025?").contains("date range:"))
        #expect(WebSearchTool.preparedQuery("what happened last weekend?") == "what happened last weekend?")
        #expect(WebSearchTool.preparedQuery("上周末有什么活动？") == "上周末有什么活动？")
    }

    @Test("Grounding sources are extracted from formatted search output")
    func sourceExtraction() {
        let output = """
        Web search via DuckDuckGo: query — 2 results

        1. First [story]
           https://example.com/one
           Snippet one

        2. Second story
           https://example.com/two
           Snippet two
        """
        #expect(ChatViewModel.groundingSources(from: output) == [
            .init(title: "First [story]", url: "https://example.com/one"),
            .init(title: "Second story", url: "https://example.com/two")
        ])
    }
}

@MainActor
private final class NativeRoutingRegistry: ToolRegistry {
    var definitions: [ToolDefinition] { [WeatherTool.definition, WebSearchTool.definition] }
    private(set) var runCount = 0

    func run(_ call: ToolCall) async -> ToolCallResult {
        runCount += 1
        return ToolCallResult(toolCallID: call.id, content: "unexpected execution", isError: true)
    }
}

private final class NativeRoutingCaptureProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var lastRequestBody: Data?

    static func reset() { lastRequestBody = nil }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [NativeRoutingCaptureProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.lastRequestBody = Self.bodyData(from: request)
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("""
        data: {"choices":[{"delta":{"content":"Ask the model to choose."},"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.utf8))
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
            let count = buffer.withUnsafeMutableBufferPointer {
                stream.read($0.baseAddress!, maxLength: $0.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return nil }
        }
    }
}
