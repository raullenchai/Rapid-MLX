import Foundation
import Testing
@testable import Rapid

@Suite("Fresh web routing")
struct FreshWebRoutingTests {
    private let enabled: Set<String> = ["web_search", "browse", "weather"]

    @Test("Explicit recent-news prompt forces web search")
    func explicitRecentNews() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What's a major news story from the last week?",
            priorMessages: [],
            enabledToolNames: enabled
        ) == "web_search")
    }

    @Test("Current weather uses the dedicated Weather tool")
    func currentWeatherUsesWeatherTool() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What is the current weather in Tokyo? Use the Weather tool.",
            priorMessages: [],
            enabledToolNames: enabled
        ) == "weather")
        #expect(ChatViewModel.forcedToolForUserTurn(
            "东京今天天气怎么样？",
            priorMessages: [],
            enabledToolNames: enabled
        ) == "weather")
    }

    @Test("Current weather falls back to web search when Weather is disabled")
    func currentWeatherFallsBackToSearch() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What is the current weather in Tokyo?",
            priorMessages: [],
            enabledToolNames: ["web_search"]
        ) == "web_search")
    }

    @Test("Weather substrings do not force the Weather tool")
    func weatherSubstringIsNotAWeatherRequest() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "Why is weathering steel corrosion resistant?",
            priorMessages: [],
            enabledToolNames: enabled
        ) == nil)
        #expect(ChatViewModel.forcedToolForUserTurn(
            "Explain how weather forecasting works.",
            priorMessages: [],
            enabledToolNames: enabled
        ) == nil)
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What is Tokyo's forecast next week?",
            priorMessages: [],
            enabledToolNames: enabled
        ) == nil)
    }

    @Test("Current-weather location extraction is conservative")
    func currentWeatherLocationExtraction() {
        #expect(ChatViewModel.weatherLocation(
            for: "What is the current weather in Tokyo? Use the Weather tool."
        ) == "Tokyo")
        #expect(ChatViewModel.weatherLocation(
            for: "What is the current weather in Springfield, Illinois?"
        ) == "Springfield, Illinois")
        #expect(ChatViewModel.weatherLocation(
            for: "What is the current weather in Washington, D.C.?"
        ) == "Washington, D.C.")
        #expect(ChatViewModel.weatherLocation(for: "东京今天天气怎么样？") == "东京")
        #expect(ChatViewModel.weatherLocation(
            for: "Explain how weather forecasting works."
        ) == nil)
    }

    @Test("Current World Cup research prompt forces web search in Chinese")
    func chineseCurrentWorldCup() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "Codex，你研究下今年世界杯为什么 Spain 夺冠了",
            priorMessages: [],
            enabledToolNames: enabled
        ) == "web_search")
    }

    @Test("Restored-thread follow-up inherits fresh evidence intent")
    func restoredFollowUp() {
        let history = [
            ChatMessage(role: .user, content: "What's a major news story from the last week?"),
            ChatMessage(role: .assistant, content: "I found several sources."),
            ChatMessage(role: .tool, content: "Web search via DuckDuckGo: results")
        ]
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What about technology? Find one concrete story and summarize it.",
            priorMessages: history,
            enabledToolNames: enabled
        ) == "web_search")
        #expect(ChatViewModel.webSearchQuery(
            for: "What about technology? Find one concrete story and summarize it.",
            priorMessages: history
        ).contains("last week"))
        #expect(ChatViewModel.forcedToolForUserTurn(
            "Why is the sky blue?",
            priorMessages: history,
            enabledToolNames: enabled
        ) == nil)
    }

    @Test("Evergreen and casual prompts remain automatic")
    func evergreenPrompts() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What is the capital of France?",
            priorMessages: [],
            enabledToolNames: enabled
        ) == nil)
        #expect(ChatViewModel.forcedToolForUserTurn(
            "Tell me a joke",
            priorMessages: [],
            enabledToolNames: enabled
        ) == nil)
        #expect(ChatViewModel.forcedToolForUserTurn(
            "Explain electric current and a current account",
            priorMessages: [],
            enabledToolNames: enabled
        ) == nil)
    }

    @Test("Follow-up survives a long multi-tool preceding turn")
    func longToolTurnFollowUp() {
        var history = [ChatMessage(
            role: .user,
            content: "What happened in the news last week?"
        )]
        for index in 0..<6 {
            history.append(ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "c\(index)", name: "web_search", arguments: "{}")]
            ))
            history.append(ChatMessage(
                role: .tool,
                content: "result \(index)",
                toolCallID: "c\(index)"
            ))
        }
        #expect(ChatViewModel.forcedToolForUserTurn(
            "What about technology?",
            priorMessages: history,
            enabledToolNames: enabled
        ) == "web_search")
    }

    @Test("Disabled web search is never forced")
    func disabledSearch() {
        #expect(ChatViewModel.forcedToolForUserTurn(
            "latest technology news",
            priorMessages: [],
            enabledToolNames: ["weather"]
        ) == nil)
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
