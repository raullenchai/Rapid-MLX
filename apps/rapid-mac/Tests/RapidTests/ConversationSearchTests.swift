import Foundation
import Testing
@testable import Rapid

@Suite("Conversation search")
struct ConversationSearchTests {
    private func conversation(
        title: String,
        messages: [ChatMessage] = [],
        updatedAt: Date = Date(),
        isArchived: Bool = false
    ) -> ChatConversation {
        ChatConversation(
            id: UUID(),
            title: title,
            messages: messages,
            createdAt: updatedAt,
            updatedAt: updatedAt,
            isArchived: isArchived
        )
    }

    @Test("An empty query returns every conversation newest first")
    func emptyQueryReturnsAll() {
        let now = Date()
        let older = conversation(title: "Older", updatedAt: now.addingTimeInterval(-60))
        let archived = conversation(title: "Archived", updatedAt: now, isArchived: true)

        let results = ConversationSearch.results(in: [older, archived], matching: "  ")

        #expect(results.map(\.id) == [archived.id, older.id])
    }

    @Test("Search matches title and visible transcript text without case sensitivity")
    func matchesTitleAndTranscript() {
        let titleMatch = conversation(title: "Swift Concurrency Notes")
        let bodyMatch = conversation(
            title: "Untitled",
            messages: [
                ChatMessage(role: .user, content: "How should the prefix CACHE work?"),
                ChatMessage(role: .assistant, content: "Use an actor for serialization."),
            ]
        )
        let miss = conversation(title: "Image generation")

        #expect(
            ConversationSearch.results(
                in: [miss, bodyMatch, titleMatch],
                matching: "swift"
            ).map(\.id) == [titleMatch.id]
        )
        #expect(
            ConversationSearch.results(
                in: [miss, bodyMatch, titleMatch],
                matching: "PREFIX cache"
            ).map(\.id) == [bodyMatch.id]
        )
    }

    @Test("Invisible system and tool payloads are not searchable")
    func ignoresInvisiblePayloads() {
        let conversation = conversation(
            title: "Weather",
            messages: [
                ChatMessage(role: .system, content: "internal-orchid"),
                ChatMessage(role: .tool, content: "tool-orchid"),
            ]
        )

        #expect(ConversationSearch.results(in: [conversation], matching: "orchid").isEmpty)
    }

    @Test("Date sections cover recent, monthly, and older history")
    func groupsByDate() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try #require(TimeZone(secondsFromGMT: 0))
        let now = try #require(
            calendar.date(from: DateComponents(year: 2026, month: 8, day: 12, hour: 12))
        )
        let today = conversation(title: "Today", updatedAt: now)
        let yesterday = conversation(
            title: "Yesterday",
            updatedAt: try #require(calendar.date(byAdding: .day, value: -1, to: now))
        )
        let week = conversation(
            title: "Week",
            updatedAt: try #require(calendar.date(byAdding: .day, value: -5, to: now))
        )
        let month = conversation(
            title: "Month",
            updatedAt: try #require(calendar.date(byAdding: .day, value: -20, to: now))
        )
        let older = conversation(
            title: "Older",
            updatedAt: try #require(calendar.date(byAdding: .day, value: -45, to: now))
        )

        let sections = ConversationSearch.sections(
            for: [month, older, week, today, yesterday],
            now: now,
            calendar: calendar
        )

        #expect(sections.map(\.bucket) == [
            .today, .yesterday, .previous7Days, .previous30Days, .older,
        ])
        #expect(sections.flatMap(\.conversations).map(\.id) == [
            today.id, yesterday.id, week.id, month.id, older.id,
        ])
    }

}
