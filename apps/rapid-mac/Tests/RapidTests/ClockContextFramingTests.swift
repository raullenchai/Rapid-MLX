import Foundation
import Testing
@testable import Rapid

@Suite("Clock context framing")
struct ClockContextFramingTests {
    private var calendar: Calendar {
        var value = Calendar(identifier: .gregorian)
        value.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        return value
    }
    private var instant: Date {
        ISO8601DateFormatter().date(from: "2026-10-18T21:14:00Z")!
    }
    private func request(_ rows: [ChatMessage], at now: Date) -> [ChatMessage] {
        ChatViewModel.addingInstructionLayers(
            to: rows,
            dateContext: ChatViewModel.currentDateContext(
                now: now, calendar: calendar,
                conversationStart: ChatViewModel.conversationStart(in: rows)
            ),
            global: "", conversation: ""
        )
    }
    private func encoded(_ rows: [ChatMessage]) throws -> [[String: Any]] {
        let request = ChatStreamClient.Request(alias: "test", messages: rows, supportsImageInput: true)
        return try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(request.messages)) as? [[String: Any]])
    }

    @Test("Start is the first user in history order, not the oldest timestamp or latest turn")
    func firstUserStart() {
        let first = ChatMessage(role: .user, content: "first", createdAt: instant)
        let earlier = ChatMessage(role: .assistant, content: "context", createdAt: instant.addingTimeInterval(-600))
        let second = ChatMessage(role: .user, content: "second", createdAt: instant.addingTimeInterval(-60))
        #expect(ChatViewModel.conversationStart(in: [earlier, first, second]) == first.createdAt)
        #expect(ChatViewModel.conversationStart(in: []) == nil)
        #expect(ChatViewModel.conversationStart(in: [earlier, ChatMessage(role: .system, content: "system")]) == nil)
    }

    @Test("Two-turn wire shape has one stable system clock and untouched user prose")
    func wireShapeAndPrefix() throws {
        let first = ChatMessage(role: .user, content: "你好", createdAt: instant)
        let answer = ChatMessage(role: .assistant, content: "你好！", createdAt: instant.addingTimeInterval(30))
        let second = ChatMessage(role: .user, content: "明天天气？", createdAt: instant.addingTimeInterval(600))
        let one = try encoded(request([first], at: instant))
        let two = try encoded(request([first, answer, second], at: second.createdAt))
        #expect(two.map { $0["role"] as? String } == ["system", "user", "assistant", "user"])
        let system = try #require(two[0]["content"] as? String)
        #expect(system == """
        [CURRENT DATE]
        Today is Sunday, October 18, 2026 (America/Los_Angeles).
        This conversation started on Sunday, October 18, 2026 at 2:14 PM (local time).
        """)
        #expect(two[1]["content"] as? String == first.content)
        #expect(two[2]["content"] as? String == answer.content)
        #expect(two[3]["content"] as? String == second.content)
        #expect(!String(decoding: try JSONSerialization.data(withJSONObject: two), as: UTF8.self).contains("(Local time:"))
        #expect(two.count > one.count)
        #expect(try JSONSerialization.data(withJSONObject: Array(two.prefix(one.count)), options: [.sortedKeys]) == JSONSerialization.data(withJSONObject: one, options: [.sortedKeys]))
    }

    @Test("Restoration preserves the start and encoded prompt; editing the first row changes it")
    func restoredConversation() throws {
        let rows = [ChatMessage(role: .user, content: "Please draft an email.", createdAt: instant),
            ChatMessage(role: .assistant, content: "Sure.", createdAt: instant),
            ChatMessage(role: .user, content: "Make it warmer.", createdAt: instant.addingTimeInterval(600))]
        var transient = rows
        transient[0].wireSuffix = "request-only guidance"
        let data = try JSONEncoder().encode(transient)
        #expect(!String(decoding: data, as: UTF8.self).contains("wireSuffix"))
        let decoded = try JSONDecoder().decode([ChatMessage].self, from: data)
        #expect(decoded.allSatisfy { $0.wireSuffix == nil })
        #expect(decoded.map(\.content) == rows.map(\.content))
        #expect(decoded.map(\.modelContent) == rows.map(\.modelContent))
        #expect(ChatViewModel.conversationStart(in: decoded) == instant)
        #expect(request(decoded, at: instant).map(\.modelContent) == request(rows, at: instant).map(\.modelContent))
        let edited = [ChatMessage(role: .user, content: "Edited question", createdAt: instant.addingTimeInterval(600))]
        #expect(request(edited, at: instant)[0].content != request(rows, at: instant)[0].content)
    }

    @Test("User attachment extracts and blank prose receive no clock")
    func attachmentOrdering() throws {
        let attachment = try ChatFileAttachment(filename: "invoice.pdf", kind: .pdf,
            extractedText: "TOTAL DUE 1,204.55", sourceByteCount: 18)
        for prose in ["What is the total?", "   "] {
            let row = ChatMessage(role: .user, content: prose, fileAttachments: [attachment], createdAt: instant)
            let rows = request([row], at: instant)
            #expect(rows[1].content == prose)
            #expect(rows[1].modelContent == row.modelContent)
            #expect(rows[1].modelContent.contains("TOTAL DUE 1,204.55"))
            #expect(!rows[1].modelContent.contains("local time"))
            #expect(try encoded(rows)[1]["content"] as? String == row.modelContent)
        }
    }

    @Test("Image focus stays on the original user row, including inherited focus")
    func imageFocusUsesOriginalIndices() throws {
        let image = try ChatImageAttachment(filename: "test.png", mimeType: "image/png", data: Data("image".utf8))
        let first = ChatMessage(role: .user, content: "describe", imageAttachments: [image], createdAt: instant)
        for history in [[first], [first, ChatMessage(role: .assistant, content: "answer"), ChatMessage(role: .user, content: "more")]] {
            let rows = request(history, at: instant)
            let request = ChatStreamClient.Request(alias: "test", messages: rows, supportsImageInput: true)
            #expect(request.imageMessageID == first.id)
            let wire = try encoded(rows)
            #expect(wire.count == rows.count)
            let parts = try #require(wire[1]["content"] as? [[String: Any]])
            #expect(parts.first?["text"] as? String == "describe")
            #expect(parts.last?["type"] as? String == "image_url")
        }
    }

    @MainActor
    @Test("System start context counts in the trim budget without moving to the latest retained turn")
    func systemContextBudget() {
        let first = ChatMessage(role: .user, content: String(repeating: "old context ", count: 100), createdAt: instant)
        let latest = ChatMessage(role: .user, content: "latest", createdAt: instant.addingTimeInterval(600))
        let rows = request([first, ChatMessage(role: .assistant, content: "answer"), latest], at: latest.createdAt)
        let budget = rows.reduce(0) { $0 + max(1, TokenEstimate.tokens(in: $1.modelContent)) }
        #expect(ChatViewModel.trimMessagesForContextWindow(rows, contextWindow: budget, keepFraction: 1).count == rows.count)
        let trimmed = ChatViewModel.trimMessagesForContextWindow(rows, contextWindow: budget - 1, keepFraction: 1)
        #expect(trimmed.count < rows.count)
        #expect(!trimmed.contains { $0.id == first.id })
        #expect(trimmed.first?.content == rows.first?.content)
        #expect(trimmed.last?.id == latest.id)
    }
}
