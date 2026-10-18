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
        ChatViewModel.stampingClockContext(
            on: ChatViewModel.addingInstructionLayers(
                to: rows,
                dateContext: ChatViewModel.currentDateContext(now: now, calendar: calendar),
                global: "", conversation: ""
            ), calendar: calendar
        )
    }

    @Test("User prose stays untouched and the separate clock row is one neutral annotation")
    func neutralWireOnlyAnnotation() throws {
        let typed = "Help me write a short, polite email to my landlord."
        let row = ChatMessage(role: .user, content: typed, createdAt: instant)
        let stamped = ChatViewModel.stampingClockContext(on: [row], calendar: calendar)[0]
        let trailer = try #require(stamped.wirePrelude)
        #expect(stamped.content == typed)
        #expect(stamped.modelContent == typed)
        #expect(trailer == "(Local time: Sunday, October 18, 2026 at 2:14 PM)")
        #expect(!trailer.contains("PDT"))
        #expect(!trailer.contains("PST"))
        #expect(!trailer.contains("America/Los_Angeles"))
        #expect(!trailer.contains("[MESSAGE SENT]"))
        #expect(!trailer.lowercased().contains("sent"))
        #expect(!trailer.contains("\n"))
    }

    @Test("Request construction keeps date context in one leading system row")
    func dateOnlySystemRow() throws {
        let rows = request([
            ChatMessage(role: .user, content: "Hello", createdAt: instant),
            ChatMessage(role: .assistant, content: "Hi", createdAt: instant),
            ChatMessage(role: .user, content: "What time is it?", createdAt: instant)
        ], at: instant)
        let system = try #require(rows.first)
        #expect(system.role == .system)
        #expect(system.modelContent.contains("America/Los_Angeles"))
        #expect(system.modelContent == ChatViewModel.currentDateContext(now: instant, calendar: calendar))
        #expect(!system.modelContent.contains("(Local time:"))
        #expect(!system.modelContent.contains("(Current time:"))
        #expect(system.wirePrelude == nil)
        #expect(rows.dropFirst().allSatisfy { $0.role != .system })
        #expect(rows[2].modelContent == "Hi")
        #expect(rows[2].wirePrelude == nil)
    }

    @Test("Follow-ups preserve earlier wire bytes and annotate every user row from createdAt")
    func multiTurnStrictExtension() {
        let first = ChatMessage(role: .user, content: "Draft a cover letter.", createdAt: instant)
        let answer = ChatMessage(role: .assistant, content: "Dear hiring team,", createdAt: instant.addingTimeInterval(30))
        let second = ChatMessage(role: .user, content: "Make it shorter.", createdAt: instant.addingTimeInterval(600))
        let third = ChatMessage(role: .user, content: "What time is it?", createdAt: instant.addingTimeInterval(1200))
        let turnOne = request([first], at: instant)
        let turnTwo = request([first, answer, second], at: second.createdAt)
        let turnThree = request([first, answer, second, answer, third], at: third.createdAt)
        #expect(turnTwo.prefix(turnOne.count).map { Array($0.modelContent.utf8) } == turnOne.map { Array($0.modelContent.utf8) })
        #expect(turnThree.prefix(turnTwo.count).map { Array($0.modelContent.utf8) } == turnTwo.map { Array($0.modelContent.utf8) })
        for row in turnThree where row.role == .user {
            #expect(row.wirePrelude == ChatViewModel.clockContext(at: row.createdAt, calendar: calendar))
            #expect(row.modelContent == row.content)
        }
        #expect(turnThree[1].wirePrelude != turnThree[3].wirePrelude)
        #expect(turnThree[3].wirePrelude != turnThree[5].wirePrelude)
    }

    @Test("Persistence drops wire annotations and restoration reconstructs identical bytes")
    func restoredConversation() throws {
        let typed = ["Please draft an email.", "Thanks!", "Make it warmer."]
        let rows = [
            ChatMessage(role: .user, content: typed[0], createdAt: instant),
            ChatMessage(role: .assistant, content: typed[1], createdAt: instant.addingTimeInterval(30)),
            ChatMessage(role: .user, content: typed[2], createdAt: instant.addingTimeInterval(600))
        ]
        let stamped = ChatViewModel.stampingClockContext(on: rows, calendar: calendar)
        let data = try JSONEncoder().encode(stamped)
        let persisted = String(decoding: data, as: UTF8.self)
        #expect(!persisted.contains("wirePrelude"))
        #expect(!persisted.contains("wireSuffix"))
        #expect(!persisted.contains("(Local time:"))
        let decoded = try JSONDecoder().decode([ChatMessage].self, from: data)
        #expect(decoded.allSatisfy { $0.wirePrelude == nil && $0.wireSuffix == nil })
        #expect(decoded.map(\.content) == typed)
        #expect(decoded.map(\.modelContent) == typed)
        #expect(ChatViewModel.stampingClockContext(on: decoded, calendar: calendar).map(\.wirePrelude) == stamped.map(\.wirePrelude))
    }

    @Test("Retry appends current time only to the newest user row when the minute changes")
    func retryTiming() {
        let newest = instant.addingTimeInterval(600)
        let rows = [
            ChatMessage(role: .user, content: "Hello", createdAt: instant),
            ChatMessage(role: .assistant, content: "Hi", createdAt: instant),
            ChatMessage(role: .user, content: "What time is it?", createdAt: newest)
        ]
        let baseline = ChatViewModel.stampingClockContext(on: rows, calendar: calendar)
        let sameMinute = ChatViewModel.stampingClockContext(on: baseline, calendar: calendar, answeringAt: newest.addingTimeInterval(20))
        #expect(sameMinute == baseline)
        let later = newest.addingTimeInterval(45 * 60)
        let retried = ChatViewModel.stampingClockContext(on: sameMinute, calendar: calendar, answeringAt: later)
        #expect(retried.prefix(2).map(\.modelContent) == baseline.prefix(2).map(\.modelContent))
        let extra = "\n(Current time: \(ChatViewModel.clockStampText(at: later, calendar: calendar)))"
        #expect(retried[2].wirePrelude == ChatViewModel.clockContext(at: newest, calendar: calendar) + extra)
        #expect(retried[2].modelContent == baseline[2].modelContent)
        #expect(retried.map(\.content) == rows.map(\.content))
    }

    @Test("A late retry sends exactly one clock row per user turn and one current-time line, however often it is stamped")
    func retryStampingIsIdempotentOnTheWire() throws {
        let newest = instant.addingTimeInterval(600)
        let later = newest.addingTimeInterval(45 * 60)
        var rows = [
            ChatMessage(role: .user, content: "Hello", createdAt: instant),
            ChatMessage(role: .assistant, content: "Hi", createdAt: instant),
            ChatMessage(role: .user, content: "What time is it?", createdAt: newest)
        ]
        // The send pipeline stamps before and after the trim, around both
        // guidance passes, and every tool round re-runs it on the same array.
        for _ in 0..<3 {
            rows = ChatViewModel.stampingClockContext(on: rows, calendar: calendar, answeringAt: later)
            rows = ChatViewModel.stampingToolGuidance(on: rows, toolsAdvertised: true)
            rows = ChatViewModel.stampingClockContext(on: rows, calendar: calendar, answeringAt: later)
        }
        let request = ChatStreamClient.Request(alias: "test", messages: rows)
        let wire = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(request.messages)) as? [[String: Any]])
        let contents = wire.compactMap { $0["content"] as? String }
        #expect(wire.map { $0["role"] as? String } == ["user", "user", "assistant", "user", "user"])
        #expect(contents.filter { $0.contains("(Local time:") }.count == 2)
        #expect(contents.joined().components(separatedBy: "(Current time:").count == 2)
        #expect(contents[3] == ChatViewModel.clockContext(at: newest, calendar: calendar)
            + "\n(Current time: \(ChatViewModel.clockStampText(at: later, calendar: calendar)))")
        #expect(contents[4] == "What time is it?")
        // Nothing of it is persisted.
        let decoded = try JSONDecoder().decode([ChatMessage].self, from: JSONEncoder().encode(rows))
        #expect(decoded.allSatisfy { $0.wirePrelude == nil && $0.wireSuffix == nil })
    }

    @Test("Retry appends current time during the repeated daylight-saving fall-back hour")
    func retryDuringRepeatedHour() {
        let formatter = ISO8601DateFormatter()
        let asked = formatter.date(from: "2026-11-01T08:30:00Z")!
        let answeringAt = formatter.date(from: "2026-11-01T09:30:00Z")!
        #expect(ChatViewModel.clockStampText(at: asked, calendar: calendar)
            == ChatViewModel.clockStampText(at: answeringAt, calendar: calendar))

        let line = ChatViewModel.answeringNowLine(
            asked: asked, answeringAt: answeringAt, calendar: calendar
        )
        #expect(!line.isEmpty)
        #expect(line == "\n(Current time: Sunday, November 1, 2026 at 1:30 AM)")
    }

    @Test("Clock prelude preserves attachment extracts without changing their bytes")
    func attachmentOrdering() throws {
        let attachment = try ChatFileAttachment(
            filename: "invoice.pdf", kind: .pdf,
            extractedText: "TOTAL DUE 1,204.55", sourceByteCount: 18
        )
        let row = ChatMessage(role: .user, content: "What is the total?", fileAttachments: [attachment], createdAt: instant)
        let stamped = ChatViewModel.stampingClockContext(on: [row], calendar: calendar)[0]
        #expect(stamped.content == row.content)
        #expect(stamped.modelContent == row.modelContent)
        #expect(stamped.modelContent.contains("TOTAL DUE 1,204.55"))
        #expect(!stamped.modelContent.contains("(Local time:"))
        #expect(stamped.wirePrelude == ChatViewModel.clockContext(at: instant, calendar: calendar))
    }
}

@Suite("Clock prelude wire encoding")
struct ClockPreludeWireTests {
    private func encoded(_ rows: [ChatMessage]) throws -> [[String: Any]] {
        let request = ChatStreamClient.Request(alias: "test", messages: rows, supportsImageInput: true)
        return try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(request.messages)) as? [[String: Any]])
    }

    @Test("Clock rows precede real users and preserve the complete encoded prefix")
    func wireShapeAndPrefix() throws {
        let first = ChatMessage(role: .user, content: "first", createdAt: Date(timeIntervalSince1970: 1000))
        let second = ChatMessage(role: .user, content: "second", createdAt: Date(timeIntervalSince1970: 1600))
        let system = ChatMessage(role: .system, content: "fixed date")
        let one = try encoded(ChatViewModel.stampingClockContext(on: [system, first]))
        let two = try encoded(ChatViewModel.stampingClockContext(on: [system, first, ChatMessage(role: .assistant, content: "answer"), second]))
        #expect(two.map { $0["role"] as? String } == ["system", "user", "user", "assistant", "user", "user"])
        #expect(two[1]["content"] as? String == ChatViewModel.clockContext(at: first.createdAt))
        #expect(two[4]["content"] as? String == ChatViewModel.clockContext(at: second.createdAt))
        #expect(two[2]["content"] as? String == "first")
        #expect(two.last { $0["role"] as? String == "user" }?["content"] as? String == "second")
        #expect(try JSONSerialization.data(withJSONObject: Array(two.prefix(one.count)), options: [.sortedKeys]) == JSONSerialization.data(withJSONObject: one, options: [.sortedKeys]))
    }

    @Test("Prelude rows never receive images, including inherited image focus")
    func imageFocusUsesOriginalIndices() throws {
        let image = try ChatImageAttachment(filename: "test.png", mimeType: "image/png", data: Data("image".utf8))
        let first = ChatMessage(role: .user, content: "describe", imageAttachments: [image])
        for history in [[first], [first, ChatMessage(role: .assistant, content: "answer"), ChatMessage(role: .user, content: "more")]] {
            let stamped = ChatViewModel.stampingClockContext(on: history)
            let request = ChatStreamClient.Request(alias: "test", messages: stamped, supportsImageInput: true)
            #expect(request.imageMessageID == first.id)
            let rows = try encoded(stamped)
            #expect(rows[0]["content"] as? String == ChatViewModel.clockContext(at: first.createdAt))
            let parts = try #require(rows[1]["content"] as? [[String: Any]])
            #expect(parts.first?["text"] as? String == "describe")
            #expect(parts.last?["type"] as? String == "image_url")
        }
    }

    @MainActor
    @Test("Clock preludes count toward the trim budget")
    func preludeBudget() {
        let rows = [ChatMessage(role: .system, content: "fixed"),
            ChatMessage(role: .user, content: "first"),
            ChatMessage(role: .assistant, content: "answer"),
            ChatMessage(role: .user, content: "latest")]
        let budget = rows.reduce(0) { $0 + max(1, TokenEstimate.tokens(in: $1.modelContent)) }
        #expect(ChatViewModel.trimMessagesForContextWindow(rows, contextWindow: budget, keepFraction: 1).count == rows.count)
        let stamped = ChatViewModel.stampingClockContext(on: rows)
        #expect(ChatViewModel.trimMessagesForContextWindow(stamped, contextWindow: budget, keepFraction: 1).count < rows.count)
        let preludeBudget = stamped.compactMap(\.wirePrelude).reduce(0) {
            $0 + TokenEstimate.tokens(in: $1) + TokenEstimate.messageFramingTokens
        }
        let fullBudget = budget + preludeBudget
        #expect(ChatViewModel.trimMessagesForContextWindow(stamped, contextWindow: fullBudget, keepFraction: 1).count == rows.count)
        #expect(ChatViewModel.trimMessagesForContextWindow(stamped, contextWindow: fullBudget - 1, keepFraction: 1).count < rows.count)
    }

    @MainActor
    @Test("Blank preludes add neither content nor framing to the trim budget", arguments: ["", " \n\t "])
    func blankPreludeBudget(prelude: String) {
        let rows = [ChatMessage(role: .user, content: "first", wirePrelude: prelude),
            ChatMessage(role: .assistant, content: "answer"),
            ChatMessage(role: .user, content: "latest", wirePrelude: prelude)]
        let budget = rows.reduce(0) { $0 + max(1, TokenEstimate.tokens(in: $1.modelContent)) }
        #expect(ChatViewModel.trimMessagesForContextWindow(rows, contextWindow: budget, keepFraction: 1).count == rows.count)
    }

    @Test("Blank preludes are omitted")
    func blankPrelude() throws {
        let rows = try encoded([ChatMessage(role: .user, content: "hello", wirePrelude: " \n ")])
        #expect(rows.count == 1)
        #expect(rows[0]["content"] as? String == "hello")
    }
}
