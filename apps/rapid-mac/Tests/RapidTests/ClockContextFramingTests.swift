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

    @Test("User prose stays untouched and the wire trailer is one neutral annotation")
    func neutralWireOnlyAnnotation() throws {
        let typed = "Help me write a short, polite email to my landlord."
        let row = ChatMessage(role: .user, content: typed, createdAt: instant)
        let stamped = ChatViewModel.stampingClockContext(on: [row], calendar: calendar)[0]
        let trailer = try #require(stamped.wireSuffix)
        #expect(stamped.content == typed)
        #expect(stamped.modelContent == typed + "\n\n" + ChatViewModel.clockContext(at: row.createdAt, calendar: calendar))
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
        #expect(system.wireSuffix == nil)
        #expect(rows.dropFirst().allSatisfy { $0.role != .system })
        #expect(rows[2].modelContent == "Hi")
        #expect(rows[2].wireSuffix == nil)
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
            #expect(row.wireSuffix == ChatViewModel.clockContext(at: row.createdAt, calendar: calendar))
            #expect(row.modelContent == row.content + "\n\n" + ChatViewModel.clockContext(at: row.createdAt, calendar: calendar))
        }
        #expect(turnThree[1].wireSuffix != turnThree[3].wireSuffix)
        #expect(turnThree[3].wireSuffix != turnThree[5].wireSuffix)
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
        let decoded = try JSONDecoder().decode([ChatMessage].self, from: JSONEncoder().encode(stamped))
        #expect(decoded.allSatisfy { $0.wireSuffix == nil })
        #expect(decoded.map(\.content) == typed)
        #expect(decoded.map(\.modelContent) == typed)
        #expect(ChatViewModel.stampingClockContext(on: decoded, calendar: calendar).map(\.modelContent) == stamped.map(\.modelContent))
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
        #expect(sameMinute.map(\.modelContent) == baseline.map(\.modelContent))
        let later = newest.addingTimeInterval(45 * 60)
        let retried = ChatViewModel.stampingClockContext(on: sameMinute, calendar: calendar, answeringAt: later)
        #expect(retried.prefix(2).map(\.modelContent) == baseline.prefix(2).map(\.modelContent))
        let extra = "\n(Current time: \(ChatViewModel.clockStampText(at: later, calendar: calendar)))"
        #expect(retried[2].wireSuffix == ChatViewModel.clockContext(at: newest, calendar: calendar) + extra)
        #expect(retried[2].modelContent == baseline[2].modelContent + extra)
        #expect(retried.map(\.content) == rows.map(\.content))
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

    @Test("Clock annotation follows attachment extracts without changing their bytes")
    func attachmentOrdering() throws {
        let attachment = try ChatFileAttachment(
            filename: "invoice.pdf", kind: .pdf,
            extractedText: "TOTAL DUE 1,204.55", sourceByteCount: 18
        )
        let row = ChatMessage(role: .user, content: "What is the total?", fileAttachments: [attachment], createdAt: instant)
        let stamped = ChatViewModel.stampingClockContext(on: [row], calendar: calendar)[0]
        #expect(stamped.content == row.content)
        #expect(stamped.modelContent == row.modelContent + "\n\n" + ChatViewModel.clockContext(at: instant, calendar: calendar))
        let extract = try #require(stamped.modelContent.range(of: "TOTAL DUE 1,204.55"))
        let annotation = try #require(stamped.modelContent.range(of: "(Local time:"))
        #expect(extract.upperBound < annotation.lowerBound)
    }
}
