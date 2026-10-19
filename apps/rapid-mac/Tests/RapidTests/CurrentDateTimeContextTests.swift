import Foundation
import Testing
@testable import Rapid

/// Date and fixed conversation-start context preserve the prompt prefix.
@Suite("Current date context")
struct CurrentDateTimeContextTests {

    private static func calendar(_ timeZoneID: String) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: timeZoneID)
            ?? TimeZone(secondsFromGMT: 0)!
        return calendar
    }

    private static var iso: ISO8601DateFormatter {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }

    private static func instant(_ string: String) -> Date {
        // Force-unwrap: the test inputs are fixed literals.
        iso.date(from: string)!
    }

    @Test("The injected clock/time-zone pins today's local date and zone")
    func pinnedLocalDate() {
        let calendar = Self.calendar("America/Los_Angeles")
        let out = ChatViewModel.currentDateContext(
            now: Self.instant("2026-08-25T14:37:00Z"),
            calendar: calendar
        )
        #expect(out == """
        [CURRENT DATE]
        Today is Tuesday, August 25, 2026 (America/Los_Angeles).
        """)
    }

    @Test("Conversation start adds exactly one fixed local-time line")
    func pinnedConversationStart() {
        let calendar = Self.calendar("America/Los_Angeles")
        let start = Self.instant("2026-08-25T14:37:00Z")
        let expected = """
        [CURRENT DATE]
        Today is Tuesday, August 25, 2026 (America/Los_Angeles).
        This conversation started on Tuesday, August 25, 2026 at 7:37 AM (local time).
        """
        for now in [start, start.addingTimeInterval(600)] {
            #expect(ChatViewModel.currentDateContext(now: now, calendar: calendar, conversationStart: start) == expected)
        }
    }

    @Test("The date-only preview survives a minute boundary")
    func systemRowIsStableAcrossAMinute() {
        let calendar = Self.calendar("America/Los_Angeles")
        // 07:37:59 and 07:38:01 local — a minute boundary crossed mid-chat,
        // which is the common case for a follow-up question.
        let before = ChatViewModel.currentDateContext(
            now: Self.instant("2026-08-25T14:37:59Z"), calendar: calendar
        )
        let after = ChatViewModel.currentDateContext(
            now: Self.instant("2026-08-25T14:38:01Z"), calendar: calendar
        )
        #expect(before == after,
                "A minute tick must not change the head of the prompt — that is the whole prefix-cache fix; a differing system row re-prefills the entire conversation.")
        // No minute, no AM/PM, no ":" — belt and braces against someone
        // re-adding a "7:38 AM" to this row.
        #expect(!before.contains(":"))
        #expect(!before.contains("AM"))
        #expect(!before.contains("PM"))
        #expect(!before.contains("time"))
    }

    @Test("The day-stable block carries no instant-specific zone abbreviation")
    func systemRowHoldsStillAcrossADaylightSavingTransition() {
        // codex round 4: "PST"/"PDT" flips mid-day at a DST transition, which
        // would re-prefill every open conversation twice a year inside the one
        // block this change exists to hold still. Same day, either side of the
        // 2 a.m. spring-forward, must render identically.
        let calendar = Self.calendar("America/Los_Angeles")
        let beforeSpringForward = Self.instant("2026-03-08T09:30:00Z")  // 01:30 PST
        let afterSpringForward = Self.instant("2026-03-08T11:30:00Z")   // 04:30 PDT
        let before = ChatViewModel.currentDateContext(now: beforeSpringForward, calendar: calendar)
        let after = ChatViewModel.currentDateContext(now: afterSpringForward, calendar: calendar)
        #expect(before == after)
        #expect(!before.contains("PST"))
        #expect(!before.contains("PDT"))
        #expect(before.contains("America/Los_Angeles"))
        // The start-time formatter uses the correct local wall clock.
        #expect(ChatViewModel.clockStampText(at: beforeSpringForward, calendar: calendar).hasSuffix("1:30 AM"))
        #expect(ChatViewModel.clockStampText(at: afterSpringForward, calendar: calendar).hasSuffix("4:30 AM"))
    }

    @Test("Guidance-only suffix on blank prose has no leading blank line")
    func blankProseGuidance() {
        let message = ChatMessage(role: .user, content: "   ", wireSuffix: ChatViewModel.toolGuidance)
        #expect(message.modelContent == ChatViewModel.toolGuidance)
    }

    @Test("A blank trailer is dropped rather than appended")
    func blankTrailerIsDropped() {
        var message = ChatMessage(role: .user, content: "hello", status: .complete)
        message.wireSuffix = "  \n "
        #expect(message.modelContent == "hello")
    }

    @Test("A fixed start survives midnight while today's date rolls over")
    func conversationStartSurvivesMidnight() {
        let calendar = Self.calendar("America/Los_Angeles")
        let start = Self.instant("2026-08-25T06:59:59Z")
        let before = ChatViewModel.currentDateContext(now: start, calendar: calendar, conversationStart: start)
        let after = ChatViewModel.currentDateContext(now: start.addingTimeInterval(2), calendar: calendar, conversationStart: start)
        #expect(before.split(separator: "\n").last == after.split(separator: "\n").last)
        #expect(before.contains("Today is Monday, August 24, 2026"))
        #expect(after.contains("Today is Tuesday, August 25, 2026"))
        #expect(after.hasSuffix("This conversation started on Monday, August 24, 2026 at 11:59 PM (local time)."))
    }

    @Test("Start context is stable across daylight-saving transitions")
    func fixedStartAcrossDaylightSaving() {
        let calendar = Self.calendar("America/Los_Angeles")
        for (beforeText, afterText) in [
            ("2026-03-08T09:30:00Z", "2026-03-08T11:30:00Z"),
            ("2026-11-01T08:30:00Z", "2026-11-01T09:30:00Z")
        ] {
            let start = Self.instant(beforeText)
            let before = ChatViewModel.currentDateContext(now: start, calendar: calendar, conversationStart: start)
            let after = ChatViewModel.currentDateContext(now: Self.instant(afterText), calendar: calendar, conversationStart: start)
            #expect(before == after)
            #expect(!before.contains("PST"))
            #expect(!before.contains("PDT"))
        }
    }

    @Test("The same instant reads a different local date across time zones")
    func timeZoneChangesTheLocalDate() {
        let instant = Self.instant("2026-08-25T06:00:00Z")
        let la = ChatViewModel.currentDateContext(
            now: instant, calendar: Self.calendar("America/Los_Angeles")
        )
        let tokyo = ChatViewModel.currentDateContext(
            now: instant, calendar: Self.calendar("Asia/Tokyo")
        )
        // 06:00Z is still August 24 in Los Angeles but already August 25 in
        // Tokyo — the injected zone is authoritative, not UTC.
        #expect(la.contains("Monday, August 24, 2026"))
        #expect(tokyo.contains("Tuesday, August 25, 2026"))
        #expect(la.contains("America/Los_Angeles"))
        #expect(tokyo.contains("Asia/Tokyo"))
    }

    @Test("The context rolls over to a new date across local midnight")
    func dateRollsOverAtLocalMidnight() {
        let before = ChatViewModel.currentDateContext(
            now: Self.instant("2026-08-25T06:59:59Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        let after = ChatViewModel.currentDateContext(
            now: Self.instant("2026-08-25T07:00:01Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        // 23:59 PDT is still Monday Aug 24; 00:00 PDT is Tuesday Aug 25.
        // Day granularity means a conversation spanning local midnight
        // re-prefills exactly once — the accepted price of #2330's guarantee.
        #expect(before.contains("Monday, August 24, 2026"))
        #expect(after.contains("Tuesday, August 25, 2026"))
        #expect(before != after)
    }

    @Test("Date context merges into a restored conversation's single system row")
    func restoredConversationMergesIntoOneSystemRow() {
        // A restored conversation already carries a leading app/system row.
        let restored = ChatMessage(
            role: .system,
            content: "App system context", status: .complete
        )
        let user = ChatMessage(role: .user, content: "Hello", status: .complete)
        let dateContext = ChatViewModel.currentDateContext(
            now: Self.instant("2026-08-25T14:37:00Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        let result = ChatViewModel.addingInstructionLayers(
            to: [restored, user],
            dateContext: dateContext,
            global: "",
            conversation: ""
        )

        #expect(result.filter { $0.role == .system }.count == 1,
                "restored + date context must stay one system row")
        let content = result.first?.content ?? ""
        #expect(content.contains("[CURRENT DATE]"))
        #expect(content.contains("Today is Tuesday, August 25, 2026"))
        #expect(content.contains("App system context"))
        #expect(result.last?.id == user.id)
    }
}

/// Wire-side capture so the production `send` path is exercised with the real
/// `ChatViewModel` and the date context is asserted to actually reach the
/// request body's single system row (not just a helper in isolation).
///
/// The stored body is written on the `URLProtocol` loading thread and read from
/// the main actor after the stream completes, so it is guarded by an ``NSLock``
/// (the pattern other test captures here use) rather than an unsynchronized
/// static — the write/read happen on different threads.
/// Lock-guarded body store. `URLProtocol` writes on its loading thread and the
/// main actor reads afterward, so the pair is boxed behind ``@unchecked
/// Sendable`` (Swift 6 rejects a bare lock-protected mutable global); all access
/// funnels through the lock.
private final class BodyStore: @unchecked Sendable {
    private let lock = NSLock()
    private var body: Data?

    func get() -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return body
    }

    func set(_ value: Data?) {
        lock.lock()
        defer { lock.unlock() }
        body = value
    }
}

private final class DateContextWireCaptureProtocol: URLProtocol, @unchecked Sendable {
    private static let store = BodyStore()

    static var lastRequestBody: Data? { store.get() }

    static func reset() { store.set(nil) }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [DateContextWireCaptureProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.store.set(Self.bodyData(from: request))
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let body = """
        data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n
        data: [DONE]\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
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
            let count = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.read(pointer.baseAddress!, maxLength: pointer.count)
            }
            if count > 0 { data.append(buffer, count: count) }
            if count == 0 { return data }
            if count < 0 { return nil }
        }
    }
}

@MainActor
@Suite("Current date context on the wire")
struct CurrentDateContextWireTests {

    @Test("A send puts the current-date block in the wire system message")
    func sendIncludesCurrentDateOnWire() async throws {
        DateContextWireCaptureProtocol.reset()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://date-context")!,
                session: DateContextWireCaptureProtocol.session()
            ),
            persistsConversations: false
        )

        model.send("what is the date today", alias: "test-model")
        for _ in 0..<200 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(!model.isStreaming)
        let body = try #require(DateContextWireCaptureProtocol.lastRequestBody)
        let json = try #require(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )
        let messages = try #require(json["messages"] as? [[String: Any]])
        #expect(messages.filter { $0["role"] as? String == "system" }.count == 1)
        let system = try #require(messages.first?["content"] as? String)
        #expect(system.contains("[CURRENT DATE]"))
        #expect(system.contains("Today is "))
        let row = try #require(model.messages.first { $0.role == .user })
        #expect(system.hasSuffix("This conversation started on \(ChatViewModel.clockStampText(at: row.createdAt)) (local time)."))
        #expect(messages.map { $0["role"] as? String } == ["system", "user"])
        #expect(messages[1]["content"] as? String == "what is the date today")
        #expect(!String(decoding: body, as: UTF8.self).contains("(Local time:"))
    }

}
