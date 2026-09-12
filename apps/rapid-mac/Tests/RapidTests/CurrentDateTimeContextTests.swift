import Foundation
import Testing
@testable import Rapid

/// Issue #2330 — Desktop Chat had no authoritative current-date context. A
/// small model answered "what is the date today" with a training-memory date
/// ("Friday, May 24, 2024") and then claimed it could not know today's date.
///
/// The fix injects the Mac's local date/time/time-zone as a request-time
/// template variable (the established desktop-chat pattern), so the model
/// never has to guess the date or infer it must search for it.
///
/// 0.14.1 dogfood split that context in two, and these tests pin the split as
/// much as the original contract. The DATE goes in the leading system row; the
/// wall CLOCK rides each user turn as a wire-only trailer. Putting a
/// minute-resolution string at the head of the prompt rewrote the first tokens
/// of every request the moment the clock ticked, which cost the engine's
/// prompt cache its prefix and re-prefilled the whole conversation — measured
/// at 7.3 s against 0.6 s on a 2.3k-token prompt, and 15–17 s per follow-up
/// with an 8-page PDF attached. So the assertions below care about two things
/// that look pedantic and are not:
///
///   * the system row contains NO minute, and is byte-identical across a
///     minute boundary (`systemRowIsStableAcrossAMinute`);
///   * the clock trailer is derived from each row's immutable `createdAt`, so
///     every request is a strict EXTENSION of the previous one rather than a
///     rewrite of its tail (`stampsEveryUserRowFromItsOwnCreatedAt`).
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

    @Test("The system row carries no wall clock and survives a minute boundary")
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
        // The abbreviation still rides on the message trailer, where it
        // describes a specific instant and so is correct.
        #expect(ChatViewModel.clockContext(at: beforeSpringForward, calendar: calendar).contains("PST"))
        #expect(ChatViewModel.clockContext(at: afterSpringForward, calendar: calendar).contains("PDT"))
    }

    @Test("The clock trailer pins the wall clock of the instant it stamps")
    func clockContextPinsTheInstant() {
        let out = ChatViewModel.clockContext(
            at: Self.instant("2026-08-25T14:37:00Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        // Send-time wording, and the DATE as well as the clock. Both because
        // every user row in the history wears one of these: "the current
        // local time" would have the prompt asserting three contradictory
        // current times, and a bare clock on an old row would attach
        // yesterday's time to today's [CURRENT DATE] across midnight.
        #expect(out == """
        [MESSAGE SENT]
        This message was sent Tuesday, August 25, 2026 at 7:37 AM (PDT, America/Los_Angeles).
        """)
    }

    @Test("Every user row is stamped from its own createdAt; other roles are untouched")
    func stampsEveryUserRowFromItsOwnCreatedAt() {
        let calendar = Self.calendar("America/Los_Angeles")
        let first = ChatMessage(
            role: .user, content: "first question", status: .complete,
            createdAt: Self.instant("2026-08-25T14:37:00Z")
        )
        let answer = ChatMessage(
            role: .assistant, content: "first answer", status: .complete,
            createdAt: Self.instant("2026-08-25T14:37:30Z")
        )
        let second = ChatMessage(
            role: .user, content: "follow-up", status: .complete,
            createdAt: Self.instant("2026-08-25T14:41:00Z")
        )
        let system = ChatMessage(
            role: .system, content: "app context", status: .complete,
            createdAt: Self.instant("2026-08-25T14:36:00Z")
        )

        let stamped = ChatViewModel.stampingClockContext(
            on: [system, first, answer, second], calendar: calendar
        )

        // The OLDER user row keeps its own 7:37 — not "now". This is what
        // makes turn two a strict extension of turn one: had the first row
        // been restamped (or its trailer removed), the stored prefix would no
        // longer match and the engine would re-prefill from that row onward.
        #expect(stamped[1].modelContent.hasSuffix("7:37 AM (PDT, America/Los_Angeles)."))
        #expect(stamped[3].modelContent.hasSuffix("7:41 AM (PDT, America/Los_Angeles)."))
        #expect(stamped[1].modelContent.hasPrefix("first question\n\n[MESSAGE SENT]"))
        // Non-user rows are never stamped: the assistant transcript has to
        // stay byte-identical to what the model produced, and the system row
        // is exactly where the clock must not be.
        #expect(stamped[0].wireSuffix == nil)
        #expect(stamped[2].wireSuffix == nil)
        #expect(stamped[2].modelContent == "first answer")
    }

    @Test("Re-stamping the same history is idempotent")
    func stampingIsIdempotent() {
        let calendar = Self.calendar("America/Los_Angeles")
        let history = [
            ChatMessage(
                role: .user, content: "hello", status: .complete,
                createdAt: Self.instant("2026-08-25T14:37:00Z")
            )
        ]
        let once = ChatViewModel.stampingClockContext(on: history, calendar: calendar)
        let twice = ChatViewModel.stampingClockContext(on: once, calendar: calendar)
        #expect(once.first?.modelContent == twice.first?.modelContent,
                "Stamping twice must not glue on two trailers — the wire text has to be a pure function of the row.")
    }

    @Test("A history with no user row passes through untouched")
    func noUserRowPassesThrough() {
        let history = [
            ChatMessage(role: .system, content: "app context", status: .complete)
        ]
        let stamped = ChatViewModel.stampingClockContext(
            on: history, calendar: Self.calendar("America/Los_Angeles")
        )
        #expect(stamped.first?.wireSuffix == nil)
        #expect(stamped.first?.modelContent == "app context")
    }

    @Test("The clock trailer lands AFTER the attachment extract")
    func trailerFollowsTheDocumentExtract() throws {
        // Ordering is load-bearing, not cosmetic. The document extract is the
        // expensive part of the prompt and the part the cache must find
        // unchanged; a trailer placed in FRONT of it would make two requests
        // diverge BEFORE the document, so none of its tokens could be reused.
        let attachment = try ChatFileAttachment(
            filename: "invoice.pdf",
            kind: .pdf,
            extractedText: "TOTAL DUE 1,204.55",
            sourceByteCount: 18
        )
        var message = ChatMessage(
            role: .user, content: "what is the total?",
            fileAttachments: [attachment], status: .complete
        )
        message.wireSuffix = "[MESSAGE SENT]\nThis message was sent at 7:37 AM."

        let wire = message.modelContent
        let extractIndex = try #require(wire.range(of: "TOTAL DUE 1,204.55"))
        let trailerIndex = try #require(wire.range(of: "[MESSAGE SENT]"))
        #expect(extractIndex.upperBound < trailerIndex.lowerBound,
                "The trailer must follow the extract, or the prompt diverges before the document and the cache cannot reuse it.")
        #expect(wire.hasPrefix("what is the total?"))
    }

    @Test("An empty-prose row gains no leading blank line from the trailer")
    func blankPromptDoesNotGainLeadingNewlines() {
        var message = ChatMessage(role: .user, content: "   ", status: .complete)
        message.wireSuffix = "[MESSAGE SENT]\nThis message was sent at 7:37 AM."
        #expect(message.modelContent.hasPrefix("[MESSAGE SENT]"))
    }

    @Test("A blank trailer is dropped rather than appended")
    func blankTrailerIsDropped() {
        var message = ChatMessage(role: .user, content: "hello", status: .complete)
        message.wireSuffix = "  \n "
        #expect(message.modelContent == "hello")
    }

    @Test("The wire-only trailer is not persisted")
    func trailerIsNotPersisted() throws {
        var message = ChatMessage(role: .user, content: "hello", status: .complete)
        message.wireSuffix = "[MESSAGE SENT]\nThis message was sent at 7:37 AM."
        let round = try JSONDecoder().decode(
            ChatMessage.self, from: JSONEncoder().encode(message)
        )
        #expect(round.wireSuffix == nil,
                "The clock belongs to one request, not to the conversation — persisting it would replay a stale time forever and bloat conversations.json.")
        #expect(round.content == "hello")
        #expect(round.modelContent == "hello")
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
        // The clock trailer rolls over too, from the same instants.
        let clockBefore = ChatViewModel.clockContext(
            at: Self.instant("2026-08-25T06:59:59Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        let clockAfter = ChatViewModel.clockContext(
            at: Self.instant("2026-08-25T07:00:01Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        #expect(clockBefore.contains("11:59 PM"))
        #expect(clockAfter.contains("12:00 AM"))
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
            ambientPreamble: nil,
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
        // The wall clock must NOT be in the system row — it is the first
        // thing in the prompt and a minute tick there costs a full re-prefill.
        #expect(!system.contains("[MESSAGE SENT]"))
        // …and it must still be on the wire, on the user turn, so #2330's
        // "the model is told the current time" contract survives the split.
        let user = try #require(
            messages.first { $0["role"] as? String == "user" }?["content"] as? String
        )
        #expect(user.contains("what is the date today"))
        #expect(user.contains("[MESSAGE SENT]"))
        #expect(user.contains("This message was sent "))
    }
    @MainActor
    @Test("An edited send gets the live clock; regenerate keeps the original ask time")
    func reSendPathsReportTheRightClock() throws {
        // codex asked what happens on retry/regenerate/edit. The answer is
        // not uniform and is worth pinning, because each path reaches the
        // stamper with a different row.
        let calendar = Calendar(identifier: .gregorian)
        let asked = Date(timeIntervalSince1970: 1_757_000_000)     // the original send
        let later = asked.addingTimeInterval(13 * 60)              // thirteen minutes on

        // Regenerate / Retry reuse the SAME row, so its trailer keeps the
        // time the question was asked. Stamping is a pure function of the
        // row, so re-running it later must not move the clock.
        let reused = ChatMessage(role: .user, content: "what time is it?", createdAt: asked)
        let firstPass = ChatViewModel.stampingClockContext(on: [reused], calendar: calendar)
        let secondPass = ChatViewModel.stampingClockContext(on: firstPass, calendar: calendar)
        #expect(firstPass[0].wireSuffix == secondPass[0].wireSuffix,
                "Re-answering the same row must not change what that row renders, or every later turn loses the shared prefix.")
        #expect(try #require(firstPass[0].wireSuffix).contains(
            ChatViewModel.clockContext(at: asked, calendar: calendar)))

        // An edited send mints a NEW row (editUserMessage rewinds and calls
        // send, which defaults createdAt to Date()), so it reports the live
        // clock — and that costs nothing, because the edited text has already
        // broken the shared prefix at that row.
        let freshlySent = ChatMessage(role: .user, content: "what time is it now?", createdAt: later)
        let edited = ChatViewModel.stampingClockContext(on: [reused, freshlySent], calendar: calendar)
        #expect(try #require(edited[1].wireSuffix).contains(
            ChatViewModel.clockContext(at: later, calendar: calendar)))
        #expect(edited[0].wireSuffix != edited[1].wireSuffix,
                "Two rows sent thirteen minutes apart must not render the same clock.")
        // And the earlier row is untouched by the new one arriving: that is
        // the append-only property the prefix cache needs.
        #expect(edited[0].wireSuffix == firstPass[0].wireSuffix)
    }

    @Test("A late regenerate appends the answer time without rewriting the ask")
    func lateRegenerateAppendsTheAnswerTime() throws {
        // codex round 5, blocking: reusing the row meant re-answering "what
        // time is it?" 45 minutes later reported the 45-minute-old clock —
        // worse than the pre-change behaviour, which recomputed per request.
        let calendar = Calendar(identifier: .gregorian)
        let asked = Date(timeIntervalSince1970: 1_757_000_000)
        let muchLater = asked.addingTimeInterval(45 * 60)

        let earlier = ChatMessage(role: .user, content: "hello", createdAt: asked.addingTimeInterval(-3600))
        let reused = ChatMessage(role: .user, content: "what time is it?", createdAt: asked)
        let regenerated = ChatViewModel.stampingClockContext(
            on: [earlier, reused],
            calendar: calendar,
            answeringAt: muchLater
        )
        let trailer = try #require(regenerated[1].wireSuffix)

        // The ask stamp is untouched and still leads: APPEND, not rewrite, so
        // the row's own rendering never changes retroactively.
        #expect(trailer.hasPrefix(ChatViewModel.clockContext(at: asked, calendar: calendar)))
        // And the answer time is actually there, so the model can answer the
        // question it was asked.
        #expect(trailer.contains(ChatViewModel.clockStampText(at: muchLater, calendar: calendar)))
        #expect(trailer.contains("This answer is being generated"))

        // Only the newest user row. An earlier turn renders exactly as it did
        // before, which is the append-only property the prefix cache needs.
        #expect(regenerated[0].wireSuffix
            == ChatViewModel.clockContext(at: earlier.createdAt, calendar: calendar))
    }

    @Test("An ordinary send and a same-minute regenerate add nothing at all")
    func sameMinuteAnswerAddsNothing() throws {
        let calendar = Calendar(identifier: .gregorian)
        let asked = Date(timeIntervalSince1970: 1_757_000_000)
        let row = ChatMessage(role: .user, content: "what time is it?", createdAt: asked)

        // The common path: the row was minted by this very request.
        let fresh = ChatViewModel.stampingClockContext(on: [row], calendar: calendar, answeringAt: asked)
        let unstamped = ChatViewModel.stampingClockContext(on: [row], calendar: calendar)
        #expect(fresh[0].wireSuffix == unstamped[0].wireSuffix,
                "Passing this request's own instant must be byte-identical to passing nothing, or every ordinary send pays for the regenerate feature.")

        // A regenerate twenty seconds on renders the same minute, so it still
        // says nothing new and still costs no prefix.
        let prompt = ChatViewModel.stampingClockContext(
            on: [row], calendar: calendar, answeringAt: asked.addingTimeInterval(20))
        #expect(prompt[0].wireSuffix == unstamped[0].wireSuffix)
    }

    @Test("answeringNowLine speaks only when it has something to say")
    func answeringNowLineContract() {
        let calendar = Calendar(identifier: .gregorian)
        let asked = Date(timeIntervalSince1970: 1_757_000_000)
        #expect(ChatViewModel.answeringNowLine(asked: asked, answeringAt: asked, calendar: calendar) == "")
        #expect(ChatViewModel.answeringNowLine(
            asked: asked, answeringAt: asked.addingTimeInterval(30), calendar: calendar) == "",
                "Sub-minute differences render identically; emitting a line for them would break the prefix for nothing.")
        #expect(!ChatViewModel.answeringNowLine(
            asked: asked, answeringAt: asked.addingTimeInterval(90), calendar: calendar).isEmpty)
    }

    @Test("The system row and the message trailer agree on the date")
    func oneInstantRendersOneDate() {
        // codex round 6, blocking: the send site sampled `Date()` twice, once
        // for the system row and once for the clock trailer, so a request
        // assembled across local midnight could say "Today is the 11th" and
        // "this message was sent … the 12th" in the same prompt. The send site
        // now threads one `requestInstant` into both. This pins the other half
        // — that both renderings agree on the calendar day for a shared
        // instant, including right at the boundary, where a mismatched
        // calendar or time zone between the two formatters would show up.
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try! #require(TimeZone(identifier: "America/Los_Angeles"))
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 11
        components.hour = 23; components.minute = 59; components.second = 59
        let justBefore = calendar.date(from: components)!

        for instant in [justBefore, justBefore.addingTimeInterval(1)] {
            let systemRow = ChatViewModel.currentDateContext(now: instant, calendar: calendar)
            let trailer = ChatViewModel.clockContext(at: instant, calendar: calendar)
            // "Today is <EEEE, MMMM d, yyyy> (zone)." vs "… sent <EEEE,
            // MMMM d, yyyy> at <time> (…)." — the date text is shared.
            let day = systemRow
                .replacingOccurrences(of: "[CURRENT DATE]\nToday is ", with: "")
                .replacingOccurrences(of: " (America/Los_Angeles).", with: "")
            #expect(trailer.contains(day),
                    "The system row says \(day) but the trailer is \(trailer).")
        }
    }

}
