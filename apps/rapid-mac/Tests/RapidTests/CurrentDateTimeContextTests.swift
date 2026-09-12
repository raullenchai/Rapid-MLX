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
        Today is Tuesday, August 25, 2026 (PDT, America/Los_Angeles).
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

    @Test("The clock trailer pins the wall clock of the instant it stamps")
    func clockContextPinsTheInstant() {
        let out = ChatViewModel.clockContext(
            at: Self.instant("2026-08-25T14:37:00Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        #expect(out == """
        [CURRENT LOCAL TIME]
        The current local time is 7:37 AM (PDT, America/Los_Angeles).
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
        #expect(stamped[1].modelContent.hasPrefix("first question\n\n[CURRENT LOCAL TIME]"))
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
        message.wireSuffix = "[CURRENT LOCAL TIME]\nThe current local time is 7:37 AM."

        let wire = message.modelContent
        let extractIndex = try #require(wire.range(of: "TOTAL DUE 1,204.55"))
        let trailerIndex = try #require(wire.range(of: "[CURRENT LOCAL TIME]"))
        #expect(extractIndex.upperBound < trailerIndex.lowerBound,
                "The trailer must follow the extract, or the prompt diverges before the document and the cache cannot reuse it.")
        #expect(wire.hasPrefix("what is the total?"))
    }

    @Test("An empty-prose row gains no leading blank line from the trailer")
    func blankPromptDoesNotGainLeadingNewlines() {
        var message = ChatMessage(role: .user, content: "   ", status: .complete)
        message.wireSuffix = "[CURRENT LOCAL TIME]\nThe current local time is 7:37 AM."
        #expect(message.modelContent.hasPrefix("[CURRENT LOCAL TIME]"))
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
        message.wireSuffix = "[CURRENT LOCAL TIME]\nThe current local time is 7:37 AM."
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
        #expect(!system.contains("[CURRENT LOCAL TIME]"))
        // …and it must still be on the wire, on the user turn, so #2330's
        // "the model is told the current time" contract survives the split.
        let user = try #require(
            messages.first { $0["role"] as? String == "user" }?["content"] as? String
        )
        #expect(user.contains("what is the date today"))
        #expect(user.contains("[CURRENT LOCAL TIME]"))
        #expect(user.contains("The current local time is "))
    }
}
