import Foundation
import Testing
@testable import Rapid

/// Issue #2330 — Desktop Chat had no authoritative current-date context. A
/// small model answered "what is the date today" with a training-memory date
/// ("Friday, May 24, 2024") and then claimed it could not know today's date.
///
/// The fix injects the Mac's local date/time/time-zone into the leading
/// system prompt row as a request-time template variable (the established
/// desktop-chat pattern), so the model never has to guess the date or infer it
/// must search for it. These tests pin the pure formatter with an injected
/// clock/time-zone and prove it stays correct across time zones, midnight
/// rollover, and a restored conversation.
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

    @Test("The injected clock/time-zone pins today's local date, time, and zone")
    func pinnedLocalDateTime() {
        let calendar = Self.calendar("America/Los_Angeles")
        let out = ChatViewModel.currentDateTimeContext(
            now: Self.instant("2026-08-25T14:37:00Z"),
            calendar: calendar
        )
        #expect(out.contains("[CURRENT DATE AND TIME]"))
        #expect(out.contains(
            "Today is Tuesday, August 25, 2026. The current local time is 7:37 AM"
        ))
        #expect(out.contains("(PDT, America/Los_Angeles)"))
    }

    @Test("The same instant reads a different local date across time zones")
    func timeZoneChangesTheLocalDate() {
        let instant = Self.instant("2026-08-25T06:00:00Z")
        let la = ChatViewModel.currentDateTimeContext(
            now: instant, calendar: Self.calendar("America/Los_Angeles")
        )
        let tokyo = ChatViewModel.currentDateTimeContext(
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
        let before = ChatViewModel.currentDateTimeContext(
            now: Self.instant("2026-08-25T06:59:59Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        let after = ChatViewModel.currentDateTimeContext(
            now: Self.instant("2026-08-25T07:00:01Z"),
            calendar: Self.calendar("America/Los_Angeles")
        )
        // 23:59 PDT is still Monday Aug 24; 00:00 PDT is Tuesday Aug 25.
        #expect(before.contains("Monday, August 24, 2026"))
        #expect(before.contains("11:59 PM"))
        #expect(after.contains("Tuesday, August 25, 2026"))
        #expect(after.contains("12:00 AM"))
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
        let dateContext = ChatViewModel.currentDateTimeContext(
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
        #expect(content.contains("[CURRENT DATE AND TIME]"))
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
        #expect(system.contains("[CURRENT DATE AND TIME]"))
        #expect(system.contains("Today is "))
    }
}
