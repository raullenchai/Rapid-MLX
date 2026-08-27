import Foundation
import Testing
@testable import Rapid

/// End-to-end cover for the throughput caption: a real ``ChatViewModel``
/// streaming a real SSE response through a fake transport that spends most
/// of the turn on prefill.
///
/// The arithmetic in ``MessageStatsTests`` constructs ``MessageStats``
/// directly, which pins the formula but says nothing about whether anything
/// ever *populates* it. Deleting both `firstTokenAt` assignments in
/// ``ChatViewModel/runOneStream`` leaves every one of those tests green
/// while production quietly reverts to whole-turn rates — the exact defect
/// this change exists to remove. These tests fail in that world, because
/// the only thing they read is what the view model actually persisted.
@MainActor
@Suite("Throughput caption integration", .serialized)
struct ThroughputCaptionIntegrationTests {

    /// Prefill is deliberately the dominant term, mirroring the reported
    /// case: a tool-carrying prompt spent 0.75 s in prefill and then emitted
    /// 8 tokens, and the caption divided by the whole turn.
    @Test("The persisted stats separate prefill from decode on a real stream")
    func streamRecordsTimeToFirstToken() async throws {
        let clock = TestStreamClock()
        var deliveredValues: [ProductValueKind] = []
        PrefillHeavyProtocol.reset(
            prefillDelay: 0.45,
            contentDeltas: 12,
            completionTokens: 12,
            clock: clock
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://prefill")!,
                session: PrefillHeavyProtocol.session(),
                now: clock.now
            ),
            persistsConversations: false,
            onProductValueDelivered: { deliveredValues.append($0) }
        )

        model.send("anything", alias: "test-model")
        for _ in 0..<400 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!model.isStreaming)
        #expect(deliveredValues == [.chatReply])

        let stats = try #require(model.messages.last?.stats)

        // 1. The wiring ran at all. Nil here means nobody stamped the first
        //    token, which is the regression this file exists to catch.
        let ttft = try #require(
            stats.timeToFirstTokenSeconds,
            "no time-to-first-token was recorded — runOneStream did not stamp the first delta"
        )

        // 2. It measured the prefill, not something incidental. The fake
        //    holds the response for 0.45 s before the first delta.
        #expect(abs(ttft - 0.45) < 0.000_001,
                "TTFT must use the fixture's deterministic prefill interval")
        #expect(ttft < stats.elapsedSeconds, "TTFT must fall inside the turn it describes")

        // 3. The reported rate uses the decode window. The whole-turn
        //    arithmetic this replaced would divide by `elapsedSeconds`,
        //    which is dominated by the 0.45 s of prefill — so the two
        //    differ by a wide, jitter-proof margin.
        let rate = try #require(stats.reportedTokensPerSecond)
        let wholeTurnRate = Double(stats.completionTokens ?? 0) / stats.elapsedSeconds
        #expect(
            rate > wholeTurnRate * 1.5,
            "rate \(rate) is not meaningfully above the whole-turn rate \(wholeTurnRate) — prefill is still in the denominator"
        )
    }

    /// The estimate path is the one a non-conforming server falls back to.
    /// It must not resurrect the whole-turn denominator either.
    @Test("A server that reports no usage still separates prefill from decode")
    func estimateAlsoExcludesPrefill() async throws {
        let clock = TestStreamClock()
        PrefillHeavyProtocol.reset(
            prefillDelay: 0.45,
            contentDeltas: 12,
            completionTokens: nil,
            clock: clock
        )
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://prefill")!,
                session: PrefillHeavyProtocol.session(),
                now: clock.now
            ),
            persistsConversations: false
        )

        model.send("anything", alias: "test-model")
        for _ in 0..<400 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        let stats = try #require(model.messages.last?.stats)
        #expect(stats.completionTokens == nil, "fixture must not report usage for this case")
        let ttft = try #require(stats.timeToFirstTokenSeconds)
        #expect(abs(ttft - 0.45) < 0.000_001)

        let estimate = try #require(stats.estimatedTokensPerSecond)
        let wholeTurnEstimate = (Double(stats.charCount) / 4.0) / stats.elapsedSeconds
        #expect(estimate > wholeTurnEstimate * 1.5)
    }

    /// A reasoning turn stamps the clock on the reasoning trace, and records
    /// that it had one.
    @Test("A reasoning turn times the trace, not the prose that follows it")
    func reasoningTurnStampsTheTraceAndIsMarked() async throws {
        ReasoningFirstProtocol.reset()
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://reasoning")!,
                session: ReasoningFirstProtocol.session()
            ),
            persistsConversations: false
        )

        model.send("think about it", alias: "test-model")
        for _ in 0..<400 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }

        let last = try #require(model.messages.last)
        #expect(!last.reasoning.isEmpty, "fixture must emit a reasoning trace")
        let stats = try #require(last.stats)

        // Recorded, and recorded EARLY — the reasoning delta arrives at
        // ~0.30 s, the prose only at ~0.75 s. A TTFT up at the prose would
        // mean the reasoning lane never stamped the clock.
        let ttft = try #require(stats.timeToFirstTokenSeconds)
        #expect(ttft < 0.6, "TTFT \(ttft)s looks like it was taken at the prose, not the reasoning trace")

        // And the turn is marked, so the char-count estimate knows its
        // numerator and denominator disagree.
        #expect(stats.emittedReasoning)
        #expect(stats.estimatedTokensPerSecond == nil)
    }

    /// The third lane, and the one with no coverage until now.
    ///
    /// ``Event/firstToken`` fires on content, reasoning, OR tool calls.
    /// Every other test in this file drives a stream that opens with
    /// content or reasoning, so deleting just the `tool_calls` clause from
    /// the `generated` predicate leaves all of them green — the clause
    /// would be load-bearing in production and unprotected in the suite,
    /// which is the shape of a guard that cannot fail.
    ///
    /// A tool-first turn is the exact case the clause exists for, and it is
    /// also the case that motivated this change: the reported defect was a
    /// tool-carrying prompt whose prefill was 93 % of the turn. With the
    /// clause gone the clock starts at the prose instead, the tool-call
    /// generation lands inside "prefill", and the decode window shrinks to
    /// the prose alone — a rate that reads plausible and is far too high.
    @Test("A turn whose first output is a tool call times the tool call, not the prose")
    func toolCallFirstStartsTheClock() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://toolfirst")!,
            session: ToolCallFirstProtocol.session()
        )

        var firstTokenAt: ContinuousClock.Instant?
        var order: [String] = []
        let start = ContinuousClock.now
        try await client.send(
            ChatStreamClient.Request(
                alias: "test-model",
                messages: [ChatMessage(role: .user, content: "weather?")]
            )
        ) { event in
            switch event {
            case .firstToken(let at):
                if firstTokenAt == nil { firstTokenAt = at }
                order.append("firstToken")
            case .content: order.append("content")
            case .toolCalls: order.append("toolCalls")
            default: break
            }
        }

        let at = try #require(firstTokenAt, "no first-token event on a tool-first stream")
        let offset = start.duration(to: at).seconds

        // The tool-call fragment lands at ~0.30 s, the prose only at
        // ~0.75 s. Anything past the midpoint means the clock skipped the
        // tool-call lane and waited for text.
        #expect(
            offset > 0.2 && offset < 0.55,
            "first token stamped \(offset)s in — the tool-call delta at ~0.30s did not start the clock"
        )
        #expect(order.first == "firstToken", "event order was \(order)")
        #expect(order.contains("content"), "fixture must also emit prose after the tool call")
    }

    /// Where the instant is SAMPLED, as distinct from when it is delivered.
    ///
    /// Every other test here leaves the main actor idle, so `MainActor.run`
    /// is entered immediately and a stamp taken inside it reads the same as
    /// one taken outside. Moving `ContinuousClock.now` back into the hop
    /// would pass all of them — the fix would be unprotected by the tests
    /// written for it.
    ///
    /// So block the main actor straight through the window the first delta
    /// lands in. Delivery necessarily waits; the question is whether the
    /// value carried was read before the wait. If it was not, the number
    /// being called "time to first token" is partly a measurement of how
    /// busy the UI was, and it errs by shrinking the decode window — which
    /// inflates the rate, the same direction as the original defect.
    @Test("The first-token instant is sampled before the main actor is free to deliver it")
    func stampIsSampledBeforeTheActorHop() async throws {
        PrefillHeavyProtocol.reset(prefillDelay: 0.30, contentDeltas: 4, completionTokens: 4)
        let box = InstantBox()
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://hop")!,
            session: PrefillHeavyProtocol.session()
        )
        let request = ChatStreamClient.Request(
            alias: "test-model",
            messages: [ChatMessage(role: .user, content: "hi")]
        )

        let sending = Task.detached {
            try await client.send(request) { event in
                if case .firstToken(let at) = event, box.value == nil {
                    box.value = at
                }
            }
        }

        // Wait until the transport has actually begun before starting the
        // timed block. `Task.detached` only ENQUEUES the send; on a loaded
        // runner it can sit unscheduled long enough that the 0.30 s prefill
        // ends after a block started at enqueue time, and correct production
        // code would then fail. The stream's own start is the reference
        // point, not this line. Waiting here is safe because the transport
        // runs on URLSession's threads, not on the actor being held.
        // Then hold the actor past the end of the whole stream, so there is
        // no arrangement of scheduling in which a hop-side stamp could sneak
        // in early. `Task.sleep` would be the wrong tool and would make the
        // test vacuous: it SUSPENDS, which frees the actor to run exactly the
        // hop this is trying to keep waiting.
        let transportBegan = holdMainActor(
            until: PrefillHeavyProtocol.loadingStarted, thenFor: 0.9
        )
        let released = ContinuousClock.now
        // Awaited first so a transport that failed surfaces its real error
        // rather than the generic timeout below.
        try await sending.value
        try #require(
            transportBegan,
            "the transport never started, so nothing was timed — the stream failed before `startLoading`"
        )

        let at = try #require(box.value, "no first-token event arrived")
        #expect(
            at < released,
            "the stamp was taken after the main actor was released, so it is timing UI contention rather than prefill"
        )
    }
}

/// Carries the instant back out of the `@MainActor` event handler. A local
/// `var` cannot be captured by an escaping closure, and the handler runs
/// while the test itself is blocking the actor, so the value has to outlive
/// the call.
@MainActor
private final class InstantBox {
    var value: ContinuousClock.Instant?
}

/// A lock-protected monotonic clock shared by the fake transport and client.
/// Advancing virtual time at SSE boundaries tests the timestamp plumbing,
/// without asking an overloaded test runner to wake within a narrow window.
private final class TestStreamClock: @unchecked Sendable {
    private let lock = NSLock()
    private var instant = ContinuousClock.now
    private var reads = 0
    private let firstTokenSampled = DispatchSemaphore(value: 0)

    func now() -> ContinuousClock.Instant {
        lock.withLock {
            reads += 1
            // Read one is ChatViewModel's stream start; read two is the SSE
            // parser sampling its first generated delta.
            if reads == 2 { firstTokenSampled.signal() }
            return instant
        }
    }

    func advance(by seconds: TimeInterval) {
        lock.withLock {
            instant = instant.advanced(by: .seconds(seconds))
        }
    }

    func waitUntilFirstTokenIsSampled() -> Bool {
        firstTokenSampled.wait(timeout: .now() + 5) == .success
    }
}

/// Occupies the main actor for real, rather than yielding it: waits for the
/// transport to signal that the stream has begun, then holds the actor for a
/// fixed window.
///
/// Both halves are `noasync` — `DispatchSemaphore.wait` for the same reason as
/// `Thread.sleep`, because blocking a cooperative thread is normally a bug.
/// Here it is the subject of the test: the property being measured is what
/// happens to a timestamp while the actor is unavailable. The annotation
/// propagates only through direct calls, so one synchronous function is both
/// the supported way to express this and an honest marker that the blocking is
/// deliberate. Keeping the wait and the sleep together also guarantees the
/// window starts at the stream's beginning with nothing awaitable in between.
///
/// Returns false if the transport never signalled. The wait is BOUNDED for
/// that case: an unbounded one would, if the send failed before
/// `startLoading`, hold the main actor forever and hang the whole run rather
/// than failing a single test — the worst failure mode available to a test
/// whose entire job is to occupy the actor.
@MainActor
private func holdMainActor(
    until started: DispatchSemaphore, thenFor seconds: TimeInterval
) -> Bool {
    guard started.wait(timeout: .now() + 5) == .success else { return false }
    Thread.sleep(forTimeInterval: seconds)
    return true
}

/// Opens with a tool-call fragment carrying no content and no reasoning,
/// then falls silent, then emits prose. The gap is what makes the two
/// candidate stamps distinguishable.
private final class ToolCallFirstProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ToolCallFirstProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        Thread.sleep(forTimeInterval: 0.30)
        emit("""
        data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",\
        "type":"function","function":{"name":"get_weather","arguments":"{}"}}]}}]}\n\n
        """)
        Thread.sleep(forTimeInterval: 0.45)
        emit("data: {\"choices\":[{\"delta\":{\"content\":\"Sunny.\"}}]}\n\n")
        Thread.sleep(forTimeInterval: 0.10)
        emit("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
        emit("data: [DONE]\n\n")
        client?.urlProtocolDidFinishLoading(self)
    }

    private func emit(_ chunk: String) {
        client?.urlProtocol(self, didLoad: Data(chunk.utf8))
    }

    override func stopLoading() {}
}

/// Emits a reasoning trace first, then visible prose, with a gap between —
/// so a clock stamped on the wrong lane lands measurably late.
private final class ReasoningFirstProtocol: URLProtocol, @unchecked Sendable {
    static func reset() {}

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ReasoningFirstProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        Thread.sleep(forTimeInterval: 0.30)
        emit("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking hard \"}}]}\n\n")
        Thread.sleep(forTimeInterval: 0.45)
        emit("data: {\"choices\":[{\"delta\":{\"content\":\"Answer.\"}}]}\n\n")
        Thread.sleep(forTimeInterval: 0.10)
        emit("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
        emit("data: [DONE]\n\n")
        client?.urlProtocolDidFinishLoading(self)
    }

    private func emit(_ chunk: String) {
        client?.urlProtocol(self, didLoad: Data(chunk.utf8))
    }

    override func stopLoading() {}
}

/// Streams a response whose time is spent almost entirely before the first
/// content delta, so a turn's prefill and decode halves are far enough apart
/// to tell which one a rate was divided by.
private final class PrefillHeavyProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var prefillDelay: TimeInterval = 0.45
    /// Held open between the first delta and the rest, so the decode window
    /// is comfortably above production's 50 ms noise floor no matter how
    /// fast the machine is.
    nonisolated(unsafe) static var decodeWindow: TimeInterval = 0.25
    nonisolated(unsafe) static var contentDeltas = 12
    nonisolated(unsafe) static var completionTokens: Int? = 12
    /// Signalled the moment the transport begins, so a caller can start a
    /// timed window knowing the stream is genuinely underway rather than
    /// merely enqueued.
    nonisolated(unsafe) static var loadingStarted = DispatchSemaphore(value: 0)
    nonisolated(unsafe) static var clock: TestStreamClock?

    static func reset(
        prefillDelay: TimeInterval,
        decodeWindow: TimeInterval = 0.25,
        contentDeltas: Int,
        completionTokens: Int?,
        clock: TestStreamClock? = nil
    ) {
        Self.prefillDelay = prefillDelay
        Self.decodeWindow = decodeWindow
        Self.contentDeltas = contentDeltas
        Self.completionTokens = completionTokens
        Self.clock = clock
        Self.loadingStarted = DispatchSemaphore(value: 0)
    }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [PrefillHeavyProtocol.self]
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        Self.loadingStarted.signal()

        // The prefill. Nothing reaches the view model during this window, so
        // a correctly-wired TTFT lands at or after it.
        if let clock = Self.clock {
            clock.advance(by: Self.prefillDelay)
        } else {
            Thread.sleep(forTimeInterval: Self.prefillDelay)
        }

        // First token, then a deliberate decode window. Emitting every
        // delta back-to-back would leave a decode window of microseconds,
        // which production correctly refuses to rate (the 50 ms noise
        // floor) — so the rate assertions below would fail on a fast
        // runner and pass on a slow one, for reasons having nothing to do
        // with the code under test. The window is staged, not hoped for.
        emit("data: {\"choices\":[{\"delta\":{\"content\":\"word \"}}]}\n\n")
        if let clock = Self.clock {
            guard clock.waitUntilFirstTokenIsSampled() else {
                client?.urlProtocol(self, didFailWithError: URLError(.timedOut))
                return
            }
        }
        if let clock = Self.clock {
            clock.advance(by: Self.decodeWindow)
        } else {
            Thread.sleep(forTimeInterval: Self.decodeWindow)
        }

        for _ in 1..<Self.contentDeltas {
            emit("data: {\"choices\":[{\"delta\":{\"content\":\"word \"}}]}\n\n")
        }
        if let tokens = Self.completionTokens {
            emit("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":900,\"completion_tokens\":\(tokens)}}\n\n")
        } else {
            emit("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
        }
        emit("data: [DONE]\n\n")
        client?.urlProtocolDidFinishLoading(self)
    }

    private func emit(_ chunk: String) {
        client?.urlProtocol(self, didLoad: Data(chunk.utf8))
    }

    override func stopLoading() {}
}
