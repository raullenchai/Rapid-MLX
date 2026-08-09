import Foundation
import Testing
@testable import Rapid

/// Audit P1 (ChatStreamClient.swift — per-line JSON decode +
/// main-actor hop per SSE event): the coalescer reduces MainActor
/// traffic on the hot path by accumulating ``content`` /
/// ``reasoning_content`` deltas inside a 16 ms window. This suite
/// pins the contract that:
///
///   * First delta of each type surfaces immediately (zero perceived
///     first-token latency for the typing indicator).
///   * Subsequent deltas inside the 16 ms window are coalesced into
///     one MainActor callback.
///   * ``flush`` always emits reasoning BEFORE content (the order
///     rapid-mlx puts them on the wire when both are present).
///   * Independent buffers: a reasoning delta doesn't drain a
///     pending content buffer and vice versa.
@Suite("SSEDeltaCoalescer")
struct SSEDeltaCoalescerTests {

    /// Record fed to the coalescer's onEvent callback so tests can
    /// inspect the event stream in order.
    @MainActor
    final class EventRecorder {
        var events: [ChatStreamClient.Event] = []
        func capture(_ e: ChatStreamClient.Event) { events.append(e) }
    }

    @Test("first content delta surfaces immediately")
    @MainActor
    func first_content_delta_flushes_synchronously() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        await coalescer.appendContent("hello") { recorder.capture($0) }
        // The first-delta-flushes-immediately invariant means the
        // single call above must have produced exactly one .content
        // event with the full payload. A regression that delayed
        // the first flush until the window expired would make this
        // assertion see zero events.
        #expect(recorder.events.count == 1)
        if case .content(let s) = recorder.events[0] {
            #expect(s == "hello")
        } else {
            Issue.record("expected .content first event, got \(recorder.events[0])")
        }
    }

    @Test("first reasoning delta surfaces immediately")
    @MainActor
    func first_reasoning_delta_flushes_synchronously() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        await coalescer.appendReasoning("thinking") { recorder.capture($0) }
        #expect(recorder.events.count == 1)
        if case .reasoning(let r) = recorder.events[0] {
            #expect(r == "thinking")
        } else {
            Issue.record("expected .reasoning first event, got \(recorder.events[0])")
        }
    }

    @Test("rapid-fire content after first flush is coalesced")
    @MainActor
    func rapid_fire_content_coalesces() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        // Burst 50 small deltas back-to-back; the first surfaces
        // immediately, the remaining 49 land inside the 16 ms
        // window and must collapse into a single trailing event
        // via the explicit flush below.
        await coalescer.appendContent("a", onEvent: { recorder.capture($0) })
        for _ in 0..<49 {
            await coalescer.appendContent("b", onEvent: { recorder.capture($0) })
        }
        await coalescer.flush(onEvent: { recorder.capture($0) })
        // Total content bytes received must equal what we pushed.
        var received = ""
        var contentEventCount = 0
        for event in recorder.events {
            if case .content(let s) = event {
                received += s
                contentEventCount += 1
            }
        }
        #expect(received == "a" + String(repeating: "b", count: 49))
        // The coalescer can never invent or lose events, so the
        // upper bound is the burst size + 1 (final flush). The
        // strict lower bound asserts coalescing engaged: 50
        // sequential deltas should NOT have produced 50 events.
        #expect(contentEventCount <= 50)
        #expect(contentEventCount < 50, "expected coalescing to reduce events; got \(contentEventCount)")
    }

    @Test("codex r1 BLOCKING-2: flush preserves append (wire) order across kinds")
    @MainActor
    func flush_preserves_wire_order_across_kinds() async {
        // Wire shape: rapid-mlx CAN emit content first, then
        // reasoning_content on the next chunk (reasoning models
        // interleave when the parser surfaces a mid-stream
        // reasoning block). The coalescer must preserve that
        // append order — emitting reasoning before content when
        // content arrived first would silently reorder text.
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        // Prime both flush latches without disturbing the test
        // sequence: append + flush each kind so subsequent appends
        // accumulate rather than first-flush.
        await coalescer.appendContent("seed-c", onEvent: { recorder.capture($0) })
        await coalescer.appendReasoning("seed-r", onEvent: { recorder.capture($0) })
        await coalescer.flush(onEvent: { recorder.capture($0) })
        recorder.events.removeAll()
        // Interleave deliberately: content, reasoning, content.
        await coalescer.appendContent("C1", onEvent: { recorder.capture($0) })
        await coalescer.appendReasoning("R1", onEvent: { recorder.capture($0) })
        await coalescer.appendContent("C2", onEvent: { recorder.capture($0) })
        await coalescer.flush(onEvent: { recorder.capture($0) })
        // Reassemble the per-kind text streams from the recorder.
        var contentStream = ""
        var reasoningStream = ""
        // Also record the kind sequence so we can assert ordering.
        var kindOrder: [String] = []
        for event in recorder.events {
            switch event {
            case .content(let s):
                contentStream += s
                kindOrder.append("C")
            case .reasoning(let r):
                reasoningStream += r
                kindOrder.append("R")
            default: break
            }
        }
        #expect(contentStream == "C1C2")
        #expect(reasoningStream == "R1")
        // Wire order across kinds: C must precede R must precede C.
        // The exact event count depends on window timing, but the
        // sequence of kinds (collapsing duplicates) must be C, R, C.
        var collapsed: [String] = []
        for k in kindOrder {
            if collapsed.last != k { collapsed.append(k) }
        }
        #expect(collapsed == ["C", "R", "C"], "kind order leaked: got \(kindOrder)")
    }

    /// #1743: the repaint RATE has to fall as the message grows.
    ///
    /// Every flush makes the chat view rebuild, and that rebuild re-parses
    /// the WHOLE accumulated message through MarkdownUI. Parsing is
    /// O(length), so a fixed 16 ms cadence costs O(length²) over a turn.
    /// Measured on a fake stream before the fix, main-thread CPU climbed
    /// 47 % → 68 % → 94 % → 100 % as the answer grew and stayed pinned;
    /// a real 9-minute answer left the app at 100 % with its window gone
    /// and RSS climbing ~11 MB/s until it was force-killed.
    ///
    /// This pins the two halves of the contract that matter:
    ///
    ///   1. a SHORT message is untouched — 16 ms still flushes, so the
    ///      overwhelming majority of replies stream exactly as before;
    ///   2. a LONG message does not, because the window has widened.
    ///
    /// Both directions are asserted deliberately. A "fix" that simply
    /// slowed everything down would pass (2) and fail (1), and would be
    /// a visible regression on every ordinary answer.
    @Test("#1743: the coalescing window widens once the message is long")
    @MainActor
    func window_widens_with_accumulated_length() async {
        // (1) Short message: 16 ms is still enough to flush.
        let short = SSEDeltaCoalescer()
        let shortRec = EventRecorder()
        await short.appendContent("seed") { shortRec.capture($0) }   // first flush
        let afterSeed = shortRec.events.count
        try? await Task.sleep(for: .milliseconds(40))
        await short.appendContent("more") { shortRec.capture($0) }
        #expect(
            shortRec.events.count > afterSeed,
            "a short message must still flush on the 16 ms cadence — widening it unconditionally would slow every normal reply"
        )

        // (2) Long message: the same 40 ms pause is no longer enough.
        let long = SSEDeltaCoalescer()
        let longRec = EventRecorder()
        await long.appendContent("seed") { longRec.capture($0) }
        // Push well past the adaptive threshold. Each append is separated
        // by enough real time to be flushed on the old cadence, so by the
        // end the coalescer has genuinely emitted this much text.
        let chunk = String(repeating: "x", count: 4_000)
        for _ in 0..<6 {
            try? await Task.sleep(for: .milliseconds(40))
            await long.appendContent(chunk) { longRec.capture($0) }
        }
        let afterBulk = longRec.events.count
        try? await Task.sleep(for: .milliseconds(40))
        await long.appendContent("tail") { longRec.capture($0) }
        #expect(
            longRec.events.count == afterBulk,
            "after ~24k characters a 40 ms gap still flushed — the window did not widen, so the repaint rate is still O(length²) (#1743)"
        )
    }

    @Test("flush on empty buffers is a no-op")
    @MainActor
    func empty_flush_emits_nothing() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        await coalescer.flush(onEvent: { recorder.capture($0) })
        #expect(recorder.events.isEmpty)
    }

    @Test("codex r1 NIT: empty append is a hard no-op (no spurious event)")
    @MainActor
    func empty_append_emits_nothing() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        await coalescer.appendContent("", onEvent: { recorder.capture($0) })
        await coalescer.appendReasoning("", onEvent: { recorder.capture($0) })
        await coalescer.flush(onEvent: { recorder.capture($0) })
        #expect(recorder.events.isEmpty, "empty append must produce zero events; got \(recorder.events)")
    }

    @Test("flush after content+reasoning preserves single event per kind")
    @MainActor
    func flush_coalesces_adjacent_same_kind() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        // Prime the latches with a first-flush content+reasoning
        // pair so subsequent appends accumulate. Then push a
        // 5-deep run of content followed by a 5-deep run of
        // reasoning. The trailing flush must emit exactly TWO
        // events (one merged content + one merged reasoning),
        // not 10.
        await coalescer.appendContent("seed-c", onEvent: { recorder.capture($0) })
        await coalescer.appendReasoning("seed-r", onEvent: { recorder.capture($0) })
        await coalescer.flush(onEvent: { recorder.capture($0) })
        recorder.events.removeAll()
        for _ in 0..<5 { await coalescer.appendContent("c", onEvent: { recorder.capture($0) }) }
        for _ in 0..<5 { await coalescer.appendReasoning("r", onEvent: { recorder.capture($0) }) }
        await coalescer.flush(onEvent: { recorder.capture($0) })
        var contentEventCount = 0
        var reasoningEventCount = 0
        var contentText = ""
        var reasoningText = ""
        for event in recorder.events {
            switch event {
            case .content(let s): contentEventCount += 1; contentText += s
            case .reasoning(let r): reasoningEventCount += 1; reasoningText += r
            default: break
            }
        }
        #expect(contentText == "ccccc")
        #expect(reasoningText == "rrrrr")
        // Strict: window-flush + final flush both happen across
        // these appends, so up to 2-3 events per kind is acceptable
        // here, but never 5 — coalescing must reduce that to a
        // small number bounded by flush count, not append count.
        #expect(contentEventCount < 5, "adjacent same-kind appends should coalesce; got \(contentEventCount) content events")
        #expect(reasoningEventCount < 5, "adjacent same-kind appends should coalesce; got \(reasoningEventCount) reasoning events")
    }
}

// MARK: - Throw-path flush integration test (codex r1 BLOCKING-1)

/// URLProtocol that emits N back-to-back content deltas, then a
/// mid-stream `{"error": {...}}` envelope. The error envelope makes
/// ``ChatStreamClient.send`` throw ``ChatStreamError.transport`` —
/// codex r2 sharpened r1: pre-fix the buffered deltas #2..N would be
/// silently dropped because the throw bypassed the explicit flush.
/// Post-fix the do/catch in send() drains the coalescer queue first.
final class ContentBurstThenErrorProtocol: URLProtocol, @unchecked Sendable {
    /// First delta flushes immediately (first-flush latch); deltas
    /// 2..deltaCount are coalesced into the pending queue. The error
    /// envelope follows immediately so the 16 ms window hasn't
    /// elapsed — guaranteeing the catch path is the only thing that
    /// can drain the tail.
    static let deltaCount = 8
    static let errorMessage = "server-side hiccup"

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ContentBurstThenErrorProtocol.self] + (config.protocolClasses ?? [])
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
        var body = ""
        for i in 0..<Self.deltaCount {
            body += "data: {\"choices\":[{\"delta\":{\"content\":\"d\(i) \"}}]}\n"
        }
        body += "data: {\"error\":{\"message\":\"\(Self.errorMessage)\"}}\n"
        client?.urlProtocol(self, didLoad: body.data(using: .utf8)!)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Suite("ChatStreamClient throw-path flush invariant")
struct ChatStreamClientThrowFlushTests {
    @Test("codex r1 BLOCKING-1 sharpened: error-envelope throw drains coalesced tail")
    @MainActor
    func error_envelope_throw_drains_coalesced_tail() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: ContentBurstThenErrorProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "perf-fake",
            messages: [ChatMessage(role: .user, content: "probe", status: .complete)]
        )
        var receivedText = ""
        var thrownErrorMessage: String?
        do {
            try await client.send(req) { event in
                if case .content(let c) = event { receivedText += c }
            }
        } catch let ChatStreamError.transport(message) {
            thrownErrorMessage = message
        } catch {
            Issue.record("unexpected error: \(error)")
        }
        #expect(thrownErrorMessage == ContentBurstThenErrorProtocol.errorMessage)
        var expectedText = ""
        for i in 0..<ContentBurstThenErrorProtocol.deltaCount {
            expectedText += "d\(i) "
        }
        // Pre-r1-fix: only the first delta (the first-flush) made it
        // out before the throw; deltas 1..7 were stranded in the
        // pending queue. Post-fix the catch's flush drains them.
        #expect(
            receivedText == expectedText,
            "coalescer dropped buffered tail on error-envelope throw; expected '\(expectedText)' got '\(receivedText)'"
        )
    }
}

// MARK: - Codex r2 BLOCKING coverage: pending-queue cap

extension SSEDeltaCoalescerTests {
    /// Codex r2 BLOCKING (unbounded queue): alternating-kind deltas
    /// would otherwise enqueue one segment per delta until the next
    /// time-based flush. The 32-segment cap forces a flush so the
    /// queue can't grow without bound under adversarial input.
    @Test("codex r2 BLOCKING: queue cap force-flushes on adversarial alternation")
    @MainActor
    func queue_cap_forces_flush_on_adversarial_alternation() async {
        let coalescer = SSEDeltaCoalescer()
        let recorder = EventRecorder()
        // Prime first-flush latches so subsequent appends accumulate.
        await coalescer.appendContent("seed-c", onEvent: { recorder.capture($0) })
        await coalescer.appendReasoning("seed-r", onEvent: { recorder.capture($0) })
        await coalescer.flush(onEvent: { recorder.capture($0) })
        recorder.events.removeAll()
        // Push 100 alternating deltas back-to-back so the 16 ms
        // window can never fire within this synchronous burst. The
        // 32-segment cap is the only way the queue can drain
        // mid-burst — without it the queue would grow to ~100 and
        // the trailing flush would be the only drain.
        for i in 0..<100 {
            if i % 2 == 0 {
                await coalescer.appendContent("c\(i)", onEvent: { recorder.capture($0) })
            } else {
                await coalescer.appendReasoning("r\(i)", onEvent: { recorder.capture($0) })
            }
        }
        // Codex r3 NIT sharpening: snapshot emissions BEFORE the
        // explicit final flush. Without the cap, no events surface
        // during the synchronous burst (window can't elapse, no
        // first-flush latch trigger after the seeded prime), so the
        // snapshot would be zero. With the cap, the queue hits 33
        // multiple times during the burst — emissions are non-zero.
        // The earlier `events.count > 1 after final flush` shape
        // passed either way (final flush of 100 segments emits 100
        // events regardless of cap engagement); this snapshot
        // strictly proves mid-burst cap behaviour.
        let emittedDuringBurst = recorder.events.count
        await coalescer.flush(onEvent: { recorder.capture($0) })
        #expect(
            emittedDuringBurst > 0,
            "queue cap did not force-flush mid-burst; expected non-zero in-burst emissions, got \(emittedDuringBurst)"
        )
        // Also assert full payload reassembles (cap never drops bytes).
        var assembled = ""
        for event in recorder.events {
            switch event {
            case .content(let s): assembled += s
            case .reasoning(let r): assembled += r
            default: break
            }
        }
        var expected = ""
        for i in 0..<100 {
            expected += (i % 2 == 0) ? "c\(i)" : "r\(i)"
        }
        #expect(assembled == expected, "queue cap dropped or reordered bytes")
    }
}
