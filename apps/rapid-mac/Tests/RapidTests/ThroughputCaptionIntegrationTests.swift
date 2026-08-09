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
        PrefillHeavyProtocol.reset(prefillDelay: 0.45, contentDeltas: 12, completionTokens: 12)
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://prefill")!,
                session: PrefillHeavyProtocol.session()
            ),
            persistsConversations: false
        )

        model.send("anything", alias: "test-model")
        for _ in 0..<400 where model.isStreaming {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(!model.isStreaming)

        let stats = try #require(model.messages.last?.stats)

        // 1. The wiring ran at all. Nil here means nobody stamped the first
        //    token, which is the regression this file exists to catch.
        let ttft = try #require(
            stats.timeToFirstTokenSeconds,
            "no time-to-first-token was recorded — runOneStream did not stamp the first delta"
        )

        // 2. It measured the prefill, not something incidental. The fake
        //    holds the response for 0.45 s before the first delta.
        #expect(ttft >= 0.4, "TTFT \(ttft)s is below the 0.45s the transport withheld the first token for")
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
        PrefillHeavyProtocol.reset(prefillDelay: 0.45, contentDeltas: 12, completionTokens: nil)
        let model = ChatViewModel(
            client: ChatStreamClient(
                baseURL: URL(string: "fake://prefill")!,
                session: PrefillHeavyProtocol.session()
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
        #expect(ttft >= 0.4)

        let estimate = try #require(stats.estimatedTokensPerSecond)
        let wholeTurnEstimate = (Double(stats.charCount) / 4.0) / stats.elapsedSeconds
        #expect(estimate > wholeTurnEstimate * 1.5)
    }
}

/// Streams a response whose time is spent almost entirely before the first
/// content delta, so a turn's prefill and decode halves are far enough apart
/// to tell which one a rate was divided by.
private final class PrefillHeavyProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var prefillDelay: TimeInterval = 0.45
    nonisolated(unsafe) static var contentDeltas = 12
    nonisolated(unsafe) static var completionTokens: Int? = 12

    static func reset(prefillDelay: TimeInterval, contentDeltas: Int, completionTokens: Int?) {
        Self.prefillDelay = prefillDelay
        Self.contentDeltas = contentDeltas
        Self.completionTokens = completionTokens
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

        // The prefill. Nothing reaches the view model during this window, so
        // a correctly-wired TTFT lands at or after it.
        Thread.sleep(forTimeInterval: Self.prefillDelay)

        for _ in 0..<Self.contentDeltas {
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
