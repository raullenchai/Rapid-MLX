import Foundation
import Testing
@testable import Rapid

/// Contract for v0.4.12 ``MessageStats`` — the throughput caption
/// that surfaces "~84 tok/s · 2.4 s" under each completed assistant
/// turn. Pins the formatter switchover points + the
/// reported-vs-estimated precedence so a refactor can't quietly
/// drop the "~" estimate prefix or invert which branch wins when
/// both are populated.
@Suite("MessageStats + caption formatters")
struct MessageStatsTests {

    // MARK: - estimatedTokensPerSecond / reportedTokensPerSecond

    @Test("Estimate fires from char count when promptTokens / completionTokens are nil")
    func estimateFires() {
        let s = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: nil,
            completionTokens: nil
        )
        // 800 chars / 4 chars-per-token = 200 tokens
        // 200 / 2 s = 100 tokens/s
        #expect(s.estimatedTokensPerSecond ?? 0 == 100.0)
        #expect(s.reportedTokensPerSecond == nil)
    }

    @Test("Reported overrides estimate when the server populated completionTokens")
    func reportedOverrides() {
        let s = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: 50,
            completionTokens: 180
        )
        // No TTFT recorded (pre-existing transcript) → the whole turn is
        // the denominator, and 179 intervals span 180 tokens.
        #expect(s.reportedTokensPerSecond ?? 0 == 89.5)
        // Estimate is still computable — the UI just prefers reported.
        #expect(s.estimatedTokensPerSecond != nil)
    }

    // MARK: - Prefill must not be charged to the decode rate

    /// The user-reported defect, with numbers measured on an M3 Ultra
    /// against `qwen3.5-4b-4bit` through the app's own bundled sidecar.
    ///
    /// One chat turn carrying the three built-in tool schemas: 970 prompt
    /// tokens prefilled in 0.69 s, then 171 tokens generated over the
    /// remaining 1.30 s. Charging the prefill to throughput captions it at
    /// 86 tok/s; the model's actual decode rate is ~131. The picker
    /// advertises ~61 for the same alias and the old benchmark card
    /// measured 143 — three numbers for one model, all labelled "tok/s".
    @Test("A prefill-dominated turn reports the decode rate, not the whole-turn rate")
    func prefillIsNotChargedToThroughput() throws {
        let s = MessageStats(
            elapsedSeconds: 1.99,
            charCount: 852,
            promptTokens: 970,
            completionTokens: 171,
            timeToFirstTokenSeconds: 0.69
        )
        // (171 - 1) tokens over the 1.30 s decode window.
        let rate = try #require(s.reportedTokensPerSecond)
        #expect(abs(rate - 130.77) < 0.01)
        // The whole-turn arithmetic this replaced (171 / 1.99 = 85.9).
        // Pinned as an explicit NOT so reintroducing
        // `completionTokens / elapsedSeconds` fails here rather than
        // silently shipping the understated number again.
        #expect(abs(rate - 85.9) > 1.0)
    }

    /// The same conversation's opening turn: 0.75 s of prefill and an
    /// 8-token answer leaves a 50 ms decode window. Seven tokens across
    /// 50 ms is the noise floor, not a measurement — reporting "140 tok/s"
    /// off it would trade one misleading number for another. Say nothing
    /// about the rate and let time-to-first-token carry the turn.
    @Test("A decode window at the noise floor reports no rate at all")
    func decodeWindowBelowNoiseFloorHasNoRate() {
        let s = MessageStats(
            elapsedSeconds: 0.78,
            charCount: 31,
            promptTokens: 957,
            completionTokens: 8,
            timeToFirstTokenSeconds: 0.75
        )
        #expect(s.reportedTokensPerSecond == nil)
        #expect(s.estimatedTokensPerSecond == nil)
        // The turn is still describable — TTFT is what actually cost the
        // user the wait, and it survives.
        #expect(s.timeToFirstTokenSeconds == 0.75)
    }

    @Test("Decode window excludes TTFT")
    func decodeWindowExcludesTTFT() throws {
        let s = MessageStats(
            elapsedSeconds: 2.00,
            charCount: 400,
            promptTokens: 900,
            completionTokens: 101,
            timeToFirstTokenSeconds: 1.00
        )
        #expect(try #require(s.decodeSeconds) == 1.00)
        #expect(try #require(s.reportedTokensPerSecond) == 100.0)
        // The char-count fallback uses the same window.
        #expect(try #require(s.estimatedTokensPerSecond) == 100.0)
    }

    @Test("No TTFT recorded → the whole turn stays the denominator (old transcripts)")
    func missingTTFTFallsBackToWholeTurn() throws {
        let s = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: nil,
            completionTokens: nil,
            timeToFirstTokenSeconds: nil
        )
        #expect(try #require(s.decodeSeconds) == 2.0)
        #expect(try #require(s.estimatedTokensPerSecond) == 100.0)
    }

    @Test("A one-token reply reports no rate rather than dividing by a window it never occupied")
    func singleTokenReplyHasNoRate() throws {
        let s = MessageStats(
            elapsedSeconds: 1.50,
            charCount: 3,
            promptTokens: 900,
            completionTokens: 1,
            timeToFirstTokenSeconds: 0.75
        )
        // The 0.75 s decode window is far above the noise floor, so the
        // token count is the ONLY thing that can make this nil. Without
        // that isolation the noise guard would satisfy the assertion and
        // the `completionTokens > 1` guard could be deleted unnoticed.
        #expect(try #require(s.decodeSeconds) > 0.5)
        #expect(s.reportedTokensPerSecond == nil)
    }

    @Test("TTFT at or past the end of the turn yields no rate, never a negative one")
    func nonPositiveDecodeWindowHasNoRate() {
        let s = MessageStats(
            elapsedSeconds: 1.00,
            charCount: 40,
            promptTokens: 900,
            completionTokens: 20,
            timeToFirstTokenSeconds: 1.00
        )
        #expect(s.decodeSeconds == nil)
        #expect(s.reportedTokensPerSecond == nil)
        #expect(s.estimatedTokensPerSecond == nil)
    }

    @Test("Sub-50ms elapsed returns nil — divide-by-near-zero would print a garbage TPS")
    func nearZeroElapsedNoTPS() {
        let s = MessageStats(
            elapsedSeconds: 0.001,
            charCount: 5,
            promptTokens: nil,
            completionTokens: nil
        )
        #expect(s.estimatedTokensPerSecond == nil)
        #expect(s.reportedTokensPerSecond == nil)
    }

    // MARK: - formatTPS

    @Test("TPS under 10 keeps one decimal so 4-bit 27B doesn't read as 9 tok/s")
    func tpsSubTen() {
        #expect(AssistantStatsFormatter.formatTPS(9.4) == "9.4")
        #expect(AssistantStatsFormatter.formatTPS(0.7) == "0.7")
    }

    @Test("TPS 10+ rounds to int because nobody cares about tenths at 80 tok/s")
    func tpsTenPlus() {
        #expect(AssistantStatsFormatter.formatTPS(10.0) == "10")
        #expect(AssistantStatsFormatter.formatTPS(83.7) == "84")
        #expect(AssistantStatsFormatter.formatTPS(120.4) == "120")
    }

    // MARK: - formatElapsed

    @Test("Sub-second elapsed renders as milliseconds")
    func elapsedMs() {
        #expect(AssistantStatsFormatter.formatElapsed(0.42) == "420 ms")
        #expect(AssistantStatsFormatter.formatElapsed(0.05) == "50 ms")
    }

    @Test("1-60s elapsed renders as 'X.Xs'")
    func elapsedSeconds() {
        #expect(AssistantStatsFormatter.formatElapsed(1.0) == "1.0 s")
        #expect(AssistantStatsFormatter.formatElapsed(8.34) == "8.3 s")
        #expect(AssistantStatsFormatter.formatElapsed(59.9) == "59.9 s")
    }

    @Test("Past 60s elapsed renders as 'Xm Ys' so tool-call rounds aren't 94.7s walls")
    func elapsedMinutes() {
        #expect(AssistantStatsFormatter.formatElapsed(60.0) == "1m 0s")
        #expect(AssistantStatsFormatter.formatElapsed(94.7) == "1m 34s")
        #expect(AssistantStatsFormatter.formatElapsed(125.0) == "2m 5s")
    }

    // MARK: - VoiceOver caption

    @Test("Accessibility caption resolves the tilde + middle-dot into plain English")
    func a11yCaption() {
        let est = MessageStats(
            elapsedSeconds: 2.4,
            charCount: 800,
            promptTokens: nil,
            completionTokens: nil
        )
        let caption = AssistantStatsFormatter.accessibilityCaption(for: est)
        #expect(caption.contains("approximately"))
        #expect(caption.contains("tokens per second"))
        #expect(caption.contains("took"))

        let reported = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: 50,
            completionTokens: 180
        )
        let captionReported = AssistantStatsFormatter.accessibilityCaption(for: reported)
        // Reported branch drops the "approximately" prefix — the
        // number IS authoritative when usage is wired (v0.4.13).
        #expect(!captionReported.contains("approximately"))
        #expect(captionReported.contains("tokens per second"))
        // No TTFT recorded on either fixture, so neither caption may
        // claim one.
        #expect(!caption.contains("to the first token"))
        #expect(!captionReported.contains("to the first token"))
    }

    @Test("Caption names time-to-first-token separately instead of blending it into the rate")
    func a11yCaptionNamesTTFT() {
        let s = MessageStats(
            elapsedSeconds: 2.00,
            charCount: 852,
            promptTokens: 970,
            completionTokens: 171,
            timeToFirstTokenSeconds: 0.75
        )
        let caption = AssistantStatsFormatter.accessibilityCaption(for: s)
        #expect(caption.contains("750 ms to the first token"))
        #expect(caption.contains("tokens per second"))
        #expect(caption.contains("took 2.0 s"))
    }

    // MARK: - Schema compat (old sessions without stats)

    @Test("Decodes a pre-v0.4.12 message envelope that has no 'stats' key — defaults to nil")
    func decodesLegacyEnvelope() throws {
        // Hand-rolled JSON matching the v0.4.11 schema (no stats field).
        // Autosynth Codable should treat missing optional fields as nil.
        let legacyJSON = """
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "role": "assistant",
            "content": "Hello from v0.4.11",
            "reasoning": "",
            "status": "complete",
            "createdAt": 770000000.0
        }
        """.data(using: .utf8)!
        let dec = JSONDecoder()
        let msg = try dec.decode(ChatMessage.self, from: legacyJSON)
        #expect(msg.stats == nil)
        #expect(msg.content == "Hello from v0.4.11")
    }

    @Test("Decodes a stats block written before timeToFirstTokenSeconds existed")
    func decodesStatsWithoutTTFT() throws {
        let priorSchema = """
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "role": "assistant",
            "content": "Hello from before the TTFT split",
            "reasoning": "",
            "status": "complete",
            "createdAt": 770000000.0,
            "stats": {
                "elapsedSeconds": 2.0,
                "charCount": 800,
                "promptTokens": 50,
                "completionTokens": 180
            }
        }
        """.data(using: .utf8)!
        let msg = try JSONDecoder().decode(ChatMessage.self, from: priorSchema)
        let stats = try #require(msg.stats)
        #expect(stats.timeToFirstTokenSeconds == nil)
        // Still renders a rate — an old transcript must not go blank.
        #expect(stats.reportedTokensPerSecond != nil)
    }

    @Test("Round-trips a populated stats field through JSON encode → decode")
    func roundTripsStats() throws {
        var msg = ChatMessage(role: .assistant, content: "Hi", status: .complete)
        msg.stats = MessageStats(
            elapsedSeconds: 2.4,
            charCount: 100,
            promptTokens: 50,
            completionTokens: 25,
            timeToFirstTokenSeconds: 0.4
        )
        let data = try JSONEncoder().encode(msg)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(back.stats?.elapsedSeconds == 2.4)
        #expect(back.stats?.charCount == 100)
        #expect(back.stats?.promptTokens == 50)
        #expect(back.stats?.completionTokens == 25)
        #expect(back.stats?.timeToFirstTokenSeconds == 0.4)
    }
}
