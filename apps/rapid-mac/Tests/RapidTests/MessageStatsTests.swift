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
        // the denominator, and because that window opens BEFORE the first
        // token, all 180 fall inside it. Unchanged from what this row was
        // originally captioned with.
        #expect(s.reportedTokensPerSecond ?? 0 == 90.0)
        // The estimate is now SUPPRESSED whenever usage exists. It is a
        // fallback for missing data, not a second opinion — letting it
        // answer alongside the reported path is how a one-token reply got
        // captioned "~1 tok/s" after the reported path declined to rate it.
        #expect(s.estimatedTokensPerSecond == nil)
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
        // The char-count fallback is suppressed here (usage present), so
        // assert the window it WOULD have used, on the same fixture minus
        // the server-reported counts. 400 chars ≈ 100 estimated tokens, and
        // the estimate carries the same -1 as the reported path above: the
        // token that ENDED prefill was not produced inside the window, so
        // 99 intervals span it. The two paths differing by anything other
        // than their token source would mean one of them is measuring
        // something the other is not.
        var noUsage = s
        noUsage.completionTokens = nil
        noUsage.promptTokens = nil
        #expect(try #require(noUsage.estimatedTokensPerSecond) == 99.0)
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
        // The field is ABSENT, not invalid: this row predates the
        // measurement, so the whole turn is the only denominator there has
        // ever been for it and 200 tokens all fall inside it.
        #expect(!s.measuresPrefill)
        #expect(try #require(s.decodeSeconds) == 2.0)
        #expect(try #require(s.estimatedTokensPerSecond) == 100.0)
    }

    /// The pair that ``measuresPrefill`` separates, side by side on
    /// otherwise identical numbers. Collapsing the two — by keying the
    /// fallback off ``validTimeToFirstToken`` instead of the field's
    /// presence — makes this test fail on whichever row it wrongs.
    @Test("Absent TTFT and rejected TTFT are not the same row")
    func absentAndCorruptTTFTAreDistinguished() throws {
        let legacy = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: 50,
            completionTokens: 100,
            timeToFirstTokenSeconds: nil        // written before the field existed
        )
        let corrupt = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: 50,
            completionTokens: 100,
            timeToFirstTokenSeconds: 5.0        // a modern row, measured wrong
        )
        #expect(!legacy.measuresPrefill)
        #expect(corrupt.measuresPrefill)
        #expect(corrupt.validTimeToFirstToken == nil)

        #expect(try #require(legacy.reportedTokensPerSecond) == 50.0)
        #expect(corrupt.reportedTokensPerSecond == nil)

        var legacyNoUsage = legacy
        legacyNoUsage.completionTokens = nil
        var corruptNoUsage = corrupt
        corruptNoUsage.completionTokens = nil
        #expect(try #require(legacyNoUsage.estimatedTokensPerSecond) == 100.0)
        #expect(corruptNoUsage.estimatedTokensPerSecond == nil)
    }

    /// The estimate's own single-token case. ``singleTokenReplyHasNoRate``
    /// covers it for a server that reports usage; a server that reports
    /// none reaches the char-count path instead, where four visible
    /// characters are one estimated token and zero token intervals.
    @Test("A one-token reply with no server usage is not estimated either")
    func singleTokenEstimateHasNoRate() throws {
        let s = MessageStats(
            elapsedSeconds: 1.0,
            charCount: 4,               // ≈ 1 estimated token
            promptTokens: nil,
            completionTokens: nil,      // server reported no usage
            timeToFirstTokenSeconds: 0.5,
            reasoningEmitted: false
        )
        #expect(s.estimatedTokensPerSecond == nil)
        // Two tokens' worth of characters is the first length that spans an
        // interval, and it is rated: (2 - 1) / 0.5 s.
        var twoTokens = s
        twoTokens.charCount = 8
        #expect(try #require(twoTokens.estimatedTokensPerSecond) == 2.0)
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
        // …and nothing else may quietly supply one in its place. The
        // estimate used to fire here and print "~1 tok/s" for a reply the
        // reported path had deliberately declined to rate, so asserting
        // only the reported property left the user-visible bug uncovered.
        #expect(s.estimatedTokensPerSecond == nil)
        let caption = AssistantStatsFormatter.accessibilityCaption(for: s)
        #expect(!caption.contains("tokens per second"), "caption still shows a rate: \(caption)")
        #expect(caption.contains("to the first token"))
    }

    @Test("A legacy transcript keeps N/elapsed, not (N-1)/elapsed")
    func legacyTranscriptKeepsWholeTurnArithmetic() throws {
        let s = MessageStats(
            elapsedSeconds: 2.0,
            charCount: 800,
            promptTokens: 50,
            completionTokens: 100,
            timeToFirstTokenSeconds: nil
        )
        // Without a TTFT the denominator is the whole turn, which begins
        // BEFORE the first token — so all N tokens fall inside it and the
        // -1 has no interval to stand on. Re-rendering an old row as
        // 49.5 instead of the 50 it was captioned with is a silent
        // rewrite of history, not a fix.
        #expect(try #require(s.reportedTokensPerSecond) == 50.0)
    }

    @Test("A TTFT outside the turn is rejected for BOTH the rate and the caption")
    func impossibleTTFTIsRejectedEverywhere() throws {
        let s = MessageStats(
            elapsedSeconds: 1.0,
            charCount: 40,
            promptTokens: 900,
            completionTokens: 20,
            timeToFirstTokenSeconds: 1.2   // clock step, or a hand-edited session file
        )
        #expect(s.validTimeToFirstToken == nil)
        // The row CLAIMS a measurement and the claim is nonsense, so it
        // gets no rate — not a fallback to 20 tok/s, which is the
        // whole-turn number this change exists to stop printing and which
        // no reader could tell apart from a real one.
        #expect(s.reportedTokensPerSecond == nil)
        #expect(s.decodeSeconds == nil)
        // …and the caption must not print "1.2 s to first token · 1.0 s".
        let caption = AssistantStatsFormatter.accessibilityCaption(for: s)
        #expect(!caption.contains("to the first token"), "caption rendered an impossible TTFT: \(caption)")
    }

    @Test("A reasoning turn with no server usage reports no estimate")
    func reasoningWithoutUsageSuppressesTheEstimate() {
        let s = MessageStats(
            elapsedSeconds: 12.0,
            charCount: 40,          // one short sentence of visible prose…
            promptTokens: nil,
            completionTokens: nil,  // …server reported no usage
            timeToFirstTokenSeconds: 0.5,
            reasoningEmitted: true  // …after 11.5 s of thinking
        )
        // The decode window opened at the first REASONING token, so it
        // covers the whole think; charCount covers only the prose. Dividing
        // one by the other understates the rate by however long the model
        // thought — the same distortion this change removed from the
        // reported path, sneaking back in through the fallback.
        #expect(s.estimatedTokensPerSecond == nil)
        // The same turn without reasoning is still estimated.
        var proseOnly = s
        proseOnly.reasoningEmitted = false
        #expect(proseOnly.estimatedTokensPerSecond != nil)
    }

    @Test("A TTFT equal to the turn length is not a measurement either")
    func ttftEqualToElapsedIsRejected() throws {
        let s = MessageStats(
            elapsedSeconds: 1.00,
            charCount: 40,
            promptTokens: 900,
            completionTokens: 20,
            timeToFirstTokenSeconds: 1.00
        )
        // A first token that arrived exactly when the turn ended leaves no
        // decode window, so it is not prefill data — and, coming from a
        // build that stamps the field, not something to paper over with
        // whole-turn arithmetic either.
        #expect(s.validTimeToFirstToken == nil)
        #expect(s.reportedTokensPerSecond == nil)
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
            timeToFirstTokenSeconds: 0.4,
            reasoningEmitted: true
        )
        let data = try JSONEncoder().encode(msg)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(back.stats?.elapsedSeconds == 2.4)
        #expect(back.stats?.charCount == 100)
        #expect(back.stats?.promptTokens == 50)
        #expect(back.stats?.completionTokens == 25)
        #expect(back.stats?.timeToFirstTokenSeconds == 0.4)
        // Both new fields must survive the disk round trip. Without this,
        // dropping `reasoningEmitted` from persistence would silently
        // re-enable the mismatched estimate on every reloaded transcript.
        #expect(back.stats?.reasoningEmitted == true)
    }
}
