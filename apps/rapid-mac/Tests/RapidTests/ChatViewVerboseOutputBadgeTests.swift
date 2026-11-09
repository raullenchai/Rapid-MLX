import Foundation
import Testing
@testable import Rapid

/// Cycle-13 (2026-06-20) regression pin for the chat-view verbose-output
/// length-truncation badge filed by cycle-7 fuzz-perf F-5.
///
/// Symptom: a verbose non-reasoning dense model (nemotron-30b-4bit and
/// similar) answering "What is 17*23?" with ``enable_thinking=false``
/// and the default ``max_tokens=200`` emits a 200-token LaTeX
/// derivation and hits ``finish_reason == "length"``. Pre-cycle-13 the
/// row landed as ``.complete`` with no visible indicator that the
/// derivation was half-finished — the user read the truncated
/// derivation as the actual answer.
///
/// Fix: ``ChatViewModel.runOneStream`` sets
/// ``ChatMessage.contentTruncated = true`` when ``finish_reason ==
/// "length"`` AND ``content`` is non-empty AND ``reasoning`` is empty
/// (i.e. the model burned its budget mid-answer rather than
/// mid-think). ``MessageRow.assistantBlock`` reads the flag and paints
/// a subtle "Answer cut off (Max Tokens hit). Increase Max Tokens to
/// see the rest." caption under the bubble.
///
/// Disjoint from PR #317's ``reasoningTruncated`` fallback (which
/// covers ``finish_reason: length`` + empty content + populated
/// reasoning). The two flags can never both be true on the same row
/// because the predicates short-circuit on opposite emptiness gates.
///
/// This file covers (i) the structural predicate
/// ``ChatMessage.shouldFlagContentTruncated`` directly via the 4-cell
/// truth table the assignment specifies; (ii) the badge copy + VoiceOver
/// accessibility label snapshots so an accidental rewording fails CI;
/// (iii) the back-compat surface — old on-disk sessions saved before
/// this cycle decode with the flag false; (iv) the disjoint-with-
/// reasoningTruncated invariant.
@MainActor
@Suite("Cycle-13 verbose-output length-truncation badge (chat-view F-5)")
struct ChatViewVerboseOutputBadgeTests {

    // MARK: - The 4-cell truth table the assignment specifies

    /// Cell (1, 1): ``finish_reason == "length"`` × reasoning empty →
    /// badge VISIBLE. The verbose-output case the fix targets.
    @Test("(length, empty-reasoning, populated-content) → badge VISIBLE — the cycle-13 verbose-output case")
    func badgeVisibleOnLengthWithEmptyReasoning() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "Let me derive this. 17 * 23 = 17 * (20 + 3) = 17 * 20 + 17 * 3 = 340 + 51 = ...",
            reasoning: "",
            finishReason: "length"
        )
        #expect(flag == true,
                "Verbose-output length-truncation must surface the badge so the user knows the answer is half-finished")
    }

    /// Cell (1, 2): ``finish_reason == "length"`` × reasoning non-empty
    /// → badge HIDDEN. PR #317's reasoning-only fallback owns this
    /// shape via ``reasoningTruncated``; a second caption on the same
    /// row would be redundant and visually noisy.
    @Test("(length, non-empty-reasoning, populated-content) → badge HIDDEN — PR #317 owns the reasoning-truncated lane")
    func badgeHiddenOnLengthWithReasoning() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "Paris is the capital of France. It is also a major cultural...",
            reasoning: "User wants a capital city fact.",
            finishReason: "length"
        )
        #expect(flag == false,
                "PR #317's reasoning-truncated path owns this shape; the verbose-output badge must stay out of it")
    }

    /// Cell (2, 1): ``finish_reason == "stop"`` × reasoning empty →
    /// badge HIDDEN. Normal completion of a non-reasoning model —
    /// nothing to flag.
    @Test("(stop, empty-reasoning, populated-content) → badge HIDDEN — normal completion of a non-reasoning model")
    func badgeHiddenOnStopWithEmptyReasoning() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "17 * 23 = 391.",
            reasoning: "",
            finishReason: "stop"
        )
        #expect(flag == false,
                "Clean stop is a real completion; badge must not falsely accuse the row of being truncated")
    }

    /// Cell (2, 2): ``finish_reason == "stop"`` × reasoning non-empty
    /// → badge HIDDEN. Normal hybrid-thinking completion (Qwen3.6 /
    /// GLM 4.7 / Qwopus shape) — nothing to flag.
    @Test("(stop, non-empty-reasoning, populated-content) → badge HIDDEN — normal hybrid-thinking completion")
    func badgeHiddenOnStopWithReasoning() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "The capital of France is Paris.",
            reasoning: "User asked about a capital city.",
            finishReason: "stop"
        )
        #expect(flag == false,
                "Hybrid-thinking real completion must not wear the truncation badge")
    }

    // MARK: - Additional gate coverage beyond the 4-cell table

    @Test("Empty content + length never flags — classifyTerminal owns that lane (reasoningOnly or emptyTurnFailure)")
    func badgeHiddenOnEmptyContent() {
        // The empty-content + length shape is split between PR #317
        // (.reasoningOnlyTruncated when reasoning is populated) and
        // the v0.4.35 max_tokens failure copy (.emptyTurnFailure when
        // both are empty). The badge must NOT compete with either —
        // it's specifically the "answer body exists but was cut off"
        // case.
        let flagWithReasoning = ChatMessage.shouldFlagContentTruncated(
            content: "",
            reasoning: "Let me think about this.",
            finishReason: "length"
        )
        let flagEverythingEmpty = ChatMessage.shouldFlagContentTruncated(
            content: "",
            reasoning: "",
            finishReason: "length"
        )
        #expect(flagWithReasoning == false,
                "Reasoning-only truncation is PR #317's domain, not the verbose-output badge")
        #expect(flagEverythingEmpty == false,
                "Zero-output failure is classifyTerminal's .emptyTurnFailure domain, not the badge")
    }

    @Test("Whitespace-only content does not flag — \"   \\n\" is not a real answer body")
    func badgeHiddenOnWhitespaceContent() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "  \n\t  ",
            reasoning: "",
            finishReason: "length"
        )
        #expect(flag == false,
                "Whitespace-only content has no body to flag as truncated — defer to classifyTerminal")
    }

    @Test("Whitespace-only reasoning is treated as empty — badge VISIBLE when content is populated and reason is length")
    func badgeVisibleOnWhitespaceReasoning() {
        // Mirror the existing
        // ``ReasoningContentFallbackTests.whitespaceOnlyReasoningIsTreatedAsEmpty``
        // contract: a reasoning lane containing only whitespace is
        // not a useful surface, so PR #317's fallback does NOT fire
        // — which means the verbose-output badge SHOULD fire if the
        // visible content was truncated. Pin so both helpers agree
        // on the trim-then-compare-empty convention.
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "Working out the answer step by step. First...",
            reasoning: "  \n\t  ",
            finishReason: "length"
        )
        #expect(flag == true,
                "Whitespace-only reasoning is treated as empty by classifyTerminal; the badge must agree")
    }

    @Test("finish_reason: tool_calls never flags — tool-call termination is a real completion, not a truncation")
    func badgeHiddenOnToolCalls() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "I'll look that up.",
            reasoning: "",
            finishReason: "tool_calls"
        )
        #expect(flag == false)
    }

    @Test("finish_reason: nil never flags — non-conforming server emitting no reason is not a length truncation")
    func badgeHiddenOnNilReason() {
        let flag = ChatMessage.shouldFlagContentTruncated(
            content: "Some content",
            reasoning: "",
            finishReason: nil
        )
        #expect(flag == false)
    }

    // MARK: - ChatMessage.contentTruncated marker semantics

    @Test("contentTruncated defaults to false on every ChatMessage shape")
    func contentTruncatedDefaultsFalse() {
        // Pre-cycle-13 callers — and ChatMessages decoded from on-disk
        // sessions saved before the flag existed — must arrive with
        // the flag false. Otherwise the chat view would hallucinate
        // "Answer cut off" on every old session.
        #expect(ChatMessage(role: .user).contentTruncated == false)
        #expect(ChatMessage(role: .assistant).contentTruncated == false)
        #expect(
            ChatMessage(
                role: .assistant,
                content: "real reply",
                status: .complete
            ).contentTruncated == false
        )
    }

    @Test("contentTruncated and reasoningTruncated are disjoint — both can be set but the predicates short-circuit on opposite emptiness gates")
    func disjointWithReasoningTruncated() {
        // The two flags own opposite halves of the length-truncated
        // space:
        //   * reasoningTruncated: empty content + populated reasoning
        //   * contentTruncated:   populated content + empty reasoning
        // If a row ever managed to hit both, the chat view would
        // paint the badge AND auto-expand the reasoning disclosure
        // claiming the answer was cut off mid-think, which is
        // user-hostile. Pin the disjoint-predicate invariant directly:
        // there is NO (content, reasoning) pair such that
        // shouldFlagContentTruncated returns true at the same time
        // as the empty-content + populated-reasoning shape that
        // classifyTerminal routes to .reasoningOnlyTruncated.
        //
        // Concretely: the helper returns true ONLY when content is
        // non-empty AND reasoning is empty; the .reasoningOnlyTruncated
        // outcome fires ONLY when content is empty AND reasoning is
        // non-empty. These are mutually exclusive on the (content,
        // reasoning) emptiness axis, by construction.
        let pairs: [(String, String, Bool)] = [
            // (content, reasoning, expected badge)
            ("answer body", "", true),
            ("", "thinking trace", false),
            ("", "", false),
            ("answer body", "thinking trace", false),
        ]
        for (c, r, expected) in pairs {
            let flag = ChatMessage.shouldFlagContentTruncated(
                content: c,
                reasoning: r,
                finishReason: "length"
            )
            #expect(flag == expected,
                    "shouldFlagContentTruncated(content: \"\(c)\", reasoning: \"\(r)\") expected \(expected), got \(flag)")
        }
    }

    // MARK: - Codable back-compat (old sessions decode cleanly)

    @Test("Pre-cycle-13 session envelopes decode with contentTruncated defaulted to false")
    func preCycle13SessionsDecodeWithFlagFalse() {
        // Old on-disk sessions saved before this cycle have no
        // ``contentTruncated`` key. The custom decoder uses
        // decodeIfPresent + ?? false so they load cleanly. Without
        // this shim Swift's synthesised init(from:) would throw
        // ``keyNotFound`` and every old session would refuse to load
        // post-upgrade.
        //
        // Build a JSON envelope that mirrors what a v0.7.x session
        // saved to ~/Library/Application Support would look like —
        // including the ``reasoningTruncated`` field PR #317 added,
        // but WITHOUT the new ``contentTruncated`` key.
        let json = """
        {
            "id": "8D8B7E7C-2F4B-4E5A-9F9C-7A8B6C5D4E3F",
            "role": "assistant",
            "content": "old session reply",
            "reasoning": "",
            "status": "complete",
            "reasoningTruncated": false,
            "createdAt": 750000000.0
        }
        """
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let decoded = try? decoder.decode(ChatMessage.self, from: data)
        #expect(decoded != nil, "Pre-cycle-13 session envelope must decode cleanly")
        #expect(decoded?.contentTruncated == false,
                "Missing key must default to false, not throw or true")
        #expect(decoded?.content == "old session reply")
        #expect(decoded?.reasoningTruncated == false)
    }

    @Test("Encode then decode round-trips contentTruncated through Codable")
    func codableRoundTripsContentTruncated() {
        // Pin the JSON shape so an accidental CodingKeys typo (or a
        // dropped case in the enum) surfaces immediately. The flag
        // must be present in the encoded blob AND deserialise back
        // to the same value.
        let original = ChatMessage(
            role: .assistant,
            content: "Half-finished answer because the cap was hit at...",
            status: .complete,
            contentTruncated: true
        )
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        guard let data = try? encoder.encode(original),
              let decoded = try? decoder.decode(ChatMessage.self, from: data) else {
            #expect(Bool(false), "Codable round-trip must succeed")
            return
        }
        #expect(decoded.contentTruncated == true)
        #expect(decoded.content == original.content)

        // And the JSON blob actually contains the key — so a future
        // refactor that drops the case from CodingKeys fails CI
        // here, not silently in production sessions.
        let blob = String(data: data, encoding: .utf8) ?? ""
        #expect(blob.contains("contentTruncated"),
                "Encoded JSON must include the contentTruncated key; got: \(blob)")
    }

    // MARK: - Copy + accessibility snapshots (codex r1 future-proofing)

    @Test("Visible badge copy is pinned — \"Answer cut off (Max Tokens hit). Increase Max Tokens to see the rest.\"")
    func badgeCopySnapshot() {
        // Pin the exact string the user sees so an accidental
        // rewording in MessageRow.assistantBlock surfaces in CI as a
        // failing snapshot, not silently. ChatGPT-Desktop's
        // "Continue generating" prompt is the reference UX; the copy
        // explicitly names (i) what happened ("Answer cut off") and
        // (ii) the user-facing knob to raise ("Max Tokens"), which
        // the assignment requires.
        //
        // Lives on ``ChatMessage`` (the model layer) instead of
        // ``MessageRow`` (private) so the test target can read it
        // without piercing private visibility — see the constant's
        // docstring for the rationale.
        #expect(ChatMessage.lengthTruncationBadgeCopy ==
                "Answer cut off (Max Tokens hit). Increase Max Tokens to see the rest.")
    }

    @Test("VoiceOver accessibility label is pinned — well-formed sentence, names cause + remediation")
    func badgeAccessibilityLabelSnapshot() {
        // VoiceOver pacing breaks on the parenthetical in the
        // visible string, so the accessibility caption is a separate
        // well-formed sentence. Pin both so an a11y regression
        // (e.g. accidentally falling back to the visible string)
        // surfaces as a test failure.
        let label = ChatMessage.lengthTruncationBadgeAccessibilityLabel
        #expect(label ==
                "Answer cut off because the Max Tokens limit was hit. Increase Max Tokens in Settings to see the rest of the answer.")
        // Sanity: caption MUST name the trigger AND the user-facing
        // knob — even if the snapshot above is later revised, a
        // reword that drops either of these regresses the UX.
        #expect(label.contains("Max Tokens"),
                "Accessibility label must name the user-facing knob to raise")
        #expect(label.lowercased().contains("cut off"),
                "Accessibility label must tell the user what happened")
    }
}
