# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for r6-B R6-C3 streaming Action: hold-back.

Bug fixed: the streaming UI-TARS tool parser used to leak raw
``Action: <verb>`` bytes into ``delta.content`` BEFORE the structured
``delta.tool_calls`` flush. Two leak vectors:

1. ``has_pending_tool_call("Action: type")`` returned ``False`` because
   the strict ``_ACTION_LINE`` regex requires ``(``. The streaming
   postprocessor's fast-path then short-circuited the delta as plain
   ``content`` — leaking the bare ``Action: <verb>`` bytes.
2. ``extract_tool_calls_streaming`` returned ``{"content": delta_text}``
   when ``"Action:" not in current_text``, even when the trailing bytes
   were a STRICT PREFIX of ``Action:`` (e.g. delta ended with ``"\\nAc"``).
   The next delta resolved the token but those candidate-opener bytes
   had already left the wire.

R6-C3 fix: parser-level hold-back state machine.

* ``has_pending_tool_call`` now returns True for bare ``Action:``
  (no verb / parens yet) AND for trailing strict-prefix of ``Action:``
  at the buffer tail.
* ``extract_tool_calls_streaming`` clips off the trailing partial-opener
  candidate before flushing the safe prefix as content.

Repro: dogfood-087 R2-6 (Aki r2.md) — chat completions streaming with
UI-TARS + computer tool + click prompt; streamed chunks contained
``delta.content="Action: type"`` raw text BEFORE the ``delta.tool_calls``
event fired with the structured action.
"""

from __future__ import annotations

import pytest

from vllm_mlx.tool_parsers.ui_tars_tool_parser import UiTarsToolParser


@pytest.fixture
def parser():
    p = UiTarsToolParser(tokenizer=None)
    p.reset()
    return p


def _drive_stream(parser, deltas):
    """Drive the streaming parser through ``deltas`` (list of str) and
    return ``(events, accumulated_content, tool_calls)``.

    ``events`` is the list of parser return values per delta;
    ``accumulated_content`` is the concatenation of every emitted
    content chunk (the bytes that would be surfaced on
    ``delta.content`` SSE events); ``tool_calls`` is the list of
    structured tool_call dicts emitted across the stream.
    """
    events: list = []
    accumulated_content_parts: list[str] = []
    tool_calls: list = []
    previous = ""
    for delta in deltas:
        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            request=None,
        )
        events.append(result)
        if result is None:
            previous = current
            continue
        if "content" in result and result["content"]:
            accumulated_content_parts.append(result["content"])
        if "tool_calls" in result and result["tool_calls"]:
            tool_calls.extend(result["tool_calls"])
        previous = current
    return events, "".join(accumulated_content_parts), tool_calls


# ---------------------------------------------------------------------------
# Bug 1 — has_pending_tool_call covers bare Action: + trailing prefix
# ---------------------------------------------------------------------------


class TestHasPendingToolCall:
    """The fast-path gate that decides whether the postprocessor
    routes the delta through the full streaming parser. Pre-r6-B,
    the only "pending" signal was the strict ``_ACTION_LINE`` regex
    (``Action: verb(``); a delta carrying the bare ``Action: <verb>``
    token (no ``(`` yet) hit the False arm and was leaked as content.
    """

    def test_action_with_open_paren_still_pending(self, parser):
        # Original contract: open paren, no balanced close → pending.
        assert parser.has_pending_tool_call("Action: click(") is True

    def test_action_with_completed_call_not_pending(self, parser):
        # Original contract: balanced close → not pending.
        assert parser.has_pending_tool_call("Action: wait()") is False

    def test_bare_action_with_verb_pending(self, parser):
        # R6-C3 NEW: ``Action: type`` (no paren) MUST be pending so
        # the postprocessor's fast-path routes through the streaming
        # parser instead of leaking the bytes as content.
        assert parser.has_pending_tool_call("Action: type") is True

    def test_bare_action_no_verb_pending(self, parser):
        # ``Action:`` with no verb yet — about to emit a verb on the
        # next delta. Hold the buffer.
        assert parser.has_pending_tool_call("Action:") is True
        assert parser.has_pending_tool_call("Action: ") is True

    @pytest.mark.parametrize(
        "tail",
        ["A", "Ac", "Act", "Acti", "Actio", "Action"],
    )
    def test_trailing_action_prefix_pending(self, parser, tail):
        # R6-C3 NEW: trailing strict-prefix of ``Action:`` at the
        # buffer tail. e.g. ``"Thought: do X.\\nAc"`` — the next delta
        # might land ``"tion: click(...)"`` and we MUST hold the
        # candidate-opener bytes off the wire until then.
        text = f"Some prose.\n{tail}"
        assert parser.has_pending_tool_call(text) is True

    def test_trailing_non_action_prefix_not_pending(self, parser):
        # Trailing bytes that are NOT a prefix of ``Action:``. Plain
        # content — fast-path should let it through.
        assert parser.has_pending_tool_call("Thought: I'm done.") is False
        assert parser.has_pending_tool_call("Some prose ending in xyz.") is False

    def test_no_action_signal_not_pending(self, parser):
        # No Action: bytes anywhere, no trailing prefix candidate.
        assert parser.has_pending_tool_call("") is False
        assert parser.has_pending_tool_call("Thought: hi") is False
        assert parser.has_pending_tool_call("Hello world!") is False


# ---------------------------------------------------------------------------
# Bug 1 — streaming parser content channel never leaks Action: prefix
# ---------------------------------------------------------------------------


class TestStreamingActionHoldback:
    """The streaming parser's content channel — ``delta.content`` —
    must NEVER carry raw bytes that are about to become the prefix of
    a structured ``Action: verb(...)`` tool_call.
    """

    def test_no_action_prefix_leaks_into_content_single_chunk(self, parser):
        # The dogfood R2-6 shape: model emits ``"Action: type"`` in
        # one chunk then completes the args on the next. Pre-r6-B
        # the parser returned None (correctly held), but the
        # postprocessor's fast-path bypass leaked the bytes via
        # ``has_pending_tool_call`` returning False.
        deltas = ["Action: type", "(content='hi')"]
        events, content, tool_calls = _drive_stream(parser, deltas)
        assert "Action:" not in content, (
            f"delta.content leaked the Action: prefix: {content!r}"
        )
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "computer"
        assert '"action": "type"' in tool_calls[0]["function"]["arguments"]

    def test_no_action_prefix_leaks_token_by_token(self, parser):
        # Sub-character deltas around the Action: token. Each prefix
        # ``"A"``, ``"Ac"``, … must be held until the colon resolves.
        deltas = [
            "A",
            "c",
            "t",
            "i",
            "o",
            "n",
            ":",
            " ",
            "click",
            "(point='<point>10 20</point>')",
        ]
        events, content, tool_calls = _drive_stream(parser, deltas)
        assert "Action" not in content, (
            f"delta.content leaked Action prefix bytes: {content!r}"
        )
        assert "Acti" not in content
        assert len(tool_calls) == 1

    def test_no_action_prefix_leaks_with_prior_content(self, parser):
        # Plain content arrives first, THEN the partial Action prefix
        # accumulates on the tail. Only the trailing candidate-opener
        # bytes must be held; the preceding plain-content bytes pass
        # through normally.
        deltas = ["Some prose.\n", "Ac", "tion: wait()"]
        events, content, tool_calls = _drive_stream(parser, deltas)
        # The "Some prose.\n" must reach delta.content; the "Ac"
        # bytes must NOT (they're the held opener candidate).
        assert "Some prose." in content
        assert "Action" not in content
        assert len(tool_calls) == 1
        assert '"action": "wait"' in tool_calls[0]["function"]["arguments"]

    def test_action_completes_before_tool_calls_event(self, parser):
        # End-to-end: structured tool_calls event MUST fire by stream
        # end, and the accumulated content MUST be free of any
        # Action: bytes.
        deltas = [
            "Action: click",
            "(point='<point>500 300</point>')",
        ]
        events, content, tool_calls = _drive_stream(parser, deltas)
        assert "Action:" not in content
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "computer"
        assert '"action": "click"' in tool_calls[0]["function"]["arguments"]
        assert '"coordinate"' not in tool_calls[0]["function"]["arguments"]
        # Chat-completions OpenAI lane stays on the parser's native
        # ``point`` key (the Anthropic / Responses lanes do the
        # translation to ``coordinate``).
        assert '"point": [500, 300]' in tool_calls[0]["function"]["arguments"]

    def test_action_after_thought_no_leak(self, parser):
        # Realistic UI-TARS Computer-Use stream: ``Thought:`` preamble
        # → blank line → ``Action: verb(...)``. The reasoning parser
        # routes the Thought block to the reasoning channel BEFORE
        # the tool parser sees it, so the tool parser's input here is
        # whatever falls in the content channel — the ``Action:`` line
        # and its arguments. Both vectors must be held until the
        # tool_call fires.
        deltas = ["Action: ", "click", "(point='<point>500 300</point>')"]
        events, content, tool_calls = _drive_stream(parser, deltas)
        assert "Action:" not in content
        assert "click" not in content
        assert len(tool_calls) == 1

    def test_tool_choice_none_passthrough_unchanged(self, parser):
        # Defense-in-depth: tool_choice="none" short-circuit is the
        # OpenAI spec contract — even if the model emits ``Action:``
        # bytes, no tool_call event is allowed. The bytes ARE allowed
        # to leak through delta.content because the route documented
        # that as the contracted shape (text-only turn).
        deltas = ["Action: click()"]
        previous = ""
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=deltas[0],
            delta_text=deltas[0],
            request={"tool_choice": "none"},
        )
        # No tool_calls; bytes pass through as content per the
        # tool_choice="none" contract.
        assert result == {"content": "Action: click()"}


# ---------------------------------------------------------------------------
# Bug 1 — non-streaming path stays clean (regression guard)
# ---------------------------------------------------------------------------


class TestNonStreamingActionParity:
    """The non-streaming path already fully consumes the prefix before
    emitting; the r6-B fix is to bring the streaming path to parity.
    Test that the non-streaming path still works correctly so the
    fix didn't accidentally break it.
    """

    def test_non_stream_strips_action_prefix(self, parser):
        text = "Action: click(point='<point>10 20</point>')"
        result = parser.extract_tool_calls(text, request=None)
        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "computer"
        # Residual content is empty / None — the Action: prefix was
        # fully consumed.
        assert not result.content


# ---------------------------------------------------------------------------
# Codex r1 BLOCKING + HIGH — replay + word-boundary gate
# ---------------------------------------------------------------------------


class TestCodexBlockingHeldBytesReplayed:
    """Codex r1 BLOCKING: when a held trailing prefix DOES NOT become
    ``Action:`` on the next delta (the model emitted a non-action
    follow-up byte), the previously-held bytes MUST be released as
    content so the SSE stream stays bytes-equivalent to the model's
    output. Pre-fix the streaming parser dropped the held bytes
    permanently — e.g. ``["Ac", "me"]`` lost the ``"Ac"``.
    """

    @pytest.mark.parametrize(
        ("chunks", "expected_content"),
        [
            # The codex example: "Ac" then "me" → "Acme" content,
            # no tool_calls.
            (["Ac", "me"], "Acme"),
            # Longer prefix that resolves to non-action prose.
            (["Actio", "n is required."], "Action is required."),
            # Held single byte → resolved to non-action.
            (["A", "n answer"], "An answer"),
            # Prose then short partial prefix then continuation.
            (["Hello\nA", "B test"], "Hello\nAB test"),
            # Multi-stage hold + release across many deltas.
            (["Act", "io", "ns help"], "Actions help"),
        ],
    )
    def test_held_prefix_released_on_non_action_disambiguation(
        self, parser, chunks, expected_content
    ):
        events, content, tool_calls = _drive_stream(parser, chunks)
        assert tool_calls == []
        assert content == expected_content, (
            f"Held bytes were dropped: events={events!r}, "
            f"content={content!r}, expected={expected_content!r}"
        )

    def test_held_prefix_resolved_into_real_action(self, parser):
        # Held bytes that DO become Action: stay held until the
        # structured tool_calls event flushes them.
        chunks = ["Act", "ion: ", "wait()"]
        events, content, tool_calls = _drive_stream(parser, chunks)
        # No Action: bytes leak as content.
        assert "Action" not in content
        assert len(tool_calls) == 1
        assert '"action": "wait"' in tool_calls[0]["function"]["arguments"]


class TestCodexHighWordBoundaryGate:
    """Codex r1 HIGH: the trailing-prefix hold-back must be
    word-boundary aware. The action grammar is ``\\bAction:`` so a
    candidate-opener tail that ISN'T at a word boundary (e.g.
    ``"PlanA"`` — trailing ``"A"`` preceded by ``"n"``) can never
    become a real action and must not be held. Pre-fix the hold-back
    didn't check the preceding char's word-class.
    """

    @pytest.mark.parametrize(
        ("text", "expected_held"),
        [
            # Word-boundary-aligned candidates: SHOULD hold.
            ("Ac", 2),  # start of text
            ("\nAc", 2),  # preceded by \n — non-word
            (" Ac", 2),  # preceded by space — non-word
            ("Plan Acti", 4),  # space before "Acti" — non-word
            ("(Acti", 4),  # paren — non-word
            # Non-word-boundary candidates: MUST NOT hold.
            ("PlanA", 0),  # preceded by 'n' — word char
            ("123A", 0),  # preceded by '3' — word char
            ("foo_Ac", 0),  # preceded by '_' — word char
            # Pure non-prefix tails: not held regardless.
            ("China", 0),
            ("banana", 0),
            ("hello world", 0),
        ],
    )
    def test_trailing_prefix_word_boundary(self, text, expected_held):
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            _trailing_action_prefix_len,
        )

        assert _trailing_action_prefix_len(text) == expected_held, (
            f"text={text!r}: hold-back grammar regressed"
        )

    def test_streaming_passthrough_for_non_word_boundary_prefix(self, parser):
        # End-to-end: a stream that ends with a non-word-boundary
        # candidate (``"PlanA"``) MUST flush as plain content with no
        # hold-back, no replay, no leak.
        events, content, tool_calls = _drive_stream(parser, ["PlanA"])
        assert content == "PlanA"
        assert tool_calls == []


# ---------------------------------------------------------------------------
# Codex r2 BLOCKING — bare Action: + non-signature prose released as content
# ---------------------------------------------------------------------------


class TestCodexR2BareActionProseRelease:
    """Codex r2 BLOCKING: pre-fix, the streaming path treated every
    ``Action:`` occurrence as "in-flight action — hold buffer." That
    meant streams whose ``Action:`` token was followed by non-signature
    prose (``"Action: is required."``, ``["Act", "ion: item"]``) held
    the bytes indefinitely; ``len(cur_actions) <= len(prev_actions)``
    returned ``None`` forever and the postprocessor never received the
    bytes.

    The fix is two-layered:

    1. Mid-stream, ``_action_signature_could_complete`` discriminates
       "this Action: could still become Action: verb(" (held)
       from "this Action: is now provably prose" (released as content).
    2. At stream end, ``flush_held_content`` releases ANY still-held
       suffix — by definition the model committed to "no more tokens"
       so the held bytes are plain content.
    """

    def test_action_followed_by_prose_releases_mid_stream(self, parser):
        # ``Action: is required.`` — after ``Action: ``, the model emits
        # ``is required.`` which IS a valid ident (``is``) but then a
        # SPACE before any open-paren. Signature can't complete →
        # release.
        events, content, tool_calls = _drive_stream(parser, ["Action: is required."])
        assert content == "Action: is required."
        assert tool_calls == []

    def test_action_followed_by_punctuation_releases_mid_stream(self, parser):
        # ``Action: !`` — after the colon and whitespace, ``!`` is NOT
        # a valid ident start. Signature can never complete → release.
        events, content, tool_calls = _drive_stream(parser, ["Action: !"])
        assert content == "Action: !"
        assert tool_calls == []

    def test_action_followed_by_digit_releases_mid_stream(self, parser):
        # Digit can't start a verb-ident (Python rules) → release.
        events, content, tool_calls = _drive_stream(parser, ["Action: 42 items"])
        assert content == "Action: 42 items"
        assert tool_calls == []

    def test_bare_action_split_across_deltas_held_until_finalize(self, parser):
        # ``["Act", "ion: item"]`` — after delta 2 the buffer is
        # ``"Action: item"``. The signature "Action: item" could
        # still complete (no terminating char yet — ``item`` could
        # extend or the next char could be ``(``). Mid-stream MUST
        # hold; the postprocessor's ``finalize()`` calls
        # ``flush_held_content`` to release at stream end.
        accumulated_text = ""
        content_parts: list[str] = []
        previous = ""
        for delta in ["Act", "ion: item"]:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=delta,
                request=None,
            )
            if result and result.get("content"):
                content_parts.append(result["content"])
            previous = current
            accumulated_text = current
        # Mid-stream emit: nothing — bytes were held.
        assert content_parts == []
        # Finalize releases.
        held = parser.flush_held_content(accumulated_text)
        assert held == "Action: item"

    def test_action_with_completed_verb_paren_still_holds(self, parser):
        # Positive control: ``Action: type(`` mid-stream MUST stay held
        # — the signature DID commit to a real action, just no balanced
        # close yet. flush_held_content releases at stream end too
        # (the action is incomplete bytes, but we don't drop them).
        events, content, tool_calls = _drive_stream(parser, ["Action: type("])
        assert content == ""
        assert tool_calls == []
        # At finalize, the partial action bytes flush as content
        # (no balanced close → no structured tool_call to surface).
        held = parser.flush_held_content("Action: type(")
        assert held == "Action: type("

    def test_safe_emit_end_helper(self):
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import _safe_emit_end

        # No Action: anywhere → safe_end is at end (or before trailing prefix).
        assert _safe_emit_end("Hello world!") == len("Hello world!")
        # Trailing prefix: safe_end clips the partial-opener.
        assert _safe_emit_end("Hello\nAc") == len("Hello\n")
        # Action: with non-signature prose → safe_end at end (no hold).
        text = "Action: is required."
        assert _safe_emit_end(text) == len(text)
        # Action: with completable signature → safe_end at Action: start.
        text = "prose. Action: clic"
        assert _safe_emit_end(text) == len("prose. ")
        # Completed Action: verb(...) — handled by tool_calls path,
        # safe_end is at end of the completed action.
        text = "prose. Action: wait() done"
        # The completed action span ends at ``Action: wait()`` close;
        # bytes ``" done"`` after are residual content. safe_end
        # should be at len(text) — no in-flight signal AND no
        # trailing prefix.
        assert _safe_emit_end(text) == len(text)


class TestCodexR2FlushHeldContent:
    """End-of-stream flush hook: ``flush_held_content`` returns the
    suffix of the accumulated text that the streaming parser was still
    holding when the stream finished. Without this hook, the bytes
    held mid-stream would be permanently dropped from the response.
    """

    def test_no_action_no_held_bytes(self, parser):
        assert parser.flush_held_content("Plain content.") == ""

    def test_trailing_prefix_flushed_at_eos(self, parser):
        # Stream ended with ``"...thing\nAc"`` — the trailing prefix
        # never resolved, so flush as content.
        assert parser.flush_held_content("Hello\nAc") == "Ac"

    def test_inflight_action_flushed_at_eos(self, parser):
        # Stream ended with ``"Action: cli"`` — partial in-flight
        # action that didn't get the open paren. flush_held_content
        # releases the held bytes as content.
        assert parser.flush_held_content("Action: cli") == "Action: cli"

    def test_completed_action_does_not_flush_as_content(self, parser):
        # Completed action bytes are emitted as structured tool_calls,
        # NOT as content via flush_held_content.
        assert parser.flush_held_content("Action: wait()") == ""

    def test_prose_action_does_not_flush_as_content(self, parser):
        # ``Action:`` followed by definitely-non-signature prose was
        # already released mid-stream — flush returns empty.
        assert parser.flush_held_content("Action: is required.") == ""


# ---------------------------------------------------------------------------
# Codex r3 BLOCKING — held prefix + new action in same delta
# ---------------------------------------------------------------------------


class TestCodexR3HeldPrefixPlusNewAction:
    """Codex r3 BLOCKING: when a held trailing prefix later disambiguates
    in the SAME chunk that also completes a real action, the resolved
    prose-prefix MUST be emitted as content alongside the structured
    ``tool_calls`` event. Pre-fix, the new-action branch only walked
    the delta-window (``delta_start_in_current`` cursor) and never
    folded in the previously-held bytes — ``["Ac", "me Action: wait()"]``
    lost the ``"Ac"`` entirely.

    Fix: the residual walk now starts from ``prev_safe_end`` (the
    cursor up to which prior calls have already emitted) and ends at
    ``cur_safe_end`` (the new safe-emit boundary), folding the
    prev-held bytes into the content slice between the prior emit
    and the first new action.
    """

    def test_held_prefix_released_alongside_new_action(self, parser):
        # The codex repro shape: "Ac" held, then "me " disambiguates
        # to plain prose AND "Action: wait()" completes a tool call
        # all in the same second chunk.
        events, content, tool_calls = _drive_stream(parser, ["Ac", "me Action: wait()"])
        assert content == "Acme ", (
            f"Held 'Ac' was dropped when new action fired: events={events!r}"
        )
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "computer"
        assert '"action": "wait"' in tool_calls[0]["function"]["arguments"]

    def test_held_prefix_with_action_in_third_delta(self, parser):
        # Multi-step: held "Ac" → resolved "Acme " → real action.
        events, content, tool_calls = _drive_stream(
            parser, ["Ac", "me ", "Action: wait()"]
        )
        assert content == "Acme "
        assert len(tool_calls) == 1

    def test_held_prefix_resolves_via_real_action_token(self, parser):
        # Held "Ac" actually IS the start of a real Action: token.
        # No prose to release — all held bytes get folded into the
        # tool_call's span.
        events, content, tool_calls = _drive_stream(parser, ["Ac", "tion: wait()"])
        assert content == ""
        assert len(tool_calls) == 1
        assert '"action": "wait"' in tool_calls[0]["function"]["arguments"]

    def test_prev_held_prose_action_followed_by_real_action(self, parser):
        # Prev buffer had a prose-Action; new delta adds a real one.
        # The prose-Action bytes should already have flushed as content
        # on the first delta (because the signature was provably
        # incomplete), and the second delta only flushes the new
        # action.
        events, content, tool_calls = _drive_stream(
            parser, ["Action: is required.\n", "Action: wait()"]
        )
        assert content == "Action: is required.\n"
        assert len(tool_calls) == 1


# ---------------------------------------------------------------------------
# Codex r3 HIGH — has_pending_tool_call no longer reports prose as pending
# ---------------------------------------------------------------------------


class TestCodexR3HasPendingSemantics:
    """Codex r3 HIGH: ``has_pending_tool_call`` used to return True for
    every bare ``Action:`` in the buffer, even when the post-``Action:``
    tail was provably non-signature prose. The streaming parser
    already releases those bytes via ``_safe_emit_end`` (so it wasn't a
    data-loss bug after the r2 fix), but the public helper's contract
    was stale — the postprocessor's fast-path stayed on the slow
    route unnecessarily.

    Fix: case 2 of ``has_pending_tool_call`` now also runs
    ``_action_signature_could_complete`` and only reports True when
    the bare ``Action:`` could still become a real action signature.
    """

    def test_prose_action_not_pending(self, parser):
        # Codex r3 HIGH repro: provably-non-signature prose.
        assert parser.has_pending_tool_call("Action: is required.") is False
        assert parser.has_pending_tool_call("Action: !") is False
        assert parser.has_pending_tool_call("Action: 42 items") is False

    def test_real_in_flight_action_still_pending(self, parser):
        # Bare ``Action:`` whose signature COULD complete — held.
        assert parser.has_pending_tool_call("Action:") is True
        assert parser.has_pending_tool_call("Action: ") is True
        assert parser.has_pending_tool_call("Action: cli") is True
        # Verb fully written but no paren yet — held.
        assert parser.has_pending_tool_call("Action: click") is True

    def test_completed_action_not_pending(self, parser):
        # Existing contract preserved — completed action means
        # the tool_call has fired already.
        assert parser.has_pending_tool_call("Action: wait()") is False

    def test_open_paren_no_close_pending(self, parser):
        # Existing contract — verb+paren commit but no close.
        assert parser.has_pending_tool_call("Action: click(") is True


class TestCodexR4TrailingPrefixIsolated:
    """Codex r4 BLOCKING: ``_trailing_action_prefix_len`` used to
    short-circuit with ``if _ACTION_TOKEN in text: return 0``, which
    disabled trailing-prefix holdback for the entire buffer once any
    full ``Action:`` token appeared earlier. That re-opened R6-C3 for
    streams that mix prior prose/completed actions with a new split
    action opener — the trailing strict prefix leaked as content
    BEFORE the structured ``tool_calls`` flush.

    Fix: evaluate only the tail overlap; earlier ``Action:`` tokens
    (completed or prose) must not suppress holding a later trailing
    strict prefix.
    """

    def test_prose_action_then_split_real_action_holds_prefix(self, parser):
        # Codex r4 BLOCKING repro #1: ``["Action: is required.\nAc",
        # "tion: wait()"]``. The trailing ``Ac`` must be held on
        # delta 1 so chunk 2's ``tion: wait()`` resolves into the
        # structured tool_call without ``Ac`` leaking as content.
        parser.reset()
        chunks = ["Action: is required.\nAc", "tion: wait()"]
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        previous = ""
        for delta in chunks:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=delta,
                request=None,
            )
            if result and result.get("content"):
                content_parts.append(result["content"])
            if result and result.get("tool_calls"):
                tool_calls.extend(result["tool_calls"])
            previous = current
        final_content = "".join(content_parts)
        # The trailing ``Ac`` must NOT have leaked between deltas
        # — on chunk 1 the streaming parser should hold the strict
        # prefix; on chunk 2 the real action resolves.
        assert final_content == "Action: is required.\n", (
            f"unexpected content: {final_content!r}"
        )
        assert len(tool_calls) == 1
        assert '"action": "wait"' in tool_calls[0]["function"]["arguments"]

    def test_completed_action_then_split_real_action_holds_prefix(self, parser):
        # Codex r4 BLOCKING repro #2: ``["Action: wait() Ac",
        # "tion: click(point='<point>10 20</point>')"]``. The prior
        # completed action must not disable trailing-prefix holdback —
        # ``Ac`` is held until chunk 2 resolves it into the real
        # second tool_call.
        parser.reset()
        chunks = [
            "Action: wait() Ac",
            "tion: click(point='<point>10 20</point>')",
        ]
        content_parts: list[str] = []
        tool_calls: list[dict] = []
        previous = ""
        for delta in chunks:
            current = previous + delta
            result = parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=delta,
                request=None,
            )
            if result and result.get("content"):
                content_parts.append(result["content"])
            if result and result.get("tool_calls"):
                tool_calls.extend(result["tool_calls"])
            previous = current
        final_content = "".join(content_parts)
        # ``Ac`` must not appear; only the inter-action whitespace
        # surfaces as content.
        assert "Ac" not in final_content, f"trailing Ac leaked: {final_content!r}"
        assert final_content == " ", f"unexpected content: {final_content!r}"
        assert len(tool_calls) == 2
        assert '"action": "wait"' in tool_calls[0]["function"]["arguments"]
        assert '"action": "click"' in tool_calls[1]["function"]["arguments"]

    def test_trailing_prefix_helper_does_not_short_circuit_on_earlier_token(self):
        # Direct unit-level check of the fix — verify
        # ``_trailing_action_prefix_len`` ignores earlier full tokens
        # and only inspects the tail.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            _trailing_action_prefix_len,
        )

        # Earlier completed action — trailing ``Ac`` still held.
        assert _trailing_action_prefix_len("Action: wait() Ac") == 2
        # Earlier prose ``Action:`` — trailing ``Acti`` still held.
        assert _trailing_action_prefix_len("Action: is required.\nActi") == 4
        # No trailing strict prefix — returns 0 even with earlier token.
        assert _trailing_action_prefix_len("Action: wait() done") == 0

    def test_trailing_prefix_with_no_earlier_token_still_works(self):
        # Pre-existing behavior preserved for the simple case.
        from vllm_mlx.tool_parsers.ui_tars_tool_parser import (
            _trailing_action_prefix_len,
        )

        assert _trailing_action_prefix_len("hello\nAc") == 2
        assert _trailing_action_prefix_len("hello\nAction") == 6
        # Word-boundary gate still active: 'A' AFTER a word char (no
        # whitespace separation) isn't a candidate-opener.
        assert _trailing_action_prefix_len("PlanA") == 0
        # 'A' after a non-word char IS a candidate-opener — same word-
        # boundary semantics the streaming parser uses for ``\bAction:``.
        assert _trailing_action_prefix_len("Plan A") == 1
