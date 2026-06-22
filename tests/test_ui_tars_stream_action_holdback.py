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
