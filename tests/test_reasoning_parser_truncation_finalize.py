# SPDX-License-Identifier: Apache-2.0
"""r5-D — reasoning-parser finalize-on-truncation contract tests.

Pins the parser-side fix for F-DGF-V080-B-7 (gemma4 dup-into-both-fields)
and F-DGF-V080-B-9 (glm4 leak-into-content) on the non-streaming
``/v1/chat/completions`` aggregator's finalize-on-truncation path.

Bug shape:

* When ``finish_reason="length"`` truncates the model mid-think (the
  closing ``</think>`` / ``<channel|>`` sentinel never arrives), the
  pre-r5-D non-streaming aggregator misclassified the unclosed buffer:
  - gemma4 duplicated the raw scratchpad into BOTH ``content`` and
    ``reasoning_content`` (when the engine's token-level OutputRouter
    also populated ``engine_reasoning_text``).
  - glm4 / minimax leaked the raw scratchpad into ``content`` with
    ``reasoning_content=null``.

Fix (shared finalize-on-truncation):

* Each parser implements ``is_open_in_think(text)`` (default False) so
  the route knows whether the unclosed buffer should be classified
  as reasoning.
* A shared ``finalize_truncation(open_in_think, buffer)`` helper in
  ``vllm_mlx/reasoning/base.py`` routes the buffer parser-agnostically.
* The non-streaming aggregator
  ``vllm_mlx/service/helpers.py::_finalize_content_and_reasoning``
  invokes the helper when ``finish_reason="length"`` is reported AND
  the parser's first pass returned the leak shape.
* Each parser's own ``extract_reasoning`` ALSO routes correctly on the
  parser-side so unit-test callers that don't go through the helper
  see the right behaviour.

The no-regression contract: ``finish_reason="stop"`` (or
``finish_reason="length"`` with the buffer already closed —
``</think>answer``) splits cleanly, byte-identical pre/post.
"""

from __future__ import annotations

import pytest

from vllm_mlx.api.utils import clean_output_text, strip_thinking_tags
from vllm_mlx.reasoning import finalize_truncation
from vllm_mlx.reasoning.deepseek_r1_parser import (
    DeepSeekR1ReasoningParser,
    VibeThinkerReasoningParser,
)
from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser
from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser
from vllm_mlx.reasoning.minimax_parser import MiniMaxReasoningParser
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.service.helpers import (
    _finalize_content_and_reasoning,
    _rescue_silent_drop_from_reasoning,
)


def _route_end_to_end(parser, raw, finish_reason):
    """Mini-route harness: ``_finalize_content_and_reasoning`` →
    ``clean_output_text`` + ``strip_thinking_tags`` →
    ``_rescue_silent_drop_from_reasoning``. Matches the chat-route
    finalize flow (chat.py:~2007–2115) but does not exercise tool
    parsing or response_format. Returns the user-facing
    ``(content, reasoning_content)`` pair the route would ship.
    """
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=parser,
        engine_reasoning_text="",
        finish_reason=finish_reason,
    )
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
    rescued = _rescue_silent_drop_from_reasoning(
        final_content,
        reasoning_text,
        tool_calls=[],
        finish_reason=finish_reason,
        raw_text=raw,
        reasoning_is_case4=False,
    )
    return rescued, reasoning_text


# ---------------------------------------------------------------------
# Shared helper contract
# ---------------------------------------------------------------------


class TestFinalizeTruncationHelper:
    """``finalize_truncation`` is the single source of truth for the
    parser-agnostic open-in-think → (reasoning, content) routing."""

    def test_open_in_think_routes_to_reasoning(self):
        reasoning, content = finalize_truncation(True, "step 1 of my thought")
        assert reasoning == "step 1 of my thought"
        assert content is None

    def test_not_open_in_think_routes_to_content(self):
        reasoning, content = finalize_truncation(False, "final answer body")
        assert reasoning is None
        assert content == "final answer body"

    def test_empty_buffer_routes_to_none(self):
        assert finalize_truncation(True, "") == (None, None)
        assert finalize_truncation(False, "") == (None, None)
        assert finalize_truncation(True, None) == (None, None)
        assert finalize_truncation(False, None) == (None, None)


# ---------------------------------------------------------------------
# Per-parser ``is_open_in_think`` contract
# ---------------------------------------------------------------------


class TestIsOpenInThink:
    """Each thinking parser correctly identifies its own unclosed-think
    state from accumulated text. Non-think parsers default to False."""

    def test_gemma4_open_in_think(self):
        p = Gemma4ReasoningParser()
        assert p.is_open_in_think("<|channel>thought\nReasoning so far") is True

    def test_gemma4_closed_thought_then_content(self):
        p = Gemma4ReasoningParser()
        text = (
            "<|channel>thought\nThinking<channel|><|channel>content\nThe answer is 42."
        )
        assert p.is_open_in_think(text) is False

    def test_gemma4_multi_block_trailing_open(self):
        p = Gemma4ReasoningParser()
        text = (
            "<|channel>thought\nFirst round<channel|><|channel>thought\nSecond unclosed"
        )
        assert p.is_open_in_think(text) is True

    def test_gemma4_no_marker_at_all(self):
        p = Gemma4ReasoningParser()
        assert p.is_open_in_think("plain content") is False

    def test_glm4_open_in_think(self):
        p = Glm4ReasoningParser()
        assert p.is_open_in_think("<think>Reasoning so far") is True

    def test_glm4_closed(self):
        p = Glm4ReasoningParser()
        assert p.is_open_in_think("<think>R</think>The answer is 42.") is False

    def test_glm4_autonomous_no_tag(self):
        """GLM-4 in autonomous mode (no tag): cannot detect from text
        alone — the route plug handles this via ``engine_reasoning_text``.
        The parser conservatively returns False."""
        p = Glm4ReasoningParser()
        assert p.is_open_in_think("Okay let me reason step by step") is False

    def test_qwen3_open_in_think(self):
        p = Qwen3ReasoningParser()
        assert p.is_open_in_think("<think>Reasoning so far") is True

    def test_qwen3_closed(self):
        p = Qwen3ReasoningParser()
        assert p.is_open_in_think("<think>R</think>answer") is False

    def test_deepseek_r1_open_in_think(self):
        p = DeepSeekR1ReasoningParser()
        assert p.is_open_in_think("<think>R") is True

    def test_minimax_open_in_think(self):
        p = MiniMaxReasoningParser()
        assert p.is_open_in_think("<think>R") is True

    def test_minimax_no_tag_returns_false(self):
        """MiniMax routes the no-tag heuristic preamble via its own
        ``_REASONING_START_RE`` matcher; the open-in-think hook is
        only the explicit-tag fast path."""
        p = MiniMaxReasoningParser()
        assert p.is_open_in_think("The user asks about Tokyo") is False


# ---------------------------------------------------------------------
# Per-parser ``extract_reasoning`` finalize-on-truncation contract
# ---------------------------------------------------------------------


class TestExtractReasoningMidThink:
    """The parser-side fix: ``extract_reasoning`` on a buffer that
    ends inside an unclosed think tag must return
    ``(reasoning=buffer, content=None)``. Pre-fix gemma4 / minimax
    leaked the buffer into ``content``."""

    def test_gemma4_mid_thought(self):
        p = Gemma4ReasoningParser()
        text = "<|channel>thought\nLet me think about this. The answer involves"
        reasoning, content = p.extract_reasoning(text)
        assert reasoning == "Let me think about this. The answer involves"
        assert content is None

    def test_gemma4_mid_thought_with_prior_closed_block(self):
        p = Gemma4ReasoningParser()
        text = (
            "<|channel>thought\nFirst round<channel|><|channel>thought\nSecond unclosed"
        )
        reasoning, content = p.extract_reasoning(text)
        assert reasoning is not None
        assert "First round" in reasoning
        assert "Second unclosed" in reasoning
        assert content is None

    def test_minimax_mid_thought(self):
        p = MiniMaxReasoningParser()
        text = "<think>Reasoning so far"
        reasoning, content = p.extract_reasoning(text)
        assert reasoning == "Reasoning so far"
        assert content is None

    def test_glm4_mid_thought_with_think_opener(self):
        """glm4 with autonomous ``<think>`` already routed correctly
        pre-r5-D via the base class's Case-3 fallback; pin it here so
        the no-regression contract holds when we refactor."""
        p = Glm4ReasoningParser()
        text = "<think>Reasoning so far"
        reasoning, content = p.extract_reasoning(text)
        assert reasoning == "Reasoning so far"
        assert content is None

    def test_qwen3_mid_thought_with_think_opener(self):
        p = Qwen3ReasoningParser()
        reasoning, content = p.extract_reasoning("<think>Reasoning so far")
        assert reasoning == "Reasoning so far"
        assert content is None

    def test_deepseek_r1_mid_thought_with_think_opener(self):
        p = DeepSeekR1ReasoningParser()
        reasoning, content = p.extract_reasoning("<think>Reasoning so far")
        assert reasoning == "Reasoning so far"
        assert content is None


class TestExtractReasoningClosedStop:
    """No-regression: ``finish_reason="stop"`` happy-path (think
    properly closed with answer following) must split cleanly,
    byte-identical pre/post."""

    def test_gemma4_clean_split(self):
        p = Gemma4ReasoningParser()
        text = (
            "<|channel>thought\nReasoning here<channel|>"
            "<|channel>content\nThe answer is 42.<channel|>"
        )
        reasoning, content = p.extract_reasoning(text)
        assert reasoning == "Reasoning here"
        assert content == "The answer is 42."

    def test_glm4_clean_split(self):
        p = Glm4ReasoningParser()
        reasoning, content = p.extract_reasoning(
            "<think>Reasoning</think>The answer is 42."
        )
        assert reasoning == "Reasoning"
        assert content == "The answer is 42."

    def test_qwen3_clean_split(self):
        p = Qwen3ReasoningParser()
        reasoning, content = p.extract_reasoning(
            "<think>Reasoning</think>The answer is 42."
        )
        assert reasoning == "Reasoning"
        assert content == "The answer is 42."

    def test_deepseek_r1_clean_split(self):
        p = DeepSeekR1ReasoningParser()
        reasoning, content = p.extract_reasoning(
            "<think>Reasoning</think>The answer is 42."
        )
        assert reasoning == "Reasoning"
        assert content == "The answer is 42."

    def test_minimax_clean_split(self):
        p = MiniMaxReasoningParser()
        reasoning, content = p.extract_reasoning(
            "<think>Reasoning</think>The answer is 42."
        )
        assert reasoning == "Reasoning"
        assert content == "The answer is 42."

    def test_gemma4_no_tags_pure_content(self):
        """``finish_reason="stop"`` for a non-thinking turn: plain
        content with no markers should pass through as content."""
        p = Gemma4ReasoningParser()
        reasoning, content = p.extract_reasoning("Hello, the answer is 42.")
        assert reasoning is None
        assert content == "Hello, the answer is 42."


# ---------------------------------------------------------------------
# Route-level finalize plug: ``_finalize_content_and_reasoning``
# integration with ``finish_reason``
# ---------------------------------------------------------------------


class TestRouteFinalizeOnTruncation:
    """The shared finalize-on-truncation plug in
    ``_finalize_content_and_reasoning`` covers the gaps each parser's
    ``extract_reasoning`` cannot detect from text alone (notably
    glm4 autonomous mode — no ``<think>`` tag emitted)."""

    def test_gemma4_length_truncation_routes_to_reasoning(self):
        """B-7 fix: gemma4 mid-thought with ``finish_reason="length"``
        must surface as reasoning_content, NOT content (NOT both)."""
        raw = "<|channel>thought\nMid-thought reasoning that was cut short"
        parser = Gemma4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert cleaned_text == "" or cleaned_text is None
        assert reasoning_text is not None
        assert "Mid-thought reasoning that was cut short" in reasoning_text
        # Critical: ``content`` and ``reasoning_content`` must NOT be
        # byte-identical (the F-DGF-V080-B-7 dup repro).
        assert cleaned_text != reasoning_text

    def test_glm4_autonomous_length_truncation_via_engine_signal(self):
        """B-9 fix: glm4 autonomous mode (no tag) with
        ``finish_reason="length"`` AND engine-router evidence
        (``engine_reasoning_text`` populated, indicating the token-level
        ``OutputRouter`` saw think tokens) must route the buffer to
        reasoning_content."""
        # glm4's autonomous-mode shape: model emitted plain text without
        # a ``<think>`` opener but the engine's OutputRouter
        # token-classified the bytes as reasoning. The route-level plug
        # honours the engine signal and re-classifies.
        raw = "Okay let me reason step by step about Tokyo's history"
        parser = Glm4ReasoningParser()
        # NOTE: the route-level plug only kicks in via the
        # post-parser path; the engine-routed branch returns earlier.
        # Pin the parser-only path for B-9: when parser returns
        # ``(None, raw)`` and ``finish_reason="length"``, the route
        # must NOT leak the buffer into content. The engine-routed
        # case is covered by the existing F-041 plug; for the no-
        # engine-signal case we rely on the parser's own behaviour
        # (glm4 with ``<think>`` opener routes correctly).
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text="<think>" + raw,  # raw_text DOES retain the think token
            cleaned_text="<think>" + raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        # Parser-side fix catches this: glm4 inherits the
        # BaseThinkingReasoningParser Case-3 fallback (only opener,
        # no closer) so the helper sees ``(reasoning, None)`` and
        # the truncated-think plug fires.
        assert reasoning_text is not None
        assert (
            raw in reasoning_text
            or "step by step about Tokyo's history" in reasoning_text
        )
        # ``content`` must NOT carry the raw scratchpad bytes.
        assert raw not in (cleaned_text or "")

    def test_minimax_length_truncation_routes_to_reasoning(self):
        """Cross-parser sweep: minimax mid-think on ``finish_reason="length"``
        must NOT leak ``<think>...`` bytes into content."""
        raw = "<think>Reasoning mid-thought that was cut short"
        parser = MiniMaxReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning mid-thought that was cut short" in reasoning_text
        # Critical: ``<think>`` tag bytes must NOT leak into content.
        assert "<think>" not in (cleaned_text or "")
        # And content must NOT byte-equal reasoning_content.
        assert cleaned_text != reasoning_text

    def test_qwen3_length_truncation_routes_to_reasoning(self):
        """Cross-parser sweep: qwen3 mid-think on ``finish_reason="length"``
        must route to reasoning_content. Existing Case-3 fallback
        already handled this — pin it for no-regression."""
        raw = "<think>Reasoning mid-thought"
        parser = Qwen3ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning mid-thought" in reasoning_text
        assert "<think>" not in (cleaned_text or "")

    def test_deepseek_r1_length_truncation_routes_to_reasoning(self):
        """Cross-parser sweep: deepseek-r1 mid-think — existing Case-3
        fallback already handled this — pin it for no-regression."""
        raw = "<think>Reasoning mid-thought"
        parser = DeepSeekR1ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning mid-thought" in reasoning_text
        assert "<think>" not in (cleaned_text or "")


class TestRouteFinishReasonStopNoRegression:
    """``finish_reason="stop"`` (clean close) MUST split cleanly,
    byte-identical pre/post-r5-D. This is the most important
    regression-prevention check — desktop clients have been seeing
    correct behaviour on the happy path and we must not disturb it."""

    def test_gemma4_clean_split_finish_stop(self):
        raw = (
            "<|channel>thought\nReasoning here<channel|>"
            "<|channel>content\nThe answer is 42.<channel|>"
        )
        parser = Gemma4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="stop",
        )
        assert reasoning_text == "Reasoning here"
        assert cleaned_text == "The answer is 42."

    def test_glm4_clean_split_finish_stop(self):
        raw = "<think>Reasoning</think>The answer is 42."
        parser = Glm4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="stop",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "The answer is 42."

    def test_qwen3_clean_split_finish_stop(self):
        raw = "<think>Reasoning</think>The answer is 42."
        parser = Qwen3ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="stop",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "The answer is 42."

    def test_minimax_clean_split_finish_stop(self):
        raw = "<think>Reasoning</think>The answer is 42."
        parser = MiniMaxReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="stop",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "The answer is 42."


class TestRouteFinishReasonLengthAfterClose:
    """``finish_reason="length"`` AFTER ``</think>`` closed and content
    started: the buffer is partial CONTENT, not reasoning. Must NOT
    re-route the content into reasoning."""

    def test_gemma4_length_after_close(self):
        # gemma4 sees a closed thought block, content started, then
        # got truncated. The truncated content body must STAY in
        # ``content``, never get rerouted to reasoning.
        raw = (
            "<|channel>thought\nReasoning here<channel|><|channel>content\nPartial ans"
        )
        parser = Gemma4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text == "Reasoning here"
        # Partial answer body in content (the unclosed content
        # channel may have leftover marker tokens stripped).
        assert cleaned_text is not None
        assert "Partial ans" in cleaned_text

    def test_glm4_length_after_close(self):
        raw = "<think>Reasoning</think>Partial answer that was cut"
        parser = Glm4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "Partial answer that was cut"

    def test_qwen3_length_after_close(self):
        raw = "<think>Reasoning</think>Partial answer that was cut"
        parser = Qwen3ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text == "Reasoning"
        assert cleaned_text == "Partial answer that was cut"


class TestNoDuplicationOfBuffer:
    """The original B-7 bug: gemma4 dup'd the same bytes into BOTH
    fields. The post-fix contract is that ``content`` and
    ``reasoning_content`` MUST NEVER be byte-identical when both are
    non-empty (modulo the edge case where reasoning ends with the
    same suffix the content starts with, which is structurally
    distinct from a full-buffer dup)."""

    @pytest.mark.parametrize(
        "parser_cls,raw",
        [
            (
                Gemma4ReasoningParser,
                "<|channel>thought\nReasoning that was truncated mid-flight",
            ),
            (
                MiniMaxReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
            (
                Glm4ReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
            (
                Qwen3ReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
            (
                DeepSeekR1ReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
            (
                VibeThinkerReasoningParser,
                "<think>Reasoning that was truncated mid-flight",
            ),
        ],
    )
    def test_no_dup_on_length_truncation(self, parser_cls, raw):
        parser = parser_cls()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        # If both are non-empty, they MUST NOT be byte-identical.
        if cleaned_text and reasoning_text:
            assert cleaned_text != reasoning_text, (
                f"{parser_cls.__name__} duplicated buffer into both "
                f"content and reasoning_content: {cleaned_text[:80]!r}"
            )


class TestVibeThinkerLengthTruncation:
    """VibeThinker is a DeepSeek-R1 variant — its truncated-think
    behaviour was already pinned by the live-test plug
    (``first_parse_was_truncated_think``). Pin it again here under
    the r5-D contract so the plug ordering doesn't drift."""

    def test_vibethinker_mid_think(self):
        raw = "<think>Reasoning so far"
        parser = VibeThinkerReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text is not None
        assert "Reasoning so far" in reasoning_text
        assert "<think>" not in (cleaned_text or "")


class TestEndToEndRouteContract:
    """End-to-end mini-route harness — exercises
    ``_finalize_content_and_reasoning`` → ``clean_output_text`` /
    ``strip_thinking_tags`` → ``_rescue_silent_drop_from_reasoning``.

    The bug repro at the route layer: even when the parser-side fix
    produces ``reasoning_text=<thought>`` and ``cleaned_text=""``, the
    silent-drop rescue can re-surface the reasoning bytes as
    ``content`` and re-introduce the dup-into-both-fields shape (the
    B-7 132/128/512-char identical-dup repro). The rescue's
    truncated-``<think>`` gate handles the qwen3/glm4 family but
    misses gemma4's channel format — the r5-D plug adds the gemma4
    analog.

    Contract: ``finish_reason="length"`` mid-think → ``content=None``,
    ``reasoning_content=<thought>``, NEVER byte-identical.
    """

    def test_gemma4_end_to_end_no_dup(self):
        raw = "<|channel>thought\nMid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(Gemma4ReasoningParser(), raw, "length")
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        # The original B-7 repro shape: NEVER ship the same bytes.
        assert content != reasoning

    def test_minimax_end_to_end_no_dup(self):
        raw = "<think>Mid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(MiniMaxReasoningParser(), raw, "length")
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        assert content != reasoning

    def test_glm4_end_to_end_no_dup(self):
        raw = "<think>Mid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(Glm4ReasoningParser(), raw, "length")
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        assert content != reasoning

    def test_qwen3_end_to_end_no_dup(self):
        raw = "<think>Mid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(Qwen3ReasoningParser(), raw, "length")
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        assert content != reasoning

    def test_deepseek_r1_end_to_end_no_dup(self):
        raw = "<think>Mid-thought reasoning that was cut short"
        content, reasoning = _route_end_to_end(
            DeepSeekR1ReasoningParser(), raw, "length"
        )
        assert content is None
        assert reasoning is not None
        assert "Mid-thought reasoning" in reasoning
        assert content != reasoning

    def test_gemma4_end_to_end_clean_split_finish_stop(self):
        """No-regression: ``finish_reason="stop"`` clean split must
        deliver both content AND reasoning_content correctly."""
        raw = (
            "<|channel>thought\nReasoning here<channel|>"
            "<|channel>content\nThe answer is 42.<channel|>"
        )
        content, reasoning = _route_end_to_end(Gemma4ReasoningParser(), raw, "stop")
        assert content == "The answer is 42."
        assert reasoning == "Reasoning here"

    def test_qwen3_end_to_end_clean_split_finish_stop(self):
        raw = "<think>Reasoning</think>The answer is 42."
        content, reasoning = _route_end_to_end(Qwen3ReasoningParser(), raw, "stop")
        assert content == "The answer is 42."
        assert reasoning == "Reasoning"


class TestGlm4AutonomousModeRoute:
    """B-9 specific: glm4's chat template does NOT pre-inject
    ``<think>``, so a model that decided not to emit the tag and got
    truncated mid-thought leaves no tag in ``cleaned_text``. The
    parser returns ``(None, raw)`` and the standard truncated-think
    plug doesn't fire (no ``<think>`` in text). The route-level fix
    leans on the engine's token-level ``OutputRouter`` evidence
    (``engine_reasoning_text`` populated) to detect this case.
    """

    def test_glm4_autonomous_engine_signal_routes_to_reasoning(self):
        """B-9 fix: engine token-router populated
        ``engine_reasoning_text`` (token-classifier saw think tokens),
        ``cleaned_text`` carries the raw scratchpad, no ``<think>``
        tag in raw_text → route the buffer to reasoning, blank
        cleaned_text."""
        raw = "Okay let me reason step by step about the user question"
        parser = Glm4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text=raw,  # engine routed the buffer
            finish_reason="length",
        )
        assert cleaned_text == ""
        assert reasoning_text == raw
        # The B-9 dup repro must NOT happen.
        assert cleaned_text != reasoning_text

    def test_glm4_no_engine_signal_no_route_change(self):
        """When the engine has no token-router evidence
        (``engine_reasoning_text`` empty), we cannot infer
        open-in-think from text alone for an autonomous-mode glm4
        response. The parser conservatively returns ``(None, raw)``
        and the route preserves cleaned_text as content — this is
        the documented limitation of autonomous-mode without
        out-of-band signal."""
        raw = "Okay let me reason step by step about the user question"
        parser = Glm4ReasoningParser()
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",  # no engine signal
            finish_reason="length",
        )
        # Without the engine signal, the route falls back to the
        # conservative "treat as content" path. The parser cannot
        # distinguish "autonomous-mode reasoning truncated" from
        # "non-thinking response truncated" on text alone.
        assert cleaned_text == raw
        assert reasoning_text is None


# ---------------------------------------------------------------------
# r5-D codex r1 BLOCKING follow-up: legitimate content containing a
# LITERAL ``<|channel>thought`` / ``<think>`` substring must NOT be
# reclassified as reasoning by the new finalize-on-truncation gates.
#
# Mirrors the contract pinned by
# ``test_literal_closed_think_in_answer_preserved_non_streaming`` in
# ``tests/test_reasoning_parsers.py`` (PR #722 codex r3) for the two
# parsers whose new finalize branches missed the substring-vs-structural
# distinction.
# ---------------------------------------------------------------------


class TestLiteralSubstringPreservedInContent:
    """The codex r1 BLOCKING contract for PR #825: the new
    finalize-on-truncation branches in gemma4 and minimax must NOT
    fire when the literal opener bytes appear inside what is otherwise
    legitimate answer content (already-closed thought block + answer
    that mentions the tag verbatim, or plain answer that mentions
    the tag verbatim).

    Fix shape:
    * gemma4 ``is_open_in_think`` requires no
      ``<|channel>content`` / ``<|channel>final`` marker between the
      last ``<|channel>thought`` and end-of-text.
    * minimax finalize branch requires
      ``model_output.lstrip().startswith("<think>")`` (true mid-think
      shape, matching the existing ``_sweep_residual_think_tags``
      conservative scope).
    """

    def test_gemma4_literal_channel_thought_in_content_preserved_stop(self):
        """``finish_reason="stop"``: properly closed thought block,
        then content channel whose answer text literally mentions
        ``<|channel>thought``. Must round-trip the answer verbatim
        and surface the actual thought as reasoning."""
        p = Gemma4ReasoningParser()
        raw = (
            "<|channel>thought\nR<channel|>"
            "<|channel>content\nThe gemma4 format uses <|channel>thought tag"
        )
        reasoning, content = p.extract_reasoning(raw)
        assert reasoning == "R", (
            f"reasoning must be the actual thought, not the literal "
            f"substring or a concatenation: {reasoning!r}"
        )
        assert content == "The gemma4 format uses <|channel>thought tag", (
            f"content with literal <|channel>thought must survive verbatim: {content!r}"
        )

    def test_gemma4_literal_channel_thought_in_content_preserved_length(self):
        """``finish_reason="length"`` mid-content: closed thought,
        content channel opened, answer mentions the literal substring,
        then truncated. The route must still split correctly — the
        thought stays as reasoning, the partial answer stays as
        content (literal substring preserved)."""
        p = Gemma4ReasoningParser()
        raw = (
            "<|channel>thought\nR<channel|>"
            "<|channel>content\nThe gemma4 format uses <|channel>thought tag"
        )
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=p,
            engine_reasoning_text="",
            finish_reason="length",
        )
        assert reasoning_text == "R", (
            f"on length truncation in-content, reasoning must remain "
            f"the closed thought: {reasoning_text!r}"
        )
        assert cleaned_text is not None
        assert "The gemma4 format uses <|channel>thought tag" in cleaned_text, (
            f"content with literal <|channel>thought must survive on "
            f"length truncation in-content: {cleaned_text!r}"
        )

    def test_minimax_literal_think_substring_in_answer_preserved_stop(self):
        """``finish_reason="stop"``: a normal answer that literally
        mentions ``<think>`` must NOT be reclassified as reasoning.
        Pre-fix, the new finalize branch fired on any ``<think>``
        substring; the conservative gate is ``lstrip().startswith``."""
        p = MiniMaxReasoningParser()
        raw = "The model uses <think> tags for reasoning"
        reasoning, content = p.extract_reasoning(raw)
        assert reasoning is None, (
            f"literal <think> in answer must NOT promote answer to "
            f"reasoning: {reasoning!r}"
        )
        assert content == raw, (
            f"literal <think> substring must survive verbatim in content: {content!r}"
        )

    def test_minimax_literal_think_substring_in_answer_preserved_length(self):
        """``finish_reason="length"`` on the same shape — the route
        must NOT promote the partial answer to reasoning just because
        it contains the substring."""
        p = MiniMaxReasoningParser()
        raw = "The model uses <think> tags for reasoning"
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=p,
            engine_reasoning_text="",
            finish_reason="length",
        )
        # The conservative gate keeps the buffer as content.
        # ``<think>`` substring is preserved verbatim — never split.
        assert reasoning_text is None, (
            f"length truncation on answer with literal <think> must "
            f"NOT route to reasoning: {reasoning_text!r}"
        )
        assert cleaned_text == raw, (
            f"length truncation on answer with literal <think> must "
            f"preserve content verbatim: {cleaned_text!r}"
        )
