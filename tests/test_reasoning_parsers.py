# SPDX-License-Identifier: Apache-2.0
"""Tests for reasoning parsers (base, think_parser, deepseek_r1, gpt_oss)."""

import pytest

from vllm_mlx.reasoning.base import DeltaMessage, ReasoningParser
from vllm_mlx.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from vllm_mlx.reasoning.gemma4_parser import Gemma4ReasoningParser
from vllm_mlx.reasoning.gpt_oss_parser import (
    _CHANNEL_RE,
    _STRUCTURAL_TOKENS,
    GptOssReasoningParser,
    _extract_channel,
)
from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from vllm_mlx.reasoning.minimax_parser import MiniMaxReasoningParser
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

# ---------------------------------------------------------------------------
# DeltaMessage
# ---------------------------------------------------------------------------


class TestDeltaMessage:
    def test_reasoning_only(self):
        dm = DeltaMessage(reasoning="thinking")
        assert dm.reasoning == "thinking"
        assert dm.content is None

    def test_content_only(self):
        dm = DeltaMessage(content="answer")
        assert dm.content == "answer"
        assert dm.reasoning is None

    def test_both(self):
        dm = DeltaMessage(reasoning="r", content="c")
        assert dm.reasoning == "r"
        assert dm.content == "c"

    def test_reasoning_content_alias(self):
        dm = DeltaMessage(reasoning="r")
        assert dm.reasoning_content == "r"

    def test_defaults(self):
        dm = DeltaMessage()
        assert dm.role is None
        assert dm.content is None
        assert dm.reasoning is None


# ---------------------------------------------------------------------------
# ReasoningParser (abstract base)
# ---------------------------------------------------------------------------


class TestReasoningParserBase:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ReasoningParser()

    def test_reset_state_noop(self):
        class Dummy(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output

            def extract_reasoning_streaming(self, prev, curr, delta):
                return None

        d = Dummy()
        d.reset_state()  # should not raise

    def test_finalize_streaming_default_none(self):
        class Dummy(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output

            def extract_reasoning_streaming(self, prev, curr, delta):
                return None

        d = Dummy()
        assert d.finalize_streaming("some text") is None


# ---------------------------------------------------------------------------
# BaseThinkingReasoningParser (via DeepSeek-R1 as concrete subclass)
# ---------------------------------------------------------------------------


class TestBaseThinkExtractReasoning:
    """Tests for extract_reasoning using DeepSeekR1ReasoningParser."""

    def setup_method(self):
        self.parser = DeepSeekR1ReasoningParser()

    def test_both_tags(self):
        text = "<think>step by step</think>The answer is 42."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "step by step"
        assert content == "The answer is 42."

    def test_both_tags_empty_reasoning(self):
        text = "<think></think>Just content"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "Just content"

    def test_both_tags_empty_content(self):
        text = "<think>reasoning only</think>"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "reasoning only"
        assert content is None

    def test_both_tags_whitespace_reasoning(self):
        text = "<think>   </think>content"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "content"

    def test_only_end_tag_implicit(self):
        text = "implicit reasoning</think>final answer"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "implicit reasoning"
        assert content == "final answer"

    def test_only_start_tag(self):
        text = "<think>incomplete reasoning without close"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "incomplete reasoning without close"
        assert content is None

    def test_no_tags_pure_content(self):
        text = "Just a simple response with no thinking."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    # ---- #575 — implicit-thinking truncation fallback -----------------

    def test_575_no_tags_enable_thinking_true_routes_to_reasoning(self):
        """The autoresearch repro: Qwen3 chat template pre-injected
        ``<think>\\n`` into the prompt, model was truncated mid-thought
        (``finish_reason="length"``), neither tag appears in output.
        Pre-#575 this leaked the whole thought to ``content``; post-fix
        it routes to ``reasoning`` symmetric with the streaming path.
        """
        text = (
            "Here's a thinking process that leads to the solution:\n\n"
            "1.  **Analyze the Problem:**\n"
            "    *   **Entities:** Two trains.\n"
            "    *   **Start Points:** Boston and New York.\n"
            "    [... 4000+ chars of pure thought ...]"
        )
        reasoning, content = self.parser.extract_reasoning(text, enable_thinking=True)
        assert reasoning == text.strip(), (
            "with enable_thinking=True the whole truncated trace MUST "
            "land in reasoning, not leak into content (Round-2 repro)"
        )
        assert content is None, (
            "content MUST be None on a truncated thought — empty "
            "assistant bubble in the UI > wall of meta-cognition"
        )

    def test_575_no_tags_enable_thinking_false_preserves_legacy_behaviour(self):
        """Backward-compat pin: passing ``enable_thinking=False``
        keeps the pre-#575 contract — no tags → content. Only the
        ``True`` path activates the new symmetric-with-streaming
        fallback; older callers that don't thread the flag at all
        (None) get the same legacy behaviour."""
        text = "Just a simple response with no thinking."
        for flag in (False, None):
            reasoning, content = self.parser.extract_reasoning(
                text, enable_thinking=flag
            )
            assert reasoning is None
            assert content == text

    def test_575_enable_thinking_true_does_not_affect_normal_split(self):
        """``enable_thinking=True`` MUST NOT change behaviour when the
        output already contains the closing tag — Case 2 (only end tag)
        is the well-behaved path that already routes correctly and the
        new flag must be a no-op there. Otherwise we'd silently swap
        ``reasoning`` and ``content`` on every successful thought."""
        text = "step by step reasoning</think>The answer is 42."
        reasoning, content = self.parser.extract_reasoning(text, enable_thinking=True)
        assert reasoning == "step by step reasoning"
        assert content == "The answer is 42."

    def test_575_empty_truncated_thought_routes_to_none(self):
        """A truncated thought that's only whitespace shouldn't ship as
        a non-empty reasoning string — ``.strip() or None`` returns
        None so callers don't render a placeholder reasoning bubble."""
        reasoning, content = self.parser.extract_reasoning(
            "   \n\t  ", enable_thinking=True
        )
        assert reasoning is None
        assert content is None

    def test_multiline_reasoning(self):
        text = "<think>Line 1\nLine 2\nLine 3</think>Answer"
        reasoning, content = self.parser.extract_reasoning(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert content == "Answer"

    def test_multiple_think_tags_uses_first_partition_only(self):
        # Codex r3-final scope (PR #722): the parser-level sweep only
        # handles the TRAILING unclosed ``<think>`` opener — the
        # round-1 fuzz repro shape on phi-4-mini-reasoning-4bit.
        # CLOSED ``<think>…</think>`` blocks in content are LEFT for
        # the downstream ``strip_thinking_tags`` regex which already
        # matches them (see ``vllm_mlx/api/utils.py::THINK_PATTERN``).
        # This preserves pre-PR behaviour for the rare case of an
        # answer that legitimately contains literal
        # ``<think>…</think>`` text.
        text = "<think>first</think>middle<think>second</think>end"
        reasoning, content = self.parser.extract_reasoning(text)
        # First-pair partition: reasoning = the first thought only.
        assert reasoning == "first"
        # Second closed block survives the parser sweep — the
        # downstream ``strip_thinking_tags`` (called from
        # ``routes/chat.py`` after the helper) is the right layer
        # to strip closed blocks. The first-partition leaves it
        # in content as ``"middle<think>second</think>end"``.
        assert "middle" in (content or "")
        assert "end" in (content or "")


# ---------------------------------------------------------------------------
# BaseThinkingReasoningParser streaming
# ---------------------------------------------------------------------------


class TestBaseThinkStreaming:
    def setup_method(self):
        self.parser = DeepSeekR1ReasoningParser()
        self.parser.reset_state()

    def test_skip_start_token(self):
        result = self.parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None

    def test_skip_end_token(self):
        result = self.parser.extract_reasoning_streaming(
            "<think>reasoning", "<think>reasoning</think>", "</think>"
        )
        assert result is None

    def test_reasoning_after_start(self):
        prev = "<think>"
        delta = "step 1"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == "step 1"
        assert result.content is None

    def test_content_after_end(self):
        prev = "<think>reasoning</think>"
        delta = "content"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.content == "content"
        assert result.reasoning is None

    def test_transition_in_delta(self):
        prev = "<think>reasoning"
        delta = " more</think>content"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == " more"
        assert result.content == "content"

    def test_both_tags_in_single_delta(self):
        prev = ""
        delta = "<think>reason</think>content"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == "reason"
        assert result.content == "content"

    def test_start_tag_only_in_delta(self):
        prev = ""
        delta = "<think>beginning"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == "beginning"

    def test_no_tags_early_defaults_to_reasoning(self):
        """Before any tags seen, base class defaults to reasoning."""
        prev = ""
        delta = "hello"
        curr = "hello"
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        # DeepSeek has threshold logic, but under threshold defaults to reasoning
        assert result.reasoning == "hello" or result.content == "hello"

    def test_implicit_end_only(self):
        """Implicit mode: </think> without <think>."""
        prev = "some reasoning"
        delta = "</think>answer"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        # Should transition from reasoning to content
        assert result is not None

    def test_reset_state(self):
        self.parser._saw_any_tag = True
        self.parser.reset_state()
        assert self.parser._saw_any_tag is False


# ---------------------------------------------------------------------------
# DeepSeekR1ReasoningParser specifics
# ---------------------------------------------------------------------------


class TestDeepSeekR1:
    def setup_method(self):
        self.parser = DeepSeekR1ReasoningParser()

    def test_tokens(self):
        assert self.parser.start_token == "<think>"
        assert self.parser.end_token == "</think>"

    def test_no_tag_threshold_constant(self):
        # Codex r2 P2 — kept at 64 on the base ``deepseek_r1`` parser
        # so distilled-on-Qwen aliases that open with ``<think>``
        # immediately don't pay the wider-buffer cost. The Qwen2-derived
        # VibeThinker family that needs a 1024-char window lives in
        # ``VibeThinkerReasoningParser`` (registered as ``vibethinker``).
        assert self.parser.NO_TAG_CONTENT_THRESHOLD == 64

    def test_no_start_only_end(self):
        """DeepSeek-R1 handles implicit start tag."""
        text = "thinking about it</think>42"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "thinking about it"
        assert content == "42"

    def test_no_tags_returns_content(self):
        text = "direct answer"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "direct answer"

    def test_standard_both_tags(self):
        text = "<think>r</think>c"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "r"
        assert content == "c"

    def test_streaming_no_tag_past_threshold(self):
        """After threshold chars without tags, treat as content."""
        self.parser.reset_state()
        long_text = "x" * 100
        result = self.parser.extract_reasoning_streaming("", long_text, long_text)
        assert result.content == long_text

    def test_streaming_no_tag_under_threshold(self):
        """Under threshold without tags, delegates to base (reasoning)."""
        self.parser.reset_state()
        short = "hi"
        result = self.parser.extract_reasoning_streaming("", short, short)
        assert result.reasoning == short

    def test_finalize_short_no_tag_correction(self):
        """Short output without tags gets corrected from reasoning to content."""
        self.parser.reset_state()
        self.parser._saw_any_tag = False
        result = self.parser.finalize_streaming("short answer")
        assert result is not None
        assert result.content == "short answer"

    def test_finalize_long_no_tag_no_correction(self):
        """Long output without tags: no correction (already handled by threshold)."""
        self.parser.reset_state()
        self.parser._saw_any_tag = False
        result = self.parser.finalize_streaming("x" * 100)
        assert result is None

    def test_finalize_with_tags_no_correction(self):
        """Output with tags: no correction needed."""
        self.parser.reset_state()
        self.parser._saw_any_tag = True
        result = self.parser.finalize_streaming("<think>r</think>c")
        assert result is None

    def test_finalize_empty_no_correction(self):
        self.parser.reset_state()
        result = self.parser.finalize_streaming("")
        assert result is None


class TestThinkParserSSEBoundary:
    """SSE-boundary withhold for split ``<think>`` / ``</think>`` tags
    (PR #715 bundle, fuzz finding C).

    The 2026-06-18 fuzz battery against PR #714 hit
    ``phi-4-mini-reasoning-4bit`` with streaming requests and observed
    ``content=">\\n", reasoning="<thinkOkay..."`` — the parser was
    splitting the literal ``<think>`` open tag across SSE chunk
    boundaries and falling through ``_handle_explicit_think``'s
    "treat as content" fallback for the trailing tag bytes.

    Tested via the ``DeepSeekR1ReasoningParser`` concrete class (the
    base parser is abstract); ``Qwen3ReasoningParser`` inherits the
    same streaming machinery and gets coverage via the
    inheritance-sensitive ``test_qwen3_sse_boundary_inherited``
    test below.
    """

    @staticmethod
    def _run_stream(parser, chunks):
        """Replay ``chunks`` through the parser's streaming interface
        exactly the way ``stream_chat_completion`` does and return
        ``(joined_reasoning, joined_content)``."""
        parser.reset_state()
        prev = ""
        reasoning = ""
        content = ""
        for ch in chunks:
            cur = prev + ch
            msg = parser.extract_reasoning_streaming(prev, cur, ch)
            if msg:
                if msg.reasoning:
                    reasoning += msg.reasoning
                if msg.content:
                    content += msg.content
            prev = cur
        return reasoning, content

    def test_start_tag_straddles_sse_boundary(self):
        """``<think>`` split as ``<thi`` / ``nk>`` MUST produce clean
        reasoning + clean content — no literal tag bytes in either
        channel. This is the exact phi-4-mini-reasoning fuzz repro."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<thi", "nk>", "Okay, ", "thinking.", "</think>", "Hello!"],
        )
        assert "<thi" not in reasoning, (
            f"partial start tag leaked into reasoning: {reasoning!r}"
        )
        assert "<thi" not in content, (
            f"partial start tag leaked into content: {content!r}"
        )
        assert reasoning == "Okay, thinking."
        assert content == "Hello!"

    def test_start_tag_split_one_char_at_a_time(self):
        """Worst-case: every character is its own SSE chunk. The parser
        must still reconstruct ``<think>`` from 7 one-char deltas
        without leaking any of the partial bytes."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            list("<think>") + ["Okay"] + ["</think>"] + ["Hi"],
        )
        assert reasoning == "Okay"
        assert content == "Hi"

    def test_end_tag_straddles_sse_boundary(self):
        """``</think>`` split as ``</thi`` / ``nk>`` MUST not leak the
        literal closing tag bytes into either channel."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<think>", "thinking", "</thi", "nk>", "answer"],
        )
        assert "</thi" not in reasoning, (
            f"partial end tag leaked into reasoning: {reasoning!r}"
        )
        assert "</thi" not in content
        assert reasoning == "thinking"
        assert content == "answer"

    def test_false_prefix_lt_recovered(self):
        """A lone ``<`` that turns out to NOT be a tag must be flushed
        into the stream on the next delta (not silently dropped).

        Regression for the held-buffer flush: an aggressive withhold
        without recovery would swallow user-visible characters."""
        parser = DeepSeekR1ReasoningParser()
        # Use the streaming interface directly so we don't hit the
        # NO_TAG_CONTENT_THRESHOLD path in the subclass.
        reasoning, _ = self._run_stream(parser, ["<", "angle bracket"])
        assert reasoning == "<angle bracket", (
            f"held '<' was dropped instead of flushed: {reasoning!r}"
        )

    def test_qwen3_sse_boundary_inherited(self):
        """Qwen3 parser inherits the streaming machinery and must get
        the same SSE-boundary safety as deepseek_r1."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<thi", "nk>", "Okay", "</think>", "Hello!"],
        )
        assert "<thi" not in reasoning
        assert "<thi" not in content
        assert reasoning == "Okay"
        assert content == "Hello!"

    def test_classic_in_delta_tag_unaffected(self):
        """Regression: when the entire ``<think>...</think>`` arrives
        in normal chunks (no straddle), behaviour must be unchanged."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<think>", "reasoning here", "</think>", "the answer"],
        )
        assert reasoning == "reasoning here"
        assert content == "the answer"

    def test_no_tags_at_all_streams_normally(self):
        """Regression: no-tag streams must still reach the Case-3
        fallback (reasoning) without being held indefinitely."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["plain ", "answer"],
        )
        # Case 3 routes no-tag to reasoning; finalize_streaming corrects
        # it. We only check the streamed bytes here are intact.
        assert reasoning == "plain answer"
        assert content == ""

    def test_false_partial_tag_inside_think_block_flushed(self):
        """Codex r1 P2 follow-up: when a ``<`` appears mid-reasoning
        and the next delta resolves it to non-tag content, the held
        ``<`` must be flushed back into reasoning. Without the
        recovery the byte was silently dropped — reasoning text
        ``2 < 5`` would render as ``2  5`` (the comparison operator
        eaten by the partial-tag withhold).

        This case fires inside an OPEN ``<think>`` block
        (``start_in_prev`` True), distinct from the Case-3 fallback
        path the prior tests cover. Codex flagged the asymmetry
        before merge."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<think>2 ", "<", " 5</think>", "ans"],
        )
        assert reasoning == "2 < 5", (
            f"false partial tag inside <think> block must flush — got "
            f"reasoning={reasoning!r}"
        )
        assert content == "ans"

    def test_in_think_block_with_lt_and_gt_chars(self):
        """Counter-test: reasoning that includes both ``<`` and ``>``
        characters mid-text (common in math / code) must not be
        garbled by the partial-tag withhold."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<think>x > 5 and y < 10", "</think>", "done"],
        )
        assert reasoning == "x > 5 and y < 10"
        assert content == "done"

    def test_split_opener_completed_with_reasoning_no_duplication(self):
        """Codex r2 P2 follow-up: when the start tag straddles SSE
        chunks AND the completing chunk also carries reasoning bytes
        (``['<thi', 'nk>Okay', ' more']``), the held suffix from the
        prior ``<thi`` delta must be cleared after consumption. A
        leftover held value caused the next ``start_in_prev`` delta to
        compute ``already_emitted_after_opener`` as if the held bytes
        were un-emitted reasoning, re-emitting the just-sent text and
        duplicating streamed reasoning."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<thi", "nk>Okay", " more", "</think>", "done"],
        )
        assert reasoning == "Okay more", (
            f"split opener + reasoning in same delta must not duplicate "
            f"reasoning on subsequent deltas — got reasoning={reasoning!r}"
        )
        assert content == "done"

    def test_split_opener_completes_with_partial_end_tag_no_leak(self):
        """Codex r4 P2 follow-up: when the same chunk that completes a
        split opener ALSO ends with a partial ``</think>``, the recovery
        branch must withhold the partial-end-tag suffix from the emit
        so the next chunk can complete the close. Otherwise the literal
        ``</thi`` bytes leak into ``reasoning_content`` and the client
        sees stray closing-tag bytes (live-fuzz repro shape:
        ``['<thi', 'nk>OK</thi', 'nk>ans']``)."""
        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<thi", "nk>OK</thi", "nk>ans"],
        )
        assert reasoning == "OK", (
            f"split opener + split closer in same completing delta must "
            f"not leak partial-end-tag bytes — got reasoning={reasoning!r}"
        )
        assert content == "ans", (
            f"content after split closer must be clean — got content={content!r}"
        )


class TestMultiBlockThinkStreaming:
    """Streaming-path multi-block ``<think>`` handling — companion to
    the non-streaming sweep in ``TestResidualThinkTagSweep``.

    The streaming postprocessor calls
    ``extract_reasoning_streaming`` per chunk and pre-fix the
    ``end_in_prev`` branch in ``_handle_explicit_think`` returned the
    whole delta as ``content`` regardless of whether a new
    ``<think>`` opener appeared. Phi-4-mini-reasoning emits 6–7
    ``<think>`` blocks across a single 2 K-token response when
    ``reasoning_max_tokens`` truncates the first one (2026-06-19
    round-1 fuzz), so the literal tag bytes streamed straight to the
    SSE consumer.

    Tests drive the parser directly with synthetic chunks (same
    ``_run_stream`` helper the SSE-boundary tests use) and assert
    no tag bytes survive on either channel.
    """

    @staticmethod
    def _run_stream(parser, chunks):
        parser.reset_state()
        prev = ""
        reasoning = ""
        content = ""
        for ch in chunks:
            cur = prev + ch
            msg = parser.extract_reasoning_streaming(prev, cur, ch)
            if msg:
                if msg.reasoning:
                    reasoning += msg.reasoning
                if msg.content:
                    content += msg.content
            prev = cur
        return reasoning, content

    def test_phi4_repro_second_block_after_answer(self):
        """The streaming analogue of the non-streaming phi-4-mini
        repro: ``<think>R1</think>answer<think>R2</think>tail``
        streamed in chunks must NOT leak ``<think>`` or ``</think>``
        bytes to ``content``."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        chunks = [
            "<think>",
            "thought1 with details",
            "</think>",
            "The answer is ",
            "\\boxed{Paris}",
            "<think>",
            "thought2 continues",
            "</think>",
            "tail",
        ]
        reasoning, content = self._run_stream(parser, chunks)
        assert "<think>" not in content, f"<think> leaked into content: {content!r}"
        assert "</think>" not in content, f"</think> leaked into content: {content!r}"
        assert "<think>" not in reasoning
        assert "</think>" not in reasoning
        assert "thought1 with details" in reasoning
        assert "thought2 continues" in reasoning
        assert "The answer is " in content
        assert "\\boxed{Paris}" in content
        assert "tail" in content

    def test_unclosed_second_block_truncated(self):
        """Second ``<think>`` is never closed (max_tokens hit) — the
        trailing reasoning must go to ``reasoning`` channel, not
        leak into ``content``."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        chunks = [
            "<think>",
            "first thought",
            "</think>",
            "Paris",
            "<think>",
            "second thought truncated",
        ]
        reasoning, content = self._run_stream(parser, chunks)
        assert "<think>" not in content
        assert "</think>" not in content
        assert "first thought" in reasoning
        assert "second thought truncated" in reasoning
        assert content == "Paris"

    def test_multi_block_qwen3_inherits(self):
        """Qwen3 inherits the multi-block fix via the base streaming
        machinery — same shape as the deepseek_r1 test above."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        chunks = [
            "<think>",
            "R1",
            "</think>",
            "answerA",
            "<think>",
            "R2",
            "</think>",
            "answerB",
        ]
        reasoning, content = self._run_stream(parser, chunks)
        assert "<think>" not in content
        assert "</think>" not in content
        assert reasoning == "R1R2"
        assert content == "answerAanswerB"

    def test_three_consecutive_blocks_streamed(self):
        """Three ``<think>…</think>`` blocks streamed back-to-back
        with intermediate content — all reasoning bytes accumulate
        on reasoning channel, all content bytes on content."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        chunks = [
            "<think>t1</think>A<think>t2</think>B<think>t3</think>C",
        ]
        reasoning, content = self._run_stream(parser, chunks)
        assert "<think>" not in content
        assert "</think>" not in content
        assert reasoning == "t1t2t3"
        assert content == "ABC"

    def test_normal_single_block_streaming_unchanged(self):
        """Regression: single-block streaming (the common case) must
        behave identically to pre-fix behaviour."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            ["<think>", "reasoning", "</think>", "answer"],
        )
        assert reasoning == "reasoning"
        assert content == "answer"

    def test_post_close_second_opener_straddles_sse_boundary(self):
        """Codex r1 BLOCKING on PR #722: the second ``<think>``
        opener split across SSE chunk boundaries (``A<thi`` then
        ``nk>R``) AFTER the first close was not recognised by the
        multi-block router because it scanned from ``prev_len``
        only — the held suffix from the prior chunk was missed
        and ``nk>R`` leaked as content with the closing tag
        bytes on the wire. Fix backs the scan window up by
        ``_held_tag_suffix_len`` so straddles span the boundary
        correctly. Symmetric to the existing single-block SSE
        boundary test."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        # First block closes, then content, then a SECOND
        # ``<think>`` opener split as ``<thi`` / ``nk>``.
        reasoning, content = self._run_stream(
            parser,
            [
                "<think>",
                "R1",
                "</think>",
                "answer",
                "<thi",  # opener withheld
                "nk>",  # opener completes
                "R2",
                "</think>",
                "tail",
            ],
        )
        assert "<think>" not in content, (
            f"split second-opener leaked into content: {content!r}"
        )
        assert "<thi" not in content, (
            f"partial tag bytes leaked into content: {content!r}"
        )
        assert "</think>" not in content
        assert reasoning == "R1R2"
        assert content == "answertail"

    def test_post_close_second_closer_straddles_sse_boundary(self):
        """Closer ``</think>`` split across the SSE boundary in a
        multi-block stream — must not leak the partial closer
        bytes into either channel."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            [
                "<think>",
                "R1",
                "</think>",
                "answer",
                "<think>",
                "R2",
                "</thi",  # closer withheld
                "nk>",  # closer completes
                "tail",
            ],
        )
        assert "<think>" not in content
        assert "</think>" not in content
        assert "</thi" not in content
        assert "</thi" not in reasoning
        assert reasoning == "R1R2"
        assert content == "answertail"

    def test_post_close_false_partial_opener_released_as_content(self):
        """Codex r2 NIT on PR #722: when held bytes from the prior
        chunk turn out to NOT complete a tag (e.g. prev ended with
        ``<thi`` but the next chunk is ``x tail`` not ``nk>``), the
        withheld bytes must be FLUSHED as content — not silently
        dropped. Regression for the held-suffix backtracking in
        ``_handle_multi_block_after_close``.

        The risky path: post-close phase, prev chunk's
        ``_held_partial_tag_len`` matched ``<thi`` as a potential
        ``<think>`` prefix, but the next chunk's first byte is ``x``
        so the partial-tag is a false alarm. Backtracking the scan
        window to ``prev_len - prev_held`` and starting ``cursor``
        there means the trailing-emit segment now spans those
        withheld bytes too — they flow back to the wire as
        content phase, matching what the user sees in the answer."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        reasoning, content = self._run_stream(
            parser,
            [
                "<think>",
                "R1",
                "</think>",
                "answer ",
                "<thi",  # partial-tag withheld
                "x tail",  # turns out NOT to be a tag — flush
            ],
        )
        assert reasoning == "R1"
        # The withheld ``<thi`` is released as content alongside the
        # following ``x tail`` so the client doesn't see truncated
        # answer text.
        assert content == "answer <thix tail", (
            f"false partial bytes not released cleanly: {content!r}"
        )

    def test_literal_closed_think_in_answer_preserved_non_streaming(self):
        """Codex r3 BLOCKING on PR #722: the conservative
        parser-level sweep MUST preserve a literal closed
        ``<think>…</think>`` substring inside the model's answer
        text. The pre-PR ``partition`` path already left such
        text in content (and the downstream ``strip_thinking_tags``
        regex stripped it), so this PR must not regress that
        behaviour.

        Note: the STREAMING router does still reclassify a literal
        ``<think>`` opener inside content as a structural opener —
        the streaming protocol has no closure-lookahead, so the
        bug is fundamental to text-based streaming. This
        non-streaming test pins the conservative scope for the
        path that DOES have closure lookahead.

        Streaming may regress this edge case but the round-1 fuzz
        repro (trailing unclosed opener after the answer) is the
        production-observed bug; literal ``<think>…</think>`` in
        content is a theoretical edge case the operator can
        observe via the existing ``strip_thinking_tags`` output."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "<think>R</think>The user said: <think>is literal</think> tag"
        )
        assert reasoning == "R"
        # Codex r4 BLOCKING on PR #722: the test must pin the EXACT
        # content so a future tightening of the conservative sweep
        # that strips the closed literal block fails this test
        # immediately. The parser leaves the closed literal block
        # verbatim — downstream ``strip_thinking_tags`` will strip
        # the tag wrapper but that is the operator-visible step,
        # not the parser-level contract this test pins.
        assert content == ("The user said: <think>is literal</think> tag"), (
            "literal closed <think>is literal</think> must survive "
            f"the conservative sweep verbatim: {content!r}"
        )

    def test_streaming_phase_uses_explicit_state_not_history_counts(self):
        """Codex r3 BLOCKING on PR #722: the multi-block router
        previously computed ``in_reasoning_prev`` from whole-history
        ``previous_text.count(<think>)`` vs ``count(</think>)``.
        A literal ``</think>`` substring inside already-emitted
        content (e.g. a closed pair that survived the conservative
        non-streaming sweep when echoed back in the answer, or a
        user prompt repeating tag-like text) inflated the close
        count and flipped the phase decision, causing subsequent
        structural reasoning chunks to leak into ``content``.

        Fix: track ``_streaming_phase`` as instance state, updated
        ONLY when this router crosses structural tags within the
        delta. This test pins that an already-emitted literal
        ``</think>`` does not corrupt the phase decision for a
        subsequent structural ``<think>`` block.
        """
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        # Stream: <think>R1</think> then content containing a
        # LITERAL </think> token (just text), then a structural
        # <think>R2</think> block. The router must keep R2 in
        # reasoning regardless of how many </think> bytes appear
        # in the already-emitted content.
        chunks = [
            "<think>",
            "R1",
            "</think>",
            "answer mentions </think> literally",
            "<think>",
            "R2",
            "</think>",
            "tail",
        ]
        reasoning, content = self._run_stream(parser, chunks)
        # Structural reasoning bytes are preserved.
        assert "R1" in reasoning
        assert "R2" in reasoning
        # No structural tag bytes leak via the router.
        # (The literal </think> in the answer text survives in
        # content — by the streaming protocol there is no way to
        # distinguish it from a structural close, and the test
        # ``test_literal_closed_think_in_answer_preserved_non_streaming``
        # documents the analogous limitation.)
        assert "R2" not in content

    def test_standalone_second_opener_immediately_after_first_close_seeds_phase(
        self,
    ):
        """Codex r4 BLOCKING on PR #722: when the SECOND ``<think>``
        opener arrives as a stand-alone SSE delta IMMEDIATELY after
        the first ``</think>`` (no intermediate content delta to
        trigger the multi-block router), ``_streaming_phase`` was
        still ``None`` because the single-block streaming path never
        sets it. The early-skip branch in
        ``extract_reasoning_streaming`` skipped the tag but did NOT
        seed the phase — so the next reasoning chunk entered the
        router with ``in_reasoning_prev = False`` (the documented
        fall-through when ``_streaming_phase is None``) and leaked
        the second reasoning block into ``content``.

        Fix: when the bare ``<think>`` delta arrives and the prior
        text already contains a ``</think>``, seed
        ``_streaming_phase = "reasoning"`` so the next delta routes
        the bytes back into reasoning. Symmetric seeding for the
        bare ``</think>`` after an opener.
        """
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        # Exact codex repro shape: tags as their OWN deltas with no
        # answer content between the first close and the second
        # opener.
        chunks = [
            "<think>",
            "R1",
            "</think>",
            "<think>",
            "R2_secret",
            "</think>",
            "tail",
        ]
        reasoning, content = self._run_stream(parser, chunks)
        # The second reasoning block MUST land in reasoning, not
        # content.
        assert "R2_secret" in reasoning
        assert "R2_secret" not in content
        # First-block reasoning still routed correctly.
        assert "R1" in reasoning
        # Trailing answer surfaces in content.
        assert "tail" in content

    def test_standalone_tags_phase_seed_does_not_affect_single_block(self):
        """Negative control for the codex r4 seed: in a NORMAL
        single-block streaming run (one ``<think>...</think>``
        followed by answer), the new seed must not flip the phase
        prematurely or alter the output. The seed only fires when
        the delta is the BARE tag — a typical reasoning chunk that
        merely contains the tag substring elsewhere is unaffected.
        """
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        chunks = ["<think>", "thoughts", "</think>", "answer here"]
        reasoning, content = self._run_stream(parser, chunks)
        assert "thoughts" in reasoning
        assert "thoughts" not in content
        assert "answer here" in content
        assert "answer here" not in reasoning


class TestResidualThinkTagSweep:
    """Multi-block ``<think>`` sweep (2026-06-19 round-1 fuzz repro).

    ``phi-4-mini-reasoning-4bit`` was observed emitting a SECOND
    ``<think>`` opener after the answer when ``reasoning_max_tokens``
    truncated mid-thought — the model returns to thinking mode after
    delivering ``\\boxed{Paris}`` and runs out of ``max_tokens`` before
    closing the second block. The naive first-pair partition in
    ``BaseThinkingReasoningParser.extract_reasoning`` consumed only the
    first ``<think>…</think>`` pair, leaving the trailing opener (and
    any subsequent closer-only sequences) literally in
    ``message.content``. Neither ``strip_thinking_tags`` (closed
    blocks only) nor ``sanitize_output`` (stray ``</think>`` only)
    catches an orphan ``<think>`` opener so the bytes survived to the
    wire.

    The sweep added in ``_sweep_residual_think_tags`` lives in the
    base class, so every thinking-tag parser (DeepSeek-R1, Qwen3,
    VibeThinker, …) inherits the fix. Tests drive each subclass's
    public ``extract_reasoning`` to keep coverage at the layer the
    route actually exercises.
    """

    def _check(self, parser, text, *, expected_content, expected_in_reasoning):
        reasoning, content = parser.extract_reasoning(text)
        # Hard guarantee: no literal tag bytes survive to either channel.
        assert "<think>" not in (content or ""), (
            f"<think> opener leaked into content: {content!r}"
        )
        assert "</think>" not in (content or ""), (
            f"</think> closer leaked into content: {content!r}"
        )
        assert "<think>" not in (reasoning or ""), (
            f"<think> opener leaked into reasoning: {reasoning!r}"
        )
        assert "</think>" not in (reasoning or ""), (
            f"</think> closer leaked into reasoning: {reasoning!r}"
        )
        if expected_content is not None:
            assert content == expected_content, (
                f"unexpected content: {content!r} (expected {expected_content!r})"
            )
        for needle in expected_in_reasoning:
            assert needle in (reasoning or ""), (
                f"expected {needle!r} in reasoning, got {reasoning!r}"
            )

    def test_phi4_repro_second_unclosed_block_after_answer(self):
        """The exact 2026-06-19 round-1 fuzz repro shape:
        ``<think>thought1</think>answer<think>thought2`` where the
        second block is truncated by ``max_tokens`` before its
        ``</think>``. Pre-fix this leaked ``<think>thought2…`` into
        ``message.content``; post-fix the trailing thought is
        appended to ``reasoning`` and content stops at the second
        opener."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        text = (
            "<think>thought1 with details</think>"
            "The answer is \\boxed{Paris}"
            "<think>\nthought2 trailing, truncated"
        )
        self._check(
            parser,
            text,
            expected_content="The answer is \\boxed{Paris}",
            expected_in_reasoning=["thought1", "thought2"],
        )

    def test_three_closed_blocks_no_trailing_unclosed(self):
        """Three closed ``<think>…</think>`` blocks all closed cleanly
        — the parser-level sweep is a NO-OP (no trailing unclosed
        opener) and the closed blocks reach downstream
        ``strip_thinking_tags`` which strips them. Codex r3-final
        scope: the parser deliberately does NOT touch closed blocks
        in content so a literal ``<think>…</think>`` substring (rare
        but possible in answer prose) is preserved at the parser
        layer."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        text = "<think>t1</think>A<think>t2</think>B<think>t3</think>C"
        reasoning, content = parser.extract_reasoning(text)
        # First-pair partition: reasoning is just t1.
        assert reasoning == "t1"
        # Remaining closed blocks survive in content for the
        # downstream regex to strip. The KEY invariant for this PR
        # is: no UNCLOSED ``<think>`` opener survives. Closed blocks
        # are downstream's responsibility.
        assert content is not None
        # No trailing unclosed opener at the parser layer (every
        # ``<think>`` in content has a matching ``</think>`` after it).
        last_open = content.rfind("<think>")
        if last_open >= 0:
            assert "</think>" in content[last_open + 7 :], (
                f"trailing unclosed <think> survived the sweep: content={content!r}"
            )

    def test_answer_text_before_trailing_unclosed_think_preserved(self):
        """Codex r4 BLOCKING on PR #722 (finding 1): codex flagged
        that the conservative sweep ``treats the last unclosed
        <think> in content as structural unconditionally``,
        claiming an answer like
        ``<think>R</think>The literal token is <think>``
        ``loses user-visible answer text``.

        This test pins the actual behaviour: the answer text BEFORE
        the trailing unclosed opener (``"The literal token is"``)
        IS preserved in content. Only the text AFTER the opener is
        rerouted to reasoning — that is the documented trade-off
        for fixing the production-observed phi-4-mini-reasoning
        leak (a real model emits a trailing ``<think>`` opener
        after the answer when ``reasoning_max_tokens`` truncates
        mid-thought; preserving those trailing bytes in content
        was the original bug). The trade-off favours the
        production case over the theoretical literal-tag case.
        """
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "<think>R</think>The literal token is <think>"
        )
        # First-pair reasoning preserved.
        assert reasoning == "R"
        # Answer text BEFORE the trailing unclosed opener is
        # preserved in content — rebuts the codex r4 finding-1
        # claim that user-visible answer text is lost.
        assert content == "The literal token is", (
            "answer text before the trailing unclosed opener must "
            f"survive in content: {content!r}"
        )

    def test_orphan_closer_left_for_downstream(self):
        """Codex r3-final scope (PR #722): a stray ``</think>``
        with no matching opener is LEFT for ``sanitize_output``
        (which already strips stray ``</think>`` via
        ``_FINAL_SANITIZER``). Earlier draft stripped it at the
        parser layer; codex r3 BLOCKING #2 pointed out that this
        layered guarantee was already in place downstream, and the
        parser layer doing it again risked obscuring an orphan
        closer that was actually a model artefact the operator
        wanted to debug. The parser leaves it for the sanitizer."""
        from vllm_mlx.api.utils import sanitize_output
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        text = "<think>thought</think>part1</think>part2"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning == "thought"
        # Stray </think> survives the parser sweep (intentional —
        # see the conservative scope note above).
        assert "</think>" in (content or "")
        # ``sanitize_output`` (the last-mile route filter) strips it.
        final = sanitize_output(content or "")
        assert "</think>" not in (final or ""), (
            f"downstream sanitiser failed to strip orphan closer: {final!r}"
        )

    def test_qwen3_inherits_sweep(self):
        """Qwen3 parser inherits the sweep via
        ``super().extract_reasoning`` — same multi-block shape as the
        phi-4-mini-reasoning repro."""
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        text = "<think>first thought</think>middle text<think>\nsecond truncated"
        self._check(
            parser,
            text,
            expected_content="middle text",
            expected_in_reasoning=["first thought", "second truncated"],
        )

    def test_vibethinker_inherits_sweep(self):
        """VibeThinker parser is a thin DeepSeek-R1 subclass —
        inherits the sweep transparently."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            VibeThinkerReasoningParser,
        )

        parser = VibeThinkerReasoningParser()
        text = "<think>vibe analysis</think>The answer<think>more thought"
        self._check(
            parser,
            text,
            expected_content="The answer",
            expected_in_reasoning=["vibe analysis", "more thought"],
        )

    def test_normal_single_block_unchanged(self):
        """Regression guard: the happy path (single closed block) must
        behave identically to pre-sweep behaviour — single-block
        outputs are by far the common case and the sweep must be a
        no-op for them."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )

        parser = DeepSeekR1ReasoningParser()
        reasoning, content = parser.extract_reasoning(
            "<think>thinking</think>final answer"
        )
        assert reasoning == "thinking"
        assert content == "final answer"

    def test_case2_implicit_think_orphan_closer_left_for_downstream(self):
        """Case 2 (chat template injects ``<think>`` so only
        ``</think>`` appears in output) plus a stray secondary
        ``</think>`` later in the content — the orphan closer is
        LEFT for ``sanitize_output`` (the last-mile route filter),
        matching the codex r3-final conservative scope."""
        from vllm_mlx.api.utils import sanitize_output
        from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        parser = Qwen3ReasoningParser()
        # Note: no leading <think> — it was injected in the prompt.
        text = "implicit reasoning</think>answer</think>tail"
        reasoning, content = parser.extract_reasoning(text)
        assert "<think>" not in (content or "")
        assert "implicit reasoning" in (reasoning or "")
        # Stray </think> survives the parser; downstream sanitizer
        # strips it. After ``sanitize_output`` the content reads
        # cleanly without tag bytes.
        final = sanitize_output(content or "")
        assert "</think>" not in (final or ""), (
            f"downstream sanitiser failed to strip orphan closer: {final!r}"
        )
        assert "answer" in (final or "")
        assert "tail" in (final or "")

    def test_reasoning_max_tokens_cap_no_tag_leak(self):
        """End-to-end through the route helper: when
        ``reasoning_max_tokens`` overflow is prepended to ``content``
        AND the parsed ``content`` carries the residue of multiple
        ``<think>`` blocks, the combined output stays tag-free.
        This is the integration shape that PR #715's
        ``_truncate_reasoning_only`` did NOT cover (its plug only
        fires for the single-truncated-thought case, not for
        ``<think>R1</think>answer<think>R2`` where R1 closes
        cleanly)."""
        from vllm_mlx.reasoning.deepseek_r1_parser import (
            DeepSeekR1ReasoningParser,
        )
        from vllm_mlx.service.helpers import (
            _finalize_content_and_reasoning,
        )

        parser = DeepSeekR1ReasoningParser()
        # Long first thought to trigger the reasoning cap (>400 chars).
        long_first = "thought1 " + "X" * 1000
        raw = (
            f"<think>{long_first}</think>"
            "\\boxed{Paris}"
            "<think>\nthought2 trailing, truncated by max_tokens"
        )
        cleaned_text, reasoning_text = _finalize_content_and_reasoning(
            raw_text=raw,
            cleaned_text=raw,
            tool_calls=[],
            reasoning_parser=parser,
            engine_reasoning_text="",
            enable_thinking=True,
            reasoning_max_tokens=100,  # 100 * 4 = 400-char cap
        )
        assert "<think>" not in (cleaned_text or ""), (
            f"opener leaked through cap: cleaned_text={cleaned_text!r}"
        )
        assert "</think>" not in (cleaned_text or ""), (
            f"closer leaked through cap: cleaned_text={cleaned_text!r}"
        )
        # The final answer survives (somewhere — overflow may prepend
        # part of the reasoning tail, but ``\\boxed{Paris}`` is intact
        # because the sweep restitched the content segments).
        assert "\\boxed{Paris}" in (cleaned_text or "")
        # Reasoning got capped at 400 chars and starts with the first
        # thought (sweep appended thought2 AFTER thought1 in emission
        # order).
        assert reasoning_text is not None
        assert len(reasoning_text) <= 400


# ---------------------------------------------------------------------------
# GptOssReasoningParser
# ---------------------------------------------------------------------------


class TestGptOssHelpers:
    def test_extract_channel_analysis(self):
        text = "<|channel|>analysis<|message|>my reasoning<|start|>assistant"
        result = _extract_channel(text, "analysis")
        assert result == "my reasoning"

    def test_extract_channel_final(self):
        text = "<|channel|>final<|message|>the answer<|return|>"
        result = _extract_channel(text, "final")
        assert result == "the answer"

    def test_extract_channel_not_found(self):
        text = "<|channel|>analysis<|message|>reasoning"
        result = _extract_channel(text, "final")
        assert result is None

    def test_extract_channel_empty_content(self):
        text = "<|channel|>analysis<|message|><|start|>"
        result = _extract_channel(text, "analysis")
        assert result is None

    def test_extract_channel_with_constrain(self):
        text = "<|channel|>final <|constrain|>JSON<|message|>content here<|return|>"
        result = _extract_channel(text, "final")
        assert result == "content here"

    def test_channel_regex_matches_analysis(self):
        text = "<|channel|>analysis<|message|>"
        m = _CHANNEL_RE.search(text)
        assert m is not None
        assert m.group(1) == "analysis"

    def test_channel_regex_matches_final(self):
        text = "<|channel|>final<|message|>"
        m = _CHANNEL_RE.search(text)
        assert m is not None
        assert m.group(1) == "final"

    def test_channel_regex_matches_constrain(self):
        text = "<|channel|>final <|constrain|>JSON<|message|>"
        m = _CHANNEL_RE.search(text)
        assert m is not None
        assert m.group(1) == "final"

    def test_structural_tokens_regex(self):
        for tok in [
            "<|start|>",
            "<|end|>",
            "<|channel|>",
            "<|return|>",
            "<|call|>",
            "<|constrain|>",
        ]:
            assert _STRUCTURAL_TOKENS.search(tok) is not None


class TestGptOssExtractReasoning:
    def setup_method(self):
        self.parser = GptOssReasoningParser()

    def test_full_format(self):
        text = (
            "<|channel|>analysis<|message|>Step by step reasoning"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "Step by step reasoning"
        assert content == "The answer is 42"

    def test_analysis_only(self):
        text = "<|channel|>analysis<|message|>just reasoning"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "just reasoning"
        assert content is None

    def test_final_only(self):
        text = "<|channel|>final<|message|>just content<|return|>"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "just content"

    def test_no_channels(self):
        text = "plain text without channels"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_empty_input(self):
        reasoning, content = self.parser.extract_reasoning("")
        assert reasoning is None
        assert content is None

    def test_none_like_empty(self):
        reasoning, content = self.parser.extract_reasoning("")
        assert reasoning is None

    def test_constrain_format(self):
        text = (
            "<|channel|>analysis<|message|>thinking"
            '<|start|>assistant<|channel|>final <|constrain|>JSON<|message|>{"key": "val"}<|return|>'
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "thinking"
        assert content == '{"key": "val"}'

    def test_structural_tokens_stripped(self):
        text = (
            "<|channel|>analysis<|message|>reason<|start|>"
            "<|channel|>final<|message|>answer<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert "<|" not in (reasoning or "")
        assert "<|" not in (content or "")


class TestGptOssStreaming:
    def setup_method(self):
        self.parser = GptOssReasoningParser()

    def test_detect_phase_init(self):
        assert GptOssReasoningParser._detect_phase("") == "init"
        assert GptOssReasoningParser._detect_phase("random text") == "init"

    def test_detect_phase_analysis(self):
        text = "<|channel|>analysis<|message|>reasoning"
        assert GptOssReasoningParser._detect_phase(text) == "analysis"

    def test_detect_phase_final(self):
        text = "<|channel|>analysis<|message|>r<|start|>assistant<|channel|>final<|message|>c"
        assert GptOssReasoningParser._detect_phase(text) == "final"

    def test_detect_phase_transition(self):
        text = "<|channel|>analysis<|message|>reason<|start|>"
        assert GptOssReasoningParser._detect_phase(text) == "transition"

    def test_streaming_analysis_phase(self):
        prev = "<|channel|>analysis<|message|>part1"
        delta = " part2"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.reasoning == " part2"

    def test_streaming_final_phase(self):
        prev = "<|channel|>analysis<|message|>r<|start|>assistant<|channel|>final<|message|>part1"
        delta = " part2"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.content == " part2"

    def test_streaming_phase_transition_to_analysis(self):
        prev = ""
        delta = "<|channel|>analysis<|message|>reasoning start"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.reasoning is not None
        assert "reasoning start" in result.reasoning

    def test_streaming_phase_transition_to_final(self):
        prev = "<|channel|>analysis<|message|>reason<|start|>assistant"
        delta = "<|channel|>final<|message|>content start"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.content is not None
        assert "content start" in result.content

    def test_streaming_init_phase_skips(self):
        prev = ""
        delta = "<|start|>"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is None

    def test_streaming_structural_token_stripped(self):
        prev = "<|channel|>analysis<|message|>reasoning"
        delta = "<|start|>"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        # Phase transitions to "transition", delta is structural → skip
        assert result is None or (
            result and "<|start|>" not in (result.reasoning or "")
        )

    def test_strip_return(self):
        assert GptOssReasoningParser._strip_return("text<|return|>") == "text"
        assert GptOssReasoningParser._strip_return("no return") == "no return"

    def test_extract_content_after_marker(self):
        text = "<|channel|>analysis<|message|>the content"
        result = GptOssReasoningParser._extract_content_after_marker_in_delta(
            text, "analysis"
        )
        assert result == "the content"

    def test_extract_content_after_marker_not_found(self):
        text = "<|channel|>analysis<|message|>content"
        result = GptOssReasoningParser._extract_content_after_marker_in_delta(
            text, "final"
        )
        assert result is None


# ---------------------------------------------------------------------------
# Full streaming simulation tests
# ---------------------------------------------------------------------------


class TestFullStreamingSimulation:
    """Simulate realistic streaming token-by-token delivery."""

    def test_think_parser_full_stream(self):
        """Simulate: <think>step 1\nstep 2</think>The answer."""
        parser = DeepSeekR1ReasoningParser()
        parser.reset_state()

        chunks = ["<think>", "step ", "1\n", "step 2", "</think>", "The ", "answer."]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "".join(reasoning_parts) == "step 1\nstep 2"
        assert "".join(content_parts) == "The answer."

    def test_deepseek_implicit_stream(self):
        """Simulate implicit mode: reasoning</think>content (no <think>)."""
        parser = DeepSeekR1ReasoningParser()
        parser.reset_state()

        chunks = ["reas", "oning", "</think>", "content"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "reas" in "".join(reasoning_parts)
        assert "content" in "".join(content_parts)

    def test_gpt_oss_full_stream(self):
        """Simulate GPT-OSS channel-based streaming."""
        parser = GptOssReasoningParser()

        chunks = [
            "<|channel|>analysis<|message|>",
            "reasoning ",
            "here",
            "<|start|>",
            "assistant",
            "<|channel|>final<|message|>",
            "the ",
            "answer",
            "<|return|>",
        ]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        reasoning_text = "".join(reasoning_parts)
        content_text = "".join(content_parts)
        assert "reasoning" in reasoning_text
        assert "answer" in content_text


# ---------------------------------------------------------------------------
# Qwen3ReasoningParser
# ---------------------------------------------------------------------------


class TestQwen3:
    def setup_method(self):
        self.parser = Qwen3ReasoningParser()

    def test_tokens(self):
        assert self.parser.start_token == "<think>"
        assert self.parser.end_token == "</think>"

    def test_both_tags(self):
        reasoning, content = self.parser.extract_reasoning(
            "<think>analysis</think>answer"
        )
        assert reasoning == "analysis"
        assert content == "answer"

    def test_only_end_tag(self):
        reasoning, content = self.parser.extract_reasoning(
            "implicit reasoning</think>answer"
        )
        assert reasoning == "implicit reasoning"
        assert content == "answer"

    def test_no_end_tag_pure_content(self):
        """Qwen3 overrides: if no end token at all, return as content."""
        reasoning, content = self.parser.extract_reasoning("just content")
        assert reasoning is None
        assert content == "just content"

    def test_only_start_tag_no_end(self):
        """Start tag without end tag: truncated thinking → reasoning, not content."""
        reasoning, content = self.parser.extract_reasoning("<think>incomplete")
        assert reasoning == "incomplete"
        assert content is None

    def test_empty_tags(self):
        reasoning, content = self.parser.extract_reasoning("<think></think>content")
        assert reasoning is None
        assert content == "content"

    # ---- #575 fast-path coverage (Qwen3 override branch) ----------------

    def test_575_qwen3_fast_path_no_tags_enable_thinking_true(self):
        """Qwen3's override has its own no-tag branch (not the base class
        Case 4). With ``enable_thinking=True`` it must also route to
        reasoning so the explicit + base paths stay in sync."""
        text = "implicit reasoning continuation"
        reasoning, content = self.parser.extract_reasoning(text, enable_thinking=True)
        assert reasoning == text
        assert content is None

    def test_575_qwen3_fast_path_no_tags_enable_thinking_false_legacy(self):
        text = "just content with no tags"
        for flag in (False, None):
            reasoning, content = self.parser.extract_reasoning(
                text, enable_thinking=flag
            )
            assert reasoning is None
            assert content == text

    # -----------------------------------------------------------------------
    # Bare-text "thinking process" preamble (issue #570).
    #
    # Qwen3 chat templates inject ``<think>\n`` after the assistant
    # generation marker when ``enable_thinking=True``. The model is
    # supposed to emit its chain-of-thought followed by ``</think>`` and
    # then the user-facing answer. Sometimes the model restates the
    # channel boundary inline as a bare-text prefix (``Here's a thinking
    # process:`` and variants); when that happens AND the model is
    # truncated by ``max_tokens`` before producing ``</think>``, neither
    # tag is in the output. The default branch then routes the whole
    # response — which is pure chain-of-thought — into ``content`` and
    # leaves ``reasoning_content`` empty, leaking reasoning to any
    # OpenAI-compatible client. These tests pin the bare-text fallback.
    # The fallback runs *after* the ``enable_thinking is True`` fast
    # path above, so it only fires when callers leave the kwarg
    # defaulted but the model still emits a bare-text think prefix.
    # -----------------------------------------------------------------------

    def test_bare_thinking_process_prefix_no_close_tag(self):
        text = (
            "Here's a thinking process:\n\n"
            "1.  **Analyze User Input:** route Seattle to San Diego.\n"
            "2.  **Evaluate Each Option (Food Scene Reputation):**\n"
            "   - Portland, OR: World-renowned food scene."
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None
        assert "thinking process" in reasoning
        # ``""`` (not ``None``) so the upstream finalize step overwrites
        # ``cleaned_text`` and the raw bare-text reasoning does not leak
        # through to the client's ``content`` field.
        assert content == ""

    def test_bare_thinking_process_variants(self):
        # Only the ``Here's [my/a/the] <scratchpad-noun>:`` shape (and
        # the ``My thought process:`` form) trigger the fallback. The
        # excluded shapes have their own regression pins below.
        for prefix in [
            "Here is my thinking process:",
            "Here is the chain-of-thought:",
            "Here's the thought process:",
            "Here's the scratchpad:",
            "My thought process:",
        ]:
            text = f"{prefix}\n\n1. First consider..."
            reasoning, content = self.parser.extract_reasoning(text)
            assert reasoning is not None, f"expected reasoning for prefix={prefix!r}"
            # ``""`` signals "overwrite to empty" to the finalize helper.
            assert content == "", f"expected empty content for prefix={prefix!r}"

    def test_thinking_verb_form_no_longer_matches(self):
        # Codex r5 BLOCKING regression pin: the verb-form
        # ``Thinking step by step:`` / ``Thinking out loud:`` /
        # ``Thinking through this:`` / ``Thinking carefully:`` /
        # ``Thinking aloud:`` are conversational answer openers
        # ("Thinking carefully: Portland is the safest option") and
        # would clobber valid responses on the default
        # ``enable_thinking=None`` code path. Only the noun-led
        # ``Here's [my/a/the] <noun>:`` shape stays in the regex —
        # the verb-led form is too conversational. This pin prevents
        # a future regex rewrite from re-adding them.
        for ambiguous in [
            "Thinking step by step: first drive south on I-5, then turn east.",
            "Thinking out loud: Portland has the best Vietnamese food.",
            "Thinking through this: the cheapest option is the train.",
            "Thinking carefully: Portland is the safest pick.",
            "Thinking aloud: I'd weight food culture higher than scenery.",
        ]:
            reasoning, content = self.parser.extract_reasoning(ambiguous)
            assert reasoning is None, (
                f"verb-form ``Thinking X:`` must no longer match — "
                f"clobbered direct answer: {ambiguous!r}"
            )
            assert content == ambiguous

    def test_bare_reasoning_label_no_longer_matches(self):
        # Codex r4 BLOCKING regression pin: ``reasoning`` (alone) and
        # ``reasoning process`` are excluded from the regex because
        # ``Here's my reasoning: …`` and ``My reasoning process: …``
        # are common direct-answer openers. Most callers default to
        # ``enable_thinking=None`` (legacy), so matching these labels
        # there would clobber valid answers on the busiest code path.
        # This test pins the exclusion so a future regex rewrite
        # cannot silently re-add them.
        for ambiguous in [
            "Here's my reasoning: Portland wins on food.",
            "Here is my reasoning: Pittsburgh outperforms in winter.",
            "Here is the reasoning: route through Salt Lake.",
            "Here is the reasoning process: sort then score.",
            "My reasoning process: weigh each option against criteria.",
        ]:
            reasoning, content = self.parser.extract_reasoning(ambiguous)
            assert reasoning is None, (
                f"bare ``reasoning(:|\\s+process:)`` must no longer "
                f"match — clobbered direct answer: {ambiguous!r}"
            )
            assert content == ambiguous

    def test_ambiguous_phrases_not_misclassified(self):
        # When ``enable_thinking=False`` (or the model otherwise emits a
        # direct answer), conversational openers like "Let me think" or
        # "I need to analyze" or "Analyzing the request" must NOT be
        # rerouted to ``reasoning_content`` — they are common answer
        # phrasings and clobbering them would leave the client with an
        # empty ``message.content``. Pinned per codex r1 BLOCKING on
        # PR #572. ``Step by step:`` / ``Step-by-step:`` added per
        # codex r2 — that bare form is the canonical heading for direct
        # "explain step by step" answers (tutorials, how-tos).
        for answer in [
            "Let me think about that — Portland is the best food stop.",
            "Let me analyze the options. The clear winner is San Francisco.",
            "Let me reason through this: Portland wins.",
            "I need to analyze the route first. The trip takes 7 days.",
            "I'll analyze each city: Portland has world-class food.",
            "I will think about this carefully — Portland wins.",
            "I should break this down: 1. Portland 2. San Francisco.",
            "Analyzing the user's request, the answer is Portland.",
            "Analyzing the question — the food capital is Portland.",
            "Step by step:\n1. Drive south on I-5\n2. Stop in Portland",
            "Step-by-step: first preheat the oven to 350F.",
        ]:
            reasoning, content = self.parser.extract_reasoning(answer)
            assert reasoning is None, (
                f"ambiguous phrase misclassified as reasoning: {answer!r}"
            )
            assert content == answer

    def test_bare_think_prefix_with_tool_call_markup_not_routed(self):
        # When the model embeds a tool call inside what looks like a
        # thinking preamble, the bare-text fallback must NOT echo the
        # raw output (tool markup and all) into ``reasoning_content``.
        # The tool parser already stripped tool tags from ``content``;
        # surfacing them in ``reasoning_content`` would leak the same
        # tags to clients via the reasoning channel. Pinned per codex
        # r2 BLOCKING on PR #572 — both branches (matched preamble +
        # tool tag in body) must defer to the upstream tool/text
        # pipeline and return ``(None, model_output)``.
        text_with_tool = (
            "Here's a thinking process:\n\n"
            'Need to call the weather API.\n<tool_call>\n{"name": '
            '"weather", "arguments": {"city": "Seattle"}}\n</tool_call>'
        )
        reasoning, content = self.parser.extract_reasoning(text_with_tool)
        assert reasoning is None, (
            "tool markup must not leak into reasoning_content via the "
            "bare-text fallback"
        )
        assert content == text_with_tool

        # All tool-tag flavors the rest of the stack recognises. The
        # preamble MUST match ``_BARE_THINK_PREFIX_RE`` first
        # (otherwise the bare-text branch wouldn't even consider
        # routing to reasoning, and the tool-markup detector would
        # never run — the loop would assert trivially). ``Here's the
        # reasoning:`` is excluded from the regex (codex r4), so
        # this loop uses ``Here's a thinking process:`` so each tag
        # flavor exercises ``_TOOL_CALL_MARKUP_RE`` for real (codex
        # r5 BLOCKING #2).
        for tag in [
            "<tool_call>",
            "<function=foo>",
            "<|tool_call|>",
            "<invoke ",
            "<minimax:tool_call>",
        ]:
            text = f"Here's a thinking process:\n\nThinking. {tag}stuff"
            reasoning, content = self.parser.extract_reasoning(text)
            assert reasoning is None, (
                f"tool tag {tag!r} should suppress bare-text fallback"
            )
            assert content == text

    def test_bare_thinking_prefix_with_close_tag_uses_normal_split(self):
        # When ``</think>`` IS present, the bare-text fallback must not
        # fire — the normal implicit-think split applies and the answer
        # after the close tag goes to ``content``.
        text = (
            "Here's a thinking process:\n1. think\n2. think more</think>"
            "The answer is Portland."
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None
        assert "thinking process" in reasoning
        assert content == "The answer is Portland."

    def test_bare_thinking_prefix_with_start_tag(self):
        # Explicit ``<think>`` in output already routes to reasoning via
        # the existing branch; the bare-text check must not interfere.
        text = "<think>Here's a thinking process: I should think harder."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None
        assert "Here's a thinking process" in reasoning
        assert content is None

    def test_normal_answer_not_misclassified_as_reasoning(self):
        # Answers that merely mention "thinking" mid-sentence must NOT
        # be reclassified as reasoning. The bare-text fallback matches
        # only at the very start of the output.
        for answer in [
            "Portland has the best food scene of those options.",
            "The answer is 42.",
            "```python\nprint('hi')\n```",
            "Yes, that's correct.",
            (
                "Sure! Portland is the standout for food. Many people think it's "
                "world-class — let me think of an example... Pok Pok was iconic."
            ),
        ]:
            reasoning, content = self.parser.extract_reasoning(answer)
            assert reasoning is None, f"misclassified as reasoning: {answer!r}"
            assert content == answer

    def test_finalize_streaming_bare_think_preamble_routes_to_reasoning(self):
        # Streaming counterpart: when the chat template injected
        # ``<think>`` and the model was truncated mid-thought before
        # ``</think>``, ``finalize_streaming`` previously emitted a
        # correction with the full text as ``content``. With the
        # bare-text fallback it surfaces in ``reasoning`` instead.
        parser = Qwen3ReasoningParser()
        accumulated = (
            "<think>Here's a thinking process:\n\n"
            "1. Analyze the user's request.\n"
            "2. Compare options."
        )
        result = parser.finalize_streaming(accumulated)
        assert result is not None
        assert result.reasoning is not None
        assert "thinking process" in result.reasoning
        assert result.content is None

    def test_finalize_streaming_close_tag_present_no_correction(self):
        parser = Qwen3ReasoningParser()
        result = parser.finalize_streaming(
            "<think>reasoning</think>The answer is Portland."
        )
        assert result is None

    def test_finalize_streaming_bare_preamble_without_think_prefix_routes_to_content(
        self,
    ):
        # Codex r3 BLOCKING symmetry: ``finalize_streaming`` has no
        # ``enable_thinking`` kwarg, so the leading ``<think>`` token
        # is the only evidence the stream is in thinking mode. Without
        # that evidence, a bare-text preamble in the accumulated text
        # is more likely a casual answer opener (the user asked the
        # model to "explain your thinking process") than an actual
        # truncated thought trace. Pre-fix the streaming side fired
        # the bare-text fallback regardless of context and silently
        # routed valid non-thinking answers into the dead-code
        # ``reasoning`` channel — the route consumer ignores
        # ``final_msg.reasoning`` so the answer never reached the
        # client. Symmetric with the explicit-False gate in
        # ``extract_reasoning``.
        parser = Qwen3ReasoningParser()
        result = parser.finalize_streaming(
            "Here's a thinking process I followed to solve the puzzle: "
            "first I sorted the items, then I picked the largest."
        )
        assert result is not None
        # ``content`` because the lack of a leading ``<think>`` means
        # the streaming Case-3 default reasoning emission was wrong
        # and the correction belongs in the content channel — the
        # same protocol the parser used pre-#570 for the no-evidence
        # branch.
        assert result.content is not None
        assert result.reasoning is None

    def test_bare_thinking_label_without_process_no_longer_matches(self):
        # Codex r3 BLOCKING: ``Here's my thinking:`` (no ``process``)
        # is normal user-facing phrasing — "Here's my thinking on
        # X..." — and the broader regex (``thinking(?:\s+process)?``)
        # generated false positives on direct answers. Tightened to
        # require ``thinking\s+process`` for the scratchpad-label
        # form. This test pins the regression so the optional ``\s+
        # process`` does not re-creep back into the regex.
        for ambiguous in [
            "Here's my thinking: Portland is the right pick for food.",
            "Here is my thinking: Pittsburgh outperforms in winter.",
            "Here is the thinking: route through Salt Lake first.",
        ]:
            reasoning, content = self.parser.extract_reasoning(ambiguous)
            assert reasoning is None, (
                f"bare ``thinking:`` (no ``process``) must no longer "
                f"match — clobbered direct answer: {ambiguous!r}"
            )
            assert content == ambiguous

    def test_extract_reasoning_explicit_false_skips_bare_preamble(self):
        # Codex r3 BLOCKING regression pin: when ``enable_thinking=False``
        # the caller has affirmatively said "no thinking is happening";
        # the bare-text fallback MUST defer and leave a valid answer
        # alone even if it opens with a scratchpad-shaped phrase. The
        # gate exists for legitimate teaching / tutorial content —
        # explaining a thinking-process methodology in non-thinking
        # mode is a real use case (``Here's a thinking process you
        # can use for any optimisation problem: …``) and must not be
        # reclassified as the model's own chain-of-thought.
        text = (
            "Here's a thinking process: first survey the available "
            "options, then score each one against your criteria, "
            "then pick the top result. This is a teaching answer "
            "the user explicitly asked for."
        )
        reasoning, content = self.parser.extract_reasoning(text, enable_thinking=False)
        assert reasoning is None
        assert content == text

    def test_extract_reasoning_unspecified_thinking_still_fires_fallback(self):
        # Mirror of the explicit-False test above: legacy callers that
        # don't thread ``enable_thinking`` at all (it stays ``None``)
        # still get defensive routing for bare-text preambles. The
        # gate is ``enable_thinking is not False`` so the None case
        # passes through and the pattern check decides.
        text = (
            "Here's a thinking process:\n"
            "1. Sort the options by relevance.\n"
            "2. Score each one against the criteria.\n"
        )
        reasoning, content = self.parser.extract_reasoning(text, enable_thinking=None)
        assert reasoning is not None
        assert "thinking process" in reasoning
        # ``""`` not None so the upstream finalize overwrites cleaned_text.
        assert content == ""


# ---------------------------------------------------------------------------
# Glm4ReasoningParser
# ---------------------------------------------------------------------------


class TestGlm4EnableThinking:
    """#575 codex R1 BLOCKING — GLM-4 does NOT prompt-inject ``<think>``,
    so the new ``enable_thinking`` kwarg must be a no-op on this parser
    even when ``True``. Otherwise legitimate no-tag GLM content gets
    silently re-routed to reasoning, diverging from streaming."""

    def setup_method(self):
        from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser

        self.parser = Glm4ReasoningParser()

    def test_no_tags_enable_thinking_true_still_routes_to_content(self):
        text = "GLM-4 plain answer with no think tags."
        reasoning, content = self.parser.extract_reasoning(text, enable_thinking=True)
        assert reasoning is None
        assert content == text

    def test_no_tags_enable_thinking_false_routes_to_content(self):
        text = "Another no-tag GLM response."
        for flag in (False, None):
            reasoning, content = self.parser.extract_reasoning(
                text, enable_thinking=flag
            )
            assert reasoning is None
            assert content == text


# ---------------------------------------------------------------------------
# MiniMaxReasoningParser
# ---------------------------------------------------------------------------


class TestMiniMaxExtractReasoning:
    def setup_method(self):
        self.parser = MiniMaxReasoningParser()

    def test_direct_content_code_block(self):
        text = "```python\nprint('hello')\n```"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_direct_content_json(self):
        text = '{"key": "value"}'
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_direct_content_tool_call(self):
        text = "<minimax:tool_call>some tool call"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_reasoning_pattern_english(self):
        text = "The user asks about Python.\n\nHere is the answer: Python is great."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None
        assert "user asks" in reasoning
        assert content is not None

    def test_reasoning_pattern_i_need(self):
        text = "I need to analyze this code.\n\nThe answer is 42."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None
        assert content is not None
        assert "answer" in content.lower()

    def test_reasoning_pattern_let_me(self):
        text = "Let me think about this.\n\nHere is the solution."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None
        assert content is not None

    def test_reasoning_pattern_chinese(self):
        text = "用户想知道Python怎么用。\n\n以下是答案。"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is not None

    def test_no_reasoning_pattern(self):
        text = "Python is a great language for beginners."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_explicit_think_tags(self):
        text = "<think>reasoning</think>content"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "reasoning"
        assert content == "content"

    def test_short_reasoning_not_stripped(self):
        """Very short 'reasoning' (<10 chars) treated as false positive."""
        text = "The user\n\nanswer"
        reasoning, content = self.parser.extract_reasoning(text)
        # "The user" is < 10 chars reasoning → returned as pure content
        assert content is not None

    def test_double_newline_split(self):
        text = "The user asks a question about Python.\n\nPython was created by Guido."
        reasoning, content = self.parser.extract_reasoning(text)
        # First part matches reasoning pattern, double newline splits
        assert reasoning is not None or content is not None


class TestMiniMaxStreaming:
    def setup_method(self):
        self.parser = MiniMaxReasoningParser()
        self.parser.reset_state()

    def test_reset_state(self):
        self.parser._decided = True
        self.parser._buffer = "stuff"
        self.parser.reset_state()
        assert self.parser._decided is False
        assert self.parser._buffer == ""
        assert self.parser._is_reasoning is False

    def test_explicit_think_tag_in_delta(self):
        result = self.parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None  # tag stripped, nothing left

    def test_explicit_think_tag_with_content(self):
        result = self.parser.extract_reasoning_streaming(
            "", "<think>reasoning", "<think>reasoning"
        )
        assert result.reasoning == "reasoning"

    def test_end_think_tag_transition(self):
        self.parser._decided = True
        self.parser._is_reasoning = True
        result = self.parser.extract_reasoning_streaming(
            "thinking", "thinking</think>answer", "</think>answer"
        )
        assert result.content == "answer"

    def test_buffering_phase(self):
        """Short text should be buffered (returns None)."""
        result = self.parser.extract_reasoning_streaming("", "hi", "hi")
        assert result is None

    def test_direct_content_detected_early(self):
        """Code blocks detected immediately as content."""
        result = self.parser.extract_reasoning_streaming(
            "", "```python\n", "```python\n"
        )
        assert result is not None
        assert result.content is not None

    def test_content_phase_passthrough(self):
        self.parser._decided = True
        self.parser._is_reasoning = False
        result = self.parser.extract_reasoning_streaming("prev", "prev more", " more")
        assert result.content == " more"

    def test_finalize_undecided(self):
        self.parser._decided = False
        result = self.parser.finalize_streaming("some short text")
        assert result is not None
        assert result.content == "some short text"

    def test_finalize_undecided_empty(self):
        self.parser._decided = False
        result = self.parser.finalize_streaming("")
        assert result is None

    def test_finalize_content_phase(self):
        self.parser._decided = True
        self.parser._is_reasoning = False
        result = self.parser.finalize_streaming("content")
        assert result is None

    def test_finalize_reasoning_reclassifies(self):
        self.parser._decided = True
        self.parser._is_reasoning = True
        result = self.parser.finalize_streaming("Just a simple answer")
        assert result is not None
        assert result.content is not None


# ---------------------------------------------------------------------------
# HarmonyReasoningParser
# ---------------------------------------------------------------------------


class TestHarmonyExtractReasoning:
    def setup_method(self):
        self.parser = HarmonyReasoningParser()

    def test_full_format(self):
        text = (
            "<|channel|>analysis<|message|>My reasoning here<|end|>"
            "<|channel|>final<|message|>The answer<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "My reasoning here"
        assert content == "The answer"

    def test_analysis_only(self):
        text = "<|channel|>analysis<|message|>reasoning only<|end|>"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "reasoning only"
        assert content is None

    def test_final_only(self):
        text = "<|channel|>final<|message|>answer only<|return|>"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "answer only"

    def test_no_channels(self):
        text = "plain text"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content is None

    def test_multiple_analysis_blocks(self):
        text = (
            "<|channel|>analysis<|message|>Block 1<|end|>"
            "<|channel|>analysis<|message|>Block 2<|end|>"
            "<|channel|>final<|message|>Answer<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert "Block 1" in reasoning
        assert "Block 2" in reasoning
        assert content == "Answer"


class TestHarmonyStreaming:
    def setup_method(self):
        self.parser = HarmonyReasoningParser()
        self.parser.reset_state()

    def test_reset_state(self):
        self.parser._current_channel = "analysis"
        self.parser._in_message = True
        self.parser.reset_state()
        assert self.parser._current_channel is None
        assert self.parser._in_message is False

    def test_analysis_channel_switch(self):
        result = self.parser.extract_reasoning_streaming(
            "", "<|channel|>analysis", "<|channel|>analysis"
        )
        assert result is None
        assert self.parser._current_channel == "analysis"

    def test_final_channel_switch(self):
        result = self.parser.extract_reasoning_streaming(
            "", "<|channel|>final", "<|channel|>final"
        )
        assert result is None
        assert self.parser._current_channel == "final"

    def test_commentary_channel_switch(self):
        result = self.parser.extract_reasoning_streaming(
            "", "<|channel|>commentary", "<|channel|>commentary"
        )
        # Commentary passes through as content for tool parser
        assert result is not None
        assert result.content == "<|channel|>commentary"
        assert self.parser._current_channel == "commentary"

    def test_message_start_skipped(self):
        self.parser._current_channel = "analysis"
        result = self.parser.extract_reasoning_streaming(
            "<|channel|>analysis", "<|channel|>analysis<|message|>", "<|message|>"
        )
        assert result is None
        assert self.parser._in_message is True

    def test_analysis_content_emitted(self):
        self.parser._current_channel = "analysis"
        self.parser._in_message = True
        result = self.parser.extract_reasoning_streaming(
            "<|channel|>analysis<|message|>",
            "<|channel|>analysis<|message|>reasoning",
            "reasoning",
        )
        assert result.reasoning == "reasoning"

    def test_final_content_emitted(self):
        self.parser._current_channel = "final"
        self.parser._in_message = True
        result = self.parser.extract_reasoning_streaming(
            "<|channel|>final<|message|>", "<|channel|>final<|message|>answer", "answer"
        )
        assert result.content == "answer"

    def test_end_token_stops_message(self):
        self.parser._current_channel = "analysis"
        self.parser._in_message = True
        result = self.parser.extract_reasoning_streaming(
            "<|channel|>analysis<|message|>r",
            "<|channel|>analysis<|message|>r<|end|>",
            "<|end|>",
        )
        assert result is None
        assert self.parser._in_message is False

    def test_return_token_stops_message(self):
        self.parser._current_channel = "final"
        self.parser._in_message = True
        result = self.parser.extract_reasoning_streaming(
            "<|channel|>final<|message|>c",
            "<|channel|>final<|message|>c<|return|>",
            "<|return|>",
        )
        assert result is None
        assert self.parser._in_message is False

    def test_commentary_passed_through(self):
        self.parser._current_channel = "commentary"
        self.parser._in_message = True
        result = self.parser.extract_reasoning_streaming(
            "prev", "prev tool_call", " tool_call"
        )
        # Commentary passes through as content for tool parser
        assert result is not None
        assert result.content == " tool_call"

    def test_control_tokens_skipped(self):
        result = self.parser.extract_reasoning_streaming("", "<|start|>", "<|start|>")
        assert result is None

    def test_full_streaming_simulation(self):
        parser = HarmonyReasoningParser()
        parser.reset_state()

        chunks = [
            "<|channel|>analysis",
            "<|message|>",
            "thinking ",
            "step 1",
            "<|end|>",
            "<|channel|>final",
            "<|message|>",
            "the ",
            "answer",
            "<|return|>",
        ]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "thinking" in "".join(reasoning_parts)
        assert "answer" in "".join(content_parts)


# ---------------------------------------------------------------------------
# Gemma4ReasoningParser
# ---------------------------------------------------------------------------


class TestGemma4Streaming:
    """Streaming behavior for Gemma4's <|channel>thought / <channel|> /
    <|channel>content channel format.

    The pre-#219 implementation classified the entire delta_text into one
    channel based on the channel state at the *end* of current_text. That
    worked when each delta was a single token (stream_interval=1) but
    misrouted bytes when stream_interval > 1 produced a buffered delta
    that straddled a channel marker.
    """

    def setup_method(self):
        self.parser = Gemma4ReasoningParser()
        self.parser.reset_state()

    def test_empty_delta_returns_none(self):
        result = self.parser.extract_reasoning_streaming("", "", "")
        assert result is None

    def test_no_channel_seen_defaults_to_content(self):
        delta = "hello world"
        result = self.parser.extract_reasoning_streaming("", delta, delta)
        assert result.content == "hello world"
        assert result.reasoning is None

    def test_thought_open_then_text_routes_to_reasoning(self):
        d1 = "<|channel>thought\n"
        m1 = self.parser.extract_reasoning_streaming("", d1, d1)
        # Marker-only delta: nothing to emit after stripping.
        assert m1 is None or (m1.reasoning is None and m1.content is None)

        d2 = "thinking step 1"
        prev = d1
        curr = prev + d2
        m2 = self.parser.extract_reasoning_streaming(prev, curr, d2)
        assert m2.reasoning == "thinking step 1"
        assert m2.content is None

    @pytest.mark.parametrize(
        "content_marker",
        ["<|channel>content", "<|channel>final"],
        ids=["content", "final"],
    )
    def test_delta_straddles_thought_close_then_content_open(self, content_marker):
        """Regression for issue #219.

        At stream_interval > 1 a single buffered delta can contain the tail of
        the thought channel, the channel-close marker, the content-channel-open
        marker (either <|channel>content or <|channel>final, since the parser
        treats final as a content-channel variant), and the start of the actual
        content. The pre-fix parser classified the entire delta as content
        (because state at end of current_text was in_content), so bytes before
        the close marker leaked from reasoning into content. This test asserts
        the split for both content and final markers.
        """
        prev = "<|channel>thought\nworking through it"
        self.parser.extract_reasoning_streaming("", prev, prev)
        assert self.parser._in_thought is True

        delta = f" final guess<channel|>{content_marker}\nThe answer is 42."
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == " final guess", (
            f"reasoning bytes from before the close marker should stay in "
            f"reasoning, got {result.reasoning!r}"
        )
        assert result.content == "The answer is 42.", (
            f"content bytes from after the {content_marker} marker should land "
            f"in content, got {result.content!r}"
        )
        assert self.parser._in_content is True
        assert self.parser._in_thought is False

    def test_delta_straddles_implicit_close_only(self):
        """Thought-close with no explicit content marker must still split,
        with the post-close bytes going to content (matches the original
        parser's implicit-content semantic)."""
        prev = "<|channel>thought\nreasoning"
        self.parser.extract_reasoning_streaming("", prev, prev)
        assert self.parser._in_thought is True

        delta = " done<channel|>plain answer"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == " done"
        assert result.content == "plain answer"
        assert self.parser._in_content is True

    def test_delta_with_no_marker_routes_whole_to_current_channel(self):
        """No marker in delta = original whole-delta dispatch (regression
        guard so the new split branch doesn't break the common case)."""
        prev = "<|channel>thought\nstart"
        self.parser.extract_reasoning_streaming("", prev, prev)
        delta = " more thinking text"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == " more thinking text"
        assert result.content is None

    def test_finished_content_phase_routes_to_content(self):
        """Once in content phase, deltas without markers route to content."""
        prev = "<|channel>thought\nx<channel|><|channel>content\nA"
        self.parser.extract_reasoning_streaming("", prev, prev)
        assert self.parser._in_content is True
        delta = "BC"
        result = self.parser.extract_reasoning_streaming(prev, prev + delta, delta)
        assert result.content == "BC"
        assert result.reasoning is None

    def test_reset_state(self):
        self.parser._in_thought = True
        self.parser._in_content = True
        self.parser._saw_any_channel = True
        self.parser.reset_state()
        assert self.parser._in_thought is False
        assert self.parser._in_content is False
        assert self.parser._saw_any_channel is False
