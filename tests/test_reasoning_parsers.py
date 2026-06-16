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

    def test_multiple_think_tags_uses_first(self):
        text = "<think>first</think>middle<think>second</think>end"
        reasoning, content = self.parser.extract_reasoning(text)
        # partition finds first occurrence
        assert reasoning == "first"
        assert "middle" in content


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
        # Only explicit scratchpad labels (terminating ``:``) trigger the
        # fallback. Conversational phrases like "Let me think..." are
        # deliberately NOT included — they're common direct-answer
        # openers when ``enable_thinking=False`` and matching them would
        # blank out valid responses (codex r1 BLOCKING on PR #572).
        # ``Step by step:`` / ``Step-by-step:`` are also deliberately
        # NOT included — that bare form is the canonical heading for
        # direct "explain step by step" answers (codex r2 BLOCKING).
        for prefix in [
            "Here is my thinking process:",
            "Here's my reasoning:",
            "Here is the chain-of-thought:",
            "Here's the thought process:",
            "Here is the reasoning process:",
            "Thinking step by step:",
            "Thinking out loud:",
            "Thinking through this:",
            "My thought process:",
            "My reasoning process:",
        ]:
            text = f"{prefix}\n\n1. First consider..."
            reasoning, content = self.parser.extract_reasoning(text)
            assert reasoning is not None, f"expected reasoning for prefix={prefix!r}"
            # ``""`` signals "overwrite to empty" to the finalize helper.
            assert content == "", f"expected empty content for prefix={prefix!r}"

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

        # All tool-tag flavors the rest of the stack recognises.
        for tag in [
            "<tool_call>",
            "<function=foo>",
            "<|tool_call|>",
            "<invoke ",
            "<minimax:tool_call>",
        ]:
            text = f"Here's the reasoning:\n\nThinking. {tag}stuff"
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
        # alone even if it opens with a scratchpad-shaped phrase.
        # Without this gate, ``Here's my reasoning: ...`` answers from
        # non-thinking-mode requests had ``content`` clobbered.
        text = (
            "Here's my reasoning: Portland has the strongest food "
            "scene of the three options on this route."
        )
        reasoning, content = self.parser.extract_reasoning(
            text, enable_thinking=False
        )
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
        reasoning, content = self.parser.extract_reasoning(
            text, enable_thinking=None
        )
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
