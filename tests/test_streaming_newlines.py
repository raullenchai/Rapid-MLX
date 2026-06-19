# SPDX-License-Identifier: Apache-2.0
"""
Tests for streaming markdown newline preservation and reasoning parser behavior.

Reproduces two bugs reported by users:
1. Markdown newlines stripped in streaming mode (whitespace-only chunks dropped)
2. Qwen3 reasoning parser eating all content when model doesn't use <think> tags

These tests exercise the reasoning parser directly, independent of the HTTP server.
"""

import pytest

from vllm_mlx.reasoning import get_parser


class TestQwen3NoTagStreaming:
    """Test Qwen3 parser behavior when model output has NO <think> tags.

    Bug: When --reasoning-parser qwen3 is active but the model doesn't
    emit <think> tags (e.g., 8-bit quantized models that think inline),
    ALL output goes to reasoning stream and content is empty.
    """

    @pytest.fixture
    def parser(self):
        return get_parser("qwen3")()

    def test_no_tags_streaming_corrected_by_finalize(self, parser):
        """When no <think> tags appear, finalize_streaming corrects to content.

        During streaming, the base parser defaults all output to reasoning
        (to support implicit think mode where </think> hasn't arrived yet).
        At stream end, finalize_streaming detects no tags were ever seen and
        emits the full text as a content correction.
        """
        parser.reset_state()

        text = "Hello! Here is a markdown example:\n\n# Heading\n\n- Item 1\n- Item 2\n"

        accumulated = ""
        reasoning_parts = []

        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result and result.reasoning:
                reasoning_parts.append(result.reasoning)

        # During streaming: everything goes to reasoning (correct — can't know
        # yet whether </think> will come)
        assert "".join(reasoning_parts) == text

        # finalize_streaming corrects: no tags seen → reclassify as content
        correction = parser.finalize_streaming(accumulated)
        assert correction is not None
        assert correction.content == text

    def test_no_tags_nonstreaming_is_fine(self, parser):
        """Non-streaming extraction correctly handles no-tag output."""
        text = "Hello! Here is a markdown example."
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_with_tags_still_works(self, parser):
        """Ensure fix doesn't break normal <think>...</think> flow."""
        parser.reset_state()

        tokens = ["<think>", "Let me think", "</think>", "The answer is 42."]
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        assert "Let me think" in "".join(reasoning_parts)
        assert "The answer is 42." in "".join(content_parts)

    def test_short_no_tags_finalized_as_content(self, parser):
        """Short no-tag output (under threshold) should be corrected by finalize."""
        parser.reset_state()

        text = "Short answer."
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        # finalize_streaming should emit correction
        correction = parser.finalize_streaming(accumulated)
        assert correction is not None
        assert correction.content == text

    def test_implicit_mode_still_works(self, parser):
        """Ensure fix doesn't break implicit mode (only </think> in output)."""
        parser.reset_state()

        tokens = ["thinking", " about ", "it", "</think>", "The answer."]
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        assert "thinking about it" in "".join(reasoning_parts)
        assert "The answer." in "".join(content_parts)


class TestNewlinePreservation:
    """Test that newline-only chunks survive the streaming pipeline.

    Bug: `\n` chunks were being dropped by whitespace suppression,
    breaking markdown formatting (headings, bullet lists, code blocks).
    """

    @pytest.fixture
    def parser(self):
        return get_parser("qwen3")()

    def test_newline_chunks_in_content(self, parser):
        """Newlines in content stream should not be dropped."""
        parser.reset_state()

        # Simulate: <think>ok</think>Hello\n\n# Heading\n
        tokens = ["<think>", "ok", "</think>", "Hello", "\n", "\n", "# Heading", "\n"]
        accumulated = ""
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result and result.content is not None:
                content_parts.append(result.content)

        full = "".join(content_parts)
        # Newlines should be preserved
        assert "\n\n" in full, f"Double newline lost in streaming. Got: {full!r}"
        assert "# Heading" in full

    def test_newline_only_delta_not_dropped(self, parser):
        """A delta that is exactly '\n' should produce content, not be skipped."""
        parser.reset_state()

        # After think tags, a \n-only delta should be content
        prev = "<think>x</think>Hello"
        delta = "\n"
        curr = prev + delta

        # First process up to "Hello" so parser knows we're past </think>
        accumulated = ""
        for char in prev:
            p = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(p, accumulated, char)

        # Now the \n delta
        result = parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None, "Newline delta should not be None"
        assert result.content == "\n", f"Expected content='\\n', got {result!r}"


class TestDeepSeekNoTagComparison:
    """Verify DeepSeek-R1 already handles no-tag case correctly (for reference)."""

    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_r1")()

    def test_no_tags_streaming_becomes_content(self, parser):
        """DeepSeek-R1 correctly switches to content after threshold."""
        parser.reset_state()

        text = "This is a regular response without any thinking tags. It should be content. "
        assert len(text) > parser.NO_TAG_CONTENT_THRESHOLD, (
            "test fixture must exceed threshold to exercise the flip"
        )
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        full_content = "".join(content_parts)
        # Once past the threshold, DeepSeek-R1 starts treating no-tag
        # deltas as content.
        assert len(full_content) > 0, "DeepSeek should have content for no-tag output"

    def test_vibethinker_preamble_before_think_routes_reasoning_correctly(self):
        """VibeThinker live-test regression (2026-06-17): the model emits a
        chatty multi-sentence preamble (~13 tokens, ~80 chars) BEFORE its
        ``<think>`` opener. With the base 64-char threshold, streaming
        flipped routing to ``content`` mid-preamble; by the time the
        literal ``<think>`` arrived, the whole reasoning trace leaked
        into ``content`` deltas.

        The fix is the ``vibethinker`` parser — a ``DeepSeekR1ReasoningParser``
        subclass with a 1024-char threshold (codex r2 P2 scoped narrowly
        to the VibeThinker family). The base ``deepseek_r1`` parser keeps
        its 64-char threshold to avoid widening the reasoning-buffer
        window globally for distilled-on-Qwen aliases that open with
        ``<think>`` immediately.

        This test pins the new ``vibethinker`` parser end-to-end through
        the parser registry so a future refactor that loses the
        registration trips here.
        """
        vibethinker_parser = get_parser("vibethinker")()
        vibethinker_parser.reset_state()
        assert vibethinker_parser.NO_TAG_CONTENT_THRESHOLD == 1024, (
            "vibethinker parser must register the larger 1024-char threshold"
        )

        # 80-char preamble (~13 tokens), then ``<think>...</think>``,
        # then the final answer. Mirrors the failing merge_intervals
        # case from the live test.
        preamble = "Okay, let me think about this carefully and work through it step by step.\n\n"
        assert 64 < len(preamble) < vibethinker_parser.NO_TAG_CONTENT_THRESHOLD, (
            "preamble must straddle the base 64-char threshold (fail-on-base "
            "guarantee) and stay under the vibethinker subclass's 1024-char "
            "threshold."
        )
        reasoning_body = (
            "<think>\nStep 1: scan the intervals. Step 2: merge overlaps.\n</think>"
        )
        answer = "def merge_intervals(intervals):\n    return sorted(intervals)"

        full_text = preamble + reasoning_body + answer
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for char in full_text:
            prev = accumulated
            accumulated += char
            result = vibethinker_parser.extract_reasoning_streaming(
                prev, accumulated, char
            )
            if result is None:
                continue
            if result.content:
                content_parts.append(result.content)
            if result.reasoning:
                reasoning_parts.append(result.reasoning)

        joined_reasoning = "".join(reasoning_parts)
        joined_content = "".join(content_parts)

        # The final answer (post-``</think>``) MUST land in content.
        assert "merge_intervals" in joined_content, (
            f"final answer leaked out of content. content={joined_content!r}"
        )
        # The reasoning trace from inside ``<think>...</think>`` MUST
        # land in reasoning_content (this is the live-test bug
        # signature: previously the trace leaked into content after the
        # 64-char flip).
        assert "scan the intervals" in joined_reasoning, (
            f"reasoning trace lost. reasoning={joined_reasoning!r}"
        )
        # And the final answer must NOT appear in reasoning_content.
        assert "merge_intervals" not in joined_reasoning, (
            "final answer leaked into reasoning_content"
        )
