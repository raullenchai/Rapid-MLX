# SPDX-License-Identifier: Apache-2.0
"""
End-to-end streaming simulator that reproduces the server's SSE pipeline.

Simulates the full flow: model tokens → reasoning parser → content filtering
→ SSE emission → client-side reassembly.

Tests three scenarios matching real-world usage:
1. Normal Qwen3 with <think> tags (should work)
2. OpenCode implicit think mode (<think> in prompt, only </think> in output)
3. No-tag model (8-bit quantized, never emits <think> tags)
4. Markdown with newlines (the original bug report)

Each test checks both the streaming content AND the final assembled output.
"""

import pytest

from vllm_mlx.reasoning import get_parser

# Mirror SPECIAL_TOKENS_PATTERN from server.py
from vllm_mlx.api.utils import SPECIAL_TOKENS_PATTERN


def simulate_server_streaming_no_parser(tokens: list[str]) -> list[str]:
    """
    Simulate the server's standard streaming path WITHOUT a reasoning parser.

    This mirrors the actual logic in stream_chat_completion() lines 2636+
    (the `else` branch when _reasoning_parser is None):
    - Special tokens are filtered via SPECIAL_TOKENS_PATTERN
    - Empty-string content is filtered
    - No think buffer — think tags pass through to the client

    Returns:
        list of SSE content strings the client would receive
    """
    sse_chunks = []

    for i, token in enumerate(tokens):
        content = token

        # Filter special tokens (server.py line 2642)
        if content:
            content = SPECIAL_TOKENS_PATTERN.sub("", content)

        # Skip empty-string content (server.py line 2698)
        if content is not None and content == "":
            content = None

        # Compute finish reason
        is_finished = (i == len(tokens) - 1)
        finish_reason = "stop" if is_finished else None

        # Skip empty chunks (server.py line 2709)
        if not content and not finish_reason:
            continue

        if content:
            sse_chunks.append(content)

    return sse_chunks


def simulate_tokenizer_decode(tokens: list[str]) -> list[str]:
    """
    Simulate the multi-byte character guard in MLXLanguageModel.stream_generate().

    Each entry in `tokens` represents the FULL decode of all IDs so far
    (simulating tokenizer.decode(_generated_ids)). This tests the delta
    extraction logic with the U+FFFD guard.

    Args:
        tokens: List of cumulative decode results (what tokenizer.decode()
                returns as more IDs are appended).

    Returns:
        list of delta strings that would be yielded as StreamingOutput.text
    """
    _prev_raw_text = ""
    deltas = []

    for _raw_text in tokens:
        new_text = _raw_text[len(_prev_raw_text):]

        # Guard against multi-byte character boundaries (llm.py)
        if "\ufffd" in new_text:
            new_text = ""
        else:
            _prev_raw_text = _raw_text

        deltas.append(new_text)

    return deltas


def simulate_server_streaming(tokens: list[str], use_reasoning_parser: str = "qwen3"):
    """
    Simulate the server's streaming pipeline from server.py.

    This mirrors the actual logic in stream_chat_completion():
    - Reasoning parser extracts content/reasoning from each delta
    - Reasoning-only chunks are DROPPED (line 2615: `if not content: continue`)
    - finalize_streaming correction is emitted at the end
    - Empty-string content is filtered (line 2729)

    Returns:
        list of SSE content strings the client would receive
    """
    parser = get_parser(use_reasoning_parser)()
    parser.reset_state()

    accumulated_text = ""
    sse_chunks = []  # What the client receives as content

    for i, token in enumerate(tokens):
        delta_text = token
        is_finished = (i == len(tokens) - 1)

        previous_text = accumulated_text
        accumulated_text += delta_text

        delta_msg = parser.extract_reasoning_streaming(
            previous_text, accumulated_text, delta_text
        )

        if delta_msg is None:
            continue

        content = delta_msg.content
        reasoning = delta_msg.reasoning

        # Skip empty-string content (server.py line 2729)
        if content is not None and content == "":
            content = None

        # Server line 2615: skip if no content and not finished
        finish_reason = "stop" if is_finished else None
        if not content and not finish_reason:
            continue

        if content:
            sse_chunks.append(content)

    # Server line 2761: finalize_streaming correction
    if hasattr(parser, 'finalize_streaming'):
        correction = parser.finalize_streaming(accumulated_text)
        if correction and correction.content:
            sse_chunks.append(correction.content)

    return sse_chunks


class TestScenario1_ExplicitThinkTags:
    """Normal Qwen3 usage with <think>reasoning</think>content."""

    def test_basic_think_then_content(self):
        """Standard flow: think tags → reasoning extracted, content streamed."""
        tokens = ["<think>", "Let me ", "analyze.", "</think>", "The ", "answer ", "is 42."]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "The answer is 42." in full
        assert "<think>" not in full
        assert "analyze" not in full  # reasoning should NOT be in content

    def test_multiline_reasoning_then_content(self):
        tokens = [
            "<think>", "Step 1\n", "Step 2\n", "Step 3\n", "</think>",
            "# Result\n", "\n", "The answer is **42**.\n",
        ]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "# Result" in full
        assert "Step 1" not in full  # reasoning excluded
        assert "\n" in full  # newlines preserved


class TestScenario2_ImplicitThinkMode:
    """OpenCode injects <think> in prompt. Only </think> appears in output.

    This is the most critical scenario — OpenCode/Cursor/Continue.dev all do this.
    The model output starts with reasoning text (no <think>), then </think>, then content.
    """

    def test_short_implicit_reasoning(self):
        """Short reasoning (< 64 chars) before </think>."""
        tokens = ["Let ", "me ", "think.", "</think>", "Answer: ", "42"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "Answer: 42" in full
        assert "Let me think" not in full  # reasoning excluded from content

    def test_long_implicit_reasoning(self):
        """Long reasoning (> 64 chars) before </think>.

        This is the critical regression test — the NO_TAG_CONTENT_THRESHOLD
        must NOT kick in and start routing reasoning as content.
        """
        # Generate reasoning > 64 chars
        reasoning_tokens = [
            "Let me think about this problem step by step.\n",  # 47 chars
            "First, I need to consider the constraints.\n",     # 44 chars (total: 91)
            "Then apply the algorithm.\n",                       # 27 chars (total: 118)
            "Finally verify the result.\n",                      # 28 chars (total: 146)
        ]
        content_tokens = ["The ", "answer ", "is ", "42."]

        tokens = reasoning_tokens + ["</think>"] + content_tokens
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        # Content should have ONLY the actual content
        assert "The answer is 42." in full
        # Reasoning should NOT leak into content
        assert "Let me think" not in full, f"Reasoning leaked into content: {full!r}"
        assert "constraints" not in full, f"Reasoning leaked into content: {full!r}"
        assert "algorithm" not in full, f"Reasoning leaked into content: {full!r}"

    def test_very_long_implicit_reasoning(self):
        """Very long reasoning (> 500 chars) before </think>."""
        reasoning = "Analyzing step by step. " * 30  # ~720 chars
        # Break into tokens
        reasoning_tokens = [reasoning[i:i+20] for i in range(0, len(reasoning), 20)]
        content_tokens = ["Here is the answer."]

        tokens = reasoning_tokens + ["</think>"] + content_tokens
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        assert "Here is the answer." in full
        assert "Analyzing" not in full, f"Reasoning leaked: {full!r}"


class TestScenario3_NoTagModel:
    """Model never emits <think> tags (e.g., 8-bit quantized Qwen3).

    This is the original user-reported bug: content is empty because
    the reasoning parser classifies everything as reasoning.
    """

    def test_short_no_tag_output(self):
        """Short output (< 64 chars) with no tags at all."""
        tokens = ["Hello ", "world!"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert full == "Hello world!", f"Expected 'Hello world!', got {full!r}"

    def test_long_no_tag_output(self):
        """Long output (> 64 chars) with no tags — the core bug."""
        text = "Here is a markdown example:\n\n# Heading\n\n- Item 1\n- Item 2\n\nDone."
        tokens = [text[i:i+5] for i in range(0, len(text), 5)]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        assert "# Heading" in full, f"Content missing: {full!r}"
        assert "- Item 1" in full, f"Content missing: {full!r}"
        assert "Done." in full, f"Content missing: {full!r}"
        # ALL text should be in content, nothing lost
        assert full == text, f"Content mismatch:\n  Expected: {text!r}\n  Got:      {full!r}"

    def test_very_long_no_tag_output(self):
        """Very long output with no tags — no chars should be lost."""
        text = "The quick brown fox. " * 20  # 420 chars
        tokens = [text[i:i+10] for i in range(0, len(text), 10)]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert full == text, f"Length mismatch: expected {len(text)}, got {len(full)}"

    def test_no_tag_with_newlines(self):
        """No-tag output with markdown newlines — the original Reddit bug."""
        tokens = [
            "# Title", "\n", "\n",
            "Some text.", "\n", "\n",
            "- bullet 1", "\n",
            "- bullet 2", "\n", "\n",
            "```python", "\n",
            "print('hello')", "\n",
            "```", "\n",
        ]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        assert "# Title\n\n" in full, f"Heading newlines lost: {full!r}"
        assert "- bullet 1\n- bullet 2" in full, f"Bullets lost: {full!r}"
        assert "```python\nprint('hello')\n```" in full, f"Code block lost: {full!r}"

    def test_no_tag_emojis(self):
        """Emoji characters should pass through correctly."""
        tokens = ["Hello ", "🎉", " ", "🚀", " world!"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "🎉" in full
        assert "🚀" in full


class TestScenario4_NewlinePreservation:
    """Test that newline-only chunks survive the pipeline.

    Original bug: \n chunks were dropped by whitespace suppression.
    """

    def test_newline_between_paragraphs(self):
        """Double newline between paragraphs must be preserved."""
        tokens = ["<think>", "ok", "</think>", "Para 1.", "\n", "\n", "Para 2."]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "Para 1.\n\nPara 2." in full, f"Newlines lost: {full!r}"

    def test_newline_in_markdown_list(self):
        tokens = ["<think>", "ok", "</think>", "- a", "\n", "- b", "\n", "- c"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "- a\n- b\n- c" in full

    def test_newline_in_code_block(self):
        tokens = [
            "<think>", "ok", "</think>",
            "```", "\n", "line1", "\n", "line2", "\n", "```",
        ]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "```\nline1\nline2\n```" in full


class TestScenario5_EdgeCases:
    """Edge cases and mixed scenarios."""

    def test_empty_think_tags(self):
        """<think></think>content — empty reasoning."""
        tokens = ["<think>", "</think>", "Just content."]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert full == "Just content."

    def test_deepseek_no_tag_threshold(self):
        """DeepSeek-R1 should also handle no-tag output correctly."""
        text = "A regular response without thinking tags, should be content."
        tokens = [text[i:i+5] for i in range(0, len(text), 5)]
        chunks = simulate_server_streaming(tokens, use_reasoning_parser="deepseek_r1")
        full = "".join(chunks)
        assert full == text, f"DeepSeek no-tag mismatch: {full!r}"

    def test_single_char_no_tag(self):
        """Single character output, no tags."""
        chunks = simulate_server_streaming(["Y"], use_reasoning_parser="qwen3")
        full = "".join(chunks)
        assert full == "Y"

    def test_whitespace_only_not_emitted_as_content(self):
        """Pure empty string should be filtered, but whitespace should pass."""
        tokens = ["<think>", "ok", "</think>", "a", "\n", "\t", "b"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "a\n\tb" in full


class TestScenario6_NoParserThinkTagPassthrough:
    """Test the standard path (no reasoning parser).

    After removing the think buffer, <think>...</think> tags should pass
    through to the client so UIs like Open WebUI can render them natively.
    """

    def test_think_tags_pass_through(self):
        """<think> and </think> tags should appear in the output."""
        tokens = ["<think>", "reasoning here", "</think>", "\n\n", "Answer: 42"]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert "<think>" in full
        assert "</think>" in full
        assert "reasoning here" in full
        assert "Answer: 42" in full

    def test_emojis_stream_immediately(self):
        """Emojis should stream token-by-token, not be buffered."""
        tokens = ["😀", "😃", "😄", "😁", "😆"]
        chunks = simulate_server_streaming_no_parser(tokens)
        # Each emoji should be a separate chunk (immediate streaming)
        assert len(chunks) == 5, f"Expected 5 chunks, got {len(chunks)}: {chunks}"
        full = "".join(chunks)
        assert full == "😀😃😄😁😆"

    def test_100_emojis(self):
        """The exact test case from the Reddit report: 'Output 100 different emojis'."""
        emojis = list("😀😃😄😁😆😅🤣😂🙂🙃😉😊😇🥰😍🤩😘😗☺😚😙"
                       "🥲😋😛😜🤪😝🤑🤗🤭🤫🤔🫡🤐🤨😐😑😶🫥😏😒"
                       "🙄😬🤥🫨😌😔😪🤤😴😷🤒🤕🤢🤮🤧🥵🥶🥴😵🤯"
                       "🤠🥳🥸😎🤓🧐😕🫤😟🙁☹😮😯😲😳🥺🥹😦😧😨"
                       "😰😥😢😭😱😖😣😞😓😩😫🥱😤😡😠🤬😈👿💀☠💩")
        # Take first 100
        emoji_tokens = emojis[:100]
        chunks = simulate_server_streaming_no_parser(emoji_tokens)
        full = "".join(chunks)
        # All emojis should be present
        assert len(full) == len(emoji_tokens), (
            f"Expected {len(emoji_tokens)} chars, got {len(full)}"
        )
        for e in emoji_tokens:
            assert e in full, f"Missing emoji: {e}"

    def test_think_then_emojis(self):
        """Model thinks first, then outputs emojis — both should pass through."""
        tokens = [
            "<think>", "Let me list emojis", "</think>", "\n\n",
            "😀", " ", "😃", " ", "😄", " ", "😁",
        ]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert "<think>" in full
        assert "Let me list emojis" in full
        assert "</think>" in full
        assert "😀" in full
        assert "😁" in full

    def test_no_think_tags_plain_text(self):
        """Plain text without think tags streams normally."""
        tokens = ["Hello ", "world ", "this ", "is ", "a ", "test."]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert full == "Hello world this is a test."

    def test_special_tokens_still_filtered(self):
        """Special tokens like <|im_end|> should still be removed."""
        tokens = ["Hello", "<|im_end|>", " world"]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert full == "Hello world"
        assert "<|im_end|>" not in full

    def test_mixed_emojis_and_text_with_special_tokens(self):
        """Emojis mixed with text and special tokens."""
        tokens = ["Great ", "job! ", "🎉", "🎊", "<|im_end|>"]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert full == "Great job! 🎉🎊"

    def test_empty_think_block_passthrough(self):
        """<think></think> with empty reasoning should pass through."""
        tokens = ["<think>", "</think>", "Direct answer."]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert "<think>" in full
        assert "</think>" in full
        assert "Direct answer." in full


class TestScenario7_MultiByteBoundary:
    """Test multi-byte character boundary handling in tokenizer decode.

    When emojis span multiple tokens, tokenizer.decode() may produce
    U+FFFD (replacement character) for incomplete byte sequences.
    The guard should hold the delta until the character is complete.
    """

    def test_emoji_split_across_two_tokens(self):
        """Emoji completed on second token — should emit once, correctly."""
        # Simulate: tok1 decodes to "Hello \ufffd", tok2 completes to "Hello 😀"
        cumulative_decodes = [
            "Hello ",       # tok0: normal text
            "Hello \ufffd", # tok1: partial emoji (first bytes)
            "Hello 😀",     # tok2: emoji complete
        ]
        deltas = simulate_tokenizer_decode(cumulative_decodes)
        assert deltas[0] == "Hello "
        assert deltas[1] == ""  # skipped due to U+FFFD
        assert deltas[2] == "😀"  # complete emoji emitted
        full = "".join(deltas)
        assert full == "Hello 😀"
        assert "\ufffd" not in full

    def test_emoji_split_across_three_tokens(self):
        """Emoji needing 3 tokens to complete."""
        cumulative_decodes = [
            "Hi ",
            "Hi \ufffd",      # partial
            "Hi \ufffd",      # still partial (different bytes, same result)
            "Hi 🏳️‍🌈",     # complete (rainbow flag, multi-codepoint)
        ]
        deltas = simulate_tokenizer_decode(cumulative_decodes)
        assert deltas[1] == ""  # skipped
        assert deltas[2] == ""  # skipped
        full = "".join(deltas)
        assert "🏳️‍🌈" in full
        assert "\ufffd" not in full

    def test_no_replacement_chars_normal_flow(self):
        """Normal text without multi-byte issues — no deltas dropped."""
        cumulative_decodes = [
            "H",
            "He",
            "Hel",
            "Hell",
            "Hello",
        ]
        deltas = simulate_tokenizer_decode(cumulative_decodes)
        assert deltas == ["H", "e", "l", "l", "o"]

    def test_multiple_emojis_some_split(self):
        """Multiple emojis, some split across tokens, some single-token."""
        cumulative_decodes = [
            "😀",           # emoji as single token
            "😀\ufffd",     # next emoji partial
            "😀🎉",         # second emoji complete
            "😀🎉 done",    # trailing text
        ]
        deltas = simulate_tokenizer_decode(cumulative_decodes)
        assert deltas[0] == "😀"
        assert deltas[1] == ""       # skipped
        assert deltas[2] == "🎉"
        assert deltas[3] == " done"
        full = "".join(deltas)
        assert full == "😀🎉 done"

    def test_replacement_char_at_end_of_stream(self):
        """If stream ends with a replacement char, it's still suppressed."""
        cumulative_decodes = [
            "text",
            "text\ufffd",  # partial at end — suppressed
        ]
        deltas = simulate_tokenizer_decode(cumulative_decodes)
        full = "".join(deltas)
        assert full == "text"
        assert "\ufffd" not in full

    def test_chinese_characters_single_token(self):
        """CJK characters (multi-byte UTF-8) as single tokens work fine."""
        cumulative_decodes = [
            "你",
            "你好",
            "你好世",
            "你好世界",
        ]
        deltas = simulate_tokenizer_decode(cumulative_decodes)
        assert deltas == ["你", "好", "世", "界"]


class TestScenario8_RegressionNoParser:
    """Regression tests: the no-parser path should NOT break existing behavior.

    With reasoning parser enabled, behavior is unchanged (tested by Scenarios 1-5).
    Without a reasoning parser, the old think buffer stripped think tags.
    The NEW behavior: think tags pass through. These tests verify the
    transition is intentional and complete.
    """

    def test_reasoning_parser_still_strips_think_tags(self):
        """WITH a reasoning parser, think tags are still extracted (not in content)."""
        tokens = ["<think>", "reasoning", "</think>", "content"]
        chunks = simulate_server_streaming(tokens, use_reasoning_parser="qwen3")
        full = "".join(chunks)
        assert "content" in full
        assert "<think>" not in full
        assert "reasoning" not in full

    def test_no_parser_preserves_full_output(self):
        """WITHOUT a reasoning parser, nothing is stripped — full output preserved."""
        tokens = [
            "<think>", "I need to reason.", "</think>", "\n",
            "The answer is 42.",
        ]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        # Everything should be present
        assert "<think>" in full
        assert "I need to reason." in full
        assert "</think>" in full
        assert "The answer is 42." in full

    def test_no_parser_markdown_with_emojis(self):
        """Markdown + emojis stream correctly without parser."""
        tokens = [
            "# Emoji List ", "🎯", "\n\n",
            "1. ", "😀", " - Grinning", "\n",
            "2. ", "🎉", " - Party", "\n",
            "3. ", "🚀", " - Rocket", "\n",
        ]
        chunks = simulate_server_streaming_no_parser(tokens)
        full = "".join(chunks)
        assert "# Emoji List 🎯" in full
        assert "😀 - Grinning" in full
        assert "🎉 - Party" in full
        assert "🚀 - Rocket" in full

    def test_no_parser_long_emoji_sequence(self):
        """Long emoji-only output streams immediately, not buffered."""
        # 50 emojis — old code would buffer all, new code streams immediately
        emojis = list("🌍🌎🌏🌐🗺🧭🏔⛰🌋🗻🏕🏖🏜🏝🏞🏟🏛🏗🧱🪨"
                       "🪵🛖🏘🏚🏠🏡🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰💒")
        chunks = simulate_server_streaming_no_parser(emojis)
        # Each emoji should be a separate chunk (not batched)
        assert len(chunks) == len(emojis), (
            f"Expected {len(emojis)} chunks (immediate streaming), got {len(chunks)}"
        )
