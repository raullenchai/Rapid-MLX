"""Tests for StreamingToolCallFilter - suppresses tool call XML during streaming."""

import unittest

from vllm_mlx.api.utils import StreamingToolCallFilter


class TestStreamingToolCallFilter(unittest.TestCase):
    def test_normal_text_passes_through(self):
        f = StreamingToolCallFilter()
        assert f.process("Hello world") == "Hello world"

    def test_minimax_tool_call_suppressed(self):
        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>")
        f.process('<invoke name="read">')
        f.process('<parameter name="path">/tmp/test.txt</parameter>')
        f.process("</invoke>")
        result = f.process("</minimax:tool_call>")
        assert result == ""

    def test_text_after_tool_call_emits(self):
        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>content</minimax:tool_call>")
        assert f.process("After") == "After"

    def test_text_before_and_after_same_delta(self):
        f = StreamingToolCallFilter()
        result = f.process("Before <minimax:tool_call>inside</minimax:tool_call>After")
        assert result == "Before After"

    def test_split_across_deltas(self):
        f = StreamingToolCallFilter()
        r1 = f.process("Before <minim")
        r2 = f.process("ax:tool_call>inside</minimax:tool_call>After")
        assert r1 + r2 == "Before After"

    def test_qwen_format_suppressed(self):
        f = StreamingToolCallFilter()
        result = f.process('Text <tool_call>{"name":"fn"}</tool_call> more')
        assert result == "Text  more"

    def test_multiple_tool_calls(self):
        f = StreamingToolCallFilter()
        result = f.process(
            "A <minimax:tool_call>x</minimax:tool_call>"
            " B <minimax:tool_call>y</minimax:tool_call> C"
        )
        assert result == "A  B  C"

    def test_flush_partial_tag_emits(self):
        f = StreamingToolCallFilter()
        r = f.process("text <minim")
        fl = f.flush()
        assert r + fl == "text <minim"

    def test_flush_unterminated_block_discards(self):
        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>partial content")
        assert f.flush() == ""

    def test_large_tool_call_content(self):
        """Simulates a Read tool returning a large file."""
        f = StreamingToolCallFilter()
        big = "x" * 10000
        result = f.process(f"Before <minimax:tool_call>{big}</minimax:tool_call>After")
        assert result == "Before After"

    def test_think_tags_not_filtered(self):
        f = StreamingToolCallFilter()
        result = f.process("<think>reasoning here</think>answer")
        assert "<think>" in result
        assert "reasoning here" in result

    def test_mixed_think_and_tool_call(self):
        f = StreamingToolCallFilter()
        result = f.process(
            "<think>thinking</think>"
            "<minimax:tool_call>tool stuff</minimax:tool_call>"
            "final answer"
        )
        assert "<think>thinking</think>" in result
        assert "tool stuff" not in result
        assert "final answer" in result

    def test_gradual_token_by_token(self):
        """Simulate token-by-token streaming."""
        f = StreamingToolCallFilter()
        parts = [
            "Hello ",
            "<",
            "mini",
            "max:",
            "tool_call",
            ">",
            '<invoke name="test">',
            "</invoke>",
            "</minimax:tool_call>",
            " world",
        ]
        result = ""
        for part in parts:
            result += f.process(part)
        result += f.flush()
        assert result == "Hello  world", f"Got: {result!r}"

    def test_empty_deltas(self):
        f = StreamingToolCallFilter()
        assert f.process("") == ""
        assert f.process("text") == "text"
        assert f.process("") == ""

    def test_calling_tool_bracket_suppressed(self):
        """Qwen3 bracket-style: [Calling tool: func({...})]\n"""
        f = StreamingToolCallFilter()
        result = f.process('[Calling tool: search({"q": "test"})]\n')
        assert result == ""

    def test_calling_tool_multiline_json(self):
        """Multi-line JSON args in bracket-style tool call."""
        f = StreamingToolCallFilter()
        r1 = f.process('[Calling tool: search({"q": "test",')
        r2 = f.process(' "limit": 5})]\n')
        r3 = f.process("After")
        assert r1 + r2 + r3 == "After"

    def test_buffer_cap_on_unclosed_block(self):
        """Buffer should be capped if tool call block never closes."""
        from vllm_mlx.api.utils import _MAX_TOOL_BUFFER_BYTES

        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>")
        # Feed data exceeding the cap
        chunk = "x" * 10000
        for _ in range(_MAX_TOOL_BUFFER_BYTES // 10000 + 2):
            f.process(chunk)
        # After cap, filter should have exited the block
        assert not f._in_block
        # New text should pass through
        assert f.process("after") == "after"


class TestStreamingToolCallFilterGemma4(unittest.TestCase):
    """Gemma 4 native wire-format suppression regression tests (issue #686).

    The opener ``<|tool_call>`` and closer ``<tool_call|>`` are asymmetric:
    the opener has no closing ``|>`` and the closer has no leading ``<|``.
    These tests verify that StreamingToolCallFilter correctly suppresses the
    full envelope (including the inner ``<|"|>`` string-quoting markers),
    handles chunk-split deltas, and bounds unterminated envelopes.
    """

    # Smoking-gun snippet captured from issue #686 (gemma-4-12b-4bit + Codex
    # CLI successful trial). The raw `response.output_text.delta` payload
    # leaked the full wire envelope into user-visible text.
    GEMMA4_LEAK_SNIPPET = (
        '<|tool_call>call:exec_command{cmd:<|"|>head -n 1 pyproject.toml<|"|>}'
        "<tool_call|>"
    )

    def test_gemma4_envelope_suppressed_whole_delta(self):
        """Full envelope arrives in a single delta — emit nothing."""
        f = StreamingToolCallFilter()
        result = f.process(self.GEMMA4_LEAK_SNIPPET)
        assert result == ""
        # And no leftover markup hiding in the flush
        assert f.flush() == ""

    def test_gemma4_envelope_with_text_before_and_after(self):
        """Visible text surrounding the envelope is preserved."""
        f = StreamingToolCallFilter()
        result = f.process(f"Sure! {self.GEMMA4_LEAK_SNIPPET}Done.")
        assert result == "Sure! Done."

    def test_gemma4_envelope_inner_quote_markers_not_leaked(self):
        """The inner ``<|"|>`` markers must not survive into visible text
        when wrapped in the envelope."""
        f = StreamingToolCallFilter()
        result = f.process(self.GEMMA4_LEAK_SNIPPET)
        assert '<|"|>' not in result
        assert "<|tool_call>" not in result
        assert "<tool_call|>" not in result

    def test_gemma4_opener_split_across_three_deltas(self):
        """Opener ``<|tool_call>`` arrives split across multiple streaming
        chunks (``<|t`` / ``ool_`` / ``call>``). Prefix-matching in
        ``_scan_for_open`` must hold back the partial match."""
        f = StreamingToolCallFilter()
        parts = [
            "Before ",
            "<|t",
            "ool_",
            "call>",
            "call:exec_command{cmd:",
            '<|"|>ls<|"|>}',
            "<tool_call|>",
            " After",
        ]
        out = "".join(f.process(p) for p in parts)
        out += f.flush()
        assert out == "Before  After", f"Got: {out!r}"

    def test_gemma4_closer_split_across_deltas(self):
        """Asymmetric closer ``<tool_call|>`` arrives split. The opener and
        closer share NO common stem, so this exercises the close-tag path
        in ``_consume_block`` independently from the open path."""
        f = StreamingToolCallFilter()
        parts = [
            "<|tool_call>call:f{a:1}",
            "<tool",
            "_call",
            "|>",
            "tail",
        ]
        out = "".join(f.process(p) for p in parts)
        out += f.flush()
        assert out == "tail", f"Got: {out!r}"

    def test_gemma4_token_by_token(self):
        """Maximum-granularity streaming: every char its own delta."""
        f = StreamingToolCallFilter()
        full = f"head {self.GEMMA4_LEAK_SNIPPET} tail"
        result = ""
        for ch in full:
            result += f.process(ch)
        result += f.flush()
        assert result == "head  tail", f"Got: {result!r}"

    def test_gemma4_unterminated_envelope_discarded(self):
        """Model emits opener but never the closer — flush must NOT leak
        the partial body, and the buffer cap must keep memory bounded."""
        f = StreamingToolCallFilter()
        f.process('<|tool_call>call:exec_command{cmd:<|"|>rm -rf')
        # Flush mid-stream: unterminated block is discarded entirely.
        assert f.flush() == ""

    def test_gemma4_buffer_cap_on_unclosed_envelope(self):
        """The 1 MB cap protects against pathologically unclosed envelopes
        even for the asymmetric gemma4 markers."""
        from vllm_mlx.api.utils import _MAX_TOOL_BUFFER_BYTES

        f = StreamingToolCallFilter()
        f.process("<|tool_call>")
        chunk = "x" * 10000
        for _ in range(_MAX_TOOL_BUFFER_BYTES // 10000 + 2):
            f.process(chunk)
        # After cap, filter has exited the block and recovers
        assert not f._in_block
        assert f.process("after") == "after"

    def test_gemma4_does_not_swallow_unrelated_text(self):
        """A bare ``<|`` in unrelated text must not accidentally swallow
        downstream content (verifies the close-tag detection in
        ``_consume_block`` doesn't grab the asymmetric closer prematurely)."""
        f = StreamingToolCallFilter()
        # No tool_call markers at all — should pass through unchanged.
        result = f.process("Hello <|im_end|> world")
        result += f.flush()
        assert result == "Hello <|im_end|> world"

    def test_gemma4_multiple_envelopes_in_one_stream(self):
        """Two consecutive gemma4 tool calls in the same stream."""
        f = StreamingToolCallFilter()
        result = f.process(
            "A "
            '<|tool_call>call:f{x:<|"|>1<|"|>}<tool_call|>'
            " B "
            '<|tool_call>call:g{y:<|"|>2<|"|>}<tool_call|>'
            " C"
        )
        assert result == "A  B  C"


class TestToolCallTagsRegistry(unittest.TestCase):
    """Verify the gemma4 markers are wired up in the global tags list."""

    def test_gemma4_pair_registered(self):
        from vllm_mlx.api.utils import get_tool_call_tags

        tags = get_tool_call_tags()
        assert ("<|tool_call>", "<tool_call|>") in tags, (
            "gemma4 tool-call wire markers must be registered in "
            "_TOOL_CALL_TAGS so StreamingToolCallFilter suppresses them "
            "by default (issue #686 regression)."
        )


if __name__ == "__main__":
    unittest.main()
