"""Tests for StreamingThinkRouter - routes <think> blocks to Anthropic thinking content blocks."""

import unittest

from rapid_mlx.api.utils import StreamingThinkRouter


class TestStreamingThinkRouter(unittest.TestCase):
    """Unit tests for StreamingThinkRouter."""

    # --- Basic routing ---

    def test_plain_text_routes_as_text(self):
        r = StreamingThinkRouter()
        assert r.process("Hello world") == [("text", "Hello world")]

    def test_think_block_routes_as_thinking(self):
        r = StreamingThinkRouter()
        assert r.process("<think>reasoning</think>") == [("thinking", "reasoning")]

    def test_text_then_think_then_text(self):
        r = StreamingThinkRouter()
        result = r.process("before<think>middle</think>after")
        assert result == [("text", "before"), ("thinking", "middle"), ("text", "after")]

    # --- start_in_thinking mode ---

    def test_start_in_thinking_mode(self):
        """When model injects <think> into prompt, output starts in thinking mode."""
        r = StreamingThinkRouter(start_in_thinking=True)
        result = r.process("reasoning here")
        assert result == [("thinking", "reasoning here")]

    def test_start_in_thinking_then_close(self):
        """Thinking closes with </think>, then text follows."""
        r = StreamingThinkRouter(start_in_thinking=True)
        result = r.process("reasoning</think>answer")
        assert result == [("thinking", "reasoning"), ("text", "answer")]

    def test_start_in_thinking_close_across_deltas(self):
        """</think> split across multiple deltas."""
        r = StreamingThinkRouter(start_in_thinking=True)
        p1 = r.process("thinking stuff</th")
        p2 = r.process("ink>now text")
        # First delta should hold back partial </think>
        assert ("thinking", "thinking stuff") in p1
        # Second delta should transition
        all_pieces = p1 + p2
        types = [t for t, _ in all_pieces]
        assert "text" in types

    # --- Partial tag handling ---

    def test_partial_open_tag_held_back(self):
        """Partial <think at end of delta should be held back."""
        r = StreamingThinkRouter()
        p1 = r.process("Hello <thi")
        p2 = r.process("nk>reasoning</think>")
        # p1 should emit "Hello " but hold back "<thi"
        assert p1 == [("text", "Hello ")]
        # p2 completes the tag
        assert ("thinking", "reasoning") in p2

    def test_partial_close_tag_in_thinking(self):
        """Partial </think at end of delta while in thinking mode."""
        r = StreamingThinkRouter(start_in_thinking=True)
        p1 = r.process("deep thought</thi")
        p2 = r.process("nk>answer")
        # p1 should emit thinking content but hold back partial
        assert ("thinking", "deep thought") in p1
        # p2 should transition to text
        assert ("text", "answer") in p2

    def test_partial_tag_false_alarm(self):
        """Partial match that turns out not to be a tag."""
        r = StreamingThinkRouter()
        p1 = r.process("Hello <this")
        p2 = r.process(" is not a tag>")
        # After p2, the held-back "<thi" + "s is not a tag>" should emit as text
        all_text = "".join(t for bt, t in p1 + p2 if bt == "text")
        assert all_text == "Hello <this is not a tag>"

    # --- Multiple think blocks ---

    def test_multiple_think_blocks(self):
        r = StreamingThinkRouter()
        result = r.process("<think>first</think>middle<think>second</think>end")
        assert result == [
            ("thinking", "first"),
            ("text", "middle"),
            ("thinking", "second"),
            ("text", "end"),
        ]

    # --- Streaming across deltas ---

    def test_streaming_token_by_token(self):
        """Simulate character-by-character streaming."""
        r = StreamingThinkRouter()
        text = "<think>abc</think>xyz"
        all_pieces = []
        for ch in text:
            all_pieces.extend(r.process(ch))
        all_pieces.extend(r.flush())
        thinking = "".join(t for bt, t in all_pieces if bt == "thinking")
        text_out = "".join(t for bt, t in all_pieces if bt == "text")
        assert thinking == "abc"
        assert text_out == "xyz"

    def test_streaming_with_start_in_thinking(self):
        """Token-by-token with start_in_thinking."""
        r = StreamingThinkRouter(start_in_thinking=True)
        text = "reasoning</think>the answer"
        all_pieces = []
        for ch in text:
            all_pieces.extend(r.process(ch))
        all_pieces.extend(r.flush())
        thinking = "".join(t for bt, t in all_pieces if bt == "thinking")
        text_out = "".join(t for bt, t in all_pieces if bt == "text")
        assert thinking == "reasoning"
        assert text_out == "the answer"

    # --- Flush behavior ---

    def test_flush_emits_remaining_text(self):
        """Text without partial tags is emitted by process(), flush() is empty."""
        r = StreamingThinkRouter()
        pieces = r.process("partial text")
        assert pieces == [("text", "partial text")]
        assert r.flush() == []

    def test_flush_emits_remaining_thinking(self):
        """Thinking without partial tags is emitted by process(), flush() is empty."""
        r = StreamingThinkRouter(start_in_thinking=True)
        pieces = r.process("unfinished thought")
        assert pieces == [("thinking", "unfinished thought")]
        assert r.flush() == []

    def test_flush_with_held_back_partial(self):
        """Flush should emit held-back partial tag as content."""
        r = StreamingThinkRouter()
        r.process("text<thi")
        pieces = r.flush()
        all_text = "".join(t for _, t in pieces)
        assert "<thi" in all_text

    def test_flush_resets_state(self):
        """After flush, router should be back to default state."""
        r = StreamingThinkRouter(start_in_thinking=True)
        r.process("thinking")
        r.flush()
        # After flush, _in_think should be False
        result = r.process("new text")
        assert result == [("text", "new text")]

    # --- Empty inputs ---

    def test_empty_delta(self):
        r = StreamingThinkRouter()
        assert r.process("") == []

    def test_flush_empty(self):
        r = StreamingThinkRouter()
        assert r.flush() == []


if __name__ == "__main__":
    unittest.main()
