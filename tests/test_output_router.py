# SPDX-License-Identifier: Apache-2.0
"""
Tests for the token-level OutputRouter.

Uses real Gemma 4 token IDs (from tokenizer vocabulary) to verify
routing correctness without any text-level matching.
"""

import pytest

from vllm_mlx.output_router import Channel, OutputRouter, RouterState, TokenMap


# === Gemma 4 Token IDs (from tokenizer) ===
GEMMA4_MAP = TokenMap(
    channel_start=100,   # <|channel>
    channel_end=101,     # <channel|>
    thought_word=45518,  # "thought"
    content_word=3955,   # "content"
    final_word=10218,    # "final"
    turn_start=105,      # <|turn>
    turn_end=106,        # <turn|>
    tool_call_start=48,  # <|tool_call>
    tool_call_end=49,    # <tool_call|>
    tool_quote=52,       # <|"|>
    tool_start=46,       # <|tool>
    tool_end=47,         # <tool|>
    tool_response_start=50,  # <|tool_response>
    tool_response_end=51,    # <tool_response|>
    bos=2,
    eos=1,
    pad=0,
)


class FakeTokenizer:
    """Minimal tokenizer that maps token IDs to text."""

    def __init__(self, vocab: dict[str, int]):
        self._id_to_text = {v: k for k, v in vocab.items()}
        self._vocab = vocab

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_text.get(i, f"<UNK:{i}>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


# Gemma 4 vocabulary (subset for testing)
VOCAB = {
    "<pad>": 0, "<eos>": 1, "<bos>": 2,
    "<|tool>": 46, "<tool|>": 47,
    "<|tool_call>": 48, "<tool_call|>": 49,
    "<|tool_response>": 50, "<tool_response|>": 51,
    '<|"|>': 52,
    "<|channel>": 100, "<channel|>": 101,
    "<|turn>": 105, "<turn|>": 106,
    "\n": 107,
    "thought": 45518, "content": 3955, "final": 10218,
    "Hello": 9259, "Four": 73440, "call": 6639, ":": 236787,
    "get": 828, "_": 236779, "weather": 19323,
    "{": 236782, "}": 236783, "city": 13319,
    "Tokyo": 89265, " ": 235248,
    "The": 651, "user": 2364, "wants": 10388,
}

TOKENIZER = FakeTokenizer(VOCAB)


@pytest.fixture
def router():
    r = OutputRouter(GEMMA4_MAP, TOKENIZER)
    r.reset()
    return r


class TestBasicRouting:
    """Test fundamental token routing."""

    def test_content_passthrough(self, router):
        """Plain content tokens go to CONTENT channel."""
        event = router.feed(9259)  # "Hello"
        assert event is not None
        assert event.channel == Channel.CONTENT
        assert event.text == "Hello"

    def test_bos_eos_pad_suppressed(self, router):
        """Control tokens are suppressed."""
        assert router.feed(0) is None   # pad
        assert router.feed(1) is None   # eos
        assert router.feed(2) is None   # bos

    def test_turn_tokens_suppressed(self, router):
        """Turn markers are suppressed."""
        assert router.feed(105) is None  # <|turn>
        assert router.feed(106) is None  # <turn|>


class TestThinkingChannel:
    """Test thought channel routing."""

    def test_thought_channel_detected(self, router):
        """<|channel> + thought → THINKING state, tokens go to REASONING."""
        assert router.feed(100) is None   # <|channel> suppressed
        assert router.feed(45518) is None  # "thought" suppressed
        assert router.state == RouterState.THINKING

        event = router.feed(651)  # "The"
        assert event is not None
        assert event.channel == Channel.REASONING

    def test_thought_ends_at_channel_close(self, router):
        """<channel|> ends thinking, switches to CONTENT."""
        router.feed(100)    # <|channel>
        router.feed(45518)  # thought
        router.feed(651)    # "The" (reasoning)

        assert router.feed(101) is None  # <channel|> suppressed
        assert router.state == RouterState.CONTENT

        event = router.feed(73440)  # "Four"
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_thought_then_content_channel(self, router):
        """Full cycle: thought → <channel|> → content channel → answer."""
        # Thought channel
        router.feed(100)     # <|channel>
        router.feed(45518)   # thought
        e1 = router.feed(651)  # "The"
        assert e1.channel == Channel.REASONING

        # End thought
        router.feed(101)  # <channel|>

        # Content channel
        router.feed(100)   # <|channel>
        router.feed(3955)  # content
        e2 = router.feed(73440)  # "Four"
        assert e2.channel == Channel.CONTENT

    def test_implicit_content_after_thought(self, router):
        """After <channel|>, if no new <|channel>, tokens are content."""
        router.feed(100)     # <|channel>
        router.feed(45518)   # thought
        router.feed(651)     # reasoning
        router.feed(101)     # <channel|>

        # No explicit content channel — should still be content
        e = router.feed(73440)  # "Four"
        assert e.channel == Channel.CONTENT


class TestToolCallRouting:
    """Test tool call accumulation."""

    def test_tool_call_accumulated(self, router):
        """Tokens between <|tool_call> and <tool_call|> are accumulated."""
        assert router.feed(48) is None   # <|tool_call> → accumulate
        assert router.feed(6639) is None  # "call" → accumulate
        assert router.feed(236787) is None  # ":" → accumulate

    def test_tool_call_emitted_on_close(self, router):
        """Complete tool call emitted as TOOL_CALL event."""
        router.feed(48)      # <|tool_call>
        router.feed(6639)    # call
        router.feed(236787)  # :
        router.feed(828)     # get
        router.feed(236779)  # _
        router.feed(19323)   # weather
        router.feed(236782)  # {
        router.feed(13319)   # city
        router.feed(236787)  # :
        router.feed(52)      # <|"|>
        router.feed(89265)   # Tokyo
        router.feed(52)      # <|"|>
        router.feed(236783)  # }

        event = router.feed(49)  # <tool_call|>
        assert event is not None
        assert event.channel == Channel.TOOL_CALL
        assert "get_weather" in event.text
        assert "Tokyo" in event.text

    def test_content_after_tool_call(self, router):
        """After tool call completes, back to content mode."""
        router.feed(48)   # <|tool_call>
        router.feed(6639) # call
        router.feed(49)   # <tool_call|>

        event = router.feed(9259)  # "Hello"
        assert event.channel == Channel.CONTENT


class TestOrphanTokens:
    """Test handling of orphaned/leaked special tokens."""

    def test_orphan_tool_call_end_suppressed(self, router):
        """<tool_call|> without <|tool_call> should be suppressed."""
        assert router.feed(49) is None  # <tool_call|> orphan

    def test_orphan_tool_response_suppressed(self, router):
        """Tool response markers should always be suppressed."""
        assert router.feed(50) is None  # <|tool_response>
        assert router.feed(51) is None  # <tool_response|>

    def test_orphan_tool_markers_suppressed(self, router):
        """<|tool> and <tool|> should be suppressed."""
        assert router.feed(46) is None  # <|tool>
        assert router.feed(47) is None  # <tool|>

    def test_content_after_orphan_tokens(self, router):
        """Content after orphan tokens routes correctly."""
        router.feed(49)   # orphan <tool_call|>
        router.feed(51)   # orphan <tool_response|>
        event = router.feed(9259)  # "Hello"
        assert event is not None
        assert event.channel == Channel.CONTENT


class TestFeedSequence:
    """Test batch processing of token sequences."""

    def test_thought_then_content(self, router):
        """Process a full thought→content sequence."""
        tokens = [
            100, 45518, 107,  # <|channel> thought \n
            651, 2364,        # "The" "user" (reasoning)
            101,              # <channel|>
            100, 3955, 107,   # <|channel> content \n
            73440,            # "Four" (content)
            101,              # <channel|>
        ]
        result = router.feed_sequence(tokens)
        assert result["content"] == "Four"
        assert "The" in result["reasoning"]
        assert result["tool_calls"] is None

    def test_tool_call_sequence(self, router):
        """Process a tool call sequence."""
        tokens = [
            48,               # <|tool_call>
            6639, 236787,     # call :
            828, 236779, 19323,  # get _ weather
            236782,           # {
            13319, 236787,    # city :
            52, 89265, 52,    # <|"|> Tokyo <|"|>
            236783,           # }
            49,               # <tool_call|>
        ]
        result = router.feed_sequence(tokens)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert "Tokyo" in result["tool_calls"][0]

    def test_plain_content(self, router):
        """No special tokens → all content."""
        tokens = [9259, 235248, 73440]  # Hello Four
        result = router.feed_sequence(tokens)
        assert result["content"] is not None
        assert result["reasoning"] is None
        assert result["tool_calls"] is None


class TestFromTokenizer:
    """Test auto-detection from tokenizer."""

    def test_gemma4_detected(self):
        """Gemma 4 tokenizer auto-detected."""
        router = OutputRouter.from_tokenizer(TOKENIZER)
        assert router is not None
        assert router.map.channel_start == 100

    def test_unknown_tokenizer(self):
        """Non-Gemma tokenizer returns None."""
        plain_vocab = {"hello": 1, "world": 2}
        plain_tok = FakeTokenizer(plain_vocab)
        router = OutputRouter.from_tokenizer(plain_tok)
        assert router is None


class TestStateReset:
    """Test state management."""

    def test_reset_clears_state(self, router):
        """Reset returns to INIT state."""
        router.feed(100)     # enter channel
        router.feed(45518)   # thinking
        assert router.state == RouterState.THINKING

        router.reset()
        assert router.state == RouterState.INIT
        assert router._tool_tokens == []

    def test_multiple_requests(self, router):
        """Router works correctly across multiple reset cycles."""
        # Request 1
        router.feed(100); router.feed(45518)  # thinking
        e1 = router.feed(651)
        assert e1.channel == Channel.REASONING

        # Request 2
        router.reset()
        e2 = router.feed(9259)  # "Hello"
        assert e2.channel == Channel.CONTENT
