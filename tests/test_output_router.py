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
    channel_start=100,  # <|channel>
    channel_end=101,  # <channel|>
    thought_word=45518,  # "thought"
    content_word=3955,  # "content"
    final_word=10218,  # "final"
    turn_start=105,  # <|turn>
    turn_end=106,  # <turn|>
    tool_call_start=48,  # <|tool_call>
    tool_call_end=49,  # <tool_call|>
    tool_quote=52,  # <|"|>
    tool_start=46,  # <|tool>
    tool_end=47,  # <tool|>
    tool_response_start=50,  # <|tool_response>
    tool_response_end=51,  # <tool_response|>
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
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<|tool>": 46,
    "<tool|>": 47,
    "<|tool_call>": 48,
    "<tool_call|>": 49,
    "<|tool_response>": 50,
    "<tool_response|>": 51,
    '<|"|>': 52,
    "<|channel>": 100,
    "<channel|>": 101,
    "<|turn>": 105,
    "<turn|>": 106,
    "\n": 107,
    "thought": 45518,
    "content": 3955,
    "final": 10218,
    "Hello": 9259,
    "Four": 73440,
    "call": 6639,
    ":": 236787,
    "get": 828,
    "_": 236779,
    "weather": 19323,
    "{": 236782,
    "}": 236783,
    "city": 13319,
    "Tokyo": 89265,
    " ": 235248,
    "The": 651,
    "user": 2364,
    "wants": 10388,
}

TOKENIZER = FakeTokenizer(VOCAB)

# Qwen3 vocabulary subset from mlx-community/Qwen3.5-4B-MLX-4bit.
QWEN3_VOCAB = {
    "<think>": 248068,
    "</think>": 248069,
    "Reason": 1,
    "ing": 2,
    "Answer": 3,
    "Plain": 4,
}

QWEN3_TOKENIZER = FakeTokenizer(QWEN3_VOCAB)

# DeepSeek R1 vocabulary subset from deepseek-ai/DeepSeek-R1-Distill-Qwen-7B.
DEEPSEEK_R1_VOCAB = {
    "<｜end▁of▁sentence｜>": 151643,
    "<｜begin▁of▁sentence｜>": 151646,
    "<think>": 151648,
    "</think>": 151649,
    "Step": 1,
    " one": 2,
    "Answer": 3,
    "Plain": 4,
}

DEEPSEEK_R1_TOKENIZER = FakeTokenizer(DEEPSEEK_R1_VOCAB)

# GPT-OSS/Harmony vocabulary subset from openai/gpt-oss-20b.
HARMONY_VOCAB = {
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|call|>": 200012,
    "analysis": 35644,
    "final": 17196,
    " json": 1,
    "Reason": 2,
    "ing": 3,
    "Answer": 4,
    "Plain": 5,
    # Role-name token following ``<|start|>``. Real harmony tokenizes
    # ``assistant`` as a single token; if it weren't suppressed it would
    # decode as literal ``assistant`` text and leak into delta.content.
    "assistant": 173781,
    "user": 173782,
    " to=functions.get_weather": 173783,
}

HARMONY_TOKENIZER = FakeTokenizer(HARMONY_VOCAB)


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
        assert router.feed(0) is None  # pad
        assert router.feed(1) is None  # eos
        assert router.feed(2) is None  # bos

    def test_turn_tokens_suppressed(self, router):
        """Turn markers are suppressed."""
        assert router.feed(105) is None  # <|turn>
        assert router.feed(106) is None  # <turn|>


class TestThinkingChannel:
    """Test thought channel routing."""

    def test_thought_channel_detected(self, router):
        """<|channel> + thought → THINKING state, tokens go to REASONING."""
        assert router.feed(100) is None  # <|channel> suppressed
        assert router.feed(45518) is None  # "thought" suppressed
        assert router.state == RouterState.THINKING

        event = router.feed(651)  # "The"
        assert event is not None
        assert event.channel == Channel.REASONING

    def test_thought_ends_at_channel_close(self, router):
        """<channel|> ends thinking, switches to CONTENT."""
        router.feed(100)  # <|channel>
        router.feed(45518)  # thought
        router.feed(651)  # "The" (reasoning)

        assert router.feed(101) is None  # <channel|> suppressed
        assert router.state == RouterState.CONTENT

        event = router.feed(73440)  # "Four"
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_thought_then_content_channel(self, router):
        """Full cycle: thought → <channel|> → content channel → answer."""
        # Thought channel
        router.feed(100)  # <|channel>
        router.feed(45518)  # thought
        e1 = router.feed(651)  # "The"
        assert e1.channel == Channel.REASONING

        # End thought
        router.feed(101)  # <channel|>

        # Content channel
        router.feed(100)  # <|channel>
        router.feed(3955)  # content
        e2 = router.feed(73440)  # "Four"
        assert e2.channel == Channel.CONTENT

    def test_implicit_content_after_thought(self, router):
        """After <channel|>, if no new <|channel>, tokens are content."""
        router.feed(100)  # <|channel>
        router.feed(45518)  # thought
        router.feed(651)  # reasoning
        router.feed(101)  # <channel|>

        # No explicit content channel — should still be content
        e = router.feed(73440)  # "Four"
        assert e.channel == Channel.CONTENT


class TestToolCallRouting:
    """Test tool call accumulation."""

    def test_tool_call_accumulated(self, router):
        """Tokens between <|tool_call> and <tool_call|> are accumulated."""
        assert router.feed(48) is None  # <|tool_call> → accumulate
        assert router.feed(6639) is None  # "call" → accumulate
        assert router.feed(236787) is None  # ":" → accumulate

    def test_tool_call_emitted_on_close(self, router):
        """Complete tool call emitted as TOOL_CALL event."""
        router.feed(48)  # <|tool_call>
        router.feed(6639)  # call
        router.feed(236787)  # :
        router.feed(828)  # get
        router.feed(236779)  # _
        router.feed(19323)  # weather
        router.feed(236782)  # {
        router.feed(13319)  # city
        router.feed(236787)  # :
        router.feed(52)  # <|"|>
        router.feed(89265)  # Tokyo
        router.feed(52)  # <|"|>
        router.feed(236783)  # }

        event = router.feed(49)  # <tool_call|>
        assert event is not None
        assert event.channel == Channel.TOOL_CALL
        assert "get_weather" in event.text
        assert "Tokyo" in event.text

    def test_content_after_tool_call(self, router):
        """After tool call completes, back to content mode."""
        router.feed(48)  # <|tool_call>
        router.feed(6639)  # call
        router.feed(49)  # <tool_call|>

        event = router.feed(9259)  # "Hello"
        assert event.channel == Channel.CONTENT


class TestFinalize:
    """Test end-of-stream state draining."""

    def test_finalize_emits_incomplete_tool_call(self, router):
        """Unclosed TOOL_CALL state is drained as a TOOL_CALL event."""
        router.feed(48)  # <|tool_call>
        router.feed(6639)  # call
        router.feed(236787)  # :
        router.feed(828)  # get
        router.feed(236779)  # _
        router.feed(19323)  # weather
        router.feed(236782)  # {
        router.feed(13319)  # city
        router.feed(236787)  # :
        router.feed(52)  # <|"|>
        router.feed(89265)  # Tokyo
        router.feed(52)  # <|"|>
        router.feed(236783)  # }

        event = router.finalize()
        assert event is not None
        assert event.channel == Channel.TOOL_CALL
        assert "get_weather" in event.text
        assert "Tokyo" in event.text
        assert router.state == RouterState.CONTENT
        assert router._tool_tokens == []
        assert router.finalize() is None

    def test_finalize_drains_pending_harmony_analysis_message(self):
        """Harmony analysis content is preserved if <|message|> is missing."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        router.feed(200005)  # <|channel|>
        router.feed(35644)  # analysis
        router.feed(2)  # Reason
        router.feed(3)  # ing

        event = router.finalize()
        assert event is not None
        assert event.channel == Channel.REASONING
        assert event.text == "Reasoning"
        assert router.state == RouterState.CONTENT


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
        router.feed(49)  # orphan <tool_call|>
        router.feed(51)  # orphan <tool_response|>
        event = router.feed(9259)  # "Hello"
        assert event is not None
        assert event.channel == Channel.CONTENT


class TestFeedSequence:
    """Test batch processing of token sequences."""

    def test_thought_then_content(self, router):
        """Process a full thought→content sequence."""
        tokens = [
            100,
            45518,
            107,  # <|channel> thought \n
            651,
            2364,  # "The" "user" (reasoning)
            101,  # <channel|>
            100,
            3955,
            107,  # <|channel> content \n
            73440,  # "Four" (content)
            101,  # <channel|>
        ]
        result = router.feed_sequence(tokens)
        assert result["content"] == "Four"
        assert "The" in result["reasoning"]
        assert result["tool_calls"] is None

    def test_tool_call_sequence(self, router):
        """Process a tool call sequence."""
        tokens = [
            48,  # <|tool_call>
            6639,
            236787,  # call :
            828,
            236779,
            19323,  # get _ weather
            236782,  # {
            13319,
            236787,  # city :
            52,
            89265,
            52,  # <|"|> Tokyo <|"|>
            236783,  # }
            49,  # <tool_call|>
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

    def test_qwen3_think_tags_detected(self):
        """Qwen3 tokenizer think tags are auto-detected."""
        router = OutputRouter.from_tokenizer(QWEN3_TOKENIZER)
        assert router is not None
        assert router.map.think_start == 248068
        assert router.map.think_end == 248069

    def test_deepseek_r1_think_tags_detected(self):
        """DeepSeek R1 tokenizer think tags and unicode controls are detected."""
        router = OutputRouter.from_tokenizer(DEEPSEEK_R1_TOKENIZER)
        assert router is not None
        assert router.map.think_start == 151648
        assert router.map.think_end == 151649
        assert router.map.bos == 151646
        assert router.map.eos == 151643

    def test_harmony_detected(self):
        """GPT-OSS/Harmony tokenizer auto-detected."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None
        assert router.map.harmony_channel == 200005
        assert router.map.harmony_message == 200008
        assert router.map.harmony_return == 200002
        assert router.map.harmony_call == 200012


class TestQwen3ThinkRouting:
    """Test Qwen3 <think> routing."""

    def test_think_block_routes_to_reasoning_then_content(self):
        """<think> payload routes to reasoning until </think>."""
        router = OutputRouter.from_tokenizer(QWEN3_TOKENIZER)
        assert router is not None

        assert router.feed(248068) is None  # <think>
        assert router.state == RouterState.THINKING

        reasoning = router.feed(1)  # Reason
        assert reasoning is not None
        assert reasoning.channel == Channel.REASONING
        assert reasoning.text == "Reason"

        assert router.feed(248069) is None  # </think>
        assert router.state == RouterState.CONTENT

        content = router.feed(3)  # Answer
        assert content is not None
        assert content.channel == Channel.CONTENT
        assert content.text == "Answer"

    def test_orphan_think_end_switches_to_content(self):
        """A lone </think> is suppressed and following tokens are content."""
        router = OutputRouter.from_tokenizer(QWEN3_TOKENIZER)
        assert router is not None

        assert router.feed(248069) is None
        assert router.state == RouterState.CONTENT

        event = router.feed(3)
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_no_tag_output_routes_as_content(self):
        """Plain Qwen3 output with no think tags passes through as content."""
        router = OutputRouter.from_tokenizer(QWEN3_TOKENIZER)
        assert router is not None

        event = router.feed(4)
        assert event is not None
        assert event.channel == Channel.CONTENT
        assert event.text == "Plain"

    def test_feed_sequence_separates_reasoning_and_content(self):
        """Batch routing separates Qwen3 reasoning and answer text."""
        router = OutputRouter.from_tokenizer(QWEN3_TOKENIZER)
        assert router is not None

        result = router.feed_sequence([248068, 1, 2, 248069, 3])
        assert result["reasoning"] == "Reasoning"
        assert result["content"] == "Answer"
        assert result["tool_calls"] is None


class TestDeepSeekR1ThinkRouting:
    """Test DeepSeek R1 <think> routing."""

    def test_unicode_bos_eos_are_suppressed(self):
        """DeepSeek's unicode BOS/EOS special tokens do not leak."""
        router = OutputRouter.from_tokenizer(DEEPSEEK_R1_TOKENIZER)
        assert router is not None

        assert router.feed(151646) is None
        assert router.feed(151643) is None

    def test_think_block_routes_to_reasoning_then_content(self):
        """<think> payload routes to reasoning until </think>."""
        router = OutputRouter.from_tokenizer(DEEPSEEK_R1_TOKENIZER)
        assert router is not None

        assert router.feed(151648) is None  # <think>
        assert router.state == RouterState.THINKING

        reasoning = router.feed(1)  # Step
        assert reasoning is not None
        assert reasoning.channel == Channel.REASONING
        assert reasoning.text == "Step"

        assert router.feed(151649) is None  # </think>
        assert router.state == RouterState.CONTENT

        content = router.feed(3)  # Answer
        assert content is not None
        assert content.channel == Channel.CONTENT
        assert content.text == "Answer"

    def test_orphan_think_end_switches_to_content(self):
        """A lone </think> is suppressed and following tokens are content."""
        router = OutputRouter.from_tokenizer(DEEPSEEK_R1_TOKENIZER)
        assert router is not None

        assert router.feed(151649) is None
        assert router.state == RouterState.CONTENT

        event = router.feed(3)
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_no_tag_output_routes_as_content(self):
        """Plain DeepSeek output with no think tags passes through as content."""
        router = OutputRouter.from_tokenizer(DEEPSEEK_R1_TOKENIZER)
        assert router is not None

        event = router.feed(4)
        assert event is not None
        assert event.channel == Channel.CONTENT
        assert event.text == "Plain"

    def test_feed_sequence_separates_reasoning_and_content(self):
        """Batch routing separates DeepSeek reasoning and answer text."""
        router = OutputRouter.from_tokenizer(DEEPSEEK_R1_TOKENIZER)
        assert router is not None

        result = router.feed_sequence([151648, 1, 2, 151649, 3])
        assert result["reasoning"] == "Step one"
        assert result["content"] == "Answer"
        assert result["tool_calls"] is None


class TestHarmonyRouting:
    """Test GPT-OSS/Harmony channel routing."""

    def test_analysis_channel_routes_message_to_reasoning(self):
        """Harmony analysis payload routes to REASONING."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        assert router.feed(200005) is None  # <|channel|>
        assert router.state == RouterState.AWAITING_CHANNEL_TYPE
        assert router.feed(35644) is None  # analysis
        assert router.state == RouterState.AWAITING_MESSAGE
        assert router.feed(200008) is None  # <|message|>
        assert router.state == RouterState.THINKING

        event = router.feed(2)  # Reason
        assert event is not None
        assert event.channel == Channel.REASONING
        assert event.text == "Reason"

        assert router.feed(200007) is None  # <|end|>
        assert router.state == RouterState.CONTENT

    def test_final_channel_routes_message_to_content(self):
        """Harmony final payload routes to CONTENT."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        assert router.feed(200005) is None  # <|channel|>
        assert router.feed(17196) is None  # final
        assert router.feed(200008) is None  # <|message|>

        event = router.feed(4)  # Answer
        assert event is not None
        assert event.channel == Channel.CONTENT
        assert event.text == "Answer"

        assert router.feed(200002) is None  # <|return|>
        assert router.state == RouterState.CONTENT

    def test_premessage_metadata_is_suppressed(self):
        """Tokens before <|message|> in a Harmony channel do not leak."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        router.feed(200005)  # <|channel|>
        router.feed(17196)  # final
        assert router.feed(200003) is None  # <|constrain|>
        assert router.feed(1) is None  # " json"
        assert router.feed(200008) is None  # <|message|>

        event = router.feed(4)
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_orphan_harmony_controls_are_suppressed(self):
        """Harmony structural tokens do not leak outside a channel."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        assert router.feed(200006) is None  # <|start|>
        assert router.feed(200007) is None  # <|end|>
        assert router.feed(200002) is None  # <|return|>
        assert router.feed(200012) is None  # <|call|>
        assert router.feed(200008) is None  # <|message|>

        event = router.feed(5)
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_start_followed_by_role_name_swallows_role(self):
        """``<|start|>assistant<|channel|>`` — the assistant role token
        must not leak as CONTENT.

        Regression for the second-and-later assistant-turn streaming leak
        surfaced by the v0.6.65 fresh-PyPI onboarding smoke on
        ``mlx-community/gpt-oss-20b-MXFP4-Q8``: streamed ``delta.content``
        began with literal ``assistant`` before the real answer. Without
        ``AFTER_START`` state, the role-name token fell through to the
        default CONTENT emission path.
        """
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        assert router.feed(200006) is None  # <|start|>
        assert router.state == RouterState.AFTER_START
        # The role-name token MUST be suppressed (returns None). Before
        # the fix this returned a CONTENT event with text='assistant'.
        assert router.feed(173781) is None  # ``assistant`` — role token
        assert router.state == RouterState.AFTER_START
        assert router.feed(200005) is None  # <|channel|>
        assert router.state == RouterState.AWAITING_CHANNEL_TYPE
        assert router.feed(17196) is None  # final
        assert router.feed(200008) is None  # <|message|>

        event = router.feed(4)  # Answer
        assert event is not None
        assert event.channel == Channel.CONTENT
        assert event.text == "Answer"

    def test_start_with_recipient_info_swallows_all(self):
        """``<|start|>assistant to=functions.get_weather<|channel|>`` —
        every header token (role + recipient metadata) is swallowed,
        only the payload after ``<|message|>`` is emitted.
        """
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        assert router.feed(200006) is None  # <|start|>
        assert router.feed(173781) is None  # assistant
        assert router.feed(173783) is None  # " to=functions.get_weather"
        assert router.feed(200005) is None  # <|channel|>
        assert router.feed(17196) is None  # final (channel)
        assert router.feed(200008) is None  # <|message|>

        event = router.feed(4)
        assert event is not None
        assert event.channel == Channel.CONTENT

    def test_start_message_path_skips_channel(self):
        """``<|start|>{role}<|message|>{body}`` — system/user/tool turns
        with no channel marker still suppress the role and emit only
        the body. Harmony spec allows this shorter header form.
        """
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        assert router.feed(200006) is None  # <|start|>
        assert router.feed(173782) is None  # user (role)
        assert router.feed(200008) is None  # <|message|>
        assert router.state == RouterState.CONTENT

        event = router.feed(5)  # Plain
        assert event is not None
        assert event.channel == Channel.CONTENT
        assert event.text == "Plain"

    def test_second_assistant_turn_no_role_leak_end_to_end(self):
        """Full reproduction of the bug. The model emits an analysis
        channel terminated with ``<|end|>``, then opens a SECOND header
        ``<|start|>assistant<|channel|>final<|message|>…``. Before the
        fix, the second turn's ``assistant`` token leaked into CONTENT
        because ``<|end|>`` transitioned state to CONTENT but the
        next-up ``<|start|>`` didn't establish AFTER_START.
        """
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        result = router.feed_sequence(
            [
                200005,  # <|channel|>
                35644,  # analysis
                200008,  # <|message|>
                2,  # Reason
                3,  # ing
                200007,  # <|end|>  → state=CONTENT
                200006,  # <|start|>  → state=AFTER_START (the fix)
                173781,  # assistant ← THIS USED TO LEAK AS CONTENT
                200005,  # <|channel|>
                17196,  # final
                200008,  # <|message|>
                4,  # Answer
                200002,  # <|return|>
            ]
        )
        assert result["reasoning"] == "Reasoning"
        assert result["content"] == "Answer", (
            f"role token leaked into content: {result['content']!r}"
        )

    def test_feed_sequence_separates_analysis_and_final(self):
        """Batch routing separates Harmony analysis and final channels."""
        router = OutputRouter.from_tokenizer(HARMONY_TOKENIZER)
        assert router is not None

        result = router.feed_sequence(
            [
                200005,
                35644,
                200008,
                2,
                3,
                200007,
                200005,
                17196,
                200008,
                4,
                200002,
            ]
        )
        assert result["reasoning"] == "Reasoning"
        assert result["content"] == "Answer"
        assert result["tool_calls"] is None


class TestStateReset:
    """Test state management."""

    def test_reset_clears_state(self, router):
        """Reset returns to INIT state."""
        router.feed(100)  # enter channel
        router.feed(45518)  # thinking
        assert router.state == RouterState.THINKING

        router.reset()
        assert router.state == RouterState.INIT
        assert router._tool_tokens == []
        assert router._pending_channel_style is None
        assert router._pending_message_channel is None
        assert router._pending_control_tokens == []

    def test_multiple_requests(self, router):
        """Router works correctly across multiple reset cycles."""
        # Request 1
        router.feed(100)
        router.feed(45518)  # thinking
        e1 = router.feed(651)
        assert e1.channel == Channel.REASONING

        # Request 2
        router.reset()
        e2 = router.feed(9259)  # "Hello"
        assert e2.channel == Channel.CONTENT


# ============================================================================
# #516 — HarmonyStreamingRouter auto-upgrade escape hatches (release-SOP G11)
# ============================================================================


class TestStreamingFactoryEscapeHatches:
    """The streaming-factory auto-upgrade from the legacy custom harmony
    state machine to ``HarmonyStreamingRouter`` (PR #515) is a binary
    auto-routing decision. The SOP requires both a force-on and a
    force-off CLI flag plumbed to the factory; these tests pin the
    branches at the factory level.

    We monkey-patch the two dependencies (``is_openai_harmony_compatible``
    and ``HarmonyStreamingRouter``) so we don't need a real harmony
    tokenizer — only the dispatch tree is under test here.
    """

    def _patch_factory(self, monkeypatch, *, gate_returns: bool):
        """Substitute the harmony compat probe + constructor with stubs
        so the test exercises the factory branches without needing a
        real tokenizer.
        """
        import vllm_mlx.output_router_harmony as orh

        def fake_is_compat(_token_map, _tokenizer):
            return gate_returns

        class FakeHarmonyRouter:
            map = None  # populated in __init__

            def __init__(self, token_map, _tokenizer):
                self.map = token_map
                self.format_tag = "harmony"

        monkeypatch.setattr(orh, "is_openai_harmony_compatible", fake_is_compat)
        monkeypatch.setattr(orh, "HarmonyStreamingRouter", FakeHarmonyRouter)
        return FakeHarmonyRouter

    def _harmony_legacy(self, monkeypatch):
        """Stub ``from_tokenizer`` to return a legacy router whose
        ``format_tag`` is 'harmony'. Returns the stub instance.
        """

        class _LegacyStub:
            class _Map:
                format_tag = "harmony"

            map = _Map()

        monkeypatch.setattr(
            OutputRouter, "from_tokenizer", classmethod(lambda cls, tok: _LegacyStub())
        )
        return _LegacyStub

    def _non_harmony_legacy(self, monkeypatch):
        class _LegacyStub:
            class _Map:
                format_tag = "gemma4"

            map = _Map()

        monkeypatch.setattr(
            OutputRouter, "from_tokenizer", classmethod(lambda cls, tok: _LegacyStub())
        )
        return _LegacyStub

    def test_default_returns_harmony_when_gate_accepts(self, monkeypatch):
        FakeHarmony = self._patch_factory(monkeypatch, gate_returns=True)
        self._harmony_legacy(monkeypatch)

        router = OutputRouter.from_tokenizer_for_streaming(object())
        assert isinstance(router, FakeHarmony)

    def test_default_returns_legacy_when_gate_rejects(self, monkeypatch):
        FakeHarmony = self._patch_factory(monkeypatch, gate_returns=False)
        legacy_cls = self._harmony_legacy(monkeypatch)

        router = OutputRouter.from_tokenizer_for_streaming(object())
        assert not isinstance(router, FakeHarmony)
        assert isinstance(router, legacy_cls)

    def test_no_harmony_streaming_forces_legacy_even_when_gate_accepts(
        self, monkeypatch
    ):
        FakeHarmony = self._patch_factory(monkeypatch, gate_returns=True)
        legacy_cls = self._harmony_legacy(monkeypatch)

        router = OutputRouter.from_tokenizer_for_streaming(
            object(), no_harmony_streaming=True
        )
        assert not isinstance(router, FakeHarmony)
        assert isinstance(router, legacy_cls)

    def test_force_harmony_streaming_bypasses_gate(self, monkeypatch):
        """Force-on must construct HarmonyStreamingRouter even when the
        compat probe returns False — debug-only path. The constructor
        will surface real failures on its own.
        """
        FakeHarmony = self._patch_factory(monkeypatch, gate_returns=False)
        self._harmony_legacy(monkeypatch)

        router = OutputRouter.from_tokenizer_for_streaming(
            object(), force_harmony_streaming=True
        )
        assert isinstance(router, FakeHarmony)

    def test_force_harmony_streaming_rejects_non_harmony_format(self, monkeypatch):
        """If the tokenizer isn't even harmony-shaped, --force-on must
        raise pointing at the right escape hatch — silently constructing
        a harmony router on a non-harmony tokenizer would crash later
        with an opaque error.
        """
        self._patch_factory(monkeypatch, gate_returns=True)
        self._non_harmony_legacy(monkeypatch)

        with pytest.raises(ValueError, match="not 'harmony'"):
            OutputRouter.from_tokenizer_for_streaming(
                object(), force_harmony_streaming=True
            )

    def test_returns_none_when_legacy_returns_none(self, monkeypatch):
        """If the legacy factory returns None (unsupported tokenizer),
        the override flags must NOT manufacture a router — that would
        crash downstream.
        """
        monkeypatch.setattr(
            OutputRouter, "from_tokenizer", classmethod(lambda cls, tok: None)
        )

        assert OutputRouter.from_tokenizer_for_streaming(object()) is None
        assert (
            OutputRouter.from_tokenizer_for_streaming(
                object(), no_harmony_streaming=True
            )
            is None
        )
        assert (
            OutputRouter.from_tokenizer_for_streaming(
                object(), force_harmony_streaming=True
            )
            is None
        )

    def test_both_flags_true_raises(self, monkeypatch):
        """PR #518 round-2 codex NIT: setting BOTH force-on and force-off
        is incoherent — the CLI layer enforces mutual exclusion, but
        direct API callers (third-party engines, tests) deserve the
        same guardrail. Factory must raise instead of silently honoring
        one over the other.
        """
        self._patch_factory(monkeypatch, gate_returns=True)
        self._harmony_legacy(monkeypatch)

        with pytest.raises(ValueError, match="mutually exclusive"):
            OutputRouter.from_tokenizer_for_streaming(
                object(),
                force_harmony_streaming=True,
                no_harmony_streaming=True,
            )
