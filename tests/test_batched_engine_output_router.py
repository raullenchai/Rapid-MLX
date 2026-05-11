# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine token-level output routing."""

from collections.abc import AsyncIterator

import pytest

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine


class FakeTokenizer:
    """Minimal tokenizer for OutputRouter detection and decoding."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self._id_to_text = {v: k for k, v in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_text.get(i, f"<UNK:{i}>") for i in ids)


# Harmony vocab IDs mirror the GPT-OSS tokenizer subset used by OutputRouter.
HARMONY_VOCAB = {
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|call|>": 200012,
    "<|endoftext|>": 200019,
    "analysis": 35644,
    "final": 17196,
    "Reason": 2,
    "ing": 3,
    "Answer": 4,
    "Fallback": 5,
}

QWEN3_VOCAB = {
    "<think>": 248068,
    "</think>": 248069,
    "Reason": 2,
    "Answer": 4,
}

GEMMA4_VOCAB = {
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
    "thought": 45518,
    "content": 3955,
    "final": 10218,
    "call": 6639,
    ":": 236787,
    "get": 828,
    "_": 236779,
    "weather": 19323,
    "{": 236782,
    "}": 236783,
    "city": 13319,
    "Tokyo": 89265,
}


def _make_engine(tokenizer: FakeTokenizer) -> BatchedEngine:
    engine = BatchedEngine("fake-model")
    engine._loaded = True
    engine._tokenizer = tokenizer
    engine._apply_chat_template = lambda *args, **kwargs: "prompt"
    engine._compute_prefix_boundary = lambda *args, **kwargs: 0
    return engine


async def _collect(
    outputs: AsyncIterator[GenerationOutput],
) -> list[GenerationOutput]:
    return [output async for output in outputs]


@pytest.mark.asyncio
async def test_stream_chat_routes_supported_tokenizer_channels():
    """Supported tokenizers emit channel-tagged chunks and suppress controls."""
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<|channel|>analysis<|message|>Reason",
            tokens=[200005, 35644, 200008, 2],
            finished=False,
        )
        yield GenerationOutput(
            text="",
            new_text="ing<|start|><|channel|>final<|message|>Answer",
            tokens=[3, 200006, 200005, 17196, 200008, 4],
            finished=True,
            finish_reason="stop",
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert [(o.new_text, o.channel, o.finished) for o in outputs] == [
        ("Reason", "reasoning", False),
        ("ing", "reasoning", False),
        ("Answer", "content", True),
    ]
    assert all("<|channel|>" not in output.new_text for output in outputs)
    assert all(output.logprobs is None for output in outputs)


@pytest.mark.asyncio
async def test_stream_chat_keeps_think_tag_tokenizers_on_legacy_path():
    """Think-tag routers are detected but not engine-enabled until validated."""
    engine = _make_engine(FakeTokenizer(QWEN3_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<think>Reason</think>Answer",
            tokens=[248068, 2, 248069, 4],
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert len(outputs) == 1
    assert outputs[0].new_text == "<think>Reason</think>Answer"
    assert outputs[0].channel is None


@pytest.mark.asyncio
async def test_stream_chat_routes_tool_call_channel_on_finish():
    """Truncated tool calls are drained as tool_call channel output."""
    engine = _make_engine(FakeTokenizer(GEMMA4_VOCAB))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="<|tool_call>call:get_weather{city:Tokyo}",
            tokens=[
                48,
                6639,
                236787,
                828,
                236779,
                19323,
                236782,
                13319,
                236787,
                89265,
                236783,
            ],
            finished=True,
            finish_reason="length",
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert [(o.channel, o.finished, o.finish_reason) for o in outputs] == [
        ("tool_call", True, "length")
    ]
    assert "get_weather" in outputs[0].new_text
    assert "Tokyo" in outputs[0].new_text
    assert outputs[0].logprobs is None


@pytest.mark.asyncio
async def test_stream_chat_uses_incremental_new_text_for_single_token_events():
    """Single-token routed chunks preserve scheduler detokenizer text."""
    vocab = {
        **HARMONY_VOCAB,
        "decoded-wrong": 6,
    }
    tokenizer = FakeTokenizer(vocab)
    tokenizer._id_to_text[6] = "decoded-wrong"
    engine = _make_engine(tokenizer)

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="",
            new_text="decoded-right",
            tokens=[6],
            finished=True,
            finish_reason="stop",
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert outputs[0].new_text == "decoded-right"
    assert outputs[0].channel == "content"


@pytest.mark.asyncio
async def test_stream_chat_leaves_unsupported_tokenizer_on_legacy_path():
    """Unsupported tokenizers preserve raw chunks with channel=None."""
    engine = _make_engine(FakeTokenizer({"Hello": 1}))

    async def fake_stream_generate(**kwargs):
        yield GenerationOutput(
            text="Hello",
            new_text="Hello",
            tokens=[1],
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    engine.stream_generate = fake_stream_generate

    outputs = await _collect(
        engine.stream_chat(messages=[{"role": "user", "content": "hi"}])
    )

    assert len(outputs) == 1
    assert outputs[0].new_text == "Hello"
    assert outputs[0].tokens == [1]
    assert outputs[0].channel is None
    assert outputs[0].finished is True


@pytest.mark.asyncio
async def test_stream_chat_falls_back_after_router_failure():
    """A mid-stream router failure disables routing for later chunks."""
    engine = _make_engine(FakeTokenizer(HARMONY_VOCAB))

    class FailingRouter:
        def feed(self, token_id):
            raise RuntimeError("boom")

    async def fake_outputs():
        yield GenerationOutput(
            text="",
            new_text="Fallback",
            tokens=[5],
            finished=False,
            channel=None,
        )
        yield GenerationOutput(
            text="",
            new_text="Answer",
            tokens=[4],
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    outputs = await _collect(
        engine._stream_with_output_router(fake_outputs(), FailingRouter())
    )

    assert [(o.new_text, o.channel, o.finished) for o in outputs] == [
        ("Fallback", None, False),
        ("Answer", None, True),
    ]


def test_create_output_router_catches_tokenizer_property_errors():
    """Tokenizer access failures fall back to legacy parsing."""

    class BrokenTokenizerEngine(BatchedEngine):
        @property
        def tokenizer(self):
            raise RuntimeError("not loaded")

    engine = BrokenTokenizerEngine("fake-model")

    assert engine._create_output_router() is None
