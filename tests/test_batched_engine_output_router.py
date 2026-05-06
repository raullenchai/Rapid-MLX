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
