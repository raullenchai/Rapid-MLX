# SPDX-License-Identifier: Apache-2.0
"""Prompt metadata follows each request through buffered and streamed engines."""

import asyncio
from types import SimpleNamespace

import pytest

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.engine.batched import BatchedEngine
from tests.test_prompt_parse_context import Tokenizer


@pytest.fixture
def engine(monkeypatch):
    engine = BatchedEngine("test-model", force_text=True)
    engine._loaded = True
    engine._is_mllm = False
    engine._tokenizer = Tokenizer()
    monkeypatch.setattr(engine, "_create_output_router", lambda: None)
    monkeypatch.setattr(engine, "_needs_prefix_boundary_snapshot", lambda: False)
    monkeypatch.setattr(engine, "encode_prompt_text", Tokenizer.encode)
    monkeypatch.setattr(
        engine,
        "_apply_chat_template",
        lambda messages, *a, **kw: (
            "<|im_start|>assistant\n"
            + ("<think>\n" if kw.get("enable_thinking") else "<think></think>\n")
        ),
    )
    return engine


@pytest.mark.parametrize("prefix", [None, '<tool_call>{"name":"lookup","arguments":'])
@pytest.mark.parametrize("thinking", [False, True])
async def test_stream_and_buffered_transport(engine, monkeypatch, prefix, thinking):
    async def generate(**kwargs):
        return GenerationOutput(text="answer")

    async def stream_generate(**kwargs):
        for text in ["first", "second"]:
            yield GenerationOutput(text=text, new_text=text, finished=False)
        yield GenerationOutput(text="firstsecond", finished=True)

    monkeypatch.setattr(engine, "generate", generate)
    monkeypatch.setattr(engine, "stream_generate", stream_generate)
    kwargs = dict(enable_thinking=thinking, forced_assistant_prefix=prefix)
    messages = [{"role": "user", "content": "hello"}]
    buffered = await engine.chat(messages, **kwargs)
    streamed = [out async for out in engine.stream_chat(messages, **kwargs)]
    context = buffered.prompt_parse_context
    assert context.initial_channel == (
        None if thinking and prefix else "reasoning" if thinking else "content"
    )
    assert context.forced_assistant_prefix == (prefix or "")
    assert context.assistant_suffix_ids == tuple(
        Tokenizer.encode(context.assistant_suffix)
    )
    assert all(out.prompt_parse_context == context for out in streamed)
    assert len({id(out.prompt_parse_context) for out in streamed}) == 1
    assert streamed[0].new_text == (prefix or "first")


async def test_concurrent_requests_keep_context_separate(engine, monkeypatch):
    async def generate(**kwargs):
        await asyncio.sleep(0)
        return GenerationOutput(text="answer")

    monkeypatch.setattr(engine, "generate", generate)
    outputs = await asyncio.gather(
        *[
            engine.chat([{"role": "user", "content": "hello"}], enable_thinking=flag)
            for flag in [True, False]
        ]
    )
    assert [o.prompt_parse_context.initial_channel for o in outputs] == [
        "reasoning",
        "content",
    ]


def test_native_and_unknown_contracts(engine, monkeypatch):
    assert engine._prompt_parse_context([1, 2], False, None, None, None) is None
    unknown = engine._prompt_parse_context("native", False, None, None, None)
    assert unknown.initial_channel is None
    engine._is_mllm = True
    engine._processor = SimpleNamespace(chat_template="custom")
    assert engine._prompt_parse_context("native", False, None, None, None) == unknown


async def test_admitted_forced_prefix_context_precedes_model_output(
    engine, monkeypatch
):
    admitted = asyncio.Event()

    async def stream_generate(**kwargs):
        admitted.set()
        await asyncio.sleep(0)
        yield GenerationOutput(text="body", new_text="body")

    monkeypatch.setattr(engine, "stream_generate", stream_generate)
    outputs = [
        output
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            enable_thinking=False,
            forced_assistant_prefix="<tool_call>",
            request_admitted_event=admitted,
        )
    ]
    assert [o.new_text for o in outputs] == ["<tool_call>", "body"]
    assert outputs[0].prompt_parse_context is outputs[1].prompt_parse_context


@pytest.mark.parametrize("streaming", [False, True])
async def test_optional_suffix_encoder_failure_preserves_generation(
    engine, monkeypatch, streaming
):
    class BrokenTokenizer:
        def encode(self, *args, **kwargs):
            raise TypeError("suffix encoding not supported")

    engine._tokenizer = BrokenTokenizer()

    async def generate(**kwargs):
        return GenerationOutput(text="answer")

    async def stream_generate(**kwargs):
        yield GenerationOutput(text="answer", new_text="answer")

    monkeypatch.setattr(engine, "generate", generate)
    monkeypatch.setattr(engine, "stream_generate", stream_generate)
    messages = [{"role": "user", "content": "hello"}]
    if streaming:
        outputs = [
            out async for out in engine.stream_chat(messages, enable_thinking=True)
        ]
    else:
        outputs = [await engine.chat(messages, enable_thinking=True)]
    assert outputs[0].text == "answer"
    assert outputs[0].prompt_parse_context.assistant_suffix_ids is None
    assert outputs[0].prompt_parse_context.initial_channel == "reasoning"
