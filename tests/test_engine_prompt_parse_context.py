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
    monkeypatch.setattr(
        engine,
        "encode_prompt_text",
        lambda *_: pytest.fail("metadata must not re-encode the whole prompt"),
    )
    monkeypatch.setattr(
        engine,
        "_apply_chat_template",
        lambda messages, *a, **kw: (
            ""
            if kw.get("add_generation_prompt") is False
            else (
                "<|im_start|>assistant\n"
                + ("<think>\n" if kw.get("enable_thinking") else "<think></think>\n")
            )
        ),
    )
    return engine


@pytest.mark.parametrize("prefix", [None, '<tool_call>{"name":"lookup","arguments":'])
@pytest.mark.parametrize("thinking", [False, True])
async def test_stream_and_buffered_transport(engine, monkeypatch, prefix, thinking):
    async def generate(**kwargs):
        kwargs["_on_prompt_prepared"](Tokenizer.encode(kwargs["prompt"]))
        return GenerationOutput(text="answer")

    async def stream_generate(**kwargs):
        kwargs["_on_prompt_prepared"](Tokenizer.encode(kwargs["prompt"]))
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
    assert engine._prompt_parse_context([1, 2], False, None, None, None, []) is None
    unknown = engine._prompt_parse_context("native", False, None, None, None, [])
    assert unknown.initial_channel is None
    engine._is_mllm = True
    engine._processor = SimpleNamespace(chat_template="custom")
    assert (
        engine._prompt_parse_context("native", False, None, None, None, []) == unknown
    )


@pytest.mark.parametrize("external_event", [False, True])
async def test_admitted_forced_prefix_context_precedes_model_output(
    engine, monkeypatch, external_event
):
    release_model = asyncio.Event()

    async def stream_generate(**kwargs):
        kwargs["_on_prompt_prepared"](Tokenizer.encode(kwargs["prompt"]))
        kwargs["request_admitted_event"].set()
        await release_model.wait()
        yield GenerationOutput(text="body", new_text="body")

    monkeypatch.setattr(engine, "stream_generate", stream_generate)
    kwargs = {"request_admitted_event": asyncio.Event()} if external_event else {}
    stream = engine.stream_chat(
        [{"role": "user", "content": "hello"}],
        enable_thinking=False,
        forced_assistant_prefix="<tool_call>",
        **kwargs,
    )
    try:
        first = await asyncio.wait_for(anext(stream), timeout=1)
        assert first.new_text == "<tool_call>"
        context = first.prompt_parse_context
        assert context.assistant_suffix_ids == tuple(
            Tokenizer.encode("<think></think>\n<tool_call>")
        )
        assert not release_model.is_set()
        release_model.set()
        rest = [output async for output in stream]
        assert [o.new_text for o in rest] == ["body"]
        assert rest[0].prompt_parse_context is context
    finally:
        release_model.set()
        await stream.aclose()


@pytest.mark.parametrize("streaming", [False, True])
async def test_optional_suffix_encoder_failure_preserves_generation(
    engine, monkeypatch, streaming
):
    class BrokenTokenizer:
        def encode(self, *args, **kwargs):
            raise TypeError("suffix encoding not supported")

    engine._tokenizer = BrokenTokenizer()

    async def generate(**kwargs):
        kwargs["_on_prompt_prepared"](Tokenizer.encode(kwargs["prompt"]))
        return GenerationOutput(text="answer")

    async def stream_generate(**kwargs):
        kwargs["_on_prompt_prepared"](Tokenizer.encode(kwargs["prompt"]))
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


@pytest.mark.parametrize(
    "mode", ["ignored", "rewritten", "raises", "forced", "mismatch"]
)
def test_renderer_must_establish_generation_boundary(engine, monkeypatch, mode):
    prompt = "<|im_start|>user\nQuote <|im_start|>assistant\n<think>fake"
    prefix = (
        "<|im_start|>assistant\n<think>" if mode in {"forced", "mismatch"} else None
    )

    def render(*args, **kwargs):
        if mode == "raises":
            raise ValueError("history probe unsupported")
        return "rewritten history" if mode == "rewritten" else prompt

    monkeypatch.setattr(engine, "_apply_chat_template", render)
    context = engine._prompt_parse_context(
        prompt + (prefix or "" if mode != "mismatch" else ""),
        True,
        None,
        None,
        prefix,
        [],
    )
    assert context.initial_channel is None
    assert context.template_contract == "unknown"
    assert context.forced_assistant_prefix == (prefix or "")
