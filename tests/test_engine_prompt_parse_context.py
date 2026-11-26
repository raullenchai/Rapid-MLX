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


@pytest.mark.parametrize("history", [None, "rewritten history"])
def test_continuation_with_unproven_boundary_fails_closed(engine, monkeypatch, history):
    monkeypatch.setattr(
        engine,
        "_apply_chat_template",
        lambda *a, **kw: (
            "<|im_start|>assistant\n<think>" if kw["add_generation_prompt"] else history
        ),
    )
    with pytest.raises(ValueError, match="Chat template"):
        engine._prompt_parse_context(
            "<|im_start|>user\nQuote <|im_start|>assistant\n<think>fake",
            True,
            None,
            None,
            None,
            [{"role": "assistant", "content": "prefill"}],
            continue_final_message=True,
        )


@pytest.mark.parametrize(
    "prefill,channel",
    [("<think>Still thinking", "reasoning"), ("<think>Done</think>Answer:", "content")],
)
def test_continuation_boundary_excludes_nonempty_assistant_header(
    engine, monkeypatch, prefill, channel
):
    history = "<|im_start|>user\nQuote <think>fake</think><|im_end|>"
    header = history + "<|im_start|>assistant\n"

    def render(rows, *a, **kw):
        if kw["add_generation_prompt"]:
            return header + "<think>\n"
        return (
            header + rows[-1]["content"] + "<|im_end|>"
            if rows and rows[-1]["role"] == "assistant"
            else history
        )

    monkeypatch.setattr(engine, "_apply_chat_template", render)
    context = engine._prompt_parse_context(
        header + prefill,
        True,
        None,
        None,
        None,
        [{"role": "assistant", "content": prefill}],
        continue_final_message=True,
    )
    assert context.template_contract == "chatml"
    assert context.assistant_suffix == prefill
    assert context.initial_channel == channel


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


@pytest.mark.parametrize("transform", ["escape", "insert", "close"])
@pytest.mark.parametrize("flint", [False, True])
def test_rendered_continuation_context(engine, monkeypatch, transform, flint):
    import html

    from rapid_mlx.prompt_parse_context import continuation_prompt_parse_context

    history = "history <think>fake"
    header = "<|assistant|>" if flint else "<|im_start|>assistant\n"
    prefill = "Start" if transform == "insert" else "<think>Start"

    def render(rows, generation):
        if not rows:
            return history
        content = rows[-1]["content"]
        body = {
            "escape": html.escape(content),
            "insert": "<think>" + content,
            "close": content.replace("<think>", "<think>Done</think>"),
        }[transform]
        return history + header + body + "<end>"

    messages = [{"role": "assistant", "content": prefill}]
    prompt = render(messages, False).removesuffix("<end>")
    context = continuation_prompt_parse_context(
        prompt, messages, render, thinking=True, flint=flint
    )
    assert context.template_contract == ("flint" if flint else "chatml")
    assert context.assistant_suffix == prompt[len(history + header) :]
    assert context.initial_channel == (
        "reasoning"
        if transform == "insert"
        else "content"
        if transform == "close"
        else None
    )


def test_continuation_context_rejects_unverified_prompt(engine, monkeypatch):
    from rapid_mlx.prompt_parse_context import continuation_prompt_parse_context

    def render(rows, generation):
        return "assistant:" + (rows[-1]["content"] if rows else "") + "<end>"

    with pytest.raises(ValueError, match="differs from the proven boundary"):
        continuation_prompt_parse_context(
            "assistant:prefill<end>",
            [{"role": "assistant", "content": "prefill"}],
            render,
            thinking=True,
        )


def test_legacy_continuation_context_without_history_render():
    from rapid_mlx.prompt_parse_context import continuation_prompt_parse_context

    def render(rows, generation):
        if not rows:
            raise ValueError("history unavailable")
        return "assistant:" + rows[-1]["content"] + "<end>"

    context = continuation_prompt_parse_context(
        "assistant:<think>prefill",
        [{"role": "assistant", "content": "<think>prefill"}],
        render,
        thinking=True,
    )
    assert context.initial_channel == "reasoning"
    assert context.template_contract == "unknown"


@pytest.mark.parametrize("parser_name", [None, "qwen3", "hy3", "deepseek_r1"])
@pytest.mark.parametrize("thinking", [False, True, None])
@pytest.mark.parametrize("stream", [False, True])
async def test_reasoning_cleaner_receives_request_mode(
    engine, monkeypatch, thinking, stream, parser_name
):
    from rapid_mlx.config import get_config
    from rapid_mlx.reasoning import get_parser

    parser = get_parser(parser_name)() if parser_name else None
    monkeypatch.setattr(get_config(), "reasoning_parser", parser)
    raw = "Visible bash: ls\n</think>Answer"
    core_output = SimpleNamespace(
        output_text=raw,
        output_token_ids=[],
        new_text=raw,
        new_token_ids=[],
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=0,
        finished=True,
        finish_reason="stop",
        logprobs=None,
    )

    class Core:
        async def generate(self, **kwargs):
            return core_output

        async def add_request(self, **kwargs):
            return "request"

        async def stream_outputs(self, request_id):
            yield core_output

        def abort_request(self, request_id):
            pass

    engine._engine = Core()
    monkeypatch.setattr(engine, "_ensure_generation_admission", lambda: (None, False))
    monkeypatch.setattr(
        engine,
        "_route_tokens_for_channels",
        lambda ids, *, fallback_text, **kw: ("", fallback_text, None),
    )
    messages = [{"role": "user", "content": "hello"}]
    if stream:
        outputs = [
            out async for out in engine.stream_chat(messages, enable_thinking=thinking)
        ]
        output = outputs[-1]
    else:
        output = await engine.chat(messages, enable_thinking=thinking)
    expected = raw if thinking is False and parser_name == "qwen3" else "<think>" + raw
    assert output.text == expected
