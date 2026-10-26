"""Prompt evidence is captured once, after scheduler preparation, without GPU work."""

import asyncio
from types import SimpleNamespace

import pytest

from rapid_mlx.engine_core import EngineCore


@pytest.mark.parametrize("fails", [False, True])
async def test_prompt_capture_uses_admitted_ids_before_commit(fails):
    core = EngineCore.__new__(EngineCore)
    ids = [4, 5, 6]
    events = []

    def prepare(request):
        # Represents the scheduler's actual tokenization/compression result.
        request.prompt_token_ids = ids
        events.append("prepared")

    def capture(actual):
        assert actual is ids  # No full-prompt copy on the metadata path.
        events.append("captured")
        if fails:
            raise ValueError("optional metadata failure")

    core.scheduler = SimpleNamespace(add_request=prepare)
    core._hybrid_throttle = False
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core._mlx_executor = None
    core._idle_event = asyncio.Event()
    core.config = SimpleNamespace(stream_interval=1)
    result = await core.add_request(
        "rendered",
        request_id="context",
        on_prompt_prepared=capture,
        on_request_committed=lambda: events.append("committed"),
    )
    assert result == "context"
    assert events == ["prepared", "captured", "committed"]
    assert core._idle_event.is_set()


@pytest.mark.parametrize("streaming", [False, True])
async def test_batched_transport_encodes_full_prompt_only_in_scheduler(
    monkeypatch, streaming
):
    from unittest.mock import AsyncMock, MagicMock

    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.request import RequestOutput

    encoded = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            encoded.append(text)
            return list(text.encode())

    tokenizer = Tokenizer()
    prompt = (
        "<|im_start|>user\n"
        + "history" * 10000
        + "<|im_end|><|im_start|>assistant\n<think>"
    )
    engine = BatchedEngine("test-model", force_text=True)
    engine._loaded = True
    engine._tokenizer = tokenizer
    engine._engine = MagicMock()
    engine._engine.engine.scheduler.config.max_concurrent_requests = 1
    monkeypatch.setattr(engine, "_create_output_router", lambda: None)
    monkeypatch.setattr(engine, "_needs_prefix_boundary_snapshot", lambda: False)
    monkeypatch.setattr(
        engine,
        "_apply_chat_template",
        lambda *a, **kw: (
            prompt[: prompt.rfind("<|im_start|>assistant\n")]
            if kw.get("add_generation_prompt") is False
            else prompt
        ),
    )
    monkeypatch.setattr(
        engine, "encode_prompt_text", lambda *_: pytest.fail("duplicate encode")
    )
    output = RequestOutput(
        request_id="context",
        output_token_ids=[42],
        new_token_ids=[42],
        output_text="answer",
        new_text="answer",
        finished=True,
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=1,
    )

    async def generate(**kwargs):
        ids = tokenizer.encode(kwargs["prompt"])
        kwargs["on_prompt_prepared"](ids)
        kwargs["on_request_committed"]()
        return output

    async def add_request(**kwargs):
        await generate(**kwargs)
        return "context"

    async def stream_outputs(request_id):
        yield output

    engine._engine.generate = generate
    engine._engine.add_request = add_request
    engine._engine.stream_outputs = stream_outputs
    engine._engine.abort_request = AsyncMock()
    if streaming:
        results = [
            out async for out in engine.stream_chat([{"role": "user", "content": "hi"}])
        ]
        result = results[-1]
    else:
        result = await engine.chat([{"role": "user", "content": "hi"}])
    assert result.prompt_parse_context.assistant_suffix_ids == tuple(b"<think>")
    assert encoded == [prompt, "<think>"]
