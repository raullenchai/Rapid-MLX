# SPDX-License-Identifier: Apache-2.0
"""Wiring test: a completed STREAMING chat completion emits exactly one
``request`` telemetry event carrying ``stream=True``, the inbound
User-Agent (for ``caller_agent``), and a ``ttft_ms`` derived from the
FIRST emitted token — not total latency.

This is the streaming analogue of ``test_telemetry_request_wiring.py``.
Real agent traffic (Cursor / Claude Code / Aider) is almost entirely
streaming, so the non-streaming emit alone leaves ``caller_agent`` +
``ttft_ms`` essentially empty in production; this path is what turns
them real.

Drives ``stream_chat_completion`` directly against a fake engine (no
model load, no real generation) — the same harness as
``tests/test_447_stream_tool_choice_auto.py`` — and intercepts
``emit.request`` to assert the call-site contract. The bucketing of the
raw UA and the sampling gate are covered in test_telemetry_redact.py /
test_telemetry_emit.py; here we only assert the route reads the UA and
threads the right fields (including a first-token-derived TTFT) through.
"""

from __future__ import annotations

import asyncio
import time

import pytest

# Importing ``vllm_mlx.routes.chat`` pulls in ``mlx`` transitively; the
# Linux pr_validate runner has no mlx, so skip cleanly there.
pytest.importorskip("mlx")

from vllm_mlx.api.models import ChatCompletionRequest  # noqa: E402


class _FakeStreamingOutput:
    """Minimal ``GenerationOutput`` shim for the streaming loop."""

    def __init__(self, new_text: str, finished: bool):
        self.new_text = new_text
        self.text = new_text
        self.finished = finished
        self.finish_reason = "stop" if finished else None
        self.channel = None
        self.prompt_tokens = 11
        self.completion_tokens = 7
        self.cached_tokens = 0
        self.tokens = []
        self.logprobs = None
        self.tool_calls = None
        self.matched_stop = None
        self.raw_text = new_text


class _FakeEngine:
    """Yields a fixed plain-text delta sequence. Sleeps AFTER the first
    token so the first-token wall-clock is clearly distinct from total
    latency — this is what lets the test assert TTFT is measured at the
    first token, not the end of the stream."""

    def __init__(self, deltas: list[str], gap_after_first: float = 0.0):
        self._deltas = deltas
        self._gap = gap_after_first
        self.tokenizer = None
        self.is_mllm = False
        self.supports_tool_calls = False
        self.supports_guided_generation = False

    async def stream_chat(self, **kwargs):
        for i, d in enumerate(self._deltas):
            if i == 1 and self._gap:
                # Delay the REST of the stream, not the first token.
                await asyncio.sleep(self._gap)
            yield _FakeStreamingOutput(d, finished=(i == len(self._deltas) - 1))

    def build_prompt(self, *args, **kwargs):
        return "prompt"

    def estimate_new_tokens(self, *args, **kwargs):
        return (11, 7)


@pytest.fixture(autouse=True)
def _patch_cfg(monkeypatch):
    """Minimal plain-text streaming config: no reasoning / tool parser so
    each delta surfaces as a ``content`` event immediately."""
    from vllm_mlx.config import server_config

    cfg = server_config.get_config()
    monkeypatch.setattr(cfg, "tool_call_parser", None, raising=False)
    monkeypatch.setattr(cfg, "reasoning_parser_name", None, raising=False)
    monkeypatch.setattr(cfg, "reasoning_parser", None, raising=False)
    monkeypatch.setattr(cfg, "enable_auto_tool_choice", False, raising=False)
    monkeypatch.setattr(cfg, "cloud_router", None, raising=False)
    monkeypatch.setattr(cfg, "gc_control", False, raising=False)
    yield


def _request(model="test-model"):
    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        stream=True,
    )


def _drive_stream(engine, request, *, caller_agent, emit_calls):
    """Run ``stream_chat_completion`` to completion, capturing emit.request
    call kwargs. Returns total wall-clock seconds spent in the stream."""
    from vllm_mlx.routes import chat
    from vllm_mlx.telemetry import emit

    monkeypatch_target = emit
    orig = monkeypatch_target.request
    monkeypatch_target.request = lambda **kw: emit_calls.append(kw)

    async def _run():
        gen = chat.stream_chat_completion(
            engine,
            [{"role": "user", "content": "hi"}],
            request,
            caller_agent=caller_agent,
        )
        async for _sse in gen:
            # Drain the stream; we only care about the terminal emit.
            pass

    t0 = time.perf_counter()
    try:
        asyncio.run(_run())
    finally:
        monkeypatch_target.request = orig
    return time.perf_counter() - t0


def test_streaming_completion_emits_request_event_with_first_token_ttft():
    calls: list[dict] = []
    # 0.2s gap after the first token → total latency dwarfs TTFT.
    engine = _FakeEngine(["Hello", " ", "there", "!"], gap_after_first=0.2)
    total_s = _drive_stream(
        engine,
        _request(),
        caller_agent="cursor/1.9.0",
        emit_calls=calls,
    )

    assert len(calls) == 1, f"expected exactly one request event, got {calls}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/chat/completions"
    assert kw["stream"] is True
    assert kw["status"] == 200
    assert kw["model_alias"] == "test-model"
    # The call site passes the RAW UA through; emit.request buckets it
    # (asserted separately). It must READ the UA, not drop it.
    assert kw["caller_agent"] == "cursor/1.9.0"
    assert kw["prompt_tokens"] == 11
    assert kw["completion_tokens"] == 7
    assert kw["tool_call_used"] is False
    assert isinstance(kw["ttft_ms"], (int, float))
    assert isinstance(kw["tps"], (int, float))

    # TTFT must be measured at the FIRST token, so it is a small fraction
    # of total latency (which includes the 0.2s post-first-token gap) —
    # NOT the total-latency value the non-streaming path would report.
    total_ms = total_s * 1000.0
    assert kw["ttft_ms"] > 0.0
    assert kw["ttft_ms"] < 0.5 * total_ms, (
        f"ttft_ms={kw['ttft_ms']:.1f} should be well under half of total "
        f"latency {total_ms:.1f}ms — it must reflect first-token timing, "
        f"not total"
    )


def test_streaming_request_event_caller_agent_absent_header():
    calls: list[dict] = []
    engine = _FakeEngine(["hi", " ", "world"])
    _drive_stream(engine, _request(), caller_agent=None, emit_calls=calls)

    assert len(calls) == 1
    # No UA → the route passes None; emit.request maps it to "unknown".
    assert calls[0]["caller_agent"] is None
    assert calls[0]["stream"] is True
