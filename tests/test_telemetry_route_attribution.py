# SPDX-License-Identifier: Apache-2.0
"""Task C (instrumentation-release) — END-TO-END caller attribution.

These tests close the gap codex flagged on the first pass of #2436: the
earlier telemetry tests called ``emit.request`` / ``normalize_caller_agent``
directly, so they stayed green even if the route wiring was deleted. These
drive the ACTUAL routes (``/v1/messages`` and ``/v1/completions``) through
``TestClient`` with telemetry opted-in + stubbed, and assert the emitted
``request`` event — proving the User-Agent reaches the payload bucketed to
the correct caller label, end to end.

This is the attribution fix the instrument-release drop depends on for the
agent-strategy decision: claude-code traffic over the Anthropic surface must
land in ``claude-code`` (not ``other``), and openai-python ride-through must
land in ``openai-python``.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ------------------------------------------------------------------ fixtures
# Mirror the opted_in + stub_queue pattern from test_telemetry_emit.py so
# telemetry is on (consent + sampling=1) and every emitted payload is
# captured in-memory instead of hitting the real queue sink.


@pytest.fixture
def telemetry_on(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)

    import vllm_mlx.telemetry.emit as emit
    import vllm_mlx.telemetry.state as state

    importlib.reload(state)
    importlib.reload(emit)
    emit._reset_for_tests()

    from vllm_mlx.telemetry.state import record_consent

    record_consent(True, rapid_mlx_version="0.0.0+test")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_REQUEST_SAMPLE", "1")
    return emit


@pytest.fixture
def captured(monkeypatch):
    """Capture every ``emit.request`` payload into a list."""
    from vllm_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    return captured


def _request_events(captured) -> list[dict]:
    """Pull ``request``-type payloads ('request' is the schema descriminator
    key inside each captured envelope)."""
    events = []
    for payload in captured:
        req = payload.get("request")
        if req is not None:
            events.append(req)
    return events


async def _sleep(seconds: float) -> None:
    """Async sleep helper so fake engines can inject a real post-first-token
    gap without blocking the event loop (mirrors the fake engine in
    ``test_telemetry_streaming_request_wiring.py``).

    TTFT rigor (codex r3-NIT#3 deferral): we use a REAL clock + an artificial
    post-first-token gap and assert ``ttft_ms < 0.5 * total``, exactly the
    approach the repo's accepted ``test_telemetry_streaming_request_wiring``
    uses — rather than patching the global ``time.perf_counter`` (which leaks
    into the HTTP client's own elapsed-time arithmetic and is flaky). Under
    pathological CI slowness where pre-token processing could exceed the gap,
    the assertion would trip on a genuinely-regressed (total-latency) TTFT —
    the exact regression the test exists to catch."""
    import asyncio

    await asyncio.sleep(seconds)


def _capture_request(monkeypatch, captured):
    """Capture every ``emit.request`` KWARG (not the enqueued payload) so a
    test can assert on raw values (e.g. exact ``ttft_ms``) that are otherwise
    bucketed in the wire payload. Mirrors the emit.request-capture pattern in
    ``test_telemetry_streaming_request_wiring.py``."""
    from vllm_mlx.telemetry import emit

    def _wrapper(**kw):
        captured.append(kw)

    monkeypatch.setattr(emit, "request", _wrapper)
    return captured


# ---------------------------------------------------------------- anthropic


class _AnthropicEngine:
    """Combined non-stream ``chat`` + stream ``stream_chat`` fake engine.

    Mirrors the ``_StreamingEngine`` from ``test_anthropic_route_streaming``
    and adds a non-streaming ``chat`` that returns a token-bearing output so
    the same harness drives both the ``/v1/messages`` branches.
    """

    preserve_native_tool_format = False
    tokenizer = SimpleNamespace(
        chat_template=None,
        apply_chat_template=lambda *a, **k: "templated",
        decode=lambda *a, **k: "",
        encode=lambda *a, **k: [1, 2, 3],
    )

    def __init__(self, *, prefill_delay=0.0, decode_delay=0.2):
        self.nonstream_calls = 0
        self.stream_calls = 0
        self.prefill_delay = prefill_delay
        self.decode_delay = decode_delay

    async def chat(self, messages, **kwargs):
        self.nonstream_calls += 1
        return SimpleNamespace(
            text="hello there",
            raw_text="hello there",
            prompt_tokens=9,
            completion_tokens=7,
            finish_reason="stop",
            tool_calls=None,
            matched_stop=None,
            reasoning_text=None,
            model="test-model",
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls += 1
        if self.prefill_delay:
            await _sleep(self.prefill_delay)
        for i, text in enumerate(["Hello ", "world"], start=1):
            if i == len(["Hello ", "world"]):
                await _sleep(self.decode_delay)
            yield SimpleNamespace(
                new_text=text,
                prompt_tokens=9,
                completion_tokens=i,
            )


def _anthropic_client(
    engine: _AnthropicEngine, *, reasoning_parser_name: str | None = None
) -> TestClient:
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.reasoning_parser_name = reasoning_parser_name

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_anthropic_messages_nonstream_emits_claude_code_attribution(
    telemetry_on, captured
):
    """A claude-code User-Agent on a completed non-streaming ``/v1/messages``
    request must surface a bucketed ``request`` event with
    ``caller_agent == "claude-code"``, the correct endpoint, stream=False,
    token counts and a positive TTFT. This is the structural proof codex
    asked for: it would FAIL if the route's ``emit.request`` were deleted or
    the User-Agent were not threaded through."""
    # The ``/v1/messages`` route imports ``mlx`` at module load, so this test
    # can only drive it where MLX is installed. Skip cleanly on the no-MLX
    # pr_validate CI env (where the whole anthropic route suite is MLX-gated)
    # rather than erroring at collection; it runs to completion locally.
    pytest.importorskip("mlx")
    engine = _AnthropicEngine()
    client = _anthropic_client(engine)

    resp = client.post(
        "/v1/messages",
        headers={"user-agent": "claude-code/1.0.3"},
        json={
            "model": "test-model",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _request_events(captured)
    assert engine.nonstream_calls == 1
    assert len(events) >= 1, f"no request telemetry emitted: {captured!r}"
    ev = events[-1]
    assert ev["endpoint"] == "/v1/messages"
    assert ev["stream"] is False
    assert ev["caller_agent"] == "claude-code"
    # Token counts are bucketed to a fixed allowlist (red-line: raw counts
    # are a soft fingerprint) — the 9/7 token counts land in "0-256".
    assert ev["prompt_tokens_bucket"] == "0-256"
    assert ev["completion_tokens_bucket"] == "0-256"
    assert ev["completion_empty"] is False
    # TTFT is bucketed (never a raw ms value) — just assert a valid label.
    assert ev["ttft_ms_bucket"] in (
        "<100ms",
        "100-500ms",
        "500-1500ms",
        "1.5-5s",
        ">5s",
    )
    assert ev["status"] == 200


def test_anthropic_messages_stream_emits_claude_code_attribution(
    telemetry_on, monkeypatch
):
    """Same contract on the streaming ``/v1/messages`` branch: the emit fires
    after the stream drains, caller_agent is the bucketed claude-code label,
    stream=True, and TTFT is TRUE first-token latency.

    Captures raw ``emit.request`` kwargs (before bucketing) while driving the
    route end-to-end; the fake engine injects 0.1s of prefill before the first
    token and a 0.2s decode gap after it. This makes both timing windows large
    enough to prove that TTFT and TPS use their respective clocks.
    """
    pytest.importorskip("mlx")
    import time

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)

    engine = _AnthropicEngine(prefill_delay=0.1, decode_delay=0.2)
    client = _anthropic_client(engine)

    t0 = time.perf_counter()
    resp = client.post(
        "/v1/messages",
        headers={"user-agent": "Claude-Code/2.0 (macOS) Anthropic/API"},
        json={
            "model": "test-model",
            "max_tokens": 2048,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert engine.stream_calls == 1
    assert len(calls) == 1, f"expected one request emit, got {calls!r}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/messages"
    assert kw["stream"] is True
    assert kw["status"] == 200
    assert kw["caller_agent"] == "Claude-Code/2.0 (macOS) Anthropic/API"
    assert kw["prompt_tokens"] == 9
    assert kw["completion_tokens"] == 2
    # TTFT is measured at the FIRST token, so it is a small fraction of total
    # latency (which includes the 0.2s post-first-token gap) — NOT total.
    assert kw["ttft_ms"] > 0.0
    assert kw["ttft_ms"] < 0.5 * total_ms, (
        f"ttft_ms={kw['ttft_ms']:.1f} should be well under half of total "
        f"latency {total_ms:.1f}ms — it must reflect first-token timing, not total"
    )
    total_rate = kw["completion_tokens"] / (total_ms / 1000.0)
    assert kw["tps"] > 1.25 * total_rate, (
        f"tps={kw['tps']:.1f} must use the post-first-token decode window, "
        f"not total request time ({total_rate:.1f} tok/s)"
    )


def test_anthropic_stream_ttft_ignores_parser_suppressed_reasoning(
    telemetry_on, monkeypatch
):
    """Raw reasoning held by a parser is not visible client output.

    The first engine delta is deliberately suppressed, followed by a 0.2s
    delay and a visible text delta. TTFT must include that delay; latching on
    the raw engine delta would report near-zero latency instead.
    """
    pytest.importorskip("mlx")

    class _HeldPrefixEngine(_AnthropicEngine):
        async def stream_chat(self, messages, **kwargs):
            self.stream_calls += 1
            yield SimpleNamespace(
                new_text="hidden reasoning",
                prompt_tokens=9,
                completion_tokens=1,
            )
            await _sleep(0.2)
            yield SimpleNamespace(
                new_text="visible answer",
                prompt_tokens=9,
                completion_tokens=2,
            )

    class _HeldPrefixParser:
        implicit_reasoning_until_close = False
        sanitize_when_thinking_disabled = True

        def configure_request(self, **kwargs):
            return None

        def extract_reasoning_streaming(self, previous, current, delta):
            if delta == "hidden reasoning":
                return None
            return SimpleNamespace(reasoning=None, content=delta)

        def finalize_streaming(self, *args, **kwargs):
            return None

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)
    monkeypatch.setattr(
        "vllm_mlx.reasoning.get_parser", lambda _name: _HeldPrefixParser
    )

    engine = _HeldPrefixEngine()
    client = _anthropic_client(engine, reasoning_parser_name="held-prefix")
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 2048,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 200, resp.text
    assert "visible answer" in resp.text
    assert "hidden reasoning" not in resp.text
    assert engine.stream_calls == 1
    assert len(calls) == 1
    assert calls[0]["ttft_ms"] >= 150.0, calls[0]


def test_anthropic_stream_ttft_latches_when_pinned_tool_buffer_replays(
    telemetry_on, monkeypatch
):
    """A valid named-tool stream delays TTFT until buffered output is replayed.

    The engine emits adjacent text and the pinned structured tool call in
    consecutive chunks. The route must first buffer the text while it validates
    the forced tool contract, then replay it before the tool block. Both are
    visible output, and telemetry must latch exactly once on that replay.
    """
    pytest.importorskip("mlx")

    class _PinnedToolEngine(_AnthropicEngine):
        async def stream_chat(self, messages, **kwargs):
            self.stream_calls += 1
            # Text and structured calls are separate engine chunks: the route
            # deliberately short-circuits a chunk carrying ``tool_calls``.
            yield SimpleNamespace(
                new_text="Checking weather. ",
                prompt_tokens=9,
                completion_tokens=1,
                finish_reason=None,
                tool_calls=None,
            )
            yield SimpleNamespace(
                new_text="",
                prompt_tokens=9,
                completion_tokens=3,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "arguments": '{"location":"SF"}',
                    }
                ],
                channel=None,
            )

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)
    engine = _PinnedToolEngine()
    client = _anthropic_client(engine)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "stream": True,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "weather in SF"}],
        },
    )

    assert resp.status_code == 200, resp.text
    assert "Checking weather." in resp.text
    assert '"type": "tool_use"' in resp.text
    assert resp.text.index("Checking weather.") < resp.text.index('"type": "tool_use"')
    assert engine.stream_calls == 1
    assert len(calls) == 1
    assert calls[0]["tool_call_used"] is True
    assert calls[0]["ttft_ms"] > 0.0


def test_anthropic_empty_stream_uses_total_latency_as_ttft(telemetry_on, monkeypatch):
    """A successful stream with no visible output reports total time as TTFT."""
    pytest.importorskip("mlx")

    class _EmptyEngine(_AnthropicEngine):
        async def stream_chat(self, messages, **kwargs):
            self.stream_calls += 1
            await _sleep(0.05)
            yield SimpleNamespace(
                new_text="",
                prompt_tokens=9,
                completion_tokens=0,
                finish_reason="stop",
                finished=True,
                tool_calls=None,
            )

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)
    engine = _EmptyEngine()
    client = _anthropic_client(engine)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "say nothing"}],
        },
    )

    assert resp.status_code == 200, resp.text
    assert engine.stream_calls == 1
    assert len(calls) == 1
    assert calls[0]["completion_tokens"] == 0
    assert calls[0]["ttft_ms"] >= 40.0, calls[0]
    assert calls[0]["tps"] == 0.0


# ---------------------------------------------------------------- completions


def _completions_client(monkeypatch, stream_generate=None):
    """Drive ``/v1/completions`` end-to-end with a fake engine, mirroring the
    proven harness from ``test_completions_log_redaction`` (engine + admission
    + context-length shims so the route completes without loading a model).

    ``stream_generate`` may be replaced by a caller-supplied async generator
    to control *where* the latency lands in the stream (codex r7-B#1). The
    default places the post-first-token gap AFTER the first generated chunk —
    the correct shape for proving true TTFT on a plain (non-echo) stream.
    """
    from vllm_mlx.routes import completions as completions_mod

    fake_engine = MagicMock()

    async def _finish(*_a, **_k):
        return SimpleNamespace(
            text="done",
            finish_reason="stop",
            completion_tokens=5,
            prompt_tokens=3,
            cached_tokens=0,
        )

    fake_engine.generate = AsyncMock(side_effect=_finish)

    if stream_generate is None:

        async def _stream_finish(*_a, **_k):
            # First chunk: non-empty content, not finished.
            yield SimpleNamespace(
                new_text="done",
                finished=False,
                finish_reason=None,
                completion_tokens=0,
                prompt_tokens=3,
            )
            # Post-first-token gap so total latency dwarfs true TTFT.
            await _sleep(0.2)
            # Final chunk: finished, carries the engine's final usage.
            yield SimpleNamespace(
                new_text="",
                finished=True,
                finish_reason="stop",
                completion_tokens=5,
                prompt_tokens=3,
            )

        fake_engine.stream_generate = _stream_finish
    else:
        fake_engine.stream_generate = stream_generate

    monkeypatch.setattr(completions_mod, "get_engine", lambda _name: fake_engine)
    monkeypatch.setattr(completions_mod, "_check_admission_or_503", lambda _e: None)
    monkeypatch.setattr(
        completions_mod, "_release_admission_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(
        completions_mod, "enforce_context_length_for_prompt", lambda *a, **k: None
    )
    monkeypatch.setattr(completions_mod, "_validate_model_name", lambda _m: None)
    monkeypatch.setattr(completions_mod, "_resolve_model_name", lambda m: m)
    monkeypatch.setattr(completions_mod, "_resolve_max_tokens", lambda m: m or 16)
    monkeypatch.setattr(completions_mod, "_resolve_temperature", lambda t: t)
    monkeypatch.setattr(completions_mod, "_resolve_top_p", lambda p: p)
    monkeypatch.setattr(
        completions_mod, "build_extended_sampling_kwargs", lambda _r: {}
    )

    async def _passthrough(coro, *_a, **_k):
        return await coro

    monkeypatch.setattr(completions_mod, "_wait_with_disconnect", _passthrough)

    with (
        patch("vllm_mlx.middleware.auth.verify_api_key", new=lambda *a, **k: None),
        patch("vllm_mlx.middleware.auth.check_rate_limit", new=lambda *a, **k: None),
    ):
        app = FastAPI()
        app.include_router(completions_mod.router)
        return TestClient(app)


def test_completions_nonstream_emits_openai_python_attribution(
    telemetry_on, captured, monkeypatch
):
    """A completed non-streaming ``/v1/completions`` with an openai-python
    User-Agent surfaces ``caller_agent == "openai-python"``, endpoint
    ``/v1/completions``, stream=False, and the engine's token counts."""
    client = _completions_client(monkeypatch)
    resp = client.post(
        "/v1/completions",
        headers={"user-agent": "OpenAI/Python 1.30.1"},
        json={"model": "test-model", "prompt": "hi", "max_tokens": 8},
    )
    assert resp.status_code == 200, resp.text
    events = _request_events(captured)
    assert len(events) >= 1, f"no request telemetry emitted: {captured!r}"
    ev = events[-1]
    assert ev["endpoint"] == "/v1/completions"
    assert ev["stream"] is False
    assert ev["caller_agent"] == "openai-python"
    # engine's 3 prompt + 5 completion tokens → "0-256" bucket
    assert ev["prompt_tokens_bucket"] == "0-256"
    assert ev["completion_tokens_bucket"] == "0-256"
    assert ev["ttft_ms_bucket"] in (
        "<100ms",
        "100-500ms",
        "500-1500ms",
        "1.5-5s",
        ">5s",
    )
    assert ev["status"] == 200


def test_completions_stream_emits_openai_python_attribution(telemetry_on, monkeypatch):
    """The streaming ``/v1/completions`` branch emits stream=True with TRUE
    first-token TTFT and final-usage token counts.

    Captures raw ``emit.request`` kwargs; the fake stream injects a real 0.2s
    gap after the first chunk so total latency dwarfs true TTFT. Asserting
    ``ttft_ms`` is a small fraction of total proves ``_first_token_ts`` is
    latched and used (a total-latency regression would blow past 0.5*total).
    """
    import time

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)

    async def _prefill_then_decode(*_a, **_k):
        await _sleep(0.1)
        yield SimpleNamespace(
            new_text="done",
            finished=False,
            finish_reason=None,
            completion_tokens=0,
            prompt_tokens=3,
        )
        await _sleep(0.2)
        yield SimpleNamespace(
            new_text="",
            finished=True,
            finish_reason="stop",
            completion_tokens=5,
            prompt_tokens=3,
        )

    client = _completions_client(monkeypatch, stream_generate=_prefill_then_decode)
    t0 = time.perf_counter()
    resp = client.post(
        "/v1/completions",
        headers={"user-agent": "OpenAI/Python 1.30.1"},
        json={"model": "test-model", "prompt": "hi", "max_tokens": 8, "stream": True},
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200, resp.text
    assert len(calls) == 1, f"expected one request emit, got {calls!r}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/completions"
    assert kw["stream"] is True
    assert kw["status"] == 200
    assert kw["caller_agent"] == "OpenAI/Python 1.30.1"
    assert kw["prompt_tokens"] == 3
    assert kw["completion_tokens"] == 5
    assert kw["ttft_ms"] > 0.0
    assert kw["ttft_ms"] < 0.5 * total_ms, (
        f"ttft_ms={kw['ttft_ms']:.1f} should be well under half of total "
        f"latency {total_ms:.1f}ms — it must reflect first-token timing, not total"
    )
    total_rate = kw["completion_tokens"] / (total_ms / 1000.0)
    assert kw["tps"] > 1.25 * total_rate, (
        f"tps={kw['tps']:.1f} must use the post-first-token decode window, "
        f"not total request time ({total_rate:.1f} tok/s)"
    )


def test_completions_stream_echo_latches_ttft_at_echo_yield(telemetry_on, monkeypatch):
    """In `echo=True` streaming, the echoed prompt is the client-visible first
    content, so TTFT is latched at the echo yield (completions.py:754) rather
    than at the first generated token later in the loop (line 906).

    The route renders the echo SYNCHRONOUSLY before ``stream_generate`` is even
    awaited, so we inject the model latency INSIDE the fake engine, BEFORE its
    first generated chunk: that makes the first generated chunk arrive only
    after a real 0.2s gap, while the echo yield is immediate. This is what makes
    the test DISCRIMINATING (codex r7-B#1):
      - with the echo latch:  TTFT latches at the immediate echo yield (~0ms)
      - without the echo latch: line 906 latches at the first generated chunk,
        which lands AFTER the 0.2s gap -> TTFT ~= the gap
    Asserting ``ttft_ms < 0.5 * total`` then fails precisely when the echo latch
    is deleted and the route silently falls back to generated-chunk timing.
    """
    import time

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)

    async def _echo_stream(*_a, **_k):
        # Model latency between the (immediate) echo yield and the first
        # generated chunk. Without the echo latch this makes line 906 latch
        # ~0.2s late, blowing past the 0.5*total bound below.
        await _sleep(0.2)
        yield SimpleNamespace(
            new_text="done",
            finished=False,
            finish_reason=None,
            completion_tokens=0,
            prompt_tokens=3,
        )
        yield SimpleNamespace(
            new_text="",
            finished=True,
            finish_reason="stop",
            completion_tokens=5,
            prompt_tokens=3,
        )

    client = _completions_client(monkeypatch, stream_generate=_echo_stream)
    t0 = time.perf_counter()
    resp = client.post(
        "/v1/completions",
        headers={"user-agent": "OpenAI/Python 1.30.1"},
        json={
            "model": "test-model",
            "prompt": "echo this",
            "max_tokens": 8,
            "stream": True,
            "echo": True,
        },
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200, resp.text
    assert len(calls) == 1, f"expected one request emit, got {calls!r}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/completions"
    assert kw["stream"] is True
    assert kw["caller_agent"] == "OpenAI/Python 1.30.1"
    # The echo latch fires at the immediate echo yield, so TTFT is a small
    # fraction of total latency (which includes the 0.2s pre-generated delay).
    # Deleting the latch makes line 906 latch at the generated chunk, ~0.2s
    # in -> TTFT would be ~the gap and this assertion would fail.
    assert kw["ttft_ms"] > 0.0
    assert kw["ttft_ms"] < 0.5 * total_ms, (
        f"ttft_ms={kw['ttft_ms']:.1f} should be well under half of total "
        f"latency {total_ms:.1f}ms — it must reflect the immediate echo-yield "
        f"latency, not the delayed first generated chunk"
    )
    assert kw["status"] == 200
