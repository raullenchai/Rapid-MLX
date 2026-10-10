# SPDX-License-Identifier: Apache-2.0
"""Benchmark evidence must reject incomplete or failed SSE generations."""

import asyncio
import json

import httpx
import pytest

from scripts.benchmark_hybrid_vision_text_batching import measure


def _transport(mode, fail_request, observed):
    async def handle(request):
        limit = json.loads(request.content)["max_tokens"]
        current_mode = mode if len(observed) + 1 == fail_request else "valid"
        observed.append((limit, current_mode))
        content = {"choices": [{"delta": {"content": "ok"}}]}
        finish = {"choices": [{"delta": {}, "finish_reason": "length"}]}
        usage = {"choices": [], "usage": {"completion_tokens": limit}}
        if current_mode == "wrong_finish":
            finish["choices"][0]["finish_reason"] = "stop"
        if current_mode == "wrong_tokens":
            usage["usage"]["completion_tokens"] = limit - 1
        events = []
        if current_mode != "no_content":
            events.append(content)
        if current_mode != "no_finish":
            events.append(finish)
        if current_mode != "no_usage":
            events.append(usage)
        if current_mode == "error":
            events.append({"error": {"message": "generation failed"}})
        data = "data: [DONE]\n\n" if current_mode == "early_done" else ""
        data += "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        if current_mode != "no_done":
            data += "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode())

    return httpx.MockTransport(handle)


def _install_client(monkeypatch, mode, fail_request=None):
    client = httpx.AsyncClient
    observed = []
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(
            transport=_transport(mode, fail_request, observed), **kwargs
        ),
    )
    return observed


@pytest.mark.asyncio
async def test_complete_benchmark_streams_produce_each_batch_width(monkeypatch):
    _install_client(monkeypatch, "valid")
    rows = await measure("http://benchmark.test", "test-model", tokens=16, reps=1)
    assert [r["b"] for r in rows] == [1, 2, 4]
    assert [len(r["outputs"]) for r in rows] == [1, 2, 4]
    assert all(o["tokens"] == 16 for r in rows for o in r["outputs"])
    assert all(r["tok_s"] > 0 for r in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_request", [2, 3, 5], ids=["B1", "B2", "B4"])
@pytest.mark.parametrize(
    "mode",
    [
        "no_finish",
        "no_done",
        "early_done",
        "wrong_finish",
        "no_usage",
        "wrong_tokens",
        "no_content",
        "error",
    ],
)
async def test_incomplete_or_failed_stream_cannot_be_benchmark_evidence(
    monkeypatch, mode, fail_request
):
    observed = _install_client(monkeypatch, mode, fail_request)
    with pytest.raises(RuntimeError):
        await measure("http://benchmark.test", "test-model", tokens=16, reps=1)
    assert observed[0] == (8, "valid")  # Warmup must succeed before the fault.
    assert observed[fail_request - 1] == (16, mode)


@pytest.mark.asyncio
async def test_failed_batch_drains_siblings_before_client_closes(monkeypatch):
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    cleanup_client_states = []
    calls = 0
    client_type = httpx.AsyncClient

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            sibling_started.set()
            try:
                await asyncio.Event().wait()
                yield b""
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        async def aclose(self):
            cleanup_client_states.append(client.is_closed)

    valid_transport = _transport("valid", None, [])

    async def handle(request):
        nonlocal calls
        calls += 1
        if calls <= 2:  # Successful warmup and B1.
            return await valid_transport.handle_async_request(request)
        if calls == 3:
            await sibling_started.wait()
            return httpx.Response(200, content=b'data: {"error": "failed"}\n\n')
        return httpx.Response(200, stream=WaitingStream())

    client = client_type(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="failed"):
        await asyncio.wait_for(
            measure("http://benchmark.test", "test-model", tokens=16, reps=1),
            timeout=2,
        )
    assert sibling_cancelled.is_set()
    assert cleanup_client_states == [False]
    assert client.is_closed
