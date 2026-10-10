# SPDX-License-Identifier: Apache-2.0
"""Benchmark evidence must reject incomplete or failed SSE generations."""

import json

import httpx
import pytest

from scripts.benchmark_hybrid_vision_text_batching import measure


def _transport(mode):
    async def handle(request):
        limit = json.loads(request.content)["max_tokens"]
        content = {"choices": [{"delta": {"content": "ok"}}]}
        finish = {"choices": [{"delta": {}, "finish_reason": "length"}]}
        usage = {"choices": [], "usage": {"completion_tokens": limit}}
        if mode == "wrong_finish":
            finish["choices"][0]["finish_reason"] = "stop"
        if mode == "wrong_tokens":
            usage["usage"]["completion_tokens"] = limit - 1
        events = []
        if mode != "no_content":
            events.append(content)
        if mode != "no_finish":
            events.append(finish)
        if mode != "no_usage":
            events.append(usage)
        if mode == "error":
            events.append({"error": {"message": "generation failed"}})
        data = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        if mode != "no_done":
            data += "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode())

    return httpx.MockTransport(handle)


def _install_client(monkeypatch, mode):
    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=_transport(mode), **kwargs),
    )


@pytest.mark.asyncio
async def test_complete_benchmark_streams_produce_each_batch_width(monkeypatch):
    _install_client(monkeypatch, "valid")
    rows = await measure("http://benchmark.test", "test-model", tokens=16, reps=1)
    assert [r["b"] for r in rows] == [1, 2, 4]
    assert [len(r["outputs"]) for r in rows] == [1, 2, 4]
    assert all(o["tokens"] == 16 for r in rows for o in r["outputs"])
    assert all(r["tok_s"] > 0 for r in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "no_finish",
        "no_done",
        "wrong_finish",
        "no_usage",
        "wrong_tokens",
        "no_content",
        "error",
    ],
)
async def test_incomplete_or_failed_stream_cannot_be_benchmark_evidence(
    monkeypatch, mode
):
    _install_client(monkeypatch, mode)
    with pytest.raises(RuntimeError):
        await measure("http://benchmark.test", "test-model", tokens=16, reps=1)
