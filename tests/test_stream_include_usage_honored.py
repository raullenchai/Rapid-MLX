# SPDX-License-Identifier: Apache-2.0
"""D-SSE-USAGE regression matrix: ``stream_options.include_usage`` is
honored at the SSE chunk-emit site across every OpenAI-compatible
streaming route.

Pre-v0.8.2 the SSE final delta on both ``/v1/chat/completions`` and
``/v1/completions`` carried a populated ``usage`` block regardless of
what the caller passed in ``stream_options``. The OpenAI streaming
spec says ``usage`` is opt-in:

    "If set, an additional chunk will be streamed before the data:
    [DONE] message. The usage field on this chunk shows the token
    usage statistics for the entire request, and the choices field
    will always be an empty array."

LangChain, AI-SDK, and the vercel-ai-stream parser double-count token
totals when ``usage`` appears on chunks the spec does not allow — the
parser treats the finish-chunk total as canonical AND adds the
dedicated trailing chunk's total on top of it.

This file pins the contract at the route-level chunk emitter (not in
each model parser) so the fix is parser-independent and survives any
future model-family addition. 6 confirmed-affected parser families
(qwen3-reasoning, qwen3.5-4b, llama3-1b, llama3-3b, gemma3-1b,
glm4.7) are all gated through the same emitter — the unit tests
below exercise the chunk-build site with mock engines, but the fix
applies uniformly to every parser family because the gate sits BELOW
the parser layer.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.routes.completions import router as completions_router

# ---------------------------------------------------------------------------
# Mock engines
# ---------------------------------------------------------------------------


class _PlainChatEngine:
    """Mock streaming chat engine. Emits N text deltas; the final
    ``GenerationOutput`` carries ``finished=True`` and a finish_reason.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, deltas: list[str] | None = None) -> None:
        self._deltas = deltas or ["Hello", " world", "."]

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for i, delta in enumerate(self._deltas):
            accumulated += delta
            is_last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                channel=None,
            )


class _PlainCompletionsEngine:
    """Mock streaming legacy-completions engine — parallels
    ``_PlainChatEngine`` but exposes ``stream_generate`` to match the
    route's expected signature.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, deltas: list[str] | None = None) -> None:
        self._deltas = deltas or ["foo", "bar", "baz"]

    async def stream_generate(self, prompt, **kwargs):
        accumulated = ""
        for i, delta in enumerate(self._deltas):
            accumulated += delta
            is_last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=3,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason="stop" if is_last else None,
                channel=None,
            )


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


def _make_chat_client(engine: _PlainChatEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _make_completions_client(engine: _PlainCompletionsEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(completions_router)
    return TestClient(app)


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE ``data:`` lines, excluding the ``[DONE]`` sentinel.

    Surfaces ``json.JSONDecodeError`` rather than silently swallowing —
    a regression that corrupts an intermediate chunk should fail the
    test, not be hidden by the trailing valid chunk.
    """
    events: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def _finish_chunks(events: list[dict]) -> list[dict]:
    """Chunks whose first ``choice`` carries a ``finish_reason``."""
    out: list[dict] = []
    for e in events:
        for c in e.get("choices", []) or []:
            if c.get("finish_reason"):
                out.append(e)
                break
    return out


def _dedicated_usage_chunks(events: list[dict]) -> list[dict]:
    """Chunks with empty ``choices`` and a populated ``usage`` — the
    OpenAI-spec dedicated trailing usage chunk.
    """
    return [e for e in events if not e.get("choices") and e.get("usage")]


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


def test_chat_stream_omits_usage_when_include_usage_false():
    """``stream_options.include_usage=false`` → the ``usage`` KEY must
    be absent from every chunk (not just present-but-null). Codex
    review caught that a truthiness check would let a regression
    re-serialize ``"usage": null`` and still pass — explicit key
    absence is the spec-compliant gate. Pre-fix the finish chunk
    carried a populated ``usage`` block here, double-counting on
    aggregating clients.
    """
    client = _make_chat_client(_PlainChatEngine())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"include_usage": False},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    chunks_with_usage_key = [e for e in events if "usage" in e]
    assert chunks_with_usage_key == [], (
        f"include_usage=false MUST omit the usage KEY from every "
        f"chunk (not just emit null); got {len(chunks_with_usage_key)} "
        f"chunk(s) with the key: {chunks_with_usage_key!r}"
    )


def test_chat_stream_omits_usage_when_stream_options_absent():
    """``stream_options`` omitted entirely → same contract as
    ``include_usage=false`` (the OpenAI default). Pre-fix this was the
    common bug surface — most clients omit ``stream_options`` and got
    a populated ``usage`` on the finish chunk anyway. Asserts KEY
    absence (not truthiness) so a serializer regression to
    ``"usage": null`` would still trip the gate.
    """
    client = _make_chat_client(_PlainChatEngine())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    chunks_with_usage_key = [e for e in events if "usage" in e]
    assert chunks_with_usage_key == [], (
        f"omitted stream_options MUST omit the usage KEY from every "
        f"chunk (not just emit null); got {len(chunks_with_usage_key)} "
        f"chunk(s) with the key: {chunks_with_usage_key!r}"
    )


def test_chat_stream_emits_dedicated_usage_chunk_when_include_usage_true():
    """``stream_options.include_usage=true`` → exactly one dedicated
    trailing chunk with empty ``choices`` and populated ``usage``. The
    finish chunk carries ``usage=null`` (serialized away by
    ``exclude_none=True``) so aggregating clients DO NOT double-count.
    """
    client = _make_chat_client(_PlainChatEngine())
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)

    finish = _finish_chunks(events)
    dedicated = _dedicated_usage_chunks(events)
    assert len(finish) == 1, f"expected exactly one finish chunk; got {len(finish)}"
    assert finish[0].get("usage") is None, (
        "finish chunk MUST NOT carry usage when include_usage=true "
        "(double-emission breaks aggregating clients)"
    )
    assert len(dedicated) == 1, (
        f"expected exactly one dedicated usage chunk; got {len(dedicated)}"
    )
    usage = dedicated[0]["usage"]
    assert usage["prompt_tokens"] >= 1
    assert usage["completion_tokens"] >= 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


def test_completions_stream_omits_usage_when_include_usage_false():
    """Sibling of the chat test: legacy ``/v1/completions`` must also
    honor ``stream_options.include_usage=false`` and omit the ``usage``
    KEY entirely from every SSE chunk. Pre-fix the schema didn't even
    declare ``stream_options`` so the field was silently dropped —
    the finish-chunk-build unconditionally attached usage. Asserts
    KEY absence (not truthiness) so a regression to ``"usage": null``
    still trips the gate (codex review finding).
    """
    client = _make_completions_client(_PlainCompletionsEngine())
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "stream": True,
            "max_tokens": 16,
            "stream_options": {"include_usage": False},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    chunks_with_usage_key = [e for e in events if "usage" in e]
    assert chunks_with_usage_key == [], (
        f"completions include_usage=false MUST omit the usage KEY from "
        f"every chunk (not just emit null); got "
        f"{len(chunks_with_usage_key)} chunk(s) with the key: "
        f"{chunks_with_usage_key!r}"
    )


def test_completions_stream_omits_usage_when_stream_options_absent():
    """Most common shape: bare ``/v1/completions`` streaming request
    with no ``stream_options``. The ``usage`` KEY must be absent from
    every chunk per OpenAI spec — asserts key absence (not truthiness)
    so a regression to ``"usage": null`` still trips the gate
    (codex review finding).
    """
    client = _make_completions_client(_PlainCompletionsEngine())
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "stream": True,
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    chunks_with_usage_key = [e for e in events if "usage" in e]
    assert chunks_with_usage_key == [], (
        f"omitted stream_options on completions MUST omit the usage "
        f"KEY from every chunk (not just emit null); got "
        f"{len(chunks_with_usage_key)} chunk(s) with the key: "
        f"{chunks_with_usage_key!r}"
    )


def test_completions_stream_emits_dedicated_usage_chunk_when_include_usage_true():
    """``/v1/completions`` with ``include_usage=true`` emits exactly
    one dedicated trailing chunk shaped like the OpenAI streaming
    spec: empty ``choices``, populated ``usage``. Brings parity with
    the chat-completions route.
    """
    client = _make_completions_client(_PlainCompletionsEngine())
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "stream": True,
            "max_tokens": 16,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)

    # The trailing usage chunk has ``"choices": []`` and a populated
    # ``usage`` block. The earlier per-token chunks have populated
    # ``choices`` and NO usage.
    dedicated = [e for e in events if e.get("choices") == [] and e.get("usage")]
    assert len(dedicated) == 1, (
        f"expected exactly one dedicated usage chunk on completions; "
        f"got {len(dedicated)} (full event list: {events!r})"
    )
    usage = dedicated[0]["usage"]
    assert usage["prompt_tokens"] >= 1
    assert usage["completion_tokens"] >= 1
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # The earlier per-token chunks MUST NOT carry usage.
    per_token = [e for e in events if e.get("choices") and e.get("usage")]
    assert per_token == [], (
        f"per-token chunks MUST NOT carry usage; got {len(per_token)} "
        f"that did: {per_token!r}"
    )


# ---------------------------------------------------------------------------
# Spec edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "test-model",
                "stream": True,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
        (
            "/v1/completions",
            {
                "model": "test-model",
                "prompt": "hi",
                "stream": True,
                "max_tokens": 16,
            },
        ),
    ],
)
def test_no_usage_in_per_token_chunks_when_include_usage_true(endpoint, body):
    """OpenAI spec: per-token chunks MAY carry ``"usage": null`` (or
    omit the field entirely — ``exclude_none=True`` makes the wire
    representation identical). They MUST NOT carry a populated usage
    block. Holds across both endpoints.
    """
    body = dict(body, stream_options={"include_usage": True})
    if endpoint == "/v1/chat/completions":
        client = _make_chat_client(_PlainChatEngine())
    else:
        client = _make_completions_client(_PlainCompletionsEngine())
    resp = client.post(endpoint, json=body)
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    per_token_with_usage = [
        e for e in events if e.get("choices") and e.get("usage") is not None
    ]
    assert per_token_with_usage == [], (
        f"{endpoint}: per-token chunks must not carry a populated "
        f"usage block when include_usage=true; got {len(per_token_with_usage)}"
    )
