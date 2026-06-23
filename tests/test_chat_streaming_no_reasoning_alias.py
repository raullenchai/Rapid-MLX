# SPDX-License-Identifier: Apache-2.0
"""r10-B R10-C2: chat-completions streaming emits ``reasoning_content`` only.

Sven's r10-R1 SSE evidence (2026-06-23) showed every reasoning delta
on ``POST /v1/chat/completions`` carrying BOTH ``delta.reasoning_content``
AND ``delta.reasoning`` with byte-identical values. That duplicate was
the root cause of R9-CRIT3 — ``openai-agents``'s ``Runner.run_streamed``
walks both keys when reassembling a turn, so every text_delta surfaced
to the SDK consumer twice. Mira r2 (R10-M2) independently reproduced
the same dup in test A.

The duplicate alias was introduced in r7-A R7-H2 as a one-release
deprecation window (the comment then-claimed ``reasoning`` was the
"OpenAI spec name"). It is not: the OpenAI o1-style streaming spec
uses ``reasoning_content`` only on chat-completion deltas. r10-B
closes the window and removes the alias on:

  1. The chat route's ``_fast_sse_chunk`` fast path (hot path —
     every per-token reasoning delta).
  2. ``ChatCompletionChunkDelta._serialize_chunk_delta`` (pydantic
     fallback when the route builds a chunk through the model).
  3. ``AssistantMessage._serialize_assistant_message`` /
     ``model_dump`` (non-streaming response, kept consistent so
     stream + non-stream wire shapes match).

This file pins the inverted invariant end-to-end against the real
chat route, using a mock engine that surfaces a ``reasoning``-channel
delta. Without the fix the SSE bytes show ``"reasoning":...`` after
``"reasoning_content":...`` on every reasoning-bearing chunk; with
the fix it appears nowhere.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _ReasoningChannelEngine:
    """Mock engine yielding two ``reasoning``-channel deltas, then a
    ``content``-channel delta, then a finish marker. The chat route
    routes the first two through the streaming reasoning path
    (``_fast_sse_chunk(text, "reasoning_content")``) and the third
    through the content path.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        yield GenerationOutput(
            text="Let me ",
            new_text="Let me ",
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="Let me think.",
            new_text="think.",
            prompt_tokens=4,
            completion_tokens=2,
            finished=False,
            finish_reason=None,
            channel="reasoning",
        )
        yield GenerationOutput(
            text="Let me think.Hi!",
            new_text="Hi!",
            prompt_tokens=4,
            completion_tokens=3,
            finished=True,
            finish_reason="stop",
            channel="content",
        )


def _make_client(engine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    # The mock engine already tags channels explicitly; no post-parser
    # needs to run.
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_parser = None
    cfg.no_thinking = False

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


def _parse_sse_deltas(body: str) -> list[dict]:
    deltas: list[dict] = []
    for raw_event in body.split("\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []) or []:
                if "delta" in choice:
                    deltas.append(choice["delta"])
    return deltas


def test_chat_streaming_emits_reasoning_content_only_no_reasoning_alias():
    """R10-C2: every streaming delta on ``/v1/chat/completions`` MUST
    surface reasoning under ``reasoning_content`` only — never under
    a duplicate ``reasoning`` alias. The alias double-counted every
    text_delta for any consumer that walked both keys (R9-CRIT3 root
    cause).
    """
    engine = _ReasoningChannelEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "think briefly then say hi"}
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    deltas = _parse_sse_deltas(resp.text)
    assert deltas, "expected at least one streaming delta"

    reasoning_deltas = [d for d in deltas if "reasoning_content" in d]
    assert reasoning_deltas, (
        "engine yielded two reasoning-channel deltas; expected at least "
        "one streaming delta to carry reasoning_content"
    )

    # R10-C2 invariant: NO delta carries the non-spec ``reasoning`` key.
    aliased = [d for d in deltas if "reasoning" in d]
    assert aliased == [], (
        f"R10-C2 / R9-CRIT3: no streaming delta may carry the non-spec "
        f"'reasoning' alias (use 'reasoning_content' only); offenders:\n"
        + "\n".join(json.dumps(d) for d in aliased)
    )

    # Sanity: reasoning text accumulated correctly under reasoning_content.
    joined = "".join(d.get("reasoning_content", "") for d in reasoning_deltas)
    assert "Let me" in joined and "think." in joined, (
        f"reasoning_content accumulation broke; got {joined!r}"
    )


def test_chat_streaming_sse_bytes_have_no_reasoning_alias_substring():
    """Belt-and-braces: the raw on-the-wire bytes must not contain the
    pattern ``"reasoning_content":...,"reasoning":...`` that r7-A R7-H2
    used to emit. The Pydantic-level check above can pass even if a
    future refactor wrote a parallel writer that escaped the model
    serializer (e.g. a route helper that builds JSON by hand). Match
    on the byte-level substring so any reintroduction trips.
    """
    engine = _ReasoningChannelEngine()
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "think briefly then say hi"}
            ],
        },
    )
    assert resp.status_code == 200, resp.text

    body = resp.text
    # The reasoning_content key still appears (positive control).
    assert '"reasoning_content":' in body, (
        "expected reasoning_content key on the wire; engine yielded "
        "reasoning-channel deltas"
    )
    # The dup alias key must not appear anywhere in the SSE body.
    assert '"reasoning":' not in body, (
        "R10-C2 / R9-CRIT3: the non-spec 'reasoning' alias key must not "
        "appear in the chat-completions SSE body — it caused openai-agents "
        "Runner.run_streamed to double-count every reasoning text_delta. "
        f"Body:\n{body}"
    )


def test_assistant_message_non_stream_emits_reasoning_content_only():
    """Stream/non-stream parity check at the pydantic model layer:
    the non-streaming ``AssistantMessage`` must mirror the streaming
    chunk's R10-C2 contract — ``reasoning_content`` only, no
    ``reasoning`` alias. Without parity a client that switches between
    stream and non-stream would still hit the dup on the non-stream
    surface.
    """
    from vllm_mlx.api.models import AssistantMessage

    msg = AssistantMessage(
        role="assistant",
        content="Hi!",
        reasoning_content="Let me think.",
    )
    data = json.loads(msg.model_dump_json(exclude_none=True))
    assert data["reasoning_content"] == "Let me think."
    assert "reasoning" not in data, (
        f"R10-C2: non-stream AssistantMessage must emit reasoning_content "
        f"only; saw alias 'reasoning' in {data!r}"
    )
