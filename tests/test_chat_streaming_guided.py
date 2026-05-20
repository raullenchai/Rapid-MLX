# SPDX-License-Identifier: Apache-2.0
"""Streaming + guided generation route contract.

Pins Gap #2 from the v0.6.60 onboarding sweep: ``stream=true`` requests
with ``response_format: json_schema`` must route through
``engine.generate_with_schema`` (constrained), NOT ``engine.stream_chat``
(unconstrained). Pre-fix, the stream branch of
``_create_chat_completion_impl`` ignored ``supports_guided_generation``
entirely and the model would emit unconstrained tokens (e.g. a
``\\`\\`\\`json ... \\`\\`\\`\\`` markdown fence) defeating the user's intent.

Two contract tests:

1. **Success path** — guided streaming is used, fallback engine.stream_chat
   is not called, the synthesized SSE stream carries the constrained text.

2. **Fallback path** — if ``generate_with_schema`` raises, the helper
   falls back to ``engine.stream_chat`` so request liveness is preserved
   (clients in strict-mode use cases should validate themselves; this
   matches the non-streaming fallback semantics).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.chat import router as chat_router


class _GuidedEngine:
    """Mock engine that supports guided generation.

    Records every call to ``generate_with_schema`` and ``stream_chat``
    so tests can assert which path the route dispatched to.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = True
    tokenizer = None

    def __init__(
        self, *, guided_text: str = '{"k": "v"}', raise_in_guided: bool = False
    ):
        self._guided_text = guided_text
        self._raise = raise_in_guided
        self.guided_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        # Stream branch validates the template eagerly; return a no-op
        # string so that pre-flight passes without exercising a real
        # chat-template engine.
        return "PROMPT"

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append(
            {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
        )
        if self._raise:
            raise RuntimeError("simulated outlines failure")
        return GenerationOutput(
            text=self._guided_text,
            new_text=self._guided_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        """Unconstrained fallback path: emit a single text delta."""
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        text = "FALLBACK"
        yield GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_client(engine: _GuidedEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _parse_sse_events(text: str) -> tuple[list[dict], bool]:
    """Return ``(parsed_events, saw_done)``.

    ``parsed_events`` excludes the ``[DONE]`` sentinel.
    """
    events: list[dict] = []
    saw_done = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events, saw_done


_SCHEMA = {
    "type": "object",
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "qty": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["name", "qty"],
            "additionalProperties": False,
        }
    },
    "properties": {
        "label": {"type": "string", "enum": ["red", "green", "blue"]},
        "items": {
            "type": "array",
            "items": {"$ref": "#/$defs/Item"},
            "minItems": 1,
        },
    },
    "required": ["label", "items"],
    "additionalProperties": False,
}


_GUIDED_OUTPUT = json.dumps({"label": "red", "items": [{"name": "alpha", "qty": 2}]})


def test_streaming_json_schema_routes_through_guided_generation():
    """stream=true + json_schema must call generate_with_schema, NOT stream_chat.

    The bug class this gates: a refactor that re-wires the stream branch
    to ``engine.stream_chat`` without consulting ``supports_guided_generation``
    would silently downgrade strict-mode requests to unconstrained tokens
    — invisible in unit smoke (small schemas the model would emit anyway)
    but catastrophic for adversarial / complex schemas.
    """
    engine = _GuidedEngine(guided_text=_GUIDED_OUTPUT)
    client = _make_client(engine)

    payload = {
        "model": "test-model",
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "pick a color"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": True},
        },
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")

    # Constraint dispatch: guided path called exactly once, stream_chat
    # never invoked. This is the load-bearing assertion for Gap #2.
    assert len(engine.guided_calls) == 1
    assert engine.stream_calls == []

    # The route must hand the RAW schema dict to generate_with_schema —
    # not the strict outer ``response_format`` wrapper. Hand-off through
    # the wrapper would silently re-introduce the schema-projection bug
    # that PR #419 fixed.
    assert engine.guided_calls[0]["json_schema"] == _SCHEMA

    events, saw_done = _parse_sse_events(resp.text)
    assert saw_done, "streaming response must terminate with [DONE]"

    # Reassemble the content from delta chunks; it must equal the
    # constrained text the engine returned.
    content_parts: list[str] = []
    saw_role = False
    saw_finish = False
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {}) or {}
            if delta.get("role") == "assistant":
                saw_role = True
            if "content" in delta and delta["content"]:
                content_parts.append(delta["content"])
            if choice.get("finish_reason"):
                saw_finish = True

    assert saw_role, "first SSE chunk must announce assistant role"
    assert saw_finish, "stream must emit a finish_reason chunk"
    assert "".join(content_parts) == _GUIDED_OUTPUT


def test_streaming_guided_falls_back_to_unconstrained_on_engine_failure():
    """If generate_with_schema raises, the helper must fall back to
    stream_chat so the request still returns a response.

    Fallback rationale: a failure in outlines (import error at runtime,
    grammar compilation error on a pathological schema, etc.) should
    degrade to unconstrained generation rather than 500. Strict-mode
    clients can validate the response themselves; defensive servers
    log the failure with full traceback (via logger.exception in
    GuidedGenerator.generate_json) so the regression surfaces in ops
    visibility — see knowledge/sop_gap_guided_schema_passthrough.md.
    """
    engine = _GuidedEngine(raise_in_guided=True)
    client = _make_client(engine)

    payload = {
        "model": "test-model",
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "pick a color"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Pick", "schema": _SCHEMA, "strict": True},
        },
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200, resp.text

    # Guided path was attempted and raised; fallback unconstrained path
    # was exercised. Both calls must be recorded.
    assert len(engine.guided_calls) == 1
    assert len(engine.stream_calls) == 1

    _, saw_done = _parse_sse_events(resp.text)
    assert saw_done, "fallback streaming response must still terminate with [DONE]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
