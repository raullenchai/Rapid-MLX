# SPDX-License-Identifier: Apache-2.0
"""F-220: ``_validate_tool_call_params`` is wired into ``/v1/messages``.

PR #736 (F-141 scoped fix) added JSON-schema enforcement on the model's
emitted ``tool_calls[].function.arguments`` for the OpenAI
``/v1/chat/completions`` route — enum / type / minimum / maximum /
minLength / maxLength violations turned into ``HTTPException(400)``.
The Anthropic-flavored ``/v1/messages`` route bypassed the same
validator, so the identical bad payload that 400-ed on
``/v1/chat/completions`` came back as a 200 ``tool_use`` block carrying
schema-violating arguments. This is the regression guard.

Mirrors ``tests/test_tool_param_enforcement.py`` for the chat route
but exercises the actual route handler so the cross-route adapter
(``api/anthropic_adapter._convert_tool``) is included in the path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.routes.anthropic import router


class _StubTokenizer:
    chat_template = ""


class _ToolCallEngine:
    """Stub engine that returns a pre-canned tool_calls structured payload.

    The Anthropic non-stream branch reads ``output.tool_calls`` first
    via ``getattr(output, "tool_calls", None)`` and feeds it into
    ``_parse_tool_calls_with_parser`` as ``structured_tool_calls``. By
    handing back a single tool_call whose ``arguments`` violate the
    declared enum, we exercise the F-220 validator inline.
    """

    preserve_native_tool_format = False
    tokenizer = _StubTokenizer()

    def __init__(self, tool_calls_payload: list[dict]):
        self._tool_calls = tool_calls_payload

    async def chat(self, messages, **kwargs):  # noqa: ARG002
        return SimpleNamespace(
            text="",
            raw_text="",
            tool_calls=self._tool_calls,
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="tool_calls",
            reasoning_text="",
        )

    async def stream_chat(self, messages, **kwargs):  # noqa: ARG002
        # Streaming variant: emit a single tool_call as a structured
        # payload on the final delta. Mirrors how the real engine
        # routers (HarmonyStreamingRouter) surface tool_calls.
        yield SimpleNamespace(
            new_text="",
            prompt_tokens=10,
            completion_tokens=1,
            tool_calls=self._tool_calls,
        )


def _make_client(engine: _ToolCallEngine) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = True
    cfg.reasoning_parser_name = None
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


_PAINT_TOOL_SCHEMA = {
    "name": "paint",
    "input_schema": {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "blue"]},
        },
        "required": ["color"],
    },
}


def _enum_violation_payload() -> list[dict]:
    # Structured-tool-call payload as produced by HarmonyStreamingRouter:
    # flat ``{name, arguments}`` dicts, NOT the wrapped OpenAI shape. See
    # ``_parse_tool_calls_with_parser`` in ``service/helpers.py``.
    return [{"name": "paint", "arguments": json.dumps({"color": "purple"})}]


def _enum_ok_payload() -> list[dict]:
    return [{"name": "paint", "arguments": json.dumps({"color": "red"})}]


def test_messages_tool_enum_violation_returns_400():
    """F-220: enum violation on /v1/messages is enforced as HTTP 400.

    Pre-fix: 200 ``tool_use`` block with ``input.color == "purple"``.
    Post-fix: 400 ``invalid_request_error`` with the canonical schema
    violation message mirroring ``/v1/chat/completions``.
    """
    engine = _ToolCallEngine(_enum_violation_payload())
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "paint purple"}],
            "tools": [_PAINT_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "paint"},
        },
    )

    assert response.status_code == 400, response.text
    body = response.json()
    # The unit-test FastAPI app uses the default HTTPException renderer
    # (``{"detail": "..."}``); the live server wraps it via the
    # global exception handlers into ``{"error": {...}}``. Accept either
    # representation here so the test stays agnostic to which layer of
    # the production stack wraps the error envelope.
    message = body.get("detail") or body.get("error", {}).get("message", "")
    assert "violates declared schema" in message, body
    assert "purple" in message
    assert "['red', 'blue']" in message


def test_messages_tool_enum_ok_returns_200_with_tool_use():
    """Negative control: a valid enum value still emits ``tool_use`` 200."""
    engine = _ToolCallEngine(_enum_ok_payload())
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "paint red"}],
            "tools": [_PAINT_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "paint"},
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    tool_uses = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_uses[0]["name"] == "paint"
    assert tool_uses[0]["input"] == {"color": "red"}
    assert body["stop_reason"] == "tool_use"


def test_messages_stream_tool_enum_violation_emits_error_event():
    """Streaming: headers are already sent so we cannot return 400 inline.

    The fix surfaces the validation failure as an Anthropic
    ``event: error`` SSE event with ``invalid_request_error`` and
    suppresses the would-be ``tool_use`` blocks. The final
    ``message_delta`` still fires with ``stop_reason="end_turn"`` so
    well-behaved clients see a clean termination.
    """
    engine = _ToolCallEngine(_enum_violation_payload())
    client = _make_client(engine)

    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "paint purple"}],
            "tools": [_PAINT_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "paint"},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    raw = response.text
    assert "event: error" in raw, raw
    # The error event must carry the canonical schema-violation message
    # so streaming clients see the SAME diagnostic the non-stream path
    # would have returned as a 400 body.
    assert "violates declared schema" in raw
    assert "purple" in raw
    assert "invalid_request_error" in raw

    # No tool_use content_block_start should have been emitted — the
    # violating call is dropped, not silently shipped as garbage.
    assert '"type": "tool_use"' not in raw
    # message_delta still fires with end_turn so the SSE protocol closes
    # cleanly (no half-open stream).
    assert '"stop_reason": "end_turn"' in raw
