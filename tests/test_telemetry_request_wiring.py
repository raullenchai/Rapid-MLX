# SPDX-License-Identifier: Apache-2.0
"""Wiring test: a completed non-streaming chat completion emits a ``request``
telemetry event carrying the inbound User-Agent (for ``caller_agent``) plus
the response's perf metrics.

Drives ``_create_chat_completion_impl`` with a fake engine (no model load,
no real generation), so it runs anywhere and intercepts ``emit.request`` to
assert the call-site contract. The bucketing of the raw UA and the sampling
gate are covered in test_telemetry_redact.py / test_telemetry_emit.py — here
we only assert the route reads the UA and threads the right fields through.
Harness mirrors tests/test_max_tokens_resolver.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _RawRequest:
    def __init__(self, user_agent=None):
        # A plain dict whose ``.get`` matches how the route reads the header
        # (``raw_request.headers.get("user-agent")``).
        self.headers = {} if user_agent is None else {"user-agent": user_agent}

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


class _FakeChatEngine:
    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "test-model"
    tokenizer = SimpleNamespace(encode=lambda _text: [1])

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="hello there",
            finish_reason="stop",
            prompt_tokens=12,
            completion_tokens=8,
        )


async def _await_direct(coro, *_a, **_k):
    return await coro


def _patch_route(monkeypatch, engine, emit_calls):
    from vllm_mlx.routes import chat
    from vllm_mlx.telemetry import emit

    monkeypatch.setattr(emit, "request", lambda **kw: emit_calls.append(kw))
    monkeypatch.setattr(chat, "_resolve_max_tokens", lambda *a, **k: 64)
    monkeypatch.setattr(chat, "get_engine", lambda *a, **k: engine)
    monkeypatch.setattr(chat, "_validate_model_name", lambda *a, **k: None)
    monkeypatch.setattr(chat, "_check_admission_or_503", lambda *a, **k: None)
    monkeypatch.setattr(
        chat, "_release_admission_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "_wait_with_disconnect", _await_direct)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "enforce_context_length_for_messages", lambda *a, **k: 1)


def _request(model="test-model"):
    from vllm_mlx.api.models import ChatCompletionRequest

    return ChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=None,
        stream=False,
    )


@pytest.mark.asyncio
async def test_nonstreaming_completion_emits_request_event(monkeypatch):
    from vllm_mlx.routes import chat

    calls: list[dict] = []
    engine = _FakeChatEngine()
    _patch_route(monkeypatch, engine, calls)

    await chat._create_chat_completion_impl(
        _request(),
        _RawRequest(user_agent="claude-cli/1.4.2"),
        engine,
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert len(calls) == 1, f"expected exactly one request event, got {calls}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/chat/completions"
    assert kw["stream"] is False
    assert kw["status"] == 200
    assert kw["model_alias"] == "test-model"
    # The call site passes the RAW UA through; emit.request buckets it
    # (asserted separately). It must READ the header, not drop it.
    assert kw["caller_agent"] == "claude-cli/1.4.2"
    assert kw["prompt_tokens"] == 12
    assert kw["completion_tokens"] == 8
    assert kw["tool_call_used"] is False
    assert isinstance(kw["ttft_ms"], (int, float))
    assert isinstance(kw["tps"], (int, float))


@pytest.mark.asyncio
async def test_request_event_caller_agent_absent_header(monkeypatch):
    from vllm_mlx.routes import chat

    calls: list[dict] = []
    engine = _FakeChatEngine()
    _patch_route(monkeypatch, engine, calls)

    await chat._create_chat_completion_impl(
        _request(),
        _RawRequest(user_agent=None),
        engine,
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert len(calls) == 1
    # No UA header → the route passes None; emit.request maps it to "unknown".
    assert calls[0]["caller_agent"] is None
