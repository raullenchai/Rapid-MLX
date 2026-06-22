# SPDX-License-Identifier: Apache-2.0
"""Regression tests for max_tokens cap resolution.

These tests do not load a model. They pin the shared resolver used by
chat, responses, Anthropic, and completions routes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _thinking_cfg(*, default_max_tokens_is_explicit: bool):
    from vllm_mlx.config import reset_config

    cfg = reset_config()
    cfg.default_max_tokens = 128
    cfg.default_max_tokens_is_explicit = default_max_tokens_is_explicit
    cfg.thinking_token_budget = 2048
    cfg.reasoning_parser_name = "qwen3"
    return cfg


def test_request_explicit_max_tokens_is_hard_cap_for_thinking_model():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(64, enable_thinking=True) == 64


def test_operator_explicit_default_max_tokens_is_hard_cap_for_thinking_model():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=True)

    assert _resolve_max_tokens(None, enable_thinking=True) == 128


def test_implicit_default_gets_thinking_headroom_when_request_omits_max_tokens():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(None, enable_thinking=True) == 128 + 2048


def test_non_thinking_request_does_not_get_implicit_headroom():
    from vllm_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(None, enable_thinking=False) == 128


class _RawRequest:
    def __init__(self, body: dict | None = None):
        self._body = body or {}

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


class _CaptureChatEngine:
    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "test-model"
    tokenizer = SimpleNamespace(encode=lambda _text: [1])

    def __init__(self):
        self.captured_max_tokens = None

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.captured_max_tokens = kwargs.get("max_tokens")
        return GenerationOutput(
            text="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


class _CaptureCompletionEngine:
    supports_completion_logprobs = True
    tokenizer = SimpleNamespace(encode=lambda text: [1], decode=lambda ids: "x")

    def __init__(self):
        self.captured_max_tokens = None

    async def generate(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.captured_max_tokens = kwargs.get("max_tokens")
        return GenerationOutput(
            text="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


async def _await_direct(coro, *_args, **_kwargs):
    return await coro


def _patch_common_route_deps(monkeypatch, module, engine):
    monkeypatch.setattr(module, "_resolve_max_tokens", lambda *_args, **_kw: 777)
    monkeypatch.setattr(module, "get_engine", lambda *_args, **_kw: engine)
    monkeypatch.setattr(module, "_validate_model_name", lambda *_args, **_kw: None)
    monkeypatch.setattr(module, "_check_admission_or_503", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        module, "_release_admission_unless_committed", lambda *_args, **_kw: None
    )
    monkeypatch.setattr(module, "_wait_with_disconnect", _await_direct)


@pytest.mark.asyncio
async def test_chat_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from vllm_mlx.api.models import ChatCompletionRequest
    from vllm_mlx.routes import chat

    engine = _CaptureChatEngine()
    _patch_common_route_deps(monkeypatch, chat, engine)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "enforce_context_length_for_messages", lambda *a, **k: 1)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=None,
    )

    await chat._create_chat_completion_impl(
        request,
        _RawRequest(),
        engine,
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert engine.captured_max_tokens == 777


@pytest.mark.asyncio
async def test_completions_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from vllm_mlx.api.models import CompletionRequest
    from vllm_mlx.routes import completions

    engine = _CaptureCompletionEngine()
    _patch_common_route_deps(monkeypatch, completions, engine)
    monkeypatch.setattr(
        completions, "enforce_context_length_for_prompt", lambda *a, **k: None
    )

    await completions.create_completion(
        CompletionRequest(model="test-model", prompt="hi", max_tokens=None),
        _RawRequest(),
    )

    assert engine.captured_max_tokens == 777


@pytest.mark.asyncio
async def test_responses_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from vllm_mlx.api.models import ChatCompletionRequest
    from vllm_mlx.api.responses_models import ResponsesRequest
    from vllm_mlx.routes import responses

    engine = _CaptureChatEngine()
    monkeypatch.setattr(responses, "_resolve_max_tokens", lambda *_args, **_kw: 777)
    monkeypatch.setattr(responses, "_wait_with_disconnect", _await_direct)

    openai_request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=None,
    )
    responses_request = ResponsesRequest(
        model="test-model",
        input=[{"type": "message", "role": "user", "content": "hi"}],
    )

    await responses._non_stream(
        engine,
        openai_request,
        responses_request,
        _RawRequest(),
    )

    assert engine.captured_max_tokens == 777


@pytest.mark.asyncio
async def test_anthropic_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from vllm_mlx.routes import anthropic

    engine = _CaptureChatEngine()
    _patch_common_route_deps(monkeypatch, anthropic, engine)
    monkeypatch.setattr(
        anthropic, "enforce_context_length_for_messages", lambda *a, **k: 1
    )

    await anthropic.create_anthropic_message(
        _RawRequest(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
        )
    )

    assert engine.captured_max_tokens == 777
