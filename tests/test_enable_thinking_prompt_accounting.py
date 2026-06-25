# SPDX-License-Identifier: Apache-2.0
"""rapid-mlx#280 — thread ``enable_thinking`` into prompt accounting.

Pins the systematic fix for the codex MED finding on the PR #893 review:
the R12-T1F / R12-T2F auto-disable helpers (PRs #891 / #895) mutate
``request.chat_template_kwargs`` to ``{"enable_thinking": False}`` BEFORE
the chat / responses routes call
:func:`vllm_mlx.service.helpers.enforce_context_length_for_messages`
and :func:`vllm_mlx.service.helpers.repair_messages_fit_context`, but
pre-fix those helpers rendered the prompt with ``enable_thinking=None``
(template default = ``True`` on Qwen3 / DeepSeek-R1). The mismatch had
two visible failure modes:

  A. Prompt-token estimate inflated by the ``<think>`` scaffolding the
     template would have emitted under the default, so requests that
     ACTUALLY fit the model's context window were rejected with
     ``context_length_exceeded``.
  B. The H-06 #267b strict-json-schema repair fit gate skipped retries
     that would have fit if rendered with the resolved value, surfacing
     as the original 422 ``json_schema_violation`` instead of a fixed
     ``json_repaired`` outcome.

The fix threads a new ``enable_thinking`` keyword into both helper
signatures (defaulted to ``None`` for backward compatibility) and has
``routes/chat.py`` + ``routes/responses.py`` always pass the resolved
value from the same ``_resolve_enable_thinking(...)`` they already
compute for ``chat_kwargs["enable_thinking"]``. Single source of truth
across the four call sites; no re-resolution.

These tests pin the contract at three layers:

  1. Helper-level: ``enable_thinking`` is forwarded to ``build_prompt``
     exactly once and the value is honoured (smoke against future
     refactor that drops the kwarg).
  2. /v1/chat/completions: a request whose prompt-token count DEPENDS
     on the ``enable_thinking`` value renders the way the engine will
     and the gate accepts a request that fits with auto-disable on.
  3. /v1/responses: parity with the chat lane on the same scenario.

The test engine's ``build_prompt`` mimics how a real Qwen3 chat
template would behave: it returns a SHORTER prompt when
``enable_thinking=False`` (no ``<think>`` scaffolding) and a LONGER
prompt when ``enable_thinking`` is ``None`` or ``True``. Combined with
a tight ``model_max_length`` cap that's between the two lengths, a
request that fits with auto-disable on would have been rejected
pre-fix and is accepted post-fix.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import response_format_metrics
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.service.helpers import (
    enforce_context_length_for_messages,
    repair_messages_fit_context,
)

# ---------------------------------------------------------------------------
# Shared shims
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Token count = ``len(prompt) // 4`` (1 token per 4 chars). The
    cap is set tightly by ``model_max_length`` so a small render
    difference flips the gate decision."""

    def __init__(self, *, model_max_length: int):
        self.model_max_length = model_max_length
        self.bos_token = None

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return [0] * max(1, len(text) // 4)


class _ThinkingTemplateEngine:
    """Engine shim that emulates a Qwen3-style chat template:
    ``enable_thinking=False`` shrinks the rendered prompt (no
    ``<think>`` opener scaffolding); ``None`` / ``True`` keeps the
    longer one. The chat lane build_prompt accepts ``enable_thinking``
    as a kwarg (real ``BatchedEngine.build_prompt`` does the same).
    """

    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True
    preserve_native_tool_format = False

    # Use a synthetic "long enough to matter" prompt: 600 chars (=150
    # tokens) for the thinking-on render, 300 chars (=75 tokens) for
    # the thinking-off render. With ``model_max_length=200`` and
    # ``max_tokens=80``, the thinking-on render trips the cap
    # (150 + 80 = 230 > 200) but the thinking-off render fits
    # (75 + 80 = 155 ≤ 200). Pre-fix the gate rendered with
    # ``enable_thinking=None`` (template default) and rejected; post-
    # fix the gate consults the resolved value and accepts.
    _PROMPT_THINK_ON = "X" * 600
    _PROMPT_THINK_OFF = "X" * 300

    def __init__(self):
        self.build_prompt_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        # Tight cap that sits between the two renders' token cost +
        # max_tokens budget.
        self._tokenizer = _StubTokenizer(model_max_length=200)
        # ``get_model_max_context`` walks ``model.args.max_position_embeddings``
        # before falling back to the tokenizer; we don't set it so the
        # cap is read from the tokenizer above.

    @property
    def tokenizer(self):
        return self._tokenizer

    def build_prompt(
        self,
        messages,
        tools=None,
        enable_thinking=None,
    ):
        self.build_prompt_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "enable_thinking": enable_thinking,
            }
        )
        if enable_thinking is False:
            return self._PROMPT_THINK_OFF
        return self._PROMPT_THINK_ON

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            cached_tokens=0,
        )


class _ResponsesThinkingTemplateEngine(_ThinkingTemplateEngine):
    """``/v1/responses`` invokes ``engine.chat`` with kwargs only
    (no positional messages), so override the chat signature to match
    the real engine's chat lane."""

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_RESPONSES_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


# ---------------------------------------------------------------------------
# (1) Helper-level: enable_thinking is forwarded into build_prompt
# ---------------------------------------------------------------------------


class TestHelperForwardsEnableThinking:
    """Pin the new signature contract: both helpers accept
    ``enable_thinking`` and forward it verbatim to the engine's
    ``build_prompt``. Future refactors that drop the kwarg would
    silently re-introduce the bug — these tests block that."""

    def test_enforce_forwards_enable_thinking_false(self):
        engine = _ThinkingTemplateEngine()
        enforce_context_length_for_messages(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=80,
            enable_thinking=False,
        )
        assert engine.build_prompt_calls, "build_prompt was not invoked"
        assert engine.build_prompt_calls[0]["enable_thinking"] is False

    def test_enforce_forwards_enable_thinking_true(self):
        engine = _ThinkingTemplateEngine()
        # max_tokens trimmed so the longer (think-on) render still fits
        # the 200-token cap — we just want to verify forwarding, not
        # exercise the gate.
        enforce_context_length_for_messages(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=10,
            enable_thinking=True,
        )
        assert engine.build_prompt_calls[0]["enable_thinking"] is True

    def test_enforce_defaults_enable_thinking_none(self):
        """Default ``None`` preserves legacy behaviour for call sites
        that haven't been audited (e.g. routes/anthropic.py)."""
        engine = _ThinkingTemplateEngine()
        enforce_context_length_for_messages(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=10,
        )
        assert engine.build_prompt_calls[0]["enable_thinking"] is None

    def test_repair_forwards_enable_thinking_false(self):
        engine = _ThinkingTemplateEngine()
        repair_messages_fit_context(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=80,
            enable_thinking=False,
        )
        assert engine.build_prompt_calls[0]["enable_thinking"] is False

    def test_repair_defaults_enable_thinking_none(self):
        engine = _ThinkingTemplateEngine()
        repair_messages_fit_context(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=10,
        )
        assert engine.build_prompt_calls[0]["enable_thinking"] is None


# ---------------------------------------------------------------------------
# (2) Helper-level: gate decision changes with enable_thinking
# ---------------------------------------------------------------------------


class TestHelperGateDecisionMatchesResolvedThinking:
    """The load-bearing contract: rendering with the resolved value
    can flip the gate from REJECT to ACCEPT. Pre-fix the helper
    always rendered with the template default — a request whose
    auto-disabled render would fit got rejected anyway."""

    def test_enforce_rejects_when_rendered_with_thinking_on(self):
        """Sanity that the test engine's tight cap actually trips: a
        request that renders to 600 chars (=150 tokens) + 80 max_tokens
        exceeds the 200-token cap. This is the pre-fix surface
        (helper rendered with the default → think-on long prompt)."""
        from fastapi import HTTPException

        engine = _ThinkingTemplateEngine()
        with pytest.raises(HTTPException) as exc:
            enforce_context_length_for_messages(
                engine,
                [{"role": "user", "content": "hi"}],
                tools=None,
                max_tokens=80,
                enable_thinking=None,  # pre-fix behaviour
            )
        assert exc.value.status_code == 400

    def test_enforce_accepts_when_rendered_with_thinking_off(self):
        """The fix: with the auto-disable resolved value threaded
        through, the gate renders to 300 chars (=75 tokens) + 80
        max_tokens = 155 tokens ≤ 200 cap → accepted. Pre-fix this
        would have been rejected by the gate above."""
        engine = _ThinkingTemplateEngine()
        # No exception → request accepted. The return value is the
        # prompt-token estimate; we only care that no HTTPException
        # fired.
        result = enforce_context_length_for_messages(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=80,
            enable_thinking=False,
        )
        # 300 chars / 4 chars-per-token = 75 tokens, the helper
        # returns the count it computed.
        assert result == 75

    def test_repair_rejects_when_rendered_with_thinking_on(self):
        """Mirror of the enforce test for the repair-fit helper. Pre-
        fix this returned ``False`` (skip retry) even when the repair
        prompt would actually have fit if rendered with the resolved
        value."""
        engine = _ThinkingTemplateEngine()
        fits = repair_messages_fit_context(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=80,
            enable_thinking=None,  # pre-fix behaviour
        )
        assert fits is False

    def test_repair_accepts_when_rendered_with_thinking_off(self):
        """The fix on the strict-json-schema lane: when the gate is
        threaded the resolved value, the repair-fit decision matches
        what the engine will actually render → retry runs."""
        engine = _ThinkingTemplateEngine()
        fits = repair_messages_fit_context(
            engine,
            [{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=80,
            enable_thinking=False,
        )
        assert fits is True


# ---------------------------------------------------------------------------
# Fixtures shared with the route-level tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    response_format_metrics.reset_for_tests()
    yield
    response_format_metrics.reset_for_tests()


@pytest.fixture
def _rate_limiter_state():
    from vllm_mlx.middleware.auth import rate_limiter

    saved_enabled = rate_limiter.enabled
    saved_rpm = rate_limiter.requests_per_minute
    saved_requests = dict(rate_limiter._requests)
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()
    yield rate_limiter
    rate_limiter.enabled = saved_enabled
    rate_limiter.requests_per_minute = saved_rpm
    rate_limiter._requests.clear()
    rate_limiter._requests.update(saved_requests)


# ---------------------------------------------------------------------------
# (3) /v1/chat/completions route integration
# ---------------------------------------------------------------------------


def _make_chat_client(engine):
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app)


class TestChatRoutePromptAccountingThreading:
    """End-to-end through the chat route: a request that has
    ``tools`` declared (so R12-T1F auto-disables thinking) and whose
    prompt would NOT fit if rendered with the template default but
    WOULD fit with the resolved auto-disabled value must be accepted.

    Pre-fix the route called
    ``enforce_context_length_for_messages(... )`` without forwarding
    ``enable_thinking`` → the gate rendered with ``None`` (template
    default → long render) → rejected as ``context_length_exceeded``.
    Post-fix the route passes ``resolved_thinking`` → render matches
    what the engine will emit → accepted."""

    def test_tools_request_uses_resolved_thinking_for_prompt_accounting(
        self, _rate_limiter_state
    ):
        engine = _ThinkingTemplateEngine()
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [
                    {"role": "user", "content": "Weather in Paris? Use the tool."}
                ],
                "tools": [_WEATHER_TOOL],
            },
        )
        assert resp.status_code == 200, resp.text
        # The build_prompt call from the gate must have been issued
        # with the auto-disabled value, NOT the default ``None``.
        gate_calls = [
            c for c in engine.build_prompt_calls if c["enable_thinking"] is not None
        ]
        assert gate_calls, (
            "gate did not forward enable_thinking — at least one "
            "build_prompt call with the resolved value must exist; "
            f"all calls: {engine.build_prompt_calls!r}"
        )
        assert gate_calls[0]["enable_thinking"] is False
        # And the engine's chat lane saw the same resolved value.
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_no_tools_request_does_not_force_thinking_kwarg(
        self, _rate_limiter_state
    ):
        """No-regression for the no-tools path: when the auto-disable
        does NOT fire, the gate forwards ``enable_thinking=None`` (the
        ``_resolve_enable_thinking`` result for a vanilla request).
        That keeps the legacy "let the template choose" behaviour for
        plain-prose requests."""
        engine = _ThinkingTemplateEngine()
        # The default render is the long one, and the cap is tight,
        # so a no-tools request would 400 with the test engine. That
        # is intentional: we only verify the gate forwarded ``None``.
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 10,  # small budget so the cap isn't tripped
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        gate_call = engine.build_prompt_calls[0]
        # No auto-disable → resolved_thinking is None → gate sees None.
        assert gate_call["enable_thinking"] is None


# ---------------------------------------------------------------------------
# (4) /v1/responses route integration
# ---------------------------------------------------------------------------


def _make_responses_client(engine):
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    return TestClient(app)


class TestResponsesRoutePromptAccountingThreading:
    """Mirror of the chat suite for ``/v1/responses``. Same single-
    source-of-truth contract — the route must thread the resolved
    ``enable_thinking`` into the gate so prompt accounting matches
    the engine's actual render."""

    def test_tools_request_uses_resolved_thinking_for_prompt_accounting(
        self, _rate_limiter_state
    ):
        engine = _ResponsesThinkingTemplateEngine()
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Weather in Paris? Use the tool.",
                            }
                        ],
                    }
                ],
                "max_output_tokens": 80,
                "tools": [_RESPONSES_WEATHER_TOOL],
            },
        )
        assert resp.status_code == 200, resp.text
        gate_calls = [
            c for c in engine.build_prompt_calls if c["enable_thinking"] is not None
        ]
        assert gate_calls, (
            "gate did not forward enable_thinking — at least one "
            "build_prompt call with the resolved value must exist; "
            f"all calls: {engine.build_prompt_calls!r}"
        )
        assert gate_calls[0]["enable_thinking"] is False
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False
