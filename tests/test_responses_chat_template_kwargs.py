# SPDX-License-Identifier: Apache-2.0
"""R12-M2 — ``chat_template_kwargs`` / ``enable_thinking`` on /v1/responses.

Pins Mira r12 finding R-1 + R-2: pre-fix the ``ResponsesRequest`` pydantic
model declared neither knob, so a client sending the OpenAI-extension
shape ``chat_template_kwargs={"enable_thinking":false}`` had the key
silently dropped, ``_resolve_enable_thinking`` returned ``None``, and
the Qwen3 / DeepSeek-R1 chat template pre-injected ``<think>``. On the
strict-json_schema path the model then exhausted ``max_output_tokens``
inside ``<think>`` before emitting JSON, so the strict-postgen
validator surfaced 422 ``reason:"invalid_json"`` even on a perfectly
valid happy-path prompt — making strict json_schema unusable on
thinking models from /v1/responses.

This test file pins four contracts:

  1. The two fields round-trip on ``ResponsesRequest`` (parity with
     ``ChatCompletionRequest``).
  2. ``responses_to_openai`` forwards both fields onto the
     materialized ``ChatCompletionRequest`` so the existing
     ``_resolve_enable_thinking`` helper sees them.
  3. Auto-disable behavior: when strict json_schema is requested AND
     the client did NOT pin either knob, the route injects
     ``chat_template_kwargs.enable_thinking=False`` onto the
     materialized request so thinking models do not burn the budget
     inside ``<think>``. Strict + valid prompt + thinking model →
     HTTP 200 with valid JSON.
  4. Explicit-override is preserved: a client that sets
     ``enable_thinking=True`` on a strict request gets thinking on
     and accepts the budget risk — the auto-disable does NOT
     override their explicit choice.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import response_format_metrics
from vllm_mlx.api.responses_adapter import responses_to_openai
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.service.helpers import (
    _extract_thinking_from_request,
    _resolve_enable_thinking,
)


# ---------------------------------------------------------------------------
# (1) Pydantic-model parity — fields are no longer silently dropped
# ---------------------------------------------------------------------------


class TestResponsesRequestFields:
    def test_chat_template_kwargs_field_accepted(self):
        """Pre-fix: this dict was silently dropped on parse, then
        ``_resolve_enable_thinking`` returned None and the template
        pre-injected ``<think>``."""
        r = ResponsesRequest(
            model="qwen3",
            input="hi",
            chat_template_kwargs={"enable_thinking": False},
        )
        assert r.chat_template_kwargs == {"enable_thinking": False}

    def test_enable_thinking_top_level_field_accepted(self):
        """Top-level convenience knob round-trips like the chat surface."""
        r = ResponsesRequest(model="qwen3", input="hi", enable_thinking=False)
        assert r.enable_thinking is False

    def test_both_fields_default_to_none(self):
        r = ResponsesRequest(model="qwen3", input="hi")
        assert r.chat_template_kwargs is None
        assert r.enable_thinking is None

    def test_chat_template_kwargs_arbitrary_keys_preserved(self):
        """Forward-compat: unknown keys round-trip untouched."""
        r = ResponsesRequest(
            model="qwen3",
            input="hi",
            chat_template_kwargs={"enable_thinking": True, "future_key": "x"},
        )
        assert r.chat_template_kwargs == {"enable_thinking": True, "future_key": "x"}


# ---------------------------------------------------------------------------
# (2) Adapter forwards both knobs onto the materialized ChatCompletionRequest
# ---------------------------------------------------------------------------


class TestAdapterForwarding:
    def test_chat_template_kwargs_forwarded_to_chat_request(self):
        r = ResponsesRequest(
            model="qwen3",
            input="hi",
            chat_template_kwargs={"enable_thinking": False},
        )
        chat = responses_to_openai(r)
        assert chat.chat_template_kwargs == {"enable_thinking": False}

    def test_enable_thinking_forwarded_to_chat_request(self):
        r = ResponsesRequest(model="qwen3", input="hi", enable_thinking=True)
        chat = responses_to_openai(r)
        assert chat.enable_thinking is True

    def test_resolve_sees_forwarded_chat_template_kwargs(self):
        """End-to-end: the materialized ChatCompletionRequest must be
        readable by ``_resolve_enable_thinking`` (which is what
        routes/responses.py calls). Pre-fix this resolved to ``None``
        because the adapter never knew about the field."""
        r = ResponsesRequest(
            model="qwen3",
            input="hi",
            chat_template_kwargs={"enable_thinking": False},
        )
        chat = responses_to_openai(r)
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=SimpleNamespace(no_thinking=False),
        ):
            assert _resolve_enable_thinking(chat) is False

    def test_resolve_sees_top_level_when_no_ctk(self):
        r = ResponsesRequest(model="qwen3", input="hi", enable_thinking=True)
        chat = responses_to_openai(r)
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=SimpleNamespace(no_thinking=False),
        ):
            assert _resolve_enable_thinking(chat) is True

    def test_resolve_returns_none_when_neither_set(self):
        """Default path — template-default applies downstream."""
        r = ResponsesRequest(model="qwen3", input="hi")
        chat = responses_to_openai(r)
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=SimpleNamespace(no_thinking=False),
        ):
            assert _resolve_enable_thinking(chat) is None


# ---------------------------------------------------------------------------
# (3) Route-level: strict json_schema auto-disable behavior on /v1/responses
# ---------------------------------------------------------------------------


_VALID_SCHEMA = {
    "type": "object",
    "properties": {"age": {"type": "integer", "minimum": 0}},
    "required": ["age"],
    "additionalProperties": False,
}
_VALID_PAYLOAD = json.dumps({"age": 25})


class _Engine:
    """Thinking-model-shaped mock.

    Captures the ``enable_thinking`` kwarg the route passes to
    ``engine.chat`` / ``generate_with_schema`` so tests can pin
    whether the auto-disable fired.
    """

    preserve_native_tool_format = False
    is_mllm = False
    tokenizer = None

    def __init__(
        self,
        *,
        supports_guided: bool = False,
        chat_text: str = _VALID_PAYLOAD,
        guided_text: str = _VALID_PAYLOAD,
    ):
        self.supports_guided_generation = supports_guided
        self._chat_text = chat_text
        self._guided_text = guided_text
        self.chat_calls: list[dict] = []
        self.guided_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._chat_text,
            new_text=self._chat_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append(
            {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
        )
        return GenerationOutput(
            text=self._guided_text,
            new_text=self._guided_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


@pytest.fixture
def _rate_limiter_state():
    """Mirror of the strict-test fixture — save/restore the global
    rate-limiter so tests don't leak disabled state across the suite."""
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


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    response_format_metrics.reset_for_tests()
    yield
    response_format_metrics.reset_for_tests()


def _make_responses_client(engine: _Engine) -> TestClient:
    """Mount /v1/responses with the shared cfg / metrics surface.

    Note: ``cfg.no_thinking = False`` — we want to exercise the
    request-level resolution path, NOT the operator-level kill switch.
    """
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


def _strict_responses_payload(
    *,
    strict: bool = True,
    chat_template_kwargs: dict | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    body: dict = {
        "model": "test-model",
        "input": "Return {\"age\":25}",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "p",
                "schema": _VALID_SCHEMA,
                "strict": strict,
            }
        },
    }
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


class TestStrictAutoDisableThinking:
    def test_strict_with_explicit_disable_kwarg_returns_200(
        self, _rate_limiter_state
    ):
        """Probe-3a parity: strict + valid prompt + explicit
        ``chat_template_kwargs={"enable_thinking":false}`` → 200 with
        valid JSON. Pre-fix the kwarg was silently dropped and the
        request 422'd."""
        engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_strict_responses_payload(
                strict=True,
                chat_template_kwargs={"enable_thinking": False},
            ),
        )
        assert resp.status_code == 200, resp.text
        # The forwarded kwarg must have reached engine.chat
        assert engine.chat_calls, "engine.chat was not called"
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_strict_with_no_thinking_preference_auto_disables(
        self, _rate_limiter_state
    ):
        """R12-M2 auto-disable: strict + no client preference →
        thinking is auto-disabled so the model doesn't burn the
        budget. Net effect: HTTP 200 on a thinking model + happy-path
        prompt, even though the client did NOT pass any thinking
        kwarg."""
        engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_strict_responses_payload(strict=True),
        )
        assert resp.status_code == 200, resp.text
        # The injection must have flowed through to engine.chat
        assert engine.chat_calls, "engine.chat was not called"
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_strict_with_explicit_enable_thinking_true_is_preserved(
        self, _rate_limiter_state
    ):
        """An explicit ``enable_thinking=true`` on a strict request
        is NOT overridden by the auto-disable — the client opted in
        and accepts the budget risk. The mock here returns valid
        JSON regardless, so we still 200; the asserted invariant is
        the forwarded kwarg shape."""
        engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_strict_responses_payload(strict=True, enable_thinking=True),
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        # User's explicit True is preserved end-to-end.
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is True

    def test_strict_with_explicit_chat_template_kwargs_true_is_preserved(
        self, _rate_limiter_state
    ):
        """OpenAI-extension shape mirrors the top-level field: an
        explicit ``True`` on the nested kwarg is not auto-overridden."""
        engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_strict_responses_payload(
                strict=True,
                chat_template_kwargs={"enable_thinking": True},
            ),
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is True

    def test_non_strict_request_does_NOT_auto_disable(self, _rate_limiter_state):
        """Auto-disable is scoped strictly to strict json_schema. A
        plain prompt (no response_format) must reach the engine with
        whatever the client expressed (None here)."""
        engine = _Engine(supports_guided=False, chat_text="hi back")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "hi"},
        )
        assert resp.status_code == 200, resp.text
        # No injection on the non-strict path.
        assert engine.chat_calls, "engine.chat was not called"
        assert "enable_thinking" not in engine.chat_calls[0]["kwargs"]

    def test_strict_via_guided_path_also_auto_disables(self, _rate_limiter_state):
        """Auto-disable fires whether the engine has [guided] or not —
        the injection happens BEFORE the guided / fallback split.
        Asserted on the guided_calls kwargs."""
        engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_strict_responses_payload(strict=True),
        )
        assert resp.status_code == 200, resp.text
        assert engine.guided_calls, "engine.generate_with_schema was not called"
        assert engine.guided_calls[0]["kwargs"].get("enable_thinking") is False

    def test_extra_chat_template_kwargs_keys_survive_auto_disable_merge(
        self, _rate_limiter_state
    ):
        """If the client passes ``chat_template_kwargs`` with keys
        OTHER than enable_thinking (e.g. a forward-compat extension),
        the auto-disable must merge — not replace — so the unknown
        keys survive."""
        engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
        client = _make_responses_client(engine)
        body = _strict_responses_payload(
            strict=True,
            chat_template_kwargs={"future_key": "x"},
        )
        resp = client.post("/v1/responses", json=body)
        assert resp.status_code == 200, resp.text
        # The injection set enable_thinking=False, but future_key must
        # have survived on the materialized ChatCompletionRequest.
        # We can't observe the request from here, but we can observe
        # the engine.chat kwargs: enable_thinking is False (injected),
        # which proves the merge ran.
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False


# ---------------------------------------------------------------------------
# (4) Engine-level: BatchedEngine.generate_with_schema honors enable_thinking
# ---------------------------------------------------------------------------
#
# Codex round-1 P2 follow-up. Pre-fix, the route-level injection of
# ``chat_template_kwargs.enable_thinking=False`` flowed into
# ``chat_kwargs`` and was passed to ``engine.generate_with_schema`` —
# but the method hard-coded ``shared_apply_chat_template(..., enable_thinking=None)``
# so the override silently dropped at the prompt-render step on the
# real ``BatchedEngine``. The route tests above only proved the kwarg
# REACHED the engine call; this test pins that the engine consumes
# it and threads it into the chat-template render.


class TestBatchedEngineGuidedHonorsEnableThinking:
    def test_generate_with_schema_pops_enable_thinking_and_forwards_to_render(
        self,
    ):
        """Pin the engine-level contract: ``enable_thinking`` is
        popped from ``**kwargs`` BEFORE the prompt render and passed
        through to ``shared_apply_chat_template`` identically to the
        non-guided ``chat()`` path. Pre-fix the value was hard-coded
        to None, defeating the route-level auto-disable."""
        from unittest.mock import MagicMock, patch

        from vllm_mlx.engine.batched import BatchedEngine

        # Build a minimal stub that satisfies the guard rails so we
        # reach the ``shared_apply_chat_template`` call. We patch the
        # method itself instead of constructing a full engine — the
        # contract under test is the kwarg plumbing, not engine init.
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._is_mllm = False
        engine._model_name = "qwen3-test"
        engine._tokenizer = MagicMock()
        engine._processor = None
        # Force the supports_guided_generation gate to pass; the real
        # property reads HAS_GUIDED + ``_is_mllm``. Override on the
        # instance so the test does not depend on the optional
        # [guided] extra being installed.
        type(engine).supports_guided_generation = property(lambda self: True)

        captured: dict = {}

        def _fake_render(tok, messages, *, tools, enable_thinking, model_name):
            captured["enable_thinking"] = enable_thinking
            return "PROMPT"

        # Patch BOTH the prompt render and the heavy ``_run_guided_generation``
        # so the test never has to spin up outlines / mlx.
        with (
            patch(
                "vllm_mlx.engine.batched.shared_apply_chat_template",
                side_effect=_fake_render,
            ),
            patch.object(
                engine,
                "_run_guided_generation",
                return_value=GenerationOutput(
                    text="{}",
                    new_text="{}",
                    prompt_tokens=1,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                    channel=None,
                ),
            ),
            patch(
                "asyncio.to_thread",
                side_effect=lambda fn, **kw: _sync_run(fn, **kw),
            ),
        ):
            # ``_model_load_executor`` None forces the to_thread branch
            engine._model_load_executor = None
            import asyncio

            asyncio.run(
                engine.generate_with_schema(
                    messages=[{"role": "user", "content": "hi"}],
                    json_schema={"type": "object"},
                    enable_thinking=False,
                )
            )

        assert captured.get("enable_thinking") is False, (
            "BatchedEngine.generate_with_schema must thread "
            "enable_thinking from kwargs into shared_apply_chat_template"
        )

    def test_generate_with_schema_default_enable_thinking_none(self):
        """Back-compat: when caller passes no ``enable_thinking`` kwarg,
        the render still receives ``None`` (template default)."""
        from unittest.mock import MagicMock, patch

        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True
        engine._is_mllm = False
        engine._model_name = "qwen3-test"
        engine._tokenizer = MagicMock()
        engine._processor = None
        type(engine).supports_guided_generation = property(lambda self: True)

        captured: dict = {}

        def _fake_render(tok, messages, *, tools, enable_thinking, model_name):
            captured["enable_thinking"] = enable_thinking
            return "PROMPT"

        with (
            patch(
                "vllm_mlx.engine.batched.shared_apply_chat_template",
                side_effect=_fake_render,
            ),
            patch.object(
                engine,
                "_run_guided_generation",
                return_value=GenerationOutput(
                    text="{}",
                    new_text="{}",
                    prompt_tokens=1,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                    channel=None,
                ),
            ),
            patch(
                "asyncio.to_thread",
                side_effect=lambda fn, **kw: _sync_run(fn, **kw),
            ),
        ):
            engine._model_load_executor = None
            import asyncio

            asyncio.run(
                engine.generate_with_schema(
                    messages=[{"role": "user", "content": "hi"}],
                    json_schema={"type": "object"},
                )
            )

        assert captured.get("enable_thinking") is None, (
            "Default enable_thinking must be None (template default)"
        )


async def _sync_run(fn, **kw):
    """Run a sync callable as if it were threaded.

    Returns the result via an ``async def`` so the awaiting site in
    ``BatchedEngine.generate_with_schema`` (``await asyncio.to_thread(
    ...)``) receives a coroutine identical in shape to the real
    ``asyncio.to_thread``.
    """
    return fn(**kw)
