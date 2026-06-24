# SPDX-License-Identifier: Apache-2.0
"""R12-T1F — auto-disable ``enable_thinking`` when tools are provided.

Pins the systematic fix for the 0.8.16 operator dogfood finding:
tool calling on thinking models was broken-by-default because the
model burned its entire ``max_tokens`` budget inside
``<think>...</think>`` before emitting the ``<tool_call>`` envelope.
The agent-SDK tight-budget pattern (``max_tokens=50..100``) made the
request finish with ``finish_reason="length"`` and ``tool_calls=None``
— the tool never fired.

The fix mirrors M-2 (PR #877) which solved the SAME failure mode for
strict json_schema requests: when the client did not pin a thinking
preference, the route now injects
``chat_template_kwargs.enable_thinking=False`` BEFORE
``_resolve_enable_thinking`` is consulted. The trigger here is
"tools is non-empty" instead of "response_format strict=true", but
the merge contract, single-source-of-truth helper, and explicit-
override-preservation rules are identical — so a future surface
that adds tool support inherits the fix for free by calling
``maybe_auto_disable_thinking_for_tools``.

This file pins five contracts:

  1. Helper-level: ``maybe_auto_disable_thinking_for_tools`` fires
     only when tools are non-empty AND the client did not pin a
     thinking preference; explicit overrides (top-level or nested
     kwarg, ``True`` or ``False``) are always honored; the merge
     is non-destructive (forward-compat keys survive).
  2. /v1/chat/completions: route handler triggers the helper,
     ``engine.chat`` sees ``enable_thinking=False`` when tools are
     declared without a thinking preference.
  3. /v1/chat/completions: explicit overrides (chat_template_kwargs
     or top-level enable_thinking) bypass the auto-disable.
  4. /v1/chat/completions: requests WITHOUT tools are unaffected
     (no auto-disable injection).
  5. /v1/responses: tools-only path also triggers the auto-disable
     (and the strict + tools combination, which would be rejected
     as 400 by the existing ``strict_with_tools_unsupported`` gate
     above the auto-disable, is exercised via direct helper test
     to keep the contract uniform).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import response_format_metrics
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.api.responses_adapter import responses_to_openai
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.service.helpers import (
    _resolve_enable_thinking,
    maybe_auto_disable_thinking_for_tools,
)

# ---------------------------------------------------------------------------
# (1) Helper-level: maybe_auto_disable_thinking_for_tools
# ---------------------------------------------------------------------------
#
# Helper-level coverage runs on a SimpleNamespace shim instead of a
# real ChatCompletionRequest so the gate is exercised in isolation
# from Pydantic validation. The integration tests below confirm the
# helper is wired into both real route handlers.


class TestHelperAutoDisableForTools:
    def test_no_tools_returns_false_and_leaves_request_untouched(self):
        """If the client did not declare tools, the helper is a no-op.
        Plain-prose requests on thinking models continue to default
        to thinking ON (template default) — the auto-disable must
        not bleed into the no-tools lane."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs is None
        assert req.enable_thinking is None

    def test_empty_tools_returns_false(self):
        """Empty-list ``tools=[]`` is the same as no tools — the model
        will not emit a tool_call, so the budget-burn failure mode
        does not apply."""
        req = SimpleNamespace(
            tools=[],
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs is None

    def test_tools_with_no_preference_fires_auto_disable(self):
        """The load-bearing happy path: tools declared + no thinking
        preference → inject ``enable_thinking=False`` so the model
        does not burn the budget inside ``<think>``."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_tools_with_explicit_enable_thinking_true_top_level_skipped(self):
        """Client explicitly opted IN to thinking on a tools request —
        accept the budget risk and do NOT override. The opt-in shape
        the chat surface accepts is the top-level convenience knob."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs=None,
            enable_thinking=True,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        # No injection — chat_template_kwargs stays untouched.
        assert req.chat_template_kwargs is None

    def test_tools_with_explicit_enable_thinking_false_top_level_skipped(self):
        """Client explicitly opted OUT — already what the auto-disable
        would do, but we still skip the injection because the helper's
        contract is 'do not touch the knob if the client expressed a
        preference'. Belt-and-suspenders: the resolved choice is
        still ``False``, just via the top-level field."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs=None,
            enable_thinking=False,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs is None

    def test_tools_with_explicit_enable_thinking_true_nested_kwarg_skipped(self):
        """Client opted IN via the OpenAI-extension shape
        ``chat_template_kwargs={"enable_thinking":true}``. The
        precedence order (``_extract_thinking_from_request``) honors
        this over the auto-disable injection."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        # The original ``True`` survives — the helper did not overwrite.
        assert req.chat_template_kwargs == {"enable_thinking": True}

    def test_tools_with_explicit_enable_thinking_false_nested_kwarg_skipped(self):
        """Same as above but with explicit OFF — the helper's contract
        is 'do not touch the knob if the client expressed a preference',
        regardless of which value they expressed."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_merge_preserves_forward_compat_keys(self):
        """The merge MUST be non-destructive: a client-supplied
        ``chat_template_kwargs={"future_key":"x"}`` (no enable_thinking
        key) must survive the auto-disable injection — the resulting
        dict carries BOTH the client key AND the auto-injected
        ``enable_thinking=False``. Mirrors M-2 codex round-3 BLOCKING."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs={"future_key": "x"},
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {
            "future_key": "x",
            "enable_thinking": False,
        }

    def test_tool_choice_none_skips_auto_disable(self):
        """Codex r1 BLOCKING (R12-T1F follow-up): when the client
        attached tools BUT pinned ``tool_choice="none"``, the OpenAI
        spec says the model must ignore the tool list and answer in
        prose. The budget-burn rationale for auto-disabling thinking
        does not apply — the model is not going to emit a tool_call,
        so a Qwen3 prose answer should keep default-on thinking. The
        helper MUST skip the injection so a prose request does not
        get its thinking quietly turned off solely because tool
        DEFINITIONS were attached."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="none",
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs is None
        assert req.enable_thinking is None

    def test_tool_choice_auto_still_fires_auto_disable(self):
        """Negative control for ``tool_choice="none"`` skip: the
        default ``tool_choice="auto"`` (model picks whether to call
        a tool) is still a tool-emitting path, so the budget-burn
        risk applies and the auto-disable MUST fire. Without this
        assertion the ``tool_choice == "none"`` gate could over-
        match (e.g. a typo that excluded ``"auto"`` too) and we
        would silently regress the load-bearing happy path."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="auto",
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_tool_choice_required_still_fires_auto_disable(self):
        """Negative control mirroring ``"auto"`` above: ``required``
        is the strongest tool-emitting signal (the OpenAI spec
        guarantees a tool_call), so the budget-burn risk applies
        and the helper MUST fire."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="required",
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_tool_choice_named_function_still_fires_auto_disable(self):
        """Negative control: the OpenAI ``tool_choice={"type":
        "function", "function": {"name": ...}}`` shape forces a
        specific named tool. That is the strongest tool-emitting
        signal and the budget-burn risk applies. The gate MUST be
        ``isinstance(str) and == "none"`` only — dict shapes must
        fall through."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice={"type": "function", "function": {"name": "x"}},
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_tool_choice_none_with_explicit_thinking_true_still_honors_client(self):
        """``tool_choice="none"`` is skipped early; explicit client
        preference (``enable_thinking=True``) is not reached as a
        decision point. Verify the request shape is fully preserved
        — thinking stays ON because the client asked for it AND
        because the helper did not touch the request at all."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="none",
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": True}


# ---------------------------------------------------------------------------
# (2) /v1/chat/completions: route-level integration
# ---------------------------------------------------------------------------


class _ChatEngine:
    """Thinking-model-shaped mock that captures the kwargs the route
    forwards to ``engine.chat``. The captured ``enable_thinking`` is
    the load-bearing assertion target — it's the signal that the
    route-level auto-disable actually reached the engine layer.

    Returns a clean ``stop`` finish on every call so the chat route's
    finalize / parse path doesn't hit edge cases unrelated to the
    auto-disable contract.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True
    tokenizer = None

    def __init__(self, *, text: str = "ok"):
        self._text = text
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._text,
            raw_text=self._text,
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            cached_tokens=0,
        )


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    """Mirror of the M-2 sibling — reset response_format counters so
    cross-test bleed doesn't make strict counts non-deterministic."""
    response_format_metrics.reset_for_tests()
    yield
    response_format_metrics.reset_for_tests()


@pytest.fixture
def _rate_limiter_state():
    """Mirror of the M-2 sibling — save/restore the global rate-limiter
    so tests don't leak disabled state across the suite."""
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


def _make_chat_client(engine: _ChatEngine) -> TestClient:
    """Mount /v1/chat/completions with cfg.no_thinking=False so the
    request-level resolution path is the one under test (not the
    operator kill switch)."""
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


class TestChatRouteAutoDisableForTools:
    def test_tools_no_preference_auto_disables_thinking(self, _rate_limiter_state):
        """The 0.8.16 operator dogfood repro: tools declared, no
        thinking preference → the route injects
        ``enable_thinking=False`` so the engine kwarg is False. Pre-fix
        the engine kwarg was None (template default = True for
        Qwen3 family) and the model burned the budget inside
        ``<think>``."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Weather in Paris? Use the tool."}
                ],
                "tools": [_WEATHER_TOOL],
            },
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        kwargs = engine.chat_calls[0]["kwargs"]
        # Load-bearing assertion: the auto-disable reached the engine.
        assert kwargs.get("enable_thinking") is False, (
            "tools + no preference must inject enable_thinking=False; "
            f"engine saw kwargs={kwargs!r}"
        )

    def test_tools_explicit_enable_thinking_true_preserved(self, _rate_limiter_state):
        """Explicit opt-in: client passed ``enable_thinking=true`` —
        the auto-disable must NOT override. The client accepts the
        budget risk (they will need to raise ``max_tokens`` for the
        tool_call to fit)."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_WEATHER_TOOL],
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert kwargs.get("enable_thinking") is True, (
            "explicit chat_template_kwargs.enable_thinking=true must "
            "survive end-to-end on the tools path"
        )

    def test_tools_explicit_enable_thinking_false_preserved(self, _rate_limiter_state):
        """Explicit opt-out: client passed ``enable_thinking=false`` —
        the resolved value is False either way, but we exercise the
        preservation contract for parity with the True case."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_WEATHER_TOOL],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert kwargs.get("enable_thinking") is False

    def test_tools_top_level_enable_thinking_true_preserved(self, _rate_limiter_state):
        """Top-level convenience knob is honored end-to-end on the
        tools path identically to the nested kwarg form."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [_WEATHER_TOOL],
                "enable_thinking": True,
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert kwargs.get("enable_thinking") is True

    def test_no_tools_thinking_model_unaffected(self, _rate_limiter_state):
        """When the request has no tools the auto-disable must NOT
        fire — plain-prose requests on thinking models keep the
        template default (engine kwarg absent → renderer applies the
        default). This is the no-regression gate for the no-tools
        lane."""
        engine = _ChatEngine(text="hi back")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        # No injection on the no-tools path. The chat route forwards
        # ``enable_thinking`` to the engine only when ``resolved_thinking``
        # is not None — so the kwarg must be absent here.
        assert "enable_thinking" not in kwargs, (
            "no-tools request must not inject enable_thinking; "
            f"engine saw kwargs={kwargs!r}"
        )

    def test_tools_forward_compat_key_survives_merge(self, _rate_limiter_state):
        """Codex round-3 BLOCKING contract from M-2: forward-compat
        keys the client passed in ``chat_template_kwargs`` survive the
        merge. The resolved request that
        ``_resolve_enable_thinking`` sees inside the route MUST be
        ``{"future_key":"x", "enable_thinking": False}``."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)

        captured_ctk: list[dict | None] = []
        import vllm_mlx.routes.chat as _chat_mod

        original = _chat_mod._resolve_enable_thinking

        def _spy(request):
            ctk = getattr(request, "chat_template_kwargs", None)
            captured_ctk.append(dict(ctk) if ctk is not None else None)
            return original(request)

        with patch.object(_chat_mod, "_resolve_enable_thinking", side_effect=_spy):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [_WEATHER_TOOL],
                    "chat_template_kwargs": {"future_key": "x"},
                },
            )

        assert resp.status_code == 200, resp.text
        assert captured_ctk, "_resolve_enable_thinking was never called"
        first_seen = captured_ctk[0]
        assert first_seen == {"future_key": "x", "enable_thinking": False}, (
            "auto-disable merge dropped the client's forward-compat "
            f"key: got {first_seen}"
        )
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False


# ---------------------------------------------------------------------------
# (3) /v1/responses: route-level integration
# ---------------------------------------------------------------------------
#
# The Responses route reuses ``maybe_auto_disable_thinking_for_tools``
# at the SAME plumbing layer as the chat surface (right after the
# adapter materializes ``openai_request`` from the
# ``ResponsesRequest``). Strict json_schema + tools is mutually
# exclusive on this surface and returns 400 ``strict_with_tools_
# unsupported`` BEFORE either auto-disable can fire, so the two
# triggers never both reach the helper on /v1/responses — but the
# helper itself stays oblivious (single contract).


class _ResponsesEngine:
    """Thinking-model-shaped mock for /v1/responses. Captures
    ``engine.chat`` kwargs the same way ``_ChatEngine`` does."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True
    tokenizer = None

    def __init__(self, *, text: str = "ok"):
        self._text = text
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._text,
            new_text=self._text,
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_responses_client(engine: _ResponsesEngine) -> TestClient:
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


def _responses_payload(
    *,
    chat_template_kwargs: dict | None = None,
    enable_thinking: bool | None = None,
    include_tools: bool = True,
) -> dict:
    body: dict = {
        "model": "test-model",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Weather in Paris? Use the tool."}
                ],
            }
        ],
        "max_output_tokens": 50,
    }
    if include_tools:
        body["tools"] = [_RESPONSES_WEATHER_TOOL]
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


class TestResponsesRouteAutoDisableForTools:
    def test_tools_no_preference_auto_disables_thinking(self, _rate_limiter_state):
        """/v1/responses parity: tools-only request with no thinking
        preference → engine sees ``enable_thinking=False``. Single-
        source-of-truth helper drives both surfaces."""
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post("/v1/responses", json=_responses_payload())
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_tools_explicit_enable_thinking_true_preserved(self, _rate_limiter_state):
        """Top-level opt-in on /v1/responses is honored end-to-end."""
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses", json=_responses_payload(enable_thinking=True)
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is True

    def test_tools_explicit_chat_template_kwargs_false_preserved(
        self, _rate_limiter_state
    ):
        """Nested-kwarg opt-out (False) is preserved on /v1/responses
        — the resolved value is the same as the auto-disable, but the
        client preference is honored without the merge fallback
        firing."""
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_responses_payload(chat_template_kwargs={"enable_thinking": False}),
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_no_tools_thinking_model_unaffected(self, _rate_limiter_state):
        """No-regression: a no-tools /v1/responses request must not
        inject ``enable_thinking`` (the engine kwarg is absent so the
        template default applies)."""
        engine = _ResponsesEngine(text="hi back")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses", json=_responses_payload(include_tools=False)
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert "enable_thinking" not in kwargs, (
            "no-tools request must not inject enable_thinking; "
            f"engine saw kwargs={kwargs!r}"
        )


# ---------------------------------------------------------------------------
# (4) Combined trigger: tools + strict json_schema (helper-level only)
# ---------------------------------------------------------------------------
#
# At the surface level the two are mutually exclusive on /v1/responses
# (``strict_with_tools_unsupported`` 400) and on /v1/chat/completions
# (``strict_with_tools_unsupported`` 400 — see chat.py around the
# strict_mode gate). But the helper itself must be idempotent: a future
# surface that lifts the mutual-exclusion gate (or a request that
# bypasses the gate via a back door we haven't found yet) should still
# resolve to ``enable_thinking=False`` exactly once, with both auto-
# disable triggers firing through the same merge path.


class TestCombinedTriggersHelperLevel:
    def test_tools_helper_is_idempotent(self):
        """Calling the helper twice on the same request must not
        produce a different state — once fired, the merge has already
        installed ``enable_thinking=False`` and the second call is a
        no-op (the precedence check sees the injected key)."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        # Second call: precedence check sees the injected key, returns
        # False, leaves the dict alone.
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_tools_with_forward_compat_and_then_resolve(self):
        """Round-trip from helper → ``_resolve_enable_thinking``: after
        the helper fires, the standard resolution path returns False
        for the resolved value, with the forward-compat key still
        present on the request dict."""
        req = ChatCompletionRequest(
            model="qwen3-test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "x",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            chat_template_kwargs={"future_key": "x"},
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {
            "future_key": "x",
            "enable_thinking": False,
        }
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=SimpleNamespace(no_thinking=False),
        ):
            assert _resolve_enable_thinking(req) is False


# ---------------------------------------------------------------------------
# (5) ResponsesRequest → ChatCompletionRequest round-trip
# ---------------------------------------------------------------------------
#
# Pre-fix path the bug took on /v1/responses: client sends tools, the
# adapter materializes a ``ChatCompletionRequest``, and the route
# layer fires the helper on the materialized request. This test pins
# that the helper fires AFTER the adapter step.


class TestResponsesAdapterRoundTripWithHelper:
    def test_adapter_then_helper_injects_on_materialized_request(self):
        """The route flow is:
        ResponsesRequest (tools=[...], no thinking preference)
        → responses_to_openai → ChatCompletionRequest
        → maybe_auto_disable_thinking_for_tools(openai_request)
        → chat_template_kwargs={"enable_thinking": False}
        """
        resp_req = ResponsesRequest(
            model="qwen3-test",
            input="Weather in Paris?",
            tools=[_RESPONSES_WEATHER_TOOL],
        )
        chat_req = responses_to_openai(resp_req)
        # Pre-injection: adapter forwarded ``chat_template_kwargs=None``
        # and ``tools`` was converted to the chat-shape.
        assert chat_req.chat_template_kwargs is None
        assert chat_req.tools, "adapter must forward tools onto chat request"
        # Helper fires on the materialized chat request.
        assert maybe_auto_disable_thinking_for_tools(chat_req) is True
        assert chat_req.chat_template_kwargs == {"enable_thinking": False}

    def test_adapter_preserves_explicit_thinking_then_helper_skips(self):
        """If the ResponsesRequest carried an explicit
        ``enable_thinking``, the adapter forwards it onto the chat
        request — the helper then sees the preference and skips the
        injection."""
        resp_req = ResponsesRequest(
            model="qwen3-test",
            input="Weather in Paris?",
            tools=[_RESPONSES_WEATHER_TOOL],
            enable_thinking=True,
        )
        chat_req = responses_to_openai(resp_req)
        assert chat_req.enable_thinking is True
        assert maybe_auto_disable_thinking_for_tools(chat_req) is False
        # No nested kwarg installed — adapter preserved the original
        # None, helper did nothing.
        assert chat_req.chat_template_kwargs is None


# ---------------------------------------------------------------------------
# (6) Sanity: the new test file does not regress the M-2 strict path
# ---------------------------------------------------------------------------


class TestM2StrictPathStillFires:
    """Sanity gate against accidentally re-introducing the M-2 bug.
    A pure helper-level test that the strict json_schema auto-disable
    in the /v1/responses route is independent from the new tools
    auto-disable — they share the precedence check
    (``_extract_thinking_from_request``) but live in separate route
    branches."""

    def test_strict_json_schema_helper_path_intact(self):
        """Helper-level: the M-2 contract assumes the route consults
        ``_extract_thinking_from_request`` before merging. Pin that
        the predicate the route relies on still answers ``None`` for
        a vanilla request — so the strict json_schema branch still
        fires its own merge."""
        from vllm_mlx.service.helpers import _extract_thinking_from_request

        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        # Vanilla request → returns None → route fires its own merge.
        assert _extract_thinking_from_request(req) is None
        # After someone (route or helper) installs the auto-disable,
        # the predicate now returns False — second-pass merge is a
        # no-op, which is the idempotency contract.
        req.chat_template_kwargs = {"enable_thinking": False}
        assert _extract_thinking_from_request(req) is False


def test_responses_strict_with_tools_still_rejects_with_400(_rate_limiter_state):
    """No-regression: the existing /v1/responses
    ``strict_with_tools_unsupported`` gate must still fire as 400.
    The new tools auto-disable lives AFTER the strict branch and does
    NOT inadvertently unlock the combination."""
    engine = _ResponsesEngine(text="ok")
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "tools": [_RESPONSES_WEATHER_TOOL],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "p",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                    "strict": True,
                }
            },
        },
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    # ``strict_with_tools_unsupported`` envelope shape pre-existing.
    code = body.get("error", {}).get("code") or body.get("detail", {}).get(
        "error", {}
    ).get("code")
    assert code == "strict_with_tools_unsupported", (
        f"expected strict_with_tools_unsupported, got body={body!r}"
    )
    # And no engine call happened — the route bailed at the gate.
    assert not engine.chat_calls, (
        "strict+tools must reject at the gate without dispatching to the engine"
    )


def test_chat_strict_with_tools_still_rejects_with_400(_rate_limiter_state):
    """Mirror gate on /v1/chat/completions — ``strict_with_tools_
    unsupported`` 400 is unchanged."""
    engine = _ChatEngine(text="ok")
    client = _make_chat_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_WEATHER_TOOL],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "p",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                    "strict": True,
                },
            },
        },
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    code = body.get("error", {}).get("code") or body.get("detail", {}).get(
        "error", {}
    ).get("code")
    assert code == "strict_with_tools_unsupported", (
        f"expected strict_with_tools_unsupported, got body={body!r}"
    )
    assert not engine.chat_calls
