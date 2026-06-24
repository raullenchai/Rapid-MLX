# SPDX-License-Identifier: Apache-2.0
"""R12-T2F-276 — auto-disable ``enable_thinking`` on a casual chat
completion to a thinking-capable model.

Pins the systematic fix for the 0.8.16 brand-new-user simulation
operator finding: a first-time SDK user writing

    client.chat.completions.create(
        model="qwen3.5-4b-4bit",   # thinking-capable
        messages=[{"role":"user","content":"In 8 words, what is rapid-mlx?"}],
        max_tokens=80,
    )

had the model burn the entire 80-token budget inside
``<think>...</think>`` and never emit an answer. The response
surfaced with ``finish_reason="length"`` and ``content`` carrying
the rescue-sentinel header followed by raw chain-of-thought — the
exact same shape the ``rapid-mlx chat`` REPL already solves via its
``--no-think`` default.

The fix mirrors the M-2 strict-json_schema gate (PR #877) and the
R12-T1F tools gate (PR #891) — same root cause (thinking models
burning the budget before the visible answer), same shape (auto-
disable when the client did not pin a preference), same merge
contract (non-destructive; forward-compat keys survive), same
single source of truth (one helper, both routes). Third member of
the auto-disable family.

This file pins six contracts:

  1. Helper-level: ``maybe_auto_disable_thinking_for_casual_chat``
     fires only when a reasoning parser is configured AND the client
     did NOT pin a thinking preference / express explicit reasoning
     intent; explicit overrides (top-level / nested kwarg / signals
     like ``reasoning_max_tokens`` / ``reasoning_effort`` /
     ``reasoning`` dict) skip the auto-disable.
  2. /v1/chat/completions: the route handler triggers the helper,
     ``engine.chat`` sees ``enable_thinking=False`` when no client-
     side reasoning signal is set.
  3. /v1/chat/completions: explicit overrides bypass the auto-disable
     (top-level / nested kwarg + every reasoning-intent signal).
  4. Non-thinking model: when no reasoning parser is configured the
     helper is a no-op (llama / mistral / qwen3-coder don't have
     ``<think>`` so the budget-burn failure mode does not apply).
  5. Interaction with R12-T1F (tools) and R12-M2 (strict json_schema):
     when an earlier helper already injected ``enable_thinking=False``,
     the casual-chat helper short-circuits via the
     ``_extract_thinking_from_request`` precedence check — no double-
     disable, no state corruption.
  6. /v1/responses parity: the same trigger contract holds on the
     /v1/responses surface, AND the Responses-native ``reasoning``
     dict (``{"effort":"low|medium|high"}``) is consulted as an
     explicit reasoning-intent signal even though
     ``responses_to_openai`` does not forward it onto the
     materialized ``ChatCompletionRequest``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import response_format_metrics
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.service.helpers import (
    _resolve_enable_thinking,
    maybe_auto_disable_thinking_for_casual_chat,
    maybe_auto_disable_thinking_for_tools,
)


# ---------------------------------------------------------------------------
# (1) Helper-level: maybe_auto_disable_thinking_for_casual_chat
# ---------------------------------------------------------------------------


@pytest.fixture
def _thinking_parser_cfg():
    """Patch ``get_config`` to return a thinking-capable config so the
    parser-name gate fires. Used by helper-level tests that don't go
    through the route layer (the route fixtures set ``cfg`` directly)."""
    with patch(
        "vllm_mlx.service.helpers.get_config",
        return_value=SimpleNamespace(
            no_thinking=False,
            reasoning_parser_name="qwen3",
        ),
    ):
        yield


@pytest.fixture
def _no_parser_cfg():
    """Patch ``get_config`` to return a non-thinking-capable config
    (no reasoning parser registered). Used to exercise the parser-name
    gate's negative case."""
    with patch(
        "vllm_mlx.service.helpers.get_config",
        return_value=SimpleNamespace(
            no_thinking=False,
            reasoning_parser_name=None,
        ),
    ):
        yield


class TestHelperAutoDisableForCasualChat:
    def test_no_reasoning_parser_returns_false(self, _no_parser_cfg):
        """Server is running a non-thinking model — the helper MUST be
        a no-op. Llama / mistral / qwen3-coder don't have ``<think>``
        so the budget-burn failure mode does not apply, and silently
        flipping the template flag could surprise downstream callers
        who rely on the engine kwarg being absent (template default)."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None
        assert req.enable_thinking is None

    def test_casual_chat_no_preference_fires_auto_disable(
        self, _thinking_parser_cfg
    ):
        """The 0.8.16 brand-new-user simulation repro: thinking-capable
        model + no client signal → inject ``enable_thinking=False`` so
        the model returns a clean answer instead of burning the budget
        inside ``<think>``."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_explicit_enable_thinking_true_top_level_skipped(
        self, _thinking_parser_cfg
    ):
        """Client explicitly opted IN — the helper MUST NOT override.
        The client accepts the budget risk and is responsible for
        sizing ``max_tokens`` accordingly."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=True,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_explicit_enable_thinking_false_top_level_skipped(
        self, _thinking_parser_cfg
    ):
        """Client explicitly opted OUT — same resolved value, but the
        helper still skips because the precedence contract is "do not
        touch the knob if the client expressed a preference"."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=False,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_explicit_enable_thinking_true_nested_kwarg_skipped(
        self, _thinking_parser_cfg
    ):
        """Nested OpenAI-extension shape — same skip contract."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": True}

    def test_reasoning_max_tokens_signal_skips_auto_disable(
        self, _thinking_parser_cfg
    ):
        """An explicit per-request reasoning cap is itself proof the
        caller wants reasoning ON (just bounded). Default-disabling
        here would silently collapse the contract to "no reasoning"."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=64,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_reasoning_effort_signal_skips_auto_disable(
        self, _thinking_parser_cfg
    ):
        """OpenAI-spec ``reasoning_effort="low|medium|high"`` is the
        documented opt-in signal — the helper MUST honor it."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort="low",
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_reasoning_dict_signal_skips_auto_disable(
        self, _thinking_parser_cfg
    ):
        """Responses-native ``reasoning={"effort":"low"}`` is the
        canonical /v1/responses opt-in. The helper consults the field
        directly when present on the request shape (used by tests and
        by the route via ``extra_signals``)."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
            reasoning={"effort": "low"},
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_empty_reasoning_dict_does_not_count_as_signal(
        self, _thinking_parser_cfg
    ):
        """Defensive check: ``reasoning={}`` is functionally an
        absent signal — the dict is the OpenAI Responses-spec shape
        for "client supplied nothing meaningful". Do NOT treat it as
        an opt-in (otherwise an SDK that always serializes the field
        as ``{}`` would bypass the auto-disable for every request)."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
            reasoning={},
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_merge_preserves_forward_compat_keys(self, _thinking_parser_cfg):
        """Non-destructive merge contract from M-2 codex round-3 BLOCKING:
        a forward-compat key the client passed must survive the auto-
        disable injection."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs={"future_key": "x"},
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert req.chat_template_kwargs == {
            "future_key": "x",
            "enable_thinking": False,
        }

    def test_idempotent_second_call(self, _thinking_parser_cfg):
        """Once fired, the injected ``enable_thinking=False`` survives
        the precedence check on a second call so the helper does not
        flip-flop. Mirrors the R12-T1F idempotency contract."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        # Second call: precedence check sees the injected key, returns
        # False, leaves the dict alone.
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_extra_signals_reasoning_dict_skips(self, _thinking_parser_cfg):
        """``extra_signals`` is the threading mechanism the /v1/responses
        route uses to surface signals that live on the
        ``ResponsesRequest`` only (the materialized
        ``ChatCompletionRequest`` does not declare the ``reasoning``
        field). Pin that an explicit ``reasoning`` dict on the
        secondary source skips the gate identically to the primary."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        extra = SimpleNamespace(
            reasoning={"effort": "high"},
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert (
            maybe_auto_disable_thinking_for_casual_chat(req, extra_signals=extra)
            is False
        )
        assert req.chat_template_kwargs is None

    def test_extra_signals_reasoning_max_tokens_skips(
        self, _thinking_parser_cfg
    ):
        """``reasoning_max_tokens`` on the secondary signals source
        (mirrors the ``ResponsesRequest.reasoning_max_tokens`` shape)
        also short-circuits the auto-disable."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        extra = SimpleNamespace(
            reasoning=None,
            reasoning_max_tokens=32,
            reasoning_effort=None,
        )
        assert (
            maybe_auto_disable_thinking_for_casual_chat(req, extra_signals=extra)
            is False
        )

    def test_extra_signals_no_double_consult_when_same_object(
        self, _thinking_parser_cfg
    ):
        """When ``extra_signals is request`` the helper should NOT
        double-consult the same object. Pinned via behaviour: a clean
        request passed as both sources still fires the auto-disable
        (no signal on either source → fire)."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert (
            maybe_auto_disable_thinking_for_casual_chat(req, extra_signals=req)
            is True
        )
        assert req.chat_template_kwargs == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# (1b) Helper-level: codex r1 follow-up findings
# ---------------------------------------------------------------------------
#
# Codex round-1 review on the initial implementation surfaced four
# concrete trigger-design issues. The four tests below pin the post-
# fix contract so a future refactor doesn't silently undo any of them.


class TestHelperCodexR1FollowUps:
    def test_tools_present_short_circuits_casual_helper(
        self, _thinking_parser_cfg
    ):
        """Codex r1 MEDIUM #1: ``tool_choice="none"`` was being
        defeated. The tools helper at line 1821 correctly SKIPS the
        ``tool_choice="none"`` branch, but pre-fix the casual helper
        then ran and injected ``enable_thinking=False`` anyway —
        silently turning a Qwen3 prose-on-tool-defs request into
        no-thinking. The casual gate now skips for ANY non-empty
        ``tools`` shape (handled by R12-T1F TOOLS-AUTO) so the casual
        helper governs ONLY the no-tools prose path."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="none",
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        # tools helper skips because tool_choice="none".
        from vllm_mlx.service.helpers import maybe_auto_disable_thinking_for_tools

        assert maybe_auto_disable_thinking_for_tools(req) is False
        # casual helper MUST also skip — the helper now gates on
        # ``tools`` being empty/None.
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_tools_present_without_tool_choice_none_also_short_circuits(
        self, _thinking_parser_cfg
    ):
        """Generalized form of the codex r1 MEDIUM #1 gate: ANY
        non-empty tools list owns the auto-disable decision via R12-T1F
        TOOLS-AUTO, not the casual helper. So even when ``tool_choice``
        is ``"auto"`` / ``"required"`` / a named-function dict, the
        casual helper short-circuits. (R12-T1F TOOLS-AUTO has its own
        coverage for those cases.)"""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="auto",
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs is None

    def test_no_thinking_server_flag_short_circuits(self):
        """Codex r1 NIT #4: when the operator pinned ``--no-thinking``
        server-side, ``_resolve_enable_thinking`` already forces
        ``False``. The auto-disable injection is purely cosmetic noise
        (extra log entry + mutated request) AND on non-qwen3 parsers
        feeds the L-05 spurious-warning shape this helper family is
        trying to avoid. Short-circuit so the operator kill switch
        keeps a single resolution site."""
        from unittest.mock import patch

        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=SimpleNamespace(
                no_thinking=True,
                reasoning_parser_name="qwen3",
            ),
        ):
            req = SimpleNamespace(
                tools=None,
                chat_template_kwargs=None,
                enable_thinking=None,
                reasoning_max_tokens=None,
                reasoning_effort=None,
            )
            assert maybe_auto_disable_thinking_for_casual_chat(req) is False
            assert req.chat_template_kwargs is None

    def test_reasoning_dict_with_null_effort_is_NOT_signal(
        self, _thinking_parser_cfg
    ):
        """Codex r1 MEDIUM #3: ``reasoning={"effort": null}`` is
        EXPLICITLY allowed by the Responses-API schema
        (``_validate_reasoning_dict_effort`` lets ``None`` flow
        through). Pre-fix the helper treated any non-empty dict as
        reasoning intent and skipped the auto-disable; a client that
        round-trips ``reasoning={"effort": null}`` (e.g. an SDK that
        always serializes the key even when no effort is chosen) would
        bypass the budget-burn fix. The fix gates specifically on
        ``effort is not None``."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
            reasoning={"effort": None},
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_reasoning_dict_with_only_summary_is_NOT_signal(
        self, _thinking_parser_cfg
    ):
        """Codex r1 MEDIUM #3 (sibling): ``reasoning={"summary":
        "auto"}`` is a SDK convenience flag (controls whether the
        Responses SDK emits a ``reasoning.summary`` block), NOT a
        reasoning-intent signal. The pre-fix "any non-empty dict"
        gate would have treated this as opt-in and skipped the auto-
        disable. The ``effort``-specific gate now correctly fires."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
            reasoning={"summary": "auto"},
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_marker_set_on_auto_disable_fire(self, _thinking_parser_cfg):
        """Codex r1 MEDIUM #2: the helper tags the request via
        ``_auto_disabled_thinking=True`` so the downstream L-05
        ``enable_thinking_warning_header`` can distinguish a server-
        injected ``enable_thinking=False`` from a client-supplied hint.
        Pinned here so a refactor cannot drop the marker silently."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        assert getattr(req, "_auto_disabled_thinking", False) is True

    def test_marker_NOT_set_on_skip(self, _thinking_parser_cfg):
        """Inverse: when the helper SKIPS (client opted in), the marker
        MUST NOT be set — otherwise a downstream L-05 warning would
        be suppressed for a request where the client DID supply the
        hint."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert getattr(req, "_auto_disabled_thinking", False) is False


class TestL05WarningSuppressedOnAutoDisable:
    """Codex r1 MEDIUM #2: the L-05 warning header MUST distinguish a
    server-injected ``chat_template_kwargs.enable_thinking=False``
    (auto-disable family) from a client-supplied hint. The auto-
    disable helpers tag the request via ``_auto_disabled_thinking``;
    the warning header consults that flag and skips the warning when
    set. Test the contract end-to-end at the warning-header layer."""

    def test_warning_suppressed_when_auto_disable_marker_set(self):
        from vllm_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=None,
            _auto_disabled_thinking=True,
        )
        # A non-qwen3 parser would normally fire the warning — but the
        # auto-disable marker suppresses it because the client never
        # supplied the hint.
        assert enable_thinking_warning_header(req, "deepseek_r1") == {}

    def test_warning_still_fires_when_client_supplied_hint(self):
        """Negative control: WITHOUT the auto-disable marker, the L-05
        warning still fires as before. This protects the pre-fix
        contract: a real client-supplied hint on a non-honoring parser
        still surfaces the silent-drop signal."""
        from vllm_mlx.service.helpers import enable_thinking_warning_header

        req = SimpleNamespace(
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=None,
        )
        assert enable_thinking_warning_header(req, "deepseek_r1") == {
            "X-RapidMLX-Warning": "enable_thinking ignored for parser=deepseek_r1"
        }

    def test_tools_helper_also_sets_marker(self):
        """The same warning-suppression contract MUST apply to the
        R12-T1F tools helper — the marker is set there too, so a
        tools-on-non-qwen3-parser request doesn't get a spurious
        warning."""
        from vllm_mlx.service.helpers import (
            enable_thinking_warning_header,
            maybe_auto_disable_thinking_for_tools,
        )

        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice=None,
            chat_template_kwargs=None,
            enable_thinking=None,
        )
        assert maybe_auto_disable_thinking_for_tools(req) is True
        # Warning suppressed on non-qwen3 parser because the marker
        # is set.
        assert enable_thinking_warning_header(req, "deepseek_r1") == {}


# ---------------------------------------------------------------------------
# (2) /v1/chat/completions: route-level integration
# ---------------------------------------------------------------------------


class _ChatEngine:
    """Thinking-model-shaped mock that captures the kwargs the route
    forwards to ``engine.chat``."""

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


def _make_chat_client(engine: _ChatEngine, *, reasoning_parser_name="qwen3") -> TestClient:
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser_name = reasoning_parser_name

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app)


class TestChatRouteAutoDisableForCasualChat:
    def test_casual_no_preference_auto_disables_thinking(self, _rate_limiter_state):
        """The 0.8.16 brand-new-user simulation repro: thinking-capable
        model + casual chat with no thinking signal → the engine sees
        ``enable_thinking=False``. Pre-fix the engine kwarg was None
        (template default = True) and the model burned the budget
        inside ``<think>``."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [
                    {"role": "user", "content": "In 8 words, what is rapid-mlx?"}
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        kwargs = engine.chat_calls[0]["kwargs"]
        assert kwargs.get("enable_thinking") is False, (
            "casual chat + thinking model + no preference must inject "
            f"enable_thinking=False; engine saw kwargs={kwargs!r}"
        )

    def test_casual_explicit_enable_thinking_true_preserved(
        self, _rate_limiter_state
    ):
        """Explicit opt-in is honored end-to-end — the rescue path
        still applies when the budget is exhausted, but the contract
        is that we never silently override an explicit signal."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is True

    def test_casual_explicit_enable_thinking_false_preserved(
        self, _rate_limiter_state
    ):
        """Explicit opt-out is honored — same resolved value, but the
        preservation contract is exercised for parity."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_casual_top_level_enable_thinking_true_preserved(
        self, _rate_limiter_state
    ):
        """Top-level rapid-mlx convenience knob is honored end-to-end
        identically to the nested kwarg form."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
                "enable_thinking": True,
            },
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is True

    def test_casual_reasoning_max_tokens_preserves_thinking(
        self, _rate_limiter_state
    ):
        """The ``reasoning_max_tokens=N`` signal is an explicit "I want
        reasoning, just bounded" — the auto-disable MUST NOT fire."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_max_tokens": 32,
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        # No auto-disable injection — the engine kwarg is absent (the
        # chat route only forwards ``enable_thinking`` when the
        # resolution is non-None, and no signal here resolves to one).
        assert "enable_thinking" not in kwargs, (
            "reasoning_max_tokens must signal thinking-on; "
            f"engine saw kwargs={kwargs!r}"
        )

    def test_casual_reasoning_effort_preserves_thinking(
        self, _rate_limiter_state
    ):
        """OpenAI-spec ``reasoning_effort`` opt-in keeps thinking ON."""
        engine = _ChatEngine(text="ok")
        client = _make_chat_client(engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert "enable_thinking" not in kwargs

    def test_non_thinking_model_unaffected(self, _rate_limiter_state):
        """No-regression gate: server without a reasoning parser
        (llama / mistral / qwen3-coder) MUST NOT see any auto-disable
        injection. The engine kwarg is absent so the chat-template
        default applies."""
        engine = _ChatEngine(text="hi back")
        client = _make_chat_client(engine, reasoning_parser_name=None)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "max_tokens": 80,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert "enable_thinking" not in kwargs, (
            "non-thinking model must not inject enable_thinking; "
            f"engine saw kwargs={kwargs!r}"
        )

    def test_casual_forward_compat_key_survives_merge(self, _rate_limiter_state):
        """Forward-compat keys passed by the client MUST survive the
        auto-disable merge."""
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
                    "max_tokens": 80,
                    "messages": [{"role": "user", "content": "hi"}],
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
# (3) Interaction with R12-T1F TOOLS-AUTO and R12-M2 strict-json
# ---------------------------------------------------------------------------
#
# When an earlier auto-disable trigger already injected
# ``enable_thinking=False`` on the request, the casual-chat helper must
# short-circuit via the same precedence check (no double-disable, no
# state corruption). Pinned here because the three helpers compose
# without any explicit coordination logic — the gate IS the precedence
# check.


class TestInteractionWithEarlierAutoDisable:
    def test_tools_trigger_short_circuits_casual_helper(self, _thinking_parser_cfg):
        """Sequence the chat route uses: tools helper fires first,
        injecting ``enable_thinking=False``; the casual-chat helper
        runs next and MUST be a no-op (the precedence check sees the
        already-injected key)."""
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice=None,
            chat_template_kwargs=None,
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        # Tools trigger fires.
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        # Casual-chat helper sees the injected key → short-circuit.
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_strict_json_pattern_short_circuits_casual_helper(
        self, _thinking_parser_cfg
    ):
        """Mirror of the R12-M2 plumbing — when the route has already
        merged ``enable_thinking=False`` for strict json_schema, the
        casual-chat helper must short-circuit identically."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": False}

    def test_round_trip_with_resolve_enable_thinking(self, _thinking_parser_cfg):
        """Composition test: after the casual-chat helper fires, the
        standard ``_resolve_enable_thinking`` consult resolves to
        ``False``. Mirrors the R12-T1F round-trip contract."""
        req = ChatCompletionRequest(
            model="qwen3-test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is True
        with patch(
            "vllm_mlx.service.helpers.get_config",
            return_value=SimpleNamespace(no_thinking=False),
        ):
            assert _resolve_enable_thinking(req) is False


# ---------------------------------------------------------------------------
# (4) /v1/responses: route-level integration
# ---------------------------------------------------------------------------


class _ResponsesEngine:
    """Thinking-model-shaped mock for /v1/responses."""

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


def _make_responses_client(
    engine: _ResponsesEngine, *, reasoning_parser_name="qwen3"
) -> TestClient:
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser_name = reasoning_parser_name

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    return TestClient(app)


def _responses_payload(
    *,
    chat_template_kwargs: dict | None = None,
    enable_thinking: bool | None = None,
    reasoning_max_tokens: int | None = None,
    reasoning: dict | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    body: dict = {
        "model": "test-model",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "In 8 words, what is rapid-mlx?"}
                ],
            }
        ],
        "max_output_tokens": 80,
    }
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if reasoning_max_tokens is not None:
        body["reasoning_max_tokens"] = reasoning_max_tokens
    if reasoning is not None:
        body["reasoning"] = reasoning
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


class TestResponsesRouteAutoDisableForCasualChat:
    def test_casual_no_preference_auto_disables_thinking(self, _rate_limiter_state):
        """/v1/responses parity: casual chat (no tools, no strict json,
        no reasoning signal) → engine sees ``enable_thinking=False``."""
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post("/v1/responses", json=_responses_payload())
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls, "engine.chat was not called"
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_casual_explicit_enable_thinking_true_preserved(
        self, _rate_limiter_state
    ):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses", json=_responses_payload(enable_thinking=True)
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is True

    def test_casual_explicit_chat_template_kwargs_false_preserved(
        self, _rate_limiter_state
    ):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_responses_payload(chat_template_kwargs={"enable_thinking": False}),
        )
        assert resp.status_code == 200, resp.text
        assert engine.chat_calls[0]["kwargs"].get("enable_thinking") is False

    def test_casual_reasoning_dict_preserves_thinking(self, _rate_limiter_state):
        """The Responses-native ``reasoning={"effort":"low"}`` opt-in
        must keep thinking ON. This is the surface-specific signal
        the chat route does NOT see — the helper consults it via
        ``extra_signals`` threaded from the route."""
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses", json=_responses_payload(reasoning={"effort": "low"})
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        # No auto-disable injection — the engine kwarg is absent so the
        # template default (thinking ON for non-coder Qwen3) applies.
        assert "enable_thinking" not in kwargs, (
            "reasoning={effort} must signal thinking-on; "
            f"engine saw kwargs={kwargs!r}"
        )

    def test_casual_reasoning_max_tokens_preserves_thinking(
        self, _rate_limiter_state
    ):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_responses_payload(reasoning_max_tokens=32),
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert "enable_thinking" not in kwargs

    def test_casual_reasoning_effort_preserves_thinking(self, _rate_limiter_state):
        engine = _ResponsesEngine(text="ok")
        client = _make_responses_client(engine)
        resp = client.post(
            "/v1/responses",
            json=_responses_payload(reasoning_effort="medium"),
        )
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert "enable_thinking" not in kwargs

    def test_non_thinking_model_unaffected(self, _rate_limiter_state):
        """No-regression: /v1/responses on a non-thinking model is
        untouched."""
        engine = _ResponsesEngine(text="hi back")
        client = _make_responses_client(engine, reasoning_parser_name=None)
        resp = client.post("/v1/responses", json=_responses_payload())
        assert resp.status_code == 200, resp.text
        kwargs = engine.chat_calls[0]["kwargs"]
        assert "enable_thinking" not in kwargs


# ---------------------------------------------------------------------------
# (5) Cross-surface: ResponsesRequest → adapter round-trip with helper
# ---------------------------------------------------------------------------


class TestResponsesAdapterCasualChatRoundTrip:
    def test_adapter_then_helper_injects_on_materialized_request(
        self, _thinking_parser_cfg
    ):
        """ResponsesRequest (casual, no signals) → adapter → ChatRequest
        → casual-chat helper → ``enable_thinking=False``."""
        from vllm_mlx.api.responses_adapter import responses_to_openai

        resp_req = ResponsesRequest(
            model="qwen3-test",
            input="hi",
            max_output_tokens=80,
        )
        chat_req = responses_to_openai(resp_req)
        assert chat_req.chat_template_kwargs is None
        # Pass the ResponsesRequest as extra_signals to mirror the
        # route plumbing. With no reasoning signal on either source,
        # the helper fires.
        assert (
            maybe_auto_disable_thinking_for_casual_chat(
                chat_req, extra_signals=resp_req
            )
            is True
        )
        assert chat_req.chat_template_kwargs == {"enable_thinking": False}

    def test_adapter_reasoning_dict_propagates_via_extra_signals(
        self, _thinking_parser_cfg
    ):
        """When the ResponsesRequest carries ``reasoning={"effort":"low"}``,
        the chat-shape adapter does NOT forward it (the field is not on
        ChatCompletionRequest), but the route's ``extra_signals``
        threading ensures the helper still sees it and short-circuits
        the auto-disable."""
        from vllm_mlx.api.responses_adapter import responses_to_openai

        resp_req = ResponsesRequest(
            model="qwen3-test",
            input="hi",
            max_output_tokens=80,
            reasoning={"effort": "low"},
        )
        chat_req = responses_to_openai(resp_req)
        # Adapter does not forward ``reasoning`` (the field is not on
        # ChatCompletionRequest), and the chat shape's
        # ``reasoning_max_tokens`` is also None.
        assert not hasattr(chat_req, "reasoning") or chat_req.reasoning is None  # noqa: E501
        # Without extra_signals threading the helper would mistakenly
        # fire — confirm that the threaded signal blocks it.
        assert (
            maybe_auto_disable_thinking_for_casual_chat(
                chat_req, extra_signals=resp_req
            )
            is False
        )
        assert chat_req.chat_template_kwargs is None


# ---------------------------------------------------------------------------
# (6) Rescue path positive regression
# ---------------------------------------------------------------------------
#
# The auto-disable is a default-flip for casual chat — it must NOT
# interfere with the rescue-sentinel path for callers who EXPLICITLY
# opted into thinking. That path is independently tested in the
# existing rescue suite; here we just confirm the auto-disable helper
# does not accidentally suppress reasoning when the client opted in.


class TestRescueStillFiresOnExplicitOptIn:
    def test_explicit_thinking_keeps_reasoning_signal_intact(
        self, _thinking_parser_cfg
    ):
        """An explicit ``enable_thinking=True`` survives the auto-disable
        helper untouched — so the downstream rescue path (which fires
        when the model truncates mid-think on an opted-in request)
        keeps its trigger conditions intact."""
        req = SimpleNamespace(
            tools=None,
            chat_template_kwargs={"enable_thinking": True},
            enable_thinking=None,
            reasoning_max_tokens=None,
            reasoning_effort=None,
        )
        assert maybe_auto_disable_thinking_for_casual_chat(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": True}
