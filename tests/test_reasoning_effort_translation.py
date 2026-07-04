# SPDX-License-Identifier: Apache-2.0
"""#448 — route-layer translation of the OpenAI ``reasoning_effort`` knob.

Before this fix the ``reasoning_effort`` field was declared + validated at
the schema layer (garbage 400s at parse time) but ``accepted-but-not-yet-
translated`` at the engine layer: a request with ``reasoning_effort="none"``
still emitted reasoning tokens, blew ``max_tokens`` mid-``<think>``, and had
``REASONING_CUTOFF_SENTINEL`` injected into ``content`` — corrupting the
field an OpenAI-spec agent re-feeds verbatim into the next turn.

``maybe_apply_reasoning_effort`` closes that gap:

  * ``none`` → ``chat_template_kwargs.enable_thinking=False`` (suppress the
    reasoning segment at the source, so the truncation scenario never
    arises).
  * ``minimal / low / medium / high`` → ``reasoning_max_tokens`` tier from
    ``OPENAI_REASONING_EFFORT_TO_MAX_TOKENS``.

The client's explicit, more-specific native knob (``enable_thinking`` /
``reasoning_max_tokens``) always wins over the effort translation. Helper-
level coverage runs on a ``SimpleNamespace`` shim so the gate is exercised
in isolation from Pydantic; the typed-model tests confirm it works on real
``ChatCompletionRequest`` / ``ResponsesRequest`` objects.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_mlx.api.models import (
    OPENAI_REASONING_EFFORT_TO_MAX_TOKENS,
    ChatCompletionRequest,
)
from vllm_mlx.api.responses_adapter import responses_to_openai
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.service.helpers import (
    _client_signalled_reasoning_intent,
    _extract_thinking_from_request,
    _resolve_enable_thinking,
    maybe_apply_reasoning_effort,
    maybe_auto_disable_thinking_for_tools,
)


class TestReasoningIntentPredicate:
    """The shared predicate both auto-disable gates consult."""

    def test_none_of_the_signals(self):
        assert _client_signalled_reasoning_intent(_shim()) is False

    def test_reasoning_max_tokens_signal(self):
        assert _client_signalled_reasoning_intent(_shim(reasoning_max_tokens=1)) is True

    def test_reasoning_effort_signal(self):
        assert _client_signalled_reasoning_intent(_shim(reasoning_effort="low")) is True

    def test_native_responses_reasoning_effort_dict(self):
        assert (
            _client_signalled_reasoning_intent(
                SimpleNamespace(reasoning={"effort": "low"})
            )
            is True
        )

    def test_native_responses_reasoning_null_effort_is_not_a_signal(self):
        assert (
            _client_signalled_reasoning_intent(
                SimpleNamespace(reasoning={"effort": None, "summary": "auto"})
            )
            is False
        )

    def test_none_sources_are_skipped(self):
        assert _client_signalled_reasoning_intent(None, _shim()) is False


def _shim(**overrides):
    """A request shim with the fields the helper reads, all defaulted to
    the "client set nothing" state unless overridden."""
    base = dict(
        reasoning_effort=None,
        reasoning_max_tokens=None,
        chat_template_kwargs=None,
        enable_thinking=None,
        tools=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# (0) The mapping table
# ---------------------------------------------------------------------------


class TestMappingTable:
    def test_none_absent_from_cap_table(self):
        """``none`` maps to enable_thinking=False, NOT a cap, so it must
        not appear in the graded cap table."""
        assert "none" not in OPENAI_REASONING_EFFORT_TO_MAX_TOKENS

    def test_graded_tiers_are_monotonic(self):
        m = OPENAI_REASONING_EFFORT_TO_MAX_TOKENS
        assert m["minimal"] < m["low"] < m["medium"] < m["high"]

    def test_graded_tiers_reuse_anthropic_magnitudes(self):
        """low/medium/high reuse the Anthropic surface's tiers so the same
        effort name yields the same budget across API dialects."""
        from vllm_mlx.api.anthropic_models import (
            ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS,
        )

        for tier in ("low", "medium", "high"):
            assert (
                OPENAI_REASONING_EFFORT_TO_MAX_TOKENS[tier]
                == ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS[tier]
            )


# ---------------------------------------------------------------------------
# (1) Helper: unset / no-op
# ---------------------------------------------------------------------------


class TestHelperNoOp:
    def test_unset_effort_is_noop(self):
        req = _shim()
        assert maybe_apply_reasoning_effort(req) is False
        assert req.chat_template_kwargs is None
        assert req.reasoning_max_tokens is None

    def test_empty_string_effort_is_noop(self):
        # ``""`` is not a valid effort (schema would 400) but the helper
        # must be defensive — a falsy value is treated as unset.
        req = _shim(reasoning_effort="")
        assert maybe_apply_reasoning_effort(req) is False
        assert req.chat_template_kwargs is None


# ---------------------------------------------------------------------------
# (2) Helper: reasoning_effort="none" → enable_thinking=False
# ---------------------------------------------------------------------------


class TestHelperNoneSuppressesThinking:
    def test_none_injects_enable_thinking_false(self):
        req = _shim(reasoning_effort="none")
        assert maybe_apply_reasoning_effort(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        # Resolving thinking now returns False — the load-bearing #448 fix.
        assert _resolve_enable_thinking(req) is False

    def test_none_marks_auto_disabled_to_suppress_warning(self):
        """The flag is server-injected (client set reasoning_effort, not
        chat_template_kwargs.enable_thinking), so the L-05 warning header
        must be suppressed via ``_auto_disabled_thinking``."""
        req = _shim(reasoning_effort="none")
        maybe_apply_reasoning_effort(req)
        assert getattr(req, "_auto_disabled_thinking", False) is True

    def test_none_merge_is_non_destructive(self):
        req = _shim(
            reasoning_effort="none",
            chat_template_kwargs={"custom_forward_compat_key": 7},
        )
        assert maybe_apply_reasoning_effort(req) is True
        assert req.chat_template_kwargs["custom_forward_compat_key"] == 7
        assert req.chat_template_kwargs["enable_thinking"] is False

    def test_none_yields_to_explicit_top_level_enable_thinking(self):
        """Explicit ``enable_thinking=True`` alongside reasoning_effort=none
        is contradictory; the more-specific native field wins."""
        req = _shim(reasoning_effort="none", enable_thinking=True)
        assert maybe_apply_reasoning_effort(req) is False
        assert req.enable_thinking is True
        assert req.chat_template_kwargs is None

    def test_none_yields_to_explicit_kwarg_enable_thinking(self):
        req = _shim(
            reasoning_effort="none",
            chat_template_kwargs={"enable_thinking": True},
        )
        assert maybe_apply_reasoning_effort(req) is False
        assert req.chat_template_kwargs == {"enable_thinking": True}

    def test_none_does_not_yield_to_reasoning_max_tokens_cap(self):
        """``none`` controls the on/off dimension; a cap is orthogonal.
        thinking-off makes the cap moot, so ``none`` still suppresses."""
        req = _shim(reasoning_effort="none", reasoning_max_tokens=64)
        assert maybe_apply_reasoning_effort(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert req.reasoning_max_tokens == 64  # left as-is (moot)


# ---------------------------------------------------------------------------
# (3) Helper: graded effort → reasoning_max_tokens tier
# ---------------------------------------------------------------------------


class TestHelperGradedTiers:
    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
    def test_graded_sets_reasoning_max_tokens_only(self, effort):
        """A graded effort caps reasoning via ``reasoning_max_tokens`` and
        does NOT touch ``enable_thinking`` — it is itself a reasoning-intent
        signal that the auto-disable gates step aside for (see
        TestToolAutoDisableStepsAsideForReasoning)."""
        req = _shim(reasoning_effort=effort)
        assert maybe_apply_reasoning_effort(req) is True
        assert req.reasoning_max_tokens == OPENAI_REASONING_EFFORT_TO_MAX_TOKENS[effort]
        assert req.chat_template_kwargs is None
        assert _extract_thinking_from_request(req) is None
        assert getattr(req, "_auto_disabled_thinking", False) is False

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
    def test_graded_yields_to_explicit_reasoning_max_tokens(self, effort):
        """An explicit client cap wins over the tier."""
        req = _shim(reasoning_effort=effort, reasoning_max_tokens=99)
        assert maybe_apply_reasoning_effort(req) is False
        assert req.reasoning_max_tokens == 99
        assert req.chat_template_kwargs is None

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
    def test_graded_with_explicit_thinking_false_still_caps(self, effort):
        """``enable_thinking=False`` + a graded effort is not contradictory
        the way ``none`` is — the client wants a bounded think budget IF the
        model thinks. The cap still applies; enable_thinking is untouched."""
        req = _shim(reasoning_effort=effort, enable_thinking=False)
        assert maybe_apply_reasoning_effort(req) is True
        assert req.reasoning_max_tokens == OPENAI_REASONING_EFFORT_TO_MAX_TOKENS[effort]
        assert req.enable_thinking is False
        assert req.chat_template_kwargs is None


# ---------------------------------------------------------------------------
# (4) Ordering contract vs the tool auto-disable
# ---------------------------------------------------------------------------


class TestOrderingWithToolAutoDisable:
    def test_none_before_tool_autodisable_no_ops_the_latter(self):
        """The route runs ``maybe_apply_reasoning_effort`` BEFORE the tool
        auto-disable. After ``none`` registers enable_thinking=False, the
        tool auto-disable sees a preference and no-ops — one resolved
        source of truth."""
        req = _shim(reasoning_effort="none", tools=[{"type": "function"}])
        assert maybe_apply_reasoning_effort(req) is True
        # Preference is now set, so the tool auto-disable is a no-op.
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert _resolve_enable_thinking(req) is False

    def test_graded_high_with_tools_not_forced_off(self):
        """codex #1009 r1 MAJOR: ``reasoning_effort="high"`` + tools must NOT
        be silently turned off by the tool auto-disable. The graded path
        sets the reasoning_max_tokens cap (a reasoning-intent signal), so
        the tool auto-disable steps aside and enable_thinking is NOT forced
        to False — the model keeps its template-default reasoning, bounded
        by the cap."""
        req = _shim(reasoning_effort="high", tools=[{"type": "function"}])
        assert maybe_apply_reasoning_effort(req) is True
        assert maybe_auto_disable_thinking_for_tools(req) is False
        # not clobbered to False — stays template default (None here)
        assert _extract_thinking_from_request(req) is None
        assert req.reasoning_max_tokens == OPENAI_REASONING_EFFORT_TO_MAX_TOKENS["high"]


class TestToolAutoDisableStepsAsideForReasoning:
    """codex #1009 r1 MAJOR — the tool auto-disable must treat a raw
    reasoning-intent signal as a preference and step aside, symmetric with
    the casual-chat gate, even when ``maybe_apply_reasoning_effort`` has not
    run yet (defensive: the signal alone is enough)."""

    def test_tools_plus_raw_reasoning_effort_steps_aside(self):
        req = _shim(reasoning_effort="high", tools=[{"type": "function"}])
        assert maybe_auto_disable_thinking_for_tools(req) is False
        assert _extract_thinking_from_request(req) is None

    def test_tools_plus_raw_reasoning_max_tokens_steps_aside(self):
        req = _shim(reasoning_max_tokens=64, tools=[{"type": "function"}])
        assert maybe_auto_disable_thinking_for_tools(req) is False

    def test_tools_without_reasoning_signal_still_auto_disables(self):
        """No-regression: plain tools + no reasoning signal + no thinking
        preference still auto-disables (the original R12-T1F contract)."""
        req = _shim(tools=[{"type": "function"}])
        assert maybe_auto_disable_thinking_for_tools(req) is True
        assert req.chat_template_kwargs == {"enable_thinking": False}


# ---------------------------------------------------------------------------
# (5) Typed models — field declared + helper works on real objects
# ---------------------------------------------------------------------------


class TestTypedModels:
    def test_chat_request_none_resolves_thinking_false(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="none",
        )
        assert maybe_apply_reasoning_effort(req) is True
        assert _resolve_enable_thinking(req) is False

    def test_chat_request_high_sets_cap(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
        assert maybe_apply_reasoning_effort(req) is True
        assert req.reasoning_max_tokens == OPENAI_REASONING_EFFORT_TO_MAX_TOKENS["high"]

    def test_chat_request_garbage_effort_400s_at_schema(self):
        """The schema layer stays the hard surface — garbage never reaches
        the translation helper."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="hgih",
            )

    def test_responses_request_none_resolves_thinking_false(self):
        req = ResponsesRequest(
            model="m",
            input="hi",
            reasoning_effort="none",
        )
        assert maybe_apply_reasoning_effort(req) is True
        assert _resolve_enable_thinking(req) is False


class TestResponsesAdapterForwardsEffort:
    """codex #1009 r2 MAJOR — ``responses_to_openai`` dropped
    ``reasoning_effort`` / native ``reasoning.effort`` so the route-layer
    translation silently no-op'd on /v1/responses. The adapter now collapses
    both surfaces onto the materialized ChatCompletionRequest's
    ``reasoning_effort`` field so ``maybe_apply_reasoning_effort`` fires."""

    def test_top_level_reasoning_effort_forwarded(self):
        oai = responses_to_openai(
            ResponsesRequest(model="m", input="hi", reasoning_effort="none")
        )
        assert oai.reasoning_effort == "none"
        # and the translation now works end-to-end on the materialized req
        assert maybe_apply_reasoning_effort(oai) is True
        assert _resolve_enable_thinking(oai) is False

    def test_native_reasoning_effort_dict_forwarded(self):
        oai = responses_to_openai(
            ResponsesRequest(model="m", input="hi", reasoning={"effort": "high"})
        )
        assert oai.reasoning_effort == "high"
        assert maybe_apply_reasoning_effort(oai) is True
        assert oai.reasoning_max_tokens == OPENAI_REASONING_EFFORT_TO_MAX_TOKENS["high"]

    def test_top_level_wins_over_nested(self):
        oai = responses_to_openai(
            ResponsesRequest(
                model="m",
                input="hi",
                reasoning_effort="low",
                reasoning={"effort": "high"},
            )
        )
        assert oai.reasoning_effort == "low"

    def test_null_nested_effort_is_not_forwarded(self):
        oai = responses_to_openai(
            ResponsesRequest(
                model="m", input="hi", reasoning={"effort": None, "summary": "auto"}
            )
        )
        assert oai.reasoning_effort is None

    def test_strict_json_branch_sees_reasoning_intent_after_forward(self):
        """codex #1009 r3 MAJOR — the /v1/responses strict-json auto-disable
        guards on ``_client_signalled_reasoning_intent(openai_request)``.
        Because the adapter forwards effort onto the materialized request,
        that predicate is True for a strict-json + graded-effort request, so
        the branch steps aside instead of force-disabling thinking."""
        oai = responses_to_openai(
            ResponsesRequest(model="m", input="hi", reasoning={"effort": "high"})
        )
        assert _client_signalled_reasoning_intent(oai) is True
