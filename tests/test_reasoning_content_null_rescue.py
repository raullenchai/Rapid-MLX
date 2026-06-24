# SPDX-License-Identifier: Apache-2.0
"""Reasoning-cutoff sentinel notice: regression tests (H-01 / R-01 / issue #858).

Background
----------
Every reasoning model (qwen3, deepseek_r1, phi-4-mini-reasoning, glm4,
gemma4, vibethinker, …) can be called with a low ``max_tokens`` budget
that cuts generation off BEFORE ``</think>`` (or the harmony
final-channel marker). The parser-wide rule pinned by D-STOP-THINK +
D-HARMONY-LEAK is "route everything to ``reasoning_content`` and leave
``content`` null/empty" — the in-progress thought trace is NOT the final
answer, so promoting it to ``content`` would ship byte-identical bytes
in both fields (the leak shape those PRs explicitly closed).

History
-------
* H-01 (PR #802, 0.8.3) introduced an opt-OUT sentinel that injected the
  literal string ``[truncated — reasoning incomplete; raise max_tokens]``
  into ``content`` by default so SDK consumers saw something instead of
  an empty bubble.
* R-01 (PR #815, 0.8.5 dogfood) flipped the policy to opt-IN on
  structured-purity rationale.
* Issue #858 (this PR, 0.8.12) reverts R-01 back to PR #802 semantics:
  GUI clients (rapid-desktop, OpenAI SDK consumers, OpenWebUI compat)
  showed empty bubbles under R-01's default-off because they only render
  ``message.content`` and don't walk the structured ``finish_reason``
  field. The sentinel is the user-visible cue.

Default (issue #858) is now ON.
``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` (or ``0`` / ``false`` /
``no`` / ``off``) opts out for power callers that want strict-null. The
helper lives in ``vllm_mlx.service.helpers._apply_reasoning_cutoff_notice``
and is the single source of truth for ``/v1/chat/completions``,
``/v1/responses``, and ``/v1/messages``.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.service.helpers import (
    REASONING_CUTOFF_SENTINEL,
    _apply_reasoning_cutoff_notice,
    _rescue_silent_drop_from_reasoning,
)

# ──────────────────────────────────────────────────────────────────────
# Unit tests for the helper itself
# ──────────────────────────────────────────────────────────────────────


class TestApplyReasoningCutoffNotice:
    """Unit-level predicate tests on ``_apply_reasoning_cutoff_notice``.

    The helper owns every predicate (env gate — default ON, opt out via
    ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``; finish_reason;
    content emptiness; reasoning presence; tool-call gate). These tests
    pin the truth table so route call sites can stay trivial and any
    future drift between surfaces fails here first.
    """

    def test_opt_out_env_disables_sentinel(self, monkeypatch):
        """Issue #858 opt-out: when
        ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` is set
        explicitly, the helper must be a no-op so power callers that
        want strict-null behaviour (the R-01 contract) can still get
        it. Disable-spelling parity is covered by
        ``test_env_disable_values_keep_sentinel_disabled`` below."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None, (
            f"issue #858 opt-out: explicit 'disabled' must keep the "
            f"helper as a no-op; got {result!r}"
        )

    def test_default_is_enabled_when_env_unset_regression_858(self, monkeypatch):
        """Regression pin for issue #858: with the env var UNSET (the
        default user experience in rapid-desktop and vanilla SDK
        callers), the helper MUST fire the sentinel on length-cut
        mid-think. PR #815 flipped the default to OFF, which produced
        empty bubbles in every GUI client — issue #858 reverts that
        back to PR #802 (H-01) semantics: default ON, opt-out via
        ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``."""
        monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL, (
            f"issue #858: unset env var must enable the sentinel "
            f"(default ON, PR #802 behaviour restored); got {result!r}"
        )

    @pytest.mark.parametrize(
        "enable_alias",
        ["1", "true", "TRUE", "True", "on", "yes", "enabled"],
    )
    def test_explicit_enable_aliases_keep_sentinel_on(self, monkeypatch, enable_alias):
        """Explicit enable aliases: with the default already ON since
        issue #858, these truthy spellings keep the sentinel on
        explicitly (rather than re-enabling it from off). Case-
        insensitive matching is locked so ``"True"`` and ``"TRUE"``
        both work — env vars are commonly provided in mixed case by
        shell wrappers. Useful as a defensive default for callers
        that want to be explicit about the on-state regardless of
        future default flips."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", enable_alias)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="Let me think about 17*23... 17*20=340, 17*3=",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL, (
            f"explicit enable alias {enable_alias!r} must keep the sentinel on"
        )

    @pytest.mark.parametrize(
        "disable_value",
        # Documented disable spellings. Issue #858 closes the DISABLE
        # set; only these values opt out of the default-on sentinel.
        ["0", "false", "FALSE", "no", "off", "disabled", "DISABLED"],
    )
    def test_env_disable_values_keep_sentinel_disabled(
        self, monkeypatch, disable_value
    ):
        """Issue #858 closes the disable set: only
        ``{0, false, no, off, disabled}`` (case-insensitive) opts out
        of the default-on sentinel. Power callers that want strict-null
        behaviour set the env var to one of these spellings."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", disable_value)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<truncated thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None, (
            f"env value {disable_value!r} must opt out of the sentinel"
        )

    @pytest.mark.parametrize(
        "unknown_value",
        # Arbitrary unknown strings (NOT in the disable set) — issue #858
        # default-on contract: anything outside the disable set leaves
        # the sentinel enabled. The empty string also flows through to
        # default-on so a misconfigured ``RAPID_MLX_REASONING_CUTOFF_NOTICE=``
        # does not silently swallow the user-visible cue.
        ["", "anything", "garbage", "maybe"],
    )
    def test_env_unknown_values_keep_sentinel_enabled(self, monkeypatch, unknown_value):
        """Issue #858: anything outside the disable set — including
        the empty string and arbitrary unrecognised values — keeps the
        sentinel ENABLED (default-on). This is the safe default for
        GUI clients."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", unknown_value)
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<truncated thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL, (
            f"env value {unknown_value!r} must leave the sentinel enabled"
        )

    # ----- enabled branch: the H-01 / issue #858 truth table -----
    #
    # All gates below run with the env var explicitly set to a truthy
    # value, so they exercise the sentinel path. The same predicates
    # govern when the sentinel fires whether the path was reached via
    # the default (issue #858 default-on) or via an explicit enable
    # alias — these tests pin the explicit-enable branch.

    def test_enabled_fires_on_length_cut_mid_think(self, monkeypatch):
        """Exact H-01 / issue #858 production failure shape with the
        env var explicitly enabled: ``finish_reason="length"`` + empty
        ``content`` + non-empty reasoning + no tool calls. The sentinel
        surfaces in ``content``; reasoning stays as-is."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="Let me think about 17*23... 17*20=340, 17*3=",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_enabled_fires_when_content_is_empty_string(self, monkeypatch):
        """Empty-string ``content`` (downstream sanitization collapsed
        the buffer to ``""``) is treated the same as ``None`` with the
        sentinel enabled — clients render an empty bubble either way."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content="",
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_enabled_fires_when_content_is_whitespace_only(self, monkeypatch):
        """Whitespace-only ``content`` looks identical to clients with
        the sentinel enabled — they see an empty bubble. Match the
        same semantics as the silent-drop helper's whitespace-only
        check."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content="   \n\t",
            reasoning_text="<incomplete thought>",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == REASONING_CUTOFF_SENTINEL

    def test_enabled_noop_when_content_is_populated(self, monkeypatch):
        """Happy path with sentinel enabled: model produced a real
        answer. The sentinel must NEVER overwrite legitimate content.
        Closed ``<think>...</think>answer`` flows must come through
        unchanged even when the env knob is on."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content="The answer is 391.",
            reasoning_text="17*23 = 17*(20+3) = 340 + 51 = 391",
            tool_calls=None,
            finish_reason="length",
        )
        assert result == "The answer is 391."

    def test_enabled_noop_on_stop_finish_d_stop_think_regression_guard(
        self, monkeypatch
    ):
        """D-STOP-THINK regression guard, even with sentinel enabled:
        stop-string cut mid-think keeps strict-null behaviour. The
        sentinel ONLY fires on ``finish_reason="length"``."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="<incomplete thought before stop string>",
            tool_calls=None,
            finish_reason="stop",
        )
        assert result is None

    def test_enabled_noop_on_tool_calls_finish(self, monkeypatch):
        """OpenAI spec: tool-call turns ship ``content=None``. Sentinel
        must not interfere even with the sentinel enabled."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="I should call get_weather",
            tool_calls=[{"id": "x", "type": "function"}],
            finish_reason="tool_calls",
        )
        assert result is None

    def test_enabled_noop_when_tool_calls_present_even_on_length(self, monkeypatch):
        """Even on ``finish_reason="length"``, a tool-call turn ships
        ``content=None``. The tool-call gate is independent of
        finish_reason."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="planning the call...",
            tool_calls=[{"id": "x", "type": "function"}],
            finish_reason="length",
        )
        assert result is None

    def test_enabled_noop_when_reasoning_is_none(self, monkeypatch):
        """If the model produced no reasoning either (truly empty
        response), there's nothing semantically rescue-worthy. We do
        NOT fabricate a "raise max_tokens" hint when the upstream
        bug is "model emitted zero tokens" — that's a different bug
        class."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=None,
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None

    def test_enabled_noop_when_reasoning_is_whitespace_only(self, monkeypatch):
        """Whitespace-only reasoning — same as ``None`` for rescue
        purposes."""
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        result = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text="   \n\t",
            tool_calls=None,
            finish_reason="length",
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────
# Parser-wide assembly tests: drive _finalize_content_and_reasoning +
# _rescue_silent_drop_from_reasoning + _apply_reasoning_cutoff_notice
# in the same orchestration the chat route runs, against every
# ``<think>``-style parser family. Pins the parser-INDEPENDENT contract
# the helper guards — under the opt-out env var the sentinel never
# fires; under default (issue #858 default-on) it fires uniformly
# across parsers.
# ──────────────────────────────────────────────────────────────────────


def _finalize_route_assembly(
    *,
    raw_text: str,
    reasoning_parser,
    finish_reason: str,
    enable_thinking: bool | None = None,
    engine_reasoning_text: str = "",
):
    """Mirror the chat route's final-content assembly sequence:

    1. ``_finalize_content_and_reasoning`` — extracts reasoning/content
    2. ``_rescue_silent_drop_from_reasoning`` — issue #569 rescue
    3. ``_apply_reasoning_cutoff_notice`` — cutoff sentinel (default
       ON; opt out via ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled``)

    Returns ``(final_content, reasoning_text)`` exactly as the route
    layer would set them on the AssistantMessage.
    """
    from vllm_mlx.api.utils import (
        clean_output_text,
        sanitize_output,
        strip_thinking_tags,
    )
    from vllm_mlx.service.helpers import (
        _finalize_content_and_reasoning,
        _rescue_silent_drop_from_reasoning,
    )

    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=raw_text,
        cleaned_text=raw_text,
        tool_calls=[],
        reasoning_parser=reasoning_parser,
        engine_reasoning_text=engine_reasoning_text,
        enable_thinking=enable_thinking,
    )
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
    final_content = _rescue_silent_drop_from_reasoning(
        final_content,
        reasoning_text,
        tool_calls=[],
        finish_reason=finish_reason,
        raw_text=raw_text,
        reasoning_is_case4=False,
    )
    final_content = _apply_reasoning_cutoff_notice(
        final_content,
        reasoning_text,
        [],
        finish_reason,
    )
    return final_content, reasoning_text


def _parser_cases():
    """Every ``<think>``-tag reasoning-parser family alongside shared
    raw-text shapes that produce the four scenarios this helper cares
    about: open-only (mid-think), closed+truncated-answer, stop-cut
    mid-think, and the happy-path closed pair.

    Each entry: ``(name, parser_class, raw_open_only, raw_closed_trunc,
    raw_stop_cut_mid, raw_happy)``."""
    from vllm_mlx.reasoning.deepseek_r1_parser import (
        DeepSeekR1ReasoningParser,
        VibeThinkerReasoningParser,
    )
    from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    think_open_only = "<think>Let me think about 17*23. 17 * 20 = 340. 17 * 3 ="
    think_closed_trunc = "<think>17 * 23 = 391.</think>The answer is 39"
    think_stop_mid = "<think>The user wants weather. Let me check"
    think_happy = "<think>17*23 = 391</think>The answer is 391."

    return [
        (
            "qwen3",
            Qwen3ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "deepseek_r1",
            DeepSeekR1ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "vibethinker",
            VibeThinkerReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
        (
            "glm4",
            Glm4ReasoningParser,
            think_open_only,
            think_closed_trunc,
            think_stop_mid,
            think_happy,
        ),
    ]


@pytest.fixture(params=_parser_cases(), ids=lambda p: p[0])
def parser_case(request):
    name, cls, raw_open_only, raw_closed_trunc, raw_stop_mid, raw_happy = request.param
    return {
        "name": name,
        "parser": cls(),
        "raw_open_only": raw_open_only,
        "raw_closed_trunc": raw_closed_trunc,
        "raw_stop_mid": raw_stop_mid,
        "raw_happy": raw_happy,
    }


class TestParserWideLengthCutMidThinkOptOut:
    """Parser-wide opt-out contract: with the env knob set to
    ``disabled``, length-cut mid-think must produce strict-null content
    (NO sentinel) for every reasoning parser family. The structured
    truncation signal (``finish_reason="length"`` + ``reasoning_content``)
    is the cue for power callers that take the opt-out branch.

    Each test under this class runs with the env knob explicitly set to
    ``disabled`` via the autouse fixture below. The default-on parser-
    wide contract (issue #858) is covered by
    ``TestParserWideLengthCutMidThinkEnabled`` further down.
    """

    @pytest.fixture(autouse=True)
    def _opt_out_env(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")

    def test_length_cut_mid_think_no_sentinel_when_opted_out(self, parser_case):
        """Opt-out contract: length-cut with an unclosed reasoning
        block must NOT inject the sentinel into ``content`` when the
        env var is set to ``disabled``. Reasoning stays populated so
        clients that read ``reasoning_content`` (or the equivalent
        ``thinking`` block on the Anthropic surface) see the trace."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_open_only"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        # Strict-null contract: no synthetic text injection.
        assert "truncated" not in (content or "").lower(), (
            f"opt-out [{parser_case['name']}]: env=disabled must not "
            f"inject the truncated sentinel; got content={content!r}"
        )
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"opt-out [{parser_case['name']}]: env=disabled must not "
            f"surface the sentinel literal; got content={content!r}"
        )
        # ``reasoning_content`` keeps the trace.
        assert reasoning is not None and "17" in reasoning, (
            f"reasoning_content must remain populated for "
            f"{parser_case['name']}; got {reasoning!r}"
        )

    def test_length_cut_after_close_preserves_partial_answer(self, parser_case):
        """Happy(ish) path: the reasoning block CLOSED before the
        truncation point. ``content`` carries the partial answer and
        must NOT be overwritten with anything — sentinel never gets a
        chance to fire (gate: ``content`` is non-empty)."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_closed_trunc"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert content == "The answer is 39", (
            f"length-cut AFTER reasoning-close [{parser_case['name']}]: "
            f"must preserve partial answer; got content={content!r}"
        )
        assert reasoning is not None and "391" in reasoning

    def test_stop_cut_mid_think_strict_null(self, parser_case):
        """D-STOP-THINK contract still holds: stop-cut mid-think never
        carries the sentinel. Under the opt-out branch this is also
        strict-null, but the sentinel-absence assertion is what
        D-STOP-THINK pins."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_stop_mid"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"[{parser_case['name']}]: sentinel must not fire on "
            f"finish_reason=stop; got content={content!r}"
        )
        assert reasoning is not None and reasoning.strip()

    def test_happy_path_full_generation_preserves_answer(self, parser_case):
        """Full closed reasoning + answer flow: content has the answer,
        reasoning has the trace. Sentinel must not fire (gate: content
        non-empty)."""
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_happy"],
            reasoning_parser=parser_case["parser"],
            finish_reason="stop",
        )
        assert content == "The answer is 391.", (
            f"happy-path [{parser_case['name']}]: content must equal answer; "
            f"got {content!r}"
        )
        assert reasoning is not None and "391" in reasoning


class TestParserWideLengthCutMidThinkEnabled:
    """Sentinel-enabled path (default-on since issue #858, or explicit
    ``RAPID_MLX_REASONING_CUTOFF_NOTICE=1``): every reasoning-parser
    family surfaces the sentinel uniformly on length-cut mid-think.
    Pinned with an explicit ``"1"`` here so the test is robust to any
    future default flip.
    """

    def test_enabled_length_cut_mid_think_produces_sentinel(
        self, parser_case, monkeypatch
    ):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        content, reasoning = _finalize_route_assembly(
            raw_text=parser_case["raw_open_only"],
            reasoning_parser=parser_case["parser"],
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL, (
            f"enabled [{parser_case['name']}]: length-cut mid-think must "
            f"surface sentinel; got content={content!r}"
        )
        assert reasoning is not None and "17" in reasoning


# ──────────────────────────────────────────────────────────────────────
# Gemma4 + Harmony: engine-routed reasoning shape
# ──────────────────────────────────────────────────────────────────────


class TestGemma4HarmonyEngineRouted:
    """Engine-routed reasoning families (gemma4, harmony) reach the
    helper via a different upstream path (the OutputRouter strips
    channel markers before the route's ``cleaned_text`` is computed).
    The helper-level contract is identical though: opt-out env value →
    sentinel never fires; default-on → sentinel surfaces.
    """

    def test_opt_out_no_sentinel_on_empty_cleaned_text(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=("The user wants to know about the weather. Let me think"),
            tool_calls=None,
            finish_reason="length",
        )
        assert content is None

    def test_default_on_surfaces_sentinel_on_empty_cleaned_text(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=("The user wants to know about the weather. Let me think"),
            tool_calls=None,
            finish_reason="length",
        )
        assert content == REASONING_CUTOFF_SENTINEL

    def test_harmony_analysis_only_opt_out(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
        content = _apply_reasoning_cutoff_notice(
            final_content=None,
            reasoning_text=(
                "The user wants weather. Let me think about which tool to call"
            ),
            tool_calls=None,
            finish_reason="length",
        )
        assert content is None


class TestRescueSilentDropFromReasoning:
    def test_case4_length_without_prompt_thinking_rescues_content(self):
        """A non-thinking answer truncated by max_tokens must not be
        silently dropped just because a Case-4 parser put it in reasoning."""
        content = _rescue_silent_drop_from_reasoning(
            final_content=None,
            reasoning_text="The answer is 12",
            tool_calls=None,
            finish_reason="length",
            raw_text="The answer is 12",
            reasoning_is_case4=True,
            prompt_thinking_active=False,
        )
        assert content == "The answer is 12"

    def test_case4_length_with_prompt_thinking_stays_suppressed(self):
        """Prompt-injected max_tokens mid-think remains a structured
        reasoning truncation, not user-visible content."""
        content = _rescue_silent_drop_from_reasoning(
            final_content=None,
            reasoning_text="5+7 equals 12",
            tool_calls=None,
            finish_reason="length",
            raw_text="5+7 equals 12",
            reasoning_is_case4=True,
            prompt_thinking_active=True,
        )
        assert content is None


# ──────────────────────────────────────────────────────────────────────
# Streaming SSE: opt-out env value → no sentinel emitted; default-on
# (issue #858) → one final-chunk event carrying the literal sentinel.
# ──────────────────────────────────────────────────────────────────────


class _StreamEngine:
    """Minimal streaming engine for the streaming-surface tests.

    Emits text-mode (``channel=None``) deltas so the route's reasoning
    parser is on the hot path — the test then configures a Qwen3 parser
    via ``cfg.reasoning_parser`` and the deltas start with a literal
    ``<think>`` opener so the parser's ``_saw_any_tag`` flag flips and
    the streaming rescue's truncated-think gate fires. That suppression
    is precisely the case the helper was filed for.

    Set ``include_open_tag=False`` to opt out of the ``<think>`` opener
    for tests that want to assert the plain silent-drop rescue path
    (no parser flag flip)."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(
        self,
        reasoning_deltas: list[str],
        finish_reason: str = "length",
        include_open_tag: bool = True,
    ):
        self._deltas = reasoning_deltas
        self._finish_reason = finish_reason
        self._include_open_tag = include_open_tag
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        accumulated = ""
        deltas = list(self._deltas)
        if self._include_open_tag and deltas:
            deltas[0] = "<think>" + deltas[0]
        for i, delta in enumerate(deltas):
            accumulated += delta
            is_last = i == len(deltas) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=is_last,
                finish_reason=self._finish_reason if is_last else None,
                channel=None,
            )


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def _stream_post(
    reasoning_deltas,
    finish_reason="length",
    include_open_tag=True,
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    cfg.engine = _StreamEngine(
        reasoning_deltas,
        finish_reason=finish_reason,
        include_open_tag=include_open_tag,
    )
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
        assert resp.status_code == 200, resp.text
        return _parse_sse(resp.text)
    finally:
        reset_config()


def test_streaming_opt_out_no_sentinel_in_terminal_chunk(monkeypatch):
    """Opt-out contract on the streaming surface: when reasoning
    streamed but no content streamed AND ``finish_reason="length"`` AND
    the env var is set to ``disabled``, NO sentinel must appear in any
    ``delta.content`` event. Per-delta ``reasoning_content`` chunks
    still flow during the loop."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    events = _stream_post(
        ["Let me think about ", "the weather query. ", "I should call"],
        finish_reason="length",
    )
    assert events, "expected at least the terminal SSE chunk"

    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            content = d.get("content")
            if content:
                assert "truncated" not in content.lower(), (
                    f"opt-out streaming: env=disabled, no chunk may "
                    f"carry the truncated sentinel; "
                    f"got delta.content={content!r}"
                )
                assert content != REASONING_CUTOFF_SENTINEL

    # Reasoning content stream is independent and still flows.
    streamed_reasoning = ""
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            if d.get("reasoning_content"):
                streamed_reasoning += d["reasoning_content"]
    assert "weather query" in streamed_reasoning, (
        "per-delta reasoning_content chunks must still flow under opt-out; "
        f"got streamed={streamed_reasoning!r}"
    )


def test_streaming_enabled_emits_sentinel_in_terminal_chunk(monkeypatch):
    """Sentinel-enabled streaming (env=1): the terminal chunk's
    ``delta.content`` carries the sentinel string. Per-delta
    reasoning_content unchanged."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    events = _stream_post(
        ["Let me think about ", "the weather query. ", "I should call"],
        finish_reason="length",
    )
    assert events
    terminal_events = [
        e
        for e in events
        if any(ch.get("finish_reason") is not None for ch in e.get("choices", []))
    ]
    assert terminal_events
    terminal = terminal_events[-1]
    delta = terminal["choices"][0].get("delta", {})
    assert delta.get("content") == REASONING_CUTOFF_SENTINEL, (
        f"enabled streaming: terminal chunk must carry sentinel; "
        f"got {delta.get('content')!r}"
    )


def test_streaming_enabled_sentinel_is_single_event_not_per_token(monkeypatch):
    """When sentinel is enabled, the sentinel surfaces as ONE
    final-chunk event, not per-token. Counting ``content`` deltas
    across the whole stream MUST yield exactly one chunk carrying the
    sentinel — never split."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    events = _stream_post(
        ["Buffer ", "more ", "thought"],
        finish_reason="length",
    )

    sentinel_chunks = []
    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            if d.get("content"):
                sentinel_chunks.append(d["content"])

    assert sentinel_chunks == [REASONING_CUTOFF_SENTINEL], (
        f"sentinel must surface as exactly one final-chunk content "
        f"delta carrying the literal sentinel string; got {sentinel_chunks!r}"
    )


def test_streaming_stop_cut_mid_think_no_sentinel_d_stop_think_guard(monkeypatch):
    """SSE D-STOP-THINK regression guard: stop-string cut mid-think
    keeps ``delta.content=None`` on every chunk. The sentinel ONLY
    fires on ``finish_reason="length"`` even when the env var
    explicitly enables it."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    events = _stream_post(
        ["Buffer ", "more ", "thought"],
        finish_reason="stop",
    )

    for ev in events:
        for choice in ev.get("choices", []):
            d = choice.get("delta") or {}
            assert not d.get("content"), (
                f"D-STOP-THINK: no content allowed on stop-cut mid-think; "
                f"got delta={d!r}"
            )


def test_streaming_happy_path_no_sentinel_when_content_streamed(monkeypatch):
    """Happy-path guard (opt-out branch): when content WAS streamed
    during the loop (normal turn that closed ``</think>`` and produced
    an answer), the assembled content stream MUST equal the original
    output. No sentinel sneaks in via the length-finish path when
    content was actually emitted."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.engine.base import GenerationOutput
    from vllm_mlx.routes.chat import router as chat_router

    class _ContentEngine:
        preserve_native_tool_format = False
        is_mllm = False
        supports_guided_generation = False
        tokenizer = None

        def build_prompt(self, messages, tools=None, enable_thinking=None):
            return "PROMPT"

        async def stream_chat(self, messages, **kwargs):
            for i, txt in enumerate(["Hello ", "world."]):
                is_last = i == 1
                yield GenerationOutput(
                    text=txt,
                    new_text=txt,
                    prompt_tokens=4,
                    completion_tokens=i + 1,
                    finished=is_last,
                    finish_reason="length" if is_last else None,
                    channel=None,
                )

    cfg = reset_config()
    cfg.engine = _ContentEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": True,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        events = _parse_sse(resp.text)
        streamed_content = ""
        for ev in events:
            for choice in ev.get("choices", []):
                d = choice.get("delta") or {}
                if d.get("content"):
                    streamed_content += d["content"]
        assert streamed_content == "Hello world.", (
            f"happy-path content stream must equal engine output; "
            f"got {streamed_content!r}"
        )
        assert REASONING_CUTOFF_SENTINEL not in streamed_content
    finally:
        reset_config()


# ──────────────────────────────────────────────────────────────────────
# Route-wiring tests: assert the helper is CALLED from every route
# (source-grep guard against accidental deletion of the call site) and
# both the default-on (issue #858 regression pin) and opt-out contracts
# hold end-to-end on every transport.
# ──────────────────────────────────────────────────────────────────────


def _route_source(module_name: str) -> str:
    import importlib
    import inspect

    mod = importlib.import_module(module_name)
    return inspect.getsource(mod)


def test_anthropic_route_helper_call_site_present():
    """Pins that the Anthropic ``/v1/messages`` route CALLS the helper,
    so the env-knob behaviour applies uniformly to Anthropic SDK
    consumers. Source-level grep guards against a future refactor that
    deletes the call site but leaves the import intact."""
    src = _route_source("vllm_mlx.routes.anthropic")
    assert "_apply_reasoning_cutoff_notice(" in src, (
        "Anthropic route must invoke the cutoff sentinel helper "
        "(not just import it) — single source of truth"
    )


def test_responses_route_helper_call_site_present():
    """Same call-site grep for ``/v1/responses``."""
    src = _route_source("vllm_mlx.routes.responses")
    assert "_apply_reasoning_cutoff_notice(" in src, (
        "Responses route must invoke the cutoff sentinel helper "
        "(not just import it) — single source of truth"
    )


def test_chat_route_helper_call_site_present():
    """Same call-site grep for ``/v1/chat/completions``. The chat
    module hosts BOTH the non-stream and stream paths, so the helper
    must be invoked twice."""
    src = _route_source("vllm_mlx.routes.chat")
    invocation_count = src.count("_apply_reasoning_cutoff_notice(")
    assert invocation_count >= 2, (
        "Chat route must invoke the cutoff sentinel helper from "
        "BOTH the non-stream and stream paths; "
        f"found {invocation_count} call site(s)"
    )


class _EngineLengthCutMidThink:
    """Shared mock engine for the route-wiring behavioral tests.

    Returns a single ``GenerationOutput`` with an unclosed ``<think>``
    opener and ``finish_reason="length"`` — the exact production
    failure shape the helper was filed against. Used across chat /
    responses / anthropic e2e wiring tests so the contract is identical
    on every surface.
    """

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    chat_template = ""

    def __init__(self):
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    def estimate_prompt_tokens(self, prompt):
        return 4

    async def chat(self, messages, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text="<think>Computing 17*23 step by step",
            new_text="<think>Computing 17*23 step by step",
            prompt_tokens=4,
            completion_tokens=12,
            finished=True,
            finish_reason="length",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        yield await self.chat(messages, **kwargs)


def _seed_length_cut_engine(cfg):
    """Common cfg shape for the route-wiring behavioral tests:
    qwen3 reasoning parser + length-cut mock engine."""
    from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg.engine = _EngineLengthCutMidThink()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.reasoning_parser_name = "qwen3"


def test_chat_route_opt_out_no_sentinel_on_length_cut(monkeypatch):
    """Opt-out e2e contract for ``/v1/chat/completions`` non-streaming:
    when ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` is set, a
    length-cut mid-think envelope must NOT carry the sentinel. The
    structured truncation signal (``finish_reason="length"`` +
    ``reasoning_content``) is the cue under this branch."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        msg = payload["choices"][0]["message"]
        content = msg.get("content")
        assert content != REASONING_CUTOFF_SENTINEL, (
            f"opt-out e2e (env=disabled): chat non-stream must NOT "
            f"inject sentinel; got content={content!r}"
        )
        if content:
            assert "truncated" not in content.lower(), (
                f"opt-out e2e (env=disabled): chat non-stream content "
                f"must not carry 'truncated' synthetic text; "
                f"got {content!r}"
            )
        assert payload["choices"][0]["finish_reason"] == "length"
        assert msg.get("reasoning_content"), (
            "reasoning_content must remain populated as the canonical truncation cue"
        )
    finally:
        reset_config()


def test_chat_route_enabled_surfaces_sentinel_on_length_cut(monkeypatch):
    """Sentinel-enabled (env=1, also the issue #858 default): pins
    the H-01 / #858 behaviour end-to-end on
    ``/v1/chat/completions`` with an explicit truthy value so the
    test is robust to any future default flip."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        msg = payload["choices"][0]["message"]
        assert msg.get("content") == REASONING_CUTOFF_SENTINEL, (
            "enabled (env=1): chat non-stream must surface sentinel "
            f"on length-cut mid-think; got content={msg.get('content')!r}"
        )
        assert payload["choices"][0]["finish_reason"] == "length"
    finally:
        reset_config()


def test_chat_route_default_env_surfaces_sentinel_regression_858(monkeypatch):
    """End-to-end regression pin for issue #858 on the actual route.

    With the env var UNSET (the rapid-desktop / vanilla-SDK default),
    ``/v1/chat/completions`` non-streaming MUST carry the sentinel in
    ``message.content`` on length-cut mid-think. Pure helper-level
    coverage is insufficient — if a future refactor unwires the
    chat-route call site (or skips it on a code path), the helper
    test would still pass while the user-visible bug (#858 empty
    bubble) reappears. This test drives the actual FastAPI router
    end-to-end and asserts the envelope shape clients see.
    """
    monkeypatch.delenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.chat import router as chat_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "stream": False,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        msg = payload["choices"][0]["message"]
        assert msg.get("content") == REASONING_CUTOFF_SENTINEL, (
            f"issue #858 e2e: default env must inject sentinel into "
            f"message.content on length-cut mid-think; got "
            f"content={msg.get('content')!r}"
        )
        assert payload["choices"][0]["finish_reason"] == "length"
        assert msg.get("reasoning_content"), (
            "reasoning_content must remain populated alongside the sentinel"
        )
    finally:
        reset_config()


def test_anthropic_route_opt_out_no_sentinel_on_length_cut(monkeypatch):
    """Opt-out e2e contract for ``/v1/messages``: when
    ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` is set, the route
    must NOT inject the sentinel into any content block.
    ``stop_reason="max_tokens"`` + the ``thinking`` content block are
    the canonical truncation cues under this branch."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(anthropic_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        body_str = json.dumps(payload)
        assert REASONING_CUTOFF_SENTINEL not in body_str, (
            f"opt-out e2e (env=disabled): /v1/messages must NOT carry "
            f"the sentinel; got payload={payload!r}"
        )
        assert "truncated" not in body_str.lower() or (
            # Allow harmless "truncated" inside non-content fields if
            # any future metadata mentions it — guard the content block
            # specifically.
            all(
                "truncated" not in (block.get("text") or "").lower()
                for block in payload.get("content") or []
                if block.get("type") == "text"
            )
        ), (
            f"opt-out e2e (env=disabled): /v1/messages text content "
            f"blocks must not carry 'truncated' synthetic text; "
            f"got payload={payload!r}"
        )
    finally:
        reset_config()


def test_anthropic_route_enabled_surfaces_sentinel(monkeypatch):
    """Sentinel-enabled (env=1): pins the issue #858 / H-01 behaviour
    end-to-end on ``/v1/messages`` with an explicit truthy value so
    the test is robust to any future default flip."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router as anthropic_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(anthropic_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "compute 17*23"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        # Compare against the JSON-decoded text block payloads — the raw
        # ``json.dumps`` body escapes the em-dash to ``—`` and would
        # spuriously fail a substring check against the unicode literal.
        text_blocks = [
            (block.get("text") or "")
            for block in (payload.get("content") or [])
            if block.get("type") == "text"
        ]
        assert any(REASONING_CUTOFF_SENTINEL in t for t in text_blocks), (
            f"enabled e2e (env=1): /v1/messages must surface sentinel "
            f"in a text content block on length-cut mid-think; "
            f"got payload={payload!r}"
        )
    finally:
        reset_config()


def test_responses_route_opt_out_no_sentinel_on_length_cut(monkeypatch):
    """Opt-out e2e contract for ``/v1/responses``: when
    ``RAPID_MLX_REASONING_CUTOFF_NOTICE=disabled`` is set, the route
    must NOT inject the sentinel into any output_text block.
    ``status="incomplete"`` +
    ``usage.output_tokens_details.reasoning_tokens`` are the canonical
    truncation cues under this branch."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "disabled")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(responses_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "max_output_tokens": 16,
                "input": "compute 17*23",
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        body_str = json.dumps(payload)
        assert REASONING_CUTOFF_SENTINEL not in body_str, (
            f"opt-out e2e (env=disabled): /v1/responses must NOT carry "
            f"the sentinel; got payload={payload!r}"
        )
        # Walk output[].content[] explicitly to catch any synthetic
        # output_text block.
        for item in payload.get("output") or []:
            for block in item.get("content") or []:
                text = block.get("text") or ""
                assert "truncated" not in text.lower(), (
                    f"opt-out e2e (env=disabled): /v1/responses "
                    f"output_text must not carry 'truncated' synthetic "
                    f"text; got block={block!r}"
                )
    finally:
        reset_config()


def test_responses_route_enabled_surfaces_sentinel(monkeypatch):
    """Sentinel-enabled (env=1): pins the issue #858 / H-01 behaviour
    end-to-end on ``/v1/responses`` with an explicit truthy value so
    the test is robust to any future default flip."""
    monkeypatch.setenv("RAPID_MLX_REASONING_CUTOFF_NOTICE", "1")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    _seed_length_cut_engine(cfg)

    try:
        app = FastAPI()
        app.include_router(responses_router)
        client = TestClient(app)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "max_output_tokens": 16,
                "input": "compute 17*23",
                "stream": False,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        # Walk ``output[].content[]`` for a text block carrying the
        # sentinel literal — the JSON-dumped body escapes the em-dash,
        # so a substring check against the unicode literal there is
        # brittle. The decoded text block is the authoritative payload.
        sentinel_texts: list[str] = []
        for item in payload.get("output") or []:
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    sentinel_texts.append(block.get("text") or "")
        assert any(REASONING_CUTOFF_SENTINEL in t for t in sentinel_texts), (
            f"enabled e2e (env=1): /v1/responses must surface sentinel "
            f"in an output_text block on length-cut mid-think; "
            f"got payload={payload!r}"
        )
    finally:
        reset_config()
