# SPDX-License-Identifier: Apache-2.0
"""Regression test for harmony reasoning_content silently dropping on the
non-streaming + no-tools path.

Bug surfaced 2026-05-22 in a fresh-PyPI v0.6.65 onboarding smoke test
against ``mlx-community/gpt-oss-20b-MXFP4-Q8``: a reasoning prompt with
no tools came back with ``reasoning_content=""`` while the full
chain-of-thought leaked into ``content``. Tool calls and streaming both
worked — only ``non-stream + no-tools + harmony`` broke.

Root cause: the engine calls ``clean_output_text`` on the raw harmony
output before constructing ``GenerationOutput``. ``_clean_gpt_oss_output``
extracts the final-channel content and strips all channel markers, so by
the time the route's ``_finalize_content_and_reasoning`` runs the
reasoning parser, the ``<|channel|>analysis<|message|>…<|end|>`` block
no longer exists in ``cleaned_text``. ``HarmonyReasoningParser`` finds
nothing and returns ``(None, None)``. PR #436 added a guard that
preserved ``content`` from being clobbered to None, but did not rescue
``reasoning_content`` — that's what this test pins.

Fix: ``GenerationOutput`` now carries ``raw_text`` (the pre-clean output).
Routes pass it to the helper, which retries the reasoning parser on
``raw_text`` whenever the first parse against ``cleaned_text`` yielded
no reasoning. Non-harmony parsers (``<think>``-based) are unaffected
because their first parse succeeds on cleaned_text — they never enter
the retry branch.
"""

from __future__ import annotations

from vllm_mlx.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
from vllm_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from vllm_mlx.service.helpers import _finalize_content_and_reasoning

# A realistic gpt-oss-20b-mxfp4-q8 harmony non-stream response: analysis channel
# (CoT) followed by final channel (answer), terminated with <|return|>.
_HARMONY_RAW = (
    "<|channel|>analysis<|message|>"
    "Let me think step by step. 17 * 23 = 17*20 + 17*3 = 340 + 51 = 391."
    "<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
    "The answer is 391."
    "<|return|>"
)

# What the engine's ``clean_output_text`` would emit for that raw output —
# the final-channel content only, channel markers stripped.
_HARMONY_CLEANED = "The answer is 391."


def test_harmony_no_tools_recovers_reasoning_from_raw_text():
    """The bug: ``_finalize_content_and_reasoning`` with cleaned_text that
    has had harmony channels stripped used to return ``reasoning=None``.
    With ``raw_text`` carrying the pre-clean output, the helper retries
    on it and recovers the analysis-channel content.
    """
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_HARMONY_RAW,
        cleaned_text=_HARMONY_CLEANED,
        tool_calls=[],
        reasoning_parser=HarmonyReasoningParser(),
    )
    assert reasoning is not None, (
        "harmony non-tool path dropped reasoning_content — "
        "the engine-pre-cleaned cleaned_text has no channel markers, so "
        "the parser must be re-run on raw_text"
    )
    assert "17 * 23" in reasoning, (
        f"recovered reasoning is missing analysis-channel content: {reasoning!r}"
    )
    # cleaned_text retains the parser's final-channel extraction (or the
    # input cleaned_text if the parser produced no new cleaned value).
    assert cleaned and "391" in cleaned


def test_harmony_no_tools_no_raw_text_keeps_existing_behavior():
    """When ``raw_text`` matches ``cleaned_text`` (e.g. an old caller that
    didn't populate the new ``GenerationOutput.raw_text`` field, so the
    route falls back to passing ``output.text`` for both), the retry
    branch must NOT fire — there's nothing new to try.
    """
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_HARMONY_CLEANED,
        cleaned_text=_HARMONY_CLEANED,
        tool_calls=[],
        reasoning_parser=HarmonyReasoningParser(),
    )
    # Reasoning is still lost — but cleaned_text survives (PR #436 guard).
    # This pins the pre-fix behavior so we know the new retry only kicks
    # in when raw_text was actually populated.
    assert reasoning is None
    assert cleaned == _HARMONY_CLEANED


def test_qwen3_think_parser_unaffected_by_retry():
    """``<think>`` parsers find their reasoning on the first pass against
    ``cleaned_text``, so the retry branch never executes — no double-work
    and no risk of overwriting a successful extraction with raw_text.
    """
    # The tool parser would have already stripped <think> off, leaving
    # just the answer as cleaned_text. We simulate the path where the
    # reasoning parser is the one that pulls <think> out.
    raw = "<think>compute 2+2 = 4</think>The answer is 4."
    cleaned_input = raw  # no tool parser ran first
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=cleaned_input,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(tokenizer=None),
    )
    assert reasoning is not None and "2+2" in reasoning
    # Qwen3 parser strips <think>...</think> so cleaned should not
    # contain the thinking block.
    assert cleaned is not None
    assert "<think>" not in cleaned


def test_harmony_with_tool_calls_unchanged():
    """The tool-call branch already parses raw_text directly — this test
    pins that the retry logic only lives in the no-tools branch and
    doesn't perturb tool-call behavior.
    """
    raw = (
        "<|channel|>analysis<|message|>need to call get_weather<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|message|>{"location":"Paris"}<|call|>'
    )
    # Simulate that the tool parser already extracted a tool call and
    # produced an empty cleaned_text.
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text="",
        tool_calls=[{"id": "x", "type": "function", "function": {}}],
        reasoning_parser=HarmonyReasoningParser(),
    )
    assert reasoning is not None and "get_weather" in reasoning


def test_no_reasoning_parser_short_circuits():
    """When the model has no reasoning parser configured, the helper must
    not attempt any extraction (raw_text retry included) and must return
    cleaned_text untouched, reasoning_text=None.
    """
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_HARMONY_RAW,
        cleaned_text=_HARMONY_CLEANED,
        tool_calls=[],
        reasoning_parser=None,
    )
    assert reasoning is None
    assert cleaned == _HARMONY_CLEANED


def test_engine_reasoning_text_short_circuits_parser():
    """When the engine populated ``reasoning_text`` via ``OutputRouter``,
    the helper trusts it as authoritative and skips the text-based
    parser. This is the root-cause fix for issue #442 — token-level
    routing replaces fragile regex parsing of decoded output. The
    parser would have returned None on this truncated input (no
    ``<|end|>`` to anchor against); the engine-provided text wins.
    """
    truncated_raw = (
        "<|channel|>analysis<|message|>User wants multiple actions. We must use tools."
    )
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=truncated_raw,
        cleaned_text="",
        tool_calls=[],
        reasoning_parser=HarmonyReasoningParser(),
        engine_reasoning_text="User wants multiple actions. We must use tools.",
    )
    assert reasoning == "User wants multiple actions. We must use tools."
    assert cleaned == ""


def test_engine_reasoning_text_overrides_even_when_parser_could_match():
    """When the engine populated ``reasoning_text``, its value wins even
    if the text-based parser COULD have found something — token-level
    routing is the authoritative source. This pins the precedence so a
    future change can't accidentally re-prefer the regex result.
    """
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_HARMONY_RAW,
        cleaned_text=_HARMONY_CLEANED,
        tool_calls=[],
        reasoning_parser=HarmonyReasoningParser(),
        engine_reasoning_text="ENGINE-ROUTED-VALUE",
    )
    assert reasoning == "ENGINE-ROUTED-VALUE"


def test_engine_reasoning_empty_falls_through_to_parser():
    """Empty ``engine_reasoning_text`` means the engine couldn't route
    (no ``OutputRouter`` for this tokenizer, or a non-channel model).
    Helper falls through to the existing parser-based extraction so
    older formats (e.g. plain ``<think>`` tags) keep working.
    """
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_HARMONY_RAW,
        cleaned_text=_HARMONY_CLEANED,
        tool_calls=[],
        reasoning_parser=HarmonyReasoningParser(),
        engine_reasoning_text="",
    )
    assert reasoning is not None and "17 * 23" in reasoning


# ---------------------------------------------------------------------------
# #575 — Case-4 leak plug + effective-thinking resolution + signature probe
# ---------------------------------------------------------------------------

from vllm_mlx.reasoning.glm4_parser import Glm4ReasoningParser
from vllm_mlx.service.helpers import (
    _effective_enable_thinking,
    _parser_accepts_enable_thinking,
)

_QWEN3_TRUNCATED_THOUGHT = (
    "Here's my thinking process:\n"
    "1. The user is asking about train travel between two cities.\n"
    "2. I need to compute the meeting point given two speeds...\n"
    "3. Let me set up the equation: distance = speed * time...\n"
    "[truncated mid-thought, finish_reason='length', no closing tag]"
)


def test_575_qwen3_truncated_thought_does_not_leak_to_content_when_thinking_on():
    """End-to-end behaviour pin: when ``enable_thinking=True`` AND the
    parser returns ``(reasoning, None)`` on a no-tag truncated thought,
    the helper MUST clear ``cleaned_text`` so the route's ``final_content``
    becomes ``None``. Pre-fix the same text leaked into BOTH
    ``reasoning_content`` AND ``content`` (codex R1 BLOCKING)."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_QWEN3_TRUNCATED_THOUGHT,
        cleaned_text=_QWEN3_TRUNCATED_THOUGHT,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=True,
    )
    assert reasoning == _QWEN3_TRUNCATED_THOUGHT.strip()
    # The critical assertion — falsy cleaned_text means the route
    # renders ``content=None`` and the client doesn't see the leak.
    assert not cleaned


def test_575_qwen3_truncated_thought_legacy_behaviour_when_thinking_explicit_off():
    """Backward-compat pin: with ``enable_thinking=False`` (caller
    affirmatively disabled thinking) the bare-text fallback added by
    #570 MUST NOT fire — a non-thinking answer that happens to open
    with ``Here's my reasoning:`` or similar scratchpad-shaped phrasing
    must stay in ``content`` or the client gets an empty
    ``message.content``. (Codex r3 BLOCKING on PR #573.)

    NOTE: the older form of this test pinned ``enable_thinking=None``
    to the same legacy contract, but #570 (PR #573) changed the
    None-case to fire the bare-text fallback defensively — see
    ``test_570_qwen3_truncated_thought_routes_to_reasoning_when_thinking_unspecified``
    below. Only the explicit-False path still preserves the strict
    legacy "text stays as content" behaviour, and that's the
    backward-compat shim third-party callers need."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_QWEN3_TRUNCATED_THOUGHT,
        cleaned_text=_QWEN3_TRUNCATED_THOUGHT,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=False,
    )
    # Legacy path: no Case-4 fast-path AND no bare-text fallback when
    # thinking is explicitly off — text stays as content, reasoning is None.
    assert reasoning is None
    assert cleaned == _QWEN3_TRUNCATED_THOUGHT


def test_570_qwen3_truncated_thought_routes_to_reasoning_when_thinking_unspecified():
    """#570 / PR #573 fix path: legacy callers that don't thread the
    ``enable_thinking`` flag through (it stays at ``None``) still get
    defensive routing when the model emits a recognizable bare-text
    thinking preamble. The whole truncated thought lands in
    ``reasoning`` and ``cleaned_text`` is blanked so it doesn't leak
    into ``message.content``. Distinguishes from the explicit-False
    legacy shim above — None is "no signal" → trust the pattern,
    False is "caller said no" → don't override."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_QWEN3_TRUNCATED_THOUGHT,
        cleaned_text=_QWEN3_TRUNCATED_THOUGHT,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=None,
    )
    assert reasoning == _QWEN3_TRUNCATED_THOUGHT.strip()
    # Blanked so the route renders ``content=None`` and the bare-text
    # preamble doesn't leak.
    assert not cleaned


def test_570_qwen3_valid_non_thinking_answer_not_clobbered_when_thinking_off():
    """Codex r3 BLOCKING regression pin: a teaching / tutorial answer
    that explains a chain-of-thought methodology (``Here's a thinking
    process you can use…``) must NOT have its content cleared when
    the caller passes ``enable_thinking=False`` — the user explicitly
    asked for the explanation. Without the explicit-False gate, the
    bare-text regex would match and the user would see empty
    ``message.content``."""
    valid_answer = (
        "Here's a thinking process you can use for any optimisation "
        "problem: first survey the options, then score each one "
        "against your criteria, then pick the top-scoring result."
    )
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=valid_answer,
        cleaned_text=valid_answer,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=False,
    )
    assert reasoning is None
    assert cleaned == valid_answer


def test_575_glm4_no_tags_thinking_on_does_not_clobber_content():
    """GLM-4 explicitly diverges from Qwen3 — its parser drops the
    ``enable_thinking`` kwarg before delegating, so even when the route
    passes ``True`` the helper must NOT touch ``cleaned_text``."""
    text = "GLM-4 plain answer with no think tags at all."
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=text,
        cleaned_text=text,
        tool_calls=[],
        reasoning_parser=Glm4ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=True,
    )
    assert reasoning is None
    assert cleaned == text


def test_575_qwen3_normal_split_unchanged_with_thinking_on():
    """When the model emits the normal ``…</think>answer`` shape the
    helper's Case-4 leak plug must NOT fire — the well-behaved split
    has both reasoning and content and clobbering content would break
    every successful thinking response."""
    raw = "step by step reasoning</think>The answer is 42."
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=True,
    )
    assert reasoning == "step by step reasoning"
    assert cleaned == "The answer is 42."


# ---- _effective_enable_thinking ------------------------------------------


def test_effective_enable_thinking_concrete_true_passes_through():
    assert _effective_enable_thinking(True, "qwen3.5-4b-4bit") is True


def test_effective_enable_thinking_concrete_false_passes_through():
    assert _effective_enable_thinking(False, "qwen3.5-4b-4bit") is False


def test_effective_enable_thinking_none_non_coder_defaults_true():
    """Mirrors ``vllm_mlx/utils/chat_template.py:127`` — the same
    default the prompt-render path applies. Without this, ``None`` on
    the default Qwen3 path would skip the Case-4 fallback even though
    the chat template DID inject ``<think>`` (codex R1 BLOCKING)."""
    assert _effective_enable_thinking(None, "qwen3.5-4b-4bit") is True


def test_effective_enable_thinking_none_coder_defaults_false():
    """Coder variants do NOT pre-inject ``<think>`` — keep them on the
    legacy no-tag-→-content path."""
    assert _effective_enable_thinking(None, "qwen3-coder-30b-a3b") is False


def test_effective_enable_thinking_no_model_name_preserves_none():
    """Defensive: callers without a known model name keep the legacy
    ``None`` so we don't silently flip behaviour."""
    assert _effective_enable_thinking(None, None) is None
    assert _effective_enable_thinking(None, "") is None


# ---- _parser_accepts_enable_thinking -------------------------------------


def test_parser_accepts_enable_thinking_modern_parser():
    """All in-tree parsers accept the kwarg post-#575."""
    assert _parser_accepts_enable_thinking(Qwen3ReasoningParser()) is True
    assert _parser_accepts_enable_thinking(Glm4ReasoningParser()) is True
    assert _parser_accepts_enable_thinking(HarmonyReasoningParser()) is True


def test_parser_accepts_enable_thinking_legacy_parser_returns_false():
    """A third-party parser on the old 1-arg signature must be detected
    statically — no side-effecting ``extract("")`` probe (codex R1 NIT)."""

    class LegacyParser:
        def extract_reasoning(self, model_output: str):  # noqa: ARG002
            return None, model_output

    assert _parser_accepts_enable_thinking(LegacyParser()) is False


def test_parser_accepts_enable_thinking_kwargs_catchall_returns_true():
    """Parsers that declare ``**kwargs`` should be treated as accepting
    the flag — they can either consume or ignore it."""

    class KwargsParser:
        def extract_reasoning(self, model_output: str, **kwargs):  # noqa: ARG002
            return None, model_output

    assert _parser_accepts_enable_thinking(KwargsParser()) is True


def test_parser_accepts_enable_thinking_missing_method_returns_false():
    """Defensive — a stub without the method should not crash the
    helper, just fall back to the 1-arg path (which itself will then
    AttributeError, but that's a separate caller-visible bug)."""

    class NoMethod:
        pass

    assert _parser_accepts_enable_thinking(NoMethod()) is False


def test_parser_accepts_enable_thinking_no_side_effects():
    """The static signature check MUST NOT call ``extract_reasoning``.
    A stateful parser whose ``extract_reasoning`` mutates internal
    state on every call would be silently corrupted by the previous
    ``extract("")`` probe; this pins that we never touch the body."""

    class StatefulParser:
        def __init__(self):
            self.call_count = 0

        def extract_reasoning(
            self, model_output: str, enable_thinking: bool | None = None
        ):
            self.call_count += 1
            return None, model_output

    parser = StatefulParser()
    assert _parser_accepts_enable_thinking(parser) is True
    assert parser.call_count == 0


# ---- #575 codex R2 — Harmony retry must survive Case-4 leak plug ---------


def test_575_r2_harmony_retry_with_thinking_on_does_not_clear_cleaned_text():
    """codex R2 BLOCKING: when Harmony's analysis-channel retry on
    ``raw_text`` recovers reasoning, the FIRST parse on the engine-
    cleaned ``"The answer is 391."`` returned ``(None, None)`` — NOT
    the no-tag Case-4 fallback. The leak plug must NOT mistake this
    shape for Case-4 and clobber the legitimate final-channel
    content. Pin the regression that round-1's naive guard would
    have introduced."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_HARMONY_RAW,
        cleaned_text=_HARMONY_CLEANED,
        tool_calls=[],
        reasoning_parser=HarmonyReasoningParser(),
        engine_reasoning_text="",
        # Default-on thinking for non-coder models flows through here
        # via ``_effective_enable_thinking(None, model_name) == True``;
        # exercise the True branch explicitly so a future refactor
        # can't silently regress.
        enable_thinking=True,
    )
    assert reasoning is not None and "17 * 23" in reasoning
    assert cleaned == _HARMONY_CLEANED, (
        "Harmony's clean final-channel content MUST survive the "
        "Case-4 leak plug — first parse on cleaned_text was "
        "(None, None), not the no-tag (reasoning, None) the plug "
        "targets"
    )


def test_575_r2_qwen3_case4_still_clears_when_thinking_on():
    """Sanity counter-test to the harmony case above: when the FIRST
    parse really WAS the no-tag Case-4 fallback, the plug DOES fire."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_QWEN3_TRUNCATED_THOUGHT,
        cleaned_text=_QWEN3_TRUNCATED_THOUGHT,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=True,
    )
    assert reasoning == _QWEN3_TRUNCATED_THOUGHT.strip()
    assert not cleaned


# ---------------------------------------------------------------------------
# 2026-06-17 VibeThinker live-test regression: truncated ``<think>`` opener
# without ``</think>`` (``finish_reason=length`` mid-thought) used to leak
# the byte-identical reasoning trace into BOTH ``content`` AND
# ``reasoning_content``. The helper now clears ``cleaned_text`` when the
# parser returns ``(reasoning, None)`` AND ``cleaned_text`` opens with an
# unclosed ``<think>``. See ``first_parse_was_truncated_think`` in
# ``vllm_mlx/service/helpers.py``.
# ---------------------------------------------------------------------------

_VIBETHINKER_TRUNCATED_THINK = (
    "<think>\nStep 1: Let's analyze the problem.\n"
    "Step 2: Apply the formula.\n"
    "Step 3: We need to compute 7 * 12 = 84.\n"
    "Step 4: Then we add 5 to get 89.\n"
    "Step 5: Let me double-check by another method"
    # NO ``</think>`` — finish_reason=length truncated mid-thought.
)


def test_vibethinker_truncated_think_no_duplicate_content_reasoning():
    """2026-06-17 VibeThinker live-test repro: ``<think>...`` truncated
    with no ``</think>`` used to surface the whole trace identically in
    BOTH ``content`` and ``reasoning_content`` (live-test math row:
    content_len == reasoning_len == 5449, byte-identical). The fix
    clears ``cleaned_text`` when the parser returns ``(reasoning,
    None)`` AND ``cleaned_text`` opens with an unclosed ``<think>``.

    This test would FAIL on main because the leak plug in
    ``_finalize_content_and_reasoning`` only fires on Case-4 (no tag at
    all) — Case-3 (unclosed ``<think>`` opener) falls through and the
    raw text echoes into both fields.
    """
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_VIBETHINKER_TRUNCATED_THINK,
        cleaned_text=_VIBETHINKER_TRUNCATED_THINK,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text="",
        # No ``enable_thinking`` signal from the caller — the literal
        # ``<think>`` token in the output is the model's own evidence
        # of thinking, so the fix must NOT be gated on this flag.
        enable_thinking=None,
    )
    # Reasoning trace recovered
    assert reasoning is not None
    assert "Step 3" in reasoning
    # Content explicitly blanked — must NOT duplicate the reasoning
    # trace (the live-test bug signature).
    assert not cleaned, f"truncated <think> trace leaked into content: {cleaned!r}"
    # Defensive: the trace must not appear in content.
    assert "Step 1" not in (cleaned or "")


def test_vibethinker_truncated_think_with_thinking_enabled_explicit():
    """Same as the test above but with the caller passing
    ``enable_thinking=True``. The plug must also fire on this path
    (was already implicitly covered by the Case-4 ``enable_thinking=True``
    branch, but Case-3 is gated independently — pin both paths)."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_VIBETHINKER_TRUNCATED_THINK,
        cleaned_text=_VIBETHINKER_TRUNCATED_THINK,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=True,
    )
    assert reasoning is not None
    assert "Step 3" in reasoning
    assert not cleaned


def test_normal_closed_think_block_still_splits_correctly():
    """Counter-test: a normal ``<think>...</think>answer`` shape with
    BOTH tags present must still split into reasoning + content. The
    new leak-plug gate requires ``</think>`` to be ABSENT, so this
    well-formed case must be unaffected."""
    raw = "<think>step by step reasoning</think>The answer is 42."
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=None,
    )
    assert reasoning == "step by step reasoning"
    assert cleaned == "The answer is 42."


def test_vibethinker_truncated_think_engine_routed_no_duplicate():
    """2026-06-17 VibeThinker live-test repro — engine-routed path.

    When ``engine_reasoning_text`` is non-empty the helper short-
    circuits to ``_apply_reasoning_cap`` BEFORE the reasoning-parser
    branch runs. In that path the leak-plug below the parser never
    fires, so a truncated ``<think>`` opener in ``cleaned_text``
    leaks straight into the client's ``content`` field (live-test
    math row: content_starts_with='<think>We need to find...').

    The fix blanks ``cleaned_text`` in the engine-routed branch too
    when ``cleaned_text`` opens with an unclosed ``<think>``,
    mirroring the post-parser plug.
    """
    raw = "<think>Let me work through this step by step. 7 * 12 ="
    engine_reasoning = "Let me work through this step by step. 7 * 12 ="
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        # Any parser will do — the engine-reasoning short-circuit
        # bypasses the parser entirely.
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text=engine_reasoning,
        enable_thinking=None,
    )
    # Reasoning comes from the engine (router-routed).
    assert reasoning == engine_reasoning
    # Content explicitly blanked — must NOT duplicate the trace
    # (the live-test bug signature).
    assert not cleaned, (
        f"truncated <think> trace leaked into engine-routed content: {cleaned!r}"
    )


def test_engine_routed_closed_think_block_passes_through():
    """Counter-test for the engine-routed branch: a normal closed
    ``<think>...</think>answer`` shape in ``cleaned_text`` MUST be
    preserved. The new gate requires the opening tag at lstrip-start
    AND no closing tag, so a well-formed block is unaffected."""
    raw = "<think>reasoning</think>The answer is 42."
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text="reasoning",
        enable_thinking=None,
    )
    assert reasoning == "reasoning"
    # cleaned_text passed through intact — both ``<think>`` and
    # ``</think>`` are present, the gate is OFF.
    assert cleaned == raw


def test_engine_routed_truncated_think_preserves_preamble():
    """Same as the engine-routed test above, but with a chatty
    preamble before the unclosed ``<think>``. ``partition`` preserves
    the preamble as content while dropping the leaked trace."""
    raw = (
        "Okay, let me think about this carefully and step by step.\n\n"
        "<think>I should compute 7 * 12 = 84, then add 5..."
    )
    engine_reasoning = "I should compute 7 * 12 = 84, then add 5..."
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text=engine_reasoning,
        enable_thinking=None,
    )
    assert reasoning == engine_reasoning
    # Preamble preserved, leaked thought dropped.
    assert cleaned == "Okay, let me think about this carefully and step by step."
    assert "compute 7 * 12" not in cleaned
    assert "<think>" not in cleaned


def test_truncated_think_with_pre_think_content_preserves_preamble():
    """Codex r1 P2 follow-up: when the model emits a preamble BEFORE
    ``<think>`` and is then truncated, the trim must preserve the
    preamble as content while dropping the unclosed thought trace.

    The original VibeThinker bug report explicitly calls out this
    shape (the merge_intervals streaming case — model emits a chatty
    multi-sentence intro before its ``<think>`` opener). The first
    iteration of this fix used a conservative
    ``lstrip().startswith("<think>")`` gate that LEAKED the trace
    into content for this case; codex r1 flagged it.

    The fix uses ``partition("<think>")[0]`` so:
      * Start-aligned (math row): preamble == "" → cleaned == ""
      * Preamble shape (merge_intervals): preamble preserved →
        content stays semantically meaningful while the leaked thought
        is dropped.
    """
    raw = "Here is some content\n<think>truncated thought"
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=None,
    )
    # Parser extracts the post-``<think>`` portion as reasoning.
    assert reasoning == "truncated thought"
    # The preamble is preserved; the unclosed ``<think>`` opener and
    # the leaked trace are dropped. (Trailing whitespace before
    # ``<think>`` is also stripped by ``.rstrip()`` for clean
    # downstream rendering.)
    assert cleaned == "Here is some content"
    # Defensive: the raw thought must NOT survive in content.
    assert "truncated thought" not in cleaned
    assert "<think>" not in cleaned


# Codex r3 P2 — reasoning cap must not re-leak truncated thought.


def test_truncated_think_with_reasoning_cap_does_not_leak_overflow():
    """Codex r3 P2: ``_apply_reasoning_cap`` prepends over-cap reasoning
    into ``cleaned_text`` so the wire ordering matches the model's
    emission. That's correct for closed ``<think>...</think>answer``
    splits but WRONG for truncated thoughts — the overflow IS the
    leaked thought trace.

    The fix routes the truncated-``<think>`` paths through
    ``_truncate_reasoning_only`` which caps reasoning but does NOT
    rewrite ``cleaned_text``. Without this, a client setting
    ``reasoning_max_tokens=100`` against a 4000-char truncated
    thought would still see ~3600 chars of reasoning prose in
    ``content``.

    Both branches (engine-routed and parser-routed) of
    ``_finalize_content_and_reasoning`` must use the
    reasoning-only cap when the plug fires.
    """
    # Long enough that the cap (default chars≈max_tokens*4) bites.
    long_thought = "<think>" + ("step by step reasoning " * 100)
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=long_thought,
        cleaned_text=long_thought,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=None,
        reasoning_max_tokens=10,  # 40 chars
    )
    # Reasoning capped at 40 chars.
    assert reasoning is not None
    assert len(reasoning) == 40
    # The overflow MUST NOT be written into cleaned_text — the
    # client setting a small cap explicitly asked for thought
    # truncation, not for the overflow to surface as content.
    assert cleaned == ""
    assert "step by step reasoning" not in (cleaned or "")


def test_engine_routed_truncated_think_with_cap_does_not_leak_overflow():
    """Same as above but for the engine-routed short-circuit branch."""
    raw = "<think>" + ("the model is thinking out loud " * 100)
    engine_reasoning = "the model is thinking out loud " * 100
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=DeepSeekR1ReasoningParser(),
        engine_reasoning_text=engine_reasoning,
        enable_thinking=None,
        reasoning_max_tokens=10,  # 40 chars
    )
    assert reasoning is not None
    assert len(reasoning) == 40
    assert cleaned == ""
    assert "thinking out loud" not in (cleaned or "")


# ---------------------------------------------------------------------------
# Bug B (PR #715 fuzz bundle): the truncated-``<think>`` leak plug must
# fire for ALL reasoning parsers whose ``extract_reasoning`` returns
# ``(reasoning, None)`` on unclosed-``<think>`` input — including the
# Qwen3 parser used by ``qwen3-4b-thinking-2507-4bit``. The plug lives in
# the shared ``_finalize_content_and_reasoning`` helper so coverage
# should be parser-agnostic; pinning here so a parser-specific override
# can't silently regress.
# ---------------------------------------------------------------------------


def test_qwen3_truncated_think_no_duplicate_content_reasoning():
    """Same bug shape as the VibeThinker repro above, but via the
    ``Qwen3ReasoningParser`` (the parser wired for
    ``qwen3-4b-thinking-2507-4bit`` and the rest of the Qwen3
    thinking family). The fuzz battery confirmed
    ``finish_reason=length`` mid-``<think>`` on Qwen3-4B-Thinking-2507
    produced byte-identical content + reasoning_content before the
    leak plug landed.

    Since the plug fires on the parser's ``(reasoning, None)`` return
    shape rather than the parser class, this test would FAIL if a
    future PR routed Qwen3 truncated-think through a different
    branch that bypasses the plug.
    """
    raw = _VIBETHINKER_TRUNCATED_THINK
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=None,
    )
    assert reasoning is not None
    assert "Step 3" in reasoning
    assert not cleaned, (
        f"qwen3 truncated <think> trace leaked into content: {cleaned!r}"
    )


def test_qwen3_truncated_think_with_enable_thinking_true():
    """Qwen3 + ``enable_thinking=True`` — the explicit-thinking signal
    must NOT cause the plug to mis-fire and also must NOT double-clear
    cleaned_text in a way that breaks the existing #575 Case-4 path."""
    raw = _VIBETHINKER_TRUNCATED_THINK
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=True,
    )
    assert reasoning is not None
    assert "Step 3" in reasoning
    assert not cleaned


def test_qwen3_engine_routed_truncated_think_no_duplicate():
    """Engine-routed path with Qwen3 parser wired (the typical
    Qwen3-4B-Thinking-2507 production shape: OutputRouter handles the
    ``<think>`` token boundary, then helper finalizes)."""
    raw = "<think>Let me solve this. 7 * 12 = 84, then I need to..."
    engine_reasoning = "Let me solve this. 7 * 12 = 84, then I need to..."
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=raw,
        cleaned_text=raw,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text=engine_reasoning,
        enable_thinking=None,
    )
    assert reasoning == engine_reasoning
    assert not cleaned, (
        f"qwen3 engine-routed truncated <think> trace leaked: {cleaned!r}"
    )
