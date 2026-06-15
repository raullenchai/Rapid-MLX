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


def test_575_qwen3_truncated_thought_legacy_behaviour_when_thinking_off():
    """Backward-compat pin: with ``enable_thinking`` left at ``None``
    (the pre-#575 contract) the helper does NOT clear cleaned_text and
    the legacy behaviour is preserved — useful for shimming third-party
    callers that have not yet wired the flag through."""
    cleaned, reasoning = _finalize_content_and_reasoning(
        raw_text=_QWEN3_TRUNCATED_THOUGHT,
        cleaned_text=_QWEN3_TRUNCATED_THOUGHT,
        tool_calls=[],
        reasoning_parser=Qwen3ReasoningParser(),
        engine_reasoning_text="",
        enable_thinking=None,
    )
    # Legacy path: no Case-4 → text stays as content, reasoning is None.
    assert reasoning is None
    assert cleaned == _QWEN3_TRUNCATED_THOUGHT


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
