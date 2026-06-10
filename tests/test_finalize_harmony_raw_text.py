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
