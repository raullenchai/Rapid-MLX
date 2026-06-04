# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #468 — harmony multi-channel compound leak.

Reported 2026-05-25 on ``mlx-community/gpt-oss-20b-MXFP4-Q8`` with
``tool_choice="required"``. The issue body shows two stacked failures:

1. **tool_choice="required" enforcement gap** — the model returns
   plain content instead of a tool call. This is an FSM-constrained-
   decoding problem (see PR #132 draft) and is deliberately OUT OF
   SCOPE for this streaming-parser cluster fix.

2. **Channel routing failure when analysis AND commentary BOTH fire**
   — the user-visible content reads
   ``"analysisWe have to use the get_time tool.assistantcommentary
   to=functions.get_time json{}"``. The cleaner stripped ``<|...|>``
   control tokens but left the channel-type words AND the recipient
   AND the body all in content. ``tool_calls`` was null.

This file pins the second symptom at the router level: when the model
emits a compound sequence (analysis block followed by an assistant
turn with a commentary tool call), the router MUST partition the
output into reasoning + tool_calls + empty content.

## Relationship to #455

#455's regressions exercise a single ``commentary`` channel in
isolation. #468's compound case is the regression that catches a fix
that handles single-channel ``commentary`` (e.g. by adding a
transient transition rule) but breaks when state must be properly
reset between analysis and commentary turns. Both files must flip
together when the cluster fix lands.

## Out-of-scope

* tool_choice="required" enforcement (FSM constraint, PR #132)
* ``clean_output_text`` text-cleaner gap that strips ``<|...|>`` but
  leaves channel-type words — that's a parser/cleaner-layer problem
  separate from the router's channel classification. The router fix
  in this cluster ensures the right channel events fire; the
  downstream text-cleaner gap is captured in [gotchas.md] entry on
  ``clean_output_text strips harmony channels before reasoning
  parser`` (memory PR #436).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_mlx.output_router import OutputRouter

from ..fake_tokenizer import HARMONY_VOCAB, harmony_fake_tokenizer


@dataclass
class _Case:
    id: str
    token_ids: list[int]
    expected_reasoning_marker: str
    expected_function_name: str
    expected_args_marker: str


_V = HARMONY_VOCAB


# ----- #468 BUG: analysis + commentary compound sequence ----------------


BUG_CASES: list[_Case] = [
    # Compound flow: model emits an analysis block ("We have to use
    # the get_time tool") then transitions through `<|start|>
    # assistant <|channel|>` into the commentary tool-call block.
    # Expected after fix:
    #   reasoning: "We have to use the get_time tool"  (or similar)
    #   tool_calls: [<entry containing "get_weather" + "Tokyo">]
    #     — we reuse get_weather as the recipient since the harness
    #     vocab has it registered; the #468 prompt used get_time but
    #     the bug surface is identical (any commentary tool call).
    #   content: None
    _Case(
        id="issue_468_analysis_then_commentary_get_weather",
        token_ids=[
            # analysis channel: reasoning block
            _V["<|channel|>"],
            _V["analysis"],
            _V["<|message|>"],
            _V["Reason"],
            _V["ing"],
            _V["<|end|>"],
            # assistant header turn for the tool call
            _V["<|start|>"],
            _V["assistant"],
            # commentary channel: tool call
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.get_weather"],
            _V[" json"],
            _V["<|message|>"],
            _V['{"city":"Tokyo"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_reasoning_marker="Reasoning",
        expected_function_name="get_weather",
        expected_args_marker='"Tokyo"',
    ),
    # Variant: multi-token JSON body in the commentary block. Pins
    # that the fix's commentary aggregation survives a compound
    # sequence — codex BLOCKING-2 from the #455 round-1 review
    # generalized.
    _Case(
        id="issue_468_analysis_then_commentary_multi_token_body",
        token_ids=[
            _V["<|channel|>"],
            _V["analysis"],
            _V["<|message|>"],
            _V["Reason"],
            _V["ing"],
            _V["<|end|>"],
            _V["<|start|>"],
            _V["assistant"],
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.calculate"],
            _V[" json"],
            _V["<|message|>"],
            _V["{"],
            _V['"expr'],
            _V['ession":"'],
            _V["17"],
            _V["*"],
            _V["23"],
            _V['"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_reasoning_marker="Reasoning",
        expected_function_name="calculate",
        expected_args_marker='"17*23"',
    ),
]


@pytest.fixture
def router() -> OutputRouter:
    """Fresh harmony OutputRouter wired to a synthetic tokenizer."""
    fake_tok = harmony_fake_tokenizer()
    r = OutputRouter.from_tokenizer(fake_tok)
    assert r is not None
    assert r.map.format_tag == "harmony"
    return r


def _normalize_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _stringify_structured(entry: object) -> str:
    """Flatten a structured tool-call dict to a searchable string.

    Mirrors the helper in the #455 regression so structured emissions
    are searchable for name/args markers regardless of dict shape.
    """
    if isinstance(entry, dict):
        parts: list[str] = []
        for value in entry.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                parts.append(_stringify_structured(value))
        return " ".join(parts)
    return repr(entry)


# ----- Bug tests (xfail until cluster fix lands) ------------------------


@pytest.mark.parametrize("case", BUG_CASES, ids=lambda c: c.id)
def test_harmony_router_compound_analysis_then_commentary(case: _Case, router):
    # Flipped from xfail strict → passing by the cluster fix's harmony
    # commentary handling in output_router.py. The compound sequence
    # works because the state machine resets cleanly across the
    # ``<|start|> assistant <|channel|>`` header transition — the
    # top-level ``<|channel|>`` check at output_router.py:164 fires
    # before the AFTER_START handler, so the post-header ``commentary``
    # word lands in the same AWAITING_CHANNEL_TYPE branch that the
    # initial ``analysis`` channel used.
    # tool_choice="required" enforcement remains out of scope here —
    # FSM PR #132 covers that.
    result = router.feed_sequence(case.token_ids)

    # Content must be empty — neither the analysis body nor the
    # commentary block should leak into content.
    assert _normalize_str(result["content"]) is None, (
        f"content leaked for case={case.id}: got={result['content']!r}. "
        "Compound sequence emitted analysis or commentary text as content."
    )

    # Reasoning carries the analysis-channel body — and ONLY the
    # analysis body. Codex round-1 NIT: substring containment alone
    # would miss commentary/tool metadata accidentally leaking INTO
    # reasoning (a fix that mis-routes everything to reasoning would
    # otherwise pass the "Reasoning is somewhere in here" check).
    # Add negative assertions for the tool metadata to enforce the
    # partition contract.
    reasoning = _normalize_str(result["reasoning"]) or ""
    assert case.expected_reasoning_marker in reasoning, (
        f"reasoning missing {case.expected_reasoning_marker!r} for "
        f"case={case.id}; got reasoning={result['reasoning']!r}"
    )
    for tool_marker in (
        case.expected_function_name,
        case.expected_args_marker,
        "commentary",
        " to=functions.",
    ):
        assert tool_marker not in reasoning, (
            f"Tool metadata marker {tool_marker!r} leaked into reasoning "
            f"for case={case.id}; got reasoning={result['reasoning']!r}. "
            "Channel partition contract violated — tool-call tokens "
            "should not appear in the reasoning channel."
        )

    # Tool call has both name and args (permissive shape — text-blob or
    # structured dict OK; same contract as #455 round-1 BLOCKING-1 fix).
    tool_calls = result["tool_calls"]
    assert tool_calls, (
        f"tool_calls is empty/None for case={case.id}; got={tool_calls!r}"
    )
    assert len(tool_calls) == 1, (
        f"Expected ONE aggregated tool_calls entry for case={case.id}; got "
        f"{len(tool_calls)}: {tool_calls!r}"
    )
    entry = tool_calls[0]
    payload = entry if isinstance(entry, str) else _stringify_structured(entry)
    assert case.expected_function_name in payload, (
        f"Function name {case.expected_function_name!r} missing from "
        f"tool_calls entry for case={case.id}; got={entry!r}"
    )
    assert case.expected_args_marker in payload, (
        f"Args marker {case.expected_args_marker!r} missing from "
        f"tool_calls entry for case={case.id}; got={entry!r}"
    )
