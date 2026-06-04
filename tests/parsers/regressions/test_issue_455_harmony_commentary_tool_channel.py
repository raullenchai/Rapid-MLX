# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #455 — harmony ``commentary`` channel leaks as content.

Reported 2026-05-23 on ``mlx-community/gpt-oss-20b-MXFP4-Q8``. The
Anthropic ``/v1/messages`` streaming path emits
``content_block_start type=text`` containing raw harmony commentary
tokens (``commentary to=functions.X json{...}``) instead of
``content_block_start type=tool_use`` + ``input_json_delta`` events.
Non-stream works correctly, so the bug is isolated to the streaming
route's handling of the tool-call channel.

## Root cause

``vllm_mlx/output_router.py`` ``AWAITING_CHANNEL_TYPE`` handling for
the harmony style (lines 222-238) only recognizes two channel-type
words:

* ``analysis`` → REASONING
* ``final``    → CONTENT

The harmony tool-call channel emits ``commentary`` followed by a
recipient marker (`` to=functions.X``) and an optional `` json``
constrain directive, then ``<|message|>`` and the JSON body. The
router falls into the default branch at lines 234-238 — it transitions
to CONTENT and emits the literal token text (``commentary``) — which
is why the Anthropic route then surfaces the whole commentary block as
``text`` deltas rather than ``tool_use`` blocks.

## Scope caveat

Router-level regression only. The full #455 fix also touches the
Anthropic streaming route (``routes/anthropic.py``) so the
``content_block_start`` payload switches to ``type=tool_use`` and the
JSON arguments stream as ``input_json_delta`` events. Route-level
coverage requires a separate end-to-end fixture that mocks the engine
+ route handler; deferred. This file pins the router contract so that
once the harmony commentary handling lands, downstream layers receive
the correct ``Channel.TOOL_CALL`` events to build the Anthropic
``tool_use`` block from.

## Sibling issues

Same OutputRouter channel-classification gap as:
* #444 — harmony streaming tool calls leak as content deltas on the
  OpenAI route. Covered in
  ``test_issue_444_harmony_tool_call_leak.py`` at the parser layer;
  this file is the router-layer companion.
* #468 — ``tool_choice="required"`` triggers the same commentary leak
  with extra constrained-decoding hop. Pinned separately because the
  fix interacts with the JSON-schema constraint, not just the channel
  classification.
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
    expected_content: str | None
    expected_reasoning: str | None
    expected_tool_calls: list[str] | None


_V = HARMONY_VOCAB


# ----- Sanity (works today): analysis + final channels ------------------


SANITY_CASES: list[_Case] = [
    # Canonical analysis channel — reasoning extracted, content stays None.
    _Case(
        id="canonical_analysis_only",
        token_ids=[
            _V["<|channel|>"],
            _V["analysis"],
            _V["<|message|>"],
            _V["Reason"],
            _V["ing"],
            _V["<|end|>"],
        ],
        expected_content=None,
        expected_reasoning="Reasoning",
        expected_tool_calls=None,
    ),
    # Canonical final channel — content extracted, reasoning stays None.
    _Case(
        id="canonical_final_only",
        token_ids=[
            _V["<|channel|>"],
            _V["final"],
            _V["<|message|>"],
            _V["Answer"],
            _V["<|return|>"],
        ],
        expected_content="Answer",
        expected_reasoning=None,
        expected_tool_calls=None,
    ),
    # Mixed: analysis → final. Same flow used by gpt-oss for
    # think-then-answer conversations.
    _Case(
        id="canonical_analysis_then_final",
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
            _V["final"],
            _V["<|message|>"],
            _V["Answer"],
            _V["<|return|>"],
        ],
        expected_content="Answer",
        expected_reasoning="Reasoning",
        expected_tool_calls=None,
    ),
]


# ----- #455 BUG: harmony commentary channel for tool calls --------------


BUG_CASES: list[_Case] = [
    # Issue #455 verbatim repro shape — `commentary to=functions.calculate
    # json{...}<|call|>` should produce a TOOL_CALL channel event with
    # just the JSON body. Today the router falls into the default
    # AWAITING_CHANNEL_TYPE branch and emits "commentary" + the
    # recipient + the body as CONTENT text.
    _Case(
        id="issue_455_calculate_tool_call",
        token_ids=[
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.calculate"],
            _V[" json"],
            _V["<|message|>"],
            _V['{"expression":"17*23"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_content=None,
        expected_reasoning=None,
        expected_tool_calls=['{"expression":"17*23"}'],
    ),
    # Different prompt + tool name — pins the fix against any
    # accidental hardcoding of the calculate path.
    _Case(
        id="get_weather_tool_call",
        token_ids=[
            _V["<|channel|>"],
            _V["commentary"],
            _V[" to=functions.get_weather"],
            _V[" json"],
            _V["<|message|>"],
            _V['{"city":"Tokyo"}'],
            _V["<|call|>"],
            _V["<|endoftext|>"],
        ],
        expected_content=None,
        expected_reasoning=None,
        expected_tool_calls=['{"city":"Tokyo"}'],
    ),
]


@pytest.fixture
def router() -> OutputRouter:
    """Fresh harmony OutputRouter wired to a synthetic tokenizer."""
    fake_tok = harmony_fake_tokenizer()
    r = OutputRouter.from_tokenizer(fake_tok)
    assert r is not None, (
        "OutputRouter.from_tokenizer returned None on the synthetic "
        "harmony vocab — discovery is broken or the vocab is missing "
        "required tokens (<|channel|>, <|message|>)."
    )
    assert r.map.format_tag == "harmony", (
        f"Expected harmony discovery; got format_tag={r.map.format_tag!r}"
    )
    return r


def _normalize_str(value: str | None) -> str | None:
    """Map empty/whitespace string to None (feed_sequence contract)."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


# ----- Sanity tests ------------------------------------------------------


@pytest.mark.parametrize("case", SANITY_CASES, ids=lambda c: c.id)
def test_harmony_router_sanity(case: _Case, router):
    """Pin: harmony analysis + final channels route correctly today."""
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize_str(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize_str(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize_str(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize_str(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )
    assert result["tool_calls"] == case.expected_tool_calls, (
        f"tool_calls mismatch for case={case.id}: expected="
        f"{case.expected_tool_calls!r}, got={result['tool_calls']!r}"
    )


# ----- Bug tests (xfail until cluster fix lands) ------------------------


@pytest.mark.parametrize("case", BUG_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Issue #455 — OutputRouter's harmony AWAITING_CHANNEL_TYPE "
        "handling (output_router.py:222-238) only recognizes "
        "``analysis``/``final`` as channel-type words. The tool-call "
        "channel emits ``commentary`` followed by `` to=functions.X`` + "
        "an optional `` json`` constrain directive, then the body. The "
        "router falls into the default branch, transitions to CONTENT, "
        "and leaks ``commentary`` + recipient + body as content text. "
        "Cluster fix: add ``harmony_commentary_word`` to TokenMap and "
        "extend the AWAITING_CHANNEL_TYPE branch to transition to "
        "TOOL_CALL on ``commentary``, swallowing the recipient + "
        "constrain directive metadata until ``<|message|>``. "
        "Anthropic-route content_block translation (the user-facing "
        "tool_use vs text issue) is downstream and out of scope for "
        "this router-level test."
    ),
    strict=True,
)
def test_harmony_router_commentary_tool_call(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize_str(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize_str(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize_str(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize_str(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )
    assert result["tool_calls"] == case.expected_tool_calls, (
        f"tool_calls mismatch for case={case.id}: expected="
        f"{case.expected_tool_calls!r}, got={result['tool_calls']!r}"
    )
