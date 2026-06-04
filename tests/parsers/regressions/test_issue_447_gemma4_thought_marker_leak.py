# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #447 — Gemma 4 OutputRouter ``thought`` literal leak.

Reported 2026-05-22 on ``mlx-community/gemma-4-26b-a4b-it-4bit`` (and
likely the whole Gemma 4 family). On the unconstrained generation path,
the literal token ``thought`` plus the analysis body land in
``message.content`` while ``reasoning_content`` stays empty. JSON mode
works because constrained decoding bypasses the channel state machine.

## Root cause

``vllm_mlx/output_router.py`` AWAITING_CHANNEL_TYPE entry is gated on
``token_id == m.channel_start`` (the ``<|channel>`` special token, ID
100 in Gemma 4's vocab). The ``thought`` / ``content`` / ``final``
literal-word checks at lines 240-251 are INSIDE the
``AWAITING_CHANNEL_TYPE`` block — so if the model emits the channel-
type word *without* a preceding ``<|channel>``, the router never
transitions and the word falls through to the default route at lines
306-311, which emits it as literal ``Channel.CONTENT`` text.

## Scope caveat

Tests the OutputRouter state machine in isolation against a synthetic
vocab via ``FakeTokenizer``. The end-to-end route (engine ->
``_stream_with_output_router`` -> route postprocessor) is downstream of
this surface; the cluster fix's router patch is verified here, while
route-level coverage lives in #455's regression file.

The issue body also describes a *streaming* variant where every delta
is routed to reasoning and content stays empty. That's a different
state-machine trap (router stuck in THINKING) — covered as a separate
xfail in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_mlx.output_router import OutputRouter

from ..fake_tokenizer import GEMMA4_VOCAB, gemma4_fake_tokenizer


@dataclass
class _Case:
    id: str
    token_ids: list[int]
    expected_content: str | None
    expected_reasoning: str | None


_V = GEMMA4_VOCAB


# ----- Sanity (works today): Gemma 4 protocol with explicit markers -----


SANITY_CASES: list[_Case] = [
    # Canonical Gemma 4 protocol — channel-type word comes RIGHT AFTER
    # ``<|channel>`` (router transitions to THINKING/CONTENT), then the
    # body, then ``<channel|>`` closes thinking back to CONTENT. Format
    # matches ``test_output_router.py::test_thought_then_content_channel``.
    _Case(
        id="canonical_thought_then_final",
        token_ids=[
            _V["<|channel>"],
            _V["thought"],
            _V["analysis_body"],
            _V["<channel|>"],
            _V["<|channel>"],
            _V["final"],
            _V["message_body"],
            _V["<eos>"],
        ],
        expected_content="message_body",
        expected_reasoning="analysis_body",
    ),
    # Pure content (no thought block) — just opens the content channel
    # and emits the body. Should route everything to content.
    _Case(
        id="canonical_final_only",
        token_ids=[
            _V["<|channel>"],
            _V["final"],
            _V["Hello"],
            _V[" world"],
            _V["<eos>"],
        ],
        expected_content="Hello world",
        expected_reasoning=None,
    ),
]


# ----- #447 BUG: model emits channel words without `<|channel>` -----


BUG_CASES: list[_Case] = [
    # #447 Case A (multi-turn): model emits the literal ``thought`` /
    # ``content`` / ``final`` tokens directly without the
    # ``<|channel>`` opener. Router never enters AWAITING_CHANNEL_TYPE,
    # so the literal channel-type words fall through to default and
    # leak as content. Expected (after fix): ``thought`` triggers
    # transition to THINKING, ``final`` transitions to CONTENT, body
    # routes correctly.
    _Case(
        id="issue_447_no_channel_open_marker",
        token_ids=[
            _V["thought"],
            _V["\n"],
            _V["analysis_body"],
            _V["\n"],
            _V["final"],
            _V["\n"],
            _V["message_body"],
            _V["<eos>"],
        ],
        expected_content="message_body",
        expected_reasoning="analysis_body",
    ),
    # Variant: bare ``thought`` opens a thought block with no closing
    # ``<channel|>`` and no ``final`` transition. Post-fix: bare
    # ``thought`` should transition to THINKING and the body lands in
    # reasoning, content stays empty (matches the "stuck in thought"
    # trap from issue #447 Case B, except verified at router level
    # rather than via streaming-delta accumulation).
    _Case(
        id="issue_447_thought_word_only_no_final",
        token_ids=[
            _V["thought"],
            _V["\n"],
            _V["message_body"],
            _V["<eos>"],
        ],
        # After fix: bare `thought` enters THINKING; without a `final`
        # transition the body is reasoning and content is empty.
        expected_content=None,
        expected_reasoning="message_body",
    ),
]


@pytest.fixture
def router() -> OutputRouter:
    """Fresh Gemma 4 OutputRouter wired to a synthetic tokenizer."""
    fake_tok = gemma4_fake_tokenizer()
    r = OutputRouter.from_tokenizer(fake_tok)
    assert r is not None, (
        "OutputRouter.from_tokenizer returned None on the synthetic "
        "Gemma 4 vocab — discovery is broken or the vocab is missing "
        "required tokens (<|channel>, <|tool_call>)."
    )
    assert r.map.format_tag == "gemma4", (
        f"Expected Gemma 4 discovery; got format_tag={r.map.format_tag!r}"
    )
    return r


def _normalize(value: str | None) -> str | None:
    """Map empty string to None to match feed_sequence's contract.

    ``OutputRouter.feed_sequence`` returns ``{"content": str or None,
    "reasoning": str or None}``; empty/whitespace results collapse to
    None. Mirror that in expectations.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


# ----- Sanity tests (pin working behavior) ------------------------------


@pytest.mark.parametrize("case", SANITY_CASES, ids=lambda c: c.id)
def test_gemma4_router_sanity(case: _Case, router):
    """Pin: canonical Gemma 4 protocol routes correctly today."""
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )


# ----- Bug tests (xfail until cluster fix lands) ------------------------


@pytest.mark.parametrize("case", BUG_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Issue #447 — Gemma 4 OutputRouter only enters "
        "AWAITING_CHANNEL_TYPE on the `<|channel>` special token. If "
        "the model emits the channel-type word (`thought`/`content`/"
        "`final`) without a preceding `<|channel>`, the literal word "
        "tokens fall through to the default route at output_router.py:"
        "306-311 and leak as content. JSON mode works because "
        "constrained decoding bypasses the state machine. Cluster fix: "
        "treat bare `thought`/`content`/`final` from any non-"
        "AWAITING_CHANNEL_TYPE state (including INIT, CONTENT, and "
        "THINKING — Case A's bare `final` arrives while THINKING) as "
        "channel-transition triggers, not literal tokens. Flip to "
        "passing once the router patch lands."
    ),
    strict=True,
)
def test_gemma4_router_no_channel_marker_leaks(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )
