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
    # #447 main bug (Case B from issue): bare ``thought`` opens a
    # thought block at the start of generation, no closing
    # ``<channel|>`` and no ``final`` transition. Post-fix: bare
    # ``thought`` (in INIT state) triggers transition to THINKING; the
    # body lands in reasoning and content stays empty. INIT-state
    # gating keeps the fix narrow — see output_router.py:343 comment
    # for the rationale (mid-content body tokens matching ``final`` /
    # ``content`` / ``thought`` IDs must NOT be silently swallowed).
    _Case(
        id="issue_447_thought_word_only_no_final",
        token_ids=[
            _V["thought"],
            _V["\n"],
            _V["message_body"],
            _V["<eos>"],
        ],
        expected_content=None,
        expected_reasoning="message_body",
    ),
]


# Codex re-review BLOCKING (2026-06-04): the bare-INIT gate now uses
# 1-token lookahead. A response that legitimately STARTS with one of
# the channel-word tokens (``thought`` / ``content`` / ``final``)
# followed by NON-whitespace must surface the word as content rather
# than be silently swallowed. The lookahead in ``_drain_pending_init_word``
# rolls back to emit the buffered token and pushes the current token
# onto the redo queue for the next ``feed()`` call. The rare case
# matters because a user-instructed model ("Begin your reply with the
# literal word 'final' followed by ...") or a quirky training quirk
# would otherwise drop the first token without trace.
LOOKAHEAD_ROLLBACK_CASES: list[_Case] = [
    _Case(
        id="legit_final_then_content_not_swallowed",
        token_ids=[
            _V["final"],
            _V[" world"],
            _V["<eos>"],
        ],
        # Both ``final`` and `` world`` must reach content — neither
        # silently swallowed by the bare-INIT gate.
        expected_content="final world",
        expected_reasoning=None,
    ),
    _Case(
        id="legit_content_then_body_not_swallowed",
        token_ids=[
            _V["content"],
            _V["Hello"],
            _V["<eos>"],
        ],
        expected_content="contentHello",
        expected_reasoning=None,
    ),
    _Case(
        id="legit_thought_then_body_not_swallowed",
        token_ids=[
            _V["thought"],
            _V["Hello"],
            _V["<eos>"],
        ],
        # Without ``\n`` after ``thought``, lookahead rejects the
        # channel intent — the buffered word is just literal content
        # (codex re-review BLOCKING: routing rejected bare words to
        # reasoning was hiding valid user-visible content). Emit as
        # CONTENT and let the body follow in the CONTENT state.
        expected_content="thoughtHello",
        expected_reasoning=None,
    ),
    _Case(
        id="bare_word_only_no_followup",
        token_ids=[
            _V["final"],
            _V["<eos>"],
        ],
        # Stream ends with only the bare word — finalize() emits it
        # as content rather than swallow.
        expected_content="final",
        expected_reasoning=None,
    ),
]


# Compound bare-word sequence (#447 Case A in the issue body) — model
# emits bare ``thought`` / ``final`` MID-stream after exiting the first
# channel. The INIT-only gate (see output_router.py) deliberately does
# NOT fire bare-word transitions outside INIT state, because that would
# regress canonical Gemma 4 content bodies that happen to start with the
# literal ``final`` / ``content`` / ``thought`` token (codex round-2
# BLOCKING). Tracked for the marker-preserving router followup.
KNOWN_LIMITATION_CASES: list[_Case] = [
    _Case(
        id="issue_447_compound_bare_words_known_limitation",
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
def test_gemma4_router_no_channel_marker_leaks(case: _Case, router):
    # Flipped from xfail strict → passing by the cluster fix's bare
    # Gemma 4 channel-word handling in output_router.py. Before the
    # default emit, the router checks for thought_word / content_word
    # / final_word IDs ONLY in INIT state and treats them as channel
    # transitions. Gated to INIT to avoid eating body tokens that
    # happen to match the channel-type word IDs.
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )


# ----- Lookahead rollback (codex re-review BLOCKING) --------------------


@pytest.mark.parametrize("case", LOOKAHEAD_ROLLBACK_CASES, ids=lambda c: c.id)
def test_gemma4_router_lookahead_rollback_preserves_legit_first_token(
    case: _Case, router
):
    """Pin: the INIT-state lookahead doesn't lose a literal first token.

    When a model legitimately starts a reply with the lowercase token
    ``thought`` / ``content`` / ``final`` followed by non-whitespace,
    the lookahead rolls back to emit the buffered token as the channel
    indicated by its semantic meaning, then continues processing the
    body in that channel. Without this guard, the prior implementation
    silently swallowed the first token (codex re-review BLOCKING).

    The bare word ALSO sets channel state — that's intentional: if the
    model speaks one of the channel words at INIT, treat it as
    expressing channel intent for whatever follows, even if the
    structural newline confirmation is missing. The alternative
    (always route to CONTENT post-rollback) would let a buggy ``thought
    body`` emit silently lose the reasoning marker.
    """
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content), (
        f"content mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_content)!r}, got={result['content']!r}"
    )
    assert result["reasoning"] == _normalize(case.expected_reasoning), (
        f"reasoning mismatch for case={case.id}: expected="
        f"{_normalize(case.expected_reasoning)!r}, got={result['reasoning']!r}"
    )


# ----- Known limitation: compound bare-word sequences (xfail) -----------


@pytest.mark.parametrize("case", KNOWN_LIMITATION_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Compound bare-word sequence (#447 Case A): the model emits "
        "bare ``thought`` followed later by bare ``final`` mid-stream "
        "after exiting the first channel. The INIT-only bare-word gate "
        "in output_router.py deliberately does NOT fire transitions "
        "outside INIT state — broadening the gate regresses canonical "
        "Gemma 4 bodies whose first content token happens to be "
        "``final`` / ``content`` / ``thought`` (codex round-2 BLOCKING). "
        "Marker-preserving router followup will resolve this without "
        "the trade-off."
    ),
    strict=True,
)
def test_gemma4_router_compound_bare_words_known_limitation(case: _Case, router):
    result = router.feed_sequence(case.token_ids)

    assert result["content"] == _normalize(case.expected_content)
    assert result["reasoning"] == _normalize(case.expected_reasoning)
