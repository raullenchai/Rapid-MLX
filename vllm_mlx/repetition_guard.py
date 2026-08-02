# SPDX-License-Identifier: Apache-2.0
"""Conservative token-level guard for runaway agent repetitions.

This intentionally operates on token ids rather than decoded text.  Exact token
periodicity is cheap to test, independent of tokenizer whitespace behaviour, and
avoids fuzzy heuristics that could truncate legitimate prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import mlx.core as mx


@dataclass(frozen=True)
class RepetitionMatch:
    """Description of the repeated suffix that caused a stop."""

    period_tokens: int
    repeats: int


@dataclass(frozen=True)
class RepetitionIntervention:
    """The next token that would extend a near-certain exact loop."""

    period_tokens: int
    repeats: int
    blocked_token_id: int


def detect_repeated_token_suffix(
    token_ids: Sequence[int],
    *,
    min_period_tokens: int = 1,
    max_period_tokens: int = 128,
    required_repeats: int = 3,
    min_repeated_tokens: int = 72,
    min_short_period_tokens: int = 256,
    max_window_tokens: int = 768,
) -> RepetitionMatch | None:
    """Return an exact periodic-suffix match, or ``None``.

    The defaults require at least three adjacent copies and at least 72 repeated
    tokens.  The effective repeat count is adaptive: a 12-token sentence needs
    six copies, while a 61-token paragraph needs only three.  Periods below six
    tokens use the higher ``min_short_period_tokens`` threshold, tolerating
    intentional stutters while still catching the sustained one-token CJK loop
    that previously ran for 10k+ tokens and exhausted Metal resource handles.
    Work is bounded to the most recent 768 tokens and callers are expected to
    invoke it only every few decode steps.
    """

    if required_repeats < 2 or min_period_tokens < 1:
        raise ValueError("invalid repetition guard thresholds")

    tokens = token_ids[-max_window_tokens:]
    max_period = min(max_period_tokens, len(tokens) // required_repeats)
    for period in range(min_period_tokens, max_period + 1):
        repeated_floor = min_short_period_tokens if period < 6 else min_repeated_tokens
        repeats = max(required_repeats, ceil(repeated_floor / period))
        repeated_tokens = period * repeats
        if repeated_tokens > len(tokens):
            continue
        suffix = tokens[-repeated_tokens:]
        pattern = suffix[:period]
        # Evaluate a loop at its primitive period.  Otherwise 200 copies of a
        # single token would masquerade as a 6-token pattern and inherit the
        # lower long-period threshold.
        if any(
            period % smaller == 0 and pattern == pattern[:smaller] * (period // smaller)
            for smaller in range(1, period)
        ):
            continue
        if all(
            suffix[offset : offset + period] == pattern
            for offset in range(period, repeated_tokens, period)
        ):
            return RepetitionMatch(period_tokens=period, repeats=repeats)
    return None


def predict_repeated_token_suffix(
    token_ids: Sequence[int],
    *,
    min_period_tokens: int = 1,
    max_period_tokens: int = 128,
    required_repeats: int = 3,
    min_repeated_tokens: int = 72,
    min_short_period_tokens: int = 256,
    max_window_tokens: int = 768,
) -> RepetitionIntervention | None:
    """Return the one token that would complete the guard's next loop copy.

    This is the preventative counterpart to :func:`detect_repeated_token_suffix`.
    It waits until one copy before the hard-stop threshold, then identifies the
    exact next token implied by the periodic suffix.  Masking only that token is
    enough to let sampling escape without aborting an already-streaming request.
    """

    if required_repeats < 2 or min_period_tokens < 1:
        raise ValueError("invalid repetition guard thresholds")
    tokens = token_ids[-max_window_tokens:]
    max_period = min(max_period_tokens, len(tokens) // (required_repeats - 1))
    for period in range(min_period_tokens, max_period + 1):
        repeated_floor = min_short_period_tokens if period < 6 else min_repeated_tokens
        stop_repeats = max(required_repeats, ceil(repeated_floor / period))
        precursor_repeats = stop_repeats - 1
        precursor_tokens = period * precursor_repeats
        if precursor_tokens > len(tokens):
            continue
        suffix = tokens[-precursor_tokens:]
        pattern = suffix[:period]
        if any(
            period % smaller == 0 and pattern == pattern[:smaller] * (period // smaller)
            for smaller in range(1, period)
        ):
            continue
        if all(
            suffix[offset : offset + period] == pattern
            for offset in range(period, precursor_tokens, period)
        ):
            return RepetitionIntervention(
                period_tokens=period,
                repeats=precursor_repeats,
                blocked_token_id=pattern[0],
            )
    return None


class AgentRepetitionLogitsProcessor:
    """Break exact agent loops before the scheduler has to abort the stream.

    The hot path only scans a bounded Python token-id suffix.  An MLX mask is
    allocated solely when an intervention is required, so normal decoding pays
    no per-token GPU work.  A small intervention cap leaves the existing hard
    stop authoritative for a model that repeatedly falls back into a loop.
    """

    def __init__(self, output_token_ids: list[int], *, max_interventions: int = 3):
        self.output_token_ids = output_token_ids
        self.max_interventions = max_interventions
        self.interventions = 0
        self.last_match: RepetitionIntervention | None = None
        self._last_intervention_length = -1

    def __call__(self, _tokens: mx.array, logits: mx.array) -> mx.array:
        if self.interventions >= self.max_interventions:
            return logits
        history_length = len(self.output_token_ids)
        if history_length == self._last_intervention_length:
            return logits
        match = predict_repeated_token_suffix(self.output_token_ids)
        if match is None:
            return logits
        token_id = match.blocked_token_id
        if token_id < 0 or token_id >= logits.shape[-1]:
            return logits
        self.interventions += 1
        self.last_match = match
        self._last_intervention_length = history_length
        token_mask = mx.arange(logits.shape[-1]) == token_id
        return mx.where(token_mask[None, :], -mx.inf, logits)
