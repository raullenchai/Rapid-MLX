# SPDX-License-Identifier: Apache-2.0
"""Conservative token-level guard for runaway agent repetitions.

This intentionally operates on token ids rather than decoded text.  Exact token
periodicity is cheap to test, independent of tokenizer whitespace behaviour, and
avoids fuzzy heuristics that could truncate legitimate prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RepetitionMatch:
    """Description of the repeated suffix that caused a stop."""

    period_tokens: int
    repeats: int


def detect_repeated_token_suffix(
    token_ids: Sequence[int],
    *,
    min_period_tokens: int = 6,
    max_period_tokens: int = 128,
    required_repeats: int = 6,
    min_repeated_tokens: int = 72,
    max_window_tokens: int = 768,
) -> RepetitionMatch | None:
    """Return an exact periodic-suffix match, or ``None``.

    The defaults require six adjacent copies and at least 72 repeated tokens.
    One-token stutters (punctuation, newlines, or intentional ``ha ha`` output)
    therefore cannot trip the guard.  Work is bounded to the most recent 768
    tokens and callers are expected to invoke it only every few decode steps.
    """

    if required_repeats < 2 or min_period_tokens < 1:
        raise ValueError("invalid repetition guard thresholds")

    tokens = token_ids[-max_window_tokens:]
    max_period = min(max_period_tokens, len(tokens) // required_repeats)
    for period in range(min_period_tokens, max_period + 1):
        repeated_tokens = period * required_repeats
        if repeated_tokens < min_repeated_tokens:
            continue
        suffix = tokens[-repeated_tokens:]
        pattern = suffix[:period]
        # Do not disguise a one-token/newline stutter as a longer period
        # merely because the same token also repeats in 12-token chunks.
        if any(
            period % smaller == 0 and pattern == pattern[:smaller] * (period // smaller)
            for smaller in range(1, min_period_tokens)
        ):
            continue
        if all(
            suffix[offset : offset + period] == pattern
            for offset in range(period, repeated_tokens, period)
        ):
            return RepetitionMatch(period_tokens=period, repeats=required_repeats)
    return None
