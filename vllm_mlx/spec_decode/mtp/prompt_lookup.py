# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX context_copy.py.
# Copyright 2026 Youssof Altoukhi and MTPLX contributors.
"""Prompt-lookup proposals for high-overlap speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptLookupMatch:
    """A continuation found in the immutable prompt."""

    start: int
    matched_suffix: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class PromptLookupPolicy:
    """Model-qualified prompt lookup policy captured at request start."""

    enabled_by_default: bool = False
    min_ngram: int = 8
    max_ngram: int = 10
    max_tokens: int = 24

    def __post_init__(self) -> None:
        if self.min_ngram < 2:
            raise ValueError("min_ngram must be at least 2")
        if self.max_ngram < self.min_ngram:
            raise ValueError("max_ngram must be >= min_ngram")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


class PromptLookupIndex:
    """Index prompt n-grams and propose their following token block.

    Generated text is used only as the lookup query.  It is never added to
    the index, which prevents self-repetition from turning into a fast loop.
    """

    def __init__(
        self,
        prompt: list[int] | tuple[int, ...],
        *,
        min_ngram: int = 6,
        max_ngram: int = 10,
        max_candidates: int = 32,
    ) -> None:
        if min_ngram < 2:
            raise ValueError("min_ngram must be at least 2")
        if max_ngram < min_ngram:
            raise ValueError("max_ngram must be >= min_ngram")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        self.prompt = tuple(int(token) for token in prompt)
        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.max_candidates = int(max_candidates)
        self._positions: dict[tuple[int, ...], list[int]] = {}

        # Store the position immediately following each n-gram.  A position at
        # len(prompt) has no continuation and is intentionally excluded.
        for end in range(self.min_ngram, len(self.prompt)):
            gram = self.prompt[end - self.min_ngram : end]
            self._positions.setdefault(gram, []).append(end)

    def propose(
        self,
        generated: list[int] | tuple[int, ...],
        *,
        max_tokens: int = 24,
    ) -> PromptLookupMatch | None:
        """Return the best prompt continuation for ``generated``'s suffix."""

        if max_tokens < 1 or len(generated) < self.min_ngram:
            return None
        suffix = tuple(int(token) for token in generated[-self.min_ngram :])
        candidates = self._positions.get(suffix)
        if not candidates:
            return None

        best_start: int | None = None
        best_suffix = self.min_ngram - 1
        max_extension = self.max_ngram - self.min_ngram
        history_len = len(generated)
        for start in reversed(candidates[-self.max_candidates :]):
            extension = 0
            while (
                extension < max_extension
                and start - self.min_ngram - 1 - extension >= 0
                and history_len - self.min_ngram - 1 - extension >= 0
                and self.prompt[start - self.min_ngram - 1 - extension]
                == generated[history_len - self.min_ngram - 1 - extension]
            ):
                extension += 1
            matched = self.min_ngram + extension
            if matched > best_suffix:
                best_start = start
                best_suffix = matched
                if extension == max_extension:
                    break

        if best_start is None:
            return None
        proposal = self.prompt[best_start : best_start + max_tokens]
        if not proposal:
            return None
        return PromptLookupMatch(best_start, best_suffix, proposal)


__all__ = ["PromptLookupIndex", "PromptLookupMatch", "PromptLookupPolicy"]


# A copy-draft round is judged against the speculative rounds it replaces,
# in committed tokens per millisecond. Both series are wall-time ratios that
# have to track *within* one response -- the same model and machine can be
# worth +30% on a rename turn and a small loss on an annotate turn, because
# the two turns copy different amounts -- so this borrows the cost model's
# faster alpha rather than the acceptance model's slow one.
COPY_DRAFT_GATE_EWMA_ALPHA = 0.3

# Rounds of each kind required before the comparison is trusted. A copy-draft
# is only ever refused on measurement, never on a prior, so the gate stays
# open until it has seen this many of both.
COPY_DRAFT_GATE_MIN_SAMPLES = 3

# Once refused, re-measure every N rounds so a decision cannot outlive the
# text that justified it: a turn that stops quoting the prompt and starts
# quoting it again has to be able to win the gate back. Doubles on each
# probe that changes nothing, up to the cap, and resets to the base the
# moment copy-drafts win on merit -- the same backoff shape the depth
# controller's starvation probe uses, for the same reason.
COPY_DRAFT_GATE_PROBE_INTERVAL = 8
COPY_DRAFT_GATE_PROBE_INTERVAL_MAX = 256


class CopyDraftGate:
    """Refuse copy-drafts on turns where the verify block costs more than it
    commits.

    A copy-draft is free to propose and its rejected tail costs nothing to
    undo, which makes it tempting to treat as strictly-positive. It is not:
    the proposal is verified in one ``W + 1`` row target forward, and that
    forward is charged whether the match held for 20 rows or 2. Measured on
    Qwen3.6-27B-4bit (M3 Ultra, production depth controller, 800-token
    answers), copy-drafts accepted ~4 rows per proposal and cost 2.5-4.1%
    of throughput; the same build on Qwen3.8-27B-4bit (M4 Pro) accepted
    ~10-14 rows per proposal and gained 28-33%. Even on one model the sign
    flips per turn: a rename turn gained 20% where an annotate turn lost 3%.

    So the decision cannot be made per model, per machine, or per family --
    only against what this turn's copy-drafts are actually returning. The
    gate keeps two throughput EWMAs in committed tokens per millisecond,
    one over copy-draft rounds and one over the speculative rounds they
    displace, and admits a proposal while the first is at least the second.

    Per request by construction: the state describes the text being
    generated, which is what the sign depends on, and a fresh request
    starts by measuring rather than inheriting another turn's verdict.
    """

    __slots__ = (
        "_copy_rate",
        "_copy_seen",
        "_base_rate",
        "_base_seen",
        "_probe_interval",
        "_since_probe",
        "declines",
        "probes",
    )

    def __init__(self) -> None:
        self._copy_rate = 0.0
        self._copy_seen = 0
        self._base_rate = 0.0
        self._base_seen = 0
        self._probe_interval = COPY_DRAFT_GATE_PROBE_INTERVAL
        self._since_probe = 0
        # Diagnostics, surfaced through the generator's timing stats so a
        # turn that got no copy-drafts can be told apart from a turn whose
        # copy-drafts were measured and declined.
        self.declines = 0
        self.probes = 0

    def observe(self, *, is_copy_draft: bool, committed: int, round_ms: float) -> None:
        """Fold one round's realized throughput into the matching series.

        Args:
            is_copy_draft: whether the round verified a copy-draft.
            committed: tokens the round actually delivered to the caller.
            round_ms: wall time charged to the round, including any drafter
                cost carried into it (a copy-draft carries none).
        """
        if committed <= 0 or round_ms <= 0.0:
            return
        rate = committed / round_ms
        if is_copy_draft:
            if self._copy_seen == 0:
                self._copy_rate = rate
            else:
                self._copy_rate += COPY_DRAFT_GATE_EWMA_ALPHA * (rate - self._copy_rate)
            self._copy_seen += 1
        else:
            if self._base_seen == 0:
                self._base_rate = rate
            else:
                self._base_rate += COPY_DRAFT_GATE_EWMA_ALPHA * (rate - self._base_rate)
            self._base_seen += 1

    def allow(self) -> bool:
        """Whether a copy-draft may be proposed for the upcoming round."""
        if (
            self._copy_seen < COPY_DRAFT_GATE_MIN_SAMPLES
            or self._base_seen < COPY_DRAFT_GATE_MIN_SAMPLES
        ):
            # Undersampled: the only way to learn a copy-draft's value is to
            # verify one, so an unproven gate never blocks.
            return True
        if self._copy_rate >= self._base_rate:
            self._probe_interval = COPY_DRAFT_GATE_PROBE_INTERVAL
            self._since_probe = 0
            return True
        self._since_probe += 1
        if self._since_probe >= self._probe_interval:
            self._since_probe = 0
            self._probe_interval = min(
                self._probe_interval * 2, COPY_DRAFT_GATE_PROBE_INTERVAL_MAX
            )
            self.probes += 1
            return True
        self.declines += 1
        return False

    def rates(self) -> tuple[float, float]:
        """``(copy_draft, baseline)`` throughput in tokens per millisecond."""
        return self._copy_rate, self._base_rate
