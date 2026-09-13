# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX context_copy.py.
# Copyright 2026 Youssof Altoukhi and MTPLX contributors.
"""Prompt-lookup proposals for high-overlap speculative decoding."""

from __future__ import annotations

from collections import deque
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


# A copy-draft of N tokens is verified in one N+1 row forward, and MLX's
# ``quantized_matmul`` tiles those rows in blocks of 32. Crossing that edge
# buys a second tile for a single row: measured on Qwen3.8-27B-4bit, a
# 32-row verify costs 342.26 ms and a 33-row verify 647.52 ms (+89%). The
# same step is in the kernel alone on both an M3 Ultra and an M4 Pro
# (17408x5120 4-bit g64: 655.9 -> 1034.9 us and 1442.8 -> 2567.2 us at the
# same M), so it is MLX's tiling rather than one GPU's shape. 31 proposed
# tokens is the widest block that stays inside one tile, and 63 and 95 are
# the widest that stay inside two and three -- a proposal just past any of
# those edges pays for a whole tile it does not fill.
#
# It is a ceiling on an opt-in policy, not a claim about every backend. Only
# a family whose injector sets ``PromptLookupPolicy`` reaches it at all, and
# the one that does ships quantized; a request's own ``max_tokens`` still
# narrows it further. It is also not a target: ``CopyDraftGate.width_cap``
# only proposes a width this turn's copies have earned, so on a model or a
# future MLX kernel whose tile sits elsewhere the cap costs at most the
# width of the flat band it was measured in -- about 3% of one verify -- and
# not a mis-sized block every round.
#
# It lives beside the policy rather than beside the generator that enforces
# it because the per-family policy has to name it, and this module is the
# only part of the copy-draft path that imports without MLX: the family
# capability and self-contained-repo tests import ``qwen3_5_inject`` on a
# CI lane that has no MLX at all.
COPY_DRAFT_TILE_ROWS = 32
MAX_COPY_DRAFT_TOKENS = COPY_DRAFT_TILE_ROWS - 1


__all__ = [
    # The gate and its constants are part of the module's surface: the
    # generator imports the class, and the tile contract that pins
    # ``COPY_DRAFT_FREE_WIDTH_FLOOR`` below ``MAX_COPY_DRAFT_TOKENS`` reads
    # them by name. Listing them keeps a wildcard import and the public
    # inventory agreeing with what callers already use.
    "COPY_DRAFT_ACCEPTANCE_EWMA_ALPHA",
    "COPY_DRAFT_FREE_WIDTH_FLOOR",
    "COPY_DRAFT_GATE_MIN_SAMPLES",
    "COPY_DRAFT_GATE_PROBE_INTERVAL",
    "COPY_DRAFT_GATE_PROBE_INTERVAL_MAX",
    "COPY_DRAFT_GATE_WINDOW",
    "COPY_DRAFT_MIN_WIDTH",
    "COPY_DRAFT_PROBE_WIDTH",
    "COPY_DRAFT_TILE_ROWS",
    "COPY_DRAFT_WIDTH_HEADROOM",
    "MAX_COPY_DRAFT_TOKENS",
    "CopyDraftGate",
    "PromptLookupIndex",
    "PromptLookupMatch",
    "PromptLookupPolicy",
]


# How fast the sizer forgets. Acceptance has to be read with recency,
# because the measurement is censored by the width that produced it: a turn
# whose 8-row probes all filled up can only learn that its matches run
# further by widening, and a slow average would hold every later block down
# at the probe width. This is the cost model's alpha rather than the
# acceptance model's for that reason.
#
# The throughput comparison is deliberately NOT an EWMA -- see
# ``CopyDraftGate.rates``.
COPY_DRAFT_ACCEPTANCE_EWMA_ALPHA = 0.3

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

# Rounds of each kind the throughput comparison keeps. The window has to be
# long enough that one round cannot decide it -- a copy-draft whose first row
# misses delivers one token for a whole verify, and on a turn copy-drafts win
# by 20% an alpha-0.3 EWMA turned one such sample into 52 refused proposals --
# and short enough that a verdict cannot outlive the text that earned it: an
# answer whose prose section stops paying must be able to refuse without
# waiting out the code section that did. 32 rounds is ~1-3 s of generation,
# caps any one sample's weight at a few percent, and is bounded state per
# request rather than a running total of the whole turn.
COPY_DRAFT_GATE_WINDOW = 32

# Width of the first copy-drafts of a turn, before any have been verified.
# Acceptance is censored by the width that was proposed, so the first block
# has to be a guess, and below the tile edge the rows are not cheap: on
# Qwen3.8-27B-4bit a verify costs 66.8 ms at one row, 69.7 at two, 262.5 at
# nine and 342.3 at thirty-two. Two rows is the only width that costs a
# rounding error over the round it replaces, which is what a guess should
# cost -- an eight-row guess already spends three quarters of a full-width
# block, so guessing wide-ish buys nothing that guessing narrow and
# widening on evidence does not.
COPY_DRAFT_PROBE_WIDTH = 2

# Rows above which widening is free. ``quantized_matmul`` switches from the
# vector to the GEMM path and amortises one weight read over a whole 32-row
# tile, so a verify block is within ~3% of the same cost anywhere from here
# to the tile edge: measured on Qwen3.8-27B-4bit (M4 Pro) 13 rows 331.50 ms
# against 32 rows 342.26, and on Qwen3.6-27B-4bit (M3 Ultra) 133.99 against
# 138.40. Narrowing inside that band therefore buys nothing and can only
# truncate a match, so the sizer stops narrowing once the measurement has
# earned its way above this floor.
COPY_DRAFT_FREE_WIDTH_FLOOR = 13

# Headroom over the measured acceptance when sizing a block below the free
# band. A block sized to exactly what the last ones accepted can never
# discover that the match ran further; a block sized well past it pays for
# rows nothing will reach. 1.5x plus one row leaves room to ratchet upward
# while keeping the overshoot bounded.
COPY_DRAFT_WIDTH_HEADROOM = 1.5

# Never size a copy-draft below this. Two rows is the narrowest block that
# still commits a copied token, and the throughput gate -- not the sizer -- is
# what decides that copying has stopped being worth anything at all.
COPY_DRAFT_MIN_WIDTH = 2


def _sizing_counts(accepted: int | None, proposed: int | None) -> tuple[int, int]:
    """The two counts a copy-draft round must report, checked and narrowed.

    Split out of :meth:`CopyDraftGate.observe` so the checks and the
    narrowing they establish live together: everything below reads two
    integers, not two optional ones that happen to have been checked
    several branches earlier.

    Raises:
        TypeError: either count is absent. Zero is a real measurement -- a
            block whose first row missed -- so it cannot also stand in for
            an absent one.
        ValueError: the pair cannot describe a round. The accepted rows are
            a prefix of the proposed ones, so the pair is ordered by
            construction; checked anyway because both numbers steer the next
            block's width. ``accepted > proposed`` would mark the block
            saturated and double the width off a measurement that cannot
            have happened, and a negative acceptance would drag the EWMA
            below the floor the sizer assumes it stays above.
    """
    if accepted is None or proposed is None:
        raise TypeError(
            "observe(is_copy_draft=True) requires accepted= and "
            "proposed=: the sizer reads them to pick the next block's "
            "width, and there is no value that stands in for an "
            f"absent count (got accepted={accepted!r}, "
            f"proposed={proposed!r})"
        )
    if accepted < 0 or proposed < 0 or accepted > proposed:
        raise ValueError(
            "copy-draft round reported accepted="
            f"{accepted} of proposed={proposed}: the accepted rows "
            "are a prefix of the proposed block, so 0 <= accepted "
            "<= proposed"
        )
    return accepted, proposed


def _tile_floor(rows: int) -> int:
    """``rows`` reduced to a width whose verify fills every tile it touches.

    The companion to :func:`_tile_edge`, for the one width the sizer does
    not choose: an operator ceiling. A ceiling is an upper bound on what may
    be proposed, not a width worth proposing -- and a ceiling of 32 asks for
    a 33 row verify, which buys a second tile for a single row. So a
    handback that is not itself an edge steps down to the one below it,
    giving up at most 31 rows to halve the verify they would have ridden in.

    A block that fits inside the first tile is returned unchanged: there is
    no edge below it to step down to, and nothing is wasted, since the
    verify pays for that tile either way.
    """
    edge = (rows + 1) // COPY_DRAFT_TILE_ROWS * COPY_DRAFT_TILE_ROWS - 1
    return edge if edge >= COPY_DRAFT_TILE_ROWS - 1 else rows


def _tile_edge(rows: int) -> int:
    """Widest proposal whose ``rows + 1`` row verify fills whole tiles.

    ``quantized_matmul`` charges for a whole ``COPY_DRAFT_TILE_ROWS`` tile
    once a single row lands in it, so the widths worth proposing are the ones
    that end exactly at a tile boundary: 31, 63, 95. Rounding up rather than
    down because everything between an edge and the one below it has already
    been paid for -- a 40-row block costs what a 63-row block costs.
    """
    tiles = -(-(rows + 1) // COPY_DRAFT_TILE_ROWS)
    return tiles * COPY_DRAFT_TILE_ROWS - 1


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
    gate sums committed tokens and charged milliseconds over a bounded
    window of copy-draft rounds and of the speculative rounds they displace,
    and admits a proposal while the first ratio is at least the second. It also sizes the
    block from the same rounds' accepted counts (:meth:`width_cap`), so a
    turn whose matches run short gets a short block rather than a refusal.

    Per request by construction: the state describes the text being
    generated, which is what the sign depends on, and a fresh request
    starts by measuring rather than inheriting another turn's verdict.
    """

    __slots__ = (
        "_accepted_rate",
        "_accepted_seen",
        "_measured_seen",
        "_last_width",
        "_saturated",
        "_copy_window",
        "_base_window",
        "_probe_interval",
        "_since_probe",
        "declines",
        "probes",
    )

    def __init__(self) -> None:
        # ``(committed, round_ms)`` for the last ``COPY_DRAFT_GATE_WINDOW``
        # rounds of each kind. Summed on demand rather than carried as
        # running totals: 32 additions cost nothing against a 70 ms round,
        # and a sum that is never decremented cannot drift.
        self._copy_window: deque[tuple[int, float]] = deque(
            maxlen=COPY_DRAFT_GATE_WINDOW
        )
        self._base_window: deque[tuple[int, float]] = deque(
            maxlen=COPY_DRAFT_GATE_WINDOW
        )
        self._accepted_rate = 0.0
        # Copy-draft rounds seen at all, which is what decides whether the
        # sizer has left the probe width -- and, separately, how many of
        # them actually measured the match. A block the target consumed end
        # to end reports a lower bound, not a length, so it advances the
        # first count and not the second.
        self._accepted_seen = 0
        self._measured_seen = 0
        self._last_width = COPY_DRAFT_PROBE_WIDTH
        self._saturated = False
        self._probe_interval = COPY_DRAFT_GATE_PROBE_INTERVAL
        self._since_probe = 0
        # Diagnostics, surfaced through the generator's timing stats so a
        # turn that got no copy-drafts can be told apart from a turn whose
        # copy-drafts were measured and declined.
        self.declines = 0
        self.probes = 0

    def observe(
        self,
        *,
        is_copy_draft: bool,
        committed: int,
        round_ms: float,
        accepted: int | None = None,
        proposed: int | None = None,
    ) -> None:
        """Fold one round's realized throughput into the matching series.

        ``accepted`` and ``proposed`` are required for a copy-draft round and
        meaningless for any other, which is why they are keyword arguments
        that default to ``None`` rather than to zero: zero is a real
        measurement -- a block whose first row missed -- and silently
        standing in for an absent one would read every copy-draft round as a
        total rejection and collapse the next block to
        ``COPY_DRAFT_MIN_WIDTH`` after ``COPY_DRAFT_GATE_MIN_SAMPLES``
        rounds. That failure costs throughput without failing anything, so
        the omission is rejected here instead.

        Args:
            is_copy_draft: whether the round verified a copy-draft.
            committed: tokens the round actually delivered to the caller.
            round_ms: wall time charged to the round, including any drafter
                cost carried into it (a copy-draft carries none).
            accepted: copied rows the target accepted. Required for a
                copy-draft round, where it sizes the next block -- but only
                when it is below ``proposed``, since a block the target
                consumed end to end reports a lower bound on the match
                rather than its length. Ignored for any other round.
            proposed: copied rows the round actually verified, which can be
                below what :meth:`width_cap` allowed when the match itself
                was shorter or the cache could not recover that far.
                Required for a copy-draft round, to tell a block the target
                consumed end to end from one it cut short; ignored
                otherwise.

        Raises:
            TypeError: a copy-draft round was reported without both counts.
            ValueError: the counts cannot describe a round -- negative, or an
                accepted prefix longer than the block it came from.
        """
        # Validated before the degenerate-round return below, so a
        # malformed pair is rejected whether or not the round it describes
        # delivered anything. The narrowed pair then stands in for
        # ``is_copy_draft`` further down: the counts exist exactly when the
        # round was a copy-draft.
        counts = _sizing_counts(accepted, proposed) if is_copy_draft else None
        if committed <= 0 or round_ms <= 0.0:
            return
        if counts is not None:
            rows_accepted, rows_proposed = counts
            self._accepted_seen += 1
            # A block the target took end to end says the match runs *at
            # least* this far; a block it cut short says exactly how far.
            # Only the second one is a measurement of the match, and only a
            # measurement may move the estimate the sizer narrows to.
            #
            # Folding saturated blocks in as if they were lengths is what
            # makes the fall-back late rather than immediate: a ladder that
            # climbed to the tile edge on full blocks leaves an average
            # sitting at the edge, so the first block that comes back three
            # rows long still reads as ~22 and still asks for the whole
            # tile, for as many rounds as the average needs to decay. The
            # censored rounds already have their own channel -- ``_saturated``
            # and ``_last_width`` drive the doubling -- so the average is
            # left to mean one thing: how far this turn's matches run when
            # they run out.
            if rows_proposed > 0:
                self._saturated = rows_accepted >= rows_proposed
                self._last_width = rows_proposed
                if rows_accepted < rows_proposed:
                    if self._measured_seen == 0:
                        self._accepted_rate = float(rows_accepted)
                    else:
                        self._accepted_rate += COPY_DRAFT_ACCEPTANCE_EWMA_ALPHA * (
                            float(rows_accepted) - self._accepted_rate
                        )
                    self._measured_seen += 1
            self._copy_window.append((committed, round_ms))
        else:
            self._base_window.append((committed, round_ms))

    def allow(self) -> bool:
        """Whether a copy-draft may be proposed for the upcoming round."""
        if (
            len(self._copy_window) < COPY_DRAFT_GATE_MIN_SAMPLES
            or len(self._base_window) < COPY_DRAFT_GATE_MIN_SAMPLES
        ):
            # Undersampled: the only way to learn a copy-draft's value is to
            # verify one, so an unproven gate never blocks.
            return True
        copy_rate, base_rate = self.rates()
        if copy_rate >= base_rate:
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

    def width_cap(self, proposed: int) -> int:
        """Narrow ``proposed`` to what this turn's copies actually accept.

        How long the matched suffix was says the match is real, not how far
        it runs. What it runs to is measurable, and it differs by more than
        any fixed table can express: on the same 800-token rename turn,
        Qwen3.8-27B-4bit accepted ~10-14 copied rows per proposal while
        Qwen3.6-27B-4bit accepted ~4. A 31-row block is the right answer to
        the first and the wrong answer to the second -- and on a fast host
        the wrong answer is expensive, because a 31-row verify costs about
        four baseline rounds and returns one.

        So the request proposes a ceiling and the measurement narrows it:
        two rows until some copies have been verified, then either the
        accepted EWMA plus headroom or -- while the blocks keep coming back
        full -- twice the last width, and no narrowing at all once that
        lands inside the flat band where rows are free. Never widens past
        ``proposed``.

        Doubling while saturated is the congestion-window shape, for the
        same reason TCP uses it: a block the target consumed end to end has
        not measured how much further the match runs, only that it runs at
        least this far, so the estimate is a floor and the only way to lift
        it is to widen. Additive growth off a two-row probe would take
        fifteen rounds to reach the tile edge; doubling takes four, and
        every step is paid for by the rows the step before it delivered.
        The first block that comes back short ends it and the measured
        acceptance takes over -- immediately, because a saturated block
        never enters that average (see :meth:`observe`), so the first
        measurement of a match that ran out is the whole estimate rather
        than a 30% correction to the width the ladder had reached.

        Widening never crosses a tile edge on an unsaturated estimate. Rows
        inside the tile a block already occupied are free; the first row of
        the next tile costs a whole tile. So the headroom above a short
        block is capped at that block's own tile edge, and only a block that
        filled its tile end to end can buy the next one.

        Every width this returns either ends at a tile edge or fits inside
        the first tile, whatever ceiling it was offered -- the guarantee the
        widths are chosen for, held independently of whether the ceiling
        happens to be an edge.
        """
        if self._accepted_seen < COPY_DRAFT_GATE_MIN_SAMPLES:
            return min(proposed, COPY_DRAFT_PROBE_WIDTH)
        if self._saturated:
            earned = self._last_width * 2
        else:
            earned = int(self._accepted_rate * COPY_DRAFT_WIDTH_HEADROOM) + 1
            # Headroom is discovery inside a tile that is already paid for,
            # not a reason to buy the next one. A block that accepted 21 of
            # 31 measured a match that ended inside the first tile, and 1.5x
            # of 21 is 32 -- one row into the second. Crossing an edge has
            # to be earned by filling the tile below it, which is what the
            # doubling branch above does, so an unsaturated estimate is held
            # to the tile its own block occupied.
            earned = min(earned, _tile_edge(self._last_width))
        if earned >= COPY_DRAFT_FREE_WIDTH_FLOOR:
            # Inside the flat band: the rows this turn has not been reaching
            # are already paid for by the ones it has, so widen to the edge
            # of the tile the evidence lands in and let a long match run.
            #
            # To the edge, and not past it: with an operator ceiling above
            # the first edge, handing back the whole ceiling would buy a
            # second tile for a single row on the strength of an eight-row
            # match. Crossing an edge needs its own evidence, and the
            # doubling above is what produces it -- a saturated 31-row block
            # earns 62, which rounds up to the second edge at 63.
            #
            # The floor is for the ceiling rather than the evidence: an
            # operator who caps copy-drafts at 32 rows has bounded the
            # request, not named a width worth verifying, and 32 rows verify
            # in 33 -- a whole second tile for one row past the edge. So a
            # handback the ceiling cut to a non-edge steps down to the edge
            # below it, which is what makes the tile guarantee hold for
            # every ceiling and not only the ones that are edges.
            return _tile_floor(min(proposed, _tile_edge(earned)))
        # The floor applies to the *estimate*, not to the request: a turn
        # whose match is one row long gets a one-row block, not a two-row
        # block reaching past the end of its own match.
        return min(proposed, max(COPY_DRAFT_MIN_WIDTH, earned))

    def rates(self) -> tuple[float, float]:
        """``(copy_draft, baseline)`` throughput in tokens per millisecond.

        Tokens summed over a window of rounds and divided by the
        milliseconds summed over the same rounds -- not an EWMA of per-round
        rates, and not a total over the whole turn either.

        Not an EWMA, because one sample must not decide: a copy-draft whose
        first row misses delivers one token for a full wide verify, several
        times under the baseline rate. That is a real outcome, but at
        alpha 0.3 it drags the average under the baseline for the next
        several rounds -- measured on a Qwen3.8-27B-4bit rename turn that
        copy-drafts win by more than 20%, it refused 52 proposals.

        Not a running total either, because a verdict must not outlive the
        text that earned it. Pooling the whole turn lets a copy-rich opening
        pay for an arbitrarily long copy-poor tail: at the rates above it
        takes a few hundred losing rounds to drag a winning pool back to the
        baseline, and an answer that switches from quoting code to writing
        prose would spend that stretch proposing blocks it cannot fill.
        ``COPY_DRAFT_GATE_WINDOW`` bounds both the weight of one sample and
        how long a stale verdict can stand.

        The window is the right unit for the ratio, rather than an average
        of per-round ratios: the question is "over these rounds, did copies
        deliver more per millisecond than the rounds they replaced", and
        that is a ratio of sums, which weighs each round by the time it
        actually took.

        The two windows age independently, and on a copy-rich turn the
        baseline one can hold rounds from much earlier in the answer -- when
        the KV cache was shorter and a round was therefore cheaper. That
        biases the comparison towards the baseline, so the error is
        one-sided: it can refuse a copy-draft that would have paid, never
        admit one that does not. Making it two-sided means charging the
        baseline for a round that did not happen, which is a model rather
        than a measurement.
        """
        copy_ms = sum(ms for _, ms in self._copy_window)
        base_ms = sum(ms for _, ms in self._base_window)
        copy = (
            sum(tok for tok, _ in self._copy_window) / copy_ms if copy_ms > 0.0 else 0.0
        )
        base = (
            sum(tok for tok, _ in self._base_window) / base_ms if base_ms > 0.0 else 0.0
        )
        return copy, base
