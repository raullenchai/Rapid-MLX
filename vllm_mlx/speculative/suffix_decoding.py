# SPDX-License-Identifier: Apache-2.0
"""SuffixDecoding — adaptive suffix-tree speculative drafter.

Draft-model-free speculative decoding for token-level inference. Builds a
sliding index over the prompt + generated tokens, and at each generation
step proposes a variable-length draft sequence by looking at *all* prior
occurrences of the current suffix and ranking continuations by frequency.

Compared to plain Prompt Lookup Decoding (PLD)
(``vllm_mlx/speculative/prompt_lookup.py``), this drafter:

  1. Considers **all** match positions, not just the first, so an
     occasional spurious match can't drag the draft off the most likely
     continuation.
  2. **Adaptively truncates** the draft at the first low-confidence
     position. PLD always emits ``num_draft_tokens`` regardless of
     confidence, which wastes verify cycles when the suffix tree is
     ambiguous.
  3. Tracks **per-position acceptance** so the verify loop can be greedy
     about long matches when the index is confident, and conservative
     when it's not.

Reference: SuffixDecoding (NeurIPS 2025) https://arxiv.org/abs/2411.04975
— same algorithmic idea, simpler data structure (we use a dict-based
suffix index, not a trie/automaton, because the PoC's history sizes
fit comfortably in O(N²) suffix construction time).

The ``Drafter`` class is engine-agnostic and only needs (a) a token
stream to index and (b) a way to query for a draft. Wiring it into
BatchedEngine's verify loop is the responsibility of the caller —
see ``scripts/bench_suffix_decoding.py`` for a single-request reference
integration via ``mlx_lm`` directly.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DraftStats:
    """Per-request bookkeeping for adaptive-draft acceptance."""

    total_drafts_proposed: int = 0
    total_draft_tokens_proposed: int = 0
    total_draft_tokens_accepted: int = 0
    sum_draft_length: int = 0  # for mean draft length
    n_step_calls: int = 0  # number of times get_draft() was called
    n_drafts_returned: int = 0  # subset of n_step_calls where draft was non-empty

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens_proposed == 0:
            return 0.0
        return self.total_draft_tokens_accepted / self.total_draft_tokens_proposed

    @property
    def mean_accepted_per_step(self) -> float:
        """Average accepted draft tokens per drafting step.

        ``> 0`` directly translates to an upper bound on speedup: e.g.
        0.7 → ~1.7x decode TPS (one verify forward emits 1+0.7 tokens
        on average vs. 1 token in baseline).
        """
        if self.n_step_calls == 0:
            return 0.0
        return self.total_draft_tokens_accepted / self.n_step_calls

    def as_dict(self) -> dict:
        return {
            "total_drafts_proposed": self.total_drafts_proposed,
            "total_draft_tokens_proposed": self.total_draft_tokens_proposed,
            "total_draft_tokens_accepted": self.total_draft_tokens_accepted,
            "n_step_calls": self.n_step_calls,
            "n_drafts_returned": self.n_drafts_returned,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_accepted_per_step": round(self.mean_accepted_per_step, 4),
        }


class SuffixDecodingDrafter:
    """Adaptive suffix-index drafter.

    Maintains an index from each k-gram (k in [1, max_suffix_len]) to the
    list of positions where it ends. Querying with the current history's
    last k tokens returns earlier positions; the drafter then picks the
    next ``max_draft_tokens`` continuation by majority vote across all
    matches, truncating where confidence drops.

    Args:
        max_draft_tokens: Cap on draft length per call. The verify forward
            cost grows linearly with this; pick based on your model's
            attention cost vs. expected acceptance. Typical values 4-16.
        max_suffix_len: Longest suffix prefix to index (k-gram size). Longer
            suffixes are more discriminating but slower to build. 4 is a
            good default — covers most repeated-phrase cases without
            blowing up index size.
        min_confidence: Minimum fraction of matches that must agree on the
            next token. Below this, drafting truncates. Lower → more
            optimistic drafts (more verify-then-reject); higher → fewer
            but more reliable drafts. 0.3 from the SuffixDecoding paper.
        max_history: Hard cap on indexed history. Old tokens are dropped
            from the index when this is exceeded — keeps memory bounded
            on long generations. Set to None to disable.
    """

    def __init__(
        self,
        max_draft_tokens: int = 8,
        max_suffix_len: int = 4,
        min_confidence: float = 0.3,
        max_history: int | None = 32_000,
    ):
        if max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be >= 1")
        if max_suffix_len < 1:
            raise ValueError("max_suffix_len must be >= 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

        self.max_draft_tokens = max_draft_tokens
        self.max_suffix_len = max_suffix_len
        self.min_confidence = min_confidence
        self.max_history = max_history

        # Token history (prompt + generated). Indexed positions are
        # absolute and shifted when we trim from the left.
        self._tokens: list[int] = []
        # _shift = number of tokens dropped from the head; index positions
        # in _suffix_index are *absolute* (pre-shift), so we subtract
        # _shift before accessing _tokens.
        self._shift = 0

        # _suffix_index[k][k-gram tuple] = list of absolute end-positions
        # (i.e. position of the LAST token in the k-gram)
        self._suffix_index: list[dict[tuple, list[int]]] = [
            defaultdict(list) for _ in range(max_suffix_len + 1)
        ]

        self.stats = DraftStats()

    # --- Index management ----------------------------------------------

    def add_prompt_tokens(self, tokens: list[int]) -> None:
        """Bulk-index the prompt before generation begins."""
        for tok in tokens:
            self._add_one(tok)

    def add_generated_token(self, token: int) -> None:
        """Index a newly generated token (whether from primary or draft)."""
        self._add_one(token)

    def _add_one(self, token: int) -> None:
        self._tokens.append(token)
        abs_pos = self._shift + len(self._tokens) - 1
        # Index k-grams that END at abs_pos for k in [1..max_suffix_len].
        for k in range(1, self.max_suffix_len + 1):
            start_local = len(self._tokens) - k
            if start_local < 0:
                continue
            kgram = tuple(self._tokens[start_local : start_local + k])
            self._suffix_index[k][kgram].append(abs_pos)
        # Trim head if exceeding max_history. We must drop stale index
        # entries here — leaving them in place would leak old absolute
        # positions whose continuations now resolve to *different* tokens
        # in the shifted local window, causing wrong drafts. The cost is
        # O(window) per trim; trims happen rarely (once per max_history
        # tokens) so amortized cost is O(1) per add.
        if self.max_history is not None and len(self._tokens) > self.max_history:
            drop = len(self._tokens) - self.max_history
            self._tokens = self._tokens[drop:]
            self._shift += drop
            min_valid_end = self._shift  # k-grams whose end is < shift are gone
            for k in range(1, self.max_suffix_len + 1):
                bucket = self._suffix_index[k]
                # Each k-gram needs end_abs >= shift AND start_abs >= shift,
                # i.e. end_abs - k + 1 >= shift, so end_abs >= shift + k - 1.
                threshold = min_valid_end + k - 1
                stale_keys = []
                for kgram, ends in bucket.items():
                    fresh = [e for e in ends if e >= threshold]
                    if fresh:
                        bucket[kgram] = fresh
                    else:
                        stale_keys.append(kgram)
                for key in stale_keys:
                    del bucket[key]

    # --- Drafting ------------------------------------------------------

    def get_draft(self) -> list[int]:
        """Propose a draft continuation for the current history.

        Returns the longest run of tokens that pass the confidence floor.
        Empty list means "no good draft" — caller falls back to vanilla
        single-token decode for this step.
        """
        self.stats.n_step_calls += 1

        if not self._tokens:
            return []

        # Try suffix lengths from longest to shortest. A longer matched
        # suffix is more discriminating, so we prefer its continuations
        # when available. (PLD paper showed this matters more than match
        # count for typical agent workloads.)
        for k in range(min(self.max_suffix_len, len(self._tokens)), 0, -1):
            query = tuple(self._tokens[-k:])
            positions = self._suffix_index[k].get(query, [])
            if not positions:
                continue

            current_end_abs = self._shift + len(self._tokens) - 1
            draft = self._build_draft_from_positions(positions, k, current_end_abs)
            if draft:
                self.stats.total_drafts_proposed += 1
                self.stats.total_draft_tokens_proposed += len(draft)
                self.stats.n_drafts_returned += 1
                self.stats.sum_draft_length += len(draft)
                return draft
        return []

    def _build_draft_from_positions(
        self,
        match_end_positions: list[int],
        suffix_len: int,
        current_end_abs: int,
    ) -> list[int]:
        """Vote over continuations from each match position; truncate at
        the first position where the winning vote falls below
        ``min_confidence``.
        """
        draft: list[int] = []
        # match_end_positions are absolute end-positions of the matched
        # suffix. The continuation begins at end+1 (absolute), which in
        # local coords is end - shift + 1.
        for offset in range(self.max_draft_tokens):
            counter: Counter[int] = Counter()
            for end_abs in match_end_positions:
                # Skip the current occurrence — we don't predict ourselves.
                if end_abs == current_end_abs:
                    continue
                cont_abs = end_abs + 1 + offset
                cont_local = cont_abs - self._shift
                if 0 <= cont_local < len(self._tokens):
                    counter[self._tokens[cont_local]] += 1
            if not counter:
                break
            top_tok, top_count = counter.most_common(1)[0]
            total = sum(counter.values())
            confidence = top_count / total
            if confidence < self.min_confidence:
                break
            draft.append(top_tok)
            # Tighten the next round to only positions that produced the
            # winner — long monotonic matches stay confident; ambiguous
            # ones drop fast as the candidate pool shrinks.
            match_end_positions = [
                end_abs
                for end_abs in match_end_positions
                if end_abs != current_end_abs
                and 0 <= (end_abs + 1 + offset - self._shift) < len(self._tokens)
                and self._tokens[end_abs + 1 + offset - self._shift] == top_tok
            ]
            if not match_end_positions:
                break
            # Suffix prefix length grew by 1 — promote up the index for
            # the next round (no-op here, but conceptual: the candidates
            # we kept are positions where suffix+offset-prefix matched).
            _ = suffix_len + offset + 1
        return draft

    # --- Acceptance bookkeeping ---------------------------------------

    def record_acceptance(self, num_accepted: int) -> None:
        """Caller reports how many of the most recent draft were accepted
        by the verify forward. Used purely for stats."""
        self.stats.total_draft_tokens_accepted += num_accepted

    def stats_dict(self) -> dict:
        return self.stats.as_dict()

    # --- Diagnostics ---------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SuffixDecodingDrafter(history={len(self._tokens)}, "
            f"max_draft={self.max_draft_tokens}, "
            f"max_suffix={self.max_suffix_len}, "
            f"min_conf={self.min_confidence}, "
            f"acc={self.stats.acceptance_rate:.2f})"
        )
