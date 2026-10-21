"""Request-local draft depth selection, independent of the array runtime."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

DEPTH_ENV = "RAPID_MLX_DSV41_DRAFT_DEPTH"
# M3 Ultra synchronized v2 priors; real costs are learned online. See
# docs/engineering/performance/2026-10-21-dsv41-draft-depth.md.
VERIFY_MS = (28.7, 46.2, 58.0, 69.5, 81.7)
DRAFT_MS = 21.0


def draft_depth() -> int | str:
    value = os.environ.get(DEPTH_ENV, "4").strip().lower()
    if value == "auto":
        return value
    if value in ("0", "1", "2", "3", "4"):
        return int(value)
    logging.getLogger(__name__).warning(
        "%s=%r is invalid; expected 0..4 or auto; using 4", DEPTH_ENV, value
    )
    return 4


@dataclass
class DraftDepthPolicy:
    mode: int | str
    acceptance: float = 0.5
    current: int = 0
    idle_rounds: int = 0
    probe_interval: int = 32
    verify_ms: tuple[float, ...] = VERIFY_MS
    draft_ms: float = DRAFT_MS
    learned_verify_ms: dict[int, float] = field(default_factory=dict)
    learned_draft_ms: dict[int, float] = field(default_factory=dict)
    learned_round_ms: dict[int, float] = field(default_factory=dict)

    def rate(self, depth: int) -> float:
        tokens = sum(self.acceptance**i for i in range(depth + 1))
        cost = self.learned_round_ms.get(depth)
        if cost is None:
            cost = self.learned_verify_ms.get(depth, self.verify_ms[depth])
            if depth:
                cost += self.learned_draft_ms.get(depth, self.draft_ms)
        return tokens / cost

    def observe_cost(
        self,
        depth: int,
        *,
        draft_ms: float | None = None,
        verify_ms: float | None = None,
        round_ms: float | None = None,
    ) -> None:
        """Learn completed wall-clock work, never asynchronous submission time.

        When existing sync points cannot separate phases, supply only round_ms;
        that total replaces both priors without subtracting an estimated phase.
        Each depth starts with its first sample, then uses EMA weight 0.25.
        """
        for costs, sample in (
            (self.learned_draft_ms, draft_ms),
            (self.learned_verify_ms, verify_ms),
            (self.learned_round_ms, round_ms),
        ):
            if sample is not None:
                costs[depth] = 0.75 * costs.get(depth, sample) + 0.25 * sample

    def choose(self, limit: int = 4) -> int:
        if self.mode != "auto":
            return min(int(self.mode), limit)
        best = max(range(limit + 1), key=self.rate)
        # Require a margin over target-only; disable immediately when unprofitable.
        if self.rate(best) <= self.rate(0) * 1.05:
            best = 0
        current = min(self.current, limit)
        if current and self.rate(current) <= self.rate(0) * 1.05:
            current = 0
        if best == 0 or self.rate(best) > self.rate(current) * 1.10:
            self.current = best
        else:
            self.current = current
        if self.current:
            self.idle_rounds = 0
            return self.current
        self.idle_rounds += 1
        if self.idle_rounds >= self.probe_interval and limit:
            self.idle_rounds = 0
            return limit
        return 0

    def observe(self, accepted: int, proposed: int) -> None:
        # Only tested predictions count: a rejected suffix is unobserved.
        if proposed:
            trials = min(accepted + 1, proposed)
            self.acceptance = 0.75 * self.acceptance + 0.25 * accepted / trials
