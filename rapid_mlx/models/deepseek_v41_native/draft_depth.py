"""Request-local draft depth selection, independent of the array runtime."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

DEPTH_ENV = "RAPID_MLX_DSV41_DRAFT_DEPTH"
# M3 Ultra synchronized v2 priors; real costs are learned online. See
# docs/engineering/performance/2026-10-21-dsv41-draft-depth.md.
VERIFY_MS = (28.7, 46.2, 58.0, 69.5, 81.7)
DRAFT_MS = 19.6
HOST_MS = 3.0


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
    """Empirical prefix yield from maximum-depth rounds only.

    Smaller rounds are censored: exclude them from the acceptance histogram,
    but still learn their costs. Histogram weights decay by 0.9 per full-depth
    observation (roughly ten samples of memory), independently of cost EMAs.
    """

    mode: int | str
    max_depth: int = 4
    current: int = 0
    idle_rounds: int = 0
    probe_interval: int = 32
    probe_remaining: int = 0
    warmup_rounds: int = 8
    full_rounds: int = 0
    accepted_histogram: list[float] = field(default_factory=lambda: [0.0] * 5)
    verify_ms: tuple[float, ...] = VERIFY_MS
    draft_ms: float = DRAFT_MS
    host_ms: float = HOST_MS
    learned_verify_ms: dict[int, float] = field(default_factory=dict)
    learned_draft_ms: float | None = None
    learned_plain_ms: float | None = None

    def expected_tokens(self, depth: int) -> float:
        weight = sum(self.accepted_histogram)
        if not weight:
            return 1.0
        return (
            1
            + sum(
                min(accepted, depth) * count
                for accepted, count in enumerate(self.accepted_histogram)
            )
            / weight
        )

    def verify_cost(self, depth: int) -> float:
        if depth in self.learned_verify_ms:
            return self.learned_verify_ms[depth]
        ratios = [
            cost / self.verify_ms[k] for k, cost in self.learned_verify_ms.items()
        ]
        scale = sum(ratios) / len(ratios) if ratios else 1.0
        return self.verify_ms[depth] * scale

    def rate(self, depth: int) -> float:
        if depth == 0 and self.learned_plain_ms is not None:
            return 1 / self.learned_plain_ms
        cost = self.verify_cost(depth) + self.host_ms
        if depth:
            cost += (
                self.draft_ms
                if self.learned_draft_ms is None
                else self.learned_draft_ms
            )
        return self.expected_tokens(depth) / cost

    @staticmethod
    def _ema(previous: float | None, sample: float) -> float:
        return sample if previous is None else 0.75 * previous + 0.25 * sample

    def observe_cost(
        self,
        depth: int,
        *,
        draft_ms: float | None = None,
        verify_ms: float | None = None,
        round_ms: float | None = None,
    ) -> None:
        """Learn completed phases, sharing draft cost across positive depths.

        round_ms includes host orchestration, excluding consumer/yield time.
        A plain round learns the actual plain-step cost without decomposition.
        Positive rounds require measured phases to learn the host residual.
        """
        if draft_ms is not None:
            self.learned_draft_ms = self._ema(self.learned_draft_ms, draft_ms)
        if verify_ms is not None:
            self.learned_verify_ms[depth] = self._ema(
                self.learned_verify_ms.get(depth), verify_ms
            )
        if round_ms is not None:
            if depth == 0:
                self.learned_plain_ms = self._ema(self.learned_plain_ms, round_ms)
            elif draft_ms is not None and verify_ms is not None:
                self.host_ms = self._ema(
                    self.host_ms, max(0.0, round_ms - draft_ms - verify_ms)
                )

    def choose(self, limit: int = 4) -> int:
        if self.mode != "auto":
            return min(int(self.mode), limit)
        limit = min(limit, self.max_depth)
        if not limit:
            return 0
        if self.full_rounds < self.warmup_rounds:
            return limit
        if self.probe_remaining:
            self.probe_remaining -= 1
            return limit
        best = max(range(limit + 1), key=self.rate)
        current = min(self.current, limit)
        if current and self.rate(current) <= self.rate(0):
            current = 0
        if best == 0 or self.rate(best) > self.rate(current) * 1.10:
            self.current = best
        else:
            self.current = current
        if self.current:
            self.idle_rounds = 0
            return self.current
        if self.idle_rounds >= self.probe_interval:
            self.idle_rounds = 0
            self.probe_remaining = 1
            return limit
        return 0

    def observe(self, accepted: int, proposed: int) -> None:
        if proposed == 0:
            self.idle_rounds += 1
        elif proposed == self.max_depth:
            self.accepted_histogram = [n * 0.9 for n in self.accepted_histogram]
            self.accepted_histogram[accepted] += 1
            self.full_rounds += 1
