# SPDX-License-Identifier: Apache-2.0
"""DSpark Hardware-Aware Prefix Scheduler — Algorithm 1 (paper p. 8).

Ported from "DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation" (Cheng et al., 2026), Section 3.2.2,
Algorithm 1. The paper's pseudocode is reproduced verbatim in spirit:
this implementation is a faithful drafter-agnostic Python port.

The scheduler picks a per-request verification length ell_r so that the
batched target verification step maximizes expected accepted tokens per
unit wall-clock, given:

  * c_{r,j}  — per-position confidence c_{r,j} in (0, 1) for request r,
               position j in [1, gamma]; in our spike we use the analytical
               supervision target from paper Eq. (8):
                   c_k* = 1 - 0.5 * || p_k^d - p_k^t ||_1
               (TV-distance proxy; no trained confidence head needed).
  * SPS(B)   — engine throughput in steps/sec at batch size B (in tokens
               sent to the target). Profiled once at engine init and stored
               as a small lookup table.

Public surface
--------------
- schedule_prefix_lengths(...) : run Algorithm 1
- SPSTable                    : callable cost table with O(1) lookup
- tv_confidence(...)          : compute c_k* from draft+target softmax rows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SPSTable:
    """Steps-per-second as a function of total batch tokens B.

    Stored as parallel arrays sorted by ascending B. Lookup uses the
    "largest B' <= B" semantics — equivalent to the paper's "smoothly
    decaying capacity curve" (Section 3.2.2, last paragraph). For
    Bsmaller than the smallest profiled point, returns the smallest
    point's SPS. For B above the largest, returns the largest point's
    SPS (graceful saturation).
    """

    batch_sizes: tuple[int, ...]
    sps: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.batch_sizes) != len(self.sps):
            raise ValueError("batch_sizes and sps must have the same length")
        if any(b1 >= b2 for b1, b2 in zip(self.batch_sizes, self.batch_sizes[1:])):
            raise ValueError("batch_sizes must be strictly ascending")

    def __call__(self, B: int) -> float:
        if B <= self.batch_sizes[0]:
            return self.sps[0]
        # Largest profiled B' <= B.
        for i in range(len(self.batch_sizes) - 1, -1, -1):
            if self.batch_sizes[i] <= B:
                return self.sps[i]
        return self.sps[-1]  # unreachable but defensive


def tv_confidence(p_draft_row: Sequence[float], p_target_row: Sequence[float]) -> float:
    """Compute c_k* = 1 - 0.5 * ||p_draft - p_target||_1 (paper Eq. 8).

    Both inputs must be valid probability vectors of equal length.
    Returns a scalar in [0, 1]. We don't require a torch / mlx
    dependency here — the caller is expected to materialize a list of
    floats (the spike does this once per step on a vocab-sized vector,
    O(V) Python overhead = fine for the bench).
    """
    if len(p_draft_row) != len(p_target_row):
        raise ValueError("draft and target rows must have the same vocab size")
    l1 = 0.0
    for d, t in zip(p_draft_row, p_target_row):
        l1 += abs(float(d) - float(t))
    c = 1.0 - 0.5 * l1
    # Numerical clamp: probabilities can sum to slightly off 1 with bf16.
    if c < 0.0:
        c = 0.0
    elif c > 1.0:
        c = 1.0
    return c


def schedule_prefix_lengths(
    confidence: Sequence[Sequence[float]],
    sps: SPSTable,
    *,
    R: int | None = None,
    gamma: int | None = None,
) -> tuple[list[int], float]:
    """Algorithm 1: Hardware-Aware Prefix Scheduler.

    Args:
        confidence: R x gamma matrix; confidence[r][j-1] is c_{r,j}.
        sps: SPSTable for the target engine.
        R: number of active requests (defaults to len(confidence)).
        gamma: max draft length per request (defaults to width of c).

    Returns:
        (ell_star, Theta_best) — per-request scheduled prefix lengths and
        the achieved expected system-wide token throughput
        Theta = tau * SPS(B).

    Semantics
    ---------
    Faithful to paper p. 8 lines 1-16:

      1.  Cumulative prefix survival a_{r,j} = prod_{i<=j} c_{r,i}
      2.  Candidate set E = {(r,j) : a_{r,j} > 0}
      3.  Sort E descending by a_{r,j}
      4.  Initialize ell_r = 0, B = R, tau = R, ell* = (0,...,0)
      5.  Theta_best = R * SPS(R)
      6.  for (r,j) in sorted order:
            ell_r = j; B += 1; tau += a_{r,j}; Theta = tau * SPS(B)
            if Theta > Theta_best: update; else: BREAK (non-anticipating)
      7.  return ell*, Theta_best

    The break (line 12 of the paper pseudocode) is the early-stopping
    that enforces the non-anticipating property (paper Section 3.2.2,
    "To enforce strict causality..."): the moment the throughput stops
    growing, we freeze the schedule — even if a later (r,j) would yield
    higher throughput. This guarantees lossless target-distribution
    recovery despite the greedy admit order.
    """
    if R is None:
        R = len(confidence)
    if gamma is None:
        gamma = len(confidence[0]) if R > 0 else 0
    if R < 0 or gamma < 0:
        raise ValueError("R and gamma must be non-negative")

    # Lines 1-3: cumulative survival a_{r,j}.
    a: list[list[float]] = []
    candidates: list[tuple[float, int, int]] = []  # (a_rj, r, j)
    for r in range(R):
        row = confidence[r]
        if len(row) != gamma:
            raise ValueError(
                f"confidence[{r}] has length {len(row)}; expected gamma={gamma}"
            )
        prefix = 1.0
        a_row: list[float] = []
        for j_minus_1, c_rj in enumerate(row):
            prefix *= float(c_rj)
            a_row.append(prefix)
            if prefix > 0.0:
                candidates.append((prefix, r, j_minus_1 + 1))
        a.append(a_row)

    # Line 3: sort descending by a_{r,j}. Python sort is stable; using
    # negative key gives the descending order. Ties are broken
    # deterministically by (-r, -j) to keep tests reproducible.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    # Lines 4-6: init.
    ell = [0] * R
    ell_star = [0] * R
    B = R
    tau = float(R)
    Theta_best = float(R) * sps(R)

    # Lines 7-15: greedy admit with early stop.
    for a_rj, r, j in candidates:
        ell[r] = j
        B += 1
        tau += a_rj
        Theta = tau * sps(B)
        if Theta > Theta_best:
            Theta_best = Theta
            ell_star = list(ell)
        else:
            break  # non-anticipating property (paper Section 3.2.2)

    return ell_star, Theta_best


__all__ = [
    "SPSTable",
    "schedule_prefix_lengths",
    "tv_confidence",
]
