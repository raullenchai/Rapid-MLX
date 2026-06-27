# SPDX-License-Identifier: Apache-2.0
"""DSpark Algorithm 1 (Hardware-Aware Prefix Scheduler) — paper page 8.

Reference: Cheng et al., "DSpark: Confidence-Scheduled Speculative Decoding
with Semi-Autoregressive Generation," arxiv 2026.xxxxx, DeepSeek-AI / PKU.

This module is a STAND-ALONE, engine-independent implementation kept inside
``spike/dspark/`` rather than under ``vllm_mlx/`` because the surrounding
rapid-mlx engine cannot host the algorithm in production (see REPORT.md
for the four structural blockers). The algorithm itself is small, exact,
and unit-testable in isolation — keeping it here preserves the audit trail
for the KILL recommendation.

Public surface:

* :func:`tv_confidence` — paper Eq. 8 supervision target, evaluated
  analytically from draft + target distributions.
* :func:`prefix_survival` — paper Eq. between (8) and Algorithm 1:
  a_{r,j} = prod_{i<=j} c_{r,i}. Monotone non-increasing in j.
* :func:`schedule` — paper Algorithm 1 verbatim. Returns the per-request
  truncation lengths chosen by the early-stopping greedy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


def tv_confidence(p_draft: Sequence[float], p_target: Sequence[float]) -> float:
    """Paper Eq. 8 — analytical per-step acceptance probability.

    c_k* = 1 - 0.5 * ||p_draft - p_target||_1

    Both sequences must be the same length (vocab dimension) and each
    should be a valid probability distribution (non-negative, sums to 1).
    Validation is light: the caller is the trusted draft / target stack.
    """
    if len(p_draft) != len(p_target):
        raise ValueError(
            f"p_draft length {len(p_draft)} != p_target length {len(p_target)}"
        )
    l1 = sum(abs(d - t) for d, t in zip(p_draft, p_target, strict=True))
    c = 1.0 - 0.5 * l1
    # Clip to [0, 1] — TV is bounded in [0, 1] mathematically but float
    # accumulation can drift epsilon-out of range.
    if c < 0.0:
        return 0.0
    if c > 1.0:
        return 1.0
    return c


def prefix_survival(confidences: Sequence[float]) -> list[float]:
    """Cumulative product a_{r,j} = prod_{i<=j} c_{r,i}.

    Monotone non-increasing because each c_i is in [0, 1]. The paper's
    Algorithm 1 line 4 sorts candidate space by this quantity descending,
    and the monotonicity is exactly what makes the early-stopping greedy
    sound (line 13).
    """
    out: list[float] = []
    running = 1.0
    for c in confidences:
        if c < 0.0 or c > 1.0:
            raise ValueError(f"confidence {c} outside [0, 1]")
        running *= c
        out.append(running)
    return out


@dataclass(frozen=True)
class ScheduleResult:
    """Outcome of Algorithm 1.

    ``lengths`` is the per-request truncation length ``ell_r*``; sum of
    these plus R is the verified batch size B at the optimum. ``theta``
    is the expected throughput at the optimum (``tau* * SPS(B)`` from
    paper line 9). ``admitted_in_order`` is the (r, j) admissions in
    greedy order (for telemetry / unit-test introspection).
    """

    lengths: tuple[int, ...]
    theta: float
    admitted_in_order: tuple[tuple[int, int], ...]


def schedule(
    confidences_per_request: Sequence[Sequence[float]],
    sps: Callable[[int], float],
) -> ScheduleResult:
    """Paper Algorithm 1 — Hardware-Aware Prefix Scheduler.

    Args:
        confidences_per_request: ``R`` requests; each is a sequence of
            ``gamma`` per-position confidence estimates in ``[0, 1]``.
            Lengths may differ across requests (some drafters propose
            shorter blocks); the algorithm handles that natively.
        sps: Engine throughput at a given target-model batch size, in
            ``steps/second``. ``sps(B)`` must be defined for every
            ``B`` in ``{R, R+1, ..., R + sum_r len(confidences[r])}``.

    Returns:
        :class:`ScheduleResult` — per-request truncation lengths,
        expected throughput at the optimum, and the greedy admission
        trace.

    Algorithm 1 reproduction notes:

    * Line 1-3: compute prefix survivals (one ``prefix_survival`` call
      per request).
    * Line 4: build E = {(r, j) | a_{r,j} > 0} sorted descending by a.
      Ties broken by (r, j) for determinism.
    * Line 5-6: initial state. ``B = R`` because every request always
      verifies its position-0 token (the always-emit verify pred). The
      ``+1`` in B and ``+1`` in tau* (paper sec 3.2.2 paragraph 2:
      "B = sum_r (1 + ell_r)", "tau = sum_r (1 + sum a_{r,1..ell_r})")
      lives in the initial state.
    * Line 7-15: greedy admission with strict early-stop on first
      non-improving step. The paper's Appendix A proves this preserves
      the non-anticipating property and exact target-distribution
      recovery.
    """
    r_count = len(confidences_per_request)
    if r_count == 0:
        return ScheduleResult(lengths=(), theta=0.0, admitted_in_order=())

    # Line 1-3: per-request prefix survivals.
    survivals: list[list[float]] = [
        prefix_survival(confs) for confs in confidences_per_request
    ]

    # Line 4: candidate space E = {(r, j) | a_{r,j} > 0}, sorted
    # descending by a. We store (a, r, j) tuples; Python's sort is
    # stable so the tie-break is the original insertion order (r, j),
    # which is the deterministic choice the paper implies.
    candidates: list[tuple[float, int, int]] = []
    for r, survs in enumerate(survivals):
        for j_idx, a in enumerate(survs):
            if a > 0.0:
                # j is 1-indexed in the paper (paper line 2:
                # "j = 1, ..., gamma"); we store the 1-indexed value.
                candidates.append((a, r, j_idx + 1))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    # Line 5: initial states.
    ell = [0] * r_count
    batch_size = r_count  # B = sum_r (1 + ell_r) = R when all ell_r = 0
    tau_star = float(r_count)  # tau* = sum_r (1 + 0) = R initially

    # Line 6: tracking.
    theta_best = tau_star * sps(batch_size)
    ell_best = tuple(ell)
    admitted: list[tuple[int, int]] = []
    best_admit_len = 0

    # Line 7-15: greedy admission, early-stop on first non-improving.
    for a, r, j in candidates:
        ell[r] = j
        batch_size += 1
        tau_star += a
        theta = tau_star * sps(batch_size)
        admitted.append((r, j))
        if theta > theta_best:
            theta_best = theta
            ell_best = tuple(ell)
            best_admit_len = len(admitted)
        else:
            # Paper line 13: break. The Appendix-A non-anticipating
            # proof relies on this strict early-stop — DO NOT relax to
            # "skip and continue" without re-deriving the proof.
            break

    return ScheduleResult(
        lengths=ell_best,
        theta=theta_best,
        admitted_in_order=tuple(admitted[:best_admit_len]),
    )


__all__ = [
    "ScheduleResult",
    "tv_confidence",
    "prefix_survival",
    "schedule",
]
