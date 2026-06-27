# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the stand-alone DSpark Algorithm 1 implementation.

These tests verify the math is correctly transcribed from the paper.
They DO NOT exercise an engine integration — that integration is
structurally blocked (see REPORT.md).

Run from repo root:

    pytest spike/dspark/test_algorithm.py -v
"""

from __future__ import annotations

import math

import pytest

from spike.dspark.algorithm import (
    ScheduleResult,
    prefix_survival,
    schedule,
    tv_confidence,
)


# ----------------------------- tv_confidence ---------------------------


def test_tv_confidence_identical_distributions_is_one() -> None:
    """Paper Eq. 8: if p_d == p_t, ||p_d - p_t||_1 = 0, so c = 1."""
    p = [0.25, 0.25, 0.25, 0.25]
    assert tv_confidence(p, p) == pytest.approx(1.0)


def test_tv_confidence_disjoint_distributions_is_zero() -> None:
    """Disjoint one-hot: ||p_d - p_t||_1 = 2, so c = 1 - 1 = 0."""
    p_d = [1.0, 0.0, 0.0, 0.0]
    p_t = [0.0, 1.0, 0.0, 0.0]
    assert tv_confidence(p_d, p_t) == pytest.approx(0.0)


def test_tv_confidence_handcomputed() -> None:
    """Manual: p_d = [0.6, 0.4], p_t = [0.8, 0.2].
    |0.6-0.8| + |0.4-0.2| = 0.2 + 0.2 = 0.4. c = 1 - 0.2 = 0.8.
    """
    p_d = [0.6, 0.4]
    p_t = [0.8, 0.2]
    assert tv_confidence(p_d, p_t) == pytest.approx(0.8)


def test_tv_confidence_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        tv_confidence([0.5, 0.5], [1.0])


# ----------------------------- prefix_survival -------------------------


def test_prefix_survival_monotone_non_increasing() -> None:
    """Paper claim: a_{r,j} <= a_{r,j-1}. The early-stop proof depends
    on this — if it ever fails, Algorithm 1 is unsafe.
    """
    confidences = [0.9, 0.85, 0.7, 0.6, 0.5]
    survs = prefix_survival(confidences)
    for j in range(1, len(survs)):
        assert survs[j] <= survs[j - 1] + 1e-12, (
            f"monotonicity violated at j={j}: a_{{j}}={survs[j]}, a_{{j-1}}={survs[j - 1]}"
        )


def test_prefix_survival_handcomputed() -> None:
    """c = [0.9, 0.8, 0.5]. a = [0.9, 0.72, 0.36]."""
    survs = prefix_survival([0.9, 0.8, 0.5])
    assert survs == pytest.approx([0.9, 0.72, 0.36])


def test_prefix_survival_zero_kills_tail() -> None:
    """One zero confidence collapses all subsequent survivals to 0."""
    survs = prefix_survival([0.9, 0.0, 0.8, 0.7])
    assert survs[0] == pytest.approx(0.9)
    assert survs[1] == 0.0
    assert survs[2] == 0.0
    assert survs[3] == 0.0


def test_prefix_survival_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        prefix_survival([0.5, -0.1, 0.9])
    with pytest.raises(ValueError):
        prefix_survival([0.5, 1.5, 0.9])


# ----------------------------- schedule (Algorithm 1) ------------------


def _flat_sps(rate: float):
    """SPS that is constant — every batch size yields ``rate`` steps/s.
    Useful for verifying the greedy admits everything when there is no
    throughput penalty for larger batches.
    """

    def _f(b: int) -> float:  # noqa: ARG001
        return rate

    return _f


def _decaying_sps(rate0: float, decay_per_slot: float):
    """SPS that decays linearly with batch size. Approximates the
    paper's "smoothly decaying hardware capacity curve" assumption
    (Section 3.2.2 last paragraph).
    """

    def _f(b: int) -> float:
        return max(0.0, rate0 - decay_per_slot * b)

    return _f


def test_schedule_empty_request_set() -> None:
    res = schedule([], _flat_sps(100.0))
    assert res.lengths == ()
    assert res.theta == 0.0


def test_schedule_R1_flat_sps_admits_full_block() -> None:
    """R=1, flat SPS, confidences > 0 throughout. Every admission
    strictly increases tau* without lowering SPS — so the optimum is
    ell_1* = gamma (admit the whole block).

    This is the degenerate case the KILL recommendation rests on: at
    R=1 the scheduler reduces to DFlash's existing longest-accepted-
    prefix decision and provides no additional system signal.
    """
    confs = [[0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]]
    res = schedule(confs, _flat_sps(100.0))
    assert res.lengths == (8,)


def test_schedule_R1_steeply_decaying_sps_truncates() -> None:
    """When SPS drops faster than the marginal expected accept grows,
    the scheduler truncates. This is the only regime where Algorithm 1
    differs from "verify all" at R=1.

    Construct so the math is hand-verifiable. R=1, gamma=4, confs all
    0.5 (so a_j = 0.5^j). SPS(B) = max(0, 10 - 3*B):
        B=1: SPS=7,  B=2: SPS=4,  B=3: SPS=1,  B=4: SPS=0,  B=5: SPS=0.

    Theta at ell=0: tau*=1,    B=1, theta=1*7   = 7.0
    Theta at ell=1: tau*=1.5,  B=2, theta=1.5*4 = 6.0  -> not better, BREAK.
    Optimal: ell_1* = 0 (do not verify any draft token).
    """
    confs = [[0.5, 0.5, 0.5, 0.5]]
    res = schedule(confs, _decaying_sps(10.0, 3.0))
    assert res.lengths == (0,)
    assert res.theta == pytest.approx(7.0)


def test_schedule_R2_global_sort_routes_to_higher_confidence() -> None:
    """With 2 requests, the global descending sort means the higher-
    confidence request gets its block admitted FIRST before the lower-
    confidence one — this is the system-level "route batch capacity to
    high-return tokens" claim of the paper.

    R=2, gamma=2. Request 0 confs = [0.9, 0.9] -> a = [0.9, 0.81].
    Request 1 confs = [0.3, 0.3] -> a = [0.3, 0.09].
    Sorted: (0,1)=0.9, (0,2)=0.81, (1,1)=0.3, (1,2)=0.09.

    With flat SPS, all four are admitted (greedy stops only on
    non-improvement, and tau* keeps growing while SPS stays flat).
    Verify the admission ORDER matches the paper's claim.
    """
    confs = [[0.9, 0.9], [0.3, 0.3]]
    res = schedule(confs, _flat_sps(100.0))
    assert res.lengths == (2, 2)
    # Check the admission order — request 0 fully extends before
    # request 1 starts.
    order = res.admitted_in_order
    assert order[0] == (0, 1)
    assert order[1] == (0, 2)
    assert order[2] == (1, 1)
    assert order[3] == (1, 2)


def test_schedule_zero_confidence_excluded_from_candidate_space() -> None:
    """Paper line 4: E = {(r, j) | a_{r,j} > 0}. A position with a == 0
    is never admitted, even if subsequent positions had non-zero
    standalone confidence (they don't — a is monotone non-increasing).
    """
    confs = [[0.9, 0.0, 0.9]]  # a = [0.9, 0.0, 0.0]
    res = schedule(confs, _flat_sps(100.0))
    # Only position 1 can be admitted; positions 2 and 3 have a = 0.
    assert res.lengths == (1,)


def test_schedule_R1_degenerates_to_dflash_in_no_decay_regime() -> None:
    """KEY CLAIM behind the KILL recommendation.

    When R=1 and the SPS curve is approximately flat over the range
    [1, 1+gamma] (which it is for any block size that fits in one
    target-model batch slot on a single-request engine like MLX), the
    scheduler admits every position with a > 0. This is precisely what
    DFlash's existing longest-accepted-prefix decision does — the
    scheduler adds zero system signal and only incurs its own
    pure-Python overhead.
    """
    # Realistic confidences: high at position 1, decaying as a parallel
    # drafter's suffix decay (paper Figure 2).
    confs = [[0.88, 0.84, 0.80, 0.78, 0.75, 0.72, 0.70, 0.68]]
    # SPS approximately flat over batch sizes 1..16 (the realistic MLX
    # single-request regime where there's no batching opponent).
    res = schedule(confs, _flat_sps(50.0))
    assert res.lengths == (8,)
    # All positions in the block were admitted -> scheduler is a no-op
    # vs DFlash's full-block verify.


def test_schedule_returns_total_throughput_at_optimum() -> None:
    """Theta_best from the result should equal tau*(optimum) * SPS(B*)."""
    confs = [[0.8, 0.6], [0.7, 0.5]]
    res = schedule(confs, _flat_sps(10.0))
    # ell = (2, 2) -> tau* = 2 + 0.8 + 0.48 + 0.7 + 0.35 = 4.33
    # B = 4, SPS = 10 -> theta = 43.3
    assert res.theta == pytest.approx(43.3)


def test_schedule_result_lengths_consistent_with_admitted_order() -> None:
    """Sanity: the per-request ell_r* equals the count of (r, j)
    entries in admitted_in_order with that r.
    """
    confs = [[0.9, 0.8, 0.7], [0.6, 0.5]]
    res = schedule(confs, _flat_sps(20.0))
    counts = [0] * len(confs)
    for r, _j in res.admitted_in_order:
        counts[r] += 1
    assert tuple(counts) == res.lengths


# ----------------------------- lossless sanity -------------------------


def test_lossless_when_scheduler_admits_full_block() -> None:
    """When the scheduler admits every position (which it does at R=1
    with flat SPS), the truncation lengths equal gamma for every
    request — meaning the verification batch sent to the target model
    is IDENTICAL to what DFlash would send without the scheduler.
    By construction this preserves DFlash's existing lossless
    contract (longest-accepted-prefix at temp=0 is byte-identical to
    non-spec-decode generation).

    This test does NOT exercise the real engine — it asserts the
    contract algebraically. The actual byte-identical test would
    require a real DFlash run, which the structural blockers prevent.
    """
    confs = [[1.0 - 0.05 * j for j in range(16)]]  # block_size = 16
    res = schedule(confs, _flat_sps(100.0))
    assert res.lengths == (16,)
    # The verify call would receive the full draft block of 16 tokens,
    # identical to the no-scheduler path -> bit-identical output.


# ----------------------------- numerical edge cases --------------------


def test_tv_confidence_clip_to_unit_interval() -> None:
    """Float accumulation can push c epsilon-out of [0, 1]; ensure the
    function clips so downstream prefix_survival doesn't reject.
    """
    # Construct an L1 that's larger than 2.0 by tiny epsilon due to
    # float noise (simulated by overshooting the sum).
    p_d = [0.5 + 1e-15, 0.5 - 1e-15]
    p_t = [0.5 - 1e-15, 0.5 + 1e-15]
    c = tv_confidence(p_d, p_t)
    assert 0.0 <= c <= 1.0
    assert math.isfinite(c)


def test_schedule_returns_immutable_lengths_tuple() -> None:
    """Lengths is a tuple, not a list — callers can't mutate it."""
    confs = [[0.9, 0.8]]
    res = schedule(confs, _flat_sps(10.0))
    assert isinstance(res, ScheduleResult)
    assert isinstance(res.lengths, tuple)
