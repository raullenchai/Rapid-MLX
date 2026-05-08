# SPDX-License-Identifier: Apache-2.0
"""Tests for the SuffixDecoding bench reliability gates (#269 follow-up).

The bench script's previous methodology produced bogus 1000-2700 tok/s
TPS readings when the model returned short responses (early EOS,
refusal, or single-chunk streaming). The fix adds three gates — zero
tokens, decode-time floor, TPS ceiling — implemented in
``_classify_run`` so they're testable without a live server.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the bench script as a module without invoking its CLI.
_BENCH_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "bench_suffix_decoding_integrated.py"
)
_spec = importlib.util.spec_from_file_location("bench_suffix_decoding", _BENCH_PATH)
assert _spec and _spec.loader
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_suffix_decoding"] = bench
_spec.loader.exec_module(bench)


class TestClassifyRun:
    """``_classify_run`` is the boundary that decides whether a single
    request's measurement is trustworthy."""

    def test_normal_run_passes(self):
        # 100 tokens in 1.0s decode = 100 tok/s — comfortably realistic.
        wr = bench._classify_run(completion_tokens=100, decode_time=1.0, total_time=1.5)
        assert wr.tps == pytest.approx(100.0)
        assert wr.rejected_reason is None

    def test_zero_tokens_rejected(self):
        # Server didn't report usage / model returned nothing — caller can't
        # tell decode rate from one missing data point.
        wr = bench._classify_run(0, 0.5, 1.0)
        assert wr.tps is None
        assert wr.rejected_reason == "zero_completion_tokens"

    def test_negative_tokens_rejected(self):
        wr = bench._classify_run(-1, 1.0, 1.0)
        assert wr.tps is None
        assert wr.rejected_reason == "zero_completion_tokens"

    def test_decode_time_floor_rejects_short_window(self):
        # 80 tokens generated in 0.04s → 2000 tok/s. This is the exact
        # failure mode that burned smollm3-3b's code_edit run in v2 —
        # technically >32 tokens (the old guard) but still meaningless.
        wr = bench._classify_run(completion_tokens=80, decode_time=0.04, total_time=0.6)
        assert wr.tps is None
        assert wr.rejected_reason and "decode_time" in wr.rejected_reason

    def test_decode_time_at_floor_passes(self):
        # Exactly at the boundary should be accepted (closed lower bound).
        wr = bench._classify_run(
            completion_tokens=50,
            decode_time=bench.MIN_DECODE_TIME,
            total_time=1.0,
        )
        assert wr.tps is not None
        assert wr.rejected_reason is None

    def test_tps_ceiling_rejects_implausible_speed(self):
        # 1000 tokens in 1.0s decode = 1000 tok/s. Past the floor but no
        # mlx-lm decode on M-series we ship runs that fast.
        wr = bench._classify_run(
            completion_tokens=1000, decode_time=1.0, total_time=1.5
        )
        assert wr.tps is None
        assert wr.rejected_reason and "ceiling" in wr.rejected_reason

    def test_tps_ceiling_at_boundary(self):
        # Just under the ceiling passes; at-or-above gets rejected. Models
        # that genuinely run this fast on M3 Ultra would be a surprise but
        # we keep the gate strict to catch measurement errors.
        wr_under = bench._classify_run(
            completion_tokens=int(bench.TPS_CEILING * 0.99),
            decode_time=1.0,
            total_time=1.5,
        )
        assert wr_under.tps is not None

        wr_over = bench._classify_run(
            completion_tokens=int(bench.TPS_CEILING * 1.5),
            decode_time=1.0,
            total_time=1.5,
        )
        assert wr_over.tps is None

    def test_raw_fields_preserved_on_reject(self):
        """Even rejected runs must carry the raw timing so post-mortem
        debugging doesn't need a re-run — that's the whole reason
        ``WorkloadRun`` is a dataclass instead of a float."""
        wr = bench._classify_run(completion_tokens=10, decode_time=0.05, total_time=0.4)
        assert wr.tps is None
        assert wr.completion_tokens == 10
        assert wr.decode_time == 0.05
        assert wr.total_time == 0.4
        assert wr.rejected_reason  # non-empty


class TestThresholds:
    """Thresholds are constants, not magic numbers — tests pin the
    intent so a future tweak that 'happens to make a test pass' has to
    update this assertion too."""

    def test_min_decode_time_is_half_second(self):
        # 0.5s is enough that the per-request HTTP overhead (~10-50ms on
        # localhost) is <10% of the measurement.
        assert pytest.approx(0.5) == bench.MIN_DECODE_TIME

    def test_tps_ceiling_above_observed_peak(self):
        # 500 sits comfortably above the fastest mlx-lm decode we ship
        # (~330 tok/s for Llama-3.2-1B-4bit). If a future model goes past
        # this for real we re-tune; until then it's a reliable sanity gate.
        assert pytest.approx(500.0) == bench.TPS_CEILING


class TestPayloadIsGreedy:
    """Pin the contract that bench requests force greedy sampling.

    Without ``temperature: 0.0`` in the payload the server's
    ``_resolve_temperature`` returns the 0.7 fallback,
    ``_install_suffix_decoding._is_greedy_for_uid`` returns False on
    every step, the verify path falls through to vanilla, and the
    bench measures vanilla-vs-vanilla. Every measurement collapses
    to ~1.0x and the resulting tier dataset is meaningless.

    The whole batch-1 / batch-2 tier sweep on disk before this fix
    landed was wrong for exactly this reason.
    """

    def test_payload_forces_temperature_zero(self):
        """Inspect the source so the constraint can't drift via a refactor
        that 'still imports run_workload' but reshuffles the body."""
        text = Path(bench.__file__).read_text()
        assert '"temperature": 0.0' in text, (
            "bench payload must set temperature=0.0 to force greedy sampling. "
            "Without this, server defaults to 0.7 and SuffixDecoding falls "
            "through on every step (see scheduler.py::_is_greedy_for_uid). "
            "Tier dataset becomes vanilla-vs-vanilla noise."
        )

    def test_workload_bodies_dont_set_temperature(self):
        """Workload dicts must not pre-set temperature — the central
        payload override is the single source of truth.

        If a workload sneaks in ``"temperature": 0.7``, the
        ``**workload_body`` spread (which comes AFTER the temperature
        key in ``run_workload``) would clobber the greedy default.
        Catch that before it ships.
        """
        for name, body in bench.WORKLOADS.items():
            assert "temperature" not in body, (
                f"workload '{name}' sets temperature, which would override "
                "the greedy default. Remove it from the workload body."
            )
