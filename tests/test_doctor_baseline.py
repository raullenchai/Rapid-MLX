# SPDX-License-Identifier: Apache-2.0
"""Tests for the doctor baseline comparator.

The doctor harness lives or dies on whether ``compare()`` correctly
classifies metrics relative to a baseline.  These are the safety net
for that logic — if any of these fail, the doctor's regression
detection is wrong and we need to look at it.
"""

from vllm_mlx.doctor.baseline import (
    DEFAULT_THRESHOLDS,
    DeltaStatus,
    compare,
    has_regression,
    safe_model_slug,
)

# A minimal threshold map covering the metrics we exercise below.
TIGHT_THRESHOLDS = {
    "decode_tps": {"regression_pct": 5, "improvement_pct": 10},
    "cold_ttft_ms": {"regression_pct": 10, "improvement_pct": 15},
    "tc_latency_ms": {"regression_pct": 10, "improvement_pct": 15},
    "tc_success_rate": {"regression_pct": 0, "improvement_pct": 5},
}


def _baseline(metrics: dict) -> dict:
    return {"model": "test-model", "metrics": metrics}


# ----------------------------------------------------------------------
# Higher-is-better metrics (decode_tps)
# ----------------------------------------------------------------------


class TestHigherIsBetter:
    def test_within_threshold_is_ok(self):
        deltas = compare(
            current={"decode_tps": 96.0},
            baseline=_baseline({"decode_tps": 100.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert len(deltas) == 1
        assert deltas[0].status == DeltaStatus.OK
        assert deltas[0].delta_pct == -4.0  # 4% slower, within 5% threshold

    def test_drop_beyond_threshold_is_regression(self):
        deltas = compare(
            current={"decode_tps": 90.0},  # 10% slower
            baseline=_baseline({"decode_tps": 100.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.REGRESSION
        assert deltas[0].delta_pct == -10.0
        assert has_regression(deltas)

    def test_gain_beyond_threshold_is_improvement(self):
        deltas = compare(
            current={"decode_tps": 120.0},  # 20% faster
            baseline=_baseline({"decode_tps": 100.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.IMPROVEMENT
        assert deltas[0].delta_pct == 20.0
        assert not has_regression(deltas)


# ----------------------------------------------------------------------
# Lower-is-better metrics (latency, memory)
# ----------------------------------------------------------------------


class TestLowerIsBetter:
    def test_lower_latency_is_improvement(self):
        # Latency dropped from 400ms to 320ms — 20% faster.
        deltas = compare(
            current={"cold_ttft_ms": 320.0},
            baseline=_baseline({"cold_ttft_ms": 400.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.IMPROVEMENT
        # Sign convention: positive delta_pct = better, even for
        # lower-is-better metrics.  This is the contract callers rely
        # on when rendering ↑/↓ arrows.
        assert deltas[0].delta_pct == 20.0

    def test_higher_latency_is_regression(self):
        # Latency rose from 400ms to 500ms — 25% slower.
        deltas = compare(
            current={"cold_ttft_ms": 500.0},
            baseline=_baseline({"cold_ttft_ms": 400.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.REGRESSION
        assert deltas[0].delta_pct == -25.0
        assert has_regression(deltas)

    def test_tc_latency_ms_uses_lower_is_better(self):
        """Regression-test for codex round 2 finding — tc_latency_ms
        was missing from _LOWER_IS_BETTER and gave inverted signs."""
        deltas = compare(
            current={"tc_latency_ms": 1200.0},  # 20% faster
            baseline=_baseline({"tc_latency_ms": 1500.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.IMPROVEMENT
        assert deltas[0].delta_pct == 20.0


# ----------------------------------------------------------------------
# Accuracy metrics — zero-tolerance regression
# ----------------------------------------------------------------------


class TestAccuracyZeroTolerance:
    def test_any_drop_is_regression(self):
        deltas = compare(
            current={"tc_success_rate": 0.99},
            baseline=_baseline({"tc_success_rate": 1.0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.REGRESSION
        assert has_regression(deltas)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_baseline_no_change_is_ok(self):
        deltas = compare(
            current={"composite_score": 0},
            baseline=_baseline({"composite_score": 0}),
            thresholds=TIGHT_THRESHOLDS,
        )
        assert deltas[0].status == DeltaStatus.OK

    def test_new_metric_is_marked_new(self):
        deltas = compare(
            current={"new_thing": 42},
            baseline=_baseline({}),
            thresholds={},
        )
        assert deltas[0].status == DeltaStatus.NEW
        assert deltas[0].baseline is None

    def test_dropped_metric_is_marked_dropped(self):
        deltas = compare(
            current={},
            baseline=_baseline({"old_thing": 42}),
            thresholds={},
        )
        assert deltas[0].status == DeltaStatus.DROPPED
        assert deltas[0].current is None

    def test_no_baseline_returns_new_for_everything(self):
        deltas = compare(
            current={"a": 1, "b": 2},
            baseline=None,
            thresholds={},
        )
        assert all(d.status == DeltaStatus.NEW for d in deltas)

    def test_unknown_metric_falls_back_to_default_thresholds(self):
        # With DEFAULT_THRESHOLDS = 5%/10%, a 6% drop is regression.
        deltas = compare(
            current={"unknown_metric": 94.0},
            baseline=_baseline({"unknown_metric": 100.0}),
            thresholds={},
        )
        assert deltas[0].status == DeltaStatus.REGRESSION
        assert deltas[0].threshold_pct == DEFAULT_THRESHOLDS["regression_pct"]


# ----------------------------------------------------------------------
# Slug injectivity — codex round 2 finding
# ----------------------------------------------------------------------


class TestSafeModelSlug:
    def test_distinct_inputs_produce_distinct_slugs(self):
        # Pairs that previously collided under the old replace-based
        # scheme but must stay distinct now.
        pairs = [
            ("foo/bar", "foo__bar"),
            ("foo bar", "foo_bar"),
            ("a/b", "a%2Fb"),
        ]
        for a, b in pairs:
            assert safe_model_slug(a) != safe_model_slug(b), (
                f"slug collision: {a!r} and {b!r} both map to {safe_model_slug(a)!r}"
            )

    def test_simple_aliases_unchanged(self):
        # Common case: an alias with no special chars stays human-readable.
        assert safe_model_slug("qwen3.5-4b-4bit") == "qwen3.5-4b-4bit"

    def test_hf_path_round_trips_via_unquote(self):
        import urllib.parse

        original = "mlx-community/Qwen3.5-4B-MLX-4bit"
        slug = safe_model_slug(original)
        assert urllib.parse.unquote(slug) == original
