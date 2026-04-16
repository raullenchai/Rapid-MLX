# SPDX-License-Identifier: Apache-2.0
"""Baseline storage + threshold-based regression comparison.

Baselines live at ``harness/baselines/{tier}.json`` and are checked into
git.  Thresholds live at ``harness/thresholds.yaml`` and are loaded once
per run.

A baseline file looks like::

    {
      "captured_at": "2026-04-15T21:00:00",
      "rapid_mlx_version": "0.5.1",
      "model": "qwen3.5-4b",
      "metrics": {
        "decode_tps": 156.2,
        "ttft_cold_ms": 412,
        ...
      }
    }

The comparator returns one ``MetricDelta`` per metric so the report can
render a complete delta table — not just the regressions.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .runner import HARNESS_DIR

# ---------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------

# Default thresholds used when a metric is not listed in thresholds.yaml.
# Conservative on perf (5%), zero-tolerance on accuracy.
DEFAULT_THRESHOLDS = {"regression_pct": 5, "improvement_pct": 10}


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-metric regression thresholds from YAML.

    Falls back to an empty dict if PyYAML isn't installed — callers will
    then use ``DEFAULT_THRESHOLDS`` for every metric.  We avoid making
    PyYAML a hard dependency just for this file: if it's missing the
    feature degrades gracefully rather than blocking the whole tier.
    """
    path = path or (HARNESS_DIR / "thresholds.yaml")
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


# ---------------------------------------------------------------------
# Baseline read/write
# ---------------------------------------------------------------------

def _safe_model_slug(model: str) -> str:
    """File-system-safe, *injective* slug for baseline filenames.

    Uses URL percent-encoding so the mapping is one-to-one — i.e.
    ``foo/bar``, ``foo__bar`` and ``foo bar`` all produce distinct
    files.  A simple ``replace('/', '__')`` would collide on names
    that legitimately contain ``__``.
    """
    import urllib.parse

    # safe='' so '/' (and '%') are quoted; '-' and '.' kept for
    # readability since they appear in nearly every model id.
    return urllib.parse.quote(model, safe="-.")


def baseline_path(tier: str, model: str | None = None) -> Path:
    """Path to the baseline JSON.

    Per-model file when ``model`` is given (used by check/full tiers
    where a baseline is only meaningful for one model).  The unsuffixed
    path is reserved for tiers whose metrics aren't model-specific.
    """
    base = HARNESS_DIR / "baselines"
    if model is None:
        return base / f"{tier}.json"
    return base / f"{tier}-{_safe_model_slug(model)}.json"


def load_baseline(tier: str, model: str | None = None) -> dict | None:
    """Return the baseline dict, or None if absent."""
    p = baseline_path(tier, model)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def save_baseline(tier: str, model: str, metrics: dict[str, float]) -> Path:
    """Write a fresh baseline file (called by --update-baselines flow)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        ver = version("rapid-mlx")
    except PackageNotFoundError:
        ver = "unknown"

    payload = {
        "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "rapid_mlx_version": ver,
        "model": model,
        "metrics": metrics,
    }
    p = baseline_path(tier, model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return p


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

class DeltaStatus(str, Enum):
    OK = "ok"               # within thresholds
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    NEW = "new"             # metric absent in baseline
    DROPPED = "dropped"     # metric absent in current run


@dataclass
class MetricDelta:
    metric: str
    baseline: float | None
    current: float | None
    delta_pct: float | None
    status: DeltaStatus
    threshold_pct: float | None = None


# Metrics where SMALLER values are better (latency, memory).
# Keep this set in sync with what scripts/autoresearch_bench.py emits.
_LOWER_IS_BETTER = {
    # autoresearch metric names (canonical)
    "cold_ttft_ms",
    "cached_ttft_ms",
    "mt_ttft_ms",
    "long_ttft_ms",
    "long_cached_ttft_ms",
    "tc_latency_ms",          # tool-call latency: lower = better
    "peak_ram_mb",
    # legacy / alternate names retained so older baselines and
    # external bench scripts using *ttft_cold_ms* style keys also
    # diff with the correct sign.
    "ttft_cold_ms",
    "ttft_cached_ms",
    "multiturn_ttft_ms",
    "long_prompt_ttft_ms",
    "tool_latency_s",
    "tool_latency_ms",
}


def _is_lower_better(metric: str) -> bool:
    return metric in _LOWER_IS_BETTER


def _classify(
    metric: str,
    baseline: float,
    current: float,
    thresholds: dict[str, dict[str, float]],
) -> tuple[DeltaStatus, float, float]:
    """Classify (status, delta_pct, regression_threshold)."""
    cfg = thresholds.get(metric, DEFAULT_THRESHOLDS)
    regression_pct = float(cfg.get("regression_pct", DEFAULT_THRESHOLDS["regression_pct"]))
    improvement_pct = float(cfg.get("improvement_pct", DEFAULT_THRESHOLDS["improvement_pct"]))

    if baseline == 0:
        # Avoid div-by-zero — treat any change as a status flip.
        if current == 0:
            return DeltaStatus.OK, 0.0, regression_pct
        return DeltaStatus.IMPROVEMENT if not _is_lower_better(metric) else DeltaStatus.REGRESSION, float("inf"), regression_pct

    raw_delta_pct = (current - baseline) / baseline * 100
    # For lower-is-better metrics, invert the sign so positive delta_pct
    # always means "worse".
    delta_pct = -raw_delta_pct if _is_lower_better(metric) else raw_delta_pct

    # delta_pct > 0  means improvement
    # delta_pct < 0  means regression (more negative = worse)
    if delta_pct < -regression_pct:
        return DeltaStatus.REGRESSION, delta_pct, regression_pct
    if delta_pct > improvement_pct:
        return DeltaStatus.IMPROVEMENT, delta_pct, regression_pct
    return DeltaStatus.OK, delta_pct, regression_pct


def compare(
    current: dict[str, float],
    baseline: dict | None,
    thresholds: dict[str, dict[str, float]],
) -> list[MetricDelta]:
    """Diff current metrics against baseline, return per-metric deltas.

    Caller decides what to do with regressions — typically marks the
    enclosing CheckResult as ``Status.REGRESSION``.
    """
    deltas: list[MetricDelta] = []
    base_metrics: dict[str, float] = (baseline or {}).get("metrics", {}) if baseline else {}

    all_metrics = sorted(set(current) | set(base_metrics))
    for m in all_metrics:
        cur = current.get(m)
        base = base_metrics.get(m)
        if base is None:
            deltas.append(
                MetricDelta(metric=m, baseline=None, current=cur,
                            delta_pct=None, status=DeltaStatus.NEW)
            )
            continue
        if cur is None:
            deltas.append(
                MetricDelta(metric=m, baseline=base, current=None,
                            delta_pct=None, status=DeltaStatus.DROPPED)
            )
            continue
        status, delta_pct, threshold = _classify(m, base, cur, thresholds)
        deltas.append(
            MetricDelta(
                metric=m, baseline=base, current=cur,
                delta_pct=delta_pct, status=status, threshold_pct=threshold,
            )
        )
    return deltas


def render_deltas_md(deltas: list[MetricDelta]) -> str:
    """Render a delta table as markdown.  Used in tier reports."""
    if not deltas:
        return "_no metrics captured_\n"
    lines = [
        "| Metric | Baseline | Current | Δ% | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for d in deltas:
        b = f"{d.baseline:.2f}" if isinstance(d.baseline, (int, float)) else "—"
        c = f"{d.current:.2f}" if isinstance(d.current, (int, float)) else "—"
        if d.delta_pct is None:
            dp = "—"
        else:
            sign = "+" if d.delta_pct >= 0 else ""
            dp = f"{sign}{d.delta_pct:.1f}%"
        lines.append(f"| {d.metric} | {b} | {c} | {dp} | {d.status.value} |")
    return "\n".join(lines) + "\n"


def has_regression(deltas: list[MetricDelta]) -> bool:
    return any(d.status == DeltaStatus.REGRESSION for d in deltas)
