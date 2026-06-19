# SPDX-License-Identifier: Apache-2.0
"""Prometheus ``/metrics`` exposition endpoint (issue #701).

A single unauthenticated ``GET /metrics`` route that renders the existing
counters/gauges exposed by ``engine.get_stats()`` / ``scheduler.get_stats()``
in Prometheus text exposition format.

Design choices
--------------
- **No new runtime dependency.** The text exposition format is short and
  well-specified (https://prometheus.io/docs/instrumenting/exposition_formats/).
  Hand-rolling ~40 LOC avoids pulling in ``prometheus_client`` (and its
  global default registry, which would fight with multi-engine tests).
- **No new instrumentation sites.** Every metric maps onto a field that
  ``engine.get_stats()`` already returns — no per-request hot-path cost,
  no new counters scattered across the engine.
- **Unauthenticated**, on ``probe_router`` rather than the auth-gated
  router, to match the standard Prometheus scrape model. Mirrors
  ``/healthz`` exactly.

  The disclosure surface is intentional and matches industry convention
  (Linkerd, Envoy, nginx-prom-exporter, kubelet, etcd all expose /metrics
  without auth). The trust boundary is the network — operators are
  expected to put /metrics behind a private VIP, mTLS, or a sidecar
  proxy. Prometheus 2.x scrape configs *can* carry bearer tokens
  (``authorization`` section in ``scrape_config``), so this is a
  deliberate convention choice rather than a protocol limitation:
  matching the de-facto pattern keeps rapid-mlx interoperable with the
  large body of existing Prometheus tooling that assumes an unauth
  ``/metrics`` target.
- **Engine-not-loaded** is a 200, not a 500 — Prometheus would otherwise
  drop the entire target. Build info is always emitted.
- **Counter monotonicity** — the cache stats backing the
  ``rapid_mlx_prefix_cache_*_total`` series are reset to zero whenever
  the cache is cleared (admin-triggered via ``POST /cache/clear`` or
  internal recovery paths). Prometheus counters MUST be monotonically
  non-decreasing for ``rate()`` to work; otherwise ``rate()`` will spike
  to ``+Inf`` or go negative the scrape after a clear. The
  ``_StickyCounterAccumulator`` below snapshots the previous raw value
  on every scrape and folds resets into a baseline, so the exposed
  counter never decreases for the lifetime of the process.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from .. import __version__
from ..config import get_config

router = APIRouter()

# Prometheus text exposition format 0.0.4.
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class _StickyCounterAccumulator:
    """Make a resettable underlying counter look monotonic to Prometheus.

    The prefix/paged/memory-aware caches expose ``hits``/``misses``/
    ``evictions``/``tokens_saved`` that reset to zero on ``cache.clear()``
    (admin ``POST /cache/clear`` and a few internal recovery paths). If we
    forwarded those raw values to Prometheus they would decrement, and
    ``rate()`` would either spike to ``+Inf`` (overflow detection in
    Prometheus 2.x) or go negative for one scrape — both visibly wrong on
    dashboards.

    Strategy: on each ``advance(key, raw)`` call, compare ``raw`` to the
    previously-seen raw value for that ``key``. If ``raw < last_raw`` we
    assume the underlying source was reset (e.g. ``cache.clear()``) and
    fold the previously-exposed total into a baseline. The exposed value
    is always ``baseline + raw``, which is monotonic.

    Race notes (audit-relevant):
    - All state mutations happen under a single ``threading.Lock``. A
      concurrent scrape will see either the pre-advance or post-advance
      snapshot — never a torn baseline.
    - Reads use ``int`` so the bookkeeping is allocation-free per scrape.
    - The accumulator state is process-local. A process restart resets
      all counters to whatever the cache currently reports (matches every
      other Prometheus client library — ``process_start_time_seconds`` is
      how scrapers detect this).
    """

    def __init__(self) -> None:
        # key → (last_raw_seen, baseline_added_on_resets)
        self._state: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def advance(self, key: str, raw: int) -> int:
        """Return a monotonic value for ``raw``, recording state for ``key``.

        Args:
            key: stable identifier for the underlying counter (we use the
                fully-qualified Prometheus metric name).
            raw: latest raw value read from the cache stats dict.

        Returns:
            Monotonic counter value to expose to Prometheus.
        """
        raw = max(0, int(raw))  # defensively floor at 0
        with self._lock:
            last_raw, baseline = self._state.get(key, (0, 0))
            if raw < last_raw:
                # The underlying counter was reset. Fold what we'd already
                # exposed (last_raw) into the baseline so the series
                # resumes from there.
                baseline = baseline + last_raw
            self._state[key] = (raw, baseline)
            return baseline + raw


# Module-level accumulator — one process, one cumulative cache series.
_cache_counter_accumulator = _StickyCounterAccumulator()


def _reset_accumulator_for_tests() -> None:
    """Test-only hook: clear the sticky-counter state between tests."""
    global _cache_counter_accumulator
    _cache_counter_accumulator = _StickyCounterAccumulator()


def _escape_label_value(value: str) -> str:
    """Escape a label value per the text exposition spec.

    Backslash, double-quote, and newline are the only required escapes.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_metric(
    name: str,
    metric_type: str,
    help_text: str,
    value: float | int,
    labels: dict[str, str] | None = None,
) -> list[str]:
    """Render one metric (HELP + TYPE + single sample) as line list."""
    out = [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} {metric_type}",
    ]
    if labels:
        label_str = ",".join(
            f'{k}="{_escape_label_value(str(v))}"' for k, v in labels.items()
        )
        out.append(f"{name}{{{label_str}}} {value}")
    else:
        out.append(f"{name} {value}")
    return out


def _coerce_number(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric coercion — Prometheus samples must be numbers.

    ``get_stats`` returns ``None`` for fields the active engine cannot
    populate (e.g. Metal stats on a non-Metal host). Treat those as 0
    rather than dropping the series — operator dashboards prefer a
    flat line at 0 to a missing metric (which would flip a stat panel
    to "no data").
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _render_prometheus(cfg: Any) -> str:
    """Render the full /metrics body for a snapshot of cfg.engine state."""
    lines: list[str] = []

    # Always-on: build info as a gauge fixed at 1 (Prometheus convention).
    # Lets dashboards/alerts filter by version without a separate label.
    lines.extend(
        _fmt_metric(
            "rapid_mlx_build_info",
            "gauge",
            "Build info as constant 1 (version/model carried in labels).",
            1,
            labels={
                "version": __version__,
                "model": cfg.model_name or "",
            },
        )
    )

    if cfg.engine is None:
        # No engine yet — return only build info. Prometheus must NOT see
        # a 500 here or the whole target goes "down" between restarts.
        return "\n".join(lines) + "\n"

    try:
        stats: dict[str, Any] = cfg.engine.get_stats() or {}
    except Exception:
        # Even a partially-initialized engine must not poison /metrics.
        # Fall back to build_info only so the scrape target stays up.
        return "\n".join(lines) + "\n"

    # ---- Scheduler counters & gauges -----------------------------------
    lines.extend(
        _fmt_metric(
            "rapid_mlx_requests_processed_total",
            "counter",
            "Cumulative requests that have completed processing.",
            int(_coerce_number(stats.get("num_requests_processed"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_prompt_tokens_total",
            "counter",
            "Cumulative prompt tokens consumed across all requests.",
            int(_coerce_number(stats.get("total_prompt_tokens"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_completion_tokens_total",
            "counter",
            "Cumulative completion tokens generated across all requests.",
            int(_coerce_number(stats.get("total_completion_tokens"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_requests_running",
            "gauge",
            "Requests currently in the running batch.",
            int(_coerce_number(stats.get("num_running"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_requests_waiting",
            "gauge",
            "Requests queued and waiting for a batch slot.",
            int(_coerce_number(stats.get("num_waiting"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_steps_executed_total",
            "counter",
            "Cumulative scheduler steps executed since engine start.",
            int(_coerce_number(stats.get("steps_executed"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_uptime_seconds",
            "gauge",
            "Engine uptime in seconds.",
            round(_coerce_number(stats.get("uptime_seconds")), 3),
        )
    )

    # ---- Metal memory (best-effort; may be absent on non-Metal hosts) --
    # get_stats reports GB rounded — convert back to bytes for the standard
    # Prometheus byte-unit convention. None → 0 via _coerce_number.
    for stat_key, metric_name, help_text in (
        (
            "metal_active_memory_gb",
            "rapid_mlx_metal_active_memory_bytes",
            "Active Metal memory in bytes.",
        ),
        (
            "metal_peak_memory_gb",
            "rapid_mlx_metal_peak_memory_bytes",
            "Peak Metal memory in bytes.",
        ),
        (
            "metal_cache_memory_gb",
            "rapid_mlx_metal_cache_memory_bytes",
            "Metal allocator cache in bytes.",
        ),
    ):
        gb = _coerce_number(stats.get(stat_key))
        lines.extend(
            _fmt_metric(
                metric_name,
                "gauge",
                help_text,
                int(gb * 1_000_000_000),
            )
        )

    # ---- Prefix / paged / memory-aware cache (one of the three) --------
    # Each cache variant exposes ``hits``/``misses``/``evictions``/
    # ``tokens_saved`` under different parent keys. Pick whichever is
    # present so the metric series stays stable across deploys that swap
    # cache implementations via flags.
    cache_stats: dict[str, Any] | None = None
    for cache_key in ("memory_aware_cache", "paged_cache", "prefix_cache"):
        candidate = stats.get(cache_key)
        if isinstance(candidate, dict):
            cache_stats = candidate
            break

    if cache_stats is not None:
        # The raw cache counters are reset by ``cache.clear()``; pipe each
        # one through the sticky accumulator so the exposed value never
        # decreases (Prometheus counter contract — required by rate()).
        for raw_key, metric_name, help_text in (
            (
                "hits",
                "rapid_mlx_prefix_cache_hits_total",
                "Prefix-cache lookups that hit a cached entry.",
            ),
            (
                "misses",
                "rapid_mlx_prefix_cache_misses_total",
                "Prefix-cache lookups that missed.",
            ),
            (
                "evictions",
                "rapid_mlx_prefix_cache_evictions_total",
                "Prefix-cache entries evicted by the LRU policy.",
            ),
            (
                "tokens_saved",
                "rapid_mlx_prefix_cache_tokens_saved_total",
                "Prompt tokens skipped thanks to prefix-cache hits.",
            ),
        ):
            raw = int(_coerce_number(cache_stats.get(raw_key)))
            monotonic = _cache_counter_accumulator.advance(metric_name, raw)
            lines.extend(_fmt_metric(metric_name, "counter", help_text, monotonic))

    # Prometheus requires a trailing newline.
    return "\n".join(lines) + "\n"


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint.

    Unauthenticated by design — Prometheus scrapers cannot send a bearer
    token. Mounted on the probe router so ``--api-key`` does not gate it.
    Cheap to call: one ``engine.get_stats()`` snapshot, no engine work.
    """
    cfg = get_config()
    body = _render_prometheus(cfg)
    return PlainTextResponse(content=body, media_type=_CONTENT_TYPE)
