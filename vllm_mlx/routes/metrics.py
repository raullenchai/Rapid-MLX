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

    # ---- PFlash observability (M-02 reframe) ---------------------------
    # When PFlash compression engages, the prompt skips the prefix-cache
    # fetch + store paths entirely (the compressed sequence is a
    # positional fiction — see ``compress_request_tokens`` in
    # scheduler.py). Without these two counters, /metrics looks frozen
    # at ``hits=0/misses=1`` on verified-tier aliases where PFlash is
    # always-on, and operators conclude the prefix cache is broken.
    # ``bypass_total`` counts requests that took the PFlash bypass;
    # ``compressed_tokens_total`` is cumulative tokens dropped by the
    # compressor (logical minus kept) and is the headline number for
    # capacity planning.
    #
    # These come straight from the scheduler counters which only ever
    # increment, so the sticky accumulator is not required.
    lines.extend(
        _fmt_metric(
            "rapid_mlx_pflash_bypass_total",
            "counter",
            (
                "Requests where PFlash compression engaged and the "
                "prefix-cache fetch/store was bypassed."
            ),
            int(_coerce_number(stats.get("pflash_bypass_count"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_pflash_compressed_tokens_total",
            "counter",
            (
                "Cumulative prompt tokens dropped by PFlash compression "
                "(logical minus kept) across all requests."
            ),
            int(_coerce_number(stats.get("pflash_compressed_tokens_dropped"))),
        )
    )

    # ---- Cancellation observability (M-01) -----------------------------
    # ``rapid_mlx_requests_processed_total`` deliberately excludes aborted
    # requests, so when fifty clients disconnect mid-stream the operator-
    # facing series stays at zero with no way to distinguish "model idle"
    # from "every request bailed". The total counter below ticks once per
    # public-API abort the scheduler accepted (deduplicated against
    # idempotent re-enqueues via ``_pending_abort_ids``), regardless of
    # cause — client disconnect, explicit ``/v1/requests/{id}/cancel``
    # route, timeout, or internal abort. The sub-counter attributes the
    # subset triggered by the disconnect_guard force-abort path so the
    # gap (total - via_disconnect) surfaces explicit-cancel + timeout
    # traffic for capacity planning. Both default to zero on engines
    # that never reach M-01 (mirrors the PFlash counters' flat-line
    # treatment) so dashboards never flip to "no data" after a deploy.
    lines.extend(
        _fmt_metric(
            "rapid_mlx_requests_cancelled_total",
            "counter",
            (
                "Cumulative requests aborted via the scheduler abort path "
                "(client disconnect, explicit cancel route, timeout). "
                "Disjoint from rapid_mlx_requests_processed_total which "
                "only counts completed requests."
            ),
            int(_coerce_number(stats.get("num_requests_cancelled"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_requests_cancelled_via_disconnect_total",
            "counter",
            (
                "Subset of rapid_mlx_requests_cancelled_total attributed "
                "to client disconnect (force-abort fired from the "
                "disconnect_guard streaming-route helper)."
            ),
            int(_coerce_number(stats.get("num_requests_cancelled_via_disconnect"))),
        )
    )

    # ---- D-METAL-CAP / D-METAL-PFX observability -----------------------
    # Both counters tick from the scheduler and are monotone for the
    # process lifetime, so they bypass the sticky-counter accumulator
    # (no cache.clear() path resets them).
    #
    # ``metal_cap_violations_total`` increments when ``add_request``
    # rejected a new request because Metal active already crossed the
    # ``--gpu-memory-utilization`` soft cap. Pre-fix, MLX's
    # ``set_memory_limit`` silently let the allocator grow past the
    # cap while system RAM remained available, and the only operator-
    # visible signal was an eventual macOS-paging slowdown — this
    # counter is the leading indicator that turns that silent
    # violation into a queryable series.
    #
    # ``prefix_cache_pressure_evictions_total`` increments once per
    # cache entry that the periodic engine_core memory-pressure tick
    # evicted via ``Scheduler.evict_prefix_cache_under_pressure``. This
    # is the headline number for the D-METAL-PFX decode-tps cliff:
    # pre-fix the series stayed at 0 because no pressure-driven
    # eviction existed at all (the only path was LRU-on-capacity,
    # which on 108 entries / 7.7 GB / max_entries=100 already-at-limit
    # never fired again — the cache trie held the slabs, the
    # OrderedDict count was AT limit, not over it).
    lines.extend(
        _fmt_metric(
            "rapid_mlx_metal_cap_violations_total",
            "counter",
            (
                "Requests rejected at admission because Metal active "
                "memory + waiting-request KV reservations + the new "
                "request's projected KV would exceed the "
                "gpu_memory_utilization soft cap (D-METAL-CAP). "
                "Increments on EITHER ``active >= cap`` (sustained "
                "over-cap storm) OR ``active + reserved + projected "
                ">= cap`` (single large prefill that would push the "
                "allocator past cap on its own grow path)."
            ),
            int(_coerce_number(stats.get("num_metal_cap_violations"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_prefix_cache_pressure_evictions_total",
            "counter",
            (
                "Prefix-cache entries evicted by the Metal-pressure "
                "trigger (D-METAL-PFX). Disjoint from "
                "rapid_mlx_prefix_cache_evictions_total which counts "
                "LRU-on-capacity evictions performed by the cache "
                "itself."
            ),
            int(_coerce_number(stats.get("num_prefix_cache_pressure_evictions"))),
        )
    )

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
