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


def _render_kv_cache_dtype_gauge(cfg: Any) -> list[str]:
    """Emit the R15 #300 ``rapid_mlx_kv_cache_dtype`` gauge.

    Three series — ``dtype="bf16"`` / ``"int8"`` / ``"int4"`` — exactly
    one of which is 1, the others 0. Lets a dashboard wire a single
    alert / panel against ``rapid_mlx_kv_cache_dtype{dtype="int4"} ==
    1`` without parsing string-valued samples (which Prometheus does
    not support natively).

    The dtype is read from ``cfg.engine.scheduler_config.kv_cache_dtype``
    when the engine is up, otherwise from ``cfg.kv_cache_dtype`` if the
    server stashed it pre-load, otherwise defaults to ``"bf16"`` (the
    only value that's a no-op everywhere — never silently report
    int4 when we're not actually quantized).
    """
    # codex r1 BLOCKING #2: only fall back to the pre-load stash when
    # the engine has NOT yet stamped its scheduler_config. The earlier
    # ``dtype == "bf16"`` guard let a stale stash override a live
    # engine after a bf16 load — e.g. operator loads with
    # ``--kv-cache-dtype int4`` against a sliding-window model, the
    # safelist resolves to bf16, but the stash still says int4 from
    # before the safelist ran. Distinguishing "engine reports a value"
    # from "no engine value available" prevents that ghost report.
    #
    # codex r2 BLOCKING #2: ``SchedulerConfig.kv_cache_dtype`` now
    # carries a default of ``"bf16"``, so a programmatic caller that
    # only set the pre-existing legacy fields
    # (``kv_cache_quantization=True`` + ``kv_cache_quantization_bits``)
    # without touching ``kv_cache_dtype`` would have us report ``bf16``
    # while int4 / int8 KV cache is actually live. When the dtype field
    # is unmodified-default but legacy quantization is on, derive the
    # effective dtype from the legacy bits — that's the only path that
    # keeps the gauge honest for callers that pre-date the dtype field.
    dtype: str | None = None
    try:
        engine = getattr(cfg, "engine", None)
        if engine is not None:
            sc = getattr(engine, "scheduler_config", None) or getattr(
                engine, "_scheduler_config", None
            )
            if sc is not None:
                live = getattr(sc, "kv_cache_dtype", None)
                if live:
                    dtype = live
                # Legacy-caller cross-check: if the dtype field is at
                # its default but the legacy quantization toggle is on,
                # the legacy fields tell the truth.
                if dtype in (None, "bf16") and getattr(
                    sc, "kv_cache_quantization", False
                ):
                    bits = getattr(sc, "kv_cache_quantization_bits", None)
                    if bits == 4:
                        dtype = "int4"
                    elif bits == 8:
                        dtype = "int8"
                    # Any other bits value is a misconfiguration the CLI
                    # rejects (codex r2 BLOCKING #1) — leave dtype as the
                    # honest bf16 default rather than guessing a label.
        if dtype is None:
            # Engine not loaded yet (or doesn't carry the field) — fall
            # back to the pre-load stash so /metrics still reports the
            # operator's resolved dtype during the load window.
            stashed = getattr(cfg, "kv_cache_dtype", None)
            if stashed:
                dtype = stashed
    except Exception:
        dtype = None
    # codex r3 BLOCKING: a typo / future dtype string / stale field
    # value not in {"bf16","int8","int4"} would render every series at
    # 0, violating this gauge's "exactly one is 1" contract and making
    # dashboards read "no active dtype" — which is worse than wrong, it
    # looks like the metric is broken. Validate against the known set
    # and fall back to ``"bf16"`` (the only no-op value) for unknowns,
    # so the contract holds for every input.
    if dtype not in ("bf16", "int8", "int4"):
        dtype = "bf16"

    out: list[str] = [
        "# HELP rapid_mlx_kv_cache_dtype Effective KV cache dtype "
        "(R15 #300). One series per dtype label; the value is 1 for "
        "the active dtype and 0 for the others.",
        "# TYPE rapid_mlx_kv_cache_dtype gauge",
    ]
    for candidate in ("bf16", "int8", "int4"):
        active = 1 if dtype == candidate else 0
        out.append(f'rapid_mlx_kv_cache_dtype{{dtype="{candidate}"}} {active}')
    return out


def _render_response_format_counters() -> list[str]:
    """Render the H-06 strict-mode counters as Prometheus lines.

    Pulled into its own helper so the metrics endpoint can emit these
    counters even when ``engine.get_stats()`` is unavailable (engine
    not yet loaded, or partially-initialized — both early-return
    branches must still surface the response_format counters because
    they live in their own module-level state and aren't engine-bound).
    """
    try:
        from ..api.response_format_metrics import snapshot as _rf_snapshot

        rf_stats = _rf_snapshot()
    except Exception:
        rf_stats = {
            "strict_requests_total": 0,
            "strict_violations_total": 0,
            "strict_repairs_attempted_total": 0,
            "strict_repairs_succeeded_total": 0,
            "strict_repairs_skipped_context_overflow_total": 0,
        }
    out: list[str] = []
    out.extend(
        _fmt_metric(
            "rapid_mlx_response_format_strict_total",
            "counter",
            (
                "Requests with response_format.type=json_schema and "
                "strict=true (H-06). Counts admitted strict requests "
                "regardless of whether the [guided] extra was installed "
                "— installs missing the extra now fall through to the "
                "post-generate validation + repair-retry path (R12-4)."
            ),
            int(rf_stats.get("strict_requests_total", 0)),
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_response_format_strict_violations_total",
            "counter",
            (
                "Strict json_schema responses that failed post-decode "
                "jsonschema.validate (H-06). Constrained decoding via "
                "outlines should make this unreachable — any non-zero "
                "rate on a guided install signals that the guided path "
                "silently degraded. On non-guided installs this counts "
                "the requests that ultimately surfaced 422 to the client "
                "after the R12-4 repair retry also failed."
            ),
            int(rf_stats.get("strict_violations_total", 0)),
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_response_format_strict_repairs_attempted_total",
            "counter",
            (
                "R12-4 strict-mode auto-repair attempts. Ticks once per "
                "request whose initial unconstrained output failed "
                "jsonschema.validate and was re-prompted with a "
                "system-injected repair hint. Includes attempts that "
                "ultimately still failed (those also bump "
                "rapid_mlx_response_format_strict_violations_total)."
            ),
            int(rf_stats.get("strict_repairs_attempted_total", 0)),
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_response_format_strict_repairs_succeeded_total",
            "counter",
            (
                "R12-4 strict-mode auto-repair successes. Ticks when an "
                "auto-repair attempt produced output that validated "
                "against the supplied schema. Divide by "
                "rapid_mlx_response_format_strict_repairs_attempted_total "
                "for the repair success rate — low rates suggest the "
                "client's schema is too restrictive for the model."
            ),
            int(rf_stats.get("strict_repairs_succeeded_total", 0)),
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_response_format_strict_repairs_skipped_context_overflow_total",
            "counter",
            (
                "H-06 #267b strict-mode repair-retry skips. Ticks when "
                "the post-build repair prompt (instructions + schema + "
                "up to 4 KiB of failed output) would have exceeded the "
                "engine's context window. The route skips the retry and "
                "surfaces the ORIGINAL 422 json_schema_violation envelope "
                "instead of 502 strict_repair_engine_failure, so clients "
                "see a deterministic validation outcome. A non-zero rate "
                "signals the repair prompt template is too large for the "
                "deployed model's context window."
            ),
            int(rf_stats.get("strict_repairs_skipped_context_overflow_total", 0)),
        )
    )
    return out


def _render_spec_decode_mtp_counters(cfg: Any) -> list[str]:
    """Render the R15-P1 #302 MTP speculative-decode counter triplet.

    Three counters + one gauge, all labeled ``family="qwen3.5"`` and
    ``method="mtp"`` so a future tree-MTP variant or a different model
    family (Qwen3.6 vs 3.5) lands cleanly without renaming:

    * ``rapid_mlx_spec_decode_attempts_total`` — Number of MTP draft
      proposals the generator made. Bumped once per
      ``mtp_generate_step`` outer-loop verify iteration.
    * ``rapid_mlx_spec_decode_accepts_total`` — Subset of attempts
      that the verify backbone pass accepted (after
      ``min(1, p_target/p_draft)`` at temp>0 or exact-match at
      temp=0). Always ``<= attempts``.
    * ``rapid_mlx_spec_decode_tokens_saved_total`` — Cumulative bonus
      tokens emitted from draft acceptance. Equals ``accepts`` for
      chain MTP; tree MTP would let this exceed ``accepts``.
    * ``rapid_mlx_spec_decode_accept_ratio`` — ``accepts / attempts``
      as a gauge (0.0 when attempts==0, never NaN). The lossless
      contract surface — dashboards alert on this dropping below 0.80
      for Qwen3.5-9B-w4 at temp=0.

    Process-local, no sticky accumulator needed: the underlying
    counter never resets (see
    :class:`vllm_mlx.spec_decode.mtp.MTPAcceptCounter`).

    ``cfg.model_alias`` is reported as the ``family`` label so an
    operator running multiple Qwen3.5 / 3.6 variants in a multi-model
    fleet (#387) can split the dashboard panel by alias. Falls back
    to ``"qwen3.5"`` when the alias is unknown so the series stays
    stable across cold-start (Prometheus drops series whose label
    set changes — a transient ``""`` label would break ``rate()``).
    """
    try:
        from ..spec_decode.mtp import get_global_counter
    except ImportError:
        # spec_decode.mtp is part of the rapid-mlx package and should
        # always be importable. The defensive catch keeps /metrics
        # rendering robust in case the package is partially installed
        # (e.g. mid-upgrade, stale .pyc) — emit zero-valued series so
        # dashboards don't break.
        return [
            "# HELP rapid_mlx_spec_decode_attempts_total MTP draft "
            "proposals (R15-P1 #302).",
            "# TYPE rapid_mlx_spec_decode_attempts_total counter",
            'rapid_mlx_spec_decode_attempts_total{family="qwen3.5",method="mtp"} 0',
            "# HELP rapid_mlx_spec_decode_accepts_total MTP drafts "
            "accepted by the verify backbone pass.",
            "# TYPE rapid_mlx_spec_decode_accepts_total counter",
            'rapid_mlx_spec_decode_accepts_total{family="qwen3.5",method="mtp"} 0',
            "# HELP rapid_mlx_spec_decode_accept_ratio MTP accepts / "
            "attempts. 0.0 when attempts==0.",
            "# TYPE rapid_mlx_spec_decode_accept_ratio gauge",
            'rapid_mlx_spec_decode_accept_ratio{family="qwen3.5",method="mtp"} 0',
            "# HELP rapid_mlx_spec_decode_tokens_saved_total Bonus "
            "tokens emitted from accepted MTP drafts (cumulative).",
            "# TYPE rapid_mlx_spec_decode_tokens_saved_total counter",
            'rapid_mlx_spec_decode_tokens_saved_total{family="qwen3.5",method="mtp"} 0',
        ]

    snapshot = get_global_counter().snapshot()

    # Family label sourced from cfg.model_alias when present so a
    # multi-model fleet can split by alias. The alias name typically
    # already contains the family ("qwen3.5-9b-4bit" → "qwen3.5-9b-4bit").
    # We use the alias verbatim rather than re-deriving the family from
    # config.json — that re-derivation would add a config.json round-
    # trip to /every/ scrape, which is wasteful for a hot endpoint.
    #
    # ``getattr`` rather than direct attribute access so the test
    # harness's ``types.SimpleNamespace`` cfg stubs (see
    # tests/test_metal_cap_enforcement.py::TestMetricsRoute) keep
    # working — those stubs intentionally only define the engine
    # fields they exercise and would otherwise raise AttributeError
    # here.
    family = getattr(cfg, "model_alias", None) or "qwen3.5"

    common_labels = {"family": family, "method": "mtp"}
    out: list[str] = []
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_attempts_total",
            "counter",
            (
                "MTP draft proposals (R15-P1 #302, mlx-lm PR #990). "
                "Bumped once per mtp_generate_step verify iteration. "
                "Pair with rapid_mlx_spec_decode_accepts_total to "
                "compute the accept ratio (also surfaced as "
                "rapid_mlx_spec_decode_accept_ratio)."
            ),
            int(snapshot.attempts),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_accepts_total",
            "counter",
            (
                "MTP drafts accepted by the verify backbone pass. "
                "Always <= rapid_mlx_spec_decode_attempts_total. The "
                "lossless contract surface — under the chain MTP "
                "variant a low ratio is a speedup signal, not a "
                "correctness one (tokens stay byte-identical to the "
                "non-spec-decode path)."
            ),
            int(snapshot.accepts),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_accept_ratio",
            "gauge",
            (
                "accepts / attempts. 0.0 when no attempts (Prometheus "
                "convention: no-data → 0 rather than NaN so dashboards "
                "don't flip to no-data during cold start)."
            ),
            round(snapshot.accept_ratio, 4),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_tokens_saved_total",
            "counter",
            (
                "Cumulative bonus tokens emitted from accepted MTP "
                "drafts. Equals rapid_mlx_spec_decode_accepts_total "
                "under chain MTP (one accept = one bonus token); a "
                "future tree MTP variant would let this exceed "
                "accepts."
            ),
            int(snapshot.tokens_saved),
            labels=common_labels,
        )
    )
    return out


def _render_spec_decode_dflash_counters(cfg: Any) -> list[str]:
    """Render the R15-P1 #313 DFlash speculative-decode counter triplet.

    Symmetric with :func:`_render_spec_decode_mtp_counters` so a single
    dashboard panel can graph both backends side-by-side; only the
    ``method=`` label distinguishes them.

    Four counters + one gauge, all labeled ``family="qwen3.5|qwen3.6"``
    (or the alias verbatim when present) and ``method="dflash"``:

    * ``rapid_mlx_spec_decode_dflash_attempts_total`` — DFlash block
      proposals (bumped once per outer-loop verify iteration).
    * ``rapid_mlx_spec_decode_dflash_accepts_total`` — Blocks where at
      least one position was accepted. Always ``<= attempts``.
    * ``rapid_mlx_spec_decode_dflash_tokens_saved_total`` — Cumulative
      bonus tokens emitted from accepted prefixes. For a fully-accepted
      block of size B this bumps by ``B - 1``; a partial accept of
      ``k`` positions bumps by ``max(0, k - 1)``.
    * ``rapid_mlx_spec_decode_dflash_accept_ratio`` — ``accepts /
      attempts`` gauge. 0.0 when attempts==0; the lossless contract
      surface.
    * ``rapid_mlx_spec_decode_dflash_block_size`` — Observable block
      size (default 16). Surfaced as a gauge so a future "block size 4
      prototype" run is distinguishable from the production "block
      size 16" config without re-deploying.
    """
    try:
        from ..spec_decode.dflash import DEFAULT_BLOCK_SIZE, get_global_counter
    except ImportError:
        # Defensive: same rationale as MTP — keep /metrics rendering
        # robust during partial install / mid-upgrade.
        return [
            "# HELP rapid_mlx_spec_decode_dflash_attempts_total "
            "DFlash block proposals (R15-P1 #313).",
            "# TYPE rapid_mlx_spec_decode_dflash_attempts_total counter",
            'rapid_mlx_spec_decode_dflash_attempts_total{family="qwen3.5",method="dflash"} 0',
            "# HELP rapid_mlx_spec_decode_dflash_accepts_total "
            "DFlash blocks where at least one position was accepted.",
            "# TYPE rapid_mlx_spec_decode_dflash_accepts_total counter",
            'rapid_mlx_spec_decode_dflash_accepts_total{family="qwen3.5",method="dflash"} 0',
            "# HELP rapid_mlx_spec_decode_dflash_accept_ratio "
            "DFlash accepts / attempts. 0.0 when attempts==0.",
            "# TYPE rapid_mlx_spec_decode_dflash_accept_ratio gauge",
            'rapid_mlx_spec_decode_dflash_accept_ratio{family="qwen3.5",method="dflash"} 0',
            "# HELP rapid_mlx_spec_decode_dflash_tokens_saved_total "
            "Cumulative bonus tokens from accepted DFlash prefixes.",
            "# TYPE rapid_mlx_spec_decode_dflash_tokens_saved_total counter",
            'rapid_mlx_spec_decode_dflash_tokens_saved_total{family="qwen3.5",method="dflash"} 0',
            "# HELP rapid_mlx_spec_decode_dflash_block_size "
            "Active DFlash drafter block size (paper default 16).",
            "# TYPE rapid_mlx_spec_decode_dflash_block_size gauge",
            'rapid_mlx_spec_decode_dflash_block_size{family="qwen3.5",method="dflash"} 16',
        ]

    snapshot = get_global_counter().snapshot()

    family = getattr(cfg, "model_alias", None) or "qwen3.5"
    common_labels = {"family": family, "method": "dflash"}
    out: list[str] = []
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_dflash_attempts_total",
            "counter",
            (
                "DFlash block proposals (R15-P1 #313, arxiv 2410.04097). "
                "Bumped once per dflash_generate_step verify iteration. "
                "Pair with rapid_mlx_spec_decode_dflash_accepts_total to "
                "compute the accept ratio (also surfaced as "
                "rapid_mlx_spec_decode_dflash_accept_ratio)."
            ),
            int(snapshot.attempts),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_dflash_accepts_total",
            "counter",
            (
                "DFlash blocks where at least one position was accepted "
                "by the verify forward. Always <= attempts. The lossless "
                "contract surface — a low ratio is a speedup signal, not "
                "a correctness one (tokens stay byte-identical to the "
                "non-spec-decode path)."
            ),
            int(snapshot.accepts),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_dflash_accept_ratio",
            "gauge",
            (
                "accepts / attempts. 0.0 when no attempts (Prometheus "
                "convention: no-data → 0 rather than NaN so dashboards "
                "don't flip to no-data during cold start)."
            ),
            round(snapshot.accept_ratio, 4),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_dflash_tokens_saved_total",
            "counter",
            (
                "Cumulative bonus tokens emitted from accepted DFlash "
                "prefixes. For a fully-accepted block of size B this "
                "bumps by B - 1; a partial accept of k positions bumps "
                "by max(0, k - 1)."
            ),
            int(snapshot.tokens_saved),
            labels=common_labels,
        )
    )
    out.extend(
        _fmt_metric(
            "rapid_mlx_spec_decode_dflash_block_size",
            "gauge",
            (
                "Active DFlash drafter block size. Defaults to 16 (paper "
                "bench value); a future smaller-block prototype run is "
                "distinguishable from production without re-deploying."
            ),
            int(DEFAULT_BLOCK_SIZE),
            labels=common_labels,
        )
    )
    return out


def _render_mxfp4_moe_guardrail_counters() -> list[str]:
    """Render the R15 #297 MoE+MXFP4 / MoE+NVFP4 load-time guardrail counters.

    Delegates to ``_mxfp4_moe_guardrail.render_prometheus_lines()`` so
    the rendering logic lives next to the counter state (and tests can
    exercise it without importing the route module's heavier transitive
    closure — see codex round 3/4 review on #297). On any import error
    we synthesize an all-zero block via the same helper symbol space so
    /metrics always exposes the series, even if the guardrail module
    fails to load for some reason.
    """
    try:
        from .._mxfp4_moe_guardrail import render_prometheus_lines

        return render_prometheus_lines()
    except Exception:
        # Counters never decrease but they may legitimately be missing
        # if the guardrail module fails to import — surface a 0-valued
        # series so dashboards still see the metric name and operators
        # can alert on rapid_mlx_mxfp4_moe_distributed_warnings_total
        # > 0 regardless of import state.
        return [
            "# HELP rapid_mlx_mxfp4_moe_distributed_warnings_total "
            "Load-time warnings for MoE+MXFP4+multi-device cliff (mlx#3402).",
            "# TYPE rapid_mlx_mxfp4_moe_distributed_warnings_total counter",
            "rapid_mlx_mxfp4_moe_distributed_warnings_total 0",
            "# HELP rapid_mlx_nvfp4_moe_warnings_total "
            "Load-time warnings for MoE+NVFP4 dynamic-range loss (mlx#2962).",
            "# TYPE rapid_mlx_nvfp4_moe_warnings_total counter",
            "rapid_mlx_nvfp4_moe_warnings_total 0",
        ]


def _render_turboquant_metrics(cfg: Any) -> list[str]:
    """Render the R15 Phase 4 TurboQuant metrics block.

    Three series:
      * ``rapid_mlx_turboquant_mode{mode="k8v4|v4|disabled"}`` gauge —
        set once at serve boot to the active TurboQuant mode.
      * ``rapid_mlx_turboquant_skipped_total{reason="sliding-window|mla|other"}``
        counter — incremented when a load lands on a model the skip
        list flags as incompatible.
      * ``rapid_mlx_turboquant_fused_kernel{status="available|fallback"}``
        gauge — set once at serve boot to whether the vendored Metal
        kernel compiled. ``fallback`` means callers run the pure-MLX
        reference path; observed numerical output is identical (1e-4
        RMSE in tests) but decode tput is upstream-V-only-grade.

    Resolved from ``SchedulerConfig`` first, falling back to the
    pre-load stash so dashboards see the active mode during the load
    window. Counter state lives in the module-level
    :data:`turboquant_skip_counters` dict, keyed by reason.
    """
    out: list[str] = []

    # Resolve the active mode. Default to ``"disabled"`` when the
    # operator has not flipped TurboQuant on; this keeps the gauge
    # readable on dashboards (the legend filter ``mode="k8v4"`` works
    # without parsing string samples). The pre-load stash on
    # ``ServerConfig`` is the early-boot fallback so /metrics emits a
    # truthful gauge before the engine is up.
    mode: str = "disabled"
    try:
        engine = getattr(cfg, "engine", None)
        if engine is not None:
            sc = getattr(engine, "scheduler_config", None) or getattr(
                engine, "_scheduler_config", None
            )
            if sc is not None:
                if getattr(sc, "kv_cache_turboquant", False):
                    mode = getattr(sc, "kv_cache_turboquant_mode", "v4") or "v4"
        elif getattr(cfg, "turboquant_mode", None):
            mode = cfg.turboquant_mode
    except Exception:
        mode = "disabled"
    if mode not in ("v4", "k8v4", "disabled"):
        mode = "disabled"

    out.append(
        "# HELP rapid_mlx_turboquant_mode Active TurboQuant compression "
        "mode (R15 Phase 4). One series per mode label; value is 1 for "
        "the active mode and 0 for the others."
    )
    out.append("# TYPE rapid_mlx_turboquant_mode gauge")
    for candidate in ("disabled", "v4", "k8v4"):
        active = 1 if mode == candidate else 0
        out.append(f'rapid_mlx_turboquant_mode{{mode="{candidate}"}} {active}')

    # Skip-list counter: emit one series per known reason even when the
    # underlying counter is zero so dashboards don't flip to "no data"
    # between restarts. The counter store lives in
    # ``turboquant_skip_counters`` so the engine load path can bump it
    # via ``record_turboquant_skip(reason)``.
    out.append(
        "# HELP rapid_mlx_turboquant_skipped_total Cumulative model "
        "loads where TurboQuant was requested but the skip list "
        "(sliding-window, MLA, other) forced a fall-back to FP16 KV."
    )
    out.append("# TYPE rapid_mlx_turboquant_skipped_total counter")
    for reason in ("sliding-window", "mla", "other"):
        count = int(turboquant_skip_counters.get(reason, 0))
        out.append(f'rapid_mlx_turboquant_skipped_total{{reason="{reason}"}} {count}')

    # Fused-kernel availability. Cached at module import-time on the
    # first call so /metrics doesn't pay the import cost on every
    # scrape.
    status = _resolve_fused_kernel_status()
    out.append(
        "# HELP rapid_mlx_turboquant_fused_kernel Status of the vendored "
        "TurboQuant fused Metal kernel — ``available`` means decode runs "
        "on the fused path, ``fallback`` means the pure-MLX reference "
        "path is in use (functional, slower)."
    )
    out.append("# TYPE rapid_mlx_turboquant_fused_kernel gauge")
    for candidate in ("available", "fallback"):
        active = 1 if status == candidate else 0
        out.append(
            f'rapid_mlx_turboquant_fused_kernel{{status="{candidate}"}} {active}'
        )
    return out


# Module-level skip-list counter. Keys are the canonical reason
# strings (``"sliding-window"``, ``"mla"``, ``"other"``); the engine
# load path bumps these via :func:`record_turboquant_skip`. Persists
# for the lifetime of the process — matches the convention used by the
# other startup-time counters in this module (mxfp4 guardrail, etc.).
turboquant_skip_counters: dict[str, int] = {
    "sliding-window": 0,
    "mla": 0,
    "other": 0,
}


def record_turboquant_skip(reason: str) -> None:
    """Increment the skip-list counter for ``reason``.

    Called by the engine load path when ``--kv-cache-turboquant`` is on
    but the target model trips the skip list. Unknown reasons are
    folded into ``"other"`` so a typo in a future safelist entry does
    not silently drop the metric.
    """
    key = reason if reason in turboquant_skip_counters else "other"
    turboquant_skip_counters[key] = turboquant_skip_counters.get(key, 0) + 1


_fused_kernel_status_cache: str | None = None


def _resolve_fused_kernel_status() -> str:
    """Memoize the fused-kernel availability check (single import, single Metal probe)."""
    global _fused_kernel_status_cache
    if _fused_kernel_status_cache is not None:
        return _fused_kernel_status_cache
    try:
        from ..turboquant import fused_kernel_status

        _fused_kernel_status_cache = fused_kernel_status()
    except Exception:
        _fused_kernel_status_cache = "fallback"
    return _fused_kernel_status_cache


def _reset_turboquant_state_for_tests() -> None:
    """Test-only hook: clear skip counters and fused-kernel cache."""
    global _fused_kernel_status_cache
    for key in turboquant_skip_counters:
        turboquant_skip_counters[key] = 0
    _fused_kernel_status_cache = None


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

    # R15 #300: KV cache dtype as a labeled gauge. Operators need to see
    # the EFFECTIVE dtype the resolver picked (post-safelist + post-
    # reasoning-pin), not just the requested flag value. Emitted as
    # three series with value 0/1 so a Grafana panel can filter by
    # ``dtype="int4"`` without parsing string-valued samples. Sourced
    # straight off ``SchedulerConfig.kv_cache_dtype`` so this stays
    # truthful even when the legacy ``--kv-cache-quantization`` flag
    # was the actual driver.
    lines.extend(_render_kv_cache_dtype_gauge(cfg))

    # H-06 response_format strict-mode counters — process-local state
    # that is independent of engine availability. Surface BEFORE the
    # engine-None / get_stats-failure early returns so dashboards see
    # the series even between restarts.
    lines.extend(_render_response_format_counters())

    # R15 #297 MoE+MXFP4 / MoE+NVFP4 load-time guardrail counters —
    # same engine-independence rationale: the guardrail fires at
    # ``load_model()``, so the counter MUST be visible to scrapers
    # before the engine reaches its ready state. Otherwise an operator
    # whose model trips the cliff at startup has no metric series to
    # alert on until the FIRST request lands.
    lines.extend(_render_mxfp4_moe_guardrail_counters())

    # R15-P1 #302 MTP spec-decode counter triplet + accept-ratio gauge.
    # Same engine-independence rationale: counters are process-local
    # and bumped from the generator loop, which can run before
    # ``engine.get_stats()`` is ready (warmup) — surface them as
    # zero-valued series so dashboards see the metric names even at
    # cold start. Pre-engine the counter is naturally zero anyway.
    lines.extend(_render_spec_decode_mtp_counters(cfg))

    # R15 Phase 4 TurboQuant series — mode gauge, skip-list counter
    # (one per reason), and fused-kernel availability gauge. Surface
    # BEFORE the engine-None / get_stats-failure early returns so
    # dashboards see the active mode + skip rate even between restarts.
    lines.extend(_render_turboquant_metrics(cfg))

    # R15-P1 #313 DFlash spec-decode counters (mirror of MTP). Surfaced
    # pre-engine for the same dashboard-cold-start reason.
    lines.extend(_render_spec_decode_dflash_counters(cfg))

    if cfg.engine is None:
        # No engine yet — return build info + response_format counters
        # only. Prometheus must NOT see a 500 here or the whole target
        # goes "down" between restarts.
        return "\n".join(lines) + "\n"

    try:
        stats: dict[str, Any] = cfg.engine.get_stats() or {}
    except Exception:
        # Even a partially-initialized engine must not poison /metrics.
        # Fall back to build_info + response_format counters only so the
        # scrape target stays up.
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
            (
                # R10-D (Talia r10-R1): cumulative count of entries the
                # disk loader rejected for any per-entry corruption
                # signal — schema-magic mismatch, length-prefix drift,
                # save-uuid mismatch (orphan from a previous cycle),
                # body-truncated safetensors, or an mlx_lm.load_prompt_cache
                # exception. Pair with ``loaded`` (in /v1/status payload)
                # to graph the reload dropout rate per startup. Closes
                # R9-L4 — operators previously had no Prometheus
                # surface for cache-load corruption beyond grepping
                # ``[cache_persist] SKIPPED`` log lines.
                "load_skipped",
                "rapid_mlx_prefix_cache_load_skipped_total",
                (
                    "Prefix-cache entries rejected at disk-load by the "
                    "per-entry integrity guard (R10-D format-pin: magic, "
                    "length-prefix, save_uuid, or safetensors body check)."
                ),
            ),
            (
                # R12-T1 (dogfood-0815 Talia r12 SEVERE): save-side
                # mirror of ``load_skipped``. Counts entries that
                # ``save_to_disk``'s post-write self-verify pass dropped
                # because the just-written tokens.bin disagreed with
                # the index.json we were about to commit (save_uuid
                # drift or length-prefix mismatch). A non-zero value
                # is the rescue rate — pre-R12-T1 those entries silently
                # corrupted ``cache_dir`` and the next boot refused the
                # whole snapshot via R10-D. Pair with
                # ``load_skipped_total`` to see whether drift is being
                # caught at save (good) or at load (bad — means another
                # path skipped the verify).
                "save_drift_drops",
                "rapid_mlx_prefix_cache_save_drift_drops_total",
                (
                    "Prefix-cache entries dropped at disk-save by the "
                    "post-write self-verify pass (R12-T1: save_uuid or "
                    "length-prefix drift between the just-written "
                    "tokens.bin and the in-flight index.json). A non-zero "
                    "rate is the save path catching corruption before it "
                    "reaches cache_dir."
                ),
            ),
        ):
            raw = int(_coerce_number(cache_stats.get(raw_key)))
            monotonic = _cache_counter_accumulator.advance(metric_name, raw)
            lines.extend(_fmt_metric(metric_name, "counter", help_text, monotonic))

        # ---- R7-M1: prefix-cache cap + current-usage gauges ----------
        # Dogfood-088 (Talia r2) flagged that operators tuning the
        # ``RAPID_MLX_PREFIX_CACHE_MAX_BYTES`` env override had NO
        # Prometheus surface to verify the cap was actually honored
        # at runtime — they could see ``evictions_total`` tick but
        # couldn't graph "how close are we to cap?" or "did our env
        # ceiling stick?". These two gauges close that gap by
        # exposing the same byte values the LRU evict-until-fits loop
        # in MemoryAwarePrefixCache.store() compares against:
        #
        #   * ``rapid_mlx_prefix_cache_cap_bytes`` — the resolved
        #     ceiling from ``MemoryCacheConfig.compute_memory_limit``
        #     (env > programmatic > heuristic > 8 GiB fallback).
        #     Gauge, not counter, because the value is set once at
        #     cache init and reflects current config, not cumulative
        #     work.
        #   * ``rapid_mlx_prefix_cache_current_bytes`` — the cache's
        #     live ledger of how many bytes are pinned by entries.
        #     Pair with cap_bytes to compute utilization headroom
        #     (1 - current/cap) in Prometheus or Grafana without
        #     each consumer re-implementing the math.
        #
        # Both are byte gauges in line with the Prometheus naming
        # convention ("base unit, no suffix"). They sit beside (not
        # replace) the existing ``current_memory_mb`` /
        # ``max_memory_mb`` fields in /v1/status which dashboards
        # already consume.
        lines.extend(
            _fmt_metric(
                "rapid_mlx_prefix_cache_cap_bytes",
                "gauge",
                (
                    "Prefix-cache memory ceiling in bytes (resolved from "
                    "RAPID_MLX_PREFIX_CACHE_MAX_BYTES env override, "
                    "programmatic max_memory_mb, or the heuristic "
                    "fraction-of-RAM default)."
                ),
                int(_coerce_number(cache_stats.get("max_memory_bytes"))),
            )
        )
        lines.extend(
            _fmt_metric(
                "rapid_mlx_prefix_cache_current_bytes",
                "gauge",
                (
                    "Prefix-cache memory currently pinned by cached entries, "
                    "in bytes. Compare to rapid_mlx_prefix_cache_cap_bytes "
                    "for headroom; the cache evicts LRU entries to stay "
                    "below the cap."
                ),
                int(_coerce_number(cache_stats.get("current_memory_bytes"))),
            )
        )

    # ---- R15-P1 radix-tree prefix-cache index (task #303) -------------
    # Radix counters live inside the same ``cache_stats`` dict, nested
    # under ``"radix"``. They are emitted under the
    # ``rapid_mlx_prefix_cache_radix_*`` namespace so the legacy
    # ``rapid_mlx_prefix_cache_*`` series stay byte-identical for
    # dashboards. All counters here are process-monotonic — the radix
    # never resets its cumulative counters on ``clear()`` (see
    # ``RadixStats`` doc) — so the sticky accumulator is not required.
    # The gauges (node_count, entry_count, max_depth, lookup_p50/p99)
    # naturally move up and down, so they are emitted directly without
    # a sticky-counter pin.
    radix_stats = cache_stats.get("radix") if cache_stats is not None else None
    if isinstance(radix_stats, dict):
        for raw_key, metric_name, help_text in (
            (
                "hits",
                "rapid_mlx_prefix_cache_radix_hits_total",
                "Prefix-cache radix-index lookups that resolved to a stored entry.",
            ),
            (
                "misses",
                "rapid_mlx_prefix_cache_radix_misses_total",
                "Prefix-cache radix-index lookups that resolved to no entry.",
            ),
            (
                "inserts",
                "rapid_mlx_prefix_cache_radix_inserts_total",
                "Prefix-cache radix-index inserts (one per cache store).",
            ),
            (
                "removes",
                "rapid_mlx_prefix_cache_radix_removes_total",
                "Prefix-cache radix-index removes (LRU evict + explicit remove).",
            ),
            (
                "deduped_prefix_bytes_saved",
                "rapid_mlx_prefix_cache_radix_deduped_bytes_total",
                (
                    "Cumulative wire-format bytes that the radix index "
                    "collapsed into shared prefix nodes — i.e. the on-disk "
                    "footprint a hash-keyed index would have re-stored. "
                    "Headline number for the 30-80% footprint-reduction "
                    "success criterion."
                ),
            ),
        ):
            lines.extend(
                _fmt_metric(
                    metric_name,
                    "counter",
                    help_text,
                    int(_coerce_number(radix_stats.get(raw_key))),
                )
            )
        for raw_key, metric_name, help_text in (
            (
                "node_count",
                "rapid_mlx_prefix_cache_radix_nodes",
                "Live count of radix-tree nodes (one per shared/unique token edge).",
            ),
            (
                "entry_count",
                "rapid_mlx_prefix_cache_radix_entries",
                "Live count of terminal nodes (== entries the radix indexes).",
            ),
            (
                "max_depth",
                "rapid_mlx_prefix_cache_radix_max_depth",
                "Deepest path through the radix (longest stored sequence).",
            ),
        ):
            lines.extend(
                _fmt_metric(
                    metric_name,
                    "gauge",
                    help_text,
                    int(_coerce_number(radix_stats.get(raw_key))),
                )
            )
        # Lookup-latency gauges are emitted in seconds (Prometheus convention).
        # Floats pass through ``_fmt_metric`` unchanged.
        lines.extend(
            _fmt_metric(
                "rapid_mlx_prefix_cache_radix_lookup_p50_seconds",
                "gauge",
                "p50 lookup latency over the last 256 radix queries, in seconds.",
                float(_coerce_number(radix_stats.get("lookup_p50_seconds"))),
            )
        )
        lines.extend(
            _fmt_metric(
                "rapid_mlx_prefix_cache_radix_lookup_p99_seconds",
                "gauge",
                "p99 lookup latency over the last 256 radix queries, in seconds.",
                float(_coerce_number(radix_stats.get("lookup_p99_seconds"))),
            )
        )

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

    # ---- R15-P1 disk-backed KV checkpoints (task #296) -----------------
    # Counters are process-monotonic by construction (the module-level
    # stats dataclass is only reset via the test-only hook), so the
    # sticky accumulator is unnecessary. ``bytes`` is a gauge because it
    # decreases on eviction. All four series default to 0 on engines
    # that never enabled the feature so dashboards stay flat-line rather
    # than flipping to "no data".
    kv_ckpt_stats = stats.get("kv_checkpoint")
    if not isinstance(kv_ckpt_stats, dict):
        kv_ckpt_stats = {}
    lines.extend(
        _fmt_metric(
            "rapid_mlx_kv_checkpoint_writes_total",
            "counter",
            (
                "Cumulative disk-backed KV checkpoints written at 256-tok "
                "boundaries (R15 #296). Counts the safetensors rename, "
                "not the in-flight .tmp."
            ),
            int(_coerce_number(kv_ckpt_stats.get("writes"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_kv_checkpoint_loads_total",
            "counter",
            (
                "Cumulative disk-backed KV checkpoint reloads through "
                "mlx_lm.load_prompt_cache (R15 #296). Increments only on "
                "a non-None return — partial / corrupt files are logged "
                "but not counted."
            ),
            int(_coerce_number(kv_ckpt_stats.get("loads"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_kv_checkpoint_bytes",
            "gauge",
            (
                "Live total bytes across every committed disk-backed KV "
                "checkpoint under ~/.cache/rapid-mlx/kv_checkpoints/ "
                "(R15 #296). Gauge, not counter, because the value "
                "drops on oldest-first eviction."
            ),
            int(_coerce_number(kv_ckpt_stats.get("bytes"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_kv_checkpoint_evictions_total",
            "counter",
            (
                "Cumulative oldest-first evictions performed against the "
                "disk-backed KV checkpoint root because the byte total "
                "crossed RAPID_MLX_KV_CHECKPOINT_MAX_BYTES (R15 #296)."
            ),
            int(_coerce_number(kv_ckpt_stats.get("evictions"))),
        )
    )
    lines.extend(
        _fmt_metric(
            "rapid_mlx_kv_checkpoint_hook_errors_total",
            "counter",
            (
                "Cumulative unexpected exceptions caught by the scheduler's "
                "disk-KV hook wrapper. Operators expect this to stay 0 — "
                "any non-zero value means the hook is silently bailing on "
                "every call (the typos shipped in #919 sat at debug-level "
                "for two releases without surfacing; this counter is the "
                "regression guard for the same class of bug)."
            ),
            int(_coerce_number(kv_ckpt_stats.get("hook_errors"))),
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
