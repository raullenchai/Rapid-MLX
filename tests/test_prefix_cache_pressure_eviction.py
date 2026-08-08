# SPDX-License-Identifier: Apache-2.0
"""D-METAL-PFX regression tests for Metal-pressure-triggered prefix-cache eviction.

Background
----------
Pre-fix repro on ``qwen3.5-35b-8bit``:
- Fresh server: B=1=82 / B=8=266 agg tok/s.
- After ONE 26.7k-prompt request: B=1 drops to 19 tok/s (-77%),
  B=8 to 92 (-67%) for the ENTIRE session, even after the queue drains.
- An 80-token follow-up bottomed out at 14.6 tok/s.
- Metal allocator cache stuck at 0, prefix-cache LRU-evictions=0 across
  108 entries / 7.7 GB.

Root cause: the only existing eviction policy was LRU-on-capacity, but
``max_entries=100`` was already AT limit, not over it. The cache trie
pinned ~7.7 GB worth of KV slabs and the underlying Metal allocator never
returned them, leaving every subsequent prefill competing with wired
prefix-cache state for the same Metal working set.

Fix tested here:
``Scheduler.evict_prefix_cache_under_pressure()`` — when Metal active
crosses ``metal_pressure_evict_fraction × cap``, evict prefix-cache
entries LRU until pressure drops or the per-tick bound is hit, calling
``mx.clear_cache()`` between evictions so the allocator actually returns
slabs (the bug was that the allocator held them in the free pool with
``get_cache_memory`` reporting 0).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.prefix_cache import BlockCacheEntry
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


def _make_scheduler_with_legacy_cache(
    gpu_memory_utilization: float = 0.5,
) -> Scheduler:
    """Scheduler wired with the legacy ``PrefixCacheManager`` so the
    eviction dispatch hits the trie + OrderedDict LRU branch."""
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=True,
        use_memory_aware_cache=False,  # force legacy cache
        use_paged_cache=False,
        prefix_cache_size=8,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    model = MagicMock()
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


def _make_scheduler_with_memory_aware_cache(
    gpu_memory_utilization: float = 0.5,
) -> Scheduler:
    """Scheduler wired with the ``MemoryAwarePrefixCache`` so the
    eviction dispatch hits the OrderedDict-by-memory branch."""
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        use_paged_cache=False,
        cache_memory_mb=64,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    model = MagicMock()
    return Scheduler(model=model, tokenizer=tokenizer, config=config)


class TestPressureEvictionDispatch:
    """The eviction helper must dispatch to the right cache variant."""

    def test_no_op_when_no_cache(self):
        """No cache configured → no evictions (no crash either)."""
        config = SchedulerConfig(
            enable_prefix_cache=False,
            gpu_memory_utilization=0.5,
        )
        sched = Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=config,
        )
        # Even at extreme pressure, no cache → no eviction.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure()
        assert n == 0

    def test_no_op_when_cap_zero(self):
        """``gpu_memory_utilization=0.0`` means the cap is disabled —
        the eviction helper must short-circuit so back-compat callers
        (existing tests, doctor harness) don't pay any cost."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.0)
        # Stuff some entries into the cache.
        sched.prefix_cache.store_cache([1, 2, 3], ["cache-state-1"])
        sched.prefix_cache.store_cache([4, 5, 6], ["cache-state-2"])
        n = sched.evict_prefix_cache_under_pressure()
        assert n == 0
        # Cache contents untouched.
        assert len(sched.prefix_cache) == 2


class TestLegacyPrefixCacheEviction:
    """Pressure-driven eviction on the legacy ``PrefixCacheManager``."""

    def test_no_eviction_below_threshold(self):
        """Active memory below 0.9 × cap → no eviction. The threshold
        is wide enough that one large prefill on a half-empty cache
        does not trigger a thrash loop."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        sched.prefix_cache.store_cache([1, 2, 3], ["state-1"])
        sched.prefix_cache.store_cache([4, 5, 6], ["state-2"])
        # 80 GB active = 80% of cap, below 0.9 threshold
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=80 * 10**9),
        ):
            n = sched.evict_prefix_cache_under_pressure()
        assert n == 0
        assert sched.num_prefix_cache_pressure_evictions == 0
        assert len(sched.prefix_cache) == 2

    def test_pressure_evicts_lru_entries(self):
        """When active > threshold AND pressure persists across all
        evictions, the helper drains the cache until ``max_evict``.
        This pins the core D-METAL-PFX recovery loop: one 32k prefill
        had been pinning 7.7 GB of cache entries through to end of
        session — now they get evicted under pressure."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        for i in range(4):
            sched.prefix_cache.store_cache(
                [i * 10 + j for j in range(3)], [f"state-{i}"]
            )
        assert len(sched.prefix_cache) == 4

        # Simulate persistent pressure — active never drops below the
        # threshold, so the loop drains the cache up to ``max_evict``.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=95 * 10**9),
        ):
            # bound the loop low so the test runs fast — production
            # default is 64 per tick.
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 4, f"expected to evict all 4 entries, got {n}"
        assert len(sched.prefix_cache) == 0
        assert sched.num_prefix_cache_pressure_evictions == 4

    def test_pressure_eviction_stops_when_pressure_drops(self):
        """As soon as the simulated allocator returns below threshold,
        eviction stops — we want the *minimum* slabs returned, not the
        whole cache flushed on every spike (regression-safe for
        hit-rate)."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        for i in range(4):
            sched.prefix_cache.store_cache(
                [i * 10 + j for j in range(3)], [f"state-{i}"]
            )

        # Active drops below threshold after the second eviction.
        active_seq = iter([95 * 10**9, 95 * 10**9, 70 * 10**9, 70 * 10**9])
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched,
                "_current_metal_active_bytes",
                side_effect=lambda: next(active_seq),
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 2, f"expected to stop after 2 evictions, got {n}"
        assert len(sched.prefix_cache) == 2

    def test_max_evict_bound_respected(self):
        """``max_evict`` caps the per-tick eviction count so one
        pressure spike can't trash the whole cache."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        for i in range(8):
            sched.prefix_cache.store_cache(
                [i * 10 + j for j in range(3)], [f"state-{i}"]
            )
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=3)
        assert n == 3, f"max_evict=3 must cap evictions, got {n}"
        assert len(sched.prefix_cache) == 5

    def test_clear_cache_called_after_each_eviction(self):
        """Pre-fix bug: the cache trie released its CacheEntry but the
        underlying MLX allocator still held the slab in its free pool,
        so ``mx.get_active_memory`` did not drop on the next tick.
        ``mx.clear_cache`` MUST run between evictions to force the
        allocator to actually return slabs."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        for i in range(3):
            sched.prefix_cache.store_cache(
                [i * 10 + j for j in range(3)], [f"state-{i}"]
            )
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            patch("mlx.core.clear_cache") as mock_clear,
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 3
        # Should have called clear_cache once per eviction.
        assert mock_clear.call_count == 3, (
            f"expected 3 clear_cache calls, got {mock_clear.call_count}"
        )


class TestMemoryAwareCacheEviction:
    """Pressure-driven eviction on the ``MemoryAwarePrefixCache``."""

    def test_pressure_evicts_memory_aware_entries(self):
        """The OrderedDict-backed memory-aware cache must also respond
        to the pressure trigger. Dispatch goes through ``_evict_lru``
        under the cache's lock (matching its own LRU-on-capacity
        path)."""
        sched = _make_scheduler_with_memory_aware_cache(gpu_memory_utilization=0.5)
        # The MemoryAwarePrefixCache stores by token tuple; we feed it
        # minimal placeholder caches that satisfy its storage path.
        mac = sched.memory_aware_cache
        assert mac is not None
        # Use small numeric cache stand-ins; the eviction helper only
        # cares about ``_entries`` membership, not cache contents.
        mac._entries[(1, 2, 3)] = MagicMock(memory_bytes=1024)
        mac._entries[(4, 5, 6)] = MagicMock(memory_bytes=1024)
        mac._sorted_keys = [(1, 2, 3), (4, 5, 6)]
        mac._current_memory = 2048

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 2
        assert len(mac._entries) == 0
        assert sched.num_prefix_cache_pressure_evictions == 2


class TestPressureEvictionMetric:
    """The Prometheus exporter renders
    ``rapid_mlx_prefix_cache_pressure_evictions_total`` — pin the dict
    key and the rendered series."""

    def test_get_stats_exposes_counter(self):
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        stats = sched.get_stats()
        assert "num_prefix_cache_pressure_evictions" in stats
        assert stats["num_prefix_cache_pressure_evictions"] == 0

    def test_metric_renders_after_pressure_evictions(self):
        """After pressure-driven evictions, the Prometheus series must
        reflect the new count — operators alert on rate() of this
        series when sustained pressure indicates undersized
        ``--gpu-memory-utilization`` for the workload."""
        import types

        from vllm_mlx.routes.metrics import _render_prometheus

        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        for i in range(3):
            sched.prefix_cache.store_cache(
                [i * 10 + j for j in range(3)], [f"state-{i}"]
            )
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=10)
        cfg = types.SimpleNamespace(
            model_name="test",
            engine=types.SimpleNamespace(get_stats=sched.get_stats),
        )
        body = _render_prometheus(cfg)
        assert "rapid_mlx_prefix_cache_pressure_evictions_total 3" in body
        assert "# TYPE rapid_mlx_prefix_cache_pressure_evictions_total counter" in body


class TestEngineCoreInvokesPressureEvict:
    """Codex round 1 BLOCKING + round 2 BLOCKING #2 regression —
    engine_core's memory-pressure tick must call
    ``evict_prefix_cache_under_pressure`` UNCONDITIONALLY (NOT only
    when ``active_mem`` exceeds the legacy
    ``_memory_pressure_threshold``), and a scheduler eviction failure
    MUST surface a rate-limited ``logger.warning`` in the engine log.

    Both behaviours are pinned by driving
    ``EngineCore._run_pressure_evict_tick`` directly with a stub
    scheduler. Round 1 used source-text scans; round 2 BLOCKING #2
    asked for actual behavioural assertions, which this rewrite
    provides.
    """

    def _build_minimal_engine_core(self, scheduler_stub):
        """Construct an ``EngineCore`` shell that bypasses the model-
        registry / executor setup so the tick helper can be driven
        without booting a real MLX engine. Only attributes touched by
        ``_run_pressure_evict_tick`` are populated."""
        from vllm_mlx.engine_core import EngineCore

        ec = EngineCore.__new__(EngineCore)
        ec.scheduler = scheduler_stub
        # The helper checks ``_pressure_evict_error_logged`` via
        # ``getattr(..., default=False)``, so we don't need to set
        # it up-front; it materializes on first failure.
        return ec

    def test_pressure_tick_calls_scheduler_regardless_of_legacy_threshold(self):
        """Codex round 1 BLOCKING regression: the helper must invoke
        ``scheduler.evict_prefix_cache_under_pressure`` on every
        call, irrespective of any external pressure-threshold gate.
        Behavioural pin — stubs the scheduler and counts calls."""
        scheduler = MagicMock()
        scheduler.evict_prefix_cache_under_pressure = MagicMock(return_value=0)
        scheduler.release_paged_cache_blocks_under_pressure = MagicMock(return_value=0)
        ec = self._build_minimal_engine_core(scheduler)

        ec._run_pressure_evict_tick()
        ec._run_pressure_evict_tick()
        ec._run_pressure_evict_tick()

        assert scheduler.evict_prefix_cache_under_pressure.call_count == 3, (
            "engine_core must call the scheduler eviction tick once "
            "per invocation, with no internal gating — see codex "
            "round 1 BLOCKING on PR #797."
        )
        assert scheduler.release_paged_cache_blocks_under_pressure.call_count == 3

    def test_pressure_tick_logs_warning_once_on_eviction_failure(self, caplog):
        """Codex round 2 BLOCKING #2 regression: when the scheduler
        eviction call raises, the engine MUST emit a single WARNING
        and rate-limit subsequent failures so a persistent broken
        cache variant doesn't flood the engine log at 16-step
        cadence. Behavioural pin — stubs the scheduler to raise and
        asserts exactly one D-METAL-PFX WARNING across multiple
        calls."""
        import logging

        scheduler = MagicMock()
        boom = RuntimeError("simulated cache eviction failure")
        scheduler.evict_prefix_cache_under_pressure = MagicMock(side_effect=boom)
        scheduler.release_paged_cache_blocks_under_pressure = MagicMock(return_value=0)
        ec = self._build_minimal_engine_core(scheduler)

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.engine_core"):
            for _ in range(5):
                # Helper must NOT re-raise — engine loop continues.
                ec._run_pressure_evict_tick()

        # Scheduler was hit every tick.
        assert scheduler.evict_prefix_cache_under_pressure.call_count == 5

        d_metal_warnings = [
            r for r in caplog.records if "D-METAL-PFX" in r.getMessage()
        ]
        assert len(d_metal_warnings) == 1, (
            f"expected exactly 1 D-METAL-PFX WARNING across 5 failing "
            f"ticks (rate-limit sentinel), got {len(d_metal_warnings)}: "
            f"{[r.getMessage() for r in d_metal_warnings]}"
        )
        # The warning MUST surface the underlying exception repr so
        # operators can correlate the stalled D-METAL-PFX series with
        # the root cause without enabling debug logging.
        assert "simulated cache eviction failure" in d_metal_warnings[0].getMessage()
        # Sentinel was set so further ticks stay silent.
        assert ec._pressure_evict_error_logged is True

    def test_pressure_tick_silent_on_success(self, caplog):
        """A clean eviction call must NOT emit any WARNING — the rate-
        limit warning is reserved for failures, not for every tick."""
        import logging

        scheduler = MagicMock()
        scheduler.evict_prefix_cache_under_pressure = MagicMock(return_value=0)
        scheduler.release_paged_cache_blocks_under_pressure = MagicMock(return_value=0)
        ec = self._build_minimal_engine_core(scheduler)

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.engine_core"):
            for _ in range(3):
                ec._run_pressure_evict_tick()

        d_metal_warnings = [
            r for r in caplog.records if "D-METAL-PFX" in r.getMessage()
        ]
        assert d_metal_warnings == [], (
            f"clean ticks must not log D-METAL-PFX warnings, got "
            f"{[r.getMessage() for r in d_metal_warnings]}"
        )


class TestClearCacheFailurePropagation:
    """Codex round 3 BLOCKING #2 regression — when ``mx.clear_cache``
    raises during pressure-driven eviction, the slabs have NOT actually
    been returned to MLX's free pool, so the eviction MUST NOT be
    counted as successful. Pre-fix, ``clear_cache`` failures were
    swallowed and the loop kept incrementing
    ``num_prefix_cache_pressure_evictions``, lying to operators about
    a recovery that didn't happen."""

    def test_clear_cache_failure_propagates_to_engine_path(self):
        """``mx.clear_cache`` raising → exception bubbles out of
        ``evict_prefix_cache_under_pressure`` so the engine_core
        rate-limited warning surfaces the underlying MLX failure."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        sched.prefix_cache.store_cache([1, 2, 3], ["state-1"])
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            patch("mlx.core.clear_cache", side_effect=RuntimeError("metal broken")),
            pytest.raises(RuntimeError, match="metal broken"),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=10)

    def test_counter_reflects_cache_mutation_on_clear_cache_failure(self):
        """Codex round 4 BLOCKING #3 fix: when ``mx.clear_cache``
        raises AFTER the trie has already dropped the entry, the
        counter MUST tick by the number of entries actually removed
        from the cache. Pre-fix attempts bumped the counter only
        after ``clear_cache`` succeeded, but that left cache state
        and metrics in disagreement on the failure path: the trie
        had already mutated (entry removed) but the metric stayed at
        zero, so operators saw ``rapid_mlx_prefix_cache_pressure_
        evictions_total == 0`` while ``len(prefix_cache)`` showed
        the entry was actually gone.

        The codex-round-3 propagation invariant (clear_cache
        failures must reach the engine_core warning path) still
        holds — the exception bubbles out so the warning fires —
        but the counter accurately reflects ground truth."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        sched.prefix_cache.store_cache([1, 2, 3], ["state-1"])
        sched.prefix_cache.store_cache([4, 5, 6], ["state-2"])
        before = sched.num_prefix_cache_pressure_evictions
        before_len = len(sched.prefix_cache)
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            patch("mlx.core.clear_cache", side_effect=RuntimeError("metal broken")),
            pytest.raises(RuntimeError),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=10)
        # The first eviction removed one entry from the trie BEFORE
        # clear_cache raised — counter must reflect that.
        after_len = len(sched.prefix_cache)
        removed = before_len - after_len
        assert removed == 1, (
            f"clear_cache failure should stop loop after 1 entry; "
            f"trie went from {before_len} to {after_len} entries"
        )
        assert sched.num_prefix_cache_pressure_evictions == before + removed, (
            f"counter must match cache mutation: expected "
            f"{before + removed}, got {sched.num_prefix_cache_pressure_evictions}"
        )


class TestSchedulerPropagatesEvictionErrors:
    """Codex round 2 BLOCKING #1 regression — a failing
    ``_evict_one_prefix_cache_entry`` MUST propagate to
    ``evict_prefix_cache_under_pressure`` (and from there to
    engine_core's rate-limited warning). Pre-fix the inner method
    swallowed every exception and returned ``False``, making a
    broken cache variant indistinguishable from "nothing eligible"
    so the engine warning never fired."""

    def test_memory_aware_cache_eviction_error_propagates(self):
        """``MemoryAwarePrefixCache._evict_lru`` raising under
        coordinated eviction must propagate up — not be silently
        squashed."""
        sched = _make_scheduler_with_memory_aware_cache(gpu_memory_utilization=0.5)
        mac = sched.memory_aware_cache
        assert mac is not None
        # Populate at least one entry so the early-return guard
        # (empty ``_entries``) does not short-circuit before
        # ``_evict_lru``.
        mac._entries[(1, 2, 3)] = MagicMock(memory_bytes=1024)
        mac._sorted_keys = [(1, 2, 3)]
        mac._current_memory = 1024

        with (
            patch.object(mac, "_evict_lru", side_effect=RuntimeError("eviction broke")),
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            pytest.raises(RuntimeError, match="eviction broke"),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=10)

    def test_legacy_prefix_cache_eviction_error_propagates(self):
        """``PrefixCacheManager._evict_lru`` raising must propagate up."""
        sched = _make_scheduler_with_legacy_cache(gpu_memory_utilization=0.5)
        sched.prefix_cache.store_cache([1, 2, 3], ["state-1"])

        with (
            patch.object(
                sched.prefix_cache,
                "_evict_lru",
                side_effect=RuntimeError("trie eviction broke"),
            ),
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
            pytest.raises(RuntimeError, match="trie eviction broke"),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=10)


class TestPressureEvictFractionClamp:
    """Codex round 2 NIT regression — ``metal_pressure_evict_fraction``
    must be clamped into a documented ``(0, 1]`` range so a zero or
    out-of-range configuration cannot subtly break the recovery loop."""

    def test_zero_fraction_clamped_to_default(self):
        """A zero / negative fraction must not evict on every tick —
        clamp to the safe default (0.9)."""
        config = SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            prefix_cache_size=4,
            gpu_memory_utilization=0.5,
            metal_pressure_evict_fraction=0.0,  # invalid
        )
        sched = Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=config,
        )
        sched.prefix_cache.store_cache([1, 2, 3], ["state-1"])
        # Active = 80% of cap. With the default-clamped fraction (0.9),
        # 80% < 90%, so no eviction. If the clamp were missing, a
        # threshold of 0 would mean "evict on every tick".
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=80 * 10**9),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 0
        assert len(sched.prefix_cache) == 1

    def test_above_one_fraction_clamped_to_one(self):
        """A fraction > 1.0 would push the threshold above the cap
        itself, making pressure eviction never run before admission
        starts rejecting. Clamp to 1.0 (= the cap) so eviction at
        least kicks in right before admission rejects."""
        config = SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=False,
            use_paged_cache=False,
            prefix_cache_size=4,
            gpu_memory_utilization=0.5,
            metal_pressure_evict_fraction=2.5,  # invalid
        )
        sched = Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=config,
        )
        sched.prefix_cache.store_cache([1, 2, 3], ["state-1"])
        # Active exactly = cap → with clamp-to-1.0, threshold is the cap
        # itself, so active < threshold is False and eviction triggers.
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=100 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        # Eviction triggered → entry removed.
        assert n == 1
        assert len(sched.prefix_cache) == 0


class TestMetalCacheMemoryMetric:
    """D-METAL-CACHE-ZERO regression: the
    ``rapid_mlx_metal_cache_memory_bytes`` series must reflect MLX's
    actual reported cache memory — pre-fix users saw 0 across a
    long-prefill session, but that was a real allocator state (cache
    cleared every step end), not a wiring bug. Pin the wiring so a
    refactor doesn't break the contract."""

    def test_get_stats_reads_live_cache_memory(self):
        """``scheduler.get_stats`` must read live from
        ``mx.get_cache_memory`` — not hard-code 0 or stale-snapshot."""
        config = SchedulerConfig(enable_prefix_cache=False)
        sched = Scheduler(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=config,
        )
        # Force a non-zero cache-memory reading via a stub. The contract
        # we pin is: get_stats must consult mx.get_cache_memory each
        # call, not a cached value.
        with (
            patch("mlx.core.metal.is_available", return_value=True),
            patch("mlx.core.get_active_memory", return_value=1 * 10**9),
            patch("mlx.core.get_peak_memory", return_value=2 * 10**9),
            patch("mlx.core.get_cache_memory", return_value=5 * 10**9),
        ):
            stats = sched.get_stats()
        assert stats.get("metal_cache_memory_gb") == pytest.approx(5.0, abs=0.01)
        assert stats.get("metal_active_memory_gb") == pytest.approx(1.0, abs=0.01)


class TestBlockAwareCacheEviction:
    """Pressure-driven eviction on the paged ``BlockAwarePrefixCache``.

    Regression: pre-fix this variant was a deliberate no-op in
    ``_evict_one_prefix_cache_entry`` ("blocks are released by
    PagedCacheManager ref-counts"), but completed requests never call
    ``release_cache`` — their blocks keep ref_count=1 and never re-enter
    the free queue, so every release path that scans the free queue
    (``release_pressure_blocks`` / ``evict_lru_blocks``) starves. Under
    Metal pressure the admission gate then rejects ALL new requests while
    ``active`` stays pinned above cap — a permanent 503 wedge. These tests
    pin the fix: pressure evicts the LRU prefix-index entry and drops its
    resident block tensors so active memory falls back under cap.
    """

    def _make_paged_scheduler(self, gpu_memory_utilization: float = 0.5):
        config = SchedulerConfig(
            max_num_seqs=8,
            max_concurrent_requests=64,
            enable_prefix_cache=True,
            use_memory_aware_cache=False,
            use_paged_cache=True,
            paged_cache_block_size=4,
            max_cache_blocks=256,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        tokenizer = MagicMock()
        tokenizer.encode = lambda s: list(range(len(s)))
        model = MagicMock()
        return Scheduler(model=model, tokenizer=tokenizer, config=config)

    def test_pressure_evicts_block_aware_entries(self):
        """Pressure must evict LRU prefix-index entries and drop the
        resident KV tensors of their blocks (the pre-fix wedge)."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        assert bac is not None
        paged = bac.paged_cache

        # Manually populate the prefix index + resident block tensors the
        # way store_cache does: hash entry -> (tokens, block_ids) with
        # blocks that carry cache_data (resident Metal memory).
        blocks = []
        for i in range(4):
            blk = paged.allocate_block()
            blk.cache_data = [MagicMock()]  # resident tensor
            blk.cache_class_name = "cache"
            blocks.append(blk)
        bac._prefix_index["h1"] = ([1, 2, 3, 4], [b.block_id for b in blocks[:2]])
        bac._prefix_index["h2"] = ([5, 6, 7, 8], [b.block_id for b in blocks[2:]])

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        # Pressure stays high, so BOTH entries are evicted (max_evict=10
        # is not the binding constraint; the empty index is). Each entry's
        # resident block tensors must be dropped.
        assert n == 2
        assert "h1" not in bac._prefix_index
        assert "h2" not in bac._prefix_index
        assert all(b.cache_data is None for b in blocks)
        assert sched.num_prefix_cache_pressure_evictions == 2

    def test_pressure_drains_until_index_empty(self):
        """Persistent pressure drains entries until the index is empty."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        blocks = []
        for i in range(4):
            blk = paged.allocate_block()
            blk.cache_data = [MagicMock()]
            blocks.append(blk)
        bac._prefix_index["h1"] = ([1, 2, 3, 4], [b.block_id for b in blocks[:2]])
        bac._prefix_index["h2"] = ([5, 6, 7, 8], [b.block_id for b in blocks[2:]])

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 2  # both entries evicted
        assert len(bac._prefix_index) == 0
        assert all(b.cache_data is None for b in blocks)
        assert sched.num_prefix_cache_pressure_evictions == 2

    def test_no_eviction_when_index_empty(self):
        """Empty prefix index -> no-op (no crash, no counter tick)."""
        sched = self._make_paged_scheduler()
        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        assert n == 0
        assert sched.num_prefix_cache_pressure_evictions == 0

    def test_pinned_blocks_not_evicted(self):
        """Pinned blocks (system prompt) must survive pressure eviction."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        blocks = []
        for i in range(4):
            blk = paged.allocate_block()
            blk.cache_data = [MagicMock()]
            blocks.append(blk)
        # Pin the first block of entry h1 — its tensor must survive.
        blocks[0].is_pinned = True
        bac._prefix_index["h1"] = ([1, 2, 3, 4], [b.block_id for b in blocks[:2]])
        bac._prefix_index["h2"] = ([5, 6, 7, 8], [b.block_id for b in blocks[2:]])

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)
        # Both entries are evicted under sustained pressure, but the
        # pinned block (h1's first block) keeps its tensor — only
        # non-pinned resident tensors are released.
        assert n == 2
        assert blocks[0].cache_data is not None  # pinned survives
        assert blocks[1].cache_data is None
        assert blocks[2].cache_data is None
        assert blocks[3].cache_data is None
        assert sched.num_prefix_cache_pressure_evictions == 2

    def test_pressure_releases_full_kv_request_table_entries(self):
        """Pressure must drop the FULL-KV tensor held by ``_request_tables``
        entries, not just the block-index views.

        In production the block slices are views into the request-table
        entry's buffer, so clearing only the views leaves the underlying
        Metal allocation alive (the pre-fix wedge: ``evictions_total``
        grew while ``active`` stayed pinned above cap). These are plain
        ``MagicMock`` stand-ins — they cannot model the tensor-level
        aliasing itself — so this test asserts the observable behaviour
        the fix guarantees: every request-table entry is popped AND every
        referenced block has its ``cache_data`` cleared, which is exactly
        the pair of drops needed for the real allocation to fall.
        """
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache

        # Simulate two completed requests whose store_cache left both a
        # prefix-index entry AND a full-KV request-table entry resident.
        blocks = []
        for i in range(4):
            blk = paged.allocate_block()
            blk.cache_data = [MagicMock()]  # resident view
            blocks.append(blk)
        bac._prefix_index["h1"] = ([1, 2, 3, 4], [b.block_id for b in blocks[:2]])
        bac._request_tables["req-1"] = BlockCacheEntry(
            block_table=MagicMock(block_ids=[b.block_id for b in blocks[:2]]),
            cache_data=[MagicMock()],  # full KV tensor
            last_access=1.0,
        )
        bac._request_tables["req-2"] = BlockCacheEntry(
            block_table=MagicMock(block_ids=[b.block_id for b in blocks[2:]]),
            cache_data=[MagicMock()],
            last_access=2.0,
        )

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=10)

        # One eviction per full-KV request-table entry (the resident
        # allocation owner). The LRU index entry is drained on the same
        # call, so a second tick releases the second request table.
        assert n == 2
        # Request-table entries must have been popped (their full-KV
        # tensors released) — the actual wedge fix. Under sustained
        # pressure both are drained, even though the index only had one
        # entry (the rt-first ordering guarantees this).
        assert "req-1" not in bac._request_tables
        assert "req-2" not in bac._request_tables
        # Every referenced block's cache_data must also be cleared — in
        # production these views alias the dropped request-table buffers,
        # so leaving them set would keep the Metal allocation alive.
        assert all(b.cache_data is None for b in blocks)
        assert sched.num_prefix_cache_pressure_evictions == 2

    def test_request_table_entry_with_pinned_block_survives(self):
        """A request-table entry may be evicted for its own full-KV, but a
        PINNED block it references (system prompt) must keep its tensor —
        pressure must reclaim the request's own buffer + private blocks
        while leaving the pinned block's KV live.

        (A blanket 'skip the whole entry' would be wrong: the system-prompt
        block is shared into nearly every request, so skipping any entry
        that touches it would reclaim almost nothing and re-open the wedge.)
        """
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        blocks = []
        for i in range(2):
            blk = paged.allocate_block()
            blk.cache_data = [MagicMock()]
            blk.ref_count = 1
            blocks.append(blk)
        blocks[0].is_pinned = True  # system-prompt block, referenced by req-1
        bac._request_tables["req-1"] = BlockCacheEntry(
            block_table=MagicMock(block_ids=[b.block_id for b in blocks]),
            cache_data=[MagicMock()],
            last_access=1.0,
        )

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            n = sched.evict_prefix_cache_under_pressure(max_evict=1)

        # The entry is evicted (its own full-KV buffer reclaimed) and its
        # private block cleared, but the pinned block keeps its KV.
        assert "req-1" not in bac._request_tables
        assert blocks[0].cache_data is not None  # pinned survives
        assert blocks[1].cache_data is None  # private block reclaimed
        assert n == 1

    def test_shared_block_not_cleared_under_pressure(self):
        """A block still referenced by a live request (ref_count > 1) must
        NOT have its KV cleared during pressure eviction — doing so would
        corrupt the active request that still holds it."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        shared = paged.allocate_block()
        shared.cache_data = [MagicMock()]
        shared.ref_count = 2  # cached prefix + one live request
        private = paged.allocate_block()
        private.cache_data = [MagicMock()]
        private.ref_count = 1
        bac._request_tables["req-1"] = BlockCacheEntry(
            block_table=MagicMock(block_ids=[shared.block_id, private.block_id]),
            cache_data=[MagicMock()],
            last_access=1.0,
        )

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=1)

        assert shared.cache_data is not None  # live KV preserved
        assert private.cache_data is None  # unshared block reclaimed

    def test_index_eviction_returns_blocks_to_free_queue(self):
        """Index-path eviction must not just clear tensors — it must also
        return the block slots to the free queue (decrement ref_count),
        or the pool leaks and eventually exhausts max_cache_blocks."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        blk = paged.allocate_block()  # ref_count == 1
        blk.cache_data = [MagicMock()]
        bid = blk.block_id
        assert bid in paged.allocated_blocks
        free_before = paged.free_block_queue.num_free_blocks
        bac._prefix_index["h1"] = ([1, 2, 3, 4], [bid])

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            sched.evict_prefix_cache_under_pressure(max_evict=1)

        # Tensor cleared AND the slot returned to the pool.
        assert blk.cache_data is None
        assert bid not in paged.allocated_blocks
        assert paged.free_block_queue.num_free_blocks == free_before + 1

    def test_fetch_guard_rejects_reallocated_block(self):
        """A prefix-index hit whose block was freed and REALLOCATED for
        different tokens (fresh cache_data, changed hash) must be rejected
        by the fetch-side guard — cache_data != None is not enough."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        bs = paged.block_size  # 4 in this fixture
        tokens = list(range(bs))  # exactly one full block

        blk = paged.allocate_block()
        blk.cache_data = [MagicMock()]
        # store_cache registers a full block as hash_value = hash(its tokens);
        # index is keyed by the prefix hash the fetch path recomputes.
        blk.hash_value = paged.compute_block_hash(tokens)
        bac._prefix_index[paged.compute_block_hash(tokens)] = (tokens, [blk.block_id])

        # Correct owner + resident tensor -> live hit.
        assert bac._find_best_prefix_match(tokens) is not None

        # Reallocation: same block_id, still has fresh cache_data, but now
        # holds a DIFFERENT block's content (hash for other tokens). The
        # stale index entry still points here, yet the content no longer
        # matches — the ownership check must turn this into a miss.
        blk.hash_value = paged.compute_block_hash([99, 98, 97, 96])
        assert bac._find_best_prefix_match(tokens) is None

    def test_pressure_eviction_does_not_free_reallocated_block(self):
        """A stale index must never evict a slot now owned by another request."""
        sched = self._make_paged_scheduler()
        bac = sched.block_aware_cache
        paged = bac.paged_cache
        tokens = [1, 2, 3, 4]
        blk = paged.allocate_block()
        blk.cache_data = [MagicMock()]
        bac._prefix_index[paged.compute_block_hash(tokens)] = (tokens, [blk.block_id])

        # Simulate the same physical slot being reused for unrelated live KV.
        other_tokens = [9, 8, 7, 6]
        blk.hash_value = paged.compute_block_hash(other_tokens)
        original_data = blk.cache_data

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(
                sched, "_current_metal_active_bytes", return_value=200 * 10**9
            ),
        ):
            assert sched.evict_prefix_cache_under_pressure(max_evict=1) == 1

        assert blk.cache_data is original_data
        assert blk.block_id in paged.allocated_blocks
        assert blk.ref_count == 1
        assert not bac._prefix_index

    def test_free_block_sweep_is_gated_by_metal_pressure(self):
        sched = self._make_paged_scheduler()
        paged = sched.paged_cache_manager
        free = paged.free_block_queue.fake_head.next_free_block
        free.cache_data = [MagicMock()]

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=80 * 10**9),
        ):
            assert sched.release_paged_cache_blocks_under_pressure() == 0
        assert free.cache_data is not None

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=100 * 10**9),
            patch.object(sched, "_current_metal_active_bytes", return_value=95 * 10**9),
            patch("mlx.core.clear_cache"),
        ):
            assert sched.release_paged_cache_blocks_under_pressure() == 1
        assert free.cache_data is None

    def test_free_block_sweep_uses_device_limit_without_explicit_cap(self):
        sched = self._make_paged_scheduler(gpu_memory_utilization=0.0)
        paged = sched.paged_cache_manager
        free = paged.free_block_queue.fake_head.next_free_block
        free.cache_data = [MagicMock()]

        with (
            patch.object(sched, "_resolve_metal_cap_bytes", return_value=0),
            patch.object(sched, "_current_metal_active_bytes", return_value=95),
            patch("vllm_mlx.scheduler.mx.metal.is_available", return_value=True),
            patch(
                "vllm_mlx.scheduler.mx.device_info",
                return_value={"max_recommended_working_set_size": 100},
            ),
            patch("mlx.core.clear_cache"),
        ):
            assert sched.release_paged_cache_blocks_under_pressure() == 1
        assert free.cache_data is None
