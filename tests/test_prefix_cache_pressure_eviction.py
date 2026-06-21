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
