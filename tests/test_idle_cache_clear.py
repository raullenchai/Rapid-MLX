"""Focused tests for idle prefix-cache release configuration and behavior."""

from types import SimpleNamespace
from unittest.mock import patch

from vllm_mlx.engine_core import EngineCore
from vllm_mlx.memory_cache import (
    CacheStats,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)
from vllm_mlx.scheduler import SchedulerConfig


def test_idle_cache_clear_seconds_is_opt_in_and_validated():
    assert SchedulerConfig().idle_cache_clear_seconds is None
    assert SchedulerConfig(idle_cache_clear_seconds=600).idle_cache_clear_seconds == 600

    try:
        SchedulerConfig(idle_cache_clear_seconds=-1)
    except ValueError as exc:
        assert "idle_cache_clear_seconds" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("negative idle-cache interval was accepted")


def test_idle_clear_preserves_cumulative_cache_counters():
    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(max_memory_mb=1, max_entries=10),
    )
    cache._stats = CacheStats(
        hits=7,
        misses=3,
        evictions=2,
        tokens_saved=123,
        max_memory_bytes=cache._max_memory,
    )

    cache.clear(reset_stats=False)

    stats = cache.get_stats()
    assert stats["hits"] == 7
    assert stats["misses"] == 3
    assert stats["tokens_saved"] == 123
    assert stats["current_memory_bytes"] == 0
    assert stats["entry_count"] == 0


def test_engine_idle_clear_runs_scheduler_clear_on_worker():
    calls: list[bool] = []
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(
        clear_prefix_cache=lambda *, reset_stats: calls.append(reset_stats) or True
    )

    with patch("vllm_mlx.engine_core.mx.clear_cache") as clear_cache:
        assert engine._clear_prefix_cache_on_worker() is True

    assert calls == [False]
    clear_cache.assert_called_once_with()
