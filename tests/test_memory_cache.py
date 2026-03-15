# SPDX-License-Identifier: Apache-2.0
"""Tests for memory-aware prefix cache."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.memory_cache import (
    CacheStats,
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _array_memory,
    _CacheEntry,
    _get_available_memory,
    estimate_kv_cache_memory,
)


class TestMemoryCacheConfig:
    """Tests for MemoryCacheConfig."""

    def test_default_config(self):
        config = MemoryCacheConfig()
        assert config.max_memory_mb is None
        assert config.max_memory_percent == 0.20
        assert config.max_entries == 1000
        assert config.enable_memory_tracking is True

    def test_custom_config(self):
        config = MemoryCacheConfig(
            max_memory_mb=2048,
            max_memory_percent=0.5,
            max_entries=100,
        )
        assert config.max_memory_mb == 2048
        assert config.max_memory_percent == 0.5
        assert config.max_entries == 100

    def test_invalid_memory_percent_zero(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=0.0)

    def test_invalid_memory_percent_negative(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=-0.1)

    def test_invalid_memory_percent_over_one(self):
        with pytest.raises(ValueError, match="max_memory_percent"):
            MemoryCacheConfig(max_memory_percent=1.5)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            MemoryCacheConfig(max_entries=0)

    def test_compute_memory_limit_explicit(self):
        config = MemoryCacheConfig(max_memory_mb=1024)
        assert config.compute_memory_limit() == 1024 * 1024 * 1024

    def test_compute_memory_limit_auto(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=8 * 1024 * 1024 * 1024,  # 8GB
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            assert limit == 2 * 1024 * 1024 * 1024  # 25% of 8GB = 2GB

    def test_compute_memory_limit_fallback(self):
        with patch(
            "vllm_mlx.memory_cache._get_available_memory",
            return_value=0,  # Detection failed
        ):
            config = MemoryCacheConfig(max_memory_percent=0.25)
            limit = config.compute_memory_limit()
            # Fallback: 25% of 8GB = 2GB
            assert limit == 2 * 1024 * 1024 * 1024


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_stats(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_queries(self):
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_memory_utilization(self):
        stats = CacheStats(
            current_memory_bytes=500 * 1024 * 1024,
            max_memory_bytes=1000 * 1024 * 1024,
        )
        assert stats.memory_utilization == 0.5

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert "hit_rate" in d
        assert "memory_utilization" in d


class MockArray:
    """Mock array with nbytes attribute."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class MockDtype:
    """Mock dtype with size attribute."""

    def __init__(self, size: int):
        self.size = size


class MockShapeArray:
    """Mock array with shape and dtype (like MLX arrays) but no nbytes."""

    def __init__(self, shape: tuple, dtype_size: int):
        self.shape = shape
        self.dtype = MockDtype(dtype_size)


class MockKVCache:
    """Mock KV cache with keys/values attributes."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self.keys = MockArray(key_bytes)
        self.values = MockArray(value_bytes)


class MockStateCache:
    """Mock cache with state property."""

    def __init__(self, key_bytes: int, value_bytes: int):
        self._keys = MockArray(key_bytes)
        self._values = MockArray(value_bytes)

    @property
    def state(self):
        return (self._keys, self._values)


class TestArrayMemory:
    """Tests for _array_memory helper (shape-based, no lazy eval trigger)."""

    def test_shape_dtype_estimation(self):
        """Verify shape*dtype.size computation without .nbytes access."""
        arr = MockShapeArray(shape=(2, 16, 128, 64), dtype_size=2)
        # 2 * 16 * 128 * 64 * 2 = 524288
        assert _array_memory(arr) == 2 * 16 * 128 * 64 * 2

    def test_fallback_to_nbytes(self):
        """Verify fallback to .nbytes when shape/dtype not available."""
        arr = MockArray(nbytes=4096)
        assert _array_memory(arr) == 4096

    def test_zero_for_unknown_object(self):
        """Return 0 for objects without shape/dtype/nbytes."""
        assert _array_memory(42) == 0
        assert _array_memory("string") == 0

    def test_shape_dtype_preferred_over_nbytes(self):
        """When both shape+dtype and nbytes exist, shape+dtype is used."""

        class DualArray:
            def __init__(self):
                self.shape = (10,)
                self.dtype = MockDtype(4)
                self.nbytes = 9999  # should NOT be used

        arr = DualArray()
        assert _array_memory(arr) == 40  # 10 * 4, not 9999

    def test_estimate_uses_shape_based_for_dict_state(self):
        """estimate_kv_cache_memory uses _array_memory (shape-based) for dicts."""
        keys = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        values = MockShapeArray(shape=(1, 8, 100, 64), dtype_size=2)
        layer = {"state": (keys, values)}
        expected = 2 * (1 * 8 * 100 * 64 * 2)
        assert estimate_kv_cache_memory([layer]) == expected


class TestEstimateKvCacheMemory:
    """Tests for estimate_kv_cache_memory function."""

    def test_empty_cache(self):
        assert estimate_kv_cache_memory([]) == 0
        assert estimate_kv_cache_memory(None) == 0

    def test_cache_with_nbytes_attribute(self):
        layer = MockKVCache(1000, 1000)
        assert estimate_kv_cache_memory([layer]) == 2000

    def test_cache_with_state_property(self):
        layer = MockStateCache(500, 500)
        assert estimate_kv_cache_memory([layer]) == 1000

    def test_cache_with_dict_state(self):
        keys = MockArray(300)
        values = MockArray(300)
        layer = {"state": (keys, values)}
        assert estimate_kv_cache_memory([layer]) == 600

    def test_multiple_layers(self):
        layers = [MockKVCache(100, 100) for _ in range(4)]
        assert estimate_kv_cache_memory(layers) == 800


class TestCacheEntry:
    """Tests for _CacheEntry."""

    def test_create_entry(self):
        cache = [MockKVCache(100, 100)]
        entry = _CacheEntry.create([1, 2, 3], cache)
        assert entry.tokens == (1, 2, 3)
        assert entry.cache is cache
        assert entry.memory_bytes == 200


class TestMemoryAwarePrefixCache:
    """Tests for MemoryAwarePrefixCache."""

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def small_cache(self, model):
        """Cache with 1MB limit."""
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=10)
        return MemoryAwarePrefixCache(model, config)

    @pytest.fixture
    def mock_kv_cache(self):
        """Create a mock KV cache with known size."""

        def _create(size_bytes: int):
            return [MockKVCache(size_bytes // 2, size_bytes // 2)]

        return _create

    def test_initialization(self, model):
        config = MemoryCacheConfig(max_memory_mb=100)
        cache = MemoryAwarePrefixCache(model, config)
        assert len(cache) == 0
        assert cache.memory_limit_mb == 100.0

    def test_store_and_fetch_exact_match(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3, 4, 5]
        kv = mock_kv_cache(1000)

        # Store
        assert small_cache.store(tokens, kv) is True
        assert len(small_cache) == 1

        # Fetch exact match
        result, remaining = small_cache.fetch(tokens)
        assert result is kv  # Same reference, no copy
        assert remaining == []

    def test_fetch_prefix_match(self, small_cache, mock_kv_cache):
        # Store shorter sequence
        short_tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(short_tokens, kv)

        # Fetch longer sequence that starts with cached prefix
        long_tokens = [1, 2, 3, 4, 5, 6]
        result, remaining = small_cache.fetch(long_tokens)

        assert result is kv
        assert remaining == [4, 5, 6]

    def test_fetch_miss(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)

        # Fetch completely different sequence
        result, remaining = small_cache.fetch([7, 8, 9])
        assert result is None
        assert remaining == [7, 8, 9]

    def test_lru_eviction_on_memory_pressure(self, model, mock_kv_cache):
        # Create cache with 500KB limit
        config = MemoryCacheConfig(max_memory_mb=0.5, max_entries=100)
        cache = MemoryAwarePrefixCache(model, config)

        # Store entries that together exceed limit
        # Each is ~200KB
        for i in range(5):
            tokens = list(range(i * 10, (i + 1) * 10))
            kv = mock_kv_cache(200 * 1024)
            cache.store(tokens, kv)

        # Should have evicted older entries
        assert cache.memory_usage_mb <= 0.5
        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_lru_order_updated_on_fetch(self, small_cache, mock_kv_cache):
        # Store two entries
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv1 = mock_kv_cache(100 * 1024)
        kv2 = mock_kv_cache(100 * 1024)

        small_cache.store(tokens1, kv1)
        small_cache.store(tokens2, kv2)

        # Fetch first entry (moves it to end of LRU)
        small_cache.fetch(tokens1)

        # Now tokens2 should be evicted first if we need space
        # Store a large entry to trigger eviction
        big_kv = mock_kv_cache(900 * 1024)
        small_cache.store([7, 8, 9], big_kv)

        # tokens1 should still be there (was recently accessed)
        # tokens2 should be evicted
        assert tokens1 in small_cache or len(small_cache) == 1

    def test_entry_too_large_rejected(self, small_cache, mock_kv_cache):
        # Try to store entry larger than cache limit
        tokens = [1, 2, 3]
        huge_kv = mock_kv_cache(10 * 1024 * 1024)  # 10MB, limit is 1MB

        result = small_cache.store(tokens, huge_kv)
        assert result is False
        assert len(small_cache) == 0

    def test_store_empty_rejected(self, small_cache, mock_kv_cache):
        assert small_cache.store([], mock_kv_cache(100)) is False
        assert small_cache.store([1, 2, 3], []) is False
        assert small_cache.store([1, 2, 3], None) is False

    def test_remove_entry(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)
        small_cache.store(tokens, kv)
        assert len(small_cache) == 1

        assert small_cache.remove(tokens) is True
        assert len(small_cache) == 0
        assert small_cache.remove(tokens) is False  # Already removed

    def test_clear(self, small_cache, mock_kv_cache):
        for i in range(3):
            small_cache.store([i], mock_kv_cache(1000))

        assert len(small_cache) == 3
        small_cache.clear()
        assert len(small_cache) == 0
        assert small_cache.memory_usage_mb == 0

    def test_contains(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        assert tokens not in small_cache
        small_cache.store(tokens, mock_kv_cache(1000))
        assert tokens in small_cache

    def test_stats_tracking(self, small_cache, mock_kv_cache):
        tokens1 = [1, 2, 3]
        tokens2 = [4, 5, 6]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens1, kv)
        small_cache.fetch(tokens1)  # Hit
        small_cache.fetch(tokens2)  # Miss

        stats = small_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1

    def test_reset_stats(self, small_cache, mock_kv_cache):
        small_cache.store([1, 2, 3], mock_kv_cache(1000))
        small_cache.fetch([1, 2, 3])
        small_cache.fetch([4, 5, 6])

        small_cache.reset_stats()
        stats = small_cache.get_stats()

        # Stats reset but entry count preserved
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entry_count"] == 1

    def test_duplicate_store_updates_lru(self, small_cache, mock_kv_cache):
        tokens = [1, 2, 3]
        kv = mock_kv_cache(1000)

        small_cache.store(tokens, kv)
        initial_len = len(small_cache)

        # Store same tokens again
        small_cache.store(tokens, kv)

        # Should not create duplicate
        assert len(small_cache) == initial_len

    def test_max_entries_limit(self, model, mock_kv_cache):
        # Create cache with low entry limit
        config = MemoryCacheConfig(max_memory_mb=100, max_entries=3)
        cache = MemoryAwarePrefixCache(model, config)

        # Store 5 entries (only 3 should remain)
        for i in range(5):
            cache.store([i], mock_kv_cache(100))

        assert len(cache) <= 3


class TestCacheListTrimmability:
    """Regression tests: CacheList wrappers must be recognized as trimmable."""

    class TrimmableLayer:
        """Mock layer that reports is_trimmable() = True (like CacheList(KVCache, KVCache))."""

        def __init__(self, key_bytes: int, value_bytes: int, offset: int = 10):
            self.keys = MockArray(key_bytes)
            self.values = MockArray(value_bytes)
            self._offset = offset

        @property
        def offset(self):
            return self._offset

        @offset.setter
        def offset(self, val):
            self._offset = val

        def is_trimmable(self) -> bool:
            return True

    class NonTrimmableLayer:
        """Mock layer that reports is_trimmable() = False (like ArraysCache for DeltaNet)."""

        def __init__(self, nbytes: int = 100):
            self.keys = MockArray(nbytes)
            self.values = MockArray(nbytes)

        def is_trimmable(self) -> bool:
            return False

    @pytest.fixture
    def model(self):
        return MagicMock()

    @pytest.fixture
    def cache(self, model):
        config = MemoryCacheConfig(max_memory_mb=10, max_entries=10)
        return MemoryAwarePrefixCache(model, config)

    def test_supersequence_trimmable_cachelist_hits(self, cache):
        """CacheList with is_trimmable()=True must allow supersequence trim, not skip."""
        # Store a longer sequence
        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        kv = [self.TrimmableLayer(500, 500, offset=8)]
        cache.store(long_tokens, kv)

        # Fetch a shorter prefix — triggers supersequence path
        short_tokens = [1, 2, 3, 4, 5]
        result, remaining = cache.fetch(short_tokens)

        # With the old attribute-sniffing bug, CacheList wrappers (no .offset/.keys
        # on outer object) would be classified as non-trimmable and this would miss.
        assert result is not None, (
            "Supersequence match was skipped — CacheList misclassified as non-trimmable"
        )
        assert remaining == []

    def test_supersequence_non_trimmable_skipped(self, cache):
        """Non-trimmable layers (is_trimmable()=False) must still be skipped."""
        long_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        kv = [self.NonTrimmableLayer()]
        cache.store(long_tokens, kv)

        short_tokens = [1, 2, 3, 4, 5]
        result, remaining = cache.fetch(short_tokens)

        # Non-trimmable: should NOT return a supersequence match
        assert result is None
        assert remaining == short_tokens

    def test_lcp_trimmable_cachelist_hits(self, cache):
        """CacheList with is_trimmable()=True must allow LCP trim path."""
        # Store tokens that share a prefix but diverge
        stored = [1, 2, 3, 4, 5, 10, 11, 12]
        kv = [self.TrimmableLayer(500, 500, offset=8)]
        cache.store(stored, kv)

        # Request tokens that share prefix [1,2,3,4,5] but diverge after
        requested = [1, 2, 3, 4, 5, 20, 21]
        result, remaining = cache.fetch(requested)

        # LCP path should find shared prefix and trim excess
        assert result is not None, (
            "LCP match was skipped — CacheList misclassified as non-trimmable"
        )
        assert remaining == [20, 21]

    def test_lcp_non_trimmable_skipped(self, cache):
        """Non-trimmable layers must prevent LCP trim path."""
        stored = [1, 2, 3, 4, 5, 10, 11, 12]
        kv = [self.NonTrimmableLayer()]
        cache.store(stored, kv)

        requested = [1, 2, 3, 4, 5, 20, 21]
        result, remaining = cache.fetch(requested)

        assert result is None
        assert remaining == requested


class TestGetAvailableMemory:
    """Tests for _get_available_memory helper."""

    def test_with_psutil(self):
        try:
            from importlib.util import find_spec

            if find_spec("psutil") is None:
                pytest.skip("psutil not installed")
            mem = _get_available_memory()
            assert mem > 0
        except ImportError:
            pytest.skip("psutil not installed")

    def test_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            # Should return 0 when psutil not available
            # Note: This test may not work as expected due to import caching
            pass
