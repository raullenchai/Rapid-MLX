# SPDX-License-Identifier: Apache-2.0
"""Tests for SSD KV cache tiering."""

import pytest

from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheStats


class TestSSDCacheConfig:
    """Tests for SSDCacheConfig."""

    def test_default_config(self):
        config = SSDCacheConfig()
        assert config.cache_dir is None
        assert config.max_size_gb == 10.0
        assert config.max_entries == 10000
        assert config.file_permissions == 0o600
        assert config.dir_permissions == 0o700
        assert config.spill_queue_size == 64
        assert config.retention_seconds is None

    def test_custom_config(self):
        config = SSDCacheConfig(
            cache_dir="/tmp/test-ssd-cache",
            max_size_gb=5.0,
            max_entries=500,
        )
        assert config.cache_dir == "/tmp/test-ssd-cache"
        assert config.max_size_gb == 5.0
        assert config.max_entries == 500

    def test_max_size_bytes(self):
        config = SSDCacheConfig(max_size_gb=2.0)
        assert config.max_size_bytes == 2 * 1024 * 1024 * 1024

    def test_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size_gb"):
            SSDCacheConfig(max_size_gb=0.0)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            SSDCacheConfig(max_entries=0)

    def test_invalid_spill_queue_size(self):
        with pytest.raises(ValueError, match="spill_queue_size"):
            SSDCacheConfig(spill_queue_size=0)


class TestSSDCacheStats:
    """Tests for SSDCacheStats."""

    def test_initial_stats(self):
        stats = SSDCacheStats()
        assert stats.spill_count == 0
        assert stats.spill_bytes == 0
        assert stats.ssd_hits == 0
        assert stats.ssd_misses == 0
        assert stats.reload_latency_sum == 0.0
        assert stats.reload_bytes == 0
        assert stats.promotion_failures == 0

    def test_to_dict(self):
        stats = SSDCacheStats(
            spill_count=10,
            spill_bytes=1024 * 1024,
            ssd_hits=5,
            ssd_misses=3,
            reload_latency_sum=0.5,
            reload_bytes=512 * 1024,
            promotion_failures=1,
        )
        d = stats.to_dict()
        assert d["spill_count"] == 10
        assert d["spill_bytes"] == 1024 * 1024
        assert d["ssd_hits"] == 5
        assert d["ssd_misses"] == 3
        assert d["reload_bytes"] == 512 * 1024
        assert d["promotion_failures"] == 1
        assert d["ssd_hit_rate"] == pytest.approx(5 / 8)
        assert d["avg_reload_latency_ms"] == pytest.approx(100.0)

    def test_hit_rate_no_lookups(self):
        stats = SSDCacheStats()
        d = stats.to_dict()
        assert d["ssd_hit_rate"] == 0.0
        assert d["avg_reload_latency_ms"] == 0.0


import os

from vllm_mlx import ssd_cache as ssd_cache_module
from vllm_mlx.ssd_cache import SSDIndex


class TestSSDIndex:
    """Tests for SQLite-backed SSD cache index."""

    @pytest.fixture
    def db_dir(self, tmp_path):
        return str(tmp_path / "ssd_index_test")

    @pytest.fixture
    def index(self, db_dir):
        os.makedirs(db_dir, exist_ok=True)
        idx = SSDIndex(db_dir)
        yield idx
        idx.close()

    def test_create_opens_db(self, index, db_dir):
        assert os.path.exists(os.path.join(db_dir, "index.db"))

    def test_insert_and_lookup_exact(self, index):
        tokens = (1, 2, 3, 4, 5)
        index.insert_entry(
            tokens_key=tokens,
            file_path="entry_abc123.safetensors",
            memory_bytes=4096,
            num_tokens=5,
        )
        result = index.lookup_exact(tokens)
        assert result is not None
        assert result["file_path"] == "entry_abc123.safetensors"
        assert result["memory_bytes"] == 4096
        assert result["num_tokens"] == 5

    def test_lookup_exact_miss(self, index):
        result = index.lookup_exact((99, 98, 97))
        assert result is None

    def test_lookup_prefix(self, index):
        # Insert a few entries
        index.insert_entry((1, 2, 3), "a.safetensors", 1000, 3)
        index.insert_entry((1, 2, 3, 4, 5), "b.safetensors", 2000, 5)
        index.insert_entry((1, 2, 3, 4, 5, 6, 7), "c.safetensors", 3000, 7)
        index.insert_entry((9, 8, 7), "d.safetensors", 1000, 3)

        # Lookup prefix matches for (1,2,3,4,5,6,7,8)
        results = index.lookup_prefix((1, 2, 3, 4, 5, 6, 7, 8))
        # Should return entries whose tokens are a prefix of the query
        file_paths = [r["file_path"] for r in results]
        assert "a.safetensors" in file_paths
        assert "b.safetensors" in file_paths
        assert "c.safetensors" in file_paths
        assert "d.safetensors" not in file_paths

    def test_lookup_prefix_uses_bounded_prefix_hash_candidates(self, index):
        index.insert_entry((1, 2, 3, 4), "match.safetensors", 1000, 4)
        for i in range(100):
            index.insert_entry((9000 + i, 2, 3, 4), f"miss-{i}.safetensors", 1000, 4)

        results = index.lookup_prefix((1, 2, 3, 4, 5))

        assert [r["file_path"] for r in results] == ["match.safetensors"]
        with index._db_lock:
            cur = index._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE prefix_hash = ?",
                (ssd_cache_module._prefix_hash((1, 2, 3, 4)),),
            )
            assert cur.fetchone()[0] == 1

    def test_legacy_index_backfills_prefix_hashes(self, db_dir):
        import sqlite3
        import time

        os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(os.path.join(db_dir, "index.db"))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (1);
            CREATE TABLE entries (
                token_hash   TEXT PRIMARY KEY,
                tokens_blob  BLOB NOT NULL,
                num_tokens   INTEGER NOT NULL,
                file_path    TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                accessed_at  REAL NOT NULL
            );
            """)
        tokens = (1, 2, 3)
        now = time.time()
        conn.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ssd_cache_module._tokens_hash(tokens),
                ssd_cache_module._tokens_to_blob(tokens),
                len(tokens),
                "legacy.safetensors",
                1000,
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()

        idx = SSDIndex(db_dir)
        try:
            assert (
                idx.lookup_prefix((1, 2, 3, 4))[0]["file_path"] == "legacy.safetensors"
            )
            with idx._db_lock:
                cur = idx._conn.execute(
                    "SELECT prefix_hash FROM entries WHERE file_path = ?",
                    ("legacy.safetensors",),
                )
                assert cur.fetchone()[0] == ssd_cache_module._prefix_hash(tokens)
        finally:
            idx.close()

    def test_delete_entry(self, index):
        tokens = (10, 20, 30)
        index.insert_entry(tokens, "x.safetensors", 500, 3)
        assert index.lookup_exact(tokens) is not None

        index.delete_entry(tokens)
        assert index.lookup_exact(tokens) is None

    def test_get_lru(self, index):
        import time

        index.insert_entry((1,), "a.safetensors", 100, 1)
        time.sleep(0.01)  # Ensure different timestamps
        index.insert_entry((2,), "b.safetensors", 200, 1)
        time.sleep(0.01)
        index.insert_entry((3,), "c.safetensors", 300, 1)

        # Get oldest 2 entries
        lru = index.get_lru(limit=2)
        assert len(lru) == 2
        assert lru[0]["file_path"] == "a.safetensors"
        assert lru[1]["file_path"] == "b.safetensors"

    def test_get_total_bytes(self, index):
        index.insert_entry((1,), "a.safetensors", 1000, 1)
        index.insert_entry((2,), "b.safetensors", 2000, 1)
        assert index.get_total_bytes() == 3000

    def test_get_total_bytes_empty(self, index):
        assert index.get_total_bytes() == 0

    def test_get_entry_count(self, index):
        assert index.get_entry_count() == 0
        index.insert_entry((1,), "a.safetensors", 100, 1)
        index.insert_entry((2,), "b.safetensors", 200, 1)
        assert index.get_entry_count() == 2

    def test_insert_duplicate_replaces(self, index):
        tokens = (1, 2, 3)
        index.insert_entry(tokens, "old.safetensors", 100, 3)
        index.insert_entry(tokens, "new.safetensors", 200, 3)
        result = index.lookup_exact(tokens)
        assert result["file_path"] == "new.safetensors"
        assert result["memory_bytes"] == 200
        assert index.get_entry_count() == 1

    def test_touch_updates_access_time(self, index):
        import time

        index.insert_entry((1,), "a.safetensors", 100, 1)
        time.sleep(0.01)
        index.insert_entry((2,), "b.safetensors", 100, 1)

        # Touch the older entry
        index.touch((1,))

        # Now (2,) should be the LRU entry
        lru = index.get_lru(limit=1)
        assert lru[0]["file_path"] == "b.safetensors"


import numpy as np

from vllm_mlx.ssd_cache import (
    KVCacheSerializer,
    ArraysCacheSerializer,
    get_serializer_for_layer,
    SERIALIZER_SUPPORT_MATRIX,
)


class MockMLXArray:
    """Minimal mock for MLX array with shape, dtype, and numpy conversion."""

    def __init__(self, data):
        self._data = np.array(data)
        self.shape = self._data.shape
        self.dtype = type("dtype", (), {"size": self._data.dtype.itemsize})()

    def __array__(self):
        return self._data


class MockKVCacheLayer:
    """Mock KVCache layer with keys, values, offset."""

    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset


class MockArraysCacheLayer:
    """Mock ArraysCache layer with state list."""

    def __init__(self, state):
        self.state = state


class TestLayerSerializer:
    """Tests for per-layer serializer interface."""

    def test_support_matrix_has_required_types(self):
        assert "KVCache" in SERIALIZER_SUPPORT_MATRIX
        assert "RotatingKVCache" in SERIALIZER_SUPPORT_MATRIX
        assert "ArraysCache" in SERIALIZER_SUPPORT_MATRIX

    def test_kv_cache_serializer_round_trip(self, tmp_path):
        keys = MockMLXArray(np.random.randn(1, 8, 32, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 32, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=32)

        serializer = KVCacheSerializer()
        file_path = str(tmp_path / "layer_0.safetensors")
        # serialize_layer now expects a snapshot (CPU-only dict) produced by
        # snapshot_layer, not the raw MLX-backed layer. This mirrors the
        # producer/writer thread split: snapshot runs on the request thread,
        # serialize runs on the SSD writer thread with only the snapshot.
        snapshot = serializer.snapshot_layer(layer)
        metadata = serializer.serialize_layer(snapshot, 0, file_path)

        assert os.path.exists(file_path)
        assert metadata["layer_type"] == "KVCache"
        assert metadata["offset"] == 32

        restored = serializer.deserialize_layer(file_path, metadata)
        assert restored["offset"] == 32
        np.testing.assert_array_almost_equal(
            np.array(restored["keys"]),
            np.array(keys),
        )
        np.testing.assert_array_almost_equal(
            np.array(restored["values"]),
            np.array(values),
        )

    def test_arrays_cache_serializer_round_trip(self, tmp_path):
        arr0 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        arr1 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        layer = MockArraysCacheLayer(state=[arr0, arr1])

        serializer = ArraysCacheSerializer()
        file_path = str(tmp_path / "layer_0.safetensors")
        # See KVCacheSerializer round-trip test: snapshot is the producer-
        # side step, serialize is the writer-side step.
        snapshot = serializer.snapshot_layer(layer)
        metadata = serializer.serialize_layer(snapshot, 0, file_path)

        assert os.path.exists(file_path)
        assert metadata["layer_type"] == "ArraysCache"
        assert metadata["num_arrays"] == 2

        restored = serializer.deserialize_layer(file_path, metadata)
        assert len(restored["state"]) == 2
        np.testing.assert_array_almost_equal(
            np.array(restored["state"][0]),
            np.array(arr0),
        )

    def test_get_serializer_for_kvcache(self):
        layer = MockKVCacheLayer(keys=None, values=None, offset=0)
        s = get_serializer_for_layer(layer)
        assert isinstance(s, KVCacheSerializer)

    def test_get_serializer_for_arrays_cache(self):
        layer = MockArraysCacheLayer(state=[])
        s = get_serializer_for_layer(layer)
        assert isinstance(s, ArraysCacheSerializer)

    def test_get_serializer_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_serializer_for_layer("not a cache layer")


from vllm_mlx.ssd_cache import SSDCacheTier


class TestSSDCacheTierCore:
    """Tests for SSDCacheTier initialization and directory setup."""

    def test_init_creates_directory(self, tmp_path):
        cache_dir = str(tmp_path / "ssd_tier_test")
        config = SSDCacheConfig(cache_dir=cache_dir)
        tier = SSDCacheTier(config)
        try:
            assert os.path.isdir(cache_dir)
            assert os.path.exists(os.path.join(cache_dir, "data"))
            assert os.path.exists(os.path.join(cache_dir, "index.db"))
        finally:
            tier.close()

    def test_dir_permissions(self, tmp_path):
        cache_dir = str(tmp_path / "ssd_tier_perms")
        config = SSDCacheConfig(cache_dir=cache_dir, dir_permissions=0o700)
        tier = SSDCacheTier(config)
        try:
            stat = os.stat(os.path.join(cache_dir, "data"))
            # Check owner permissions (at least rwx for owner)
            assert stat.st_mode & 0o700 == 0o700
        finally:
            tier.close()

    def test_entry_hash_deterministic(self):
        tokens = (1, 2, 3, 4, 5)
        h1 = SSDCacheTier._entry_hash(tokens)
        h2 = SSDCacheTier._entry_hash(tokens)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_entry_hash_different_for_different_tokens(self):
        h1 = SSDCacheTier._entry_hash((1, 2, 3))
        h2 = SSDCacheTier._entry_hash((1, 2, 4))
        assert h1 != h2

    def test_stats_initial(self, tmp_path):
        config = SSDCacheConfig(cache_dir=str(tmp_path / "stats_test"))
        tier = SSDCacheTier(config)
        try:
            stats = tier.get_stats()
            assert stats["spill_count"] == 0
            assert stats["ssd_hits"] == 0
        finally:
            tier.close()

    def test_close_idempotent(self, tmp_path):
        config = SSDCacheConfig(cache_dir=str(tmp_path / "close_test"))
        tier = SSDCacheTier(config)
        tier.close()
        tier.close()  # Should not raise

    def test_init_closes_index_when_later_setup_fails(self, tmp_path, monkeypatch):
        fake_index = type(
            "FakeIndex",
            (),
            {
                "closed": False,
                "close": lambda self: setattr(self, "closed", True),
            },
        )()

        monkeypatch.setattr(ssd_cache_module, "SSDIndex", lambda cache_dir: fake_index)

        def fail_queue(*args, **kwargs):
            raise RuntimeError("queue setup failed")

        monkeypatch.setattr(ssd_cache_module.queue, "Queue", fail_queue)

        config = SSDCacheConfig(cache_dir=str(tmp_path / "init_fail"))
        with pytest.raises(RuntimeError, match="queue setup failed"):
            SSDCacheTier(config)

        assert fake_index.closed

    def test_init_preserves_setup_failure_when_index_close_fails(
        self, tmp_path, monkeypatch
    ):
        class FakeIndex:
            def close(self):
                raise RuntimeError("index close failed")

        monkeypatch.setattr(ssd_cache_module, "SSDIndex", lambda cache_dir: FakeIndex())

        def fail_queue(*args, **kwargs):
            raise RuntimeError("queue setup failed")

        monkeypatch.setattr(ssd_cache_module.queue, "Queue", fail_queue)

        config = SSDCacheConfig(cache_dir=str(tmp_path / "init_close_fail"))
        with pytest.raises(RuntimeError, match="queue setup failed"):
            SSDCacheTier(config)


import time


class TestAsyncSpillWriter:
    """Tests for the async spill writer thread."""

    @pytest.fixture
    def tier_with_writer(self, tmp_path):
        config = SSDCacheConfig(cache_dir=str(tmp_path / "writer_test"))
        tier = SSDCacheTier(config)
        tier.start_writer()
        yield tier
        tier.close()

    def test_spill_writes_entry_to_disk(self, tier_with_writer):
        tokens = (1, 2, 3, 4, 5)
        keys = MockMLXArray(np.random.randn(1, 8, 16, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 16, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=16)
        cache = [layer]

        tier_with_writer.enqueue_spill(tokens, cache, memory_bytes=2048)

        # Wait for async write to complete (with timeout)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier_with_writer._stats.spill_count > 0:
                break
            time.sleep(0.05)

        assert tier_with_writer._stats.spill_count == 1
        assert tier_with_writer._stats.spill_bytes > 0

        # Verify entry in index
        result = tier_with_writer._index.lookup_exact(tokens)
        assert result is not None
        assert result["num_tokens"] == 5

    def test_spill_atomic_write(self, tier_with_writer):
        """Verify no partial files exist after spill."""
        tokens = (10, 20, 30)
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)

        tier_with_writer.enqueue_spill(tokens, [layer], memory_bytes=1024)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier_with_writer._stats.spill_count > 0:
                break
            time.sleep(0.05)

        # Check that no .tmp files remain
        data_dir = tier_with_writer._data_dir
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                assert not f.endswith(".tmp"), f"Temp file left behind: {f}"

    def test_spill_queue_full_drops(self, tmp_path):
        """When queue is full, new spills are dropped (not blocking)."""
        config = SSDCacheConfig(
            cache_dir=str(tmp_path / "queue_full_test"),
            spill_queue_size=2,
        )
        tier = SSDCacheTier(config)
        # Don't start writer — queue will fill up
        keys = MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16))
        values = MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=1)

        # Enqueue more than capacity
        for i in range(5):
            tier.enqueue_spill((i,), [layer], memory_bytes=100)

        # Queue should be at capacity, not have raised
        assert tier._spill_queue.qsize() <= 2
        tier.close()


from unittest.mock import MagicMock
from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class MockKVCacheForSpill:
    """Mock KVCache with keys, values, offset for memory estimation."""

    def __init__(self, size_bytes: int):
        # Use real numpy arrays so serialization works
        self.keys = MockMLXArray(np.zeros((1, 4, 8, 16), dtype=np.float16))
        self.values = MockMLXArray(np.zeros((1, 4, 8, 16), dtype=np.float16))
        self.offset = 8


class TestSpillPath:
    """Tests for eviction -> SSD spill integration."""

    def test_evict_lru_calls_ssd_spill(self, tmp_path):
        model = MagicMock()
        # min_prefix_tokens=1: test uses 10-token sequences (prod default 128
        # would silently reject every store()).
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=3, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        ssd_config = SSDCacheConfig(cache_dir=str(tmp_path / "spill_test"))
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()
        cache.set_ssd_tier(ssd_tier)

        # Fill cache to capacity (3 entries)
        for i in range(3):
            kv = [MockKVCacheForSpill(1000)]
            cache.store(list(range(i * 10, (i + 1) * 10)), kv)

        assert len(cache) == 3

        # Store one more to trigger eviction
        kv = [MockKVCacheForSpill(1000)]
        cache.store(list(range(100, 110)), kv)

        # Wait for spill
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        assert ssd_tier._stats.spill_count >= 1
        ssd_tier.close()

    def test_evict_without_ssd_tier_still_works(self):
        """Eviction without SSD tier should work as before (discard)."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        for i in range(5):
            kv = [MockKVCacheForSpill(1000)]
            cache.store(list(range(i * 10, (i + 1) * 10)), kv)

        # Should not raise, just discard
        assert len(cache) <= 2

    def test_snapshot_runs_on_caller_thread(self, tmp_path):
        """Regression: enqueue_spill must materialize layers on the caller
        thread (real mx.array hard-aborts the process if accessed off the
        Stream(gpu, N) it was created on)."""
        import threading as _threading

        creator_thread_id = _threading.get_ident()

        class ThreadBoundArray:
            # Python-level proxy for the MLX cross-thread Stream abort.
            def __init__(self, data, owner_tid):
                self._data = data
                self._owner_tid = owner_tid

            def __array__(self):
                if _threading.get_ident() != self._owner_tid:
                    raise RuntimeError("accessed from wrong thread")
                return self._data

        class ThreadBoundKV:
            def __init__(self, tid):
                self.keys = ThreadBoundArray(
                    np.zeros((1, 1, 1, 1), dtype=np.float16), tid
                )
                self.values = ThreadBoundArray(
                    np.zeros((1, 1, 1, 1), dtype=np.float16), tid
                )
                self.offset = 1

        ssd_config = SSDCacheConfig(cache_dir=str(tmp_path / "snapshot_test"))
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()

        try:
            tokens = tuple(range(0, 200))
            enqueued = ssd_tier.enqueue_spill(
                tokens,
                [ThreadBoundKV(creator_thread_id)],
                memory_bytes=64,
            )
            assert enqueued is True

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if ssd_tier._stats.spill_count >= 1:
                    break
                time.sleep(0.05)
            assert ssd_tier._stats.spill_count == 1
        finally:
            ssd_tier.close()


import asyncio


class TestAsyncFetchPath:
    """Tests for async SSD fetch with RAM budget reservation."""

    @pytest.fixture
    def populated_tier(self, tmp_path):
        """Create an SSD tier with one pre-written entry."""
        config = SSDCacheConfig(cache_dir=str(tmp_path / "fetch_test"))
        tier = SSDCacheTier(config)
        tier.start_writer()

        tokens = (1, 2, 3, 4, 5)
        keys = MockMLXArray(np.random.randn(1, 8, 16, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 16, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=16)

        tier.enqueue_spill(tokens, [layer], memory_bytes=2048)

        # Wait for write
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        yield tier, tokens, keys, values
        tier.close()

    def test_lookup_ssd_returns_candidate(self, populated_tier):
        tier, tokens, _, _ = populated_tier
        candidate = tier.lookup_ssd(tokens)
        assert candidate is not None
        assert candidate["num_tokens"] == 5
        assert candidate["memory_bytes"] == 2048

    def test_lookup_ssd_miss(self, populated_tier):
        tier, _, _, _ = populated_tier
        candidate = tier.lookup_ssd((99, 98, 97))
        assert candidate is None

    def test_async_promote_reserves_budget_then_reads(self, populated_tier):
        """RAM budget is reserved BEFORE the SSD read completes."""
        tier, tokens, keys, values = populated_tier

        budget_reserved = []
        budget_released = []

        def reserve_fn(nbytes):
            budget_reserved.append(nbytes)
            return True  # Budget available

        def release_fn(nbytes):
            budget_released.append(nbytes)

        result = asyncio.run(tier.async_promote(tokens, reserve_fn, release_fn))

        # Budget was reserved
        assert len(budget_reserved) == 1
        assert budget_reserved[0] == 2048

        # Result contains the cache layers
        assert result is not None
        assert len(result) == 1  # One layer

        # Stats updated
        assert tier._stats.ssd_hits == 1
        assert tier._stats.reload_bytes > 0

    def test_async_promote_budget_denied(self, populated_tier):
        """When budget reservation fails, promote returns None."""
        tier, tokens, _, _ = populated_tier

        def reserve_fn(nbytes):
            return False  # Budget denied

        def release_fn(nbytes):
            pass

        result = asyncio.run(tier.async_promote(tokens, reserve_fn, release_fn))

        assert result is None
        assert tier._stats.promotion_failures == 1

    def test_async_promote_read_failure_releases_budget(self, populated_tier):
        """If disk read fails after reservation, budget is released."""
        tier, tokens, _, _ = populated_tier

        budget_reserved = []
        budget_released = []

        def reserve_fn(nbytes):
            budget_reserved.append(nbytes)
            return True

        def release_fn(nbytes):
            budget_released.append(nbytes)

        # Corrupt the entry on disk
        entry_hash = tier._entry_hash(tokens)
        entry_dir = os.path.join(tier._data_dir, entry_hash)
        manifest_path = os.path.join(entry_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write("corrupted!")

        result = asyncio.run(tier.async_promote(tokens, reserve_fn, release_fn))

        assert result is None
        # Budget was reserved then released
        assert len(budget_reserved) == 1
        assert len(budget_released) == 1
        assert budget_released[0] == budget_reserved[0]


class TestCapacityManagement:
    """Tests for SSD disk capacity management."""

    @pytest.fixture
    def small_tier(self, tmp_path):
        """SSD tier with very small capacity for testing eviction."""
        config = SSDCacheConfig(
            cache_dir=str(tmp_path / "capacity_test"),
            max_size_gb=0.0001,  # ~100KB
            max_entries=3,
        )
        tier = SSDCacheTier(config)
        tier.start_writer()
        yield tier
        tier.close()

    def _write_and_wait(self, tier, tokens, size=1024):
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)
        tier.enqueue_spill(tokens, [layer], memory_bytes=size)

        deadline = time.monotonic() + 5.0
        initial_count = tier._stats.spill_count
        while time.monotonic() < deadline:
            if tier._stats.spill_count > initial_count:
                break
            time.sleep(0.05)

    def test_disk_lru_eviction(self, small_tier):
        """When disk capacity is exceeded, oldest entries are evicted."""
        # Fill with 3 entries (at max_entries limit)
        for i in range(3):
            self._write_and_wait(small_tier, tuple(range(i * 10, (i + 1) * 10)))

        assert small_tier._index.get_entry_count() == 3

        # Write one more — should trigger disk LRU eviction
        self._write_and_wait(small_tier, (100, 101, 102))

        # Should still be at or below max_entries
        assert small_tier._index.get_entry_count() <= 3

    def test_startup_reconciliation(self, tmp_path):
        """On startup, index is reconciled with files on disk."""
        cache_dir = str(tmp_path / "reconcile_test")
        config = SSDCacheConfig(cache_dir=cache_dir)
        tier = SSDCacheTier(config)
        tier.start_writer()

        # Write an entry
        tokens = (1, 2, 3)
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)
        tier.enqueue_spill(tokens, [layer], memory_bytes=1024)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        tier.close()

        # Delete the data file but leave the index entry
        entry_hash = SSDCacheTier._entry_hash(tokens)
        entry_dir = os.path.join(cache_dir, "data", entry_hash)
        import shutil

        shutil.rmtree(entry_dir)

        # Re-open — reconciliation should detect the missing file
        tier2 = SSDCacheTier(config)
        tier2.reconcile()

        # The orphaned index entry should have been removed
        assert tier2._index.lookup_exact(tokens) is None
        tier2.close()


class TestCLIIntegration:
    """Tests for CLI argument parsing of SSD cache flags."""

    def test_scheduler_config_has_ssd_fields(self):
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig()
        assert config.ssd_cache_dir is None
        assert config.ssd_cache_max_gb == 10.0

    def test_scheduler_config_custom_ssd(self):
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig(
            ssd_cache_dir="/tmp/test-ssd",
            ssd_cache_max_gb=5.0,
        )
        assert config.ssd_cache_dir == "/tmp/test-ssd"
        assert config.ssd_cache_max_gb == 5.0


class TestMemoryCacheSSDCheck:
    """Tests for MemoryAwarePrefixCache.check_ssd() method."""

    def test_check_ssd_returns_candidate_on_miss(self, tmp_path):
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        ssd_config = SSDCacheConfig(cache_dir=str(tmp_path / "check_ssd_test"))
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()
        cache.set_ssd_tier(ssd_tier)

        # Write an entry to SSD directly
        tokens = (1, 2, 3, 4, 5)
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)
        ssd_tier.enqueue_spill(tokens, [layer], memory_bytes=1024)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        # RAM fetch should miss
        result, remaining = cache.fetch(list(tokens))
        assert result is None

        # But SSD check should find it
        candidate = cache.check_ssd(list(tokens))
        assert candidate is not None
        assert candidate["num_tokens"] == 5

        ssd_tier.close()

    def test_check_ssd_returns_none_without_tier(self):
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        candidate = cache.check_ssd([1, 2, 3])
        assert candidate is None

    def test_check_ssd_returns_none_on_ram_hit(self, tmp_path):
        """When RAM has a hit, check_ssd should return None (not needed)."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        kv = [MockKVCacheForSpill(1000)]
        cache.store([1, 2, 3], kv)

        # RAM hit exists
        result, _ = cache.fetch([1, 2, 3])
        assert result is not None

        # check_ssd should indicate no SSD needed
        candidate = cache.check_ssd([1, 2, 3])
        assert candidate is None


class TestIntegrationSpillAndFetch:
    """End-to-end tests: spill from RAM -> SSD -> promote back."""

    @pytest.fixture
    def cache_with_ssd(self, tmp_path):
        """RAM cache + SSD tier, small limits to force eviction."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2, min_prefix_tokens=1)
        ram_cache = MemoryAwarePrefixCache(model, config)

        ssd_config = SSDCacheConfig(
            cache_dir=str(tmp_path / "integration_test"),
            max_size_gb=1.0,
        )
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()
        ram_cache.set_ssd_tier(ssd_tier)

        yield ram_cache, ssd_tier
        ssd_tier.close()

    def _make_kv_layer(self, offset=16):
        keys = MockMLXArray(np.random.randn(1, 8, offset, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, offset, 64).astype(np.float16))
        return MockKVCacheLayer(keys=keys, values=values, offset=offset)

    def test_full_round_trip(self, cache_with_ssd):
        """Store -> evict to SSD -> fetch miss in RAM -> find in SSD."""
        ram_cache, ssd_tier = cache_with_ssd

        # Store 2 entries (fills RAM cache)
        tokens_a = list(range(0, 10))
        tokens_b = list(range(10, 20))
        ram_cache.store(tokens_a, [self._make_kv_layer()])
        ram_cache.store(tokens_b, [self._make_kv_layer()])
        assert len(ram_cache) == 2

        # Store a third entry — should evict tokens_a to SSD
        tokens_c = list(range(20, 30))
        ram_cache.store(tokens_c, [self._make_kv_layer()])

        # Wait for SSD spill
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        assert ssd_tier._stats.spill_count >= 1

        # RAM miss for tokens_a
        result, remaining = ram_cache.fetch(tokens_a)
        assert result is None

        # SSD should have tokens_a
        candidate = ram_cache.check_ssd(tokens_a)
        assert candidate is not None

        # Promote from SSD
        async def do_promote():
            return await ssd_tier.async_promote(
                tuple(tokens_a),
                lambda n: True,  # Budget always available
                lambda n: None,
            )

        promoted = asyncio.run(do_promote())
        assert promoted is not None
        assert len(promoted) == 1  # One layer
        assert ssd_tier._stats.ssd_hits == 1

    def test_hybrid_cache_round_trip(self, tmp_path):
        """ArraysCache layers survive SSD round-trip."""
        config = SSDCacheConfig(cache_dir=str(tmp_path / "hybrid_test"))
        tier = SSDCacheTier(config)
        tier.start_writer()

        tokens = (1, 2, 3)
        arr0 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        arr1 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        layer = MockArraysCacheLayer(state=[arr0, arr1])

        tier.enqueue_spill(tokens, [layer], memory_bytes=2048)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        async def do_promote():
            return await tier.async_promote(
                tokens,
                lambda n: True,
                lambda n: None,
            )

        promoted = asyncio.run(do_promote())
        assert promoted is not None
        assert len(promoted) == 1
        assert "state" in promoted[0]
        assert len(promoted[0]["state"]) == 2

        # Verify data integrity
        np.testing.assert_array_almost_equal(
            np.array(promoted[0]["state"][0]),
            np.array(arr0),
        )

        tier.close()

    def test_scheduler_reconstructs_real_arrays_cache(self):
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        mx = pytest.importorskip("mlx.core")
        from vllm_mlx.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        reconstructed = Scheduler._reconstruct_ssd_layers(
            scheduler,
            [{"state": [np.array([1, 2], dtype=np.float32)]}],
        )

        assert reconstructed is not None
        assert isinstance(reconstructed[0], cache_mod.ArraysCache)
        assert reconstructed[0].state[0].tolist() == mx.array([1, 2]).tolist()

    def test_capacity_eviction_end_to_end(self, tmp_path):
        """Entries beyond max_entries are evicted from SSD."""
        config = SSDCacheConfig(
            cache_dir=str(tmp_path / "cap_e2e"),
            max_entries=2,
        )
        tier = SSDCacheTier(config)
        tier.start_writer()

        for i in range(4):
            tokens = tuple(range(i * 10, (i + 1) * 10))
            layer = MockKVCacheLayer(
                keys=MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16)),
                values=MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16)),
                offset=1,
            )
            tier.enqueue_spill(tokens, [layer], memory_bytes=100)

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if tier._stats.spill_count > i:
                    break
                time.sleep(0.05)

        # Only max_entries (2) should remain
        assert tier._index.get_entry_count() <= 2

        # The latest entries should still be there
        assert tier._index.lookup_exact(tuple(range(30, 40))) is not None

        tier.close()

    def test_metrics_accuracy(self, cache_with_ssd):
        """Verify all SSD metrics are updated correctly."""
        ram_cache, ssd_tier = cache_with_ssd

        # Fill RAM and force one spill
        ram_cache.store(list(range(10)), [self._make_kv_layer()])
        ram_cache.store(list(range(10, 20)), [self._make_kv_layer()])
        ram_cache.store(list(range(20, 30)), [self._make_kv_layer()])

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        stats = ssd_tier.get_stats()
        assert stats["spill_count"] >= 1
        assert stats["spill_bytes"] > 0

        # Promote the spilled entry
        evicted_tokens = tuple(range(10))  # First stored, first evicted
        candidate = ssd_tier.lookup_ssd(evicted_tokens)
        if candidate is not None:

            async def promote():
                return await ssd_tier.async_promote(
                    evicted_tokens, lambda n: True, lambda n: None
                )

            asyncio.run(promote())
            stats = ssd_tier.get_stats()
            assert stats["ssd_hits"] >= 1
            assert stats["reload_bytes"] > 0
            assert stats["avg_reload_latency_ms"] > 0
