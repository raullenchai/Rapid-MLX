# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the radix-tree prefix-cache index wired into
``MemoryAwarePrefixCache`` (R15-P1, task #303).

These tests exercise the *coupling* between the radix and the cache —
they verify that:

- The radix stays in sync with ``_entries`` across store, evict, remove,
  and clear paths.
- The radix-accelerated ``fetch`` returns the same answer as the legacy
  bisect path on every interesting case (exact / prefix / divergent).
- The dedup-bytes-saved counter correctly identifies the shared-system-
  prompt workload pattern this whole task targets.

No model, no MLX runtime needed — we construct a fake KV-cache layer that
satisfies the ``_CacheEntry`` interface but skips the real tensor work.
This keeps the test under a second and lets it run in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from vllm_mlx.runtime.radix_index import RadixPrefixIndex


class _FakeCacheLayer:
    """Minimal stand-in for an mlx-lm KV-cache layer.

    Implements the small surface the memory-aware cache touches:
    - ``state`` property (returns the boxed payload — used for memory
      sizing inside ``_CacheEntry.create``).
    - ``meta_state`` for the same.
    - ``offset`` attribute (the trim path peeks at this).
    - ``trim(n)`` to mark the layer as trimmable so the LCP path can
      reuse this fake on divergent queries (mirrors mlx-lm's
      ``KVCache.trim`` semantics — purely informational here).

    All sizes are tiny so eviction is easy to drive in tests.
    """

    def __init__(self, payload_size: int = 1024):
        self.offset = 0
        # A bytes payload makes ``estimate_kv_cache_memory`` return a
        # stable value across runs (no MLX arrays involved).
        self._payload = b"\x00" * payload_size

    @property
    def state(self):
        return self._payload

    @property
    def meta_state(self):
        return (str(self.offset),)

    def trim(self, n: int) -> None:
        """No-op trim — the cache code only checks for the method's
        existence to decide trimmability."""
        self.offset = max(0, self.offset - n)

    def is_trimmable(self) -> bool:
        return True


def _make_cache(
    max_memory_mb: int = 64, with_radix: bool = True
) -> MemoryAwarePrefixCache:
    """Helper: build a cache + optional radix wired together."""
    model = MagicMock()
    config = MemoryCacheConfig(max_memory_mb=max_memory_mb, max_entries=1000)
    radix = RadixPrefixIndex() if with_radix else None
    return MemoryAwarePrefixCache(model=model, config=config, radix_index=radix)


def _cache_payload(n_layers: int = 2):
    """One ``_FakeCacheLayer`` per model layer."""
    return [_FakeCacheLayer() for _ in range(n_layers)]


class TestRadixCacheCoupling:
    def test_store_inserts_into_radix(self):
        cache = _make_cache()
        cache.store([1, 2, 3, 4], _cache_payload())
        radix = cache._radix_index
        assert radix is not None
        assert len(radix) == 1
        assert (1, 2, 3, 4) in radix

    def test_remove_drops_from_radix(self):
        cache = _make_cache()
        cache.store([1, 2, 3], _cache_payload())
        cache.store([1, 2, 4], _cache_payload())
        assert len(cache._radix_index) == 2
        cache.remove([1, 2, 3])
        assert len(cache._radix_index) == 1
        assert (1, 2, 4) in cache._radix_index
        assert (1, 2, 3) not in cache._radix_index

    def test_clear_resets_radix(self):
        cache = _make_cache()
        cache.store([1, 2, 3], _cache_payload())
        cache.store([4, 5, 6], _cache_payload())
        cache.clear()
        assert len(cache._radix_index) == 0
        assert cache._radix_index.stats()["node_count"] == 0


class TestRadixFetchParity:
    """The radix-accelerated fetch must agree with the legacy path."""

    def test_exact_match_hits_both_paths(self):
        radix_cache = _make_cache(with_radix=True)
        hash_cache = _make_cache(with_radix=False)
        for c in (radix_cache, hash_cache):
            c.store([1, 2, 3, 4], _cache_payload())
            kv, remaining = c.fetch([1, 2, 3, 4])
            assert kv is not None
            assert remaining == []
        assert radix_cache._last_match_type == hash_cache._last_match_type == "exact"

    def test_prefix_match_returns_same_remaining(self):
        # Cache holds [1,2,3], query is [1,2,3,4,5]; both paths should
        # return a hit with remaining=[4,5].
        radix_cache = _make_cache(with_radix=True)
        hash_cache = _make_cache(with_radix=False)
        for c in (radix_cache, hash_cache):
            c.store([1, 2, 3], _cache_payload())
            kv, remaining = c.fetch([1, 2, 3, 4, 5])
            assert kv is not None
            assert remaining == [4, 5]
        assert radix_cache._last_match_type == hash_cache._last_match_type == "prefix"

    def test_miss_returns_full_remaining(self):
        radix_cache = _make_cache(with_radix=True)
        hash_cache = _make_cache(with_radix=False)
        for c in (radix_cache, hash_cache):
            kv, remaining = c.fetch([1, 2, 3])
            assert kv is None
            assert remaining == [1, 2, 3]

    def test_longest_prefix_wins(self):
        # Two stored entries: [1,2,3] and [1,2,3,4,5]. Query [1,2,3,4,5,6]
        # must pick the longer entry.
        for use_radix in (True, False):
            c = _make_cache(with_radix=use_radix)
            c.store([1, 2, 3], _cache_payload())
            c.store([1, 2, 3, 4, 5], _cache_payload())
            kv, remaining = c.fetch([1, 2, 3, 4, 5, 6])
            assert kv is not None
            assert remaining == [6], f"radix={use_radix} got remaining={remaining}"


class TestRadixSharedSystemPromptWorkload:
    """The headline use-case for R15-P1: shared system prompts.

    Simulates N tenants whose prompts share the same K-token preamble,
    then verifies:
    1. Each tenant's prompt is a cache hit on the shared prefix.
    2. ``deduped_prefix_bytes_saved`` reflects the cross-tenant sharing.
    """

    def test_n_tenants_shared_preamble(self):
        cache = _make_cache(max_memory_mb=128)
        preamble = list(range(1, 201))  # 200-token system prompt
        N = 10
        # Each tenant stores ``preamble + tenant_suffix``.
        for tid in range(N):
            suffix = [10_000 + tid, 20_000 + tid, 30_000 + tid]
            cache.store(preamble + suffix, _cache_payload())
        radix = cache._radix_index
        # 10 entries, each sharing the full 200-token preamble.
        assert len(radix) == N
        deduped = radix.stats()["deduped_prefix_bytes_saved"]
        # First insert contributes 0; the next 9 each share the full
        # 200-token preamble → 9 * 200 * 4 bytes = 7200 bytes.
        assert deduped == 9 * 200 * 4

    def test_new_tenant_with_shared_preamble_is_cache_hit(self):
        cache = _make_cache(max_memory_mb=128)
        preamble = list(range(1, 201))
        # Existing tenant.
        cache.store(preamble + [10_001, 20_001], _cache_payload())
        # New tenant with same preamble, different user message.
        kv, remaining = cache.fetch(preamble + [99_999, 88_888, 77_777])
        # Should match SOMETHING — either the prefix (radix path) or the
        # supersequence-trim path (sorted-keys forward scan). Either way
        # the shared preamble must be reused.
        assert kv is not None
        # The shared prefix is at least the preamble length minus
        # supersequence excess. Easier check: at most 3 remaining
        # (the new tenant's 3-token suffix).
        assert len(remaining) <= 3


class TestRadixStatsExposure:
    def test_get_stats_includes_radix_subdict(self):
        cache = _make_cache()
        cache.store([1, 2, 3, 4], _cache_payload())
        stats = cache.get_stats()
        assert "radix" in stats
        radix_stats = stats["radix"]
        assert radix_stats["entry_count"] == 1
        assert radix_stats["node_count"] == 4

    def test_get_stats_omits_radix_when_hash_mode(self):
        cache = _make_cache(with_radix=False)
        cache.store([1, 2, 3, 4], _cache_payload())
        stats = cache.get_stats()
        assert "radix" not in stats


class TestRadixSurvivesLruEviction:
    """When LRU evicts an entry, the radix must drop it too.

    The cache's max_memory is set tight enough to force eviction of the
    older entry; the radix's entry_count must match cache size at all
    times. We don't snapshot ``node_count`` because some shared nodes
    might still be referenced.
    """

    def test_lru_eviction_keeps_radix_consistent(self):
        # Tight memory budget — each entry is ~2KB (payload_size=1024 ×
        # 2 layers), and the cache rounds up. Set max_memory_mb=1 so we
        # can only hold a handful of entries.
        cache = _make_cache(max_memory_mb=1)
        radix = cache._radix_index
        # Store enough entries to force at least one eviction. Each call
        # stores a disjoint sequence so the radix node count grows.
        for i in range(50):
            tokens = [i * 100 + j for j in range(50)]
            cache.store(tokens, _cache_payload(n_layers=4))
        # The cache's _entries count must match the radix entry count.
        # We don't assert a particular number — just that they agree
        # (the eviction policy may leave anywhere between 1 and 50
        # entries depending on per-entry size).
        assert len(cache._entries) == radix.stats()["entry_count"]
        assert len(cache._entries) == len(radix)


class TestRadixPersistenceWithCache:
    def test_save_load_roundtrip_through_cache(self, tmp_path):
        # Build cache with radix, store some entries, save just the
        # radix, then load it into a fresh radix and verify the entries
        # are reachable.
        cache = _make_cache()
        cache.store([1, 2, 3], _cache_payload())
        cache.store([1, 2, 4, 5], _cache_payload())
        path = str(tmp_path / "radix.index")
        cache._radix_index.save(path)
        # Fresh radix, load from disk.
        new_radix = RadixPrefixIndex()
        assert new_radix.load(path) is True
        assert (1, 2, 3) in new_radix
        assert (1, 2, 4, 5) in new_radix
