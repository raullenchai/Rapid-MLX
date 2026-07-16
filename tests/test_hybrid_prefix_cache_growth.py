# SPDX-License-Identifier: Apache-2.0
"""Hybrid (GatedDeltaNet / Mamba MoE) recurrent-state cache policy.

History
-------
Issue #214 (oldriverno1, michaelasper) originally asked for hybrid multi-turn
conversations to hit the prefix cache the way dense models do, so TTFT would
not grow linearly with conversation length. We shipped that: stored
``[P + R1]`` was reused as a strict prefix of turn-2's ``[P + R1 + M2]`` (no
trim required — the RNN state at end-of-stored is exactly the state needed at
start-of-M2-prefill).

Issues #1025 / #1058 then showed the OTHER edge of the same behavior: those
per-request recurrent-state (``ArraysCache``) entries are stored by reference
and are NEVER a prefix of the *next* request across DIFFERENT conversations
(each request's output differs → every key is a unique superset), so
prefix-subset eviction never reclaims them. They only drop under the cache's
own byte budget (independent of ``--gpu-memory-utilization``), so Metal
``active`` ratchets up holding leaked recurrent state → D-METAL-CAP wedges /
OOM.

Resolution (direction 1, raullen 2026-07-09)
--------------------------------------------
Stop caching non-trimmable recurrent-state entries entirely. ``store`` now
DROPS any cache that carries an ``ArraysCache`` / ``CacheList``-wrapping-one
layer (``is_trimmable() == False``). This trades away the #214 within-
conversation multi-turn speedup (hybrid turns re-prefill) to stop the
cross-conversation leak. The tests below encode the NEW policy; the previous
#214 "must hit" assertions are intentionally inverted.

Dense (all-``KVCache``) models are unaffected — their state is trimmable and
still cached/reused normally.

#1103 refinement (opt-in bounded trim-free reuse)
-------------------------------------------------
Issue #1103 showed a workload where the #214 reuse was real, not synthetic:
a byte-stable system prompt + tools with append-only history produces clean
prefix-extension matches, which fetch serves WITHOUT trimming (the leak's
trim-side guards never applied to exact / prefix-extension matches). With
``hybrid_reuse_max_entries = N > 0`` (CLI ``--hybrid-cache-entries N``),
``store`` retains up to N non-trimmable entries — LRU-evicted among
themselves so #1025's unbounded unique-superset accumulation cannot recur —
and fetch keeps refusing the trim-requiring paths. The default (0) preserves
the drop-at-store policy above byte-for-byte; the original policy tests are
unchanged. The opt-in tests live in the "#1103" section at the bottom.
"""

from unittest.mock import MagicMock

import pytest

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class _MockArray:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class TrimmableLayer:
    """Stands in for KVCache (transformer attention layer)."""

    def __init__(self, nbytes: int = 200, offset: int = 0):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)
        self._offset = offset

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, val: int) -> None:
        self._offset = val

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:  # KVCache-like: defines trim
        return n


class NonTrimmableLayer:
    """Stands in for ArraysCache (DeltaNet/Mamba RNN state)."""

    def __init__(self, nbytes: int = 200):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)

    def is_trimmable(self) -> bool:
        return False


def _dense_cache(n: int = 10):
    return [TrimmableLayer() for _ in range(n)]


def _hybrid_cache(n_trimmable: int = 10, n_non_trimmable: int = 30):
    """Mirror Qwen3.5/3.6 hybrid layout: ~25% transformer, ~75% DeltaNet."""
    return [TrimmableLayer() for _ in range(n_trimmable)] + [
        NonTrimmableLayer() for _ in range(n_non_trimmable)
    ]


@pytest.fixture
def cache():
    config = MemoryCacheConfig(max_memory_mb=10, max_entries=64)
    return MemoryAwarePrefixCache(MagicMock(), config)


# ---------------------------------------------------------------------------
# Dense (all-trimmable) is unaffected — still stored and reused.
# ---------------------------------------------------------------------------


def test_dense_growing_conversation_hits_prefix(cache):
    """Dense models still hit the prefix path on growing conversations."""
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    assert cache.store(prompt, _dense_cache()) is True
    assert cache.store(prompt + response_1, _dense_cache()) is True

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is not None, "Dense growing conversation should hit prefix"
    assert remaining == new_msg


# ---------------------------------------------------------------------------
# #1025 / #1058: hybrid recurrent-state entries are DROPPED at store time.
# ---------------------------------------------------------------------------


def test_hybrid_store_is_dropped(cache):
    """A cache with any non-trimmable layer must NOT be stored (leak fix)."""
    prompt = list(range(1000, 1100))

    stored = cache.store(prompt, _hybrid_cache())

    assert stored is False, "Hybrid recurrent-state entry must be dropped, not stored"
    assert tuple(prompt) not in cache._entries, (
        "Non-trimmable entry leaked into _entries — this is the #1025/#1058 leak"
    )
    assert cache.get_stats()["non_trimmable_skips"] == 1


def test_hybrid_multiturn_does_not_leak(cache):
    """A multi-turn hybrid conversation leaves NO entries in the cache.

    Every turn stores a longer ``[P + ... ]`` superset; before the fix each
    one lingered forever (never a prefix of a *different* conversation's next
    key). After the fix none are retained → ``_entries`` stays empty and
    ``_current_memory`` returns to 0.
    """
    prompt = list(range(1000, 1100))
    r1, r2 = [9001, 9002], [9003, 9004]
    m2, m3 = list(range(2000, 2050)), list(range(3000, 3030))

    cache.store(prompt, _hybrid_cache())
    cache.store(prompt + r1, _hybrid_cache())
    cache.store(prompt + r1 + m2 + r2, _hybrid_cache())
    cache.store(prompt + r1 + m2 + r2 + m3, _hybrid_cache())

    assert len(cache._entries) == 0, (
        f"Hybrid conversation left {len(cache._entries)} lingering entries — "
        "this is the recurrent-state leak (#1025/#1058)"
    )
    assert cache._current_memory == 0
    assert cache.get_stats()["non_trimmable_skips"] == 4


def test_hybrid_fetch_always_misses(cache):
    """With hybrid stores dropped, every hybrid fetch is a clean miss."""
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    cache.store(prompt, _hybrid_cache())
    cache.store(prompt + response_1, _hybrid_cache())

    turn_2 = prompt + response_1 + new_msg
    result, remaining = cache.fetch(turn_2)

    assert result is None, "Hybrid entries are never stored → fetch must miss"
    assert remaining == turn_2


# ---------------------------------------------------------------------------
# Granularity: partial hybrid (even ONE non-trimmable layer) drops the entry.
# ---------------------------------------------------------------------------


def test_single_non_trimmable_layer_drops_entry(cache):
    """One non-trimmable layer among many trimmable ones drops the whole entry.

    A half-populated entry (trimmable layers only) can't reconstruct a hybrid
    model, so we skip the whole entry rather than store a useless subset.
    """
    prompt = list(range(1000, 1100))
    mostly_dense = _dense_cache(n=39) + [NonTrimmableLayer()]

    assert cache.store(prompt, mostly_dense) is False
    assert tuple(prompt) not in cache._entries


def test_dict_form_arrayscache_dropped(cache):
    """Block-aware (dict-form) extracted states are gated on class_name too."""
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": "KVCache", "state": (1, 2), "meta_state": ("0",)},
        {"class_name": "ArraysCache", "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is False
    assert tuple(prompt) not in cache._entries


def test_dict_form_all_kvcache_stored(cache):
    """A dict-form entry with only KVCache layers is still stored."""
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": "KVCache", "state": (1, 2), "meta_state": ("0",)},
        {"class_name": "KVCache", "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is True
    assert tuple(prompt) in cache._entries


@pytest.mark.parametrize(
    "kv_class",
    ["RotatingKVCache", "ChunkedKVCache", "ConcatenateKVCache", "QuantizedKVCache"],
)
def test_dict_form_trimmable_kv_variants_still_stored(cache, kv_class):
    """Dict-form sliding-window / other trimmable KV classes must NOT be dropped.

    Regression for codex #1075 finding: an allowlist of only KVCache would
    wrongly classify RotatingKVCache & friends as non-trimmable and drop the
    entry, regressing prefix reuse for dense / sliding-window models. The
    denylist keeps them cacheable.
    """
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": kv_class, "state": (1, 2), "meta_state": ("0",)},
        {"class_name": kv_class, "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is True, (
        f"{kv_class} is trimmable and must remain cacheable"
    )
    assert tuple(prompt) in cache._entries


def test_dict_form_mamba_variant_dropped(cache):
    """Vendor-suffixed recurrent class names are caught by substring match."""
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": "KVCache", "state": (1, 2), "meta_state": ("0",)},
        {
            "class_name": "GatedDeltaNetArraysCache",
            "state": (3, 4),
            "meta_state": ("0",),
        },
    ]

    assert cache.store(prompt, dict_cache) is False
    assert tuple(prompt) not in cache._entries


# ---------------------------------------------------------------------------
# Guards preserved from the #214 era: trim-required matches still MISS. These
# now also never even reach fetch (store dropped them), but the fetch-side
# non-trimmable guard stays as defense-in-depth for any legacy on-disk entry.
# ---------------------------------------------------------------------------


def test_hybrid_supersequence_still_skipped(cache):
    """Even if a hybrid entry existed, a trim-required supersequence match must
    skip. We inject directly into ``_entries`` to bypass the store gate and
    exercise the fetch-side guard (legacy on-disk entry defense-in-depth).
    """
    from vllm_mlx.memory_cache import _CacheEntry

    long_stored = list(range(1000, 1200))
    entry = _CacheEntry.create(long_stored, _hybrid_cache())
    cache._entries[tuple(long_stored)] = entry
    import bisect

    bisect.insort(cache._sorted_keys, tuple(long_stored))

    short_request = list(range(1000, 1100))
    result, remaining = cache.fetch(short_request)

    assert result is None, (
        "Trim-required match on non-trimmable hybrid layers must still skip"
    )
    assert remaining == short_request


# ---------------------------------------------------------------------------
# #1103: opt-in bounded trim-free reuse (hybrid_reuse_max_entries > 0).
#
# The trim-free fetch paths (exact match, prefix-extension) never needed the
# non-trimmable guard — resuming a stored prefix at its own token boundary
# requires no trim. These tests cover the opt-in policy that stores hybrid
# entries for exactly those paths, bounded so #1025 cannot recur.
# ---------------------------------------------------------------------------


class ArraysCacheLayer:
    """Mirror mlx-lm ``ArraysCache``: ``state`` is a flat list of N arrays
    (N is model-defined — NOT always 2), no ``keys``/``values`` attributes."""

    def __init__(self, n_arrays: int = 3, nbytes_each: int = 100):
        self._arrays = [_MockArray(nbytes_each) for _ in range(n_arrays)]

    @property
    def state(self):
        return self._arrays

    def is_trimmable(self) -> bool:
        return False


class CacheListLayer:
    """Mirror mlx-lm ``CacheList``: ``state`` is a NESTED list of the wrapped
    caches' states; trimmable only if every wrapped cache is."""

    def __init__(self, *caches):
        self._caches = caches

    @property
    def state(self):
        return [c.state for c in self._caches]

    def is_trimmable(self) -> bool:
        return all(c.is_trimmable() for c in self._caches)


@pytest.fixture
def reuse_cache():
    """Cache with the #1103 opt-in enabled (bound of 2 hybrid entries)."""
    config = MemoryCacheConfig(
        max_memory_mb=10, max_entries=64, hybrid_reuse_max_entries=2
    )
    return MemoryAwarePrefixCache(MagicMock(), config)


def test_hybrid_store_retained_when_enabled(reuse_cache):
    """With the opt-in, hybrid entries are stored (and not counted as skips)."""
    prompt = list(range(1000, 1100))

    assert reuse_cache.store(prompt, _hybrid_cache()) is True
    assert tuple(prompt) in reuse_cache._entries
    assert reuse_cache.get_stats()["non_trimmable_skips"] == 0
    assert reuse_cache.get_stats()["non_trimmable_entries"] == 1


def test_hybrid_exact_match_hit_when_enabled(reuse_cache):
    """Exact match is trim-free — a retained hybrid entry must serve it."""
    prompt = list(range(1000, 1100))
    reuse_cache.store(prompt, _hybrid_cache())

    result, remaining = reuse_cache.fetch(prompt)

    assert result is not None, "Exact match needs no trim — hybrid entry must hit"
    assert remaining == []


def test_hybrid_prefix_extension_hit_when_enabled(reuse_cache):
    """The #214 growing-conversation shape works again under the opt-in:
    stored ``[P + R1]`` serves turn-2's ``[P + R1 + M2]`` as a prefix match
    (no trim — the RNN state at end-of-stored is the state M2 prefill needs).
    """
    prompt = list(range(1000, 1100))
    response_1 = [9001, 9002]
    new_msg = list(range(2000, 2050))

    reuse_cache.store(prompt + response_1, _hybrid_cache())

    turn_2 = prompt + response_1 + new_msg
    result, remaining = reuse_cache.fetch(turn_2)

    assert result is not None, "Prefix-extension is trim-free — must hit"
    assert remaining == new_msg


def test_hybrid_trim_paths_still_refused_when_enabled(reuse_cache):
    """The opt-in must NOT re-open the trim-requiring paths: a shorter request
    against a longer stored hybrid entry (supersequence-with-excess, then LCP
    fallback) still misses."""
    long_stored = list(range(1000, 1200))
    reuse_cache.store(long_stored, _hybrid_cache())

    short_request = list(range(1000, 1100))
    result, remaining = reuse_cache.fetch(short_request)

    assert result is None, "Trim-required reuse of hybrid state must still be refused"
    assert remaining == short_request


def test_hybrid_divergent_lcp_still_refused_when_enabled(reuse_cache):
    """Same-prefix-different-suffix (LCP shape) requires trimming the stored
    entry — still refused for hybrid entries under the opt-in."""
    prompt = list(range(1000, 1100))
    reuse_cache.store(prompt + [9001, 9002], _hybrid_cache())

    divergent = prompt + [9003, 9004]
    result, _remaining = reuse_cache.fetch(divergent)

    assert result is None, "LCP reuse trims the stored entry — must stay refused"


def test_hybrid_bound_is_enforced(reuse_cache):
    """Storing more hybrid entries than the bound LRU-evicts the oldest —
    the #1025 unbounded unique-superset accumulation cannot recur."""
    chain_a = list(range(1000, 1100))
    chain_b = list(range(2000, 2100))
    chain_c = list(range(3000, 3100))

    reuse_cache.store(chain_a, _hybrid_cache())
    reuse_cache.store(chain_b, _hybrid_cache())
    reuse_cache.store(chain_c, _hybrid_cache())

    stats = reuse_cache.get_stats()
    assert stats["non_trimmable_entries"] == 2, "Bound of 2 must hold"
    assert tuple(chain_a) not in reuse_cache._entries, "Oldest chain evicted"
    assert tuple(chain_b) in reuse_cache._entries
    assert tuple(chain_c) in reuse_cache._entries
    assert stats["evictions"] >= 1


def test_hybrid_bound_evicts_by_recency_not_insertion_order(reuse_cache):
    """#1103 codex NIT-3: the bound is LRU, not FIFO — a cache HIT must
    refresh an entry's recency so it survives a later eviction.

    ``test_hybrid_bound_is_enforced`` only ever stores (never fetches), so
    insertion order == recency order there and it can't distinguish LRU from
    FIFO. Here we store chain_a then chain_b, then FETCH chain_a (an exact
    trim-free hit that must bump it to most-recent), then store chain_c. Under
    LRU the least-recently-used is now chain_b, so chain_b — NOT the
    first-inserted chain_a — must be the one evicted.
    """
    chain_a = list(range(1000, 1100))
    chain_b = list(range(2000, 2100))
    chain_c = list(range(3000, 3100))

    reuse_cache.store(chain_a, _hybrid_cache())
    reuse_cache.store(chain_b, _hybrid_cache())

    # Exact-match hit on chain_a — trim-free, so a retained hybrid entry
    # serves it AND its recency is refreshed to most-recently-used.
    result, remaining = reuse_cache.fetch(chain_a)
    assert result is not None, "Exact match on chain_a must hit (trim-free)"
    assert remaining == []

    # Now the LRU order is [chain_b (oldest), chain_a (newest)]. Storing
    # chain_c pushes the count to 3 > bound 2, so the OLDEST (chain_b) goes.
    reuse_cache.store(chain_c, _hybrid_cache())

    stats = reuse_cache.get_stats()
    assert stats["non_trimmable_entries"] == 2, "Bound of 2 must hold"
    assert tuple(chain_b) not in reuse_cache._entries, (
        "chain_b was least-recently-used and must be evicted (LRU, not FIFO)"
    )
    assert tuple(chain_a) in reuse_cache._entries, (
        "chain_a was refreshed by its fetch hit and must survive"
    )
    assert tuple(chain_c) in reuse_cache._entries


def test_hybrid_bound_does_not_evict_dense_entries(reuse_cache):
    """The hybrid bound only ever evicts non-trimmable entries — dense
    KV-only entries are invisible to it."""
    dense_key = list(range(500, 600))
    reuse_cache.store(dense_key, _dense_cache())

    for base in (1000, 2000, 3000, 4000):
        reuse_cache.store(list(range(base, base + 100)), _hybrid_cache())

    assert tuple(dense_key) in reuse_cache._entries, (
        "Dense entry must survive hybrid-bound evictions"
    )
    assert reuse_cache.get_stats()["non_trimmable_entries"] == 2


def test_default_config_keeps_drop_policy():
    """No config change → byte-for-byte #1075 behavior (drop at store)."""
    config = MemoryCacheConfig(max_memory_mb=10, max_entries=64)
    c = MemoryAwarePrefixCache(MagicMock(), config)

    assert c.store(list(range(1000, 1100)), _hybrid_cache()) is False
    assert c.get_stats()["non_trimmable_skips"] == 1
    assert c.get_stats()["non_trimmable_entries"] == 0


# ---------------------------------------------------------------------------
# #1103: recurrent-state byte accounting. ``ArraysCache.state`` is an N-array
# list and ``CacheList.state`` is nested — both previously slipped through the
# ``keys, values = state`` unpack and contributed 0 bytes to the eviction
# ledger, so the very entries #1025 needed the budget to see were invisible
# to it.
# ---------------------------------------------------------------------------


def test_arrayscache_state_bytes_are_counted():
    from vllm_mlx.memory_cache import estimate_kv_cache_memory

    # 3 state arrays (not the (keys, values) pair shape) × 100 bytes.
    layer = ArraysCacheLayer(n_arrays=3, nbytes_each=100)
    assert estimate_kv_cache_memory([layer]) == 300


def test_cachelist_nested_state_bytes_are_counted():
    from vllm_mlx.memory_cache import estimate_kv_cache_memory

    # CacheList wrapping two ArraysCaches: nested state lists must be
    # recursed into, not unpacked as (keys, values) and counted as 0.
    layer = CacheListLayer(
        ArraysCacheLayer(n_arrays=2, nbytes_each=100),
        ArraysCacheLayer(n_arrays=2, nbytes_each=100),
    )
    assert estimate_kv_cache_memory([layer]) == 400


def test_hybrid_entry_memory_reaches_the_ledger(reuse_cache):
    """A stored hybrid entry's recurrent-state bytes must appear in
    ``_current_memory`` so LRU / pressure eviction can actually act on it."""
    prompt = list(range(1000, 1100))
    cache_layers = [TrimmableLayer(), ArraysCacheLayer(n_arrays=2, nbytes_each=500)]

    assert reuse_cache.store(prompt, cache_layers) is True
    entry = reuse_cache._entries[tuple(prompt)]
    assert entry.non_trimmable is True
    assert entry.memory_bytes >= 1000, (
        "ArraysCache state bytes must be included in the entry's ledger size"
    )


# ---------------------------------------------------------------------------
# #1103 codex BLOCKING-1: the hybrid bound must be enforced on the
# PERSISTENT-LOAD path too, not only at store time. A snapshot written while
# the opt-in was enabled can be reloaded on restart under a DIFFERENT (or
# absent) ``hybrid_reuse_max_entries``. Before the fix, loaded entries were
# flagged non-trimmable but NEVER subjected to the bound — so a restart could
# retain entries ABOVE N, or retain them at all when N == 0, breaking BOTH the
# opt-in and the bounded guarantee (the "default is byte-for-byte unchanged"
# property). These tests use REAL mlx-lm cache objects so the save/load round-
# trip through safetensors is exercised end-to-end, not mocked.
# ---------------------------------------------------------------------------


def _real_hybrid_cache(seqlen: int = 8):
    """A real (KVCache + ArraysCache) hybrid cache populated with mx arrays so
    it survives ``save_prompt_cache`` / ``load_prompt_cache`` round-trips.

    KVCache is trimmable (transformer attention); ArraysCache is the non-
    trimmable recurrent state (GatedDeltaNet / Mamba) that the bound governs.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.update_and_fetch(
        mx.random.normal((1, 2, seqlen, 4)),
        mx.random.normal((1, 2, seqlen, 4)),
    )
    arr = ArraysCache(size=2)
    arr[0] = mx.random.normal((1, 4, 4))
    arr[1] = mx.random.normal((1, 4, 4))
    mx.eval(kv.state, arr.state)
    return [kv, arr]


def _persist_three_hybrid_entries(tmp_path) -> str:
    """Store 3 hybrid entries under a generous bound and save them to disk.
    Returns the snapshot directory. All 3 are retained on the source side so
    the RELOAD side is what exercises the bound."""
    src_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=9
    )
    src = MemoryAwarePrefixCache(MagicMock(), src_config)
    for base in (1000, 2000, 3000):
        assert src.store(list(range(base, base + 8)), _real_hybrid_cache()) is True
    assert src.get_stats()["non_trimmable_entries"] == 3

    cache_dir = str(tmp_path / "snap")
    assert src.save_to_disk(cache_dir) is True
    return cache_dir


def test_persistent_load_respects_hybrid_bound(tmp_path):
    """Reloading a snapshot of 3 hybrid entries with N=1 must retain only 1 —
    the bound is applied at commit time, exactly like the store path."""
    cache_dir = _persist_three_hybrid_entries(tmp_path)

    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    loaded = dst.load_from_disk(cache_dir, replace=True)

    stats = dst.get_stats()
    assert stats["non_trimmable_entries"] == 1, (
        "Persistent load must LRU-trim hybrid entries down to the bound"
    )
    assert len(dst._entries) == 1
    # The reported loaded count must reflect only SURVIVING entries, not the
    # 3 that were staged before the bound trimmed 2 away.
    assert loaded == 1


def test_persistent_load_drops_all_when_disabled(tmp_path):
    """Reloading with N=0 (the default / disabled state) must drop ALL non-
    trimmable entries — byte-for-byte the #1075 drop-at-store behavior applied
    after a restart. This is the core "default is unchanged" guarantee."""
    cache_dir = _persist_three_hybrid_entries(tmp_path)

    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=0
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    loaded = dst.load_from_disk(cache_dir, replace=True)

    stats = dst.get_stats()
    assert stats["non_trimmable_entries"] == 0, (
        "With reuse disabled, a restart must retain NO non-trimmable entries"
    )
    assert len(dst._entries) == 0
    assert loaded == 0


def test_persistent_load_keeps_trimmable_entries_when_disabled(tmp_path):
    """The disabled-state drop is scoped to NON-trimmable entries only — a
    persisted dense (all-KVCache) entry must still reload normally with N=0."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    src_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=9
    )
    src = MemoryAwarePrefixCache(MagicMock(), src_config)

    dense = KVCache()
    dense.update_and_fetch(
        mx.random.normal((1, 2, 8, 4)),
        mx.random.normal((1, 2, 8, 4)),
    )
    mx.eval(dense.state)
    dense_key = list(range(500, 508))
    assert src.store(dense_key, [dense]) is True
    # One hybrid entry too, so we can prove the drop is selective.
    src.store(list(range(1000, 1008)), _real_hybrid_cache())

    cache_dir = str(tmp_path / "snap")
    assert src.save_to_disk(cache_dir) is True

    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=0
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    dst.load_from_disk(cache_dir, replace=True)

    assert dst.get_stats()["non_trimmable_entries"] == 0
    assert tuple(dense_key) in dst._entries, (
        "Dense (trimmable) entries must survive an N=0 reload"
    )


def test_merge_load_into_full_hybrid_cache_counts_only_imported(tmp_path):
    """#1103 codex BLOCKING-1: merge-loading (replace=False) ONE new hybrid
    entry into a cache already AT the bound must report ``loaded == 1``.

    The bound pass runs at commit and, being LRU, evicts the PRE-EXISTING
    entry (older) to make room for the freshly imported one. That eviction
    must NOT be subtracted from the import's ``loaded`` tally — the old code
    subtracted every bound eviction and returned 0 (or corrupted the byte
    total) for a load that actually installed a new entry.
    """
    # Snapshot on disk: a single NEW hybrid entry to import.
    src_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=9
    )
    src = MemoryAwarePrefixCache(MagicMock(), src_config)
    assert src.store(list(range(7000, 7008)), _real_hybrid_cache()) is True
    cache_dir = str(tmp_path / "snap")
    assert src.save_to_disk(cache_dir) is True

    # Destination already holds a pre-existing hybrid entry and the bound is
    # N=1, so it is already FULL before the import.
    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    assert dst.store(list(range(9000, 9008)), _real_hybrid_cache()) is True
    assert dst.get_stats()["non_trimmable_entries"] == 1

    loaded = dst.load_from_disk(cache_dir, replace=False)

    # The imported entry survived the bound (it is the most-recent), the
    # pre-existing one was LRU-evicted — but that eviction belongs to the
    # destination, not this import, so the imported count stays 1.
    assert loaded == 1, (
        "Merge-load must count the surviving imported entry, not net it "
        "against the pre-existing entry the bound evicted"
    )
    assert dst.get_stats()["non_trimmable_entries"] == 1
    assert tuple(range(7000, 7008)) in dst._entries
    assert tuple(range(9000, 9008)) not in dst._entries
    # loaded_bytes ledger must stay coherent (non-negative, matches survivor).
    assert dst._last_load_bytes > 0
