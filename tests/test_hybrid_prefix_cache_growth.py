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

pytestmark = pytest.mark.requires_mlx

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
    ["RotatingKVCache", "QuantizedKVCache"],
)
def test_dict_form_trimmable_kv_variants_still_stored(cache, kv_class):
    """Dict-form sliding-window / other genuinely-trimmable KV classes must NOT
    be dropped.

    Regression for codex #1075 finding: an allowlist of only KVCache would
    wrongly classify RotatingKVCache & friends as non-trimmable and drop the
    entry, regressing prefix reuse for dense / sliding-window models. The
    denylist keeps them cacheable. (``ChunkedKVCache`` / ``ConcatenateKVCache``
    are the exception — they LIE about trimmability and are dropped; see
    ``test_dict_form_trim_unsafe_liar_variants_dropped``.)
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


@pytest.mark.parametrize("kv_class", ["ChunkedKVCache", "ConcatenateKVCache"])
def test_dict_form_trim_unsafe_liar_variants_dropped(cache, kv_class):
    """Dict-form trim-unsafe "trimmable liar" classes MUST be dropped.

    ``ChunkedKVCache`` (drops front history via ``maybe_trim_front``) and
    ``ConcatenateKVCache`` (``trim`` never slices its buffers) both report
    ``is_trimmable()==True`` yet corrupt on a trim-then-continue, so trimming
    them back to a prefix produces wrong KV. The store gate must refuse them —
    see ``_TRIM_UNSAFE_CACHE_CLASSES`` in ``memory_cache``. Neither is reachable
    by a currently-supported family (llama4 / afm7, not in ``aliases.json``);
    this is defense-in-depth for the latent gap.
    """
    prompt = list(range(1000, 1100))
    dict_cache = [
        {"class_name": kv_class, "state": (1, 2), "meta_state": ("0",)},
        {"class_name": kv_class, "state": (3, 4), "meta_state": ("0",)},
    ]

    assert cache.store(prompt, dict_cache) is False, (
        f"{kv_class} lies about trimmability and must be dropped"
    )
    assert tuple(prompt) not in cache._entries


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


def test_hybrid_exact_match_without_shorter_snapshot_misses(reuse_cache):
    """An exact recurrent-state entry cannot kick off generation.

    BatchGenerator re-forwards the final prompt token, so the cached state
    would first need to trim one token.  With no shorter snapshot available,
    correctness requires a clean miss/full prefill.
    """
    prompt = list(range(1000, 1100))
    reuse_cache.store(prompt, _hybrid_cache())

    result, remaining = reuse_cache.fetch(prompt)

    assert result is None
    assert remaining == prompt


def test_hybrid_exact_match_falls_back_to_longest_strict_prefix(reuse_cache):
    """An unusable exact hit must not mask a resumable boundary snapshot."""
    prompt = list(range(1000, 1100))
    boundary = prompt[:-7]
    reuse_cache.store(boundary, _hybrid_cache(), evict_prefixes=False)
    reuse_cache.store(prompt, _hybrid_cache(), evict_prefixes=False)

    result, remaining = reuse_cache.fetch(prompt)

    assert result is not None
    assert remaining == prompt[-7:]
    assert reuse_cache._last_match_type == "prefix"


def test_radix_hybrid_exact_match_falls_back_to_longest_strict_prefix():
    """The radix fast path must walk past an unusable exact terminal."""
    from vllm_mlx.runtime.radix_index import RadixPrefixIndex

    config = MemoryCacheConfig(
        max_memory_mb=10, max_entries=64, hybrid_reuse_max_entries=2
    )
    cache = MemoryAwarePrefixCache(MagicMock(), config, RadixPrefixIndex())
    prompt = list(range(1000, 1100))
    boundary = prompt[:-7]
    cache.store(boundary, _hybrid_cache(), evict_prefixes=False)
    cache.store(prompt, _hybrid_cache(), evict_prefixes=False)

    result, remaining = cache.fetch(prompt)

    assert result is not None
    assert remaining == prompt[-7:]
    assert cache._last_match_type == "prefix"


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


def test_hybrid_bound_does_not_promote_unusable_exact_entry(reuse_cache):
    """An exact snapshot that serves no tokens must not gain LRU priority.

    Otherwise repeated exact requests can protect an unusable full-prompt
    snapshot while evicting the older message-boundary snapshot that actually
    avoids full prefill.
    """
    chain_a = list(range(1000, 1100))
    chain_b = list(range(2000, 2100))
    chain_c = list(range(3000, 3100))

    reuse_cache.store(chain_a, _hybrid_cache())
    reuse_cache.store(chain_b, _hybrid_cache())

    # BatchGenerator cannot consume this exact non-trimmable state because it
    # re-forwards the final prompt token. The miss must not refresh recency.
    result, remaining = reuse_cache.fetch(chain_a)
    assert result is None
    assert remaining == chain_a

    # The LRU order remains [chain_a (oldest), chain_b (newest)]. Storing
    # chain_c pushes the count to 3 > bound 2, so chain_a must be evicted.
    reuse_cache.store(chain_c, _hybrid_cache())

    stats = reuse_cache.get_stats()
    assert stats["non_trimmable_entries"] == 2, "Bound of 2 must hold"
    assert tuple(chain_a) not in reuse_cache._entries
    assert tuple(chain_b) in reuse_cache._entries
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


def test_persistent_load_imports_are_protected_from_bound(tmp_path):
    """#1111 regression (PROTECTED-entry semantics): reloading a snapshot of 3
    hybrid entries with a LOW bound (N=1) must still retain ALL 3 — imported
    entries are PROTECTED (SGLang ``lock_ref`` / vLLM ``ref_cnt`` idiom) and
    exempt from the opportunistic retention bound, which governs live-store
    entries only. The bound would otherwise LRU-trim a reloaded snapshot below
    what the operator explicitly imported."""
    cache_dir = _persist_three_hybrid_entries(tmp_path)

    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    loaded = dst.load_from_disk(cache_dir, replace=True)

    stats = dst.get_stats()
    assert stats["non_trimmable_entries"] == 3, (
        "Imported (protected) entries must survive the retention bound"
    )
    assert len(dst._entries) == 3
    assert loaded == 3
    # All installed entries carry the protected marker.
    assert all(e.protected for e in dst._entries.values())


def test_persistent_load_retains_all_when_disabled(tmp_path):
    """#1111 regression: reloading with N=0 (the default / disabled state) is an
    EXPLICIT import, so it must RETAIN every imported non-trimmable entry — the
    retention bound does not gate an operator's deliberate disk import (#476).

    ``hybrid_reuse_max_entries`` governs opportunistic within-session prefix
    reuse at STORE time; N=0 disables that opportunistic retention. It does NOT
    mean "reject an explicit disk import". #1111 wrongly gated the load commit
    on N <= 0, so a default-config import dropped its own entries and reported
    ``entries_loaded == 0``. The store-path #1075 anti-leak drop is unaffected —
    the live-store loop still drops non-trimmable entries at N=0 (guarded by
    ``test_hybrid_store_is_dropped`` / ``test_default_config_keeps_drop_policy``
    above)."""
    cache_dir = _persist_three_hybrid_entries(tmp_path)

    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=0
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    loaded = dst.load_from_disk(cache_dir, replace=True)

    stats = dst.get_stats()
    assert stats["non_trimmable_entries"] == 3, (
        "An explicit import must retain all imported non-trimmable entries "
        "even with opportunistic retention disabled (N=0)"
    )
    assert len(dst._entries) == 3
    assert loaded == 3


def test_persistent_load_keeps_both_kinds_when_disabled(tmp_path):
    """#1111 regression: an explicit import with N=0 must reload BOTH a dense
    (trimmable KVCache) entry AND a hybrid (non-trimmable ArraysCache) entry —
    the retention bound gates neither on the explicit disk-import path."""
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
    # One hybrid entry too, so we can prove the reload covers both kinds.
    hybrid_key = list(range(1000, 1008))
    src.store(hybrid_key, _real_hybrid_cache())

    cache_dir = str(tmp_path / "snap")
    assert src.save_to_disk(cache_dir) is True

    dst_config = MemoryCacheConfig(
        max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=0
    )
    dst = MemoryAwarePrefixCache(MagicMock(), dst_config)
    dst.load_from_disk(cache_dir, replace=True)

    assert dst.get_stats()["non_trimmable_entries"] == 1, (
        "The hybrid (non-trimmable) entry must survive an N=0 explicit import"
    )
    assert tuple(dense_key) in dst._entries, (
        "Dense (trimmable) entries must survive an N=0 reload"
    )
    assert tuple(hybrid_key) in dst._entries, (
        "Hybrid (non-trimmable) entries must survive an N=0 explicit import"
    )


def test_store_after_import_evicts_oldest_opportunistic_not_import(tmp_path):
    """#1111 regression: with a protected import present, live-storing MORE
    opportunistic hybrid entries than the bound allows must GENUINELY evict the
    OLDEST opportunistic (unprotected) entry — never the protected import.

    Ports SGLang's protected/evictable split: the retention bound acts on the
    evictable (opportunistic live-store #1075) set ONLY. This test forces real
    eviction: under N=2 it stores THREE opportunistic entries, so the 3rd store
    pushes the evictable set over the bound and the enforcer LRU-evicts the
    oldest opportunistic entry. The protected import is excluded from the
    candidate set (SGLang ``evictable_leaves`` skips ``lock_ref > 0`` nodes), so
    it survives while the bound still holds at N=2 over opportunistic entries.
    """
    # Snapshot on disk: a single NEW hybrid entry to import (becomes protected).
    src = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=9
        ),
    )
    assert src.store(list(range(7000, 7008)), _real_hybrid_cache()) is True
    cache_dir = str(tmp_path / "snap")
    assert src.save_to_disk(cache_dir) is True

    dst = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=2
        ),
    )
    # Import first: the protected entry.
    assert dst.load_from_disk(cache_dir, replace=False) == 1
    import_key = tuple(range(7000, 7008))
    assert dst._entries[import_key].protected

    # Now store THREE opportunistic entries under the N=2 bound. Each store
    # fires the enforcer; the 3rd store pushes the EVICTABLE set to 3 > 2, so
    # the oldest opportunistic entry (9000) is genuinely LRU-evicted.
    assert dst.store(list(range(9000, 9008)), _real_hybrid_cache()) is True
    assert dst.store(list(range(9100, 9108)), _real_hybrid_cache()) is True
    assert dst.store(list(range(9200, 9208)), _real_hybrid_cache()) is True

    # Eviction genuinely fired: the oldest opportunistic entry is GONE.
    assert tuple(range(9000, 9008)) not in dst._entries, (
        "the 3rd opportunistic store must LRU-evict the oldest one (bound=2)"
    )
    # The two newest opportunistic entries remain (bound=2 over evictable set).
    assert tuple(range(9100, 9108)) in dst._entries
    assert tuple(range(9200, 9208)) in dst._entries
    # The PROTECTED import survives the eviction that just fired.
    assert import_key in dst._entries, (
        "the protected import must NOT be evicted by an opportunistic store"
    )
    assert dst._entries[import_key].protected
    # Bound is enforced over the EVICTABLE set only: 2 opportunistic + 1
    # protected import = 3 non-trimmable total, but only 2 are evictable.
    assert dst.get_stats()["non_trimmable_entries"] == 3
    evictable = sum(
        1 for e in dst._entries.values() if e.non_trimmable and not e.protected
    )
    assert evictable == 2


def test_enforcer_never_evicts_protected_imports(tmp_path):
    """The enforcer's victim list must NEVER contain a protected (imported) key,
    even when the bound is far below the number of imported entries. This is the
    direct invariant the load-commit reconciliation relies on: because staged
    imports are protected, they are never victims, so the ``loaded`` tally is
    never spuriously netted down.

    Imports 3 protected entries under N=1 and asserts the enforcer, invoked
    directly, evicts none of them (returns an empty victim list)."""
    cache_dir = _persist_three_hybrid_entries(tmp_path)
    dst = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
        ),
    )
    assert dst.load_from_disk(cache_dir, replace=True) == 3
    assert all(e.protected for e in dst._entries.values())

    # Directly invoke the enforcer: the evictable (unprotected) candidate set is
    # empty, so nothing is evicted regardless of the tight N=1 bound.
    with dst._lock:
        victims = dst._enforce_hybrid_bound_locked()
    assert victims == []
    assert dst.get_stats()["non_trimmable_entries"] == 3


def test_load_seam_imports_protected_at_any_bound(tmp_path):
    """#1111 regression guard — pins the exact seam of the fix: the SAME 3-entry
    imported snapshot retains ALL 3 under BOTH N=0 (disabled) and a low N=1
    bound, because imported entries are PROTECTED and exempt from the
    opportunistic retention bound at every N.

    Mutation checks:
    * The #1111 store-parity behavior (drop-on-load at N<=0) would make the N=0
      arm keep 0 -> fails here.
    * A "bound also trims imports at N>0" behavior would make the N=1 arm keep 1
      -> fails here.
    Only the protected/evictable split (imports never counted against the bound)
    passes both arms."""
    cache_dir = _persist_three_hybrid_entries(tmp_path)

    disabled = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=0
        ),
    )
    assert disabled.load_from_disk(cache_dir, replace=True) == 3
    assert disabled.get_stats()["non_trimmable_entries"] == 3

    low_bound = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
        ),
    )
    assert low_bound.load_from_disk(cache_dir, replace=True) == 3
    assert low_bound.get_stats()["non_trimmable_entries"] == 3


def test_import_survives_a_later_unrelated_live_store(tmp_path):
    """#1111 codex BLOCKING (load-then-store mutation-kill): the CORE bug was
    that an imported non-trimmable entry survived the load MOMENT but the next
    ordinary live ``store`` that fired the retention enforcer LRU-evicted it —
    so an import survived ZERO subsequent requests.

    With the protected-entry fix the imported entry is exempt from the enforcer
    at the LIVE-STORE call site too, so it survives an unrelated later store.
    Exercised under N>0 (where a fresh opportunistic hybrid store IS admitted
    and DOES fire the enforcer at the store call site — the only config where
    the enforcer can run after an import). Reverting the protected exclusion in
    ``_enforce_hybrid_bound_locked`` makes this assertion fail."""
    # Import a hybrid entry (protected) into an N=1 cache.
    src = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=9
        ),
    )
    assert src.store(list(range(1000, 1008)), _real_hybrid_cache()) is True
    cache_dir = str(tmp_path / "snap")
    assert src.save_to_disk(cache_dir) is True

    dst = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
        ),
    )
    assert dst.load_from_disk(cache_dir, replace=True) == 1
    imported_key = tuple(range(1000, 1008))
    assert imported_key in dst._entries
    assert dst._entries[imported_key].protected

    # An UNRELATED later live store of a fresh opportunistic hybrid entry. At
    # N=1 this IS admitted (bypasses the N<=0 store gate) and fires the enforcer
    # at the store call site over the whole non-trimmable set.
    assert dst.store(list(range(2000, 2008)), _real_hybrid_cache()) is True

    # The imported (protected) entry MUST still be present — it survived the
    # enforcer that ran during the unrelated store.
    assert imported_key in dst._entries, (
        "Imported entry was wiped by a later unrelated live store — the "
        "protected exclusion regressed (#1111 codex BLOCKING)"
    )
    # Both survive: the protected import + the 1 opportunistic entry within N=1.
    assert dst.get_stats()["non_trimmable_entries"] == 2
    assert tuple(range(2000, 2008)) in dst._entries


def test_startup_reload_cycle_does_not_grow_protected_set(tmp_path):
    """#1111 codex r3 BLOCKING (restart-cycle growth) — mutation-kill.

    The disk-load path is reached by BOTH the explicit HTTP import AND the
    process-restart startup auto-load. ``save_to_disk`` persists ALL live
    entries — including opportunistic (unprotected) non-trimmable ones. If the
    startup reload marked them protected (as the first fix did unconditionally),
    the protected set would grow ~N every restart and defeat the
    ``hybrid_reuse_max_entries`` cap: shutdown persists N opportunistic -> boot
    reloads them protected -> new opportunistic added within the bound ->
    persisted -> protected next boot -> unbounded.

    The fix threads ``protected_import`` and the STARTUP path passes ``False``,
    so reloaded non-trimmable entries stay UNPROTECTED and obey the bound at
    commit. This test simulates K shutdown/reload cycles under N and asserts the
    retained non-trimmable set NEVER exceeds N.

    Mutation-kill: revert the startup caller to ``protected_import=True`` (or
    hardcode ``protected=True`` at the commit site) and the count blows past N.
    """
    n = 2
    k_cycles = 5

    def _snapshot_of_live_cache(cache, out_dir):
        assert cache.save_to_disk(out_dir) is True

    def _hybrid_keys(cache):
        # Non-trimmable keys in LRU order (OrderedDict insertion order).
        return [key for key, e in cache._entries.items() if e.non_trimmable]

    # Persist an initial snapshot of THREE opportunistic hybrid entries (stored
    # under a generous bound so they all land on disk unprotected). Three (> n)
    # so the reload genuinely has to trim, not merely fit.
    seed = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=200, max_entries=64, hybrid_reuse_max_entries=9
        ),
    )
    for base in (100, 200, 300):
        assert seed.store(list(range(base, base + 8)), _real_hybrid_cache())
    snap = str(tmp_path / "snap")
    _snapshot_of_live_cache(seed, snap)
    # Track the persisted hybrid-key order in Python state (mirrors what is on
    # disk). save_to_disk writes _entries in LRU order, so this is the exact
    # persisted order the reload will see. On cycle 0 = the 3-entry seed.
    persisted_keys = _hybrid_keys(seed)
    assert len(persisted_keys) == 3

    # Simulate restart cycles: each boot reloads the persisted snapshot as the
    # STARTUP path does (protected_import=False), then live-stores one fresh
    # opportunistic entry, then persists the whole live cache for the next boot.
    for cycle in range(k_cycles):
        booted = MemoryAwarePrefixCache(
            MagicMock(),
            MemoryCacheConfig(
                max_memory_mb=200, max_entries=64, hybrid_reuse_max_entries=n
            ),
        )
        # The reload must retain EXACTLY min(persisted, n).
        expected_retained = min(len(persisted_keys), n)

        # STARTUP auto-load — reloaded entries must be UNPROTECTED.
        booted.load_from_disk(snap, replace=True, protected_import=False)

        # No reloaded entry may be protected (they are opportunistic on disk).
        assert not any(e.protected for e in booted._entries.values()), (
            "startup-reloaded entries must be UNPROTECTED (protected_import=False)"
        )

        reloaded_keys = _hybrid_keys(booted)
        # (a) NOT "keeps too many" (growth): retained == min(persisted, n).
        # (b) NOT "keeps nothing" (discard-all): expected_retained is > 0 here,
        #     so a reload that dropped everything would violate this equality.
        assert len(reloaded_keys) == expected_retained, (
            f"cycle {cycle}: reload retained {len(reloaded_keys)} non-trimmable, "
            f"expected exactly min(persisted={len(persisted_keys)}, n={n})="
            f"{expected_retained} — either grew (protected bug) or discarded all"
        )
        # (c) The survivors are the MOST-RECENT n persisted keys (LRU keeps the
        #     tail of the persisted order). Pins WHICH entries survived, not just
        #     the count, so a reload that keeps the wrong (or zero) set fails.
        assert reloaded_keys == persisted_keys[-expected_retained:], (
            f"cycle {cycle}: survivors {reloaded_keys} are not the most-recent "
            f"{expected_retained} persisted keys {persisted_keys[-expected_retained:]}"
        )

        # A fresh opportunistic hybrid request this session.
        booted.store(
            list(range(1000 + cycle * 10, 1000 + cycle * 10 + 8)),
            _real_hybrid_cache(),
        )
        # THE INVARIANT after the fresh store: still bounded at N, never growing
        # across accumulated restart cycles.
        retained = booted.get_stats()["non_trimmable_entries"]
        assert retained == n, (
            f"cycle {cycle}: retained non-trimmable {retained} != bound {n} "
            "— the startup-reload protected-set growth bug regressed"
        )
        # Persist the live cache for the next boot (mirrors shutdown) and update
        # the tracked persisted order to what actually got written.
        _snapshot_of_live_cache(booted, snap)
        persisted_keys = _hybrid_keys(booted)


def test_production_startup_wiring_passes_protected_import_false(tmp_path):
    """#1111 codex r4 BLOCKING-1: pin the PRODUCTION startup wiring, not just
    the ``load_from_disk`` primitive.

    The growth-prevention guarantee only holds if the real startup entry point
    ``runtime.cache.load_prefix_cache_from_disk()`` actually passes
    ``protected_import=False`` down to the engine. The lower-level tests call
    ``load_from_disk(..., protected_import=False)`` directly, so they stay green
    even if ``runtime/cache.py`` stops passing ``False``. This test drives the
    real production call site with a mocked engine and asserts the kwarg.

    Mutation-kill: delete ``protected_import=False`` from
    ``runtime/cache.py::load_prefix_cache_from_disk`` and this test FAILS.
    """
    from unittest.mock import MagicMock as _MagicMock
    from unittest.mock import patch

    import vllm_mlx.runtime.cache as runtime_cache

    fake_engine = _MagicMock()
    fake_engine.load_cache_from_disk.return_value = 0  # no entries; wiring only
    fake_cfg = _MagicMock()
    fake_cfg.engine = fake_engine

    with (
        patch.object(runtime_cache, "get_config", return_value=fake_cfg),
        patch.object(runtime_cache, "get_cache_dir", return_value=str(tmp_path)),
        # _load_radix_index_after_cache pokes the engine internals; the mock has
        # no real scheduler, so stub it out — this test is about the load kwarg.
        patch.object(runtime_cache, "_load_radix_index_after_cache"),
    ):
        runtime_cache.load_prefix_cache_from_disk()

    fake_engine.load_cache_from_disk.assert_called_once()
    _args, kwargs = fake_engine.load_cache_from_disk.call_args
    assert kwargs.get("protected_import") is False, (
        "startup auto-load must call load_cache_from_disk(protected_import=False) "
        "— otherwise reloaded opportunistic entries are immortalized as protected "
        "and the retention bound grows unbounded across restarts"
    )


def test_startup_reload_obeys_bound_while_explicit_import_pins(tmp_path):
    """#1111 codex r3: the two callers must be treated DIFFERENTLY.

    Same on-disk snapshot of 3 hybrid entries:
    * ``protected_import=False`` (startup) under N=1 -> reloaded non-trimmable
      entries obey the bound -> at most 1 retained.
    * ``protected_import=True`` (explicit #476 import) under the same N=1 ->
      all 3 pinned, exempt from the bound.
    """
    cache_dir = _persist_three_hybrid_entries(tmp_path)

    startup = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
        ),
    )
    startup.load_from_disk(cache_dir, replace=True, protected_import=False)
    assert startup.get_stats()["non_trimmable_entries"] <= 1, (
        "startup auto-load must obey the retention bound"
    )
    assert not any(e.protected for e in startup._entries.values())

    explicit = MemoryAwarePrefixCache(
        MagicMock(),
        MemoryCacheConfig(
            max_memory_mb=100, max_entries=64, hybrid_reuse_max_entries=1
        ),
    )
    assert explicit.load_from_disk(cache_dir, replace=True, protected_import=True) == 3
    assert explicit.get_stats()["non_trimmable_entries"] == 3
    assert all(e.protected for e in explicit._entries.values())
