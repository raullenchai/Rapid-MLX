"""Trim-free prefix reuse generalized to sliding-window
(RotatingKVCache) models — cache-level correctness proofs.

These are the ground-truth checks behind the correctness verdict. They run entirely
offline against mlx-lm's real ``RotatingKVCache`` (no engine boot, no network)
and assert the properties the reuse path depends on:

1. A rotated ``RotatingKVCache`` reused via the trim-free PREFIX-EXTENSION path
   (deepcopy at store/fetch, then append the continuation) yields a byte-exact
   window vs a cold full-prefill — even after the ring has rotated many times.

2. ``RotatingKVCache.trim(1)`` (the exact-hit compensation the scheduler uses to
   keep a warm exact-repeat byte-equal to cold) does NOT correctly un-write the
   last token once the buffer has rotated. This is WHY the scheduler must fall
   back to a full re-prefill on a non-trimmable exact hit rather than the
   ``trim(1)`` path (see ``scheduler.py`` exact-hit branch).

3. #1103's ``hybrid_reuse_max_entries`` gate is class-agnostic: it retains a
   rotated ``RotatingKVCache`` entry and serves it on exact + prefix-extension
   fetches, exactly as it does for hybrid recurrent-state entries.
"""

import copy

import mlx.core as mx
import pytest
from mlx_lm.models.cache import (
    ChunkedKVCache,
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
)

from vllm_mlx.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _cache_has_non_trimmable,
    _layer_forbids_trim,
    _layer_is_trim_liar,
)

B, H, D = 1, 2, 8
W = 16  # small sliding window to force rotation cheaply


def _temporal(rot: RotatingKVCache):
    k = rot._temporal_order(rot.keys)
    v = rot._temporal_order(rot.values)
    n = min(rot.offset, rot.max_size)
    return k[..., -n:, :], v[..., -n:, :]


@pytest.mark.parametrize("P,C", [(10, 5), (40, 5), (40, 30), (17, 3), (3000, 50)])
def test_rotating_prefix_extension_is_byte_exact(P, C):  # noqa: N803
    """Deepcopy(store)->append(remaining) == cold full-prefill, rotated or not."""
    mx.random.seed(0)
    fk = mx.random.normal((B, H, P + C, D))
    fv = mx.random.normal((B, H, P + C, D))

    cold = RotatingKVCache(max_size=W, keep=0)
    cold.update_and_fetch(fk, fv)
    ck, cv = _temporal(cold)

    warm = RotatingKVCache(max_size=W, keep=0)
    warm.update_and_fetch(fk[..., :P, :], fv[..., :P, :])
    resumed = copy.deepcopy(copy.deepcopy(warm))  # store then fetch
    resumed.update_and_fetch(fk[..., P:, :], fv[..., P:, :])
    wk, wv = _temporal(resumed)

    mx.eval(ck, cv, wk, wv)
    assert ck.shape == wk.shape
    assert float(mx.max(mx.abs(ck - wk))) == 0.0
    assert float(mx.max(mx.abs(cv - wv))) == 0.0


def test_rotated_rotating_cache_is_non_trimmable():
    """Once rotated, is_trimmable() is False -> classified non-trimmable ->
    can_trim_prompt_cache() is False -> the exact-hit trim(1) is unavailable."""
    rot = RotatingKVCache(max_size=W, keep=0)
    k = mx.random.normal((B, H, W + 8, D))
    rot.update_and_fetch(k, k)
    assert rot.offset >= rot.max_size
    assert rot.is_trimmable() is False
    assert can_trim_prompt_cache([KVCache(), rot]) is False
    assert _cache_has_non_trimmable([KVCache(), rot]) is True


def test_trim1_cannot_undo_last_token_on_rotated_cache():
    """The scheduler's exact-hit compensation (trim then re-forward last token)
    is INVALID for a rotated cache: trim(1) only decrements offset/_idx, it does
    not un-write the rotated slot, so re-forwarding drifts. This is the reason
    the scheduler falls back to a full re-prefill for non-trimmable exact hits."""
    mx.random.seed(3)
    P = 40  # > W -> rotated
    fk = mx.random.normal((B, H, P + 1, D))
    fv = mx.random.normal((B, H, P + 1, D))

    cold = RotatingKVCache(max_size=W, keep=0)
    cold.update_and_fetch(fk[..., :P, :], fv[..., :P, :])
    cold.update_and_fetch(fk[..., P : P + 1, :], fv[..., P : P + 1, :])
    ck, _ = _temporal(cold)

    warm = RotatingKVCache(max_size=W, keep=0)
    warm.update_and_fetch(fk[..., :P, :], fv[..., :P, :])
    resumed = copy.deepcopy(warm)
    resumed.trim(1)  # force the compensation even though can_trim is False
    resumed.update_and_fetch(fk[..., P - 1 : P, :], fv[..., P - 1 : P, :])
    resumed.update_and_fetch(fk[..., P : P + 1, :], fv[..., P : P + 1, :])
    wk, _ = _temporal(resumed)

    mx.eval(ck, wk)
    # forced trim(1) still MISMATCHES on the rotated buffer -> proves the
    # full-prefill fallback is required for correctness.
    assert float(mx.max(mx.abs(ck - wk))) > 1e-3


def test_hybrid_gate_retains_and_serves_rotated_sliding_window_entry():
    """#1103's knob is class-agnostic: N>0 retains a rotated RotatingKVCache
    entry and serves prefix extensions; an exact request cannot reuse it
    because generation kickoff would require trim(1). N=0 drops it."""

    def _mixed(n):
        kv = KVCache()
        rot = RotatingKVCache(max_size=W, keep=0)
        k = mx.random.normal((B, H, n, D))
        v = mx.random.normal((B, H, n, D))
        kv.update_and_fetch(k, v)
        rot.update_and_fetch(k, v)
        mx.eval(kv.keys, kv.values, rot.keys, rot.values)
        return [kv, rot]

    toks = list(range(40))  # 40 > W -> rotated -> non-trimmable

    off = MemoryAwarePrefixCache(None, MemoryCacheConfig(hybrid_reuse_max_entries=0))
    assert off.store(toks, _mixed(40), evict_prefixes=False) is False  # dropped

    on = MemoryAwarePrefixCache(None, MemoryCacheConfig(hybrid_reuse_max_entries=4))
    assert on.store(toks, _mixed(40), evict_prefixes=False) is True  # retained

    exact, remaining = on.fetch(toks)
    assert exact is None
    assert on._last_match_type == "miss"
    assert remaining == toks

    ext, remaining = on.fetch(toks + [40, 41, 42])
    assert ext is not None
    assert on._last_match_type == "prefix"
    assert remaining == [40, 41, 42]


def test_scheduler_exact_hit_nontrimmable_drops_cache_and_full_prefills():
    """scheduler._resolve_exact_hit_tokens: a non-trimmable (rotated
    sliding-window) exact hit must DISCARD the reused cache and full-prefill,
    so the first generated token matches a cold request. Deleting the fallback
    branch in the scheduler turns this test red (mutation-kill for the fix)."""
    from unittest.mock import MagicMock

    from vllm_mlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)  # bypass __init__

    rot = RotatingKVCache(max_size=W, keep=0)
    k = mx.random.normal((B, H, W + 8, D))
    rot.update_and_fetch(k, k)
    mx.eval(rot.keys, rot.values)
    assert can_trim_prompt_cache([KVCache(), rot]) is False  # precondition

    req = MagicMock()
    req.prompt_token_ids = list(range(40))
    req.prompt_cache = [KVCache(), rot]
    req.cached_tokens = 40
    req.remaining_tokens = []
    req.request_id = "req-nontrim-000"

    tokens = sched._resolve_exact_hit_tokens(req)

    # fallback fired: cache discarded, request reset to a cold full prefill
    assert req.prompt_cache is None
    assert req.cached_tokens == 0
    assert req.remaining_tokens == list(range(40))
    assert tokens == list(range(40))


def test_scheduler_exact_hit_trimmable_trims_and_keeps_cache():
    """Control: a fully-trimmable cache takes the trim(1) path — cache retained,
    only the last token re-forwarded (offset decremented by 1)."""
    from unittest.mock import MagicMock

    from vllm_mlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)

    kv = KVCache()
    k = mx.random.normal((B, H, 6, D))
    kv.update_and_fetch(k, k)
    mx.eval(kv.keys, kv.values)
    assert can_trim_prompt_cache([kv]) is True  # precondition
    off_before = kv.offset

    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4, 5, 6]
    req.prompt_cache = [kv]
    req.cached_tokens = 6
    req.remaining_tokens = []
    req.request_id = "req-trim-0000"

    tokens = sched._resolve_exact_hit_tokens(req)

    # trim path: cache retained, offset trimmed by 1, only last token forwarded
    assert req.prompt_cache is not None
    assert kv.offset == off_before - 1
    assert tokens == [6]
    assert req.cached_tokens == 6  # unchanged


def test_scheduler_exact_hit_trim_exception_falls_back_to_cold_prefill(monkeypatch):
    """If trim inspection/execution RAISES, the helper must NOT proceed on top of
    an un-trimmed cache (that reintroduces the exact-hit drift this helper exists
    to prevent) — it must reset the request to a cold full prefill, exactly like
    the non-trimmable branch. Mutation-kill for the except-path reset: with the
    old 'log and continue' body, prompt_cache stays set and tokens == [last],
    so this test goes red."""
    from unittest.mock import MagicMock

    import mlx_lm.models.cache as _mlx_cache

    from vllm_mlx.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)

    def _boom(*_a, **_k):
        raise RuntimeError("simulated trim inspection failure")

    # The helper does `from mlx_lm.models.cache import can_trim_prompt_cache`
    # INSIDE the try at call time, so patching the module attribute makes the
    # in-function import resolve to the raising stub -> forces the except path.
    monkeypatch.setattr(_mlx_cache, "can_trim_prompt_cache", _boom)

    kv = KVCache()  # trimmable in principle, but inspection will raise first
    k = mx.random.normal((B, H, 6, D))
    kv.update_and_fetch(k, k)
    mx.eval(kv.keys, kv.values)

    req = MagicMock()
    req.prompt_token_ids = [1, 2, 3, 4, 5, 6]
    req.prompt_cache = [kv]
    req.cached_tokens = 6
    req.remaining_tokens = []
    req.request_id = "req-exc-00000"

    tokens = sched._resolve_exact_hit_tokens(req)

    # exception path fell back to a cold full prefill (no drift)
    assert req.prompt_cache is None
    assert req.cached_tokens == 0
    assert req.remaining_tokens == [1, 2, 3, 4, 5, 6]
    assert tokens == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# ChunkedKVCache trim-liar correctness lock (defense-in-depth; llama4-only,
# not in aliases.json — see ``_TRIM_UNSAFE_CACHE_CLASSES`` in memory_cache).
# ``ChunkedKVCache.is_trimmable()`` (inherited from KVCache) returns True
# UNCONDITIONALLY, but ``maybe_trim_front()`` drops front history and advances
# ``start_position``, so a trim-back-to-prefix slices past discarded tokens →
# wrong KV. These tests pin the gate that must REFUSE such reuse.
# ---------------------------------------------------------------------------


def _front_dropped_chunked(
    n_tokens: int = 200, chunk_size: int = 128
) -> ChunkedKVCache:
    """A REAL front-dropped ``ChunkedKVCache`` (``start_position > 0``).

    Fills the cache past ``chunk_size`` then calls ``maybe_trim_front()``
    exactly as ``llama4.py`` does, so the front is discarded and
    ``start_position`` advances — the state that makes a trim-back unsafe.
    """
    ck = ChunkedKVCache(chunk_size=chunk_size)
    k = mx.random.normal((B, H, n_tokens, D))
    v = mx.random.normal((B, H, n_tokens, D))
    ck.update_and_fetch(k, v)
    ck.maybe_trim_front()
    mx.eval(ck.keys, ck.values)
    assert ck.start_position > 0  # precondition: front actually dropped
    assert ck.is_trimmable() is True  # the LIE this fix defends against
    return ck


def test_front_dropped_chunked_kvcache_classified_non_trimmable():
    """A front-dropped ChunkedKVCache is classified non-trimmable by every gate
    even though its inherited ``is_trimmable()`` lies True. Without the fix
    ``_cache_has_non_trimmable`` returns False here (mutation-kill)."""
    ck = _front_dropped_chunked()

    assert _layer_is_trim_liar(ck) is True
    assert _layer_forbids_trim(ck) is True
    # mixed dense + chunked entry, mirroring llama4's per-group layout
    assert _cache_has_non_trimmable([KVCache(), ck]) is True


def test_fetch_supersequence_skips_front_dropped_chunked_entry():
    """A stored entry containing a front-dropped ChunkedKVCache must NOT be
    reused via the supersequence-with-excess trim path — it falls through to a
    miss (full prefill). Without the fix this returns a (corrupt) supersequence
    hit."""
    # hybrid_reuse_max_entries > 0 so store RETAINS the non-trimmable entry
    # (default 0 would drop it at store time); the fetch gate is what we test.
    cache = MemoryAwarePrefixCache(None, MemoryCacheConfig(hybrid_reuse_max_entries=8))
    long_tokens = list(range(60))
    kv = KVCache()
    fk = mx.random.normal((B, H, len(long_tokens), D))
    kv.update_and_fetch(fk, fk)
    mx.eval(kv.keys, kv.values)
    assert cache.store(long_tokens, [kv, _front_dropped_chunked()]) is True

    result, remaining = cache.fetch(list(range(20)))  # supersequence of stored
    assert result is None
    assert remaining == list(range(20))
    assert cache._last_match_type == "miss"


def test_fetch_lcp_skips_front_dropped_chunked_entry():
    """A stored entry containing a front-dropped ChunkedKVCache must NOT be
    reused via the LCP (divergent-suffix) trim path — it falls through to a
    miss. Without the fix this returns a (corrupt) LCP hit."""
    cache = MemoryAwarePrefixCache(None, MemoryCacheConfig(hybrid_reuse_max_entries=8))
    stored = list(range(20)) + [900, 901, 902, 903, 904]
    kv = KVCache()
    fk = mx.random.normal((B, H, len(stored), D))
    kv.update_and_fetch(fk, fk)
    mx.eval(kv.keys, kv.values)
    assert cache.store(stored, [kv, _front_dropped_chunked()]) is True

    # shares prefix [0..19] then diverges -> only the LCP path could match
    requested = list(range(20)) + [800, 801]
    result, remaining = cache.fetch(requested)
    assert result is None
    assert remaining == requested
    assert cache._last_match_type == "miss"


def test_fix_preserves_kvcache_reuse_and_rotated_skip():
    """Regression guard: the trim-liar gate changes ONLY the two liar classes.
    An all-KVCache supersequence entry is still trim-reused, and a rotated
    RotatingKVCache entry is still skipped — exactly as before the fix."""
    # (a) all-KVCache supersequence entry -> trimmable reuse retained
    kv_cache = MemoryAwarePrefixCache(None, MemoryCacheConfig())
    kv = KVCache()
    fk = mx.random.normal((B, H, 40, D))
    kv.update_and_fetch(fk, fk)
    mx.eval(kv.keys, kv.values)
    assert kv_cache.store(list(range(40)), [kv]) is True
    result, remaining = kv_cache.fetch(list(range(15)))
    assert result is not None
    assert remaining == []
    assert kv_cache._last_match_type == "supersequence"

    # (b) rotated RotatingKVCache -> still classified non-trimmable / skipped
    rot = RotatingKVCache(max_size=W, keep=0)
    rk = mx.random.normal((B, H, W + 8, D))
    rot.update_and_fetch(rk, rk)
    assert rot.is_trimmable() is False
    assert _layer_forbids_trim(rot) is True
    assert _cache_has_non_trimmable([KVCache(), rot]) is True


# ---------------------------------------------------------------------------
# Hybrid sliding/full per-layer TYPE STABILITY across a trimming fetch
# ---------------------------------------------------------------------------
# Regression coverage for the batched-generation crash on hybrid
# sliding/full-attention models (Muse-Glimmer-30B: 52 layers,
# ``[sliding, sliding, sliding, full]`` repeating). ``_trim_cache_offset``
# rebuilt EVERY keys/offset layer as a plain ``KVCache``, so a trim-requiring
# fetch (supersequence-with-excess / LCP) returned a uniform ``KVCache`` list
# for a mixed model. mlx-lm then merged those into a uniform ``BatchKVCache``,
# and extending that batch against a correctly-typed one
# (``BatchRotatingKVCache`` on the sliding layers) raised
# ``AttributeError: 'BatchKVCache' object has no attribute 'rotated'``.
#
# The tests above already prove the ROTATED case is refused. These cover the
# UNROTATED case, which is what actually shipped broken: the rotation gate
# passes it, so the trim helper must preserve the class.

_HYBRID_LAYER_TYPES = ["sliding", "sliding", "sliding", "full"] * 3


def _make_hybrid_cache(n_tokens: int, window: int = W):
    """A Muse-Glimmer-shaped mixed cache with ``n_tokens`` pushed through."""
    caches = [
        RotatingKVCache(max_size=window, keep=0) if t == "sliding" else KVCache()
        for t in _HYBRID_LAYER_TYPES
    ]
    for c in caches:
        x = mx.random.normal((B, H, n_tokens, D))
        c.update_and_fetch(x, x)
        mx.eval(c.keys, c.values)
    return caches


def test_trim_preserves_hybrid_per_layer_cache_types():
    """An UNROTATED hybrid cache keeps its per-layer classes across a trim."""
    from vllm_mlx.memory_cache import _trim_cache_offset

    cache = _make_hybrid_cache(n_tokens=8)
    before = [type(c).__name__ for c in cache]
    assert set(before) == {"RotatingKVCache", "KVCache"}, "fixture is not mixed"

    trimmed = _trim_cache_offset(cache, 2)
    assert trimmed is not None, "unrotated hybrid cache should be trimmable"
    assert [type(c).__name__ for c in trimmed] == before

    for original, tc in zip(cache, trimmed):
        assert tc.offset == 6
        if isinstance(tc, RotatingKVCache):
            # Sliding-window identity AND the ring write cursor must survive:
            # a stale ``_idx`` would append the continuation at the wrong slot.
            assert tc.max_size == original.max_size
            assert tc.keep == original.keep
            assert tc._idx == 6


def test_trim_does_not_mutate_the_retained_hybrid_entry():
    """Trimming returns copies — the cached entry stays reusable at full length."""
    from vllm_mlx.memory_cache import _trim_cache_offset

    cache = _make_hybrid_cache(n_tokens=8)
    assert _trim_cache_offset(cache, 2) is not None
    for c in cache:
        assert c.offset == 8
        if isinstance(c, RotatingKVCache):
            assert c._idx == 8


def test_trimmed_hybrid_cache_merges_to_same_batch_types_as_fresh():
    """HIT and MISS paths must agree per layer, or ``_extend_cache`` crashes.

    This is the assertion that fails with the pre-fix helper: the restored
    (trimmed) side merged to ``BatchKVCache`` on the sliding layers while a
    fresh sequence merged to ``BatchRotatingKVCache``, and ``extend`` then read
    ``other.rotated`` off a cache that has no such attribute.
    """
    from vllm_mlx.memory_cache import _trim_cache_offset

    restored = _trim_cache_offset(_make_hybrid_cache(n_tokens=8), 2)
    assert restored is not None
    fresh = [
        RotatingKVCache(max_size=W, keep=0) if t == "sliding" else KVCache()
        for t in _HYBRID_LAYER_TYPES
    ]

    restored_batch = [type(c).merge([c]) for c in restored]
    fresh_batch = [type(c).merge([c]) for c in fresh]
    assert [type(c).__name__ for c in restored_batch] == [
        type(c).__name__ for c in fresh_batch
    ]
    # The crash was an AttributeError, so assert the attribute is really there
    # rather than trusting the class name alone.
    for a, b in zip(restored_batch, fresh_batch):
        assert hasattr(a, "rotated") == hasattr(b, "rotated")


def test_rotated_hybrid_cache_is_refused_rather_than_mistyped():
    """Once the ring wraps, refuse the trim — never silently downgrade the type.

    Returning a plain ``KVCache`` here would stop the crash but corrupt the
    sliding layers' KV, which is the failure mode this fix must NOT trade into.
    """
    from vllm_mlx.memory_cache import _trim_cache_offset

    rotated = _make_hybrid_cache(n_tokens=W + 8)
    assert any(
        isinstance(c, RotatingKVCache) and not c.is_trimmable() for c in rotated
    ), "fixture did not actually rotate"
    assert _trim_cache_offset(rotated, 2) is None


def test_trim_offset_plain_kvcache_path_unchanged():
    """Full-attention-only models keep their previous behaviour exactly."""
    from vllm_mlx.memory_cache import _trim_cache_offset

    plain = []
    for _ in range(4):
        c = KVCache()
        x = mx.random.normal((B, H, 10, D))
        c.update_and_fetch(x, x)
        mx.eval(c.keys, c.values)
        plain.append(c)

    trimmed = _trim_cache_offset(plain, 3)
    assert trimmed is not None
    assert all(type(c) is KVCache for c in trimmed)
    assert all(c.offset == 7 for c in trimmed)
    assert all(c.offset == 10 for c in plain)


def test_trim_scope_is_limited_to_rotating_layers():
    """The #1863 fix must not change behaviour for any OTHER cache class.

    ``_trim_cache_offset``'s duck-typed branch accepts anything exposing
    non-list ``keys`` + ``offset``. A "has a trim() method" test would have
    silently re-routed ``ChunkedKVCache``, ``ConcatenateKVCache`` and the
    quantized ``_QuantizableKVCache`` through the class-preserving path too —
    classes whose semantics this helper has never applied (and which
    ``_TRIM_UNSAFE_CACHE_CLASSES`` / the quantization normalizer handle
    separately). Pin the narrow scope so a future refactor can't widen it.
    """
    from mlx_lm.models.cache import ChunkedKVCache

    from vllm_mlx.memory_cache import _trim_cache_offset
    from vllm_mlx.quantized_batch_cache import _QuantizableKVCache

    def _filled(layer, n=8):
        x = mx.random.normal((B, H, n, D))
        layer.update_and_fetch(x, x)
        mx.eval(layer.keys, layer.values)
        return layer

    # Rotating -> preserved (the fix).
    out = _trim_cache_offset([_filled(RotatingKVCache(max_size=64, keep=0))], 2)
    assert out is not None and type(out[0]) is RotatingKVCache

    # Everything else -> still rebuilt as a plain KVCache, exactly as before.
    for layer in (
        KVCache(),
        ChunkedKVCache(chunk_size=256),
        _QuantizableKVCache(64, 8),
    ):
        out = _trim_cache_offset([_filled(layer)], 2)
        assert out is not None, f"{type(layer).__name__} unexpectedly refused"
        assert type(out[0]) is KVCache, (
            f"{type(layer).__name__} was re-routed off the legacy rebuild path — "
            "the fix's scope widened beyond RotatingKVCache"
        )
        assert out[0].offset == 6
