"""Numerical-equivalence tests for QuantizedBatchKVCache (#1197).

Strategy
--------
``QuantizedBatchKVCache`` stores quantized KV but returns bf16 (dequant-on-read),
so two invariants are checked:

1. **Bookkeeping is bit-exact vs BatchKVCache.** ``offset`` / ``left_padding`` /
   ``_idx`` / ``make_mask`` / ``nbytes>`` structure never touch quantization, so
   they must match mlx-lm's ``BatchKVCache`` exactly under the same op sequence.

2. **KV content equals a per-chunk quantize round-trip.** Because the sequence
   axis is 1:1 with tokens in the packed tensor, the value returned for a written
   chunk is exactly ``dequantize(quantize(chunk))`` — deterministic, so the
   comparison is bit-exact (max-abs-diff == 0), not merely "close".
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache

from vllm_mlx.quantized_batch_cache import (
    QuantizedBatchKVCache,
    _dequantize,
    _QuantizableKVCache,
    _quantize,
)

GS = 64
BITS = 8
H = 2
D = 64


def _kv(b, n, seed):
    mx.random.seed(seed)
    k = mx.random.normal((b, H, n, D)).astype(mx.bfloat16)
    v = mx.random.normal((b, H, n, D)).astype(mx.bfloat16)
    return k, v


def _qrt(x, gs=GS, bits=BITS):
    """Reference: what a quantize->dequantize round-trip yields for x."""
    return _dequantize(_quantize(x, gs, bits), gs, bits)


def _max_abs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


# --- update_and_fetch --------------------------------------------------------


def test_single_update_matches_quant_roundtrip():
    B, n = 3, 40
    k, v = _kv(B, n, 0)
    q = QuantizedBatchKVCache([0, 0, 0], GS, BITS)
    ok, ov = q.update_and_fetch(k, v)
    assert ok.shape == (B, H, n, D)
    assert _max_abs(ok, _qrt(k)) == 0.0
    assert _max_abs(ov, _qrt(v)) == 0.0


def test_multi_chunk_crosses_capacity_growth():
    # step=256; write chunks that force a capacity grow + partial-step trim path.
    B = 2
    q = QuantizedBatchKVCache([0, 0], GS, BITS)
    ref_k_chunks, ref_v_chunks = [], []
    total = 0
    for i, n in enumerate([100, 200, 50, 300]):  # crosses 256 boundary
        k, v = _kv(B, n, 10 + i)
        ok, ov = q.update_and_fetch(k, v)
        ref_k_chunks.append(_qrt(k))
        ref_v_chunks.append(_qrt(v))
        total += n
        assert q.size() == total
        assert ok.shape == (B, H, total, D)
        # full-history dequant equals concat of per-chunk round-trips
        assert _max_abs(ok, mx.concatenate(ref_k_chunks, axis=2)) == 0.0
        assert _max_abs(ov, mx.concatenate(ref_v_chunks, axis=2)) == 0.0


def test_offset_and_idx_match_batchkvcache():
    B = 3
    lp = [2, 0, 5]
    q = QuantizedBatchKVCache(lp, GS, BITS)
    b = BatchKVCache(lp)
    for i, n in enumerate([30, 30]):
        k, v = _kv(B, n, 100 + i)
        q.update_and_fetch(k, v)
        b.update_and_fetch(k, v)
    assert q._idx == b._idx
    assert mx.array_equal(q.offset, b.offset)
    assert mx.array_equal(q.left_padding, b.left_padding)


# --- make_mask (must be bit-exact; no quantization involved) ------------------


def test_make_mask_matches_batchkvcache():
    B = 3
    lp = [1, 4, 0]
    q = QuantizedBatchKVCache(lp, GS, BITS)
    b = BatchKVCache(lp)
    k, v = _kv(B, 20, 7)
    q.update_and_fetch(k, v)
    b.update_and_fetch(k, v)
    mq = q.make_mask(1)
    mb = b.make_mask(1)
    assert mq.shape == mb.shape
    assert mx.array_equal(mq, mb)


# --- prepare with left padding ----------------------------------------------


def test_prepare_left_padding_matches():
    B = 2
    q = QuantizedBatchKVCache([0, 0], GS, BITS)
    b = BatchKVCache([0, 0])
    q.prepare(left_padding=[3, 1])
    b.prepare(left_padding=[3, 1])
    assert mx.array_equal(q.offset, b.offset)
    assert mx.array_equal(q.left_padding, b.left_padding)


def test_prepare_left_padding_on_nonempty_raises():
    q = QuantizedBatchKVCache([0], GS, BITS)
    q.update_and_fetch(*_kv(1, 5, 1))
    with pytest.raises(ValueError):
        q.prepare(left_padding=[1])


# --- trim --------------------------------------------------------------------


def test_trim_matches_batchkvcache():
    B = 2
    q = QuantizedBatchKVCache([0, 0], GS, BITS)
    b = BatchKVCache([0, 0])
    k, v = _kv(B, 50, 3)
    q.update_and_fetch(k, v)
    b.update_and_fetch(k, v)
    assert q.trim(10) == b.trim(10)
    assert q._idx == b._idx
    assert mx.array_equal(q.offset, b.offset)


# --- filter (row select + left-shift) ---------------------------------------


def test_filter_matches_batchkvcache():
    B = 4
    lp = [0, 3, 1, 2]
    q = QuantizedBatchKVCache(lp, GS, BITS)
    b = BatchKVCache(lp)
    k, v = _kv(B, 25, 9)
    qk_full, _ = q.update_and_fetch(k, v)
    b.update_and_fetch(k, v)

    keep = [0, 2]
    q.filter(keep)
    b.filter(keep)

    assert q._idx == b._idx
    assert mx.array_equal(q.offset, b.offset)
    assert mx.array_equal(q.left_padding, b.left_padding)
    # Content check: the dequantized kept rows must equal the bf16 BatchKVCache's
    # kept rows put through the same quant round-trip — proves filter() selected
    # the RIGHT rows and preserved their KV, not merely the batch dimension.
    qk_kept = _dequantize(q._slice_seq(q.keys, q._idx), GS, BITS)
    assert qk_kept.shape[0] == 2
    ref_k = _qrt(b.keys[..., : b._idx, :])
    ref_v = _qrt(b.values[..., : b._idx, :])
    assert _max_abs(qk_kept, ref_k) == 0.0
    qv_kept = _dequantize(q._slice_seq(q.values, q._idx), GS, BITS)
    assert _max_abs(qv_kept, ref_v) == 0.0


# --- extend ------------------------------------------------------------------


def test_extend_matches_batchkvcache_bookkeeping():
    qa = QuantizedBatchKVCache([0, 0], GS, BITS)
    qb = QuantizedBatchKVCache([0], GS, BITS)
    ba = BatchKVCache([0, 0])
    bb = BatchKVCache([0])

    ka, va = _kv(2, 40, 11)
    kb, vb = _kv(1, 25, 12)
    qa.update_and_fetch(ka, va)
    qb.update_and_fetch(kb, vb)
    ba.update_and_fetch(ka, va)
    bb.update_and_fetch(kb, vb)

    qa.extend(qb)
    ba.extend(bb)

    assert qa._idx == ba._idx
    assert mx.array_equal(qa.offset, ba.offset)
    assert mx.array_equal(qa.left_padding, ba.left_padding)
    assert qa.keys[0].shape[0] == 3  # batch grew to 3 rows


def test_extend_both_empty():
    qa = QuantizedBatchKVCache([1, 2], GS, BITS)
    qb = QuantizedBatchKVCache([0], GS, BITS)
    qa.extend(qb)
    assert qa.offset.shape[0] == 3
    assert qa.left_padding.shape[0] == 3
    assert qa.keys is None


# --- merge (single-seq -> batch) --------------------------------------------


def _seq_cache(n, seed):
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    k, v = _kv(1, n, seed)
    c.update_and_fetch(k, v)
    return c


def test_merge_bookkeeping_matches_batchkvcache():
    caches_q = [_seq_cache(10, 1), _seq_cache(30, 2), _seq_cache(20, 3)]
    caches_b = [_seq_cache(10, 1), _seq_cache(30, 2), _seq_cache(20, 3)]
    q = QuantizedBatchKVCache.merge(caches_q, GS, BITS)
    b = BatchKVCache.merge(caches_b)
    assert q.size() == b.size()
    assert mx.array_equal(q.offset, b.offset)
    assert mx.array_equal(q.left_padding, b.left_padding)


def test_merge_empty_returns_quantized():
    from mlx_lm.models.cache import KVCache

    q = QuantizedBatchKVCache.merge([KVCache(), KVCache()], GS, BITS)
    assert isinstance(q, QuantizedBatchKVCache)
    assert q.empty()
    # subsequent update quantizes from token 0
    q.update_and_fetch(*_kv(2, 5, 1))
    assert not q.empty()


# --- extract -----------------------------------------------------------------


def test_extract_row_roundtrips():
    B = 3
    lp = [0, 2, 1]
    q = QuantizedBatchKVCache(lp, GS, BITS)
    q.update_and_fetch(*_kv(B, 15, 5))
    idx, pad = 1, lp[1]
    single = q.extract(idx)
    assert single.keys.shape[0] == 1
    assert single.offset == single.keys.shape[2]
    # Content: extract must return the dequantized stored row `idx`, sliced past
    # its left padding — proves it picked the RIGHT row and real (non-zero) KV,
    # not just a correctly-shaped tensor.
    exp_k = _dequantize([m[idx : idx + 1] for m in q.keys], GS, BITS)[
        :, :, pad : q._idx, :
    ]
    exp_v = _dequantize([m[idx : idx + 1] for m in q.values], GS, BITS)[
        :, :, pad : q._idx, :
    ]
    assert single.keys.shape[2] == q._idx - pad
    assert _max_abs(single.keys, exp_k) == 0.0
    assert _max_abs(single.values, exp_v) == 0.0
    assert float(mx.max(mx.abs(single.keys.astype(mx.float32)))) > 0.0  # not zeros


def test_extract_empty_cache_returns_empty_kvcache():
    # codex #2: a request cancelled before its first write reaches extract() on
    # the empty cache merge() produced (keys is None). Must return an empty
    # KVCache, not raise TypeError iterating None.
    from mlx_lm.models.cache import KVCache

    q = QuantizedBatchKVCache.merge([KVCache(), KVCache()], GS, BITS)
    assert q.keys is None
    single = q.extract(0)
    assert isinstance(single, KVCache)
    assert single.keys is None and single.values is None


# --- empty-cache / dtype / fallback edge cases (pr_validate codex review) ----


def test_empty_cache_state_is_none_pair():
    # BLOCKING: reconstruction dequantizes state[0]; an empty cache MUST expose
    # (None, None, ...) so the scheduler restores an empty KVCache instead of
    # dequantizing None.
    from mlx_lm.models.cache import KVCache

    q = QuantizedBatchKVCache.merge([KVCache(), KVCache()], GS, BITS)
    k, v, off, lp = q.state
    assert k is None and v is None


def test_scheduler_reconstructs_empty_quantized_cache():
    # BLOCKING (end-to-end): the scheduler's reconstruction branch must restore
    # an empty QuantizedBatchKVCache snapshot to an empty KVCache without
    # dequantizing None.
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.scheduler import Scheduler

    q = QuantizedBatchKVCache([0], GS, BITS)  # empty, never written
    layer_state = {
        "state": q.state,  # (None, None, offset, left_padding)
        "meta_state": q.meta_state,
        "class_ref": QuantizedBatchKVCache,
    }
    sched = Scheduler.__new__(Scheduler)
    out = sched._reconstruct_cache_from_states([layer_state])
    assert out is not None
    assert isinstance(out[0], KVCache)
    assert out[0].keys is None
    assert out[0].offset == 0


def test_finalize_on_empty_cache_no_write():
    # NIT: prepare(right_padding=...) then finalize() with no write must settle
    # padding bookkeeping without iterating None storage.
    q = QuantizedBatchKVCache([0, 0], GS, BITS)
    q.prepare(right_padding=[1, 0])
    q.finalize()  # must not raise
    assert q.keys is None
    assert q._right_padding is None


def test_merge_falls_back_to_bf16_on_incompatible_dims():
    # NIT: head_dim=80 admits no supported group size -> merge must degrade to a
    # plain BatchKVCache instead of raising on the first write.
    from mlx_lm.generate import BatchKVCache as _BatchKVCache
    from mlx_lm.models.cache import KVCache

    def _seq(dim, n, seed):
        c = KVCache()
        mx.random.seed(seed)
        x = mx.random.normal((1, H, n, dim)).astype(mx.bfloat16)
        c.update_and_fetch(x, x)
        return c

    merged = QuantizedBatchKVCache.merge([_seq(80, 6, 1), _seq(80, 4, 2)], GS, BITS)
    assert isinstance(merged, _BatchKVCache)
    assert not isinstance(merged, QuantizedBatchKVCache)


def test_merge_preserves_distinct_key_value_dtypes():
    # NIT: keys/values may carry different dtypes (MLA-like); merge must not cast
    # values to the key dtype.
    from mlx_lm.models.cache import KVCache

    def _seq(kd, vd, kdt, vdt, n, seed):
        c = KVCache()
        mx.random.seed(seed)
        c.keys = mx.random.normal((1, H, n, kd)).astype(kdt)
        c.values = mx.random.normal((1, H, n, vd)).astype(vdt)
        c.offset = n
        return c

    # keys bf16, values float16 — both head_dim 64 (quantizes cleanly).
    merged = QuantizedBatchKVCache.merge(
        [_seq(64, 64, mx.bfloat16, mx.float16, 5, 1)], GS, BITS
    )
    # scales carry the source dtype; values scales must stay float16, keys bfloat16
    assert merged.keys[1].dtype == mx.bfloat16
    assert merged.values[1].dtype == mx.float16


# --- state round-trip --------------------------------------------------------


def test_state_roundtrip_preserves_content():
    B = 2
    q = QuantizedBatchKVCache([0, 0], GS, BITS)
    k, v = _kv(B, 20, 6)
    out_k, _ = q.update_and_fetch(k, v)
    st = q.state

    q2 = QuantizedBatchKVCache([0, 0], GS, BITS)
    q2.state = st
    q2.meta_state = q.meta_state
    assert q2._idx == q._idx
    out_k2 = _dequantize(q2._slice_seq(q2.keys, q2._idx), GS, BITS)
    assert _max_abs(out_k, out_k2) == 0.0


# --- nbytes ------------------------------------------------------------------


@pytest.mark.parametrize("bits,ratio", [(8, 0.6), (4, 0.35)])
def test_nbytes_smaller_than_bf16(bits, ratio):
    B, n = 2, 512
    q = QuantizedBatchKVCache([0, 0], GS, bits)
    b = BatchKVCache([0, 0])
    k, v = _kv(B, n, 4)
    q.update_and_fetch(k, v)
    b.update_and_fetch(k, v)
    # quantized resident footprint must be well below bf16
    assert q.nbytes < b.nbytes * ratio


# --- _QuantizableKVCache wiring ---------------------------------------------


def test_quantizable_kvcache_merge_builds_quantized_batch():
    a = _QuantizableKVCache(GS, BITS)
    b = _QuantizableKVCache(GS, BITS)
    a.update_and_fetch(*_kv(1, 10, 1))
    b.update_and_fetch(*_kv(1, 20, 2))
    merged = _QuantizableKVCache.merge([a, b])
    assert isinstance(merged, QuantizedBatchKVCache)
    assert merged.size() == 20
    assert merged._q_group_size == GS and merged._q_bits == BITS


def test_quantizable_kvcache_empty_merge():
    a = _QuantizableKVCache(GS, BITS)
    b = _QuantizableKVCache(GS, BITS)
    merged = _QuantizableKVCache.merge([a, b])
    assert isinstance(merged, QuantizedBatchKVCache)
    assert merged.empty()


# --- install wiring ----------------------------------------------------------


def test_install_wraps_only_kvcache_layers():
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from vllm_mlx.quantized_batch_cache import install_quantized_batch_cache

    class _FakeBG:
        def _make_new_cache(self):
            return [KVCache(), RotatingKVCache(max_size=64), KVCache()]

    bg = _FakeBG()
    install_quantized_batch_cache(bg, group_size=64, bits=4)
    caches = bg._make_new_cache()
    assert isinstance(caches[0], _QuantizableKVCache)
    assert isinstance(caches[2], _QuantizableKVCache)
    assert caches[0].q_bits == 4 and caches[0].q_group_size == 64
    # sliding-window layer left as-is (quantized batched rotating is NYI upstream)
    assert type(caches[1]).__name__ == "RotatingKVCache"


# --- BLOCKING fixes: head_dim/group_size compatibility ----------------------


def test_supported_group_size():
    from vllm_mlx.quantized_batch_cache import supported_group_size

    assert supported_group_size(64, 64) == 64
    assert supported_group_size(128, 64) == 64
    assert supported_group_size(96, 64) == 32  # 96 not div by 64, is by 32
    assert supported_group_size(96, 128) == 32
    assert supported_group_size(128, 128) == 128
    assert supported_group_size(80, 64) is None  # 80 not div by 32/64/128
    assert supported_group_size(40, 128) is None


def test_probe_head_dim():
    from vllm_mlx.quantized_batch_cache import probe_head_dim

    class _Args:
        head_dim = 128

    class _M:
        args = _Args()

    assert probe_head_dim(_M()) == 128

    class _Args2:
        hidden_size = 2048
        num_attention_heads = 32

    class _M2:
        args = _Args2()

    assert probe_head_dim(_M2()) == 64
    assert probe_head_dim(object()) is None

    # Multimodal wrapper (VLM): the language model's head_dim lives on a nested
    # `language_model.args`, not the top-level `model.args`. Must descend so the
    # live KV cache is not spuriously disabled (#1199 follow-up).
    class _TextArgs:
        head_dim = 256

    class _LM:
        args = _TextArgs()

    class _VLM:
        args = object()  # top-level args carry NO attention dims
        language_model = _LM()

    assert probe_head_dim(_VLM()) == 256

    # Alternative nesting: dims exposed via `args.text_config` sub-config.
    class _TopArgsWithTextConfig:
        text_config = _TextArgs()

    class _VLM2:
        args = _TopArgsWithTextConfig()

    assert probe_head_dim(_VLM2()) == 256

    # Mixed: top-level args expose MISLEADING vision/composite dims
    # (hidden_size // num_attention_heads = 64) while the authoritative language
    # head_dim is 256. The nested language config must win — preferring the
    # top-level fallback here would mis-size the live cache (#1208).
    class _VisionishTop:
        hidden_size = 1024
        num_attention_heads = 16  # -> 64, the WRONG (non-language) head dim

    class _VLM3:
        args = _VisionishTop()
        language_model = _LM()  # language head_dim = 256

    assert probe_head_dim(_VLM3()) == 256

    # VLM whose nested language config is UNPROBEABLE must fail safe to None,
    # NOT fall back to the misleading top-level vision/composite dims (#1208
    # codex): a wrong head dim would mis-size the live cache instead of a clean
    # bf16 fallback.
    class _UnprobeableArgs:
        pass  # no head_dim, no hidden_size/num_attention_heads

    class _UnprobeableLM:
        args = _UnprobeableArgs()

    class _VLM4:
        args = _VisionishTop()  # -> 64 if wrongly trusted
        language_model = _UnprobeableLM()

    assert probe_head_dim(_VLM4()) is None

    # VLM signalled by `language_model` presence even when the submodule exposes
    # no `.args` at all: the presence of the submodule is the multimodal signal,
    # so we must still refuse the top-level vision dims and fail safe (#1208).
    class _NoArgsLM:
        pass  # language_model submodule with no `.args`

    class _VLM5:
        args = _VisionishTop()
        language_model = _NoArgsLM()

    assert probe_head_dim(_VLM5()) is None


def test_update_adjusts_group_size_for_head_dim_96():
    # head_dim=96 is not divisible by 64 but is by 32 — must auto-adjust, not crash
    q = QuantizedBatchKVCache([0], group_size=64, bits=8)
    k = mx.random.normal((1, H, 10, 96)).astype(mx.bfloat16)
    ok, ov = q.update_and_fetch(k, k)
    assert q._q_group_size == 32
    assert ok.shape == (1, H, 10, 96)
    assert _max_abs(ok, _dequantize(_quantize(k, 32, 8), 32, 8)) == 0.0


def test_update_raises_on_incompatible_head_dim():
    # head_dim=80 is divisible by no supported group size -> clear error
    q = QuantizedBatchKVCache([0], group_size=64, bits=8)
    k = mx.random.normal((1, H, 10, 80)).astype(mx.bfloat16)
    with pytest.raises(ValueError, match="group_size"):
        q.update_and_fetch(k, k)


def test_merge_adjusts_group_size_for_head_dim_96():
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    k = mx.random.normal((1, H, 12, 96)).astype(mx.bfloat16)
    c.update_and_fetch(k, k)
    merged = QuantizedBatchKVCache.merge([c], group_size=64, bits=8)
    assert merged._q_group_size == 32  # coerced 64 -> 32 for head_dim=96


# --- BLOCKING fix: prefix-cache HIT normalization ---------------------------


def test_normalize_preserves_content_and_type():
    from mlx_lm.models.cache import RotatingKVCache

    from vllm_mlx.quantized_batch_cache import normalize_caches_for_quantization

    restored = _seq_cache(20, 1)
    norm = normalize_caches_for_quantization(
        [restored, RotatingKVCache(max_size=64)], GS, BITS
    )
    assert isinstance(norm[0], _QuantizableKVCache)
    assert norm[0].q_group_size == GS and norm[0].q_bits == BITS
    assert norm[0].offset == restored.offset  # content preserved
    assert type(norm[1]).__name__ == "RotatingKVCache"  # non-KVCache untouched


def test_prefix_hit_merges_to_quantized_like_miss():
    # A normalized prefix-cache HIT must merge into QuantizedBatchKVCache (same
    # type as a MISS), so mlx-lm never mixes bf16 and quantized batches (#1197).
    from vllm_mlx.quantized_batch_cache import normalize_caches_for_quantization

    hit = normalize_caches_for_quantization([_seq_cache(20, 1)], GS, BITS)
    merged_hit = hit[0].merge([hit[0]])
    assert isinstance(merged_hit, QuantizedBatchKVCache)
    assert merged_hit.size() == 20

    miss = _QuantizableKVCache(GS, BITS)
    merged_miss = _QuantizableKVCache.merge([miss])
    assert isinstance(merged_miss, QuantizedBatchKVCache)

    # both are the same type -> extend is safe (no bf16-vs-quantized crash)
    merged_hit.extend(merged_miss)
    assert merged_hit.size() == 20


# --- MAJOR fix: CacheList (hybrid models) recursion -------------------------


def _has_cachelist():
    try:
        from mlx_lm.models.cache import CacheList  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cachelist(), reason="mlx-lm has no CacheList")
def test_install_leaves_cachelist_untouched():
    # CacheList (DeepSeek-V3.2 / LongCat / Baichuan-M1 / Falcon-H1) must pass
    # through as-is: these models touch cache.keys directly in their forward
    # pass (e.g. mx.depends(cache[0].keys, ...)), which assumes a raw array, not
    # a quantized triple. Only the top-level exact KVCache layer is swapped.
    from mlx_lm.models.cache import CacheList, KVCache

    from vllm_mlx.quantized_batch_cache import install_quantized_batch_cache

    class _FakeBG:
        def _make_new_cache(self):
            return [CacheList(KVCache(), KVCache()), KVCache()]

    bg = _FakeBG()
    install_quantized_batch_cache(bg, group_size=64, bits=8)
    layers = bg._make_new_cache()
    assert isinstance(layers[0], CacheList)  # left untouched (bf16)
    assert all(type(c) is KVCache for c in layers[0].caches)  # inner NOT wrapped
    assert isinstance(layers[1], _QuantizableKVCache)  # plain KVCache swapped


@pytest.mark.skipif(not _has_cachelist(), reason="mlx-lm has no CacheList")
def test_normalize_leaves_cachelist_untouched():
    from mlx_lm.models.cache import CacheList, KVCache

    from vllm_mlx.quantized_batch_cache import normalize_caches_for_quantization

    restored = _seq_cache(15, 1)
    cl = CacheList(_seq_cache(15, 2), _seq_cache(15, 3))
    out = normalize_caches_for_quantization([restored, cl], GS, BITS)
    assert isinstance(out[0], _QuantizableKVCache)  # plain KVCache normalized
    assert isinstance(out[1], CacheList)  # CacheList left untouched (bf16)
    assert all(type(c) is KVCache for c in out[1].caches)


# --- head_dim resolution: one group size, shared by both caches --------------


def test_probe_kv_head_dims():
    from vllm_mlx.quantized_batch_cache import probe_kv_head_dims

    class _Args:
        head_dim = 128

    class _M:
        args = _Args()

    assert probe_kv_head_dims(_M()) == (128, 128)

    # Distinct value head dim (v_head_dim) must be probed independently.
    class _ArgsV:
        head_dim = 128
        v_head_dim = 96

    class _MV:
        args = _ArgsV()

    assert probe_kv_head_dims(_MV()) == (128, 96)

    # hidden_size/num_heads fallback for K, V defaults to K.
    class _Args2:
        hidden_size = 2048
        num_attention_heads = 32

    class _M2:
        args = _Args2()

    assert probe_kv_head_dims(_M2()) == (64, 64)

    # Multimodal wrapper: both key AND value head dims resolve from the nested
    # `language_model.args`, not the top-level args (#1199 follow-up).
    class _TextArgsV:
        head_dim = 256
        v_head_dim = 128

    class _LMV:
        args = _TextArgsV()

    class _VLMV:
        args = object()
        language_model = _LMV()

    assert probe_kv_head_dims(_VLMV()) == (256, 128)

    # Mixed: misleading top-level dims must NOT win over the nested language
    # config, and v_head_dim is read from the SAME (nested) args (#1208).
    class _VisionishTopV:
        head_dim = 64  # wrong (non-language) dim; must be ignored
        v_head_dim = 64

    class _VLMV2:
        args = _VisionishTopV()
        language_model = _LMV()  # language head_dim=256, v_head_dim=128

    assert probe_kv_head_dims(_VLMV2()) == (256, 128)

    # Unknown -> (None, None).
    assert probe_kv_head_dims(object()) == (None, None)


def test_resolve_kv_quantization():
    from vllm_mlx.quantized_batch_cache import resolve_kv_quantization

    # Compatible after coercion (96 -> 32): live enabled, gs=32.
    assert resolve_kv_quantization(96, 96, 64) == (32, False)
    # Already compatible: unchanged.
    assert resolve_kv_quantization(128, 128, 64) == (64, False)
    # Asymmetric K/V dims: gs must divide BOTH (128 ok at 64, 96 forces 32).
    assert resolve_kv_quantization(128, 96, 64) == (32, False)
    # No supported size divides both (80): live disabled (retained self-coerces).
    assert resolve_kv_quantization(80, 80, 64) == (64, True)
    # Probe failure: live disabled.
    assert resolve_kv_quantization(None, None, 64) == (64, True)
    assert resolve_kv_quantization(128, None, 64) == (64, True)


def test_scheduler_wires_resolved_group_size_to_live_hook():
    # End-to-end wiring: a head_dim=96 model must drive the coerced group size
    # (32) into the live install params and leave the retained cache at its
    # configured size (self-coerced later). Exercises
    # Scheduler._init_kv_quantization directly, so deleting the wiring turns red.
    from vllm_mlx.memory_cache import MemoryCacheConfig
    from vllm_mlx.scheduler import Scheduler

    class _Args:
        head_dim = 96

    class _Model:
        args = _Args()

    class _Cfg:
        kv_cache_quantization = True
        kv_cache_quantization_bits = 8
        kv_cache_quantization_group_size = 64
        kv_cache_turboquant = None

    sched = Scheduler.__new__(Scheduler)
    sched.config = _Cfg()
    sched._init_kv_quantization(_Model())

    # Live hook: coerced to 32 and the install gate is open.
    assert sched._kv_quant_group_size == 32
    assert sched._kv_quant_live_disabled is False
    live_on = (
        sched.config.kv_cache_quantization
        and not sched.config.kv_cache_turboquant
        and not sched._kv_quant_live_disabled
    )
    assert live_on is True

    # Retained cache: MemoryCacheConfig keeps quant on and the CONFIGURED gs (64);
    # per-layer coercion happens in _quantize_cache, not here.
    mcc = MemoryCacheConfig(
        kv_quantize=sched.config.kv_cache_quantization,
        kv_bits=sched.config.kv_cache_quantization_bits,
        kv_group_size=sched.config.kv_cache_quantization_group_size,
    )
    assert mcc.kv_quantize is True
    assert mcc.kv_group_size == 64


def test_scheduler_probe_failure_disables_live_only():
    # An unprobeable model disables only the LIVE cache; the retained cache is
    # never gated by the config-level probe (it self-coerces at quantize time).
    from vllm_mlx.scheduler import Scheduler

    class _Cfg:
        kv_cache_quantization = True
        kv_cache_quantization_bits = 8
        kv_cache_quantization_group_size = 64
        kv_cache_turboquant = None

    sched = Scheduler.__new__(Scheduler)
    sched.config = _Cfg()
    sched._init_kv_quantization(object())  # unprobeable model

    assert sched._kv_quant_live_disabled is True
    assert sched._kv_quant_group_size == 64  # unchanged


def test_live_install_gate_fails_closed_when_init_bypassed():
    # codex pr_validate BLOCKING: some unit/serve paths build a Scheduler via
    # __new__ (bypassing __init__ -> _init_kv_quantization), leaving the resolve
    # attributes absent. The live-install gate MUST fail CLOSED: a missing
    # _kv_quant_live_disabled has to read as "disabled" so an unprobed,
    # possibly-incompatible head dim never installs a quantized live cache that
    # would crash on first write. The gate defaults the flag to True.
    from vllm_mlx.scheduler import Scheduler

    class _Cfg:  # quantization requested, but the probe never ran
        kv_cache_quantization = True
        kv_cache_quantization_bits = 8
        kv_cache_quantization_group_size = 64
        kv_cache_turboquant = None

    sched = Scheduler.__new__(Scheduler)
    sched.config = _Cfg()
    # NOTE: _init_kv_quantization() deliberately NOT called.
    assert not hasattr(sched, "_kv_quant_live_disabled")

    # Replicate the real install gate (scheduler.py ~2568). The missing-attr
    # default MUST be True so the gate is closed.
    live_on = (
        getattr(sched.config, "kv_cache_quantization", False)
        and not getattr(sched.config, "kv_cache_turboquant", None)
        and not getattr(sched, "_kv_quant_live_disabled", True)
    )
    assert live_on is False


def test_scheduler_mla_probe_disables_live_but_retained_still_quantizes():
    # codex MAJOR: DeepSeek-V3 (MLA) probes to (56, 128) via hidden_size/heads +
    # v_head_dim, which resolves incompatible — so the LIVE cache is (conserv-
    # atively) bf16. But the retained cache MUST still quantize: its real cached
    # dims are kv_lora_rank=512 / qk_rope=64, both group-size-64 compatible, and
    # _quantize_cache coerces against those, NOT the config head dims.
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.memory_cache import _quantize_cache
    from vllm_mlx.scheduler import Scheduler

    class _Args:  # DeepSeek-V3-0324 shape
        hidden_size = 7168
        num_attention_heads = 128
        v_head_dim = 128

    class _Model:
        args = _Args()

    class _Cfg:
        kv_cache_quantization = True
        kv_cache_quantization_bits = 8
        kv_cache_quantization_group_size = 64
        kv_cache_turboquant = None

    sched = Scheduler.__new__(Scheduler)
    sched.config = _Cfg()
    sched._init_kv_quantization(_Model())
    assert sched._kv_quant_live_disabled is True  # live conservatively bf16

    # Retained: build the MLA-shaped KVCache and quantize it with the configured
    # gs=64. It must NOT stay bf16 (the pre-fix regression) — 512 and 64 both
    # quantize at 64.
    kv = KVCache()
    klat = mx.random.normal((1, 1, 12, 512)).astype(mx.bfloat16)  # kv_latent
    kpe = mx.random.normal((1, 1, 12, 64)).astype(mx.bfloat16)  # k_pe
    kv.update_and_fetch(klat, kpe)
    out = _quantize_cache([kv], bits=8, group_size=64)
    assert type(out[0]).__name__ == "QuantizedKVCache"
    assert out[0].group_size == 64


def test_quantize_cache_coerces_and_skips_per_layer():
    # _quantize_cache resolves the group size against each layer's REAL dims:
    # head_dim=96 -> 32, head_dim=128 -> 64, head_dim=80 -> keep bf16 (no crash).
    from mlx_lm.models.cache import KVCache

    from vllm_mlx.memory_cache import _quantize_cache

    def _layer(dim):
        c = KVCache()
        x = mx.random.normal((1, 2, 8, dim)).astype(mx.bfloat16)
        c.update_and_fetch(x, x)
        return c

    out = _quantize_cache([_layer(96), _layer(128), _layer(80)], bits=8, group_size=64)
    mx.eval([m for layer in out if layer.keys is not None for m in layer.keys])
    assert type(out[0]).__name__ == "QuantizedKVCache" and out[0].group_size == 32
    assert type(out[1]).__name__ == "QuantizedKVCache" and out[1].group_size == 64
    assert type(out[2]).__name__ == "KVCache"  # head_dim=80: kept bf16, no crash


# --- extend quant-param safety (codex review) -------------------------------


def test_extend_empty_adopts_other_quant_params():
    # An empty cache carries its construction-time default; extending it with a
    # populated cache must adopt the populated params so the triples dequantize
    # with the right group_size/bits.
    empty = QuantizedBatchKVCache([0], group_size=64, bits=8)
    pop = QuantizedBatchKVCache([0], group_size=32, bits=4)
    pop.update_and_fetch(*_kv(1, 10, 1))
    empty.extend(pop)
    assert empty._q_group_size == 32 and empty._q_bits == 4


def test_extend_mismatched_params_raises():
    a = QuantizedBatchKVCache([0], group_size=64, bits=8)
    a.update_and_fetch(*_kv(1, 10, 1))
    b = QuantizedBatchKVCache([0], group_size=32, bits=4)
    b.update_and_fetch(*_kv(1, 10, 2))
    with pytest.raises(ValueError, match="mismatched quantization"):
        a.extend(b)
