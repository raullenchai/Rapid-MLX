"""Fused V4.1 decode attention kernels against the graph path and a float64 oracle."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native import (
    attention,
    decode_attention,
    sparse_attention,
)
from rapid_mlx.models.deepseek_v41_native.attention import window_idx_matrix
from rapid_mlx.models.deepseek_v41_native.cache import LayerCache
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from rapid_mlx.models.deepseek_v41_native.fakequant import fake_quant_fp8_ue8m0
from rapid_mlx.models.deepseek_v41_native.layers import (
    RMSNorm,
    precompute_freqs_cis,
    rope_tail,
)
from rapid_mlx.models.deepseek_v41_native.model import Model

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="fused kernels need Metal"
)

HEADS, DIM, WINDOW, RD = 16, 512, 128, 64


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(decode_attention, "ENABLED", True)


def _rope_table():
    return precompute_freqs_cis(RD, 24000, 65536, 160000.0, 16.0, 32, 1)


def _case(seed, b, start, n_comp, k):
    """Random decode inputs, plus the graph path's concatenated keys/indices."""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    q0 = (mx.random.normal((b, 1, HEADS, DIM)) * 0.5).astype(mx.bfloat16)
    ring = fake_quant_fp8_ue8m0(mx.random.normal((b, WINDOW, DIM)), 32)
    comp = mx.random.normal((b, max(n_comp, 1), DIM))
    sink = mx.random.normal((HEADS,))
    wp = min(start, WINDOW)
    offset = wp + 1
    cidx = None
    if k:
        rows = np.stack([rng.permutation(n_comp)[:k] for _ in range(b)])
        rows[:, ::7] = -1  # some selections masked
        cidx = mx.array(np.where(rows >= 0, rows + offset, -1).astype(np.int32))
        cidx = cidx[:, None, :]
    slots = (start - wp + mx.arange(wp)) % WINDOW
    keys = [ring[:, slots].astype(mx.bfloat16)] if wp else []
    keys.append(ring[:, start % WINDOW][:, None].astype(mx.bfloat16))
    idxs = mx.broadcast_to(
        window_idx_matrix(wp, 1, WINDOW)[None], (b, 1, min(WINDOW, wp + 1))
    )
    if k:
        keys.append(comp[:, :n_comp].astype(mx.bfloat16))
        idxs = mx.concatenate([idxs, cidx], -1)
    return q0, ring, comp, sink, offset, cidx, mx.concatenate(keys, 1), idxs


def _oracle(q, keys, idxs, sink):
    """float64 attention over the graph path's (query-dtype) keys."""
    qn = np.array(q.astype(mx.float32)).astype(np.float64)
    kn = np.array(keys.astype(mx.float32)).astype(np.float64)
    ix, sk = np.array(idxs), np.array(sink).astype(np.float64)
    out = np.zeros((q.shape[0], HEADS, DIM))
    for b in range(q.shape[0]):
        ok = ix[b, 0] >= 0
        rows = kn[b, np.maximum(ix[b, 0], 0)]
        logits = qn[b, 0] @ rows.T * DIM**-0.5
        logits[:, ~ok] = -1e30
        m = np.maximum(np.maximum(logits.max(-1), sk), -1e30)
        w = np.exp(logits - m[:, None]) * ok
        out[b] = (w @ rows) / (w.sum(-1) + np.exp(sk - m))[:, None]
    return out


_MANTISSA = {mx.bfloat16: 7, mx.float16: 10, mx.float32: 23}


def _ulp(x, dtype=mx.bfloat16):
    """Spacing of ``dtype`` values at |x| (the smaller one below a power of two)."""
    a = np.maximum(np.abs(x), np.finfo(np.float32).tiny)
    return 2.0 ** (np.floor(np.log2(a)) - _MANTISSA[dtype])


def _inverse_rope_f64(o, c, s):
    """rope_tail(..., inverse=True) of a float64 [b, h, d] array, in float64."""
    c = np.array(c.astype(mx.float32)).astype(np.float64)[0]
    s = np.array(s.astype(mx.float32)).astype(np.float64)[0]
    out = o.copy()
    re, im = o[..., DIM - RD :: 2], o[..., DIM - RD + 1 :: 2]
    out[..., DIM - RD :: 2] = re * c + im * s
    out[..., DIM - RD + 1 :: 2] = im * c - re * s
    return out


@needs_metal
@pytest.mark.parametrize(
    "b,start,n_comp,k",
    [
        (1, 0, 0, 0),
        (1, 5, 0, 0),
        (1, 127, 0, 0),
        (1, 128, 0, 0),
        (1, 3000, 750, 512),
        (2, 700, 175, 175),
        (1, 20000, 5000, 512),
        (1, 300, 2, 2),
    ],
)
def test_attend_matches_float64_oracle_at_least_as_well_as_graph(b, start, n_comp, k):
    q0, ring, comp, sink, offset, cidx, keys, idxs = _case(
        start + k, b, start, n_comp, k
    )
    cos, sin = _rope_table()
    c, s = cos[start : start + 1], sin[start : start + 1]
    q = rope_tail(q0, RD, c, s)
    graph = rope_tail(
        sparse_attention.sparse_attn(q, keys, sink, idxs, DIM**-0.5),
        RD,
        c,
        s,
        inverse=True,
    )
    fused = decode_attention.attend(
        q0, ring, start, comp if k else None, cidx, offset, sink, c, s, DIM**-0.5, RD
    )
    assert fused.shape == (b, 1, HEADS, DIM) and fused.dtype == mx.float32

    exact = _oracle(q, keys, idxs, sink)
    # the float64 oracle carried through the float64 inverse rope tail, so both
    # paths are measured in the coordinates they return
    exact_rot = _inverse_rope_f64(exact, c, s)
    # the exact result with the graph path's bf16 output rounding and rope
    rounded = rope_tail(
        mx.array(exact.astype(np.float32))[:, None].astype(mx.bfloat16),
        RD,
        c,
        s,
        inverse=True,
    )
    expected = np.array(rounded.astype(mx.float32)).reshape(b, HEADS, DIM)
    got = np.array(fused).reshape(b, HEADS, DIM)
    ref = np.array(graph.astype(mx.float32)).reshape(b, HEADS, DIM)
    # the fused values are bf16-representable
    assert np.array_equal(
        got, np.array(mx.array(got).astype(mx.bfloat16).astype(mx.float32))
    )
    # fp32 accumulation order only flips rare bf16 roundings of the exact value
    assert (got != expected).mean() <= 2e-3
    # no further from the exact value than the graph path (both round their
    # output to bf16, which dominates the error; the margin covers the rare
    # accumulation-order flips counted above)
    fused_err, graph_err = np.abs(got - exact_rot), np.abs(ref - exact_rot)
    assert fused_err.max() <= graph_err.max() * 1.05
    assert fused_err.mean() <= graph_err.mean() * 1.01


@needs_metal
def test_attend_reads_the_ring_in_chronological_order():
    """A ring rotated by one slot must change the result only through its order."""
    b, start = 1, 300
    q0, ring, comp, sink, offset, cidx, keys, idxs = _case(5, b, start, 0, 0)
    cos, sin = _rope_table()
    c, s = cos[start : start + 1], sin[start : start + 1]
    base = decode_attention.attend(
        q0, ring, start, None, None, offset, sink, c, s, 0.1, RD
    )
    shifted = decode_attention.attend(
        q0, mx.roll(ring, 1, axis=1), start + 1, None, None, offset, sink, c, s, 0.1, RD
    )
    assert np.array_equal(np.array(base), np.array(shifted))


@needs_metal
def test_attend_masks_compressed_indices_outside_the_rows():
    """Indices below the offset or past the last compressed row read nothing."""
    b, start, n_comp, k = 2, 400, 16, 8
    q0, ring, comp, sink, offset, cidx, keys, idxs = _case(9, b, start, n_comp, k)
    cos, sin = _rope_table()
    c, s = cos[start : start + 1], sin[start : start + 1]
    picks = np.array(cidx)
    picks[:, 0, :4] = [offset - 1, offset + n_comp, offset + n_comp + 4096, -5]
    masked = picks.copy()
    masked[:, 0, :4] = -1

    def run(ix):
        return np.array(
            decode_attention.attend(
                q0, ring, start, comp, mx.array(ix), offset, sink, c, s, 0.1, RD
            )
        )

    out = run(picks)
    assert np.isfinite(out).all()
    assert np.array_equal(out, run(masked))


@needs_metal
@pytest.mark.parametrize("act", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("cache_dtype", [mx.float32, mx.bfloat16])
def test_norms_kernel_matches_graph(cache_dtype, act):
    cos, sin = _rope_table()
    q_rank = 1280
    q_diff = 0
    for trial in range(24):
        mx.random.seed(100 + trial)
        # fp16 tops out near 6.5e4, so keep its squares' sum finite
        hi = 1 if act == mx.float16 else 3
        scale = float(10 ** np.random.default_rng(trial).uniform(-3, hi))
        qa = (mx.random.normal((2, 1, q_rank)) * scale).astype(act)
        kva = (mx.random.normal((2, 1, DIM)) * scale).astype(act)
        q_norm, kv_norm = RMSNorm(q_rank, 1e-20), RMSNorm(DIM, 1e-20)
        q_norm.weight = mx.random.uniform(0.5, 2.0, (q_rank,))
        kv_norm.weight = mx.random.uniform(0.5, 2.0, (DIM,))
        pos = trial * 97
        c, s = cos[pos : pos + 1], sin[pos : pos + 1]
        ref_q = np.array(q_norm(qa).astype(mx.float32))
        ref_kv = fake_quant_fp8_ue8m0(rope_tail(kv_norm(kva), RD, c, s), 32)
        got_q, got_kv = decode_attention.norms(
            qa, kva, q_norm.weight, kv_norm.weight, 1e-20, c, s, RD, cache_dtype
        )
        assert got_q.dtype == act and got_kv.dtype == cache_dtype
        # the cached KV row (rope tail and FP8 fake-quant included) is exact
        assert np.array_equal(
            np.array(ref_kv.astype(cache_dtype).astype(mx.float32)),
            np.array(got_kv.astype(mx.float32)),
        )
        got = np.array(got_q.astype(mx.float32))
        differ = got != ref_q
        q_diff += int(differ.sum())
        # a different fp32 sum order can move a bf16/fp16 q_norm by at most
        # one step; fp32 activations keep the fp32 rsqrt's few-ulp difference
        steps = 4 if act == mx.float32 else 1
        nearer = np.minimum(np.abs(got), np.abs(ref_q))[differ]
        assert np.all(np.abs(got - ref_q)[differ] <= steps * _ulp(nearer, act))
    if act != mx.float32:
        assert q_diff <= 24 * 2 * q_rank * 1e-3


def _tiny_args():
    return ModelArgs(
        vocab_size=64,
        dim=64,
        n_layers=3,
        moe_inter_dim=16,
        n_heads=8,
        head_dim=128,
        rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        window_size=8,
        n_routed_experts=4,
        n_activated_experts=2,
        compress_ratios=(0, 4, 4),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        index_n_heads=2,
        index_head_dim=64,
        # selects every compressed row, so near-tied index scores cannot flip
        # the selection between the two paths
        index_topk=16,
        hc_mult=2,
    )


def _exact_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """sparse_attn with elementwise fp32 sums (MLX's fp32 einsum may run at reduced precision)."""
    h = q.shape[2]
    sink = attn_sink.astype(mx.float32).reshape(1, 1, h, 1)
    keys = sparse_attention._gather_kv(kv, topk_idxs).astype(mx.float32)
    logits = (q.astype(mx.float32)[:, :, :, None] * keys[:, :, None]).sum(-1)
    valid = (topk_idxs >= 0)[:, :, None, :]
    logits = mx.where(valid, logits * softmax_scale, -1e30)
    peak = mx.maximum(mx.maximum(logits.max(-1, keepdims=True), sink), -1e30)
    w = mx.where(valid, mx.exp(logits - peak), 0.0)
    denom = w.sum(-1, keepdims=True) + mx.exp(sink - peak)
    return ((w[..., None] * keys[:, :, None]).sum(-2) / denom).astype(q.dtype)


@needs_metal
@pytest.mark.parametrize(
    "bsz,dtype,first,tol,stock",
    [
        (1, mx.float32, 1e-5, 3e-3, False),
        # row 1 of this seed puts one layer-2 window value on an FP8 rounding
        # boundary at the first decode step (one e4m3 code apart), so only the
        # per-step tolerance applies
        (2, mx.float32, None, 3e-3, False),
        (1, mx.bfloat16, 0.0, 5e-2, False),
        # against the shipped graph attention as well
        (1, mx.float32, 1e-5, 1e-4, True),
        (2, mx.bfloat16, 0.0, 5e-2, True),
    ],
)
def test_model_decode_with_fused_attention_tracks_graph(
    bsz, dtype, first, tol, stock, monkeypatch
):
    if not stock:
        monkeypatch.setattr(attention, "sparse_attn", _exact_sparse_attn)
    mx.random.seed(3)
    model = Model(_tiny_args())
    model.set_dtype(dtype)
    prompt = mx.random.randint(0, 64, (bsz, 21))
    steps = mx.random.randint(0, 64, (bsz, 12))

    def run(enabled):
        monkeypatch.setattr(decode_attention, "ENABLED", enabled)
        cache = model.make_cache(bsz=bsz, max_seq_len=64)
        out = [model(prompt, cache, last_logit_only=True)]
        for i in range(steps.shape[1]):
            out.append(model(steps[:, i : i + 1], cache))
        mx.eval(out)
        return [np.array(o.astype(mx.float32)) for o in out]

    used = []
    attend = decode_attention.attend
    monkeypatch.setattr(
        decode_attention,
        "attend",
        lambda *a, **kw: used.append(a[3] is not None) or attend(*a, **kw),
    )
    fused = run(True)
    # every decode step of all three layers ran fused, compressed rows included
    assert len(used) == 3 * steps.shape[1] and any(used)
    graph = run(False)
    assert len(used) == 3 * steps.shape[1]
    # the first decode step agrees to fp32 rounding (bf16: exactly); later
    # steps can drift by an FP8 code where a cached KV value sits on a
    # rounding boundary, which the random tiny model then amplifies
    np.testing.assert_array_equal(fused[0], graph[0])
    if first is not None:
        np.testing.assert_allclose(fused[1], graph[1], rtol=first, atol=first)
    for left, right in zip(graph, fused, strict=True):
        np.testing.assert_allclose(right, left, rtol=tol, atol=tol)


@needs_metal
def test_negative_control_hook_still_takes_the_graph_path(monkeypatch):
    model = Model(_tiny_args())
    for layer in model.layers:
        layer.attn._break_rope_inverse = True
    monkeypatch.setattr(
        decode_attention, "attend", lambda *a, **kw: pytest.fail("fused path used")
    )
    cache = model.make_cache(max_seq_len=32)
    model(mx.array([[1, 2, 3]]), cache)
    mx.eval(model(mx.array([[4]]), cache))


def test_supports_declines_what_the_kernels_do_not_cover(monkeypatch):
    monkeypatch.setattr(decode_attention.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(decode_attention.mx.metal, "is_available", lambda: True)
    x = mx.zeros((2, 1, 64), dtype=mx.bfloat16)
    ring = mx.zeros((2, 8, 512))

    def ok(**kw):
        args = dict(x=x, win_kv=ring, heads=64, head_dim=512, rd=64)
        args.update(kw)
        return decode_attention.supports(**args)

    assert ok()
    assert not ok(x=mx.zeros((2, 2, 64), dtype=mx.bfloat16))  # prefill chunk
    assert not ok(x=mx.zeros((2, 1, 64), dtype=mx.int32))
    assert not ok(win_kv=ring.astype(mx.int32))
    assert not ok(win_kv=mx.zeros((3, 8, 512)))  # cache rows != batch
    assert not ok(win_kv=mx.zeros((2, 8, 256)))
    assert not ok(heads=60)
    assert not ok(head_dim=640, win_kv=mx.zeros((2, 8, 640)))
    assert not ok(head_dim=192, win_kv=mx.zeros((2, 8, 192)))
    assert not ok(rd=0)
    assert not ok(rd=63)
    monkeypatch.setattr(decode_attention.mx.metal, "is_available", lambda: False)
    assert not ok()  # no Metal device
    monkeypatch.setattr(decode_attention.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(decode_attention, "ENABLED", False)
    assert not ok()
    monkeypatch.setattr(decode_attention.mx, "default_device", lambda: mx.cpu)
    monkeypatch.setattr(decode_attention, "ENABLED", True)
    assert not ok()


def test_kernel_dispatch_geometry(monkeypatch):
    """Exercise the kernel factories and launch wrappers with stand-ins on any host."""
    built, launches = [], []

    def fake_metal_kernel(**kwargs):
        built.append(kwargs["name"])

        def kernel(**call):
            launches.append(call)
            return [
                mx.zeros(shape, dtype=dtype)
                for shape, dtype in zip(
                    call["output_shapes"], call["output_dtypes"], strict=True
                )
            ]

        return kernel

    decode_attention._kernels.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", fake_metal_kernel)
    try:
        q = mx.zeros((2, 1, 16, 256), dtype=mx.bfloat16)
        ring = mx.zeros((2, 8, 256))
        cos = sin = mx.zeros((1, 32))
        out = decode_attention.attend(
            q, ring, 3, None, None, 4, mx.zeros((16,)), cos, sin, 0.5, 64
        )
        assert out.shape == (2, 1, 16, 256)
        (attend,) = launches
        # D = 256: 32 slices of one 8-wide tile; 16 heads = 2 groups
        assert attend["grid"] == (2 * 32 * 32, 2, 1)
        assert attend["threadgroup"] == (32 * 32, 1, 1)
        assert ("SL", 32) in attend["template"] and ("W", 8) in attend["template"]
        assert np.array(attend["inputs"][4]).tolist() == [3, 4, 0, 4, 0]

        launches.clear()
        comp = mx.zeros((2, 40, 256))
        cidx = mx.full((2, 1, 35), 5, dtype=mx.int32)
        decode_attention.attend(
            q, ring, 200, comp, cidx, 9, mx.zeros((16,)), cos, sin, 0.5, 64
        )
        (attend,) = launches
        assert np.array(attend["inputs"][4]).tolist() == [200, 8, 35, 9, 40]
        assert attend["inputs"][3].shape == (2, 35)

        launches.clear()
        q128 = mx.zeros((1, 1, 8, 128), dtype=mx.bfloat16)
        decode_attention.attend(
            q128,
            mx.zeros((1, 8, 128)),
            3,
            None,
            None,
            4,
            mx.zeros((8,)),
            cos,
            sin,
            0.5,
            64,
        )
        (attend,) = launches
        # D = 128: 16 slices, each still a whole 8-wide tile
        assert attend["grid"] == (16 * 32, 1, 1) and ("SL", 16) in attend["template"]

        launches.clear()
        qr, kv = decode_attention.norms(
            mx.zeros((2, 1, 96), dtype=mx.bfloat16),
            mx.zeros((2, 1, 256), dtype=mx.bfloat16),
            mx.ones((96,)),
            mx.ones((256,)),
            1e-6,
            cos,
            sin,
            64,
            mx.float32,
        )
        (norms,) = launches
        assert norms["grid"] == (512, 2, 2)
        assert ("QR", 96) in norms["template"] and ("HD", 256) in norms["template"]
        assert qr.shape == (2, 1, 96) and qr.dtype == mx.bfloat16
        assert kv.shape == (2, 1, 256) and kv.dtype == mx.float32
        assert built == [
            "rapid_v41_decode_attn",
            "rapid_v41_decode_attn_norms",
        ]
    finally:
        decode_attention._kernels.cache_clear()


@pytest.mark.parametrize("pos", [0, 7, 8, 13])
def test_single_token_window_write_matches_chunk_write(pos):
    args = ModelArgs(n_layers=1, head_dim=16, window_size=8, compress_ratios=(0,))
    single, chunk = LayerCache(2, args, 0, 32), LayerCache(2, args, 0, 32)
    mx.random.seed(pos)
    ring = mx.random.normal((2, 8, 16))
    for cache in (single, chunk):
        cache.win_kv = mx.array(ring)
    kv = mx.random.normal((2, 1, 16)).astype(mx.bfloat16)
    single.write_window(pos, kv)
    # the general path, fed the same token as a one-row chunk
    keep_slots = (pos + mx.arange(1)) % chunk.window
    chunk.win_kv[:, keep_slots] = kv.astype(chunk.dtype)
    assert np.array_equal(np.array(single.win_kv), np.array(chunk.win_kv))
