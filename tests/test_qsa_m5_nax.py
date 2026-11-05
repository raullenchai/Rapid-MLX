from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", exc_type=ImportError)

from rapid_mlx.kernels import qsa_m5_nax as nax
from rapid_mlx.kernels.qsa_block_sparse import block_sparse_attention
from rapid_mlx.kernels.qsa_stage1 import qsa_stage1_select

pytestmark = pytest.mark.requires_mlx
needs_m5 = pytest.mark.skipif(
    not nax.hardware_supported(), reason="needs M5 tensor units"
)


def test_opt_in_and_non_m5_fallback(monkeypatch):
    monkeypatch.delenv(nax.ENABLE_ENV, raising=False)
    monkeypatch.setattr(nax, "hardware_supported", lambda: True)
    assert not nax.eligible(512, batch_size=1)
    monkeypatch.setenv(nax.ENABLE_ENV, "1")
    monkeypatch.setattr(nax, "hardware_supported", lambda: False)
    monkeypatch.setattr(
        nax, "_scores_kernel", lambda: pytest.fail("compiled on non-M5")
    )
    assert not nax.eligible(512, batch_size=1)
    # No tensor evaluation or fallback graph construction is allowed here.
    assert (
        nax.select(SimpleNamespace(ndim=4, shape=(1, 512, 4, 128)), None, None) is None
    )


def test_envelope_and_layout_declines(monkeypatch):
    monkeypatch.setenv(nax.ENABLE_ENV, "1")
    monkeypatch.setattr(nax, "hardware_supported", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert nax.eligible(64, batch_size=1)
    for batch, length, training in [(2, 512, False), (1, 1, False), (1, 512, True)]:
        assert not nax.eligible(length, batch_size=batch, training=training)
    q = mx.zeros((1, 64, 4, 128), dtype=mx.float16)
    assert nax.select(q, mx.zeros((1, 513, 128)), mx.zeros((1, 64))) is None
    monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)
    assert not nax.eligible(64, batch_size=1)


def _view(shape, layout):
    if layout == "cache_view":
        x = mx.random.normal((*shape[:-2], shape[-2] + 13, shape[-1])).astype(
            mx.bfloat16
        )
        return x[..., : shape[-2], :]
    if layout == "strided":
        x = mx.random.normal((*shape[:-1], shape[-1] * 2)).astype(mx.bfloat16)
        return x[..., ::2]
    if layout == "misaligned":
        x = mx.random.normal((*shape[:-1], shape[-1] + 1)).astype(mx.bfloat16)
        return x[..., 1:]
    return mx.random.normal(shape).astype(mx.bfloat16)


def _score_reference(q, k, fp32=False):
    if fp32:
        q, k = q.astype(mx.float32), k.astype(mx.float32)
    s = q.transpose(0, 2, 1, 3) @ k.swapaxes(-1, -2)[:, None]
    return mx.maximum(s.astype(mx.float32), 0).sum(axis=1) / math.sqrt(128)


def _agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.mean(
        [
            len(set(x) & set(y)) / 512
            for x, y in zip(a.reshape(-1, 512), b.reshape(-1, 512), strict=True)
        ]
    )


@needs_m5
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "misaligned", "cache_view"]
)
@pytest.mark.parametrize("length,blocks,offset", [(64, 513, 2051), (67, 1025, 4103)])
def test_scores_and_selection_against_current_and_fp32(
    monkeypatch, layout, length, blocks, offset
):
    monkeypatch.setenv(nax.ENABLE_ENV, "1")
    mx.random.seed(23)
    q = _view((1, length, 4, 128), layout)
    k = _view((1, blocks, 128), layout)
    pos = (offset + mx.arange(length))[None].astype(mx.int32)
    actual = nax.scores(q, k)
    current = _score_reference(q, k)
    fp32 = _score_reference(q, k, fp32=True)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(current), atol=0.03, rtol=0.01
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(fp32), atol=0.03, rtol=0.01
    )
    ids = nax.select(q, k, pos)
    old = qsa_stage1_select(q, k, pos, block_topk=512, compress_ratio=4)
    from rapid_mlx.kernels.qsa_stage1 import _select_scores

    ref = _select_scores(fp32[0], pos[0], topk=512, compress_ratio=4)[None]
    assert _agreement(ids, old) >= 0.995
    assert _agreement(ids, ref) >= 0.99
    assert np.asarray(ids).max() < blocks


@needs_m5
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "misaligned", "cache_view"]
)
@pytest.mark.parametrize("offset", [0, 2051, 4099])
def test_attention_chunked_tail_and_empty_rows(monkeypatch, layout, offset):
    monkeypatch.setenv(nax.ENABLE_ENV, "1")
    mx.random.seed(31)
    length, total = 67, offset + 67
    q = _view((1, 24, length, 256), layout)
    k = _view((1, 2, total, 256), layout)
    v = _view((1, 2, total, 256), layout)
    pos = np.arange(length) + offset
    counts = np.minimum((pos + 1) // 4, 512).astype(np.int32)[None]
    counts[0, 0] = 0  # An invalid/padded query must produce zero, not NaN.
    starts = np.broadcast_to(
        np.arange(512, dtype=np.int32) * 4, (1, length, 512)
    ).copy()
    rng = np.random.default_rng(31)
    for t in range(1, length):
        available = (int(pos[t]) + 1) // 4
        n = int(counts[0, t])
        starts[0, t, :n] = np.sort(rng.choice(available, n, replace=False)) * 4
    tails = (((pos + 1) // 4 * 4)[:, None] + np.arange(4)).astype(np.int32)[None]
    tc = ((pos + 1) % 4).astype(np.int32)[None]
    tc[0, 0] = 0
    arrays = [mx.array(a) for a in (starts, counts, tails, tc)]
    actual = nax.attention(q, k, v, *arrays, scale=256**-0.5)
    old = block_sparse_attention(q, k, v, *arrays, block_size=4)
    # NumPy cannot import the MLX bf16 buffer; widen before crossing the bridge.
    a, b = [np.asarray(x.astype(mx.float32)) for x in (actual, old)]
    assert np.isfinite(a).all()
    assert np.sqrt(np.mean((a - b) ** 2)) < 0.003
    # Independent FP32 softmax oracle, one query at a time to bound memory.
    qn, kn, vn = [np.asarray(x.astype(mx.float32)) for x in (q, k, v)]
    ref = np.zeros_like(a)
    for t in range(1, length):
        indices = np.concatenate(
            [
                (starts[0, t, : counts[0, t], None] + np.arange(4)).reshape(-1),
                tails[0, t, : tc[0, t]],
            ]
        )
        if not len(indices):
            continue
        for kvh in range(2):
            sl = slice(kvh * 12, (kvh + 1) * 12)
            s = qn[0, sl, t] @ kn[0, kvh, indices].T / 16
            p = np.exp(s - s.max(axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            ref[0, sl, t] = p @ vn[0, kvh, indices]
    assert np.sqrt(np.mean((a - ref) ** 2)) < 0.003
    np.testing.assert_allclose(a[:, :, 0], 0, atol=0)


@needs_m5
def test_zero_score_ties_and_partial_causal_rows(monkeypatch):
    monkeypatch.setenv(nax.ENABLE_ENV, "1")
    q = mx.zeros((1, 64, 4, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 1025, 128), dtype=mx.bfloat16)
    pos = (2017 + mx.arange(64))[None].astype(mx.int32)
    first = nax.select(q, k, pos)
    second = nax.select(q, k, pos)
    old = qsa_stage1_select(q, k, pos, block_topk=512, compress_ratio=4)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    np.testing.assert_array_equal(np.sort(np.asarray(first)), np.sort(np.asarray(old)))


@pytest.fixture
def default_qsa_routes(monkeypatch):
    for flag in (
        "RAPID_MLX_QSA_STAGE1",
        "RAPID_MLX_QSA_BLOCK_SPARSE",
        "RAPID_MLX_QSA_INDEXED_SPLITK",
    ):
        monkeypatch.setenv(flag, "0")


def test_integrated_non_m5_fallback(monkeypatch, default_qsa_routes):
    from mlx_lm.models.cache import KVCache

    from rapid_mlx.models.qwen4_exp import QSAAttention
    from rapid_mlx.models.qwen4_exp_cache import QSAIndexCache
    from scripts.bench_qwen4_qsa_stage1_indexer import model_args

    monkeypatch.setattr(nax, "hardware_supported", lambda: False)
    monkeypatch.setattr(
        nax, "_scores_kernel", lambda: pytest.fail("unexpected NAX compile")
    )
    monkeypatch.setattr(
        nax, "_attention_kernel", lambda: pytest.fail("unexpected NAX compile")
    )
    mx.random.seed(41)
    layer = QSAAttention(model_args())
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    x = mx.random.normal((1, 2112, 256)).astype(mx.bfloat16)
    results = []
    for enabled in ("0", "1"):
        monkeypatch.setenv(nax.ENABLE_ENV, enabled)
        cache = [KVCache(), QSAIndexCache(4)]
        mx.eval(layer(x[:, :2048], cache=cache))
        results.append(np.asarray(layer(x[:, 2048:], cache=cache).astype(mx.float32)))
    np.testing.assert_array_equal(*results)
    assert layer.m5_nax_route_constructions == 0
    assert layer.indexer.m5_nax_route_constructions == 0


@needs_m5
def test_integrated_m5_prefill_parity(monkeypatch, default_qsa_routes):
    from mlx_lm.models.cache import KVCache

    from rapid_mlx.models.qwen4_exp import QSAAttention
    from rapid_mlx.models.qwen4_exp_cache import QSAIndexCache
    from scripts.bench_qwen4_qsa_stage1_indexer import model_args

    mx.random.seed(47)
    layer = QSAAttention(model_args())
    layer.set_dtype(mx.bfloat16)
    layer.eval()
    x = mx.random.normal((1, 2183, 256)).astype(mx.bfloat16)
    outputs = []
    for enabled in ("0", "1"):
        monkeypatch.setenv(nax.ENABLE_ENV, enabled)
        cache = [KVCache(), QSAIndexCache(4)]
        mx.eval(layer(x[:, :2051], cache=cache))
        outputs.append(np.asarray(layer(x[:, 2051:], cache=cache).astype(mx.float32)))
        assert (layer.m5_nax_route_constructions > 0) == (enabled == "1")
        assert (layer.indexer.m5_nax_route_constructions > 0) == (enabled == "1")
    a, b = outputs
    assert np.isfinite(b).all()
    relative_rms = np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a * a)), 1e-6)
    assert relative_rms < 0.03
    assert layer.m5_nax_route_constructions > 0
    assert layer.indexer.m5_nax_route_constructions > 0
