"""Numerical and fallback contracts for the qualified packed attention path."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.base import quantized_scaled_dot_product_attention as stock

from rapid_mlx.kernels import affine8_attention as attn

pytestmark = pytest.mark.requires_mlx


@pytest.fixture
def qualified():
    if not attn._qualified_runtime():
        pytest.skip("requires the qualified Metal runtime")


def inputs(dtype=mx.bfloat16, tq=4, tk=2049):
    mx.random.seed(81)
    q = mx.random.normal((1, 24, tq, 256)).astype(dtype)
    # Row strides from spare capacity, plus a non-unit last-axis stride.
    kv = []
    for _ in range(2):
        triple = mx.quantize(
            mx.random.normal((1, 4, tk + 33, 256)).astype(dtype), group_size=64, bits=8
        )
        kv.append(
            tuple(
                mx.stack([a, a], axis=-1).reshape(*a.shape[:-1], a.shape[-1] * 2)[
                    ..., :tk, ::2
                ]
                for a in triple
            )
        )
    return q, *kv


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("tq,tk", [(1, 16385), (3, 8193), (4, 2049), (8, 2049)])
@pytest.mark.parametrize("kind", ["causal", "gapped", "all_masked"])
def test_numerics(qualified, dtype, tq, tk, kind):
    q, k, v = inputs(dtype, tq, tk)
    if kind == "causal":
        mask = "causal"
    else:
        mask = mx.arange(tk)[None, :] <= mx.arange(tk - tq, tk)[:, None]
        mask = mask & (mx.arange(tk)[None, :] % 7 != 0)
        if kind == "all_masked":
            mask = mask & (mx.arange(tq)[:, None] != 0)
        mask = mask[None, None, None]
    result = attn.try_affine8_attention(q, k, v, scale=0.0625, mask=mask)
    assert result is not None
    # Stock scales queries in place; never reuse that query for the candidate.
    expected = stock(q + mx.array(0, dtype=dtype), k, v, scale=0.0625, mask=mask)
    mx.eval(result, expected)
    assert bool(mx.all(mx.isfinite(result)))
    rel = mx.linalg.norm(
        result.astype(mx.float32) - expected.astype(mx.float32)
    ) / mx.linalg.norm(expected.astype(mx.float32))
    assert float(rel) < 0.025


def test_declines_and_switch(qualified, monkeypatch):
    q, k, v = inputs()
    cases = [
        dict(bits=4),
        dict(group_size=32),
        dict(mask=mx.zeros((4, 2049))),
        dict(mask=None),
        dict(mask=mx.ones((1, 24, 4, 2049), dtype=mx.bool_)),
    ]
    for kwargs in cases:
        assert attn.try_affine8_attention(q, k, v, scale=0.0625, **kwargs) is None
    monkeypatch.setenv("RAPID_MLX_AFFINE8_ATTENTION", "0")
    assert attn.try_affine8_attention(q, k, v, scale=0.0625, mask="causal") is None


def test_both_dispatch_routes(qualified, monkeypatch):
    import mlx_lm.models.base as lm

    from rapid_mlx.models.mlx_vlm_vendored.models import base as vlm
    from rapid_mlx.quantized_batch_cache import (
        _install_batched_mask_safe_quantized_sdpa,
    )

    _install_batched_mask_safe_quantized_sdpa()
    q, k, v = inputs()
    sentinel = mx.ones(q.shape, dtype=q.dtype)
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(attn, "try_affine8_attention", fake)
    for base in (lm, vlm):
        assert (
            base.quantized_scaled_dot_product_attention(
                q, k, v, scale=0.0625, mask="causal"
            )
            is sentinel
        )
    assert len(calls) == 2


def test_runtime_and_geometry_fallbacks(qualified, monkeypatch):
    q, k, v = inputs()
    for bad_q in (
        mx.broadcast_to(q, (2, 24, 4, 256)),
        q[:, :, :, :128],
        q.astype(mx.float32),
    ):
        assert (
            attn.try_affine8_attention(bad_q, k, v, scale=0.0625, mask="causal") is None
        )
    assert (
        attn.try_affine8_attention(
            q,
            tuple(a[:, :, :1000] for a in k),
            tuple(a[:, :, :1000] for a in v),
            scale=0.0625,
            mask="causal",
        )
        is None
    )
    monkeypatch.setattr(attn, "_qualified_runtime", lambda: False)
    assert attn.try_affine8_attention(q, k, v, scale=0.0625, mask="causal") is None


def test_compile_failure_falls_back_once(qualified, monkeypatch):
    q, k, v = inputs()
    calls = []

    def unavailable():
        calls.append(True)
        raise RuntimeError("test compilation failure")

    monkeypatch.setattr(attn, "_kernel", unavailable)
    monkeypatch.setattr(attn, "_DISABLED", False)
    for _ in range(2):
        assert attn.try_affine8_attention(q, k, v, scale=0.0625, mask="causal") is None
    assert len(calls) == 1


def test_lazy_compile_failure_falls_back(qualified, monkeypatch):
    q, k, v = inputs()

    def fail_eval(*args):
        raise RuntimeError("test lazy compiler failure")

    monkeypatch.setattr(attn, "_WARMED_VARIANTS", set())
    monkeypatch.setattr(attn, "_DISABLED", False)
    monkeypatch.setattr(mx, "eval", fail_eval)
    assert attn.try_affine8_attention(q, k, v, scale=0.0625, mask="causal") is None
    assert attn._DISABLED
