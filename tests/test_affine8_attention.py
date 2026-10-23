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


@pytest.fixture
def dispatcher(monkeypatch):
    """Exercise host contracts on every Metal CI host without compiling MPP.

    Numerical tests above still use the real kernel on the qualified host.
    Fake split partials here have a known stable weighted mean, so this also
    checks the host-side reduction independently of the Metal implementation.
    """
    if not mx.metal.is_available():
        pytest.skip("requires MLX Metal arrays")
    factory = attn._kernel
    factory.cache_clear()
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    monkeypatch.setattr(attn, "_qualified_runtime", lambda: True)
    monkeypatch.setattr(attn, "_DISABLED", False)
    monkeypatch.setattr(attn, "_WARMED_VARIANTS", set())
    monkeypatch.setenv("RAPID_MLX_AFFINE8_ATTENTION", "1")
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        shape = kwargs["output_shapes"][0]
        first = mx.arange(shape[1])[None, :, None] == 0
        maxima = mx.broadcast_to(mx.where(first, 10000.693147, 10000.0), shape)
        denominator = mx.ones(shape)
        partial = mx.broadcast_to(
            mx.where(first[..., None], 3.0, 1.0), kwargs["output_shapes"][2]
        )
        return maxima, denominator, partial

    def construct(**kwargs):
        # The shipped source must be loadable, and the wrapper must preserve
        # cache strides rather than materializing a full contiguous history.
        assert "MetalPerformancePrimitives" in kwargs["header"]
        assert kwargs["ensure_row_contiguous"] is False
        return execute

    monkeypatch.setattr(mx.fast, "metal_kernel", construct)
    try:
        yield calls
    finally:
        factory.cache_clear()
        mx.set_default_device(previous)


def test_declines_and_switch(dispatcher, monkeypatch):
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


def test_both_dispatch_routes(dispatcher, monkeypatch):
    import mlx_lm.models.base as lm

    from rapid_mlx.models.mlx_vlm_vendored.models import base as vlm
    from rapid_mlx.quantized_batch_cache import (
        _install_batched_mask_safe_quantized_sdpa,
    )

    # Installation is process-global in production; restore it after this test.
    monkeypatch.setattr(
        lm,
        "quantized_scaled_dot_product_attention",
        lm.quantized_scaled_dot_product_attention,
    )
    monkeypatch.setattr(
        lm,
        "_rapid_batched_mask_safe_qsdpa",
        getattr(lm, "_rapid_batched_mask_safe_qsdpa", False),
        raising=False,
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


def test_runtime_and_geometry_fallbacks(dispatcher, monkeypatch):
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


def test_compile_failure_falls_back_once(dispatcher, monkeypatch):
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


def test_lazy_compile_failure_falls_back(dispatcher, monkeypatch):
    q, k, v = inputs()

    def fail_eval(*args):
        raise RuntimeError("test lazy compiler failure")

    monkeypatch.setattr(attn, "_WARMED_VARIANTS", set())
    monkeypatch.setattr(attn, "_DISABLED", False)
    monkeypatch.setattr(mx, "eval", fail_eval)
    assert attn.try_affine8_attention(q, k, v, scale=0.0625, mask="causal") is None
    assert attn._DISABLED


@pytest.mark.parametrize("boolean_mask", [False, True])
def test_stable_split_merge_and_padded_query(dispatcher, boolean_mask):
    q, k, v = inputs(tq=3, tk=8193)
    mask = mx.ones((1, 1, 1, 3, 8193), dtype=mx.bool_) if boolean_mask else "causal"
    result = attn.try_affine8_attention(q, k, v, scale=0.0625, mask=mask)
    assert result is not None and result.shape == q.shape
    call = dispatcher[-1]
    splits = call["output_shapes"][0][1]
    # One split has weight 2 and value 3; all others weight 1 and value 1.
    # Maxima near 10000 require a stable merge, not direct exponentiation.
    assert mx.allclose(
        result.astype(mx.float32), mx.array((splits + 5) / (splits + 1)), atol=0.01
    )
    padded = call["inputs"][0]
    assert padded.shape == (4, 24, 256)
    assert mx.allclose(padded[:, :18], q.reshape(4, 18, 256))
    assert bool(mx.all(padded[:, 18:] == 0))
    assert call["inputs"][1] is k[0]
    assert call["inputs"][4] is v[0]
    if boolean_mask:
        assert call["inputs"][-2].shape == (3, 8193)
        assert call["inputs"][-1].tolist() == [True, True, True]
    # Subsequent invocations of the same template stay lazy and reusable.
    again = attn.try_affine8_attention(q, k, v, scale=0.0625, mask=mask)
    assert mx.allclose(result, again)


def test_none_mask_single_query(dispatcher):
    q, k, v = inputs(tq=1, tk=16385)
    assert attn.try_affine8_attention(q, k, v, scale=0.0625) is not None


def test_malformed_cache_and_geometry_never_reach_kernel(dispatcher):
    q, k, v = inputs()
    cases = [
        (q, k[:2], v, "causal"),
        (q, (object(), *k[1:]), v, "causal"),
        (q, tuple(a[0] for a in k), v, "causal"),
        (q[:, :23], k, v, "causal"),
        (q[:, :0], k, v, "causal"),
        (q[:, :, :0], k, v, "causal"),
        (mx.ones((1, 64, 4, 256), dtype=q.dtype), k, v, "causal"),
        (q, (k[0].astype(mx.float32), *k[1:]), v, "causal"),
        (q, k, v, "unsupported"),
        (q, k, v, mx.ones((2049,), dtype=mx.bool_)),
    ]
    for query, keys, values, mask in cases:
        assert (
            attn.try_affine8_attention(query, keys, values, scale=0.0625, mask=mask)
            is None
        )
    assert dispatcher == []


def test_runtime_qualification_does_not_expand_silently(monkeypatch):
    check = attn._qualified_runtime.__wrapped__
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "__version__", "0.32.3")
    monkeypatch.setattr(attn.platform, "mac_ver", lambda: ("26.5.2", (), ""))
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M3 Ultra"})
    assert check()
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M2"})
    assert not check()
