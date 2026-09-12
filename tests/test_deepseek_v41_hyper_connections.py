from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from vllm_mlx.models.deepseek_v41_native import hyper_connections as hc
from vllm_mlx.models.deepseek_v41_native import model as model_module


def _requires_metal() -> None:
    if not hc._metal_fast_path_available():
        pytest.skip("custom hyper-connection kernels require Metal")


def _reference_hc_mixes(x, fn, scale, base, hc_mult, iters, norm_eps, hc_eps):
    flat = x.reshape(*x.shape[:2], -1).astype(mx.float32)
    inverse_rms = mx.rsqrt(mx.mean(mx.square(flat), axis=-1, keepdims=True) + norm_eps)
    mixes = (flat @ fn.astype(mx.float32).T) * inverse_rms
    m = mixes.astype(mx.float32)
    pre = mx.sigmoid(m[..., :hc_mult] * scale[0] + base[:hc_mult]) + hc_eps
    post = 2 * mx.sigmoid(
        m[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = (m[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).reshape(
        *m.shape[:-1], hc_mult, hc_mult
    )
    comb = mx.softmax(comb, axis=-1) + hc_eps
    return pre, post, hc._sinkhorn_reference(comb, hc_eps, iters)


def test_compiled_hc_mixes_matches_reference_decode_shape() -> None:
    mx.random.seed(40)
    x = mx.random.uniform(shape=(1, 1, 4, 64)).astype(mx.bfloat16)
    fn = mx.random.uniform(shape=(24, 256)).astype(mx.float32)
    scale = mx.random.uniform(shape=(3,)).astype(mx.float32)
    base = mx.random.uniform(shape=(24,)).astype(mx.float32)

    actual = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    expected = _reference_hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    mx.eval(*actual, *expected)

    for accelerated, reference in zip(actual, expected, strict=True):
        assert mx.allclose(accelerated, reference, rtol=0, atol=2e-7).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_fused_hc_post_matches_reference(dtype) -> None:
    _requires_metal()
    mx.random.seed(41)
    x = mx.random.uniform(shape=(2, 3, 64)).astype(dtype)
    residual = mx.random.uniform(shape=(2, 3, 4, 64)).astype(dtype)
    post = mx.random.uniform(shape=(2, 3, 4)).astype(mx.float32)
    comb = mx.random.uniform(shape=(2, 3, 4, 4)).astype(mx.float32)

    actual = hc.hc_post(x, residual, post, comb)
    expected = hc._hc_post_reference(x, residual, post, comb)
    mx.eval(actual, expected)

    assert actual.shape == residual.shape
    atol = 3e-7 if dtype == mx.float32 else 0
    assert mx.allclose(actual, expected, rtol=0, atol=atol).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_fused_hc_pre_norm_matches_rounded_reference(dtype) -> None:
    _requires_metal()
    mx.random.seed(42)
    x = mx.random.uniform(shape=(2, 3, 4, 64)).astype(dtype)
    pre = mx.random.uniform(shape=(2, 3, 4)).astype(mx.float32)
    weight = mx.random.uniform(shape=(64,)).astype(mx.float32)

    actual = hc.hc_pre_norm(x, pre, weight, 1e-6)
    expected = hc._hc_pre_norm_reference(x, pre, weight, 1e-6)
    mx.eval(actual, expected)

    assert actual.shape == (2, 3, 64)
    atol = 3e-7 if dtype == mx.float32 else 0
    assert mx.allclose(actual, expected, rtol=0, atol=atol).item()


def test_fused_sinkhorn_matches_reference() -> None:
    _requires_metal()
    mx.random.seed(43)
    comb = mx.random.uniform(shape=(2, 3, 4, 4)).astype(mx.float32)

    actual = hc._sinkhorn(comb, 1e-6, 20)
    expected = hc._sinkhorn_reference(comb, 1e-6, 20)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, rtol=0, atol=2e-7).item()


def test_non_release_layouts_stay_on_portable_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        hc,
        "_fused_hc_post",
        lambda *_args: pytest.fail("unexpected Metal post-mix path"),
    )
    monkeypatch.setattr(
        hc,
        "_fused_hc_pre_norm",
        lambda *_args: pytest.fail("unexpected Metal pre-norm path"),
    )
    x = mx.ones((1, 1, 32), dtype=mx.float32)
    residual = mx.ones((1, 1, 2, 32), dtype=mx.float32)
    post = mx.ones((1, 1, 2), dtype=mx.float32)
    comb = mx.ones((1, 1, 2, 2), dtype=mx.float32)
    pre = mx.ones((1, 1, 2), dtype=mx.float32)
    weight = mx.ones((32,), dtype=mx.float32)

    expanded = hc.hc_post(x, residual, post, comb)
    normalized = hc.hc_pre_norm(residual, pre, weight, 1e-6)
    mx.eval(expanded, normalized)

    assert expanded.shape == residual.shape
    assert normalized.shape == x.shape


def test_empty_release_layout_stays_on_portable_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        hc,
        "_fused_hc_post",
        lambda *_args: pytest.fail("unexpected Metal post-mix path"),
    )
    monkeypatch.setattr(
        hc,
        "_fused_hc_pre_norm",
        lambda *_args: pytest.fail("unexpected Metal pre-norm path"),
    )
    x = mx.zeros((1, 0, 16), dtype=mx.float32)
    residual = mx.zeros((1, 0, 4, 16), dtype=mx.float32)
    post = mx.zeros((1, 0, 4), dtype=mx.float32)
    comb = mx.zeros((1, 0, 4, 4), dtype=mx.float32)
    pre = mx.zeros((1, 0, 4), dtype=mx.float32)
    weight = mx.ones((16,), dtype=mx.float32)

    expanded = hc.hc_post(x, residual, post, comb)
    normalized = hc.hc_pre_norm(residual, pre, weight, 1e-6)
    mx.eval(expanded, normalized)

    assert expanded.shape == residual.shape
    assert normalized.shape == x.shape


def test_release_layout_stays_portable_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        hc,
        "_fused_hc_post",
        lambda *_args: pytest.fail("unexpected Metal post-mix path"),
    )
    monkeypatch.setattr(
        hc,
        "_fused_hc_pre_norm",
        lambda *_args: pytest.fail("unexpected Metal pre-norm path"),
    )
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        x = mx.ones((1, 1, 16), dtype=mx.float32)
        residual = mx.ones((1, 1, 4, 16), dtype=mx.float32)
        post = mx.ones((1, 1, 4), dtype=mx.float32)
        comb = mx.ones((1, 1, 4, 4), dtype=mx.float32)
        pre = mx.ones((1, 1, 4), dtype=mx.float32)
        weight = mx.ones((16,), dtype=mx.float32)

        expanded = hc.hc_post(x, residual, post, comb)
        normalized = hc.hc_pre_norm(residual, pre, weight, 1e-6)
        sinkhorn = hc._sinkhorn(comb, 1e-6, 20)
        mx.eval(expanded, normalized, sinkhorn)
    finally:
        mx.set_default_device(previous)

    assert expanded.shape == residual.shape
    assert normalized.shape == x.shape
    assert sinkhorn.shape == comb.shape


def test_block_uses_fused_pre_norm_for_attention_and_ffn(monkeypatch) -> None:
    calls = []
    pre = mx.ones((1, 1, 4), dtype=mx.float32)
    post = mx.ones((1, 1, 4), dtype=mx.float32)
    comb = mx.eye(4, dtype=mx.float32)[None, None]

    def fake_mixes(*_args):
        calls.append("mixes")
        return pre, post, comb

    def fake_pre_norm(x, mix, weight, eps):
        calls.append(("pre_norm", mix, weight, eps))
        return mx.sum(x, axis=2)

    def fake_post(value, _residual, _post, _comb):
        calls.append("post")
        return mx.repeat(value[:, :, None, :], 4, axis=2)

    monkeypatch.setattr(model_module, "hc_mixes", fake_mixes)
    monkeypatch.setattr(model_module, "hc_pre_norm", fake_pre_norm)
    monkeypatch.setattr(model_module, "hc_post", fake_post)

    block = SimpleNamespace(
        hc_attn_fn=mx.zeros((24, 32)),
        hc_ffn_fn=mx.zeros((24, 32)),
        hc_attn_scale=mx.ones((3,)),
        hc_ffn_scale=mx.ones((3,)),
        hc_attn_base=mx.zeros((24,)),
        hc_ffn_base=mx.zeros((24,)),
        hc_mult=4,
        hc_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
        attn_norm=SimpleNamespace(weight=mx.ones((8,)), eps=1e-6),
        ffn_norm=SimpleNamespace(weight=mx.ones((8,)), eps=1e-6),
        attn=lambda value, *_args: value + 1,
        ffn=lambda value: value + 2,
    )
    x = mx.ones((1, 1, 4, 8), dtype=mx.float32)

    output, next_pre = model_module.Block.__call__(block, x, pre, 0, None, None)
    mx.eval(output, next_pre)

    assert output.shape == x.shape
    assert next_pre is pre
    assert calls[0] == "mixes"
    assert calls[2:4] == ["post", "mixes"]
    assert calls[-1] == "post"
    pre_norm_calls = [call for call in calls if isinstance(call, tuple)]
    assert pre_norm_calls[0][1] is pre
    assert pre_norm_calls[1][1] is pre
    assert pre_norm_calls[0][2] is block.attn_norm.weight
    assert pre_norm_calls[1][2] is block.ffn_norm.weight
