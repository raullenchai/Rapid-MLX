from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native import hyper_connections as hc
from rapid_mlx.models.deepseek_v41_native import model as model_module


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


# The release coefficient layout at the smallest width the fused mixes take:
# four copies of 1024 values (whole 4096-wide reduction blocks).
_MIX_WIDTH = 1024


def _mix_inputs(rows: int, dtype, seed: int):
    mx.random.seed(seed)
    spread = mx.power(2.0, mx.arange(rows) - 2.0)[:, None, None, None]
    x = (mx.random.normal((rows, 1, 4, _MIX_WIDTH)) * spread).astype(dtype)
    fn = mx.random.normal((24, 4 * _MIX_WIDTH)) * 0.02
    scale = mx.array([0.6, 0.8, 1.7])
    base = mx.random.normal((24,)) * 0.5
    return x, fn, scale, base


@pytest.fixture
def fresh_mix_verdicts(monkeypatch):
    monkeypatch.setattr(hc, "_HC_MIXES_VERDICTS", {})


@pytest.mark.parametrize("param_dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_fused_hc_mixes_rows_equal_single_row_stock(
    fresh_mix_verdicts, rows: int, dtype, param_dtype
) -> None:
    _requires_metal()
    x, fn, scale, base = _mix_inputs(rows, dtype, seed=50 + rows)
    # the checkpoint stores the coefficient parameters in BF16
    fn, scale, base = (p.astype(param_dtype) for p in (fn, scale, base))

    actual = hc._fused_hc_mixes(x, fn, scale, base, 20, 1e-6, 1e-6)
    expected = [
        hc._hc_mixes_reference(x[r : r + 1], fn, scale, base, 4, 20, 1e-6, 1e-6)
        for r in range(rows)
    ]
    mx.eval(actual, expected)

    for part, shape in zip(actual, [(rows, 1, 4), (rows, 1, 4), (rows, 1, 4, 4)]):
        assert part.shape == shape and part.dtype == mx.float32
    for r in range(rows):
        for fused, stock in zip(actual, expected[r], strict=True):
            assert mx.array_equal(fused[r : r + 1], stock).item()


def test_hc_mixes_takes_the_qualified_kernels(fresh_mix_verdicts, monkeypatch) -> None:
    _requires_metal()
    x, fn, scale, base = _mix_inputs(4, mx.bfloat16, seed=60)
    calls = []
    original = hc._fused_hc_mixes

    def spy(value, *args):
        calls.append(value.shape)
        return original(value, *args)

    monkeypatch.setattr(hc, "_fused_hc_mixes", spy)
    decode = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    assert calls[-1] == x.shape
    assert list(hc._HC_MIXES_VERDICTS.values()) == [True]
    # prefill rows stay on the stock graph
    prefill = mx.concatenate([x, x, x], axis=1)
    calls.clear()
    stock = hc.hc_mixes(prefill, fn, scale, base, 4, 20, 1e-6, 1e-6)
    assert calls == []
    mx.eval(decode, stock)
    assert stock[2].shape == (4, 3, 4, 4)


def test_hc_mixes_self_check_mismatch_keeps_the_stock_path(
    fresh_mix_verdicts, monkeypatch
) -> None:
    _requires_metal()
    x, fn, scale, base = _mix_inputs(1, mx.bfloat16, seed=61)
    real = hc._hc_mixes_finish_kernel()

    def wrong(**kwargs):
        pre, post, comb = real(**kwargs)
        return [pre, post, comb + 1]

    monkeypatch.setattr(hc, "_hc_mixes_finish_kernel", lambda: wrong)
    actual = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    expected = hc._hc_mixes_reference(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    assert list(hc._HC_MIXES_VERDICTS.values()) == [False]
    for a, e in zip(actual, expected, strict=True):
        assert mx.array_equal(a, e).item()


def test_hc_mixes_launch_failure_keeps_the_stock_path(
    fresh_mix_verdicts, monkeypatch
) -> None:
    _requires_metal()
    x, fn, scale, base = _mix_inputs(1, mx.float32, seed=62)

    def broken(**kwargs):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(hc, "_hc_mixes_partial_kernel", lambda: broken)
    actual = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    expected = hc._hc_mixes_reference(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    assert list(hc._HC_MIXES_VERDICTS.values()) == [False]
    for a, e in zip(actual, expected, strict=True):
        assert mx.array_equal(a, e).item()


def test_hc_mixes_support_declines_off_the_release_layout(monkeypatch) -> None:
    monkeypatch.setattr(hc, "_metal_fast_path_available", lambda: True)
    x, fn, scale, base = _mix_inputs(2, mx.bfloat16, seed=63)
    assert hc._hc_mixes_supported(x, fn, scale, base, 4)
    assert not hc._hc_mixes_supported(x, fn, scale, base, 2)
    assert not hc._hc_mixes_supported(x.astype(mx.int32), fn, scale, base, 4)
    assert not hc._hc_mixes_supported(x[:, :, :, :512], fn, scale, base, 4)
    assert not hc._hc_mixes_supported(x, fn[:, :-1], scale, base, 4)
    assert hc._hc_mixes_supported(x, fn.astype(mx.bfloat16), scale, base, 4)
    assert not hc._hc_mixes_supported(x, fn.astype(mx.int32), scale, base, 4)
    assert not hc._hc_mixes_supported(x, fn, scale[:2], base, 4)
    assert not hc._hc_mixes_supported(x, fn, scale, base.astype(mx.int32), 4)
    assert not hc._hc_mixes_supported(x[:0], fn, scale, base, 4)
    assert not hc._hc_mixes_supported(mx.repeat(x, 5, axis=0), fn, scale, base, 4)
    monkeypatch.setattr(hc, "HC_MIXES_ENABLED", False)
    assert not hc._hc_mixes_supported(x, fn, scale, base, 4)
    monkeypatch.setattr(hc, "HC_MIXES_ENABLED", True)
    monkeypatch.setattr(hc, "_metal_fast_path_available", lambda: False)
    assert not hc._hc_mixes_supported(x, fn, scale, base, 4)


@pytest.mark.parametrize(
    "lead, width",
    [((1, 3), _MIX_WIDTH), ((2, 3), _MIX_WIDTH), ((2, 1), 2 * _MIX_WIDTH)],
)
def test_hc_mixes_dispatch_matches_single_row_stock(
    fresh_mix_verdicts, monkeypatch, lead, width
) -> None:
    """Speculative-verify rows, batched verify rows and a multi-block width
    through the public dispatch, from a strided stream."""
    _requires_metal()
    mx.random.seed(70)
    b, s = lead
    wide = mx.random.normal((b, s, 4, 2 * width)).astype(mx.bfloat16)
    x = wide[..., ::2]  # non-contiguous view
    fn = (mx.random.normal((24, 4 * width)) * 0.02).astype(mx.bfloat16)
    scale = mx.array([0.6, 0.8, 1.7]).astype(mx.bfloat16)
    base = (mx.random.normal((24,)) * 0.5).astype(mx.bfloat16)

    calls = []
    original = hc._fused_hc_mixes

    def spy(value, *args):
        calls.append(value.shape)
        return original(value, *args)

    monkeypatch.setattr(hc, "_fused_hc_mixes", spy)
    actual = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    assert x.shape in calls
    mx.eval(actual)
    assert actual[2].shape == (b, s, 4, 4)
    for i in range(b):
        for j in range(s):
            row = mx.array(x[i : i + 1, j : j + 1])
            stock = hc._hc_mixes_reference(row, fn, scale, base, 4, 20, 1e-6, 1e-6)
            for fused, ref in zip(actual, stock, strict=True):
                assert mx.array_equal(fused[i : i + 1, j : j + 1], ref).item()


@pytest.mark.parametrize("changed", [0, 1, 2])
def test_hc_mixes_verdicts_are_per_parameter_dtype(
    fresh_mix_verdicts, monkeypatch, changed: int
) -> None:
    """A verdict for one parameter dtype never authorizes another, whichever
    of the projection, scale or base differs."""
    _requires_metal()
    x, fn, scale, base = _mix_inputs(1, mx.bfloat16, seed=71)
    params = [p.astype(mx.bfloat16) for p in (fn, scale, base)]
    hc.hc_mixes(x, *params, 4, 20, 1e-6, 1e-6)
    assert list(hc._HC_MIXES_VERDICTS.values()) == [True]
    params[changed] = (fn, scale, base)[changed]  # FP32
    fn, scale, base = params

    real = hc._hc_mixes_finish_kernel()

    def wrong(**kwargs):
        pre, post, comb = real(**kwargs)
        return [pre, post, comb + 1]

    monkeypatch.setattr(hc, "_hc_mixes_finish_kernel", lambda: wrong)
    actual = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    assert sorted(hc._HC_MIXES_VERDICTS.values()) == [False, True]
    expected = hc._hc_mixes_reference(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    for a, e in zip(actual, expected, strict=True):
        assert mx.array_equal(a, e).item()


def test_hc_mixes_qualifies_inside_a_compiled_caller(fresh_mix_verdicts) -> None:
    """A cold verdict reached from an mx.compile trace qualifies on concrete
    probe inputs instead of evaluating the caller's stream placeholder."""
    _requires_metal()
    x, fn, scale, base = _mix_inputs(2, mx.bfloat16, seed=72)
    fn, scale, base = (p.astype(mx.bfloat16) for p in (fn, scale, base))

    @mx.compile
    def traced(value):
        return hc.hc_mixes(value, fn, scale, base, 4, 20, 1e-6, 1e-6)

    actual = traced(x)
    assert list(hc._HC_MIXES_VERDICTS.values()) == [True]
    eager = hc.hc_mixes(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    for a, e in zip(actual, eager, strict=True):
        assert mx.array_equal(a, e).item()


def test_hc_mixes_traced_parameters_take_the_stock_graph(fresh_mix_verdicts) -> None:
    """Parameters that are trace placeholders cannot be layout-checked: the
    stock graph runs and no verdict is recorded."""
    _requires_metal()
    x, fn, scale, base = _mix_inputs(2, mx.bfloat16, seed=73)

    @mx.compile
    def traced(value, weight, s, b):
        return hc.hc_mixes(value, weight, s, b, 4, 20, 1e-6, 1e-6)

    actual = traced(x, fn, scale, base)
    assert hc._HC_MIXES_VERDICTS == {}
    expected = hc._hc_mixes_reference(x, fn, scale, base, 4, 20, 1e-6, 1e-6)
    for a, e in zip(actual, expected, strict=True):
        assert mx.array_equal(a, e).item()


def test_hc_mixes_column_major_projection_takes_the_stock_graph(
    fresh_mix_verdicts,
) -> None:
    """The stock GEMV reduces a column-major projection in another order; a
    cancelling column exposes the difference, so that layout stays stock even
    after the row-major layout has qualified."""
    _requires_metal()
    x = mx.ones((1, 1, 4, _MIX_WIDTH))
    scale, base = mx.ones((3,)), mx.zeros((24,))
    row_major = mx.zeros((24, 4 * _MIX_WIDTH))
    hc.hc_mixes(x, row_major, scale, base, 4, 20, 1e-6, 1e-6)
    assert list(hc._HC_MIXES_VERDICTS.values()) == [True]

    column = mx.zeros((4 * _MIX_WIDTH, 24))
    column[mx.array([0, 32, 1024]), 0] = mx.array([2.0**24, 1.0, -(2.0**24)])
    column_major = column.T
    assert not hc._hc_mixes_supported(x, column_major, scale, base, 4)
    actual = hc.hc_mixes(x, column_major, scale, base, 4, 20, 1e-6, 1e-6)
    expected = hc._hc_mixes_reference(x, column_major, scale, base, 4, 20, 1e-6, 1e-6)
    for a, e in zip(actual, expected, strict=True):
        assert mx.array_equal(a, e).item()
    # the fused order differs here, which is why the layout is excluded
    fused = hc._fused_hc_mixes(x, column_major, scale, base, 20, 1e-6, 1e-6)
    assert not mx.array_equal(fused[0], expected[0]).item()


def test_fused_hc_mixes_reads_strided_scale_and_base_densely() -> None:
    """Strided scale/base views are packed before launch, so the kernels read
    the viewed elements, not the underlying buffer."""
    _requires_metal()
    x, fn, scale, base = _mix_inputs(2, mx.bfloat16, seed=74)
    scale_view = mx.stack([scale, mx.full((3,), 9.0)], axis=1).reshape(-1)[::2]
    base_view = mx.stack([base, mx.full((24,), -9.0)], axis=1).reshape(-1)[::2]
    actual = hc._fused_hc_mixes(x, fn, scale_view, base_view, 20, 1e-6, 1e-6)
    for r in range(2):
        stock = hc._hc_mixes_reference(x[r : r + 1], fn, scale, base, 4, 20, 1e-6, 1e-6)
        for fused, ref in zip(actual, stock, strict=True):
            assert mx.array_equal(fused[r : r + 1], ref).item()


def test_hc_mixes_verdict_decided_while_waiting_for_the_lock(
    fresh_mix_verdicts, monkeypatch
) -> None:
    """A thread that loses the race reuses the winner's verdict, no re-probe."""

    class RacedLock:
        def __enter__(self):
            # another thread qualified this key while we waited for the lock
            hc._HC_MIXES_VERDICTS.update(dict.fromkeys(list(seen), True))

        def __exit__(self, *exc):
            return False

    seen = []
    real_get = hc._HC_MIXES_VERDICTS.get

    class Recording(dict):
        def get(self, key, default=None):
            seen.append(key)
            return real_get(key, default)

    monkeypatch.setattr(hc, "_HC_MIXES_VERDICTS", Recording())
    monkeypatch.setattr(hc, "_HC_MIXES_LOCK", RacedLock())
    monkeypatch.setattr(
        hc, "_hc_mixes_probe", lambda *a: pytest.fail("verdict re-probed")
    )
    x = mx.zeros((1, 4, 64), dtype=mx.bfloat16)
    fn, scale, base = mx.zeros((24, 256)), mx.ones((3,)), mx.zeros((24,))
    assert hc._hc_mixes_qualified(x, fn, scale, base, 20, 1e-6, 1e-6) is True
    assert len(seen) == 1
