from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from rapid_mlx.models.deepseek_v41_native import moe_decode  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.moe import MoE  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.serving import (  # noqa: E402
    ExactDirectDownSwitchGLU,
)

DIM, INTER, EXPERTS, TOPK = 512, 256, 64, 4


def _moe(
    *, routed_bits: int = 2, shared_bits: int = 8, limit: float = 10.0, seed: int = 0
) -> MoE:
    mx.random.seed(seed)
    layer = MoE(
        ModelArgs(
            dim=DIM,
            moe_inter_dim=INTER,
            n_routed_experts=EXPERTS,
            n_activated_experts=TOPK,
            swiglu_limit=limit,
        )
    )
    layer.update(
        nn.utils.tree_map(
            lambda p: (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16),
            layer.parameters(),
        )
    )
    layer.gate.bias = mx.random.normal((EXPERTS,)) * 0.02
    layer.gate.bias_vl = mx.zeros((EXPERTS,))
    nn.quantize(
        layer,
        group_size=64,
        bits=routed_bits,
        class_predicate=lambda path, m: (
            path.startswith("experts.") and hasattr(m, "to_quantized")
        ),
    )
    nn.quantize(
        layer,
        group_size=64,
        bits=shared_bits,
        class_predicate=lambda path, m: (
            path.startswith("shared_experts.") and hasattr(m, "to_quantized")
        ),
    )
    mx.eval(layer.parameters())
    return layer


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("Metal kernel requires a GPU")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


# The product's MoE rows are BF16. Other dtypes go through the same per-layer
# self-check, which keeps the stock path wherever stock picks other kernels.
@pytest.mark.parametrize("dtype", [mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 2, 5, 8])
def test_fused_rows_equal_single_row_stock(gpu, rows: int, dtype) -> None:
    layer = _moe(seed=rows)
    # the product layout: exact direct down projection on the stock side
    layer.experts.__class__ = ExactDirectDownSwitchGLU
    layout = moe_decode._layout(layer)
    assert layout == (DIM, INTER, 8)
    x = mx.random.normal((rows, DIM)).astype(dtype)
    actual = moe_decode._fused(layer, layout, x)
    expected = mx.concatenate([layer._stock(x[r : r + 1]) for r in range(rows)])
    mx.eval(actual, expected)
    assert actual.dtype == dtype
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("limit", [10.0, 0.0])
def test_moe_call_takes_the_qualified_kernels(gpu, monkeypatch, limit) -> None:
    layer = _moe(limit=limit, shared_bits=2)
    calls = []
    original = moe_decode._experts

    def spy(*args):
        calls.append(args[2].shape)
        return original(*args)

    monkeypatch.setattr(moe_decode, "_experts", spy)
    x = mx.random.normal((1, 3, DIM)).astype(mx.bfloat16) * 4
    actual = layer(x)
    assert getattr(layer, moe_decode._CHECK_ATTR)[1][mx.bfloat16] == (DIM, INTER, 2)
    assert moe_decode._CHECK_ATTR not in layer.parameters()
    assert calls[-1] == (3, DIM)
    monkeypatch.setattr(moe_decode, "ENABLED", False)
    expected = mx.concatenate([layer(x[:, i : i + 1]) for i in range(3)], axis=1)
    mx.eval(actual, expected)
    assert actual.shape == x.shape
    assert mx.array_equal(actual, expected).item()


def test_self_check_mismatch_keeps_the_stock_path(gpu, monkeypatch) -> None:
    layer = _moe()
    real = moe_decode._down_kernel()

    def wrong(**kwargs):
        (out,) = real(**kwargs)
        return [out + 1]

    monkeypatch.setattr(moe_decode, "_down_kernel", lambda: wrong)
    x = mx.random.normal((1, DIM)).astype(mx.bfloat16)
    assert mx.array_equal(layer(x), layer._stock(x)).item()
    assert getattr(layer, moe_decode._CHECK_ATTR)[1][mx.bfloat16] is None
    # a replaced weight is qualified again
    layer.experts.gate_proj.weight = mx.array(layer.experts.gate_proj.weight)
    monkeypatch.undo()
    assert moe_decode._qualified(layer, layer._stock, mx.bfloat16) is not None


def test_launch_failure_keeps_the_stock_path(gpu, monkeypatch) -> None:
    layer = _moe()

    def broken(**kwargs):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(moe_decode, "_route_kernel", lambda normalize: broken)
    x = mx.random.normal((1, DIM)).astype(mx.bfloat16)
    assert mx.array_equal(layer(x), layer._stock(x)).item()
    assert getattr(layer, moe_decode._CHECK_ATTR)[1][mx.bfloat16] is None


def test_supports_declines_off_the_decode_shape(monkeypatch) -> None:
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    assert moe_decode.supports(mx.zeros((1, 1, DIM), dtype=mx.bfloat16))
    assert moe_decode.supports(mx.zeros((8, DIM)))
    assert not moe_decode.supports(mx.zeros((9, DIM), dtype=mx.bfloat16))
    assert not moe_decode.supports(mx.zeros((1, DIM), dtype=mx.int32))
    monkeypatch.setattr(moe_decode, "ENABLED", False)
    assert not moe_decode.supports(mx.zeros((1, DIM), dtype=mx.bfloat16))


def test_layout_declines_unsupported_weights() -> None:
    assert moe_decode._layout(_moe()) == (DIM, INTER, 8)
    assert moe_decode._layout(_moe(routed_bits=4)) is None
    assert moe_decode._layout(_moe(shared_bits=4)) is None
    unquantized = MoE(
        ModelArgs(
            dim=DIM, moe_inter_dim=INTER, n_routed_experts=8, n_activated_experts=2
        )
    )
    assert moe_decode._layout(unquantized) is None
    softmax = _moe()
    softmax.gate.score_func = "softmax"
    assert moe_decode._layout(softmax) is None
    biased = _moe()
    biased.shared_experts.w2.bias = mx.zeros((DIM,))
    assert moe_decode._layout(biased) is None


def test_stock_qmv_lane_geometry() -> None:
    # two-bit: one 16-value pack per lane on both kernels
    assert moe_decode._qmv_vpt(2, 5120, 2304) == 16
    assert moe_decode._qmv_vpt(2, 2304, 5120) == 16
    # eight-bit: two packs on the fast kernel, one on the general kernel
    assert moe_decode._qmv_vpt(8, 5120, 2304) == 8
    assert moe_decode._qmv_vpt(8, 2304, 5120) == 8
    assert moe_decode._qmv_vpt(8, 2240, 5120) == 4
    assert moe_decode._qmv_vpt(8, 5120, 2300) == 4


def test_dispatch_geometry(monkeypatch) -> None:
    """Grid/threadgroup layout of the three launches (runs without Metal)."""
    launches = {}

    def fake_kernel(name):
        def run(**kwargs):
            launches[name] = kwargs
            return [
                mx.zeros(shape, dtype=dtype)
                for shape, dtype in zip(
                    kwargs["output_shapes"], kwargs["output_dtypes"], strict=True
                )
            ]

        return run

    monkeypatch.setattr(moe_decode, "_route_kernel", lambda n: fake_kernel("route"))
    monkeypatch.setattr(moe_decode, "_up_kernel", lambda limit: fake_kernel("up"))
    monkeypatch.setattr(moe_decode, "_down_kernel", lambda: fake_kernel("down"))
    monkeypatch.setattr(moe_decode, "supports", lambda x: True)
    layer = _moe()
    layer.gate.scores = lambda x: mx.zeros((x.shape[0], EXPERTS))
    rows = 3
    out = moe_decode.moe_forward(
        layer, mx.zeros((rows, 1, DIM), dtype=mx.bfloat16), lambda x: x
    )
    assert out is None  # the zero kernels fail the self-check
    monkeypatch.setattr(
        moe_decode, "_qualified", lambda moe, stock, dtype: (DIM, INTER, 8)
    )
    out = moe_decode.moe_forward(
        layer, mx.zeros((rows, 1, DIM), dtype=mx.bfloat16), lambda x: x
    )
    assert out.shape == (rows, 1, DIM) and out.dtype == mx.bfloat16
    assert launches["route"]["grid"] == (EXPERTS, rows, 1)
    assert launches["route"]["threadgroup"] == (EXPERTS, 1, 1)
    assert launches["up"]["grid"] == (32, INTER // 4, rows * TOPK + 1)
    assert launches["up"]["threadgroup"] == (32, 2, 1)
    assert launches["up"]["output_shapes"] == [(rows * (TOPK + 1), INTER)]
    assert launches["down"]["grid"] == (32, (TOPK + 1) * DIM // 4, 1)
    assert launches["down"]["threadgroup"] == (32, TOPK + 1, 1)
    assert ("SVPT", 8) in launches["up"]["template"]
    assert ("SDVPT", 8) in launches["down"]["template"]
    assert ("ROWS", rows) in launches["down"]["template"]
    # wrong width: stock path
    monkeypatch.setattr(
        moe_decode, "_qualified", lambda moe, stock, dtype: (DIM * 2, INTER, 8)
    )
    assert moe_decode.moe_forward(layer, mx.zeros((1, DIM)), lambda x: x) is None
    monkeypatch.setattr(moe_decode, "supports", lambda x: False)
    assert moe_decode.moe_forward(layer, mx.zeros((1, DIM)), lambda x: x) is None


def test_kernel_builders(monkeypatch) -> None:
    built = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(kw) or kw)
    for builder in (moe_decode._route_kernel, moe_decode._up_kernel):
        builder.cache_clear()
    moe_decode._down_kernel.cache_clear()
    try:
        moe_decode._route_kernel(True)
        moe_decode._route_kernel(False)
        moe_decode._up_kernel(10.0)
        moe_decode._up_kernel(-1.5)
        moe_decode._down_kernel()
    finally:
        for builder in (moe_decode._route_kernel, moe_decode._up_kernel):
            builder.cache_clear()
        moe_decode._down_kernel.cache_clear()
    names = [kw["name"] for kw in built]
    assert len(set(names)) == 5
    assert "if (true)" in built[0]["source"] and "if (false)" in built[1]["source"]
    assert "(10.0f)" in built[2]["source"] and "(-1.5f)" in built[3]["source"]
