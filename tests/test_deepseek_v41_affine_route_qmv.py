from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")

from rapid_mlx.models.deepseek_v41_native.affine_route_qmv import (
    affine2_route_down_qmv,
)


def _projection(*, bits: int = 2, width: int = 64):
    switch = switch_layers.SwitchGLU(512, width, 8, bias=False)
    nn.quantize(switch, group_size=64, bits=bits)
    projection = switch.down_proj
    projection.scales = projection.scales.astype(mx.bfloat16)
    projection.biases = projection.biases.astype(mx.bfloat16)
    return projection


@pytest.mark.parametrize("width", [64, 256, 512, 576])
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("routes", [1, 6, 24, 30, 36])
def test_affine2_route_down_qmv_is_bit_exact(
    routes: int, bits: int, width: int
) -> None:
    if not mx.metal.is_available() or mx.default_device() == mx.cpu:
        pytest.skip("Metal kernel requires a GPU")
    projection = _projection(bits=bits, width=width)
    inputs = mx.random.normal((routes, width)).astype(mx.float32)
    indices = (mx.arange(routes) % 8).reshape(routes, 1).astype(mx.uint32)

    actual = affine2_route_down_qmv(projection, inputs, indices)
    expected = projection(inputs[:, None, None, :], indices).squeeze(-2)
    mx.eval(actual, expected)

    assert mx.array_equal(actual, expected).item()


def test_affine2_route_down_qmv_rejects_non_exact_shapes() -> None:
    projection = _projection()
    indices = mx.zeros((1, 1), dtype=mx.uint32)
    with pytest.raises(ValueError, match="unsupported exact"):
        affine2_route_down_qmv(
            projection, mx.zeros((1, 64), dtype=mx.bfloat16), indices
        )

    with pytest.raises(ValueError, match="unsupported exact"):
        affine2_route_down_qmv(
            projection,
            mx.zeros((37, 64), dtype=mx.float32),
            mx.zeros((37, 1), dtype=mx.uint32),
        )

    four_bit = _projection(bits=8)
    with pytest.raises(ValueError, match="unsupported exact"):
        affine2_route_down_qmv(four_bit, mx.zeros((1, 64), dtype=mx.float32), indices)


def test_affine2_route_down_qmv_rejects_structural_contract_violations() -> None:
    projection = _projection()
    inputs = mx.zeros((1, 64), dtype=mx.float32)
    indices = mx.zeros((1, 1), dtype=mx.uint32)

    with pytest.raises(ValueError, match="rank two"):
        affine2_route_down_qmv(projection, inputs[0], indices)
    with pytest.raises(ValueError, match="one row per input token"):
        affine2_route_down_qmv(projection, mx.zeros((2, 64), dtype=mx.float32), indices)
    with pytest.raises(ValueError, match="unsupported exact"):
        affine2_route_down_qmv(object(), inputs, indices)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_mixed_route_dispatch_and_installation(monkeypatch, bits) -> None:
    from types import SimpleNamespace

    from rapid_mlx.models.deepseek_v41_native import affine_route_qmv, serving

    projection = _projection(bits=bits, width=256)
    experts = switch_layers.SwitchGLU(512, 256, 8, bias=False)
    experts.down_proj = projection
    assert serving.supports_target_qmv(experts)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(experts=experts))]
    )
    assert serving.install_target_qmv(model) == 1
    assert isinstance(experts, serving.ExactDirectDownSwitchGLU)
    launches = []

    def fake(**kw):
        launches.append(kw)
        return [mx.zeros(kw["output_shapes"][0], dtype=mx.float32)]

    monkeypatch.setattr(affine_route_qmv, "_kernel", lambda: fake)
    out = affine2_route_down_qmv(
        projection,
        mx.zeros((2, 256), dtype=mx.float32),
        mx.zeros((2, 6), dtype=mx.uint32),
    )
    assert out.shape == (2, 6, 512)
    template = dict(launches[0]["template"])
    assert template["BITS"] == bits
    assert template["VPT"] == (16 if bits == 2 else 8)
    projection.group_size = 32
    assert not serving.supports_target_qmv(experts)


def test_route_kernel_builder_uses_shared_packing(monkeypatch) -> None:
    from rapid_mlx.models.deepseek_v41_native import affine_route_qmv, moe_decode

    built = []
    affine_route_qmv._kernel.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(kw) or kw)
    try:
        affine_route_qmv._kernel()
    finally:
        affine_route_qmv._kernel.cache_clear()
    assert built[0]["header"] == moe_decode._HEADER
