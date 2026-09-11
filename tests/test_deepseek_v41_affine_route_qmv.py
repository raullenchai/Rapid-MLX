from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")

from scripts.deepseek_v41_affine_route_qmv import affine2_route_down_qmv


def _projection(*, bits: int = 2):
    switch = switch_layers.SwitchGLU(512, 64, 8, bias=False)
    nn.quantize(switch, group_size=64, bits=bits)
    projection = switch.down_proj
    projection.scales = projection.scales.astype(mx.bfloat16)
    projection.biases = projection.biases.astype(mx.bfloat16)
    return projection


@pytest.mark.parametrize("routes", [1, 6, 24, 30, 36])
def test_affine2_route_down_qmv_is_bit_exact(routes: int) -> None:
    if mx.default_device() == mx.cpu:
        pytest.skip("Metal kernel requires a GPU")
    projection = _projection()
    inputs = mx.random.normal((routes, 64)).astype(mx.float32)
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

    four_bit = _projection(bits=4)
    with pytest.raises(ValueError, match="unsupported exact"):
        affine2_route_down_qmv(four_bit, mx.zeros((1, 64), dtype=mx.float32), indices)
