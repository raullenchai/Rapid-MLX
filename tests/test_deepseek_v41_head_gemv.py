from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from rapid_mlx.models.deepseek_v41_native import head_gemv  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.head_gemv import (  # noqa: E402
    head_logits,
    supports,
)

needs_metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal kernel requires a GPU"
)


def _head(outputs: int = 1003, inputs: int = 512) -> nn.Linear:
    head = nn.Linear(inputs, outputs, bias=False)
    head.weight = (mx.random.normal((outputs, inputs)) * 0.02).astype(mx.bfloat16)
    return head


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("Metal kernel requires a GPU")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@needs_metal
@pytest.mark.parametrize("shape", [(1, 1), (1,), (1, 1, 1)])
def test_single_rows_are_bit_exact_with_stock_projection(gpu, shape) -> None:
    head = _head()
    x = mx.random.normal((*shape, 512))
    assert supports(head, x)
    actual = head_logits(head, x)
    expected = head(x)
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert mx.array_equal(actual, expected).item()


@needs_metal
@pytest.mark.parametrize("rows", [2, 5, 8])
def test_each_row_matches_its_single_row_stock_result(gpu, rows: int) -> None:
    # Multi-row stock GEMM may use a different (even reduced-precision) kernel;
    # the exact head instead gives every row its single-row decode result.
    head = _head(outputs=4096, inputs=1024)
    x = mx.random.normal((1, rows, 1024))
    actual = head_logits(head, x)
    expected = mx.concatenate([head(x[:, i : i + 1]) for i in range(rows)], axis=1)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@needs_metal
def test_model_forward_uses_exact_head(gpu, monkeypatch) -> None:
    calls = []
    original = head_gemv._kernel

    def counting():
        calls.append(1)
        return original()

    monkeypatch.setattr(head_gemv, "_kernel", counting)
    head = _head()
    x = mx.random.normal((1, 1, 512))
    mx.eval(head_logits(head, x))
    assert calls


def test_unsupported_inputs_fall_back_to_stock(monkeypatch) -> None:
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    head = _head()
    rows = mx.random.normal((9, 512))
    assert not supports(head, rows)
    assert not supports(head, rows[:1].astype(mx.bfloat16))
    assert not supports(head, mx.random.normal((1, 256)))
    assert not supports(head, mx.zeros((0, 512)))
    assert not supports(head, mx.array(1.0))
    assert not supports(_head(inputs=200), mx.random.normal((1, 200)))
    biased = nn.Linear(512, 64)
    biased.weight = biased.weight.astype(mx.bfloat16)
    assert not supports(biased, rows[:1])
    fp32 = nn.Linear(512, 64, bias=False)
    assert not supports(fp32, rows[:1])
    quantized = nn.QuantizedLinear(512, 64, bias=False)
    assert not supports(quantized, rows[:1])
    weightless = nn.Linear(512, 64, bias=False)
    del weightless["weight"]
    assert not supports(weightless, rows[:1])
    flat = nn.Linear(512, 64, bias=False)
    flat.weight = mx.zeros((512,), dtype=mx.bfloat16)
    assert not supports(flat, rows[:1])
    actual = head_logits(head, rows)
    expected = head(rows)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_cpu_and_disabled_switch_use_stock(monkeypatch) -> None:
    head = _head()
    x = mx.random.normal((1, 512))
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not head_gemv._gpu_default()
        assert not supports(head, x)
    finally:
        mx.set_default_device(previous)
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    assert supports(head, x)
    monkeypatch.setattr(head_gemv, "ENABLED", False)
    assert not supports(head, x)


def test_dispatch_geometry_without_metal(monkeypatch) -> None:
    """Exercise the launch wrapper with a stand-in kernel on any host."""
    seen = {}

    def fake_kernel(**kwargs):
        seen.update(kwargs)
        rows, outputs = kwargs["output_shapes"][0]
        return [mx.zeros((rows, outputs), dtype=mx.float32)]

    monkeypatch.setattr(head_gemv, "_kernel", lambda: fake_kernel)
    monkeypatch.setattr(head_gemv, "supports", lambda head, x: True)
    head = _head(outputs=1003)
    logits = head_logits(head, mx.zeros((2, 3, 512)))
    assert logits.shape == (2, 3, 1003)
    assert seen["grid"] == (251 * 32 * 4, 6, 1)
    assert seen["threadgroup"] == (128, 1, 1)
    assert dict(seen["template"]) == {"K": 512, "N": 1003, "OUTPUTS": 4}
    assert seen["inputs"][0].shape == (6, 512)
