from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rapid_mlx.models.deepseek_v41_native import head_gemv, moe  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.head_gemv import project  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.moe import Gate, MoE  # noqa: E402


def _gate(experts: int = 384, dim: int = 1024) -> Gate:
    gate = Gate(ModelArgs(dim=dim, n_routed_experts=experts, n_activated_experts=6))
    gate.weight = (mx.random.normal((experts, dim)) * 0.05).astype(mx.bfloat16)
    gate.bias = mx.random.normal((experts,)) * 0.01
    return gate


def _stock(gate: Gate, x: mx.array) -> mx.array:
    return x.astype(mx.float32) @ gate.weight.astype(mx.float32).T


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("Metal kernel requires a GPU")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("rows", [1, 2, 6, 8])
def test_router_projection_is_bit_exact_per_row(gpu, rows: int) -> None:
    gate = _gate()
    x = mx.random.normal((rows, 1024))
    assert head_gemv._supports_weight(gate.weight, x)
    assert head_gemv._matches_stock(gate)
    actual = head_gemv._launch(gate.weight, x)
    expected = mx.concatenate([_stock(gate, x[i : i + 1]) for i in range(rows)])
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(project(gate, x), actual).item()


@pytest.mark.parametrize("rows", [1, 5, 8, 9])
def test_router_selection_matches_single_row_stock_gate(
    gpu, monkeypatch, rows: int
) -> None:
    # Like the head, every routed row (batched lanes, speculative verify)
    # gets its single-row decode scores; stock multi-row GEMM may differ.
    gate = _gate()
    x = mx.random.normal((rows, 1024)).astype(mx.bfloat16)
    weights, indices = gate(x)
    monkeypatch.setattr(head_gemv, "ENABLED", False)
    if rows > head_gemv.MAX_ROWS:
        stock_weights, stock_indices = gate(x)
    else:
        singles = [gate(x[i : i + 1]) for i in range(rows)]
        stock_weights = mx.concatenate([w for w, _ in singles])
        stock_indices = mx.concatenate([i for _, i in singles])
    mx.eval(weights, indices, stock_weights, stock_indices)
    assert mx.array_equal(indices, stock_indices).item()
    assert mx.array_equal(weights, stock_weights).item()


def test_router_projection_falls_back_off_the_fast_path(monkeypatch) -> None:
    gate = _gate(experts=64, dim=256)
    x = mx.random.normal((1, 256))
    monkeypatch.setattr(head_gemv, "ENABLED", False)
    assert mx.array_equal(project(gate, x), _stock(gate, x)).item()
    monkeypatch.setattr(head_gemv, "ENABLED", True)
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    # The router keeps a selection bias; unlike a biased nn.Linear it is not
    # part of the projection, so only the weight decides eligibility.
    assert head_gemv._supports_weight(gate.weight, x)
    assert not head_gemv._supports_weight(gate.weight, mx.random.normal((9, 256)))
    assert not head_gemv._supports_weight(gate.weight.astype(mx.float32), x)


def test_router_projection_self_check_mismatch_stays_stock(monkeypatch) -> None:
    def wrong_kernel(**kwargs):
        rows, outputs = kwargs["output_shapes"][0]
        return [mx.zeros((rows, outputs), dtype=mx.float32)]

    monkeypatch.setattr(head_gemv, "_kernel", lambda: wrong_kernel)
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    gate = _gate(experts=64, dim=256)
    x = mx.random.normal((1, 256))
    assert mx.array_equal(project(gate, x), _stock(gate, x)).item()
    assert getattr(gate, head_gemv._CHECK_ATTR)[1] is False
    assert head_gemv._CHECK_ATTR not in gate.parameters()


def test_moe_routes_through_the_exact_projection(monkeypatch) -> None:
    calls = []
    original = moe.project

    def spy(owner, x):
        calls.append((type(owner).__name__, x.dtype, x.shape))
        return original(owner, x)

    monkeypatch.setattr(moe, "project", spy)
    layer = MoE(
        ModelArgs(dim=64, moe_inter_dim=32, n_routed_experts=8, n_activated_experts=2)
    )
    out = layer(mx.random.normal((1, 1, 64)).astype(mx.bfloat16))
    mx.eval(out)
    assert calls == [("Gate", mx.float32, (1, 64))]


def test_product_router_geometry_qualifies_for_every_row_count(gpu) -> None:
    # the product router: 384 experts x 5120 hidden
    gate = _gate(experts=384, dim=5120)
    assert head_gemv._matches_stock(gate)
    for rows in range(1, head_gemv.MAX_ROWS + 1):
        x = mx.random.normal((rows, 5120))
        actual = project(gate, x)
        expected = mx.concatenate([_stock(gate, x[i : i + 1]) for i in range(rows)])
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected).item()


def test_replaced_router_weight_is_checked_again(monkeypatch) -> None:
    monkeypatch.setattr(head_gemv, "_gpu_default", lambda: True)
    checks = []
    original = head_gemv._launch

    def counting(weight, x):
        checks.append(x.shape)
        return original(weight, x)

    monkeypatch.setattr(head_gemv, "_launch", counting)
    gate = _gate(experts=64, dim=256)
    monkeypatch.setattr(head_gemv, "_kernel", lambda: _zero_kernel)
    x = mx.random.normal((1, 256))
    project(gate, x)
    project(gate, x)
    assert len(checks) == 1  # one self-check, cached verdict: stock
    gate.weight = mx.array(gate.weight)
    project(gate, x)
    assert len(checks) == 2  # the new weight is checked again


def _zero_kernel(**kwargs):
    rows, outputs = kwargs["output_shapes"][0]
    return [mx.zeros((rows, outputs), dtype=mx.float32)]
