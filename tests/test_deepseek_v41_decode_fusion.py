"""Compiled native decode contracts, including the independent eager A/B path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.deepseek_v41_native import fakequant, fusion, layers, moe
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
from scripts.bench_deepseek_v41_decode_fusion import graph_step, make_tiny_model


@pytest.fixture(autouse=True)
def restore_switches(monkeypatch):
    monkeypatch.setattr(fusion, "ENABLED", False)
    monkeypatch.setattr(fakequant, "DISABLE", False)


def array(value):
    return np.array(value.astype(mx.float32))


def compare(function, *args, **kwargs):
    fusion.ENABLED = False
    reference = function(*args, **kwargs)
    mx.eval(reference)
    fusion.ENABLED = True
    actual = function(*args, **kwargs)
    mx.eval(actual)
    references = reference if isinstance(reference, tuple) else (reference,)
    actuals = actual if isinstance(actual, tuple) else (actual,)
    for left, right in zip(references, actuals, strict=True):
        assert np.array_equal(array(left), array(right))
        assert np.array_equal(np.signbit(array(left)), np.signbit(array(right)))
    return actual


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize(
    "function",
    [
        fakequant.fake_quant_fp8_ue8m0,
        fakequant.fake_quant_fp4_ue8m0,
        fakequant.fake_quant_fp4_e4m3,
    ],
)
def test_quantization_random_and_extremes(function, n, dtype):
    mx.random.seed(11)
    x = mx.random.normal((2, n, 64)).astype(dtype)
    compare(function, x)
    # Independent blocks cover zero, the scale floors, and a large finite amax.
    values = np.resize(
        np.array(
            [0.0, -0.0, 1e-38, -1e-38, 1e-4, -1e-4, 6 * 2**-9, -6 * 2**-9, 1e4, -1e4],
            np.float32,
        ),
        (2, n, 64),
    )
    compare(function, mx.array(values).astype(dtype))
    compare(function, mx.zeros((2, n, 64), dtype))


def test_rounding_midpoints_and_neighbors():
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
    positive = np.concatenate(
        [
            midpoints,
            np.nextafter(midpoints, np.float32(0)),
            np.nextafter(midpoints, np.float32(np.inf)),
            [0.0, -0.0, 6.0],
        ]
    )
    values = np.concatenate([positive, -positive]).astype(np.float32)
    # Test the seven-pass helper directly through a compiled pure function.
    compare(fusion.fuse(fakequant._e2m1_round), mx.array(values))
    expected = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], np.float32)
    assert np.array_equal(array(fakequant._e2m1_round(mx.array(midpoints))), expected)
    # Max=6 gives unit FP4 scale; 448 gives unit FP8 scale.
    for function, maximum, block in [
        (fakequant.fake_quant_fp4_ue8m0, 6.0, 32),
        (fakequant.fake_quant_fp4_e4m3, 6.0, 16),
        (fakequant.fake_quant_fp8_ue8m0, 448.0, 32),
    ]:
        rows = np.zeros((len(values), block), np.float32)
        rows[:, 0] = maximum
        rows[:, 1] = values
        compare(function, mx.array(rows), block)


@pytest.mark.parametrize(
    "function",
    [
        fakequant.fake_quant_fp8_ue8m0,
        fakequant.fake_quant_fp4_ue8m0,
        fakequant.fake_quant_fp4_e4m3,
    ],
)
def test_quantization_bypass_and_invalid_block(function, monkeypatch):
    fusion.ENABLED = True
    x = mx.array([[0.123, -0.456]])
    monkeypatch.setattr(fakequant, "DISABLE", True)
    assert function(x) is x
    monkeypatch.setattr(fakequant, "DISABLE", False)
    for mode in (False, True):
        fusion.ENABLED = mode
        with pytest.raises(ValueError, match="not divisible"):
            function(x)
    compare(function, mx.ones((1, 32)))
    monkeypatch.setattr(fakequant, "DISABLE", True)
    assert function(x) is x  # bypass also works after the trace was cached


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("eps", [1e-20, 1e-6])
def test_rmsnorm_weights_are_inputs(dtype, eps):
    norm = layers.RMSNorm(64, eps)
    for n in (1, 4):
        x = mx.random.normal((2, n, 64)).astype(dtype)
        compare(norm, x)
        norm.weight = mx.random.normal((64,)).astype(dtype)
        compare(norm, x)
        compare(norm, mx.zeros((2, n, 64), dtype))


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("heads", [False, True])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_rotary(inverse, heads, dtype):
    for n in (1, 5):
        shape = (2, n, 3, 32) if heads else (2, n, 32)
        x = mx.random.normal(shape).astype(dtype)
        cos, sin = layers.precompute_freqs_cis(8, n, 0, 10000.0, 1.0, 32, 1)
        compare(layers.rope_tail, x, 8, cos, sin, inverse)


@pytest.mark.parametrize("score", ["sqrtsoftplus", "sigmoid", "softmax"])
@pytest.mark.parametrize("topk,normalize", [(1, True), (2, True), (2, False)])
def test_routing_ties_and_changed_bias(score, topk, normalize):
    gate = moe.Gate(
        ModelArgs(
            dim=64,
            n_routed_experts=8,
            n_activated_experts=topk,
            score_func=score,
            norm_topk_prob=normalize,
        )
    )
    for n in (1, 5):
        x = mx.random.normal((n, 64))
        compare(gate, x)  # all zero weights: exact ties
        gate.weight = mx.random.normal((8, 64)) * 0.1
        gate.bias = mx.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
        compare(gate, x)
        gate.bias = -gate.bias
        compare(gate, x)


@pytest.mark.parametrize("limit", [0.0, 10.0])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_activations_and_combine(limit, dtype):
    for n in (1, 5):
        x = (mx.random.normal((n, 64)) * 20).astype(dtype)
        gate = (mx.random.normal((n, 64)) * 20).astype(dtype)
        compare(moe.ClampedSwiGLU(limit), x, gate)
        compare(moe.SharedExpert(64, 32, limit), x)
        compare(
            moe._combine,
            mx.random.normal((n, 2, 64)).astype(dtype),
            mx.random.uniform(shape=(n, 2)),
            x,
        )


@pytest.mark.parametrize("chunks", [(8,), (3, 2, 3)])
def test_model_32_greedy_steps(chunks):
    model = make_tiny_model()
    outputs = []
    tokens = []
    for mode in (False, True):
        fusion.ENABLED = mode
        cache = model.make_cache(max_seq_len=64)
        logits_list, token_list = [], []
        offset = 0
        for n in chunks:
            logits = model(
                mx.array([[i % 64 for i in range(offset, offset + n)]]), cache
            )
            mx.eval(logits)
            logits_list.append(array(logits))
            offset += n
        for _ in range(32):
            token = mx.argmax(logits[:, -1], axis=-1)[:, None]
            mx.eval(token)
            token_list.append(np.array(token))
            logits = model(token, cache)
            mx.eval(logits)
            logits_list.append(array(logits))
        outputs.append(logits_list)
        tokens.append(token_list)
    assert np.array_equal(tokens[0], tokens[1])
    for left, right in zip(outputs[0], outputs[1], strict=True):
        assert np.array_equal(left, right)


def test_decode_node_reduction():
    model = make_tiny_model()
    fusion.ENABLED = False
    off = graph_step(model)
    fusion.ENABLED = True
    on = graph_step(model)
    assert on["nodes"] <= off["nodes"] * 0.70
    assert len(on["layers"]) == 4


@pytest.mark.parametrize(
    "function,maximum,block",
    [
        (fakequant.fake_quant_fp8_ue8m0, 448.0, 32),
        (fakequant.fake_quant_fp4_ue8m0, 6.0, 32),
        (fakequant.fake_quant_fp4_e4m3, 6.0, 16),
    ],
)
def test_fp8_scale_and_value_midpoints(function, maximum, block):
    # Every neighboring e4m3 pair in [1, 2] has a midpoint here.
    midpoints = np.arange(1.0625, 2.0, 0.125, dtype=np.float32)
    values = np.concatenate(
        [
            midpoints,
            np.nextafter(midpoints, np.float32(0)),
            np.nextafter(midpoints, np.float32(np.inf)),
        ]
    )
    rows = np.zeros((len(values), block), np.float32)
    rows[:, 0] = maximum
    rows[:, 1] = values
    compare(function, mx.array(rows), block)
    # Scales straddle rounding boundaries independently of the activation grid.
    for amax in (6 * 2**-126, 1e-4, 6 * 2**-9, 1.0, 6.0, 448.0, 1e30):
        maxima = np.array([amax], np.float32)
        maxima = np.concatenate(
            [
                maxima,
                np.nextafter(maxima, np.float32(0)),
                np.nextafter(maxima, np.float32(np.inf)),
            ]
        )
        rows = np.zeros((3, block), np.float32)
        rows[:, 0] = maxima
        rows[:, 1] = -maxima * 0.3
        compare(function, mx.array(rows), block)
