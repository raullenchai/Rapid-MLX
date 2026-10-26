"""Single-token routing/combination parity and original-path boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rapid_mlx.patches import moe_decode_fusion as fusion


def test_disabled_without_mlx(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_MOE_DECODE_FUSION", "0")
    assert fusion.install_moe_decode_fusion(object()) == 0
    assert not fusion.eligible(object(), object())


@pytest.fixture
def metal():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("requires Metal")
    return mx


def make_block(e=64, k=4, h=128):
    from mlx_lm.models.lfm2_moe import Lfm2MoeSparseMoeBlock

    args = SimpleNamespace(
        hidden_size=h,
        moe_intermediate_size=128,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        use_expert_bias=True,
    )
    return Lfm2MoeSparseMoeBlock(args)


@pytest.mark.parametrize("e,k,h", [(32, 2, 128), (64, 4, 2048), (256, 8, 2048)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("weight_dtype", ["float32", "bfloat16"])
def test_route(metal, e, k, h, dtype, weight_dtype):
    mx = metal
    b = make_block(e, k, h)
    b.gate.weight = (mx.random.normal((e, h), key=mx.random.key(71)) * 0.02).astype(
        getattr(mx, weight_dtype)
    )
    b.expert_bias = mx.random.normal((e,), key=mx.random.key(72)) * 0.001
    for seed in range(8):
        x = mx.random.normal((1, 1, h), key=mx.random.key(seed)).astype(
            getattr(mx, dtype)
        )
        gates = b.gate(x)
        gates = mx.softmax(gates.astype(mx.float32))
        gates += b.expert_bias
        reference = mx.argpartition(gates, kth=-k)[..., -k:]
        indices, scores = fusion.route(b, x)
        actual_set = set(indices.reshape(-1).tolist())
        reference_set = set(reference.reshape(-1).tolist())
        if actual_set != reference_set:
            # Only rounded probabilities tied at the selection boundary may differ.
            values = gates.reshape(-1).tolist()
            boundary = min(values[i] for i in reference_set)
            assert all(values[i] == boundary for i in actual_set ^ reference_set)
        expected = mx.take_along_axis(gates, indices, -1)
        expected = expected / (expected.sum(-1, keepdims=True) + 1e-20)
        expected = expected.astype(x.dtype)
        assert mx.allclose(scores, expected, rtol=0.02, atol=0.002).item()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("k", [2, 4, 8])
def test_combine(metal, dtype, k):
    mx = metal
    y = mx.random.normal((1, 1, k, 2048)).astype(getattr(mx, dtype))
    scores = mx.softmax(mx.random.normal((1, 1, k))).astype(y.dtype)
    expected = (y * scores[..., None]).sum(-2)
    actual = fusion.combine(y, scores)
    assert actual.dtype == expected.dtype
    tolerance = 0.02 if dtype == "bfloat16" else 0.002 if dtype == "float16" else 1e-6
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()
    if dtype in ("bfloat16", "float16"):
        assert mx.array_equal(actual, expected).item()


def test_block(metal, monkeypatch):
    mx = metal
    import mlx.nn as nn

    b = make_block()
    b.set_dtype(mx.bfloat16)
    b.gate.weight = b.gate.weight.astype(mx.float32)
    b.expert_bias = b.expert_bias.astype(mx.float32)
    original = type(b).__call__
    model = nn.Sequential(b)
    weight = b.gate.weight
    x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    expected = original(b, x)
    assert fusion.install_moe_decode_fusion(model) == 1
    assert fusion.install_moe_decode_fusion(model) == 0
    assert b.gate.weight is weight
    assert mx.allclose(b(x), expected, atol=0.02, rtol=0.02).item()
    with patch.object(
        fusion, "route", side_effect=AssertionError("unexpected fast path")
    ):
        for shape in [(1, 3, 128), (2, 1, 128)]:
            wide = mx.random.normal(shape).astype(mx.bfloat16)
            assert mx.array_equal(b(wide), original(b, wide)).item()
        monkeypatch.setenv("RAPID_MLX_MOE_DECODE_FUSION", "0")
        assert mx.array_equal(b(x), original(b, x)).item()


def test_non_metal(metal, monkeypatch):
    monkeypatch.setattr(metal.metal, "is_available", lambda: False)
    assert fusion.install_moe_decode_fusion(object()) == 0
    assert not fusion.eligible(make_block(), metal.zeros((1, 1, 128)))


def test_optional_normalization_and_bias(metal):
    mx = metal
    b = make_block()
    b.norm_topk_prob = False
    b.use_expert_bias = False
    x = mx.random.normal((1, 1, 128))
    indices, scores = fusion.route(b, x)
    gates = mx.softmax(b.gate(x).astype(mx.float32))
    expected = mx.take_along_axis(gates, indices, -1)
    assert mx.allclose(scores, expected, atol=1e-6).item()


def test_unsupported_gates(metal):
    mx = metal
    import mlx.nn as nn

    b = make_block()
    x = mx.zeros((1, 1, 128))
    b.gate = nn.Sequential(nn.Linear(128, 64, bias=False))
    assert not fusion.eligible(b, x)
    expected = b(x)
    assert fusion.install_moe_decode_fusion(nn.Sequential(b)) == 1
    assert mx.array_equal(b(x), expected).item()
    b.gate = nn.QuantizedLinear.from_linear(
        nn.Linear(128, 64, bias=False), group_size=64, bits=8
    )
    assert not fusion.eligible(b, x)
    assert mx.array_equal(b(x), type(make_block()).__call__(b, x)).item()
    b.gate = nn.Linear(128, 64, bias=True)
    assert not fusion.eligible(b, x)
    b.gate = nn.Linear(128, 64, bias=False)
    b.gate.weight = b.gate.weight.astype(mx.float16)
    assert not fusion.eligible(b, x)
    b.gate.weight = b.gate.weight.astype(mx.float32)
    b.sharding_group = object()
    assert not fusion.eligible(b, x)


def test_non_module_container(metal):
    assert fusion.install_moe_decode_fusion(object()) == 0


@pytest.mark.parametrize("e", range(32, 257, 32))
@pytest.mark.parametrize("k", range(1, 9))
@pytest.mark.parametrize("biased", [False, True])
def test_routing_ties(metal, e, k, biased):
    """Match the runtime's stable ascending sort, including boundary ties."""
    mx = metal
    import mlx.nn as nn

    b = make_block(e, k)
    b.set_dtype(mx.bfloat16)
    b.gate.weight = mx.zeros((e, 128), dtype=mx.float32)
    b.expert_bias = (
        (mx.arange(e) % 3 == 0).astype(mx.float32) * 0.001
        if biased
        else mx.zeros((e,), dtype=mx.float32)
    )
    x = mx.ones((1, 1, 128), dtype=mx.bfloat16)
    gates = mx.softmax(b.gate(x).astype(mx.float32)) + b.expert_bias
    expected_indices = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    expected_scores = mx.take_along_axis(gates, expected_indices, -1)
    expected_scores = (
        expected_scores / (expected_scores.sum(-1, keepdims=True) + 1e-20)
    ).astype(x.dtype)
    indices, scores = fusion.route(b, x)
    assert mx.array_equal(indices, expected_indices).item()
    assert mx.array_equal(scores, expected_scores).item()
    expected_output = b(x)
    assert fusion.install_moe_decode_fusion(nn.Sequential(b)) == 1
    assert mx.array_equal(b(x), expected_output).item()


def test_missing_mlx(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_mlx(name, *args, **kwargs):
        if name.startswith("mlx"):
            raise ImportError("MLX unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.delenv("RAPID_MLX_MOE_DECODE_FUSION", raising=False)
    monkeypatch.setattr(builtins, "__import__", without_mlx)
    assert fusion.install_moe_decode_fusion(object()) == 0
    assert not fusion.eligible(object(), object())


@pytest.mark.parametrize("e,h", [(64, 2048), (256, 2048), (32, 128)])
@pytest.mark.parametrize("weight_dtype", ["float32", "bfloat16"])
def test_gate_gemv_accumulation(metal, e, h, weight_dtype):
    mx = metal
    b = make_block(e, 4, h)
    b.gate.weight = b.gate.weight.astype(getattr(mx, weight_dtype))
    x = mx.random.normal((1, 1, h)).astype(mx.bfloat16)
    bn = 8 if h >= 16 * e else 1
    (actual,) = fusion._kernel("logits")(
        inputs=[x, b.gate.weight],
        template=[("H", h), ("G", mx.result_type(x, b.gate.weight)), ("BN", bn)],
        grid=(e // 4 * bn * 32, 1, 1),
        threadgroup=(bn * 32, 1, 1),
        output_shapes=[(e,)],
        output_dtypes=[mx.float32],
    )
    assert mx.array_equal(actual, b.gate(x).astype(mx.float32).reshape(-1)).item()


def test_dense_model_unchanged(metal):
    import mlx.nn as nn

    model = nn.Sequential(nn.Linear(128, 128))
    original_type = type(model.layers[0])
    weight = model.layers[0].weight
    assert fusion.install_moe_decode_fusion(model) == 0
    assert type(model.layers[0]) is original_type
    assert model.layers[0].weight is weight


@pytest.mark.parametrize("e", range(32, 257, 32))
@pytest.mark.parametrize("k", range(1, 9))
def test_routing_parameter_boundaries(metal, e, k):
    mx = metal
    import mlx.nn as nn

    b = SimpleNamespace(
        gate=nn.Linear(256, e, bias=False),
        num_experts=e,
        top_k=k,
        norm_topk_prob=True,
        use_expert_bias=False,
    )
    x = mx.random.normal((1, 1, 256))
    gates = mx.softmax(b.gate(x).astype(mx.float32))
    reference = mx.argpartition(gates, kth=-k)[..., -k:]
    indices, scores = fusion.route(b, x)
    assert set(reference.reshape(-1).tolist()) == set(indices.reshape(-1).tolist())
    expected = mx.take_along_axis(gates, indices, -1)
    expected /= expected.sum(-1, keepdims=True)
    assert mx.allclose(scores, expected, atol=1e-6).item()


def test_later_loader_patch_is_preserved(metal, monkeypatch):
    import mlx.nn as nn

    first = make_block()
    assert fusion.install_moe_decode_fusion(nn.Sequential(first)) == 1
    sentinel = object()
    monkeypatch.setattr(type(make_block()), "__call__", lambda self, x: sentinel)
    later = make_block()
    assert fusion.install_moe_decode_fusion(nn.Sequential(later)) == 1
    assert later(metal.zeros((2, 1, 128))) is sentinel


def test_non_metal_without_runtime(monkeypatch):
    """The portable no-Metal guard runs on Linux without MLX installed."""
    import builtins

    original = builtins.__import__
    core = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: False))
    fake_mlx = SimpleNamespace(core=core)
    fake_nn = SimpleNamespace(Linear=type("Linear", (), {}))

    def without_metal(name, *args, **kwargs):
        if name == "mlx.core":
            return fake_mlx
        if name == "mlx.nn":
            return SimpleNamespace(nn=fake_nn)
        return original(name, *args, **kwargs)

    monkeypatch.delenv("RAPID_MLX_MOE_DECODE_FUSION", raising=False)
    monkeypatch.setattr(builtins, "__import__", without_metal)
    assert fusion.install_moe_decode_fusion(object()) == 0
    assert not fusion.eligible(object(), object())
