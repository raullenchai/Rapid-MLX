"""Mixed-precision checkpoints qualify each fused decode lever per layer.

Later checkpoints quantize the routed gate/up projections of some layers to
3 bits (down stays 2-bit) and change which dense projections are 8-bit. Every
fused lever must decide per layer: fuse the layouts it was qualified for and
leave every other layer on the stock path with identical output.
"""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from rapid_mlx.models.deepseek_v41_native import head_gemv, moe_decode  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.moe import MoE  # noqa: E402
from rapid_mlx.models.deepseek_v41_native.serving import (  # noqa: E402
    ExactDirectDownSwitchGLU,
)

DIM, INTER, EXPERTS, TOPK = 512, 256, 64, 4


def _layer(
    gate_up_bits: int, *, down_bits: int = 2, shared_bits: int = 8, seed: int = 0
) -> MoE:
    mx.random.seed(seed)
    layer = MoE(
        ModelArgs(
            dim=DIM,
            moe_inter_dim=INTER,
            n_routed_experts=EXPERTS,
            n_activated_experts=TOPK,
            swiglu_limit=10.0,
        )
    )
    layer.update(
        nn.utils.tree_map(
            lambda p: (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16),
            layer.parameters(),
        )
    )
    layer.gate.bias = mx.random.normal((EXPERTS,)) * 0.02
    for prefix, bits in (
        ("experts.gate_proj", gate_up_bits),
        ("experts.up_proj", gate_up_bits),
        ("experts.down_proj", down_bits),
        ("shared_experts.", shared_bits),
    ):
        nn.quantize(
            layer,
            group_size=64,
            bits=bits,
            class_predicate=lambda path, m, prefix=prefix: (
                path.startswith(prefix) and hasattr(m, "to_quantized")
            ),
        )
    layer.experts.__class__ = ExactDirectDownSwitchGLU
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


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("shared", [2, 4, 8])
def test_mixed_gate_up_layers_are_fusable(bits, shared) -> None:
    assert moe_decode._layout(_layer(bits, shared_bits=shared)) == (
        DIM,
        INTER,
        shared,
        bits,
        bits,
        2,
    )
    assert moe_decode._layout(_layer(bits, down_bits=3, shared_bits=shared)) == (
        DIM,
        INTER,
        shared,
        bits,
        bits,
        3,
    )


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_mixed_2bit_3bit_stack_fuses_per_layer(gpu, rows: int) -> None:
    # e.g. layer 14 (2-bit) next to layer 15 (3-bit gate/up)
    layers = [_layer(2, seed=14), _layer(3, seed=15), _layer(2, shared_bits=2, seed=16)]
    calls = []
    original = moe_decode._experts

    def spy(moe, *args):
        calls.append(id(moe))
        return original(moe, *args)

    moe_decode._experts = spy
    try:
        x = mx.random.normal((rows, DIM)).astype(mx.bfloat16)
        for layer in layers:
            out = layer(x)
            expected = mx.concatenate([layer._stock(x[r : r + 1]) for r in range(rows)])
            mx.eval(out, expected)
            assert mx.array_equal(out, expected).item()
            x = out
    finally:
        moe_decode._experts = original
    fused = {id(layer) for layer in layers}
    assert fused <= set(calls)


def test_quantized_or_non_bf16_head_and_router_stay_stock(gpu) -> None:
    x = mx.random.normal((1, DIM))
    head = nn.Linear(DIM, 1024, bias=False)
    head.weight = head.weight.astype(mx.bfloat16)
    assert head_gemv.supports(head, x)
    quantized = nn.QuantizedLinear.from_linear(head, group_size=64, bits=8)
    assert not head_gemv.supports(quantized, x)
    fp16 = nn.Linear(DIM, 1024, bias=False)
    fp16.weight = fp16.weight.astype(mx.float16)
    assert not head_gemv.supports(fp16, x)
    gate = _layer(2).gate
    gate.weight = gate.weight.astype(mx.float16)
    scores = head_gemv.project(gate, x)
    assert mx.array_equal(scores, x @ gate.weight.astype(mx.float32).T).item()
