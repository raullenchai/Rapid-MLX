from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from deepseek_v41_native.attention import Attention, GroupedOutputLinear  # noqa: E402
from deepseek_v41_native.compressor import Compressor, CompressorState  # noqa: E402
from deepseek_v41_native.config import ModelArgs  # noqa: E402
from deepseek_v41_native.load import reshape_grouped_wo_a  # noqa: E402
from deepseek_v41_native.model import Model  # noqa: E402


def test_reshape_grouped_wo_a_restores_quantized_parameter_axes() -> None:
    args = ModelArgs(o_groups=2, o_lora_rank=3)
    weight = mx.arange(24, dtype=mx.uint32).reshape(6, 4)
    scales = mx.arange(12, dtype=mx.float32).reshape(6, 2)

    values = dict(
        reshape_grouped_wo_a(
            [
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wo_a.scales", scales),
                ("layers.0.attn.wq_a.weight", weight),
            ],
            args,
        )
    )

    assert values["layers.0.attn.wo_a.weight"].shape == (2, 3, 4)
    assert values["layers.0.attn.wo_a.scales"].shape == (2, 3, 2)
    assert values["layers.0.attn.wq_a.weight"].shape == (6, 4)


def test_reshape_grouped_wo_a_rejects_incompatible_rows() -> None:
    args = ModelArgs(o_groups=2, o_lora_rank=3)
    with pytest.raises(ValueError, match="expected 6"):
        reshape_grouped_wo_a([("layers.0.attn.wo_a.weight", mx.zeros((5, 4)))], args)


def test_quantized_grouped_wo_a_preserves_batch_sequence_and_group_axes() -> None:
    args = ModelArgs(
        dim=16,
        n_heads=8,
        head_dim=8,
        rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=3,
        o_groups=2,
        n_layers=1,
        compress_ratios=(0,),
    )
    attention = Attention(0, args)
    attention.wo_a = attention.wo_a.to_quantized(group_size=32, bits=2)
    grouped = mx.zeros((1, 5, args.o_groups, 32), dtype=mx.float32)

    projected = attention.wo_a(grouped[..., None, :]).squeeze(-2)

    assert projected.shape == (1, 5, args.o_groups, args.o_lora_rank)


def test_grouped_wo_a_accepts_release_flat_float_layout_for_prefill() -> None:
    layer = GroupedOutputLinear(input_dims=8, output_dims=3, num_heads=2)
    flat = mx.arange(48, dtype=mx.float32).reshape(6, 8)
    layer.weight = flat
    values = mx.arange(2 * 5 * 2 * 8, dtype=mx.float32).reshape(2, 5, 2, 8)

    actual = layer(values[..., None, :]).squeeze(-2)
    expected = mx.einsum("bsgd,grd->bsgr", values, flat.reshape(2, 3, 8))
    mx.eval(actual, expected)

    assert actual.shape == (2, 5, 2, 3)
    assert mx.array_equal(actual, expected).item()


def test_quantized_compressor_uses_logical_module_projection() -> None:
    args = ModelArgs(
        dim=64,
        head_dim=32,
        n_layers=1,
        compress_ratios=(2,),
    )
    compressor = Compressor(args, 0)
    compressor.wkv = compressor.wkv.to_quantized(group_size=32, bits=2)
    compressor.wgate = compressor.wgate.to_quantized(group_size=32, bits=2)
    state = CompressorState(bsz=1, ratio=2, head_dim=args.head_dim)

    output = compressor(mx.zeros((1, 2, args.dim)), start_pos=0, comp_state=state)
    mx.eval(output)

    assert output.shape == (1, 1, args.head_dim)


def test_native_model_defaults_to_watchdog_safe_layer_boundaries() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
    )

    assert Model(args).eval_interval == 1
