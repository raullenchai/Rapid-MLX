from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from vllm_mlx.models.deepseek_v41_native.attention import (  # noqa: E402
    Attention,
    GroupedOutputLinear,
)
from vllm_mlx.models.deepseek_v41_native.cache import ModelCache  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.compressor import (  # noqa: E402
    Compressor,
    CompressorState,
)
from vllm_mlx.models.deepseek_v41_native.config import ModelArgs  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.load import reshape_grouped_wo_a  # noqa: E402
from vllm_mlx.models.deepseek_v41_native.model import Model  # noqa: E402


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


def test_native_model_captures_configured_dspark_inputs() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
        dspark_target_layer_ids=(0,),
    )
    model = Model(args)

    class IdentityBlock:
        engram = None

        def __call__(self, h, pre_mix, *_args):
            return h, pre_mix

    model.layers = [IdentityBlock()]
    cache = model.make_cache(max_seq_len=8)

    logits, hidden = model(mx.array([[1, 2]]), cache, return_dspark_hidden=True)
    mx.eval(logits, hidden)

    assert logits.shape == (1, 2, args.vocab_size)
    assert hidden.shape == (1, 2, args.dim)


def test_native_model_captures_dspark_inputs_in_configured_order() -> None:
    args = ModelArgs(
        dim=64,
        vocab_size=32,
        n_layers=0,
        n_heads=1,
        head_dim=64,
        rope_head_dim=32,
        hc_mult=1,
        compress_ratios=(),
        dspark_target_layer_ids=(1, 0),
    )
    model = Model(args)

    class AddBlock:
        engram = None

        def __init__(self, value):
            self.value = value

        def __call__(self, h, pre_mix, *_args):
            return h + self.value, pre_mix

    model.layers = [AddBlock(1), AddBlock(2)]
    cache = model.make_cache(max_seq_len=8)

    _, hidden = model(mx.array([[1]]), cache, return_dspark_hidden=True)
    first, second = mx.split(hidden, 2, axis=-1)

    assert mx.array_equal(first, second + 1).item()


def test_speculative_cache_rollback_restores_partial_compressor_group() -> None:
    args = ModelArgs(
        dim=64,
        n_layers=1,
        head_dim=32,
        compress_ratios=(2,),
        kv_source_layers=(0,),
    )
    cache = ModelCache(args, max_seq_len=16)
    state = cache.layers[0].comp_state
    assert state is not None
    cache.offset = 4
    cache.begin_forward()
    state.pending_start = 4
    state.pending_kv = mx.arange(3 * 32).reshape(1, 3, 32).astype(mx.float32)
    state.pending_score = state.pending_kv + 100
    expected_kv = state.pending_kv[:, :1]
    expected_score = state.pending_score[:, :1]
    cache.offset = 7

    cache.rollback(5)

    assert cache.offset == 5
    assert mx.array_equal(state.kv_state[:, :1], expected_kv).item()
    assert mx.array_equal(state.score_state[:, :1], expected_score).item()


def test_speculative_cache_rollback_rejects_older_forward() -> None:
    args = ModelArgs(dim=64, n_layers=0, head_dim=32, compress_ratios=())
    cache = ModelCache(args, max_seq_len=16)
    cache.rollback_start = 4
    cache.offset = 7

    with pytest.raises(ValueError, match="outside latest forward"):
        cache.rollback(3)


def test_compressor_rollback_to_group_boundary_clears_partial_state() -> None:
    state = CompressorState(bsz=1, ratio=2, head_dim=4)
    state.begin_forward(3)
    state.kv_state[:] = 7
    state.score_state[:] = 9

    state.rollback(4)

    assert mx.array_equal(state.kv_state, mx.zeros_like(state.kv_state)).item()
    assert mx.all(mx.isneginf(state.score_state)).item()


def test_compressor_rollback_to_forward_start_restores_carried_partial() -> None:
    state = CompressorState(bsz=1, ratio=2, head_dim=4)
    state.kv_state[:, :1] = 3
    state.score_state[:, :1] = 5
    state.begin_forward(3)
    state.kv_state[:] = 9
    state.score_state[:] = 11

    state.rollback(3)

    assert mx.all(state.kv_state[:, :1] == 3).item()
    assert mx.all(state.score_state[:, :1] == 5).item()


def test_model_only_snapshots_cache_when_rollback_is_enabled(monkeypatch) -> None:
    args = ModelArgs(dim=64, n_layers=0, head_dim=32, compress_ratios=())
    model = Model(args)
    cache = model.make_cache(max_seq_len=8)
    calls = 0
    original = cache.begin_forward

    def record_begin_forward():
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(cache, "begin_forward", record_begin_forward)

    model(mx.array([[1]]), cache)
    assert calls == 0

    model(mx.array([[1]]), cache, enable_rollback=True)
    assert calls == 1
