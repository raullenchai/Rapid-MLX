# SPDX-License-Identifier: Apache-2.0
"""Focused synthetic coverage for the vendored GraniteMoE SWA model."""

import importlib
import sys

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache


@pytest.fixture(autouse=True)
def _clear_vendored_registration():
    sys.modules.pop("mlx_lm.models.granitemoe_swa", None)
    yield
    sys.modules.pop("mlx_lm.models.granitemoe_swa", None)


def _tiny_args(**overrides):
    from vllm_mlx.models.granitemoe_swa import ModelArgs

    values = {
        "model_type": "granitemoe_swa",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 64,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "embedding_multiplier": 1.0,
        "attention_multiplier": 0.125,
        "residual_multiplier": 1.0,
        "logits_scaling": 1.0,
        "sliding_window": 4,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "rope_parameters": {"rope_theta": 10000.0},
    }
    values.update(overrides)
    return ModelArgs(**values)


def test_registers_with_mlx_lm_loader_lookup():
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    module = importlib.import_module("mlx_lm.models.granitemoe_swa")
    assert module.__name__ == "vllm_mlx.models.granitemoe_swa"
    assert module.ModelArgs is not None


def test_mixed_cache_contract_and_tiny_forward():
    from vllm_mlx.models.granitemoe_swa import Model

    model = Model(_tiny_args())
    cache = model.make_cache()
    assert isinstance(cache[0], KVCache)
    assert isinstance(cache[1], RotatingKVCache)
    assert cache[1].max_size == 4
    assert isinstance(cache[3], KVCache)

    capped = model.make_cache(max_kv_size=8)
    assert isinstance(capped[0], RotatingKVCache)
    assert capped[0].max_size == 8
    assert capped[1].max_size == 4

    logits = model(mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32), cache=cache)
    mx.eval(logits, [entry.state for entry in cache])
    assert logits.shape == (1, 5, 128)
    assert model.supports_speculative_rollback is True


def test_sanitize_splits_gate_up_by_halves_and_drops_tied_lm_head():
    from vllm_mlx.models.granitemoe_swa import Model

    model = Model(_tiny_args(num_hidden_layers=1, layer_types=["full_attention"]))
    fused = mx.arange(2 * 8 * 4).reshape(2, 8, 4)
    sanitized = model.sanitize(
        {
            "model.layers.0.block_sparse_moe.experts.gate_up_proj": fused,
            "model.layers.0.block_sparse_moe.experts.down_proj": mx.ones((2, 4, 8)),
            "lm_head.weight": mx.ones((128, 64)),
        }
    )
    gate = sanitized["model.layers.0.block_sparse_moe.experts.gate_proj.weight"]
    up = sanitized["model.layers.0.block_sparse_moe.experts.up_proj.weight"]
    assert gate.shape == up.shape == (2, 4, 4)
    assert mx.array_equal(gate, fused[:, :4, :])
    assert mx.array_equal(up, fused[:, 4:, :])
    assert "model.layers.0.block_sparse_moe.experts.down_proj.weight" in sanitized
    assert "lm_head.weight" not in sanitized
