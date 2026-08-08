# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for the vendored mlx-lm #1488 GPT-OSS Puzzle port."""

import importlib
import json
import sys

import pytest


@pytest.fixture(autouse=True)
def _clear_puzzle_vendor_registration():
    """Keep process-global mlx-lm registration isolated across tests."""
    from vllm_mlx.utils.tokenizer import _VENDORED_MODEL_TYPES

    sys.modules.pop("mlx_lm.models.gpt_oss_puzzle", None)
    _VENDORED_MODEL_TYPES.discard("gpt_oss_puzzle")
    yield
    sys.modules.pop("mlx_lm.models.gpt_oss_puzzle", None)
    _VENDORED_MODEL_TYPES.discard("gpt_oss_puzzle")


def test_register_vendored_arch_makes_puzzle_visible_to_mlx_lm():
    from vllm_mlx.utils.tokenizer import (
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    _register_vendored_archs()

    module = importlib.import_module("mlx_lm.models.gpt_oss_puzzle")
    assert module.__name__ == "vllm_mlx.models.gpt_oss_puzzle"
    assert "gpt_oss_puzzle" in _VENDORED_MODEL_TYPES
    assert hasattr(module, "Model")


def test_vendored_arch_classifier_selects_low_level_loader(tmp_path):
    from vllm_mlx.utils.tokenizer import (
        _is_vendored_arch_model,
        _register_vendored_archs,
    )

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss_puzzle"}))
    _register_vendored_archs()

    assert _is_vendored_arch_model(str(tmp_path))


def test_loader_routes_puzzle_config_to_vendored_path(tmp_path, monkeypatch):
    from vllm_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_oss_puzzle"}))
    expected = (object(), object())
    seen = []

    def _fake_vendor_loader(model_name):
        seen.append(model_name)
        return expected

    monkeypatch.setattr(tokenizer, "_load_with_tokenizer_fallback", _fake_vendor_loader)

    assert tokenizer._load_model_with_fallback_impl(str(tmp_path)) is expected
    assert seen == [str(tmp_path)]


def test_puzzle_uses_per_layer_experts_and_cache_windows():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from vllm_mlx.models import gpt_oss_puzzle

    args = gpt_oss_puzzle.ModelArgs(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts_per_tok=2,
        yarn_rope_scaling={
            "rope_type": "yarn",
            "factor": 56.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        block_configs=[
            {"num_local_experts": 4, "sliding_window": 4},
            {"num_local_experts": 2, "sliding_window": None},
        ],
    )
    model = gpt_oss_puzzle.Model(args)
    cache = model.make_cache()
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(logits)

    assert [layer.mlp.num_local_experts for layer in model.layers] == [4, 2]
    assert args.rope_scaling == args.yarn_rope_scaling
    assert isinstance(cache[0], RotatingKVCache)
    assert cache[0].max_size == 4
    assert isinstance(cache[1], KVCache)
    assert logits.shape == (1, 3, args.vocab_size)


def test_sanitize_discards_puzzle_fp8_kv_calibration_scales():
    import mlx.core as mx

    from vllm_mlx.models import gpt_oss_puzzle

    model = gpt_oss_puzzle.Model(
        gpt_oss_puzzle.ModelArgs(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            head_dim=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts_per_tok=1,
            block_configs=[{"num_local_experts": 2, "sliding_window": None}],
        )
    )
    kept = mx.array([1.0])
    sanitized = model.sanitize(
        {
            "model.layers.0.mlp.experts.gate_proj.weight": kept,
            "model.layers.0.self_attn.k_scale": mx.array([2.0]),
            "model.layers.0.self_attn.v_scale": mx.array([3.0]),
        }
    )

    assert list(sanitized) == ["model.layers.0.mlp.experts.gate_proj.weight"]
    assert mx.array_equal(
        sanitized["model.layers.0.mlp.experts.gate_proj.weight"], kept
    )
