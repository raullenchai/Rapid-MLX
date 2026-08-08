# SPDX-License-Identifier: Apache-2.0
"""Focused synthetic coverage for the vendored Sarvam MLA adapter."""

import importlib
import sys

import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache


@pytest.fixture(autouse=True)
def _clear_vendored_registration():
    from vllm_mlx.utils.tokenizer import _VENDORED_MODEL_TYPES

    sys.modules.pop("mlx_lm.models.sarvam_mla", None)
    _VENDORED_MODEL_TYPES.discard("sarvam_mla")
    yield
    sys.modules.pop("mlx_lm.models.sarvam_mla", None)
    _VENDORED_MODEL_TYPES.discard("sarvam_mla")


def _tiny_args(**overrides):
    from vllm_mlx.models.sarvam_mla import ModelArgs

    values = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        # Sarvam's omitted group default is routed_experts / 8 with top-2
        # groups, so 16 is the smallest synthetic shape with valid 2/2
        # group routing.
        "num_experts": 16,
        "num_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "kv_lora_rank": 4,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 8,
    }
    values.update(overrides)
    return ModelArgs(**values)


def test_sarvam_config_remap_defaults_and_explicit_group_controls():
    args = _tiny_args()
    assert args.model_type == "sarvam_mla"
    assert args.q_lora_rank is None
    assert (args.n_routed_experts, args.n_shared_experts) == (16, 1)
    assert (args.n_group, args.topk_group) == (2, 2)

    explicit = _tiny_args(n_group=1, topk_group=1)
    assert (explicit.n_group, explicit.topk_group) == (1, 1)


def test_registers_with_mlx_lm_loader_lookup():
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    module = importlib.import_module("mlx_lm.models.sarvam_mla")
    assert module.__name__ == "vllm_mlx.models.sarvam_mla"
    assert module.ModelArgs is not None


def test_loader_routes_sarvam_config_to_vendored_path(tmp_path, monkeypatch):
    import json

    from vllm_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "sarvam_mla"}),
        encoding="utf-8",
    )
    expected = (object(), object())
    seen = []

    def _fake_vendor_loader(model_name):
        seen.append(model_name)
        return expected

    monkeypatch.setattr(tokenizer, "_load_with_tokenizer_fallback", _fake_vendor_loader)

    assert tokenizer._load_model_with_fallback_impl(str(tmp_path)) is expected
    assert seen == [str(tmp_path)]


def test_sanitize_and_prompt_cache_follow_deepseek_v3_contract():
    from vllm_mlx.models.sarvam_mla import Model

    model = Model(_tiny_args())
    sanitized = model.sanitize(
        {
            "model.layers.0.rotary_emb.inv_freq": mx.ones((4,)),
            "model.layers.0.input_layernorm.weight": mx.ones((64,)),
        }
    )
    assert "model.layers.0.rotary_emb.inv_freq" not in sanitized
    assert "model.layers.0.input_layernorm.weight" in sanitized

    cache = make_prompt_cache(model)
    assert len(cache) == model.args.num_hidden_layers
    logits = model(mx.array([[1, 2]], dtype=mx.int32), cache=cache)
    mx.eval(logits, [entry.state for entry in cache])
    assert logits.shape == (1, 2, model.args.vocab_size)
