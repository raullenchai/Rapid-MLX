# SPDX-License-Identifier: Apache-2.0
"""Non-GPU structural coverage for the Apertus 1.5 text vendor."""

import importlib
import sys

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def _clear_vendored_registration():
    """Registration is process-global; reset its module entry per test."""
    sys.modules.pop("mlx_lm.models.apertus1p5_text", None)
    yield
    sys.modules.pop("mlx_lm.models.apertus1p5_text", None)


def _args(*, tie_word_embeddings=False):
    from vllm_mlx.models.apertus1p5_text import ModelArgs

    return ModelArgs(
        model_type="apertus1p5_text",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=96,
        output_vocab_size=48,
        max_position_embeddings=128,
        post_norm=False,
        qk_norm=True,
        tie_word_embeddings=tie_word_embeddings,
        rope_parameters={
            "rope_theta": 10000.0,
            "type": "llama3",
            "factor": 8.0,
        },
    )


def test_config_derives_native_apertus_rope_arguments():
    args = _args()

    assert args.rope_theta == 10000.0
    assert args.rope_scaling == {"rope_type": "llama3", "factor": 8.0}


def test_split_vocab_head_and_default_cache_are_structural_only():
    """Do not forward/evaluate: the centralized MLX GPU gate is closed."""
    from mlx_lm.models.cache import make_prompt_cache

    from vllm_mlx.models.apertus1p5_text import Model

    model = Model(_args())
    assert model.model.embed_tokens.weight.shape == (96, 32)
    assert model.lm_head.weight.shape == (48, 32)
    cache = make_prompt_cache(model)
    assert len(cache) == 2
    assert all(cache_entry.offset == 0 for cache_entry in cache)


def test_sanitize_remaps_text_tower_and_scalar_xielu_parameters():
    from vllm_mlx.models.apertus1p5_text import Model

    model = Model(_args())
    weights = {
        "model.language_model.layers.0.mlp.act_fn.alpha_p": mx.array([1.0]),
        "model.language_model.layers.0.mlp.act_fn.alpha_n": mx.array([2.0]),
        "model.language_model.layers.0.mlp.act_fn.beta": mx.array([3.0]),
        "model.language_model.layers.0.mlp.act_fn.eps": mx.array([4.0]),
        "lm_head.weight": mx.ones((48, 32)),
    }

    sanitized = model.sanitize(weights)
    assert "model.layers.0.mlp.act_fn.alpha_p" in sanitized
    assert sanitized["model.layers.0.mlp.act_fn.alpha_p"].shape == ()
    assert sanitized["model.layers.0.mlp.act_fn.alpha_n"].shape == ()
    assert sanitized["model.layers.0.mlp.act_fn.beta"].shape == ()
    assert sanitized["model.layers.0.mlp.act_fn.eps"].shape == ()
    assert "lm_head.weight" in sanitized

    tied = Model(_args(tie_word_embeddings=True))
    assert "lm_head.weight" not in tied.sanitize(weights)


def test_loader_registration_and_auto_profile_dispatch():
    from mlx_lm.utils import _get_classes

    from vllm_mlx.model_auto_config import detect_model_config
    from vllm_mlx.utils import tokenizer

    tokenizer._register_vendored_archs()
    module = importlib.import_module("mlx_lm.models.apertus1p5_text")
    assert module.__name__ == "vllm_mlx.models.apertus1p5_text"
    model_class, args_class = _get_classes({"model_type": "apertus1p5_text"})
    assert model_class is module.Model
    assert args_class is module.ModelArgs

    profile = detect_model_config("apertus-v1.5-8b-text")
    assert profile.hf_path == "swiss-ai/Apertus-v1.5-8B"
    assert profile.is_text_only
    assert not profile.is_hybrid
    assert not profile.supports_spec_decode
