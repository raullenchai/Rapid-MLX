# SPDX-License-Identifier: Apache-2.0
"""Contract coverage for Rapid's vendored Cohere2-MoE architecture.

These checks intentionally never instantiate model weights or execute MLX
operations.  They protect the configuration, cache, registration, and loading
key seams that make the model reachable through the mlx-lm loader.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from vllm_mlx.model_aliases import resolve_model
from vllm_mlx.model_auto_config import detect_model_config
from vllm_mlx.models import cohere2_moe
from vllm_mlx.utils import tokenizer


def test_model_args_derives_north_attention_schedule() -> None:
    args = cohere2_moe.ModelArgs(model_type="cohere2_moe", num_hidden_layers=8)

    assert args.model_type == "cohere2_moe"
    assert (
        args.layer_types
        == [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ]
        * 2
    )


def test_model_args_rejects_mismatched_explicit_attention_schedule() -> None:
    try:
        cohere2_moe.ModelArgs(
            model_type="cohere2_moe",
            num_hidden_layers=2,
            layer_types=["full_attention"],
        )
    except ValueError as exc:
        assert "layer_types" in str(exc)
    else:
        raise AssertionError(
            "mismatched layer_types must fail before model construction"
        )


def test_vendor_registration_exposes_mlx_lm_architecture_module() -> None:
    module_name = "mlx_lm.models.cohere2_moe"
    previous_module = sys.modules.pop(module_name, None)
    was_registered = "cohere2_moe" in tokenizer._VENDORED_MODEL_TYPES
    tokenizer._VENDORED_MODEL_TYPES.discard("cohere2_moe")
    try:
        tokenizer._register_vendored_archs()
        assert sys.modules[module_name] is cohere2_moe
        assert "cohere2_moe" in tokenizer._VENDORED_MODEL_TYPES
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if not was_registered:
            tokenizer._VENDORED_MODEL_TYPES.discard("cohere2_moe")


def test_cache_layout_and_weight_key_cleanup_need_no_model_weights() -> None:
    model = SimpleNamespace(
        args=SimpleNamespace(
            sliding_window=4096,
            layer_types=["full_attention", "sliding_attention", "sliding_attention"],
            tie_word_embeddings=True,
        )
    )

    caches = cohere2_moe.Model.make_cache(model)
    assert [cache.__class__.__name__ for cache in caches] == [
        "KVCache",
        "RotatingKVCache",
        "RotatingKVCache",
    ]

    weights = {
        "language_model.model.embed_tokens.weight": object(),
        "language_model.lm_head.weight": object(),
        "language_model.rotary_emb.inv_freq": object(),
    }
    assert cohere2_moe.Model.sanitize(model, weights) == {
        "model.embed_tokens.weight": weights["language_model.model.embed_tokens.weight"]
    }


def test_checkpoint_profile_uses_native_parser_and_conservative_capabilities() -> None:
    profile = detect_model_config("mlx-community/North-Mini-Code-1.0-bf16")

    assert profile is not None
    assert profile.tool_call_parser == "cohere"
    assert profile.reasoning_parser == "cohere"
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False


def test_public_4bit_alias_uses_native_parser_and_conservative_capabilities() -> None:
    assert (
        resolve_model("north-mini-code-4bit")
        == "mlx-community/North-Mini-Code-1.0-4bit"
    )

    profile = detect_model_config("north-mini-code-4bit")

    assert profile is not None
    assert profile.hf_path == "mlx-community/North-Mini-Code-1.0-4bit"
    assert profile.tool_call_parser == "cohere"
    assert profile.reasoning_parser == "cohere"
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False
