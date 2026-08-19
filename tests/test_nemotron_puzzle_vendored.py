# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for the Nemotron Puzzle heterogeneous-MoE vendor."""

import importlib
import sys

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def _restore_nemotron_modules():
    """Registration replaces an old native module; restore test isolation."""
    from vllm_mlx.utils.tokenizer import _VENDORED_MODEL_TYPES

    original = sys.modules.get("mlx_lm.models.nemotron_h")
    if getattr(original, "__name__", "") == "vllm_mlx.models.nemotron_h":
        # A previous test already ran ``_register_vendored_archs()`` (any of
        # the vendored-arch suites calls it), leaving our vendor installed
        # under the native name. Start from the pristine state so this
        # test's registration re-evaluates the native-``block_configs``
        # probe instead of short-circuiting on our own module.
        original = None
        sys.modules.pop("mlx_lm.models.nemotron_h", None)
    sys.modules.pop("mlx_lm.models.nemotron_h_puzzle", None)
    _VENDORED_MODEL_TYPES.difference_update({"nemotron_h", "nemotron_h_puzzle"})
    yield
    sys.modules.pop("mlx_lm.models.nemotron_h_puzzle", None)
    if original is not None:
        sys.modules["mlx_lm.models.nemotron_h"] = original
    else:
        sys.modules.pop("mlx_lm.models.nemotron_h", None)
    _VENDORED_MODEL_TYPES.difference_update({"nemotron_h", "nemotron_h_puzzle"})


def _args(model_type="nemotron_h_puzzle"):
    from vllm_mlx.models.nemotron_h import ModelArgs

    return ModelArgs(
        model_type=model_type,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        max_position_embeddings=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        mamba_num_heads=4,
        mamba_head_dim=8,
        mamba_proj_bias=False,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=False,
        layers_block_type=["moe", "mamba", "attention", "moe"],
        moe_intermediate_size=48,
        moe_latent_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        block_configs=[
            {"moe_intermediate_size": 64, "num_experts_per_tok": 2},
            {},
            {},
            {"moe_intermediate_size": 96, "num_experts_per_tok": 4},
        ],
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention"],
        mtp_block_configs=[{"num_experts_per_tok": 4}],
    )


def test_heterogeneous_blocks_forward_and_cache():
    from vllm_mlx.models.nemotron_h import Model

    args = _args()
    model = Model(args)
    assert not hasattr(model, "mtp")
    first, _, _, last = model.layers
    assert first.mixer.num_experts_per_tok == 2
    assert last.mixer.num_experts_per_tok == 4
    assert first.mixer.switch_mlp.fc1.weight.shape[-2] == 64
    assert last.mixer.switch_mlp.fc1.weight.shape[-2] == 96

    cache = model.make_cache()
    assert len(cache) == 2
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    assert cache[1].offset == 3
    next_logits = model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(next_logits)
    assert next_logits.shape == (1, 1, args.vocab_size)
    assert cache[1].offset == 4


def test_loader_registration_dispatches_both_puzzle_config_names():
    from mlx_lm.utils import _get_classes

    from vllm_mlx.utils import tokenizer

    tokenizer._register_vendored_archs()
    for model_type in ("nemotron_h", "nemotron_h_puzzle"):
        module = importlib.import_module(f"mlx_lm.models.{model_type}")
        assert module.__name__ == "vllm_mlx.models.nemotron_h"
        model_class, args_class = _get_classes({"model_type": model_type})
        assert model_class is module.Model
        assert args_class is module.ModelArgs
    assert {"nemotron_h", "nemotron_h_puzzle"} <= tokenizer._VENDORED_MODEL_TYPES


def test_puzzle_quantization_keeps_lm_head_unquantized():
    from vllm_mlx.models.nemotron_h import Model

    puzzle = Model(_args("nemotron_h_puzzle"))
    assert not puzzle.quant_predicate("lm_head", puzzle.lm_head)
    assert puzzle.quant_predicate("backbone.layers.0.mixer", puzzle.layers[0].mixer)

    uniform = Model(_args("nemotron_h"))
    assert uniform.quant_predicate("lm_head", uniform.lm_head)


def test_profile_uses_nemotron_xml_parser_and_disables_speculation():
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile("nemotron-puzzle-75b-a9b-6bit")
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid and profile.is_moe
    assert not profile.supports_spec_decode
