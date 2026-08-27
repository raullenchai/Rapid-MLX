"""Optional numerical parity checks against the architecture reference.

The regular test environment does not install Torch or the unreleased model
module.  Day-0 validation runs this file in the pinned scratch environment
recorded in the engineering handoff.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

try:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpTextConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        Qwen4ExpTextPLELayer,
    )
except ImportError:
    pytest.skip("qwen4_exp reference module is not installed", allow_module_level=True)

from vllm_mlx.models.qwen4_exp import PLELayer, TextModel, TextModelArgs


def _tiny_config(**overrides):
    values = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "hc_count": 4,
        "hc_lowrank": 3,
        "layer_types": ["linear_attention", "full_attention"],
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_budget": 2,
        "indexer_compress_ratio": 2,
        "ple_layer_ids": [],
        "eos_token_id": 31,
        "rope_parameters": {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
        "output_gate_type": "sigmoid",
        "max_position_embeddings": 32768,
    }
    values.update(overrides)
    return values


def _to_mx_state(state):
    return {key: mx.array(value.detach().numpy()) for key, value in state.items()}


def test_two_layer_text_logits_match_pinned_architecture_reference():
    values = _tiny_config()
    torch.manual_seed(123)
    reference = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**values)).eval()
    candidate = TextModel(TextModelArgs(**values))
    # The reference path is always unfused. Training mode selects the matching
    # deterministic MLX recurrence rather than the production fused kernel.
    candidate.train()
    weights = candidate.sanitize(_to_mx_state(reference.state_dict()))
    candidate.load_weights(list(weights.items()), strict=True)

    input_ids = np.array([[1, 2, 3]], dtype=np.int64)
    with torch.no_grad():
        expected = (
            reference(torch.from_numpy(input_ids), use_cache=False)
            .logits.float()
            .numpy()
        )
    actual = candidate(mx.array(input_ids))
    mx.eval(actual)
    np.testing.assert_allclose(np.array(actual), expected, rtol=3e-2, atol=1.2e-3)

    with torch.no_grad():
        reference_prompt = reference(torch.from_numpy(input_ids), use_cache=True)
        reference_decode = (
            reference(
                torch.tensor([[4]]),
                past_key_values=reference_prompt.past_key_values,
                use_cache=True,
            )
            .logits.float()
            .numpy()
        )
    cache = candidate.make_cache()
    candidate_prompt = candidate(mx.array(input_ids), cache=cache)
    mx.eval(candidate_prompt, [layer.state for layer in cache])
    candidate_decode = candidate(mx.array([[4]]), cache=cache)
    mx.eval(candidate_decode, [layer.state for layer in cache])
    np.testing.assert_allclose(
        np.array(candidate_decode),
        reference_decode,
        rtol=3e-2,
        atol=1.2e-3,
    )


def test_ple_output_matches_pinned_architecture_reference():
    values = _tiny_config(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        ple_conv_kernel_size=4,
    )
    config = Qwen4ExpTextConfig(**values)
    torch.manual_seed(321)
    reference = Qwen4ExpTextPLELayer(config, layer_idx=0, ple_layer_index=0).eval()
    candidate = PLELayer(TextModelArgs(**values), ple_layer_index=0)

    state = _to_mx_state(reference.state_dict())
    embedding = state.pop("ple_embedding.ngram_embedding.weight")
    rows = embedding.shape[0] // len(candidate.ple_embedding.ngram_embedding.shards)
    for index in range(len(candidate.ple_embedding.ngram_embedding.shards)):
        state[f"ple_embedding.ngram_embedding.shards.{index}.weight"] = embedding[
            index * rows : (index + 1) * rows
        ]
    conv = state["conv1d.weight"]
    state["conv1d.weight"] = conv.moveaxis(2, 1)
    candidate.load_weights(list(state.items()), strict=True)

    rng = np.random.default_rng(11)
    hidden = rng.normal(0, 0.1, (1, 4, 32)).astype(np.float32)
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    with torch.no_grad():
        expected = (
            reference(
                torch.from_numpy(hidden),
                torch.from_numpy(input_ids),
                None,
            )
            .float()
            .numpy()
        )
    actual = candidate(mx.array(hidden), mx.array(input_ids), None)
    mx.eval(actual)
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-4, atol=2e-5)
