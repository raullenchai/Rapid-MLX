# SPDX-License-Identifier: Apache-2.0
"""Reference agreement for Rapid's vendored Cohere2-MoE (North-Mini-Code).

The reference is transformers' ``Cohere2MoeForCausalLM``.  Two config fields
change the math without changing any weight shape, so a port that ignores them
still loads cleanly:

* ``prefix_dense_sliding_window_pattern == 1`` rotates the dense-prefix layers
  (``force_rope``) even though ``layer_types`` marks them full attention;
* ``rms_norm_eps`` selects RMSNorm over the mean-centred LayerNorm for every
  norm, including the final one.

The Torch parity test runs only where Torch and a transformers release with
Cohere2-MoE are installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from rapid_mlx.models import cohere2_moe

# North-Mini-Code's shape, shrunk: one dense prefix layer, full attention every
# fourth layer, the rest sliding.
TINY = dict(
    model_type="cohere2_moe",
    hidden_size=64,
    head_dim=16,
    num_hidden_layers=8,
    intermediate_size=32,
    prefix_dense_intermediate_size=96,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=96,
    rope_theta=50000.0,
    rms_norm_eps=1e-6,
    sliding_window=6,
    num_experts=8,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    prefix_dense_sliding_window_pattern=1,
    layer_types=[
        "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(8)
    ],
)


def _model(**overrides) -> cohere2_moe.Model:
    config = {**TINY, **overrides}
    config = {key: value for key, value in config.items() if value is not None}
    return cohere2_moe.Model(cohere2_moe.ModelArgs.from_dict(config))


def _rotated(model: cohere2_moe.Model) -> list[bool]:
    return [layer.self_attn.rope is not None for layer in model.layers]


def test_dense_prefix_layer_is_rotated_when_prefix_pattern_is_one() -> None:
    assert _rotated(_model()) == [True, True, True, True, False, True, True, True]
    # The reference config class defaults the pattern to 1.
    assert _rotated(_model(prefix_dense_sliding_window_pattern=None))[0] is True


def test_dense_prefix_layer_is_not_rotated_for_other_prefix_patterns() -> None:
    model = _model(prefix_dense_sliding_window_pattern=2)
    assert _rotated(model) == [False, True, True, True, False, True, True, True]


def test_rotated_dense_prefix_attention_sees_token_order() -> None:
    # A causal NoPE layer sees the earlier tokens as an unordered set at the
    # last position, so swapping two of them cannot change that output.  The
    # rotated dense-prefix layer must see the swap; NoPE layer 4 must not.
    mx.random.seed(0)
    model = _model()
    x = mx.random.normal((1, 6, TINY["hidden_size"]))
    swapped = x[:, [1, 0, 2, 3, 4, 5]]

    def last(layer_index: int, inputs: mx.array) -> mx.array:
        return model.layers[layer_index].self_attn(inputs, mask="causal")[:, -1]

    assert not mx.allclose(last(0, x), last(0, swapped), atol=1e-4).item()
    assert mx.allclose(last(4, x), last(4, swapped), atol=1e-5).item()


def test_norm_class_follows_rms_norm_eps_like_the_reference() -> None:
    model = _model()
    norms = [layer.input_layernorm for layer in model.layers] + [model.model.norm]
    assert all(type(norm) is nn.RMSNorm for norm in norms)
    assert all(norm.eps == 1e-6 for norm in norms)

    legacy = _model(rms_norm_eps=None)
    norms = [layer.input_layernorm for layer in legacy.layers] + [legacy.model.norm]
    assert all(type(norm) is nn.LayerNorm for norm in norms)
    assert all(norm.eps == 1e-5 for norm in norms)


def test_tiny_model_matches_the_transformers_reference() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Cohere2MoeForCausalLM"):
        pytest.skip("installed transformers has no Cohere2-MoE reference")

    hf_kwargs = {key: value for key, value in TINY.items() if key != "model_type"}
    config = transformers.Cohere2MoeConfig(
        **hf_kwargs,
        layer_norm_eps=1e-5,
        logit_scale=1.0,
        num_shared_experts=0,
        norm_topk_prob=False,
        expert_selection_fn="sigmoid",
        use_parallel_block=True,
        use_qk_norm=False,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=None,
    )
    torch.manual_seed(0)
    reference = transformers.Cohere2MoeForCausalLM(config).eval()
    if not hasattr(reference.model.layers[0].self_attn, "force_rope"):
        pytest.skip("installed transformers predates the dense-prefix RoPE")
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            offset = 1.0 if name.endswith("norm.weight") else 0.0
            parameter.copy_(torch.randn_like(parameter) * 0.2 + offset)

    # Split the reference's stacked experts into the SwitchGLU layout.
    weights = {}
    for key, value in reference.state_dict().items():
        value = value.detach().float().numpy()
        if key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key.removesuffix(".experts.gate_up_proj") + ".switch_mlp"
            half = value.shape[1] // 2
            weights[prefix + ".gate_proj.weight"] = mx.array(value[:, :half])
            weights[prefix + ".up_proj.weight"] = mx.array(value[:, half:])
        elif key.endswith(".mlp.experts.down_proj"):
            prefix = key.removesuffix(".experts.down_proj") + ".switch_mlp"
            weights[prefix + ".down_proj.weight"] = mx.array(value)
        else:
            weights[key] = mx.array(value)
    model = _model()
    model.load_weights(list(model.sanitize(weights).items()), strict=True)

    tokens = np.random.RandomState(1).randint(0, TINY["vocab_size"], (1, 13))
    with torch.no_grad():
        expected = reference(torch.tensor(tokens)).logits.float().numpy()
    # Float32 GPU matmul precision varies by chip (about 2e-2 here on an M5),
    # so compare on the CPU, where this tolerance is chip-independent.
    with mx.stream(mx.cpu):
        actual = np.array(model(mx.array(tokens)).astype(mx.float32))
    assert np.abs(expected - actual).max() < 1e-4
