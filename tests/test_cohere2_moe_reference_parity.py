# SPDX-License-Identifier: Apache-2.0
"""Reference agreement for Rapid's vendored Cohere2-MoE (North-Mini-Code).

The reference is transformers' ``Cohere2MoeForCausalLM``.  With
``prefix_dense_sliding_window_pattern == 1`` it rotates the dense-prefix layers
(``force_rope``) even though ``layer_types`` marks them full attention.  The
field changes the math without changing any weight shape, so a port that
ignores it still loads cleanly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

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
