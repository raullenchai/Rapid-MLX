# SPDX-License-Identifier: Apache-2.0
"""Vendored IBM GraniteMoE sliding-window attention architecture.

Adapted from ml-explore/mlx-lm#1608 for ``granitemoe_swa`` checkpoints such
as ``ibm-granite/granite-swash-3b-a600m``. It uses the pinned mlx-lm cache and
attention primitives, with full-attention layers retaining ``KVCache`` and
sliding layers using ``RotatingKVCache``.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.activations import swiglu  # noqa: E402
from mlx_lm.models.base import (  # noqa: E402
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402
from mlx_lm.models.rope_utils import initialize_rope  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    shared_intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    embedding_multiplier: float
    attention_multiplier: float
    residual_multiplier: float
    logits_scaling: float
    sliding_window: int
    layer_types: list[str]
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    rope_parameters: dict[str, Any] | None = None

    def __post_init__(self):
        if self.rope_parameters:
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = dim // self.n_heads
        self.scale = args.attention_multiplier
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=args.attention_bias)
        self.sinks = mx.zeros((self.n_heads,))
        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            False,
            None,
            args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        batch, length, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(batch, length, self.n_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sinks
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.router = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.experts = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_local_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        logits = self.router(x).astype(mx.float32)
        indices = mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        gates = mx.softmax(
            mx.take_along_axis(logits, indices, axis=-1), precise=True, axis=-1
        )
        values = self.experts(x, indices)
        return (values * gates[..., None]).sum(axis=-2).astype(values.dtype)


class SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_linear = nn.Linear(
            args.hidden_size, args.shared_intermediate_size * 2, bias=False
        )
        self.output_linear = nn.Linear(
            args.shared_intermediate_size, args.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(swiglu(gate, up))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.residual_multiplier = args.residual_multiplier
        self.self_attn = Attention(args)
        self.block_sparse_moe = MoE(args)
        self.shared_mlp = SharedMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        hidden = x + self.self_attn(self.input_layernorm(x), mask, cache) * (
            self.residual_multiplier
        )
        normed = self.post_attention_layernorm(hidden)
        mlp_out = self.block_sparse_moe(normed) + self.shared_mlp(normed)
        return hidden + mlp_out * self.residual_multiplier


class GraniteMoeSwaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_type) for layer_type in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier
        self.layer_types = args.layer_types
        self.window_size = args.sliding_window
        self.full_index = args.layer_types.index("full_attention")
        self.sliding_index = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else self.full_index
        )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        hidden = self.embed_tokens(inputs) * self.embedding_multiplier
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(hidden, cache[self.full_index])
        sliding_mask = create_attention_mask(
            hidden,
            cache[self.sliding_index],
            window_size=self.window_size,
        )
        for layer, layer_cache in zip(self.layers, cache):
            mask = full_mask if layer.layer_type == "full_attention" else sliding_mask
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    """GraniteMoE model with full and sliding attention cache lanes."""

    supports_speculative_rollback = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoeSwaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        output = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            output = self.model.embed_tokens.as_linear(output)
        else:
            output = self.lm_head(output)
        return output / self.logits_scaling

    def sanitize(self, weights):
        if any("experts.gate_proj" in key for key in weights):
            return weights
        sanitized = {}
        for key, value in weights.items():
            if key.endswith("block_sparse_moe.experts.gate_up_proj"):
                intermediate = value.shape[1]
                sanitized[key.replace("gate_up_proj", "gate_proj") + ".weight"] = value[
                    :, : intermediate // 2, :
                ]
                sanitized[key.replace("gate_up_proj", "up_proj") + ".weight"] = value[
                    :, intermediate // 2 :, :
                ]
            elif key.endswith("block_sparse_moe.experts.down_proj"):
                sanitized[key + ".weight"] = value
            elif key == "lm_head.weight" and self.args.tie_word_embeddings:
                continue
            else:
                sanitized[key] = value
        return sanitized

    def make_cache(self, max_kv_size=None):
        return [
            (RotatingKVCache(max_size=max_kv_size) if max_kv_size else KVCache())
            if layer_type == "full_attention"
            else RotatingKVCache(max_size=self.args.sliding_window)
            for layer_type in self.model.layer_types
        ]

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
