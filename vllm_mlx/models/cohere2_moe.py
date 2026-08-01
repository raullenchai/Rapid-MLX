# SPDX-License-Identifier: Apache-2.0
"""Vendored Cohere2 MoE decoder for North-Mini-Code checkpoints.

This is the minimal architecture portion of ml-explore/mlx-lm#1487.  Rapid
registers it under ``mlx_lm.models.cohere2_moe`` so the pinned mlx-lm loader
can use its normal config, cache, and weight-loading path.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 2048
    head_dim: int = 128
    num_hidden_layers: int = 49
    intermediate_size: int = 768
    prefix_dense_intermediate_size: int = 3072
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    vocab_size: int = 262144
    rope_theta: float = 50000.0
    layer_norm_eps: float = 1e-5
    logit_scale: float = 1.0
    attention_bias: bool = False
    sliding_window: int = 4096
    max_position_embeddings: int = 500000
    tie_word_embeddings: bool = True
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 0
    norm_topk_prob: bool = False
    first_k_dense_replace: int = 1
    expert_selection_fn: str = "sigmoid"
    layer_types: list[str] | None = None
    # Accepted checkpoint keys that do not alter this architecture.
    use_parallel_block: bool = True
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Cohere2MoeSparseBlock(nn.Module):
    """Sigmoid/softmax top-k MoE with the checkpoint's native key layout."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.expert_selection_fn == "sigmoid"
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.intermediate_size, args.num_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        logits = self.gate(x).astype(mx.float32)
        scores = mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)
        indices = mx.stop_gradient(
            mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        output = self.switch_mlp(x, indices)
        return mx.sum(output * weights.astype(dtype)[..., None], axis=-2)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.rope = (
            nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)
            if self.is_sliding
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MLP(args.hidden_size, args.prefix_dense_intermediate_size)
            if layer_idx < args.first_k_dense_replace or args.num_experts == 0
            else Cohere2MoeSparseBlock(args)
        )
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=False
        )
        self.attention_type = args.layer_types[layer_idx]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        normalized = self.input_layernorm(x)
        return x + self.self_attn(normalized, mask, cache) + self.mlp(normalized)


class Cohere2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, index) for index in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps, bias=False)
        self.full_attention_index = args.layer_types.index("full_attention")
        self.sliding_attention_index = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        hidden = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(hidden, cache[self.full_attention_index])
        sliding_mask = (
            create_attention_mask(
                hidden,
                cache[self.sliding_attention_index],
                window_size=self.args.sliding_window,
            )
            if self.sliding_attention_index is not None
            else None
        )
        for layer, layer_cache in zip(self.layers, cache):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Cohere2MoeModel(args)
        self.logit_scale = args.logit_scale
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        hidden = self.model(inputs, cache)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return logits * self.logit_scale

    def make_cache(self):
        return [
            KVCache()
            if layer_type == "full_attention" or not self.args.sliding_window
            else RotatingKVCache(max_size=self.args.sliding_window)
            for layer_type in self.args.layer_types
        ]

    def sanitize(self, weights):
        if any(key.startswith("language_model.") for key in weights):
            weights = {
                key.removeprefix("language_model."): value
                for key, value in weights.items()
            }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            key: value
            for key, value in weights.items()
            if "rotary_emb.inv_freq" not in key
        }

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return {"group_size": 64, "bits": 8} if path.endswith("mlp.gate") else True

        return predicate

    @property
    def layers(self):
        return self.model.layers
