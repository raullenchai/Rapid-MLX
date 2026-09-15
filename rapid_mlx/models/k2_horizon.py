# SPDX-License-Identifier: Apache-2.0
# Copyright © 2023-2024 Apple Inc.
"""Rapid-owned dense text runtime for ``model_type: k2_horizon``.

K2 Horizon is close to the standard Llama decoder geometry, but its residual
stream is normalized in independent groups. Treating that as ordinary global
RMSNorm produces plausible-looking but incorrect logits, so the family needs a
small explicit forward graph rather than a config-only alias.

The runtime deliberately uses mlx-lm's low-level attention, RoPE, and KV-cache
contracts. It therefore runs on Rapid's standard text scheduler (batching,
prefix cache, streaming, cancellation) and never imports mlx-vlm or introduces
a family-specific generation loop. Only the released dense topology is
accepted; MoE, MoVA, QK-norm, and sliding variants fail closed.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Released K2 Horizon 7B configuration and supported dense invariants."""

    model_type: str = "k2_horizon"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 250624
    head_dim: int | None = 128
    max_position_embeddings: int = 524288
    num_key_value_heads: int | None = 8
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10_000_000.0
    rope_traditional: bool = False
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    layernorm_num_groups: int = 4
    num_experts: int = 0
    num_shared_experts: int = 0
    moe_intermediate_size: int = 0
    mova_num_experts: int = 0
    query_key_norm: bool = False
    attention_gate_func: str | None = None
    rope_head_dim: int | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]):
        config = dict(params or {})
        rope = config.pop("rope_parameters", None)
        if rope is not None:
            if not isinstance(rope, dict):
                raise ValueError("rope_parameters must be an object")
            rope = dict(rope)
            rope_type = rope.pop("rope_type", None)
            legacy_rope_type = rope.pop("type", None)
            if (
                rope_type is not None
                and legacy_rope_type is not None
                and rope_type != legacy_rope_type
            ):
                raise ValueError("rope_parameters declares conflicting rope types")
            rope_type = rope_type or legacy_rope_type or "default"
            config["rope_theta"] = rope.pop(
                "rope_theta", config.get("rope_theta", cls.rope_theta)
            )
            if rope_type == "default":
                if rope:
                    raise ValueError(
                        "default rope_parameters contains scaling metadata: "
                        f"{', '.join(sorted(rope))}"
                    )
                config["rope_scaling"] = None
            else:
                rope["rope_type"] = rope_type
                config["rope_scaling"] = rope
        return super().from_dict(config)

    def __post_init__(self):
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "vocab_size",
            "layernorm_num_groups",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} ({getattr(self, name)}) must be positive")
        if self.hidden_size % self.layernorm_num_groups:
            raise ValueError("hidden_size must be divisible by layernorm_num_groups")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads < 1:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads"
            )
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError(
                    "hidden_size must be divisible by num_attention_heads "
                    "when head_dim is omitted"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim < 1:
            raise ValueError("head_dim must be positive")
        if self.rope_head_dim is None:
            self.rope_head_dim = self.head_dim
        if self.rope_head_dim != self.head_dim:
            raise ValueError("partial rotary head dimensions are not supported")
        if self.hidden_act != "silu":
            raise ValueError(
                f"unsupported hidden_act {self.hidden_act!r}; expected 'silu'"
            )
        if (
            self.num_experts
            or self.num_shared_experts
            or self.moe_intermediate_size
            or self.mova_num_experts
        ):
            raise ValueError("K2 Horizon MoE/MoVA checkpoints are not supported")
        if self.query_key_norm:
            raise ValueError("K2 Horizon query/key normalization is not supported")
        if self.attention_gate_func is not None:
            raise ValueError("K2 Horizon gated attention is not supported")
        if self.use_sliding_window or self.sliding_window is not None:
            raise ValueError("K2 Horizon sliding attention is not supported")


class GroupRMSNorm(nn.Module):
    """RMS-normalize each contiguous hidden group independently."""

    def __init__(self, dims: int, groups: int, eps: float):
        super().__init__()
        if groups < 1 or dims % groups:
            raise ValueError(f"hidden size {dims} must be divisible by groups {groups}")
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        grouped = x.reshape((*shape[:-1], self.groups, -1))
        normalized = mx.fast.rms_norm(grouped, weight=None, eps=self.eps)
        return self.weight * normalized.reshape(shape)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.num_key_value_heads is not None
        assert args.head_dim is not None
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
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
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        queries = (
            self.q_proj(x)
            .reshape(batch, length, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = GroupRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )
        self.post_attention_layernorm = GroupRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        hidden = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class K2HorizonModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = GroupRMSNorm(
            args.hidden_size, args.layernorm_num_groups, args.rms_norm_eps
        )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = (
            self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        )
        if cache is None:
            cache = [None] * len(self.layers)
        elif len(cache) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} K2 cache entries, got {len(cache)}"
            )
        mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = K2HorizonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        output = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(output)
        return self.lm_head(output)

    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in key
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
