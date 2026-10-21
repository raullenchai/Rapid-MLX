# SPDX-License-Identifier: MIT
"""Flint backbone adapted from upstream mlx-lm PR #1998 by pcuenca.

Source: https://github.com/ml-explore/mlx-lm/pull/1998
Head: 810efca78b69a737c44ce73b87e8adbc12d10529
Semantics checked against Recursive's reference implementation (Apache-2.0).

CC-VENDOR-DELTA: absolute imports and hardware compatibility initialization;
pure MLX fp32 routing with guarded normalization; no distributed sharding;
single-kind masks and cache validation; packed storage and mixed-bit inference.

Sync policy: always register this adapter for flint, including when mlx-lm
ships a native module. PR #1998 does not support the official INT4 checkpoint
layout: U8 packed storage and an 8-bit embedding under a uniform 4-bit config.
Retain the adapter until native loading is explicitly qualified against that
layout; an mlx-lm upgrade alone must not change the loader.

MIT License

Copyright © 2026 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat

_mlx_compat.install()

from mlx_lm.models.activations import swiglu  # noqa: E402
from mlx_lm.models.base import (  # noqa: E402
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    mlp_only_layers: list[int]
    layer_types: list[str]
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0
    rope_parameters: dict[str, Any] | None = None
    tie_word_embeddings: bool = False

    _load_config: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, params):
        # CC-VENDOR-DELTA: retain the exact dict consulted after sanitize.
        args = super().from_dict(params)
        args._load_config = params
        return args

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers or not self.layer_types:
            raise ValueError("Flint layer_types must describe every layer")
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("Flint layer_types contains an unsupported attention kind")
        if self.sliding_window <= 0:
            raise ValueError("Flint sliding_window must be positive")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "Flint num_experts_per_tok must be within the expert count"
            )
        if self.rope_parameters and self.rope_parameters.get("rope_theta") is not None:
            self.rope_theta = self.rope_parameters["rope_theta"]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((n_heads,))

        self.rope: nn.RoPE | None
        if args.layer_types[layer_idx] == "sliding_attention":
            self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)
        else:
            self.rope = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.rope is not None:
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
            sinks=self.softmax_offset,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return cast(mx.array, self.o_proj(output))


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return cast(
            mx.array, self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        # CC-VENDOR-DELTA: packed routers must use the quantized module;
        # unquantized routers retain fp32 accumulation on both operands.
        if isinstance(self.gate, nn.QuantizedLinear):
            logits = self.gate(x).astype(mx.float32)
        else:
            logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        inds = mx.stop_gradient(
            mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)

        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_expert(x)

        return cast(mx.array, y)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.use_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.self_attn = Attention(args, layer_idx)
        self.mlp: MLP | SparseMoeBlock
        if layer_idx in args.mlp_only_layers:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = next(
            (i for i, k in enumerate(args.layer_types) if k == "full_attention"), None
        )
        self.swa_idx = next(
            (i for i, k in enumerate(args.layer_types) if k == "sliding_attention"),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if len(cache) != len(self.layers):
            raise ValueError("Flint cache length must match the layer count")
        fa_mask = (
            create_attention_mask(h, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        swa_mask = (
            create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )
            if self.swa_idx is not None
            else None
        )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, c)

        return cast(mx.array, self.norm(h))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return cast(mx.array, self.model.embed_tokens.as_linear(out))
        return cast(mx.array, self.lm_head(out))

    def sanitize(self, weights: dict) -> dict:
        if self.args.tie_word_embeddings:
            for key in list(weights):
                if key.startswith("lm_head."):
                    weights.pop(key)
        for index in range(self.args.num_hidden_layers):
            if index in self.args.mlp_only_layers:
                continue
            prefix = f"model.layers.{index}.mlp"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
        # CC-VENDOR-DELTA: infer packed widths before the loader quantizes.
        quantization = self.args._load_config.get("quantization")
        modules = dict(self.named_modules())
        for key, value in list(weights.items()):
            if not key.endswith(".weight") or key[:-7] + ".scales" not in weights:
                continue
            path = key[:-7]
            if quantization is None:
                raise ValueError(
                    f"Flint packed module {path} requires quantization config"
                )
            if value.dtype == mx.uint8:
                value = value.view(mx.uint32)
                weights[key] = value
            if value.dtype != mx.uint32:
                raise ValueError(f"Flint packed weight {key} must be uint8 or uint32")
            settings = dict(quantization)
            settings.update(quantization.get(path, {}))
            group_size = settings["group_size"]
            scales = weights[path + ".scales"]
            width = scales.shape[-1] * group_size
            numerator = value.shape[-1] * 32
            module = modules.get(path)
            if module is None or not hasattr(module, "to_quantized"):
                raise ValueError(f"Flint packed module {path} cannot be quantized")
            if width != module.weight.shape[-1] or width <= 0:
                raise ValueError(f"Flint quantized input width mismatch for {path}")
            bits, remainder = divmod(numerator, width)
            if (
                remainder
                or bits not in (2, 3, 4, 5, 6, 8)
                or settings.get("mode", "affine") != "affine"
            ):
                raise ValueError(f"Flint cannot infer supported affine bits for {path}")
            if value.shape[:-1] != scales.shape[:-1]:
                raise ValueError(f"Flint packed/scales shape mismatch for {path}")
            quantization[path] = {"bits": bits, "group_size": group_size}
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
