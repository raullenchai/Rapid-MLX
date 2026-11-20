# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 Flash text backbone with portable MLA, IndexPool, Engram and iHC."""

import logging
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .. import _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import CacheList, KVCache  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from . import hy_v4_flash_ops as ops  # noqa: E402
from .hy_v4_flash_config import ModelArgs as ModelArgs  # noqa: E402
from .hy_v4_flash_config import (  # noqa: E402
    checkpoint_key,
    engram_hash,
    quantization_policy,
    validate_lookup,
)

logger = logging.getLogger(__name__)


def _update_aux_cache(cache, keys):
    # Keep ordinary KVCache/BatchKVCache bookkeeping (including merge, filter,
    # trim and extraction), but never retain a second, unread activation graph.
    # Independent zero-width arrays carry sequence length without payload.
    values = mx.zeros((*keys.shape[:-1], 0), dtype=keys.dtype)
    return cache.update_and_fetch(keys, values)[0]


class HCPre(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self.hc_fn = nn.Linear(
            args.hc_mult * args.hidden_size, 2 * args.hc_mult, bias=False
        )
        self.hc_scale = mx.full((2,), 0.01)
        self.hc_base = mx.concatenate(
            (
                mx.full((args.hc_mult,), -float(math.log(args.hc_mult - 1))),
                mx.zeros((args.hc_mult,)),
            )
        )

    def __call__(self, x):
        a = self._args
        return ops.hc_pre(
            x,
            self.hc_fn.weight,
            self.hc_scale,
            self.hc_base,
            a.hc_eps,
            a.rms_norm_eps,
            a.hc_magnitude,
            mx,
        )


class HCLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hc_pre = HCPre(args)


class HCHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self.hc_head_fn = nn.Linear(
            args.hc_mult * args.hidden_size, args.hc_mult, bias=False
        )
        self.hc_head_scale = mx.full((1,), 0.01)
        self.hc_head_base = mx.full((args.hc_mult,), -float(math.log(args.hc_mult - 1)))

    def __call__(self, x):
        a = self._args
        return ops.hc_head(
            x,
            self.hc_head_fn.weight,
            self.hc_head_scale,
            self.hc_head_base,
            a.hc_eps,
            a.rms_norm_eps,
            mx,
        )


class MLP(nn.Module):
    def __init__(self, hidden, intermediate, eps=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.post_fc2_norm = nn.RMSNorm(hidden, eps=eps) if eps is not None else None

    def __call__(self, x):
        gate = self.gate_proj(x)
        out = self.down_proj(gate * mx.sigmoid(gate) * self.up_proj(x))
        return self.post_fc2_norm(out) if self.post_fc2_norm is not None else out


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate):
        if self.limit > 0:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return gate * mx.sigmoid(gate) * up


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.expert_bias = mx.zeros((args.n_routed_experts,))
        self.routed_expert_down_proj = nn.Linear(
            args.hidden_size, args.routed_expert_hidden_size, bias=False
        )
        self.routed_expert_up_proj = nn.Linear(
            args.routed_expert_hidden_size, args.hidden_size, bias=False
        )
        self.switch_mlp = SwitchGLU(
            args.routed_expert_hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )
        self.shared_experts = (
            MLP(args.hidden_size, args.moe_intermediate_size * args.n_shared_experts)
            if args.n_shared_experts
            else None
        )
        self.combine_output_norm = (
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            if args.sandwich_norm
            else None
        )

    def __call__(self, x):
        a = self._args
        indices, weights = ops.route(
            self.gate(x.astype(mx.float32)),
            self.expert_bias,
            a.num_experts_per_tok,
            a.norm_topk_prob,
            a.routed_scaling_factor,
            mx,
        )
        routed = self.switch_mlp(self.routed_expert_down_proj(x), indices)
        routed = (
            (routed.astype(mx.float32) * weights[..., None])
            .sum(axis=-2)
            .astype(x.dtype)
        )
        out = self.routed_expert_up_proj(routed)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return (
            self.combine_output_norm(out)
            if self.combine_output_norm is not None
            else out
        )


class Indexer(nn.Module):
    def __init__(self, args, theta):
        super().__init__()
        self._args, self._theta = args, theta
        d = args.compressed_indexer_head_dim
        self.query_projection = nn.Linear(
            args.q_lora_rank,
            args.compressed_indexer_num_attention_heads * d,
            bias=False,
        )
        self.head_weight_projection = nn.Linear(
            args.hidden_size, args.compressed_indexer_num_attention_heads, bias=False
        )
        self.token_key_projection = nn.Linear(args.hidden_size, d, bias=False)
        self.token_key_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.key_value_projection = nn.Linear(d, d, bias=False)
        self.key_gate_projection = nn.Linear(d, d, bias=False)
        self.key_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.intra_group_ape = mx.zeros((args.compressed_indexer_compress_ratio, d))

    def __call__(self, x, qc, query_positions, key_positions, cache):
        a = self._args
        b, length, _ = x.shape
        d = a.compressed_indexer_head_dim
        heads = a.compressed_indexer_num_attention_heads
        raw = self.token_key_norm(self.token_key_projection(x)).astype(mx.bfloat16)
        raw = _update_aux_cache(cache, raw[:, None])
        raw = raw[:, 0]
        ratio = a.compressed_indexer_compress_ratio
        count = (raw.shape[1] + ratio - 1) // ratio
        group_pos = mx.arange(count) * ratio
        padding = -key_positions[:, :1]
        gather = (
            group_pos[None, :, None]
            + mx.arange(ratio)[None, None, :]
            + padding[:, :, None]
        )
        gather = mx.clip(gather, 0, raw.shape[1] - 1)
        members = mx.take_along_axis(raw[:, None, :, :], gather[..., None], axis=2)
        pooled = ops.gated_pool(
            self.key_value_projection(members.astype(mx.float32)),
            self.key_gate_projection(members.astype(mx.float32)),
            self.intra_group_ape,
            self.key_norm.weight,
            a.rms_norm_eps,
            mx,
        ).astype(mx.bfloat16)
        pooled = ops.fwht_fp8(
            ops.rope(
                pooled,
                mx.broadcast_to(group_pos, (b, count)),
                a.qk_rope_head_dim,
                self._theta,
                mx,
                indexer=True,
            ),
            mx,
        )
        q = self.query_projection(qc).reshape(b, length, heads, d)
        q = ops.fwht_fp8(
            ops.rope(
                q, query_positions, a.qk_rope_head_dim, self._theta, mx, indexer=True
            ).astype(mx.bfloat16),
            mx,
        )
        dot = mx.maximum(mx.einsum("blhd,bgd->blhg", q, pooled), 0)
        weights = (
            self.head_weight_projection(x).astype(mx.float32) * (heads * d) ** -0.5
        )
        scores = (dot * weights[..., None]).sum(axis=-2)
        return ops.sparse_mask(
            query_positions,
            key_positions,
            scores,
            ratio,
            a.compressed_indexer_group_topk,
            mx,
        )


class Attention(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self._args = args
        self.is_window_layer = args.layer_types[layer_idx] == "sliding_attention"
        prefix = "swa_" if self.is_window_layer else ""
        self.num_heads = (
            args.swa_num_attention_heads
            if self.is_window_layer
            else args.num_attention_heads
        )
        self.nope = getattr(args, prefix + "qk_nope_head_dim")
        self.rotary = getattr(args, prefix + "qk_rope_head_dim")
        self.value_dim = getattr(args, prefix + "v_head_dim")
        self._theta = args.rope_parameters.get(args.layer_types[layer_idx], {}).get(
            "rope_theta", 10000 if self.is_window_layer else 10000000
        )
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * (self.nope + self.rotary), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size, args.kv_lora_rank + self.rotary, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            args.kv_lora_rank, self.num_heads * (self.nope + self.value_dim), bias=False
        )
        self.linear_gate = nn.Linear(
            args.hidden_size, self.num_heads * self.value_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.value_dim, args.hidden_size, bias=False
        )
        self.learnable_sink_param = mx.zeros((self.num_heads,))
        self.post_proj_norm = (
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            if args.sandwich_norm
            and (self.is_window_layer or not args.dsa_skip_attn_post_norm)
            else None
        )
        self.indexer = None if self.is_window_layer else Indexer(args, self._theta)

    def __call__(self, x, cache):
        a = self._args
        b, length, _ = x.shape
        offset = mx.array(cache[0].offset).reshape(-1, 1)
        positions = mx.broadcast_to(offset + mx.arange(length), (b, length))
        qc = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qc).reshape(
            b, length, self.num_heads, self.nope + self.rotary
        )
        kv = self.kv_a_proj_with_mqa(x)
        latent = self.kv_a_layernorm(kv[..., : a.kv_lora_rank])
        pe = ops.rope(
            kv[..., a.kv_lora_rank :], positions, self.rotary, self._theta, mx
        )
        q = ops.rope(q, positions, self.rotary, self._theta, mx)
        latent, pe = cache[0].update_and_fetch(latent[:, None], pe[:, None])
        total = latent.shape[2]
        key_positions = mx.broadcast_to(
            offset + length - total + mx.arange(total), (b, total)
        )
        expanded = self.kv_b_proj(latent[:, 0]).reshape(
            b, total, self.num_heads, self.nope + self.value_dim
        )
        k = mx.concatenate(
            (
                expanded[..., : self.nope],
                mx.broadcast_to(
                    pe[:, 0, :, None, :], (b, total, self.num_heads, self.rotary)
                ),
            ),
            axis=-1,
        )
        if self.indexer is not None:
            allowed = self.indexer(x, qc, positions, key_positions, cache[1])
        else:
            allowed = (
                (key_positions[:, None, :] <= positions[..., None])
                & (key_positions[:, None, :] > positions[..., None] - a.sliding_window)
                & (key_positions[:, None, :] >= 0)
            )
            _update_aux_cache(cache[1], mx.zeros((b, 1, length, 0), dtype=x.dtype))
        out = ops.attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            expanded[..., self.nope :].transpose(0, 2, 1, 3),
            allowed,
            self.learnable_sink_param,
            mx,
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1) * mx.sigmoid(
            self.linear_gate(x)
        )
        out = self.o_proj(out)
        return self.post_proj_norm(out) if self.post_proj_norm is not None else out


class MultiHeadEmbedding(nn.Module):
    def __init__(self, rows, width):
        super().__init__()
        self.embedding = nn.Embedding(rows, width)


class Engram(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self._args, self._layer_id = args, layer_id
        self._primes = args.engram_layer_head_vocab_sizes[str(layer_id)]
        sizes = [p for order in self._primes for p in order]
        d = args.engram_n_embed_per_ngram // args.engram_n_head_per_ngram
        width = len(sizes) * d
        self.lookup_table = mx.arange(args.vocab_size, dtype=mx.int32)
        self.multi_head_embedding = MultiHeadEmbedding(sum(sizes), d)
        self.lookup_per_head_norms = [
            nn.RMSNorm(d, eps=1.1920928955078125e-7) for _ in sizes
        ]
        self.key_projs = [
            nn.Linear(width, args.hidden_size) for _ in range(args.hc_mult)
        ]
        self.value_proj = nn.Linear(width, args.hidden_size)
        self.norm_key = [
            nn.RMSNorm(args.hidden_size, eps=1.1920928955078125e-7)
            for _ in range(args.hc_mult)
        ]
        self.norm_query = [
            nn.RMSNorm(args.hidden_size, eps=1.1920928955078125e-7)
            for _ in range(args.hc_mult)
        ]
        self.unified_pre_gamma = mx.ones((args.hidden_size,))
        self.unified_per_hc_scales = [
            mx.zeros((args.hidden_size,)) for _ in range(args.hc_mult)
        ]

    def hashes(self, inputs, history):
        return engram_hash(
            np.array(inputs),
            np.array(history),
            np.array(self.lookup_table),
            self._primes,
            self._layer_id,
            self._args,
        )

    def __call__(self, hidden, inputs, history):
        a = self._args
        hashes = mx.array(self.hashes(inputs, history))
        rows = self.multi_head_embedding.embedding(mx.maximum(hashes, 0)).astype(
            mx.float32
        )
        rows = mx.where(hashes[..., None] < 0, 0, rows)
        x = mx.stack(
            [
                norm(rows[..., i, :])
                for i, norm in enumerate(self.lookup_per_head_norms)
            ],
            axis=-2,
        ).astype(hidden.dtype)
        x = x.reshape(*inputs.shape, -1)
        gates = []
        for i in range(a.hc_mult):
            key = self.norm_key[i](self.key_projs[i](x))
            query = self.norm_query[i](hidden[..., i, :])
            dot = (key * query).sum(axis=-1) * a.hidden_size**-0.5
            gates.append(
                mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
            )
        value = self.value_proj(x).astype(mx.float32)
        signal = ops.rms(value, self.unified_pre_gamma, 1.1920928955078125e-7, mx)
        gate = mx.stack(gates, axis=-1).astype(mx.float32)
        scales = mx.stack(self.unified_per_hc_scales).astype(mx.float32)
        result = ((gate[..., None] + scales) * signal[..., None, :]).astype(
            hidden.dtype
        )
        return hidden + mx.where(
            mx.all(hashes < 0, axis=-1)[..., None, None], 0, result
        )


class DecoderLayer(nn.Module):
    def __init__(self, args, idx):
        super().__init__()
        self.self_attn = Attention(args, idx)
        self.mlp = (
            MLP(
                args.hidden_size,
                args.intermediate_size,
                args.rms_norm_eps if args.sandwich_norm else None,
            )
            if args.mlp_layer_types[idx] == "dense"
            else MoE(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.hc_attn_layer, self.hc_mlp_layer = HCLayer(args), HCLayer(args)
        self.engram_layer = (
            Engram(args, idx + 1) if idx + 1 in args.engram_layer_ids else None
        )

    @staticmethod
    def engram_inputs(inputs, prior, offset):
        # Generation can supply uint32 IDs; padding needs a signed sentinel.
        inputs, prior = inputs.astype(mx.int32), prior.astype(mx.int32)
        # Batch padding is not an original token and resets n-gram history.
        start = mx.array(offset).reshape(-1, 1) - prior.shape[1]
        prior = mx.where(start + mx.arange(prior.shape[1]) < 0, -1, prior)
        positions = mx.array(offset).reshape(-1, 1) + mx.arange(inputs.shape[1])
        return mx.where(positions < 0, -1, inputs), prior

    def __call__(self, hidden, inputs, cache):
        token_keys = (
            inputs[:, None, :, None]
            if self.engram_layer is not None
            else mx.zeros(
                (*inputs.shape[:1], 1, inputs.shape[1], 0), dtype=inputs.dtype
            )
        )
        tokens = _update_aux_cache(cache[2], token_keys)
        if self.engram_layer is not None:
            prior = tokens[:, 0, : -inputs.shape[1], 0]
            hash_inputs, prior = self.engram_inputs(inputs, prior, cache[0].offset)
            hidden = self.engram_layer(hidden, hash_inputs, prior)
        x, post = self.hc_attn_layer.hc_pre(hidden)
        hidden = ops.hc_post(
            self.self_attn(self.input_layernorm(x), cache), hidden, post, mx
        )
        x, post = self.hc_mlp_layer.hc_pre(hidden)
        return ops.hc_post(self.mlp(self.post_attention_layernorm(x)), hidden, post, mx)


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HCHead(args)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache):
        for layer, c in zip(self.layers, cache, strict=True):
            engram = layer.engram_layer
            if engram is not None and hasattr(
                engram.multi_head_embedding.embedding, "prefetch"
            ):
                tokens = c[2]
                prior = (
                    tokens.state[0][:, 0, :, 0]
                    if tokens.keys is not None
                    else inputs[:, :0]
                )
                hash_inputs, prior = layer.engram_inputs(inputs, prior, c[0].offset)
                hashes = engram.hashes(hash_inputs, prior)
                engram.multi_head_embedding.embedding.prefetch(np.maximum(hashes, 0))
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[..., None, :], (*h.shape[:-1], self._args.hc_mult, h.shape[-1])
        )
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, inputs, c)
        return self.norm(self.hc_head(h))


class Model(nn.Module):
    supports_prefix_cache = False

    def __init__(self, args):
        super().__init__()
        self.args, self.model_type = args, args.model_type
        self._loaded_lookup = None
        self.model = Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # Retain latent KV, sparse-layer index keys and Engram-layer token IDs.
        # Unused payloads have zero width, but all slots advance together.
        # Keeping full history makes ordinary prompt-cache trimming exact.
        return [CacheList(KVCache(), KVCache(), KVCache()) for _ in self.layers]

    def __call__(self, inputs, cache=None):
        if cache is None:
            cache = self.make_cache()
        if len(cache) != len(self.layers):
            raise ValueError("Hy4 Flash requires one cache per layer")
        h = self.model(inputs, cache)
        if self.args.enable_lm_head_fp32:
            h = h.astype(mx.float32)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        # Decode evaluates logits, not every cache branch. Materialize auxiliary
        # roots too, especially BatchKVCache offsets and the final Engram token
        # write, so unread lazy updates cannot retain earlier decoding graphs.
        mx.async_eval(logits, [c[i].state for c in cache for i in (1, 2)])
        return logits

    def sanitize(self, weights):
        sanitized = {}
        skipped = False
        for key, value in weights.items():
            target = checkpoint_key(key)
            if target is None:
                skipped = True
                continue
            if target in sanitized:
                raise ValueError(f"Duplicate Hy4 Flash weight: {target}")
            if not self.cast_predicate(target) and mx.issubdtype(
                value.dtype, mx.floating
            ):
                value = value.astype(mx.float32)
            if target.endswith(".engram_layer.lookup_table"):
                lookup = validate_lookup(np.array(value), self.args.vocab_size)
                if self._loaded_lookup is not None and not np.array_equal(
                    self._loaded_lookup, lookup
                ):
                    raise ValueError("Engram layers must share the same token map")
                self._loaded_lookup = lookup
                index = int(target.split(".")[2])
                self.model.layers[index].engram_layer.lookup_table = value
            sanitized[target] = value
        if skipped:
            logger.info(
                "Hy4 Flash text-only: skipping vision tower and dfly_layers weights"
            )
        for key in list(sanitized):
            if ".switch_mlp.gate_up_proj." in key:
                gate, up = mx.split(sanitized.pop(key), 2, axis=-2)
                sanitized[key.replace(".gate_up_proj.", ".gate_proj.")] = gate
                sanitized[key.replace(".gate_up_proj.", ".up_proj.")] = up
        for layer in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer}.mlp"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for leaf in ("weight", "scales", "biases"):
                    first = f"{prefix}.switch_mlp.0.{projection}.{leaf}"
                    if first in sanitized:
                        sanitized[f"{prefix}.switch_mlp.{projection}.{leaf}"] = (
                            mx.stack(
                                [
                                    sanitized.pop(
                                        f"{prefix}.switch_mlp.{e}.{projection}.{leaf}"
                                    )
                                    for e in range(self.args.n_routed_experts)
                                ]
                            )
                        )
        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        return sanitized

    @property
    def quant_predicate(self):
        return quantization_policy

    @property
    def cast_predicate(self):
        return lambda path: (
            not any(
                part in path
                for part in (
                    "hc_",
                    "expert_bias",
                    "mlp.gate.weight",
                    "lm_head",
                    ".indexer.key_",
                    ".indexer.intra_group_ape",
                )
            )
        )
