# SPDX-License-Identifier: Apache-2.0
"""Vendored MLX text model for the Qwen4-Exp architecture.

The implementation is intentionally architecture-driven: every optional
component is admitted from the checkpoint's typed ``text_config`` fields.
There are no repository-name aliases or compatibility approximations here.

M1 implements the target text decoder.  The checkpoint's MTP and vision
modules are deliberately ignored by :meth:`Model.sanitize` until their own
milestones have independent numerical and lifecycle coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.base import (  # noqa: E402
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402
from mlx_lm.models.gated_delta import gated_delta_update  # noqa: E402
from mlx_lm.models.rope_utils import initialize_rope  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from .qwen4_exp_cache import QSAIndexCache  # noqa: E402


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0

    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_parameters: dict[str, Any] | None = field(
        default_factory=lambda: {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"

    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True

    hc_count: int = 4
    hc_lowrank: int = 320

    layer_types: list[str] | None = None
    full_attention_interval: int = 4
    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None

    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    seed: int = 1234
    eos_token_id: int | list[int] | None = None
    mtp_num_hidden_layers: int = 0

    def __post_init__(self):
        rope = dict(self.rope_parameters or {})
        self.partial_rotary_factor = float(
            rope.get("partial_rotary_factor", self.partial_rotary_factor)
        )
        self.rope_theta = float(rope.get("rope_theta", self.rope_theta))
        self.ple_embed_dim = (
            self.hidden_size if self.ple_embed_dim is None else self.ple_embed_dim
        )
        self.ple_layer_ids = sorted(set(self.ple_layer_ids))
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (index + 1) % self.full_attention_interval
                else "qwen_sparse_attention"
                for index in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "qwen_sparse_attention" if kind == "full_attention" else kind
                for kind in self.layer_types
            ]
        self._validate()

    def _validate(self) -> None:
        layer_types = cast(list[str], self.layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError(
                "Qwen4-Exp layer_types must have one entry per decoder layer"
            )
        unsupported = set(layer_types) - {
            "linear_attention",
            "qwen_sparse_attention",
        }
        if unsupported:
            raise ValueError(
                f"Unsupported Qwen4-Exp layer types: {sorted(unsupported)}"
            )
        if self.hc_count <= 1 or self.hidden_size <= 0 or self.hc_lowrank <= 0:
            raise ValueError("Qwen4-Exp requires positive four-stream HC dimensions")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.output_gate_type != "sigmoid":
            raise ValueError(
                "Qwen4-Exp M1 supports the checkpoint-declared sigmoid GDN gate only"
            )
        qsa = (
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        )
        if any(value is None for value in qsa):
            raise ValueError("Qwen4-Exp QSA requires the complete indexer contract")
        qsa_values = cast(tuple[int, int, int, int, int], qsa)
        if any(value <= 0 for value in qsa_values):
            raise ValueError("Qwen4-Exp indexer values must be positive")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires one indexer KV head")
        if qsa_values[3] % qsa_values[4]:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > qsa_values[2]:
            raise ValueError("attention rotary dimensions must fit the QSA index head")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be within num_experts")
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        ple_embed_dim = cast(int, self.ple_embed_dim)
        if self.ple_layer_ids and ple_embed_dim % ngram_heads:
            raise ValueError(
                "PLE embedding width must divide evenly across n-gram heads"
            )
        if any(
            layer < 1 or layer > self.num_hidden_layers for layer in self.ple_layer_ids
        ):
            raise ValueError("ple_layer_ids are one-indexed decoder layer ids")
        if any(
            layer_types[layer - 1] != "linear_attention" for layer in self.ple_layer_ids
        ):
            raise ValueError("PLE is only valid on linear-attention layers")
        if self.ple_layer_ids and self.eos_token_id is None:
            raise ValueError("PLE requires eos_token_id for segment-local n-grams")
        if isinstance(self.eos_token_id, list) and not self.eos_token_id:
            raise ValueError("PLE eos_token_id list must not be empty")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict[str, Any]

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose checkpoint weight represents an additive delta from one."""

    def __init__(self, dim: int, *, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        if group_size is not None and dim % group_size:
            raise ValueError("grouped RMSNorm width must divide the feature width")
        self.weight = mx.zeros((dim,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
            weight = self.weight.reshape(-1, self.group_size)
        else:
            weight = self.weight
        dtype = x.dtype
        normalized = x.astype(mx.float32)
        normalized = normalized * mx.rsqrt(
            mx.mean(mx.square(normalized), axis=-1, keepdims=True) + self.eps
        )
        normalized = normalized * (1.0 + weight.astype(mx.float32))
        return normalized.astype(dtype).reshape(original_shape)


class SigmoidRMSNormGated(nn.Module):
    """Qwen4-Exp's norm-before-gate GDN output transform."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        activation: str = "sigmoid",
    ):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps
        self.activation = activation

    def __call__(self, hidden: mx.array, gate: mx.array) -> mx.array:
        dtype = hidden.dtype
        # Preserve the architecture reference's BF16 rounding boundary: the
        # fused RMSNorm returns activation dtype before the gate multiplies in
        # FP32. Adapted from mlx-vlm ecf1aa0a62958ea770bc25c35e173effe142aa3c
        # (MIT).
        normalized = mx.fast.rms_norm(hidden, self.weight, self.eps).astype(mx.float32)
        gate = gate.astype(mx.float32)
        gate = mx.sigmoid(gate) if self.activation == "sigmoid" else nn.silu(gate)
        return (normalized * gate).astype(dtype)


class GatedResidual(nn.Module):
    """Exact four-stream read mixer and per-branch write gate."""

    def __init__(self, args: TextModelArgs, *, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_width = self.hc_count * self.hidden_size
        self.hc_norm = ZeroCenteredRMSNorm(
            hc_width, group_size=self.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_width, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_width, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_width, self.hc_count, bias=False) if use_combine else None
        )

    def __call__(self, hyper_input: mx.array):
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"Qwen4-Exp HC expected {expected} features, got {hyper_input.shape[-1]}"
            )
        normalized = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        streams = normalized.reshape(
            *normalized.shape[:-1], self.hc_count, self.hidden_size
        )
        mixed = mx.mean(mix * streams, axis=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * mx.sigmoid(self.block_inject_weight(normalized) / self.hc_count)
        return mixed, hyper_input, injection


class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Softmax top-k routed experts plus the separately gated shared expert."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)
        self.sharding_group = None

    def __call__(self, x: mx.array, *, target_verify: bool = False) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if target_verify and x.ndim == 3 and x.shape[1] > 1:
            batch, steps, width = x.shape
            experts_per_token = indices.shape[-1]
            flat_x = x.reshape(batch * steps, width)
            flat_indices = indices.reshape(batch * steps, experts_per_token)
            flat_x = mx.expand_dims(flat_x, (-2, -3))
            gate_up_proj = getattr(self.switch_mlp, "gate_up_proj", None)
            if gate_up_proj is not None:
                gate_up = gate_up_proj(flat_x, flat_indices, sorted_indices=False)
                gate, up = mx.split(gate_up, 2, axis=-1)
            else:
                up = self.switch_mlp.up_proj(flat_x, flat_indices, sorted_indices=False)
                gate = self.switch_mlp.gate_proj(
                    flat_x, flat_indices, sorted_indices=False
                )
            routed = self.switch_mlp.down_proj(
                self.switch_mlp.activation(up, gate),
                flat_indices,
                sorted_indices=False,
            )
            routed = routed.squeeze(-2).reshape(batch, steps, experts_per_token, -1)
        else:
            routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return routed + shared


class GatedDeltaNet(nn.Module):
    """Qwen4-Exp GDN with 48 value heads and a sigmoid output gate."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.log(
            mx.random.uniform(low=0.01, high=16.0, shape=(self.num_v_heads,))
        )
        self.norm = SigmoidRMSNormGated(
            self.head_v_dim,
            eps=args.rms_norm_eps,
            activation=args.output_gate_type or args.hidden_act,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        *,
        record_rollback: bool = False,
    ) -> mx.array:
        batch, length, _ = inputs.shape
        mixed = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, length, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(inputs)
        alpha = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        if mask is not None:
            mixed = mx.where(mask[..., None], mixed, 0)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])
            if record_rollback:
                cache.record_slot_snapshots(
                    0,
                    [
                        mx.contiguous(conv_input[:, position : position + keep, :])
                        for position in range(1, length)
                    ],
                )
        convolved = nn.silu(self.conv1d(conv_input))
        query, key, value = [
            item.reshape(batch, length, heads, dim)
            for item, heads, dim in zip(
                mx.split(convolved, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache is not None else None
        # Exact Qwen4-Exp normalization: L2 epsilon is applied after the sum,
        # then queries receive the attention scale. Adapted from mlx-vlm
        # ecf1aa0a62958ea770bc25c35e173effe142aa3c (MIT); the previous
        # RMS-normalized form moved epsilon inside the mean and diverged on the
        # real q4 checkpoint.
        inv_scale = key.shape[-1] ** -0.5
        query = query * mx.rsqrt(
            mx.sum(mx.square(query), axis=-1, keepdims=True) + 1e-6
        )
        key = key * mx.rsqrt(mx.sum(mx.square(key), axis=-1, keepdims=True) + 1e-6)
        query = query * inv_scale
        if record_rollback and cache is not None and length > 1:
            from vllm_mlx.kernels.qwen4_gdn_verify import (
                gated_delta_verify_with_states,
            )

            output, state, state_snapshots = gated_delta_verify_with_states(
                query,
                key,
                value,
                alpha,
                beta,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )
            cache.record_slot_snapshots(
                1,
                [state_snapshots[:, position] for position in range(length - 1)],
                finalize=True,
            )
        else:
            output, state = gated_delta_update(
                query,
                key,
                value,
                alpha,
                beta,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )
        if cache is not None:
            cache[1] = state
            cache.advance(length)
        output = self.norm(output, z)
        return self.out_proj(output.reshape(batch, length, -1))


def apply_rotary_positions(
    x: mx.array,
    positions: mx.array,
    *,
    rotary_dim: int,
    base: float,
) -> mx.array:
    """Apply non-interleaved partial RoPE at explicit logical positions.

    ``x`` has shape ``[batch, tokens, heads, dim]`` and ``positions`` is
    ``[tokens]`` or ``[batch, tokens]``. Explicit positions are required by
    QSA because compressed keys rotate at each group's first token.
    """

    if rotary_dim == 0:
        return x
    if rotary_dim % 2:
        raise ValueError("Qwen4-Exp rotary dimensions must be even")
    if positions.ndim == 1:
        positions = positions[None, :]
    inverse_frequency = 1.0 / (
        base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / float(rotary_dim))
    )
    angles = positions.astype(mx.float32)[..., None] * inverse_frequency
    angles = mx.concatenate([angles, angles], axis=-1)[:, :, None, :]
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    rotary = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = mx.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
    rotated = rotary * cosine + rotated_half * sine
    return mx.concatenate([rotated.astype(x.dtype), x[..., rotary_dim:]], axis=-1)


@cache
def _qwen4_exp_rope_kernel(rotary_dim: int, position_ndim: int):
    """Build the architecture's half-split MRoPE forward kernel.

    Forward math is adopted line-for-line from mlx-vlm commit
    ecf1aa0a62958ea770bc25c35e173effe142aa3c (MIT).  Rapid only needs the
    inference forward here; the pure-MLX fallback below covers non-Metal use.
    """

    if not mx.metal.is_available():
        return None
    if position_ndim == 2:
        position_expr = "position_ids[b * q_len + t]"
    else:
        raise ValueError("Qwen4-Exp text RoPE requires 2-D position IDs")
    source = f"""
        uint elem = thread_position_in_grid.x;

        const int half_dim = {rotary_dim // 2};
        const int q_bsz = x_shape[0];
        const int q_heads = x_shape[1];
        const int q_len = x_shape[2];
        const int q_dim = x_shape[3];
        const int slots = half_dim + q_dim - {rotary_dim};
        const int work_size = q_bsz * q_heads * q_len * slots;

        if (elem >= uint(work_size)) {{
            return;
        }}

        int local = int(elem);
        int slot = local % slots;
        int tmp = local / slots;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % q_heads;
        int b = tmp / q_heads;
        int base = ((b * q_heads + h) * q_len + t) * q_dim;

        if (slot >= half_dim) {{
            int pass_d = {rotary_dim} + slot - half_dim;
            int pass_idx = base + pass_d;
            x_out[pass_idx] = x[pass_idx];
            return;
        }}

        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);

        int idx = base + d;
        float xv = static_cast<float>(x[idx]);
        float xp = static_cast<float>(x[base + pair_d]);
        x_out[idx] = static_cast<T>(xv * c - xp * s);
        x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
    """
    return mx.fast.metal_kernel(
        name=f"qwen4_exp_rope_half_split_{rotary_dim}_{position_ndim}d",
        input_names=["x", "position_ids", "inv_freq"],
        output_names=["x_out"],
        source=source,
    )


def apply_qwen4_exp_rope(
    x: mx.array,
    positions: mx.array,
    *,
    rotary_dim: int,
    base: float,
) -> mx.array:
    """Apply exact Qwen4-Exp text RoPE to ``[B, H, L, D]`` states."""

    if positions.ndim == 1:
        positions = mx.broadcast_to(positions[None], (x.shape[0], positions.size))
    inverse_frequency = 1.0 / (
        base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / float(rotary_dim))
    )
    kernel = _qwen4_exp_rope_kernel(rotary_dim, positions.ndim)
    if kernel is None:
        return apply_rotary_positions(
            x.transpose(0, 2, 1, 3),
            positions,
            rotary_dim=rotary_dim,
            base=base,
        ).transpose(0, 2, 1, 3)
    half_dim = inverse_frequency.shape[0]
    slots = half_dim + x.shape[-1] - rotary_dim
    work_size = x.shape[0] * x.shape[1] * x.shape[2] * slots
    (output,) = kernel(
        inputs=[x, positions, inverse_frequency],
        template=[("T", x.dtype)],
        grid=(work_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return output


@dataclass(frozen=True)
class _QSASelection:
    """Compact per-query physical KV indices produced by the QSA indexer."""

    token_indices: mx.array
    valid: mx.array
    physical_kv_length: int

    def dense_mask(self) -> mx.array:
        """Materialize the reference mask for unsupported sparse consumers."""
        batch, length, _ = self.token_indices.shape
        # Invalid compact slots share a sentinel column so they cannot clear a
        # real token when put_along_axis sees duplicate indices.
        sentinel = self.physical_kv_length
        indices = mx.where(self.valid, self.token_indices, sentinel)
        selected = mx.zeros(
            (batch, length, self.physical_kv_length + 1), dtype=mx.bool_
        )
        selected = mx.put_along_axis(selected, indices, self.valid, axis=-1)
        return selected[:, None, :, : self.physical_kv_length]


class QSAIndexer(nn.Module):
    """Weight-bearing QSA selector backed by raw-ring/compressed-key state."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_heads = cast(int, args.indexer_n_heads)
        self.num_kv_heads = cast(int, args.indexer_kv_heads)
        self.head_dim = cast(int, args.indexer_head_dim)
        self.token_budget = cast(int, args.indexer_budget)
        self.compress_ratio = cast(int, args.indexer_compress_ratio)
        self.block_topk = self.token_budget // self.compress_ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (self.num_heads + self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: QSAIndexCache,
        *,
        physical_kv_length: int,
    ) -> mx.array | None:
        batch, length, _ = hidden_states.shape
        cache._ensure_batch(batch)
        offsets = list(cache._offsets)
        valid_spans = cache.valid_spans(length)
        projected = self.index_qk_proj(hidden_states)
        query_width = self.num_heads * self.head_dim
        query, raw_keys = mx.split(projected, [query_width], axis=-1)
        query = query.reshape(batch, length, self.num_heads, self.head_dim)
        raw_keys = raw_keys.reshape(batch, length, self.num_kv_heads, self.head_dim)
        if self.num_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires one indexer KV head")
        raw_keys = raw_keys.squeeze(2)
        starts = mx.array([start for start, _ in valid_spans], dtype=mx.int64)
        positions = (
            mx.array(offsets, dtype=mx.int64)[:, None]
            + mx.arange(length, dtype=mx.int64)[None, :]
            - starts[:, None]
        )
        query = self.q_layernorm(query)
        query = apply_qwen4_exp_rope(
            query.transpose(0, 2, 1, 3),
            positions,
            rotary_dim=self.rotary_dim,
            base=self.rope_theta,
        ).transpose(0, 2, 1, 3)

        def transform_group(group: mx.array, start: int) -> mx.array:
            normalized = self.k_layernorm(group[:, None, :])[:, 0, :]
            return apply_qwen4_exp_rope(
                normalized[:, None, None, :],
                mx.array([[start]], dtype=mx.int64),
                rotary_dim=self.rotary_dim,
                base=self.rope_theta,
            )[:, 0, 0, :]

        cache.update(raw_keys, transform_group)
        # The architecture reference stays on ordinary causal attention while
        # every complete block fits the QSA budget. Preserve that exact math
        # and kernel selection while still updating Rapid's persistent index
        # cache for the first later token that crosses the sparse boundary.
        # Adapted from mlx-vlm ecf1aa0a62958ea770bc25c35e173effe142aa3c
        # (MIT), without its O(B*L*topk*K) token-mask materialization.
        if physical_kv_length // self.compress_ratio <= self.block_topk:
            return None
        left_padding = (
            [0] * batch
            if cache.left_padding is None
            else [int(value) for value in cache.left_padding.tolist()]
        )
        compact_indices = []
        compact_valid = []
        for batch_index in range(batch):
            input_start, valid_length = valid_spans[batch_index]
            available_blocks = cache._compressed_counts[batch_index]
            selected_blocks = None
            if available_blocks > self.block_topk:
                keys = cache.keys_for_blocks(batch_index, available_blocks)
                # Preserve the reference's single batched matmul and FP32
                # reduction. Per-token matmuls choose a different Metal
                # accumulation kernel and can perturb the top-k boundary.
                scores = mx.matmul(
                    query[batch_index].transpose(1, 0, 2),
                    keys.T,
                )
                scores = mx.sum(
                    mx.maximum(scores.astype(mx.float32), 0), axis=0
                ) / math.sqrt(self.head_dim)
                query_ends = offsets[batch_index] + mx.arange(length) - input_start + 1
                complete_counts = mx.maximum(query_ends // self.compress_ratio, 0)
                valid_blocks = (
                    mx.arange(available_blocks)[None, :] < complete_counts[:, None]
                )
                scores = mx.where(valid_blocks, scores, -mx.inf)
                selected_blocks = mx.argpartition(
                    scores, kth=-self.block_topk, axis=-1
                )[..., -self.block_topk :].astype(mx.int32)
            query_indices = mx.arange(length, dtype=mx.int32)
            logical_positions = offsets[batch_index] + query_indices - input_start
            complete_counts = mx.maximum(
                (logical_positions + 1) // self.compress_ratio, 0
            )
            if selected_blocks is None:
                max_complete = (
                    offsets[batch_index] + valid_length
                ) // self.compress_ratio
                if max_complete > self.block_topk:
                    raise RuntimeError("QSA sparse selection was not materialized")
                selected_blocks = mx.broadcast_to(
                    mx.arange(self.block_topk, dtype=mx.int32)[None, :],
                    (length, self.block_topk),
                )

            dense_blocks = mx.broadcast_to(
                mx.arange(self.block_topk, dtype=mx.int32)[None, :],
                (length, self.block_topk),
            )
            blocks = mx.where(
                complete_counts[:, None] > self.block_topk,
                selected_blocks,
                dense_blocks,
            )
            block_valid = mx.arange(self.block_topk)[None, :] < mx.minimum(
                complete_counts[:, None], self.block_topk
            )
            block_tokens = (
                blocks[:, :, None] * self.compress_ratio
                + mx.arange(self.compress_ratio, dtype=mx.int32)[None, None, :]
            ).reshape(length, self.token_budget)
            block_token_valid = mx.broadcast_to(
                block_valid[:, :, None],
                (length, self.block_topk, self.compress_ratio),
            ).reshape(length, self.token_budget)

            tail_start = complete_counts * self.compress_ratio
            tail_counts = logical_positions + 1 - tail_start
            tail_offsets = mx.arange(self.compress_ratio, dtype=mx.int32)[None, :]
            tail_tokens = tail_start[:, None] + tail_offsets
            tail_valid = tail_offsets < tail_counts[:, None]

            query_valid = (query_indices >= input_start) & (
                query_indices < input_start + valid_length
            )
            indices = mx.concatenate([block_tokens, tail_tokens], axis=-1)
            valid = mx.concatenate([block_token_valid, tail_valid], axis=-1)
            valid = valid & query_valid[:, None]
            indices = mx.clip(
                indices + left_padding[batch_index], 0, physical_kv_length - 1
            )
            compact_indices.append(indices)
            compact_valid.append(valid)

        selection = _QSASelection(
            token_indices=mx.stack(compact_indices),
            valid=mx.stack(compact_valid),
            physical_kv_length=physical_kv_length,
        )
        return selection.dense_mask()


class QSAAttention(nn.Module):
    """Qwen sparse attention with independent main-KV and index side caches."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * args.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = ZeroCenteredRMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.rotary_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )
        self.indexer = QSAIndexer(args)

    def __call__(
        self,
        x: mx.array,
        cache: Any | None = None,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        kv_cache = None if cache is None else cache[0]
        index_cache = None if cache is None else cache[1]
        if mask is None:
            # The cache offset must be observed before this layer appends the
            # current K/V chunk. Computing the mask after update double-counts
            # chunked prefills (L queries against past+2L keys).
            mask = create_attention_mask(x, kv_cache)
        offset = 0 if kv_cache is None else kv_cache.offset
        physical_length = (
            length
            if kv_cache is None
            else int(getattr(kv_cache, "_idx", kv_cache.size())) + length
        )
        if index_cache is None:
            index_cache = QSAIndexCache(self.indexer.compress_ratio)
        selected = self.indexer(
            x,
            index_cache,
            physical_kv_length=physical_length,
        )

        projected = self.q_proj(x).reshape(
            batch, length, self.num_attention_heads, self.head_dim * 2
        )
        queries, gate = mx.split(projected, 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        keys = self.k_proj(x).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        values = self.v_proj(x).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        values = values.transpose(0, 2, 1, 3)
        positions = mx.arange(length, dtype=mx.int64)
        if isinstance(offset, mx.array) and offset.ndim:
            positions = offset[:, None] + positions[None, :]
        else:
            positions = positions + int(offset)
        # Qwen4-Exp uses half-split (rotate-half) pairing for its partial RoPE.
        # This follows mlx-vlm ecf1aa0a62958ea770bc25c35e173effe142aa3c
        # (MIT) while retaining Rapid's persistent QSA cache contract.
        queries = apply_qwen4_exp_rope(
            queries.transpose(0, 2, 1, 3),
            positions,
            rotary_dim=self.rotary_dim,
            base=self.indexer.rope_theta,
        )
        keys = apply_qwen4_exp_rope(
            keys.transpose(0, 2, 1, 3),
            positions,
            rotary_dim=self.rotary_dim,
            base=self.indexer.rope_theta,
        )
        if kv_cache is not None:
            keys, values = kv_cache.update_and_fetch(keys, values)
        additive_mask = (
            mask
            if selected is None
            else mx.where(
                selected,
                mx.array(0.0, dtype=queries.dtype),
                mx.array(-1e9, dtype=queries.dtype),
            )
        )
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=kv_cache,
            scale=self.scale,
            mask=additive_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> list[int]:
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + 10007 * ple_layer_index
    return [
        2 * (_splitmix64(base_seed + 0x9E3779B97F4A7C15 * (index + 1)) % half_bound) + 1
        for index in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def find_nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not _is_prime(value):
            value += 1
    return value


class ShardedEmbedding(nn.Module):
    """Exact row-wise embedding split matching the checkpoint's 128 shards.

    Keeping shards as independent ``nn.Embedding`` leaves conversion and
    serving bounded: quantization never concatenates the 51B-parameter PLE
    table into one temporary tensor.
    """

    def __init__(self, num_embeddings: int, dims: int, parts: int):
        super().__init__()
        if num_embeddings % parts:
            raise ValueError("sharded embedding rows must divide evenly")
        self.rows_per_shard = num_embeddings // parts
        self.shards = [nn.Embedding(self.rows_per_shard, dims) for _ in range(parts)]

    def __call__(self, indices: mx.array) -> mx.array:
        shard_ids = indices // self.rows_per_shard
        # Evaluating at most 128 small integer ids selects only the embedding
        # shards needed by this request; it never materializes embedding rows.
        used = sorted({int(value) for value in shard_ids.reshape(-1).tolist()})
        output = None
        for shard_id in used:
            shard_id = int(shard_id)
            local = indices - shard_id * self.rows_per_shard
            mask = shard_ids == shard_id
            gathered = self.shards[shard_id](mx.clip(local, 0, self.rows_per_shard - 1))
            gathered = mx.where(mask[..., None], gathered, 0)
            output = gathered if output is None else output + gathered
        if output is None:
            return mx.zeros((*indices.shape, self.shards[0].weight.shape[-1]))
        return output


class NGramEmbedding(nn.Module):
    """Segment-aware hashed bigram/trigram embedding used by PLE."""

    def __init__(
        self,
        args: TextModelArgs,
        *,
        embedding_dim: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_ids = tuple(
            args.eos_token_id
            if isinstance(args.eos_token_id, list)
            else [cast(int, args.eos_token_id)]
        )
        self.eos_token_id = self.eos_token_ids[0]
        sizes = [
            find_nth_prime_after(
                args.ngram_vocab_size_base - 1,
                ple_layer_index * self.ngram_heads + head + 1,
            )
            for head in range(self.ngram_heads)
        ]
        offsets: list[int] = []
        offset = 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        divisor = args.make_ngram_vocab_size_divisible_by
        padded_rows = ((offset + divisor - 1) // divisor) * divisor
        self.layer_multipliers = mx.array(
            build_layer_multipliers(
                args.vocab_size,
                args.ngram_size,
                ple_layer_index,
                args.seed,
            ),
            dtype=mx.int64,
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = ShardedEmbedding(
            padded_rows,
            embedding_dim // self.ngram_heads,
            args.split_ngram_parts,
        )

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return token_ids
        _, length = token_ids.shape
        positions = mx.arange(length, dtype=mx.int64)
        is_eos = token_ids == self.eos_token_ids[0]
        for eos_token_id in self.eos_token_ids[1:]:
            is_eos = is_eos | (token_ids == eos_token_id)
        eos_positions = mx.where(is_eos, positions, -1)
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [
                mx.full((token_ids.shape[0], 1), -1, dtype=mx.int64),
                previous_eos_inclusive[:, :-1],
            ],
            axis=1,
        )
        position_in_segment = positions[None, :] - previous_eos - 1
        source_positions = positions - shift
        gather_positions = mx.maximum(source_positions, 0)[None, :]
        gather_positions = mx.broadcast_to(gather_positions, token_ids.shape)
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
        return mx.where(valid, shifted, self.eos_token_id)

    def compute_ids(
        self,
        input_ids: mx.array,
        cache: Any | None = None,
        *,
        record_rollback: bool = False,
    ) -> mx.array:
        input_ids = input_ids.astype(mx.int64)
        if cache is not None and cache[3] is not None:
            previous = cache[3]
        else:
            previous = mx.full(
                (input_ids.shape[0], self.context_len),
                self.eos_token_id,
                dtype=mx.int64,
            )
        history = mx.concatenate([previous, input_ids], axis=1)
        if cache is not None:
            if cache.lengths is not None:
                valid = mx.clip(cache.lengths, 0, input_ids.shape[1])
                positions = (valid[:, None] + mx.arange(self.context_len))[..., None]
                cache[3] = mx.take_along_axis(
                    history[..., None], positions, axis=1
                ).squeeze(-1)
            else:
                cache[3] = mx.contiguous(history[:, -self.context_len :])
            if record_rollback:
                cache.record_slot_snapshots(
                    3,
                    [
                        mx.contiguous(
                            history[:, position : position + self.context_len]
                        )
                        for position in range(1, input_ids.shape[1])
                    ],
                )
        shifted = [
            self._shift_right_ignore_eos(history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed,
                    shifted[position] * self.layer_multipliers[position],
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = mx.remainder(mixed[..., None], sizes) + offsets
            blocks.append(ids)
        return mx.concatenate(blocks, axis=-1)[:, -input_ids.shape[1] :]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Any | None = None,
        *,
        record_rollback: bool = False,
    ) -> mx.array:
        ids = self.compute_ids(
            input_ids,
            cache,
            record_rollback=record_rollback,
        )
        return self.ngram_embedding(ids).flatten(-2)


class PLELayer(nn.Module):
    """Exact PLE projection, gating, and dilation-three short convolution."""

    def __init__(
        self,
        args: TextModelArgs,
        *,
        ple_layer_index: int,
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_width = args.hc_count * args.hidden_size
        embed_dim = cast(int, args.ple_embed_dim)
        self.ple_embedding = NGramEmbedding(
            args,
            embedding_dim=embed_dim,
            ple_layer_index=ple_layer_index,
        )
        self.key_proj = nn.Linear(embed_dim, hc_width, bias=False)
        self.value_proj = nn.Linear(embed_dim, args.hidden_size, bias=False)
        self.norm_key = ZeroCenteredRMSNorm(
            hc_width, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_query = ZeroCenteredRMSNorm(
            hc_width, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.norm_conv = ZeroCenteredRMSNorm(
            hc_width, group_size=args.hidden_size, eps=args.rms_norm_eps
        )
        self.conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.conv1d = nn.Conv1d(
            hc_width,
            hc_width,
            kernel_size=args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=hc_width,
            bias=False,
        )

    def _short_conv(
        self,
        x: mx.array,
        cache: Any | None,
        *,
        record_rollback: bool = False,
    ) -> mx.array:
        if cache is not None and cache[2] is not None:
            state = cache[2]
        else:
            state = mx.zeros(
                (x.shape[0], self.conv_state_len, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            if cache.lengths is not None:
                valid = mx.clip(cache.lengths, 0, x.shape[1])
                positions = (valid[:, None] + mx.arange(self.conv_state_len))[..., None]
                cache[2] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[2] = mx.contiguous(conv_input[:, -self.conv_state_len :, :])
            if record_rollback:
                cache.record_slot_snapshots(
                    2,
                    [
                        mx.contiguous(
                            conv_input[:, position : position + self.conv_state_len, :]
                        )
                        for position in range(1, x.shape[1])
                    ],
                )
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: Any | None,
        mask: mx.array | None = None,
        *,
        record_rollback: bool = False,
    ) -> mx.array:
        if mask is not None:
            input_ids = mx.where(mask, input_ids, self.ple_embedding.eos_token_id)
        embeddings = self.ple_embedding(
            input_ids,
            cache,
            record_rollback=record_rollback,
        )
        keys = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        values = self.value_proj(embeddings)
        queries = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * values[..., None, :]
        gated = gated.flatten(-2)
        normalized = self.norm_conv(gated)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)
            normalized = mx.where(mask[..., None], normalized, 0)
        return gated + self._short_conv(
            normalized,
            cache,
            record_rollback=record_rollback,
        )


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_index: int):
        super().__init__()
        self.layer_type = cast(list[str], args.layer_types)[layer_index]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = QSAAttention(args)
        self.mlp = SparseMoeBlock(args)
        ple_index = (
            args.ple_layer_ids.index(layer_index + 1)
            if layer_index + 1 in args.ple_layer_ids
            else None
        )
        self.ple = (
            PLELayer(args, ple_layer_index=ple_index) if ple_index is not None else None
        )
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    @staticmethod
    def _combine(
        output: mx.array,
        residual: mx.array,
        injection: mx.array,
    ) -> mx.array:
        injected = output[..., None, :] * injection[..., :, None]
        return residual + injected.flatten(-2)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        input_ids: mx.array,
        mask: mx.array | None,
        cache: Any | None,
        record_rollback: bool = False,
    ) -> mx.array:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                cache,
                mask,
                record_rollback=record_rollback,
            )
        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        if self.is_linear:
            output = self.linear_attn(
                mixed,
                mask=mask,
                cache=cache,
                record_rollback=record_rollback,
            )
        else:
            output = self.self_attn(mixed, cache=cache, mask=mask)
        hidden_states = self._combine(output, residual, injection)

        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        output = self.mlp(mixed, target_verify=record_rollback)
        return self._combine(output, residual, injection)


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_index)
            for layer_index in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        *,
        return_hidden: bool = False,
        record_rollback: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
        if cache is None:
            cache = [None] * len(self.layers)
        linear_index = next(
            (index for index, layer in enumerate(self.layers) if layer.is_linear),
            None,
        )
        linear_mask = (
            None
            if linear_index is None
            else create_ssm_mask(hidden_states, cache[linear_index])
        )
        attention_index = next(
            (index for index, layer in enumerate(self.layers) if not layer.is_linear),
            None,
        )
        attention_cache = (
            None
            if attention_index is None or cache[attention_index] is None
            else cache[attention_index][0]
        )
        attention_mask = create_attention_mask(hidden_states, attention_cache)
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                input_ids=inputs,
                mask=linear_mask if layer.is_linear else attention_mask,
                cache=layer_cache,
                record_rollback=record_rollback,
            )
        output = self.hyper_connection_mixer(hidden_states)
        return (output, hidden_states) if return_hidden else output


class Qwen4ExpStateCache(ArraysCache):
    """Recurrent Qwen4 state with speculative-verify restore points.

    The four-slot PLE layer cache couples GDN convolution/state with PLE
    convolution/ngram history.  A rejected speculative token must restore all
    four slots to the same accepted boundary; restoring GDN alone silently
    desynchronizes later PLE inputs.
    """

    rollback_state: list[list[mx.array | None]] | None = None
    _rollback_slots: dict[int, list[mx.array]] | None = None

    def record_slot_snapshots(
        self,
        slot: int,
        snapshots: list[mx.array],
        *,
        finalize: bool = False,
    ) -> None:
        """Stage per-position recurrent state and publish atomic boundaries."""
        if not snapshots:
            return
        if self._rollback_slots is None:
            self._rollback_slots = {}
        self._rollback_slots[slot] = snapshots
        if not finalize:
            return
        expected_slots = set(range(len(self.cache)))
        if set(self._rollback_slots) != expected_slots:
            raise AssertionError(
                "Qwen4 speculative cache snapshots do not cover every state slot"
            )
        lengths = {len(items) for items in self._rollback_slots.values()}
        if len(lengths) != 1:
            raise AssertionError("Qwen4 speculative cache snapshot lengths diverged")
        count = lengths.pop()
        self.rollback_state = [
            [self._rollback_slots[slot][position] for slot in range(len(self.cache))]
            for position in range(count)
        ]
        self._rollback_slots = None

    def restore_rollback(self, n_to_drop: int, verify_size: int) -> None:
        snapshots = self.rollback_state
        if not snapshots:
            raise AssertionError("Qwen4 verify rollback has no saved boundary")
        keep = verify_size - n_to_drop
        if keep < 1 or keep > len(snapshots):
            raise AssertionError(
                f"invalid Qwen4 rollback boundary: keep={keep}, "
                f"snapshots={len(snapshots)}"
            )
        self.cache = list(snapshots[keep - 1])
        self.rollback_state = None
        self._rollback_slots = None


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array | tuple[mx.array, mx.array]:
        hidden_result = self.model(
            inputs,
            cache,
            input_embeddings,
            return_hidden=return_hidden,
            record_rollback=n_confirmed > 0 and inputs.shape[1] > 1,
        )
        if return_hidden:
            hidden, mtp_hidden = cast(tuple[mx.array, mx.array], hidden_result)
        else:
            hidden = cast(mx.array, hidden_result)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        return (logits, mtp_hidden) if return_hidden else logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(
                    Qwen4ExpStateCache(size=4 if layer.ple is not None else 2)
                )
            else:
                caches.append(
                    CacheList(
                        KVCache(),
                        QSAIndexCache(layer.self_attn.indexer.compress_ratio),
                    )
                )
        return caches

    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("mtp.") and ".visual." not in key
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        sanitized = {}
        for key, value in weights.items():
            if (
                key.endswith("conv1d.weight")
                and value.ndim == 3
                and value.shape[1] == 1
            ):
                value = value.moveaxis(2, 1)
            if ".mlp.experts.gate_up_proj" in key:
                gate_up = value
                midpoint = gate_up.shape[-2] // 2
                prefix, suffix = key.split("experts.gate_up_proj", 1)
                leaf = {"": "weight", ".scales": "scales", ".biases": "biases"}.get(
                    suffix
                )
                if leaf is None:
                    sanitized[key] = value
                    continue
                base = f"{prefix}switch_mlp"
                sanitized[f"{base}.gate_proj.{leaf}"] = gate_up[..., :midpoint, :]
                sanitized[f"{base}.up_proj.{leaf}"] = gate_up[..., midpoint:, :]
                continue
            if ".mlp.experts.down_proj" in key:
                prefix, suffix = key.split("experts.down_proj", 1)
                leaf = {"": "weight", ".scales": "scales", ".biases": "biases"}.get(
                    suffix
                )
                if leaf is not None:
                    key = f"{prefix}switch_mlp.down_proj.{leaf}"
            if ".ngram_embedding.shard_" in key:
                prefix, suffix = key.split(".ngram_embedding.shard_", 1)
                shard, leaf = suffix.split(".", 1)
                key = f"{prefix}.ngram_embedding.shards.{int(shard)}.{leaf}"
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if ".ple.ple_embedding.ngram_embedding.shards." in path:
                # The checkpoint's PLE tables have width 160.  Group 32 is the
                # largest established MLX group that divides that width, so it
                # preserves the source shape without a padding/slicing format.
                return {"group_size": 32, "bits": 4}
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith("A_log")


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        return self.language_model(
            inputs,
            cache,
            input_embeddings,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
        )

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        mapped = {}
        for key, value in weights.items():
            if key.startswith("model.visual") or key.startswith("vision_tower"):
                continue
            if key.startswith("mtp."):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model", 1)
            elif not key.startswith("language_model."):
                key = f"language_model.{key}"
            mapped[key] = value
        return self.language_model.sanitize(mapped)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
