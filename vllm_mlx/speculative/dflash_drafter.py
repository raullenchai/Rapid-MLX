# SPDX-License-Identifier: Apache-2.0
"""
DFlash block-diffusion drafter (MLX port of z-lab/dflash).

Reference: https://github.com/z-lab/dflash + dflash.py shipped with the
HF model (DFlashDraftModel). The drafter is a small bidirectional
transformer (8 layers, hidden_size=2048) that takes:

  * `noise_embedding`: BLOCK_SIZE mask-token embeddings drawn from the
    TARGET model's `embed_tokens`. These are the "queries" / draft slots.
  * `target_hidden`: a stack of TARGET hidden states from layers
    `target_layer_ids` (default [1, 10, 19, 28, 37]) at the latest
    decoded position(s). They are concatenated along the feature axis
    (giving 5 * hidden_size = 10240 features), projected back to
    `hidden_size` by `fc`, then RMS-normed by `hidden_norm`. The
    resulting vectors are used as the "context" k/v inside every
    drafter attention layer.

Drafter attention is `is_causal=False`. K/V are formed by concatenating
projections of the (already-conditioned) target context with the noise
itself, so every noise position can see every other noise position as
well as every target context position. Position embeddings are applied
to the full (context + noise) sequence with the **same** rotary
embedding the standard Qwen3 dense uses (head_dim=128).

The drafter does NOT have its own `lm_head` — caller projects the
drafter's final hidden states through the TARGET's `lm_head` (or
`embed_tokens.as_linear` when tied) to obtain logits.

This MLX port is intentionally minimal: BF16 only, no KV cache
(speculative drafting always re-runs the drafter from scratch over a
small block, ~16 positions), no sliding-window optimisation. Block
size, mask token id, and target_layer_ids are read from the model's
`config.json`.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DFlashConfig:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 8
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    # YARN scaling — drafter uses YARN; matches HF Qwen3RotaryEmbedding.
    rope_scaling_type: str = "yarn"
    rope_scaling_factor: float = 64.0
    rope_scaling_original_max_pos: int = 4096
    rope_scaling_beta_fast: float = 32.0
    rope_scaling_beta_slow: float = 1.0
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: list[int] = field(default_factory=lambda: [1, 10, 19, 28, 37])
    vocab_size: int = 248320

    @classmethod
    def from_dict(cls, raw: dict) -> "DFlashConfig":
        dflash_cfg = raw.get("dflash_config", {})
        rope_scaling = raw.get("rope_scaling") or {}
        return cls(
            hidden_size=raw.get("hidden_size", 2048),
            intermediate_size=raw.get("intermediate_size", 6144),
            num_hidden_layers=raw.get("num_hidden_layers", 8),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get("num_key_value_heads", 4),
            head_dim=raw.get("head_dim", 128),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=float(raw.get("rope_theta", 10000000.0)),
            rope_scaling_type=str(rope_scaling.get("rope_type", rope_scaling.get("type", "default"))),
            rope_scaling_factor=float(rope_scaling.get("factor", 1.0)),
            rope_scaling_original_max_pos=int(
                rope_scaling.get("original_max_position_embeddings", 4096)
            ),
            rope_scaling_beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
            rope_scaling_beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
            block_size=int(raw.get("block_size", 16)),
            mask_token_id=int(dflash_cfg.get("mask_token_id", 248070)),
            target_layer_ids=list(dflash_cfg.get("target_layer_ids", [1, 10, 19, 28, 37])),
            vocab_size=raw.get("vocab_size", 248320),
        )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_qk(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    """Match PyTorch dflash.apply_rotary_pos_emb semantics.

    cos / sin shape (B, T, D) where T = ctx_len + q_len.
    q has q_len = noise_len, k has full T length.
    The PyTorch impl slices cos/sin's LAST T positions for q (so q
    aligns with the **tail** of the position sequence), and uses the
    full cos/sin for k.
    """
    cos = mx.expand_dims(cos, 1)  # (B, 1, T, D)
    sin = mx.expand_dims(sin, 1)
    q_len = q.shape[-2]
    cos_q = cos[..., -q_len:, :]
    sin_q = sin[..., -q_len:, :]
    q_out = (q * cos_q) + (_rotate_half(q) * sin_q)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


def _yarn_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    original_max_pos: int,
    beta_fast: float,
    beta_slow: float,
) -> tuple[mx.array, float]:
    """Compute YARN-scaled inv_freq matching transformers' yarn_rope_init_fn.

    Returns (inv_freq, attention_scaling). attention_scaling is the
    extra cos/sin magnification YARN applies (typically slightly > 1).
    """
    import math

    pos_freqs = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    # find_correction_range
    def find_correction_dim(num_rotations: float) -> float:
        return (
            head_dim
            * math.log(original_max_pos / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = math.floor(find_correction_dim(beta_fast))
    high = math.ceil(find_correction_dim(beta_slow))
    low = max(low, 0)
    high = min(high, head_dim - 1)
    if low == high:
        high += 1
    # linear ramp [0..1] of length head_dim/2
    arange = mx.arange(head_dim // 2, dtype=mx.float32)
    linear_func = (arange - low) / (high - low)
    ramp = mx.clip(linear_func, 0.0, 1.0)
    inv_freq_mask = 1.0 - ramp
    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    # YARN attention scaling
    attention_scaling = 0.1 * math.log(factor) + 1.0
    return inv_freq, attention_scaling


class DFlashRotary(nn.Module):
    """Cosine/sine table generator matching dflash's Qwen3RotaryEmbedding."""

    def __init__(self, cfg: "DFlashConfig"):
        super().__init__()
        self.head_dim = cfg.head_dim
        if cfg.rope_scaling_type == "yarn":
            inv_freq, attention_scaling = _yarn_inv_freq(
                cfg.head_dim,
                cfg.rope_theta,
                cfg.rope_scaling_factor,
                cfg.rope_scaling_original_max_pos,
                cfg.rope_scaling_beta_fast,
                cfg.rope_scaling_beta_slow,
            )
        else:
            inv_freq = 1.0 / (
                cfg.rope_theta
                ** (mx.arange(0, cfg.head_dim, 2, dtype=mx.float32) / cfg.head_dim)
            )
            attention_scaling = 1.0
        self._inv_freq = inv_freq
        self._scaling = attention_scaling

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (B, T)
        # (B, T, head_dim/2)
        freqs = mx.expand_dims(position_ids.astype(mx.float32), -1) * mx.expand_dims(
            mx.expand_dims(self._inv_freq, 0), 0
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self._scaling
        sin = mx.sin(emb) * self._scaling
        return cos, sin


class DFlashAttention(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.reshape(B, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)  # (B, H, q_len, D)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = mx.concatenate([k_ctx, k_noise], axis=1).reshape(
            B, ctx_len + q_len, self.num_kv_heads, self.head_dim
        )
        v = mx.concatenate([v_ctx, v_noise], axis=1).reshape(
            B, ctx_len + q_len, self.num_kv_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q, k = _apply_rotary_qk(q, k, cos, sin)

        # Bidirectional: no causal mask. Use scaled_dot_product_attention.
        # GQA: repeat KV heads to match query heads.
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = mx.repeat(k, repeat, axis=1)
            v = mx.repeat(v, repeat, axis=1)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        out = out.transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(cfg)
        self.mlp = DFlashMLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    """Block-diffusion drafter producing BLOCK_SIZE candidate hidden states.

    No `lm_head` here; caller projects through the target's lm_head/
    embed_tokens.as_linear. No KV cache because each drafting call
    operates on a fresh block of mask tokens.
    """

    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = [DFlashDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(
            len(cfg.target_layer_ids) * cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = DFlashRotary(cfg)

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden_concat: mx.array,
        position_ids: mx.array,
    ) -> mx.array:
        """Run the drafter forward.

        Args:
            noise_embedding: (B, BLOCK, hidden_size) — embeddings of
                mask tokens, drawn from target.embed_tokens.
            target_hidden_concat: (B, ctx_len, len(layer_ids)*hidden_size)
                — concatenated target hidden states.
            position_ids: (B, ctx_len + BLOCK) absolute target positions
                for the context tokens followed by the next BLOCK
                positions for the drafted slots.

        Returns:
            (B, BLOCK, hidden_size) — final hidden states of the drafted
            positions (post-norm). Caller projects to logits.
        """
        target_hidden = self.hidden_norm(self.fc(target_hidden_concat))
        cos, sin = self.rotary(position_ids)
        h = noise_embedding
        for layer in self.layers:
            h = layer(h, target_hidden, cos, sin)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_dflash_drafter(model_path: str | Path) -> tuple[DFlashDraftModel, DFlashConfig]:
    """Load DFlash drafter weights from the HF directory.

    Expects:
      <model_path>/config.json
      <model_path>/model.safetensors
    """
    model_path = Path(model_path)
    cfg_raw = json.loads((model_path / "config.json").read_text())
    cfg = DFlashConfig.from_dict(cfg_raw)
    logger.info(
        "[dFlash] loading drafter: layers=%d hidden=%d kv_heads=%d block_size=%d "
        "target_layers=%s mask_id=%d",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_key_value_heads,
        cfg.block_size,
        cfg.target_layer_ids,
        cfg.mask_token_id,
    )

    model = DFlashDraftModel(cfg)
    raw = mx.load(str(model_path / "model.safetensors"))
    # PyTorch keys map 1-to-1 onto the MLX module names declared above.
    # Layer-attention norm weights load directly because dflash uses
    # standard RMSNorm (gain only) and we declared nn.RMSNorm.
    weights = list(raw.items())
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    logger.info("[dFlash] loaded %d weight tensors", len(weights))
    return model, cfg
