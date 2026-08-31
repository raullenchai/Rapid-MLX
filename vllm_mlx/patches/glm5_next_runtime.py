# SPDX-License-Identifier: Apache-2.0
"""Correctness overlay for the released GLM-5 Next text runtime.

mlx-vlm 0.6.17 contains the initial GLM-5 Next implementation, but its text
stack still inherits unclamped DeepSeek MLPs, bf16 router math, and two norm
epsilons from the parent architecture. Those are not interchangeable with the
GLM checkpoint contract. This module installs the architecture-owned variants
before model construction while leaving shared DeepSeek classes untouched.

The overlay also keeps the six-input KDA projection fusion lossless: a uniform
quantization uses the released fused path, while mixed quantization falls back
to the six original projections instead of interpreting five tensors with the
first tensor's quantization parameters.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn

_LOCK = threading.Lock()
_INSTALLED = False


def _projection_quantization_is_homogeneous(modules: Sequence[Any]) -> bool:
    quantized = [hasattr(module, "scales") for module in modules]
    if not all(quantized) and any(quantized):
        return False
    if not any(quantized):
        return True
    return (
        len({module.group_size for module in modules}) == 1
        and len({module.bits for module in modules}) == 1
        and len({getattr(module, "mode", "affine") for module in modules}) == 1
    )


def _keep_glm5_next_fp32(key: str) -> bool:
    return (
        ".attn_hc." in key
        or ".ffn_hc." in key
        or key.endswith("A_log")
        or key.endswith("dt_bias")
    )


def install_glm5_next_runtime_fix() -> bool:
    """Install the GLM-specific text math once, before model construction."""
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return False

        from mlx_vlm.models.deepseek_v32.language import (
            DeepseekV32MoE,
            MoEGate,
            group_expert_select,
        )
        from mlx_vlm.models.glm5_next import language
        from mlx_vlm.models.mlp import DeepseekMLP

        if getattr(language, "_RAPID_MLX_RUNTIME_FIX_INSTALLED", False):
            _INSTALLED = True
            return False

        class Glm5NextClampedSwiGLU(nn.Module):
            def __init__(self, limit: float | None):
                super().__init__()
                self.limit = limit

            def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
                if self.limit is not None:
                    x_gate = mx.clip(x_gate, a_min=None, a_max=self.limit)
                    x_up = mx.clip(x_up, a_min=-self.limit, a_max=self.limit)
                return nn.silu(x_gate) * x_up

        class Glm5NextMLP(DeepseekMLP):
            def __init__(self, config, hidden_size=None, intermediate_size=None):
                super().__init__(
                    config,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                self.limit = config.swiglu_limit

            def __call__(self, x: mx.array) -> mx.array:
                gate = self.gate_proj(x)
                up = self.up_proj(x)
                if self.limit is not None:
                    gate = mx.clip(gate, a_min=None, a_max=self.limit)
                    up = mx.clip(up, a_min=-self.limit, a_max=self.limit)
                return self.down_proj(nn.silu(gate) * up)

        class Glm5NextMoEGate(MoEGate):
            def __call__(self, x: mx.array):
                logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
                return group_expert_select(
                    logits,
                    self.e_score_correction_bias,
                    self.top_k,
                    self.n_group,
                    self.topk_group,
                    self.routed_scaling_factor,
                    self.norm_topk_prob,
                )

        class Glm5NextMoE(DeepseekV32MoE):
            def __init__(self, config):
                super().__init__(config)
                self.switch_mlp.activation = Glm5NextClampedSwiGLU(config.swiglu_limit)
                self.gate = Glm5NextMoEGate(config)
                if config.n_shared_experts is not None:
                    width = config.moe_intermediate_size * config.n_shared_experts
                    self.shared_experts = Glm5NextMLP(config, intermediate_size=width)

        released_sparse_attention = language.Glm5NextSparseAttention

        class Glm5NextSparseAttention(released_sparse_attention):
            def __init__(self, config):
                super().__init__(config)
                self.q_a_layernorm = nn.RMSNorm(
                    self.q_lora_rank, eps=config.rms_norm_eps
                )
                self.kv_a_layernorm = nn.RMSNorm(
                    self.kv_lora_rank, eps=config.rms_norm_eps
                )
                self.indexer.k_norm = nn.LayerNorm(self.indexer.head_dim, eps=1e-6)

        released_linear_attention = language.Glm5NextLinearAttention

        class Glm5NextLinearAttention(released_linear_attention):
            def _fused_in_proj(self, inputs):
                modules = (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.forget_gate.f_a_proj,
                    self.g_a_proj,
                    self.b_proj,
                )
                if not _projection_quantization_is_homogeneous(modules):
                    return tuple(module(inputs) for module in modules)
                return super()._fused_in_proj(inputs)

        released_sanitize = language.LanguageModel.sanitize

        def patched_sanitize(self, weights):
            sanitized = released_sanitize(self, weights)
            for key, value in list(sanitized.items()):
                if (
                    _keep_glm5_next_fp32(key)
                    and mx.issubdtype(value.dtype, mx.floating)
                    and value.dtype != mx.float32
                ):
                    sanitized[key] = value.astype(mx.float32)
            return sanitized

        @property
        def cast_predicate(self):
            def predicate(key):
                if "e_score_correction_bias" in key:
                    return False
                return not _keep_glm5_next_fp32(key)

            return predicate

        # These names are looked up by Glm5NextDecoderLayer at construction
        # time. Replacing only the GLM module globals avoids changing any
        # DeepSeek model that imports the shared implementations directly.
        language.DeepseekMLP = Glm5NextMLP
        language.DeepseekV32MoE = Glm5NextMoE
        language.Glm5NextSparseAttention = Glm5NextSparseAttention
        language.Glm5NextLinearAttention = Glm5NextLinearAttention
        language.LanguageModel.sanitize = patched_sanitize
        language.LanguageModel.cast_predicate = cast_predicate
        language._RAPID_MLX_RUNTIME_FIX_INSTALLED = True
        _INSTALLED = True
        return True


def is_installed() -> bool:
    return _INSTALLED


__all__ = [
    "install_glm5_next_runtime_fix",
    "is_installed",
]
