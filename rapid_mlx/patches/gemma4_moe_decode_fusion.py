# SPDX-License-Identifier: Apache-2.0
"""Opt-in, model-local Gemma 4 expert reduction and gate/up fusion."""

from __future__ import annotations

import logging
import sys
from functools import cache
from typing import Any

from .._env import env_falsey
from ..lattice.fusion import install_enabled
from .moe_decode_fusion import combine

logger = logging.getLogger(__name__)
_MODEL_TYPES = frozenset({"gemma4", "gemma4_text"})


def _expert_types():
    # Only inspect families already loaded by the selected model loader.
    return tuple(
        cls
        for name in (
            "mlx_lm.models.gemma4_text",
            "mlx_vlm.models.gemma4.language",
            "mlx_vlm.models.gemma4_text.language",
            "rapid_mlx.models.gemma4_vendored.language",
        )
        if isinstance(cls := getattr(sys.modules.get(name), "Experts", None), type)
    )


def _eligible(block, x, indices, weights, mx):
    return (
        not getattr(block, "training", False)
        and getattr(block, "sharding_group", None) is None
        and x.ndim == 3
        and x.shape[0] * x.shape[1] == 1
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and weights.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and indices.dtype in (mx.int32, mx.uint32, mx.int64, mx.uint64)
        and weights.ndim == indices.ndim == 3
        and weights.shape == indices.shape
        and weights.shape[:2] == x.shape[:2]
        and 1 <= weights.shape[-1] <= 8
    )


@cache
def _decode_type(base, original):
    def call(self, x, top_k_indices, top_k_weights):
        try:
            import mlx.core as mx

            if _eligible(self, x, top_k_indices, top_k_weights, mx):
                y = self.switch_glu(x, top_k_indices)
                if y.shape == (*top_k_weights.shape, x.shape[-1]) and y.dtype in (
                    mx.float32,
                    mx.float16,
                    mx.bfloat16,
                ):
                    return combine(y, top_k_weights)
        except Exception:  # Fail closed on unsupported runtime/kernel layouts.
            pass
        return original(self, x, top_k_indices, top_k_weights)

    return type(
        f"RapidGemma4Decode{base.__name__}",
        (base,),
        {"__call__": call, "_rapid_gemma4_moe_decode_fusion": True},
    )


def _ffn(layer, x, attended):
    h = x + layer.post_attention_layernorm(attended)
    dense = layer.post_feedforward_layernorm_1(
        layer.mlp(layer.pre_feedforward_layernorm(h))
    )
    indices, weights = layer.router(h)
    routed = layer.experts(layer.pre_feedforward_layernorm_2(h), indices, weights)
    h = h + layer.post_feedforward_layernorm(
        dense + layer.post_feedforward_layernorm_2(routed)
    )
    return h * layer.layer_scalar


@cache
def _layer_type(base, original):
    def call(
        self,
        x,
        mask=None,
        cache=None,
        per_layer_input=None,
        shared_kv=None,
        offset=None,
        shared_cache=None,
    ):
        shared = {"shared_kv": shared_kv, "offset": offset}
        if shared_cache is not None:
            shared["shared_cache"] = shared_cache
        ffn = self.__dict__.get("_rapid_gemma4_compiled_ffn")
        if (
            ffn is None
            or shared_cache is not None
            or per_layer_input is not None
            or x.ndim != 3
            or x.shape[0] * x.shape[1] != 1
            or x.dtype != self.__dict__.get("_rapid_gemma4_dtype")
        ):
            return original(self, x, mask, cache, per_layer_input, **shared)
        attended, shared_kv, offset = self.self_attn(
            self.input_layernorm(x), mask, cache, **shared
        )
        return ffn(x, attended), shared_kv, offset

    return type(
        base.__name__, (base,), {"__call__": call, "_rapid_gemma4_layer_compile": True}
    )


def _layer_types():
    return tuple(
        cls
        for name in (
            "mlx_lm.models.gemma4_text",
            "mlx_vlm.models.gemma4.language",
            "mlx_vlm.models.gemma4_text.language",
            "rapid_mlx.models.gemma4_vendored.language",
        )
        if isinstance(cls := getattr(sys.modules.get(name), "DecoderLayer", None), type)
    )


def _compile_layers(model, mx) -> int:
    """Compile each MoE layer's post-attention half for single-token decode.

    The decoder enrolls as a unit after a bit-exact probe per layer; any
    decline restores every layer's previous class and attributes.
    """
    families = _layer_types()
    layers = [
        m
        for _, m in model.named_modules()
        if isinstance(m, families)
        and getattr(m, "enable_moe", False)
        and getattr(m, "per_layer_input_gate", None) is None
        and getattr(m, "layer_scalar", None) is not None
        and not getattr(type(m), "_rapid_gemma4_layer_compile", False)
    ]
    if not layers:
        return 0
    previous = [(layer, type(layer)) for layer in layers]
    try:
        for layer in layers:
            base = type(layer)
            dtype = layer.pre_feedforward_layernorm.weight.dtype
            if dtype not in (mx.float16, mx.bfloat16):
                raise ValueError("unsupported dtype")
            ffn = mx.compile(lambda x, a, layer=layer: _ffn(layer, x, a))
            object.__setattr__(layer, "_rapid_gemma4_compiled_ffn", ffn)
            object.__setattr__(layer, "_rapid_gemma4_dtype", dtype)
            object.__setattr__(layer, "__class__", _layer_type(base, base.__call__))
            width = layer.pre_feedforward_layernorm.weight.size
            x = mx.sin(mx.arange(width, dtype=mx.float32)).astype(dtype)[None, None]
            a = mx.cos(mx.arange(width, dtype=mx.float32)).astype(dtype)[None, None]
            got, want = ffn(x, a), _ffn(layer, x, a)
            if not bool(mx.array_equal(got, want).item()):
                raise ValueError("compiled layer probe mismatch")
        return len(layers)
    except Exception:
        for layer, cls in previous:
            object.__setattr__(layer, "__class__", cls)
            layer.__dict__.pop("_rapid_gemma4_compiled_ffn", None)
            layer.__dict__.pop("_rapid_gemma4_dtype", None)
        logger.debug("Gemma 4 layer compile declined", exc_info=True)
        return 0


def is_installed(model) -> bool:
    """Probe this model's per-instance installation, never process-global state."""
    return any(
        getattr(type(block), "_rapid_gemma4_moe_decode_fusion", False)
        for _, block in model.named_modules()
    )


def install_gemma4_moe_decode_fusion(model, model_type: str) -> int:
    """Resolve policy once at load; legacy environment switches gate each part."""
    if model_type not in _MODEL_TYPES:
        return 0
    try:
        import mlx.core as mx

        if not install_enabled(
            "gemma4_moe_decode_fusion", mx, route="gemma4_moe_decode_fusion_install"
        ):
            return 0
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            return 0
        families = _expert_types()
        targets = [
            block for _, block in model.named_modules() if type(block) in families
        ]
        if not targets:
            return 0
        from ..moe_fusion import fuse_gate_up

        # The multimodal family's single-token decode leaves gate/up as two
        # selected projections; its extended contract is qualified for Gemma.
        fuse_gate_up(model, extended_call_contract=True)
        if env_falsey("RAPID_MLX_MOE_DECODE_FUSION"):
            return 0
        count = 0
        for block in targets:
            base: Any = type(block)
            block.__class__ = _decode_type(base, base.__call__)
            count += 1
        compiled = _compile_layers(model, mx)
        logger.info(
            "Gemma 4 decode fusion: %d expert blocks, %d compiled layers",
            count,
            compiled,
        )
        return count
    except Exception:
        logger.debug(
            "Gemma 4 MoE fusion unavailable; retaining legacy path", exc_info=True
        )
        return 0
