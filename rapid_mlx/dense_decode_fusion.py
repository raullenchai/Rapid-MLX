# SPDX-License-Identifier: Apache-2.0
"""Model-local, functional dense decode fusions.

Only resolved no-spec loads enroll. Every intermediate belongs to one call;
there are no pending projection results, norm caches, or request references.
Multi-row calls and hidden-state capture use the original backbone unchanged.
Set RAPID_MLX_DENSE_DECODE_FUSION=0 before loading to restore stock execution.
"""

# Attention/decoder composition adapted from Copyright (c) 2025-2026 Apple Inc.
# MIT permission notice: kernels/LICENSE-RMS.
from __future__ import annotations

import importlib
import inspect
import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .kernels.dense_residual_norm import seam

logger = logging.getLogger(__name__)
_SWITCH = "RAPID_MLX_DENSE_DECODE_FUSION"
_BACKBONES = {
    ("mlx_vlm.models.qwen3_5.language", "Qwen3_5Model"): "qwen-vlm",
    ("mlx_lm.models.qwen3_5", "Qwen3_5TextModel"): "qwen-lm",
    ("mlx_vlm.models.lfm2.language", "Lfm2Model"): "lfm",
    ("mlx_lm.models.lfm2", "Lfm2Model"): "lfm",
    ("rapid_mlx.models.lfm3", "Lfm3Model"): "lfm3",
}


def _bits_equal(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item())
    )


def _singleton(x):
    return (
        x.ndim == 3
        and x.shape[:2] == (1, 1)
        and 0 < x.shape[-1] <= 4096
        and x.dtype in (mx.float16, mx.bfloat16)
    )


def _norm_probe(norm, dtype):
    if type(norm) is not nn.RMSNorm or norm.weight.dtype != dtype:
        return False
    n = norm.weight.size
    for scale in (0.01, 1.0, 10.0):
        a = (mx.sin(mx.arange(n, dtype=mx.float32)) * scale).astype(dtype)[None, None]
        b = (mx.cos(mx.arange(n, dtype=mx.float32)) * scale).astype(dtype)[None, None]
        residual, normalized = seam(a, b, norm)
        if not (_bits_equal(residual, a + b) and _bits_equal(normalized, norm(a + b))):
            return False
    return True


class _ProjectionGroup(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.bits = parts[0].bits
        self.group_size = parts[0].group_size
        self.weight = mx.concatenate([p.weight for p in parts])
        self.scales = mx.concatenate([p.scales for p in parts])
        self.biases = mx.concatenate([p.biases for p in parts])

    def __call__(self, x):
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


def _group_attention(module, dtype):
    cls = type(module)
    if cls.__module__ not in {
        "mlx_vlm.models.qwen3_5.language",
        "mlx_vlm.models.lfm2.language",
    }:
        return False
    parts = [module.q_proj, module.k_proj, module.v_proj]
    if not all(
        type(p) is nn.QuantizedLinear
        and p.get("bias") is None
        and getattr(p, "mode", "affine") == "affine"
        and p.get("biases") is not None
        for p in parts
    ):
        return False
    metadata = {
        (
            p.bits,
            p.group_size,
            p.weight.shape[1:],
            p.scales.shape[1:],
            p.scales.dtype,
            p.biases.dtype,
            getattr(p, "mode", "affine"),
        )
        for p in parts
    }
    if len(metadata) != 1 or getattr(parts[0], "mode", "affine") != "affine":
        return False
    group = _ProjectionGroup(parts)
    widths = [p.weight.shape[0] for p in parts]
    bounds = (widths[0], widths[0] + widths[1])
    n = parts[0].scales.shape[-1] * parts[0].group_size
    for fn in (mx.sin, mx.cos):
        x = fn(mx.arange(n, dtype=mx.float32)).astype(dtype)[None, None]
        if not all(
            _bits_equal(y, p(x))
            for y, p in zip(mx.split(group(x), bounds, axis=-1), parts)
        ):
            return False
    # Unregistered: the public parameter tree keeps only the stock names.
    object.__setattr__(module, "_rapid_dense_qkv", group)
    object.__setattr__(module, "_rapid_dense_qkv_bounds", bounds)
    start = 0
    for p, width in zip(parts, widths):
        for key in ("weight", "scales", "biases"):
            setattr(p, key, getattr(group, key)[start : start + width])
        start += width
    return True


def _attention(module, x, mask, cache, api, kind, position_ids, position_embeddings):
    q, k, v = mx.split(
        module._rapid_dense_qkv(x), module._rapid_dense_qkv_bounds, axis=-1
    )
    if kind == "qwen-vlm":
        q, k, v, gate, mask = module._prepare_projected_qkv(
            q, k, v, cache, position_ids, position_embeddings, mask
        )
    else:
        q, k, v = module._prepare_projected_qkv(q, k, v, cache)
    output = api.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=module.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
    if kind == "qwen-vlm":
        return module.o_proj(output * mx.sigmoid(gate))
    return module.out_proj(output)


def _group_mlp(module, dtype):
    cls = type(module)
    names = (
        ("w1", "w3", "w2")
        if cls.__name__ == "MLP" and "lfm2" in cls.__module__
        else ("gate_proj", "up_proj", "down_proj")
    )
    if not all(hasattr(module, n) for n in names):
        return False
    parts = [getattr(module, n) for n in names[:2]]
    if not all(
        type(p) is nn.QuantizedLinear
        and p.get("bias") is None
        and getattr(p, "mode", "affine") == "affine"
        and p.get("biases") is not None
        for p in parts
    ):
        return False
    metadata = {
        (
            p.bits,
            p.group_size,
            p.weight.shape,
            p.scales.shape,
            p.scales.dtype,
            p.biases.dtype,
            getattr(p, "mode", "affine"),
        )
        for p in parts
    }
    if len(metadata) != 1 or getattr(parts[0], "mode", "affine") != "affine":
        return False
    group = _ProjectionGroup(parts)
    n = parts[0].scales.shape[-1] * parts[0].group_size
    for fn in (mx.sin, mx.cos):
        x = fn(mx.arange(n, dtype=mx.float32)).astype(dtype)[None, None]
        actual = mx.split(group(x), 2, axis=-1)
        if not all(_bits_equal(y, p(x)) for y, p in zip(actual, parts)):
            return False
    original = cls.__call__
    api = importlib.import_module(cls.__module__)
    activation = (
        api.geglu
        if cls.__module__
        in {
            "rapid_mlx.models.gemma4_vendored.language",
            "mlx_vlm.models.gemma4.language",
        }
        else api.swiglu
    )

    def call(self, x):
        if not _singleton(x) or x.dtype != dtype:
            return original(self, x)
        gate, up = mx.split(self._rapid_dense_group(x), 2, axis=-1)
        return getattr(self, names[2])(activation(gate, up))

    object.__setattr__(module, "_rapid_dense_group", group)
    # Keep original parameter names as views, including the stock wide path.
    width = parts[0].weight.shape[0]
    for i, part in enumerate(parts):
        for key in ("weight", "scales", "biases"):
            setattr(part, key, getattr(group, key)[i * width : (i + 1) * width])
    object.__setattr__(
        module, "__class__", type(cls.__name__, (cls,), {"__call__": call})
    )
    return True


def _decode(backbone, h, cache, kind, api, position_ids=None):
    """Explicit residual carry. Stock layers/cache operators retain their API."""
    if cache is None:
        cache = [None] * len(backbone.layers)
    is_lfm = kind in ("lfm", "lfm3")
    if kind == "lfm3":
        attn_mask = ssm_mask = None
    elif kind == "qwen-vlm":
        attn_mask = api._create_qwen3_5_attention_mask(h, cache[backbone.fa_idx])
        ssm_mask = api._create_qwen3_5_ssm_mask(h, cache[backbone.ssm_idx])
        api._set_qwen3_5_decode_left_padding(cache, backbone.layers, None)
    else:
        attn_mask = api.create_attention_mask(h, cache[backbone.fa_idx])
        ssm_index = backbone.conv_idx if kind == "lfm" else backbone.ssm_idx
        ssm_mask = api.create_ssm_mask(h, cache[ssm_index])
    position_embeddings = None
    if kind == "qwen-vlm" and position_ids is not None:
        for layer in backbone.layers:
            if not layer.is_linear:
                if not layer.self_attn.rotary_emb.fused_apply:
                    position_embeddings = layer.self_attn.rotary_emb(h, position_ids)
                break
    normalized = None
    for idx, (layer, c) in enumerate(zip(backbone.layers, cache), 1):
        if kind == "lfm3":
            if layer.is_attention_layer:
                window = (
                    backbone.config.sliding_window
                    if layer.layer_type == "sliding_attention"
                    else None
                )
                attn_mask = api.create_attention_mask(h, c, window_size=window)
            else:
                ssm_mask = api.create_ssm_mask(h, c)
        first = layer.operator_norm if is_lfm else layer.input_layernorm
        if normalized is None:
            normalized = first(h)
        if is_lfm:
            if layer.is_attention_layer and hasattr(
                layer.self_attn, "_rapid_dense_qkv"
            ):
                branch = _attention(
                    layer.self_attn,
                    normalized,
                    attn_mask,
                    c,
                    api,
                    kind,
                    position_ids,
                    position_embeddings,
                )
            elif layer.is_attention_layer:
                branch = layer.self_attn(
                    normalized,
                    mask=attn_mask,
                    cache=c,
                    **({"position_ids": position_ids} if kind == "lfm3" else {}),
                )
            else:
                branch = layer.conv(normalized, mask=ssm_mask, cache=c)
            second, mlp = layer.ffn_norm, layer.feed_forward
        else:
            if layer.is_linear:
                branch = layer.linear_attn(normalized, ssm_mask, c)
            elif kind == "qwen-vlm" and hasattr(layer.self_attn, "_rapid_dense_qkv"):
                branch = _attention(
                    layer.self_attn,
                    normalized,
                    attn_mask,
                    c,
                    api,
                    kind,
                    position_ids,
                    position_embeddings,
                )
            elif kind == "qwen-vlm":
                branch = layer.self_attn(
                    normalized,
                    mask=attn_mask,
                    cache=c,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
            else:
                branch = layer.self_attn(normalized, attn_mask, c)
            second, mlp = layer.post_attention_layernorm, layer.mlp
        if kind == "lfm3" and layer.use_sandwich_norm:
            branch = layer.post_operator_norm(branch)
        h, normalized = seam(h, branch, second)
        # Carry the two add operands until the next input norm, without storage
        # on a layer or norm object. Final norm consumes the same pair.
        branch = mlp(normalized)
        if kind == "lfm3" and layer.use_sandwich_norm:
            branch = layer.post_ffn_norm(branch)
        final = backbone.embedding_norm if kind in ("lfm", "lfm3") else backbone.norm
        next_norm = (
            final
            if idx == len(backbone.layers)
            else (
                backbone.layers[idx].operator_norm
                if is_lfm
                else backbone.layers[idx].input_layernorm
            )
        )
        h, normalized = seam(h, branch, next_norm)
    return normalized


def _make_call(cls, kind, dtype):
    original = cls.__call__
    signature = inspect.signature(original)
    api = importlib.import_module(cls.__module__)

    def call(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        values = bound.arguments
        inputs = values.get("inputs")
        cache = values.get("cache")
        embeddings = values.get("inputs_embeds", values.get("input_embeddings"))
        if (
            inputs is None
            or inputs.shape != (1, 1)
            or any(
                values.get(k) is not None
                for k in (
                    "capture_layer_ids",
                    "hidden_sink",
                    "gdn_sink",
                    "mask",
                )
            )
            or cache is not None
            and len(cache) != len(self.layers)
            or cache is not None
            and any(
                c is not None
                and (
                    getattr(c, "left_padding", None) is not None
                    or type(c)
                    not in (
                        (api.KVCache, api.ArraysCache, api.RotatingKVCache)
                        if kind == "lfm3"
                        else (api.KVCache, api.ArraysCache)
                    )
                )
                for c in cache
            )
        ):
            return original(self, *args, **kwargs)
        h = self.embed_tokens(inputs) if embeddings is None else embeddings
        if not _singleton(h) or h.dtype != dtype:
            return original(self, *args, **kwargs)
        return _decode(self, h, cache, kind, api, values.get("position_ids"))

    return call


def _probe_cache(backbone, kind, api):
    caches = []
    for layer in backbone.layers:
        if kind in ("lfm", "lfm3"):
            if not layer.is_attention_layer:
                cache = api.ArraysCache(size=1)
            elif kind == "lfm3" and layer.layer_type == "sliding_attention":
                cache = api.RotatingKVCache(max_size=backbone.config.sliding_window)
            else:
                cache = api.KVCache()
        else:
            cache = api.ArraysCache(size=2) if layer.is_linear else api.KVCache()
        caches.append(cache)
    return caches


def _enroll_gemma(layer):
    """Fuse the attention add into the FFN input norm, preserving branch norms."""
    cls = type(layer)
    dtype = layer.pre_feedforward_layernorm.weight.dtype
    if dtype not in (mx.float16, mx.bfloat16) or layer.per_layer_input_gate is not None:
        return False
    if not _norm_probe(layer.pre_feedforward_layernorm, dtype):
        return False
    original = cls.__call__
    has_shared_cache = "shared_cache" in inspect.signature(original).parameters

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
        if has_shared_cache or shared_cache is not None:
            # An unsupported argument reaches the original and fails there.
            shared["shared_cache"] = shared_cache
            if not has_shared_cache:
                return original(self, x, mask, cache, per_layer_input, **shared)
        if not _singleton(x) or x.dtype != dtype or per_layer_input is not None:
            return original(self, x, mask, cache, per_layer_input, **shared)
        branch, shared_kv, offset = self.self_attn(
            self.input_layernorm(x), mask, cache, **shared
        )
        residual, normalized = seam(
            x, self.post_attention_layernorm(branch), self.pre_feedforward_layernorm
        )
        branch = self.mlp(normalized)
        if self.enable_moe:
            branch = self.post_feedforward_layernorm_1(branch)
            indices, weights = self.router(residual)
            routed = self.experts(
                self.pre_feedforward_layernorm_2(residual), indices, weights
            )
            branch = branch + self.post_feedforward_layernorm_2(routed)
        out = residual + self.post_feedforward_layernorm(branch)
        if self.layer_scalar is not None:
            out = out * self.layer_scalar
        return out, shared_kv, offset

    x = mx.sin(
        mx.arange(layer.pre_feedforward_layernorm.weight.size, dtype=mx.float32)
    ).astype(dtype)[None, None]
    if not _bits_equal(call(layer, x)[0], original(layer, x)[0]):
        return False
    _group_mlp(layer.mlp, dtype)
    object.__setattr__(
        layer, "__class__", type(cls.__name__, (cls,), {"__call__": call})
    )
    return True


def _snapshot(model):
    return [
        (module, type(module), dict(module), module.__dict__.copy())
        for _, module in model.named_modules()
    ]


def _restore(snapshot):
    for module, cls, values, attrs in snapshot:
        object.__setattr__(module, "__class__", cls)
        dict.clear(module)
        dict.update(module, values)
        module.__dict__.clear()
        module.__dict__.update(attrs)


def install_dense_decode_fusion(model, *, spec_decode="none"):
    """Enroll compatible instances after real-weight device probes; fail closed."""
    if spec_decode != "none" or os.environ.get(_SWITCH, "1") == "0":
        return 0
    if not callable(getattr(model, "named_modules", None)):
        return 0
    count = 0
    modules = [m for _, m in model.named_modules()]
    gemma = [
        m
        for m in modules
        if type(m).__name__ == "DecoderLayer"
        and type(m).__module__
        in {
            "rapid_mlx.models.gemma4_vendored.language",
            "mlx_vlm.models.gemma4.language",
        }
    ]
    if gemma:
        # The decoder enrolls as a unit: any declined layer restores them all.
        snapshot = [entry for layer in gemma for entry in _snapshot(layer)]
        try:
            if all([_enroll_gemma(layer) for layer in gemma]):
                count += len(gemma)
            else:
                _restore(snapshot)
        except Exception:
            _restore(snapshot)
            logger.debug("Gemma decode enrollment declined", exc_info=True)
    for backbone in modules:
        cls = type(backbone)
        if any(backbone is layer for layer in gemma):
            continue
        kind = _BACKBONES.get((cls.__module__, cls.__name__))
        if kind is None or getattr(backbone, "_rapid_dense_enrolled", False):
            continue
        snapshot = None
        try:
            args = getattr(backbone, "args", getattr(backbone, "config", None))
            if getattr(args, "num_experts", 0):
                continue
            layers = backbone.layers
            if not layers:
                continue
            norms = [
                backbone.embedding_norm if kind in ("lfm", "lfm3") else backbone.norm
            ]
            for layer in layers:
                norms.extend(
                    (layer.operator_norm, layer.ffn_norm)
                    if kind in ("lfm", "lfm3")
                    else (layer.input_layernorm, layer.post_attention_layernorm)
                )
            dtype = norms[0].weight.dtype
            if dtype not in (mx.float16, mx.bfloat16) or norms[0].weight.size > 4096:
                continue
            if not all(_norm_probe(n, dtype) for n in norms):
                continue
            candidate = _make_call(cls, kind, dtype)
            # Enrollment runs before publication. Stage projection rewrites,
            # then verify the complete composition on independent caches.
            # Restore every class/parameter/attribute if the device declines.
            inputs = mx.array([[1]], mx.int32)
            api = importlib.import_module(cls.__module__)

            expected = backbone(inputs, cache=_probe_cache(backbone, kind, api))
            mx.eval(expected)
            snapshot = _snapshot(backbone)
            for layer in layers:
                mlp = layer.feed_forward if kind in ("lfm", "lfm3") else layer.mlp
                _group_mlp(mlp, dtype)
                if (
                    getattr(layer, "is_attention_layer", False)
                    or getattr(layer, "is_linear", True) is False
                ):
                    _group_attention(layer.self_attn, dtype)
            if not _bits_equal(
                candidate(backbone, inputs, cache=_probe_cache(backbone, kind, api)),
                expected,
            ):
                _restore(snapshot)
                continue
        except Exception as exc:
            if snapshot is not None:
                _restore(snapshot)
            logger.debug(
                "Dense decode enrollment declined: %s",
                type(exc).__name__,
                exc_info=True,
            )
            continue
        object.__setattr__(
            backbone, "__class__", type(cls.__name__, (cls,), {"__call__": candidate})
        )
        object.__setattr__(backbone, "_rapid_dense_enrolled", True)
        count += 1
    if count:
        logger.info("Enrolled functional dense decode fusions on %d modules", count)
    return count
