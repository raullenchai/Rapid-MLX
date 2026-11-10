# SPDX-License-Identifier: Apache-2.0
"""Model-local residual/RMSNorm carry for qualified singleton plain decode.

Reuse the existing exact seam and decoder composition. Enrollment requires
real-weight output and cache parity. Prefill, verification, hidden capture,
ragged caches, other devices, and unsupported call surfaces keep the original.
"""

from __future__ import annotations

import copy
import logging

import mlx.core as mx

from ._env import env_falsey
from .dense_decode_fusion import _bits_equal, _decode, _norm_probe, _probe_cache
from .tuning import QWEN35_AFFINE4, enabled

logger = logging.getLogger(__name__)
_SWITCH = "RAPID_MLX_QWEN35_RESIDUAL_NORM"
_TAG = "_rapid_qwen35_residual_norm"


def _qualified(backbone):
    cls = type(backbone)
    args = getattr(backbone, "args", None)
    return (
        cls.__module__ == "mlx_vlm.models.qwen3_5_moe.language"
        and cls.__name__ == "Qwen3_5MoeModel"
        and getattr(args, "hidden_size", None) == 2048
        and getattr(args, "num_experts", None) == 256
        and getattr(args, "num_experts_per_tok", None) == 8
        and len(backbone.layers) == 40
        and all(
            layer.input_layernorm.weight.dtype == mx.bfloat16
            for layer in backbone.layers
        )
    )


def _make_call(original, api, cache_types):
    def call(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
        **kwargs,
    ):
        if (
            env_falsey(_SWITCH)
            or inputs.shape != (1, 1)
            or inputs_embeds is not None
            or mask is not None
            or capture_layer_ids is not None
            or hidden_sink is not None
            or kwargs
            or self.training
            or cache is None
            or len(cache) != 40
            or any(type(c) not in cache_types for c in cache)
            or getattr(cache[0], "lengths", None) is not None
            or getattr(cache[0], "left_padding", None) is not None
        ):
            return original(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                capture_layer_ids=capture_layer_ids,
                hidden_sink=hidden_sink,
                **kwargs,
            )
        return _decode(
            self,
            self.embed_tokens(inputs),
            cache,
            "qwen-vlm",
            api,
            position_ids,
        )

    return call


def _probe(backbone, api):
    from mlx.utils import tree_flatten

    norms = [backbone.norm]
    for layer in backbone.layers:
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
    if not all(_norm_probe(norm, mx.bfloat16) for norm in norms):
        return False
    cache = _probe_cache(backbone, "qwen-vlm", api)
    mx.eval(
        backbone(mx.array([[1, 2, 3]], mx.int32), cache=cache), [c.state for c in cache]
    )
    left, right = copy.deepcopy(cache), copy.deepcopy(cache)
    inputs = mx.array([[4]], mx.int32)
    expected = backbone(inputs, cache=left)
    actual = _decode(backbone, backbone.embed_tokens(inputs), right, "qwen-vlm", api)
    mx.eval(expected, actual, [c.state for c in left], [c.state for c in right])
    ls = tree_flatten([c.state for c in left])
    rs = tree_flatten([c.state for c in right])
    return (
        _bits_equal(expected, actual)
        and len(ls) == len(rs)
        and all(lp == rp and _bits_equal(a, b) for (lp, a), (rp, b) in zip(ls, rs))
        and all(
            getattr(a, "offset", None) == getattr(b, "offset", None)
            for a, b in zip(left, right)
        )
    )


def install_qwen35_residual_norm(model, *, spec_decode="none"):
    """Enroll only measured plain-serving models; never patch a global class."""
    if spec_decode != "none" or env_falsey(_SWITCH):
        return 0
    if not enabled(
        "qwen35_residual_norm",
        mx.device_info(),
        family="qwen35_moe",
        geometry=QWEN35_AFFINE4,
        mode="plain",
        width=1,
    ):
        return 0
    named = getattr(model, "named_modules", None)
    if not callable(named):
        return 0
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_vlm.models.qwen3_5 import language as api

    from .engine.batched import _qwen36_text_arrays_cache_type

    cache_types = frozenset(
        (
            ArraysCache,
            KVCache,
            api.ArraysCache,
            api.KVCache,
            _qwen36_text_arrays_cache_type(),
        )
    )
    count = 0
    for _, backbone in list(named()):
        if getattr(backbone, _TAG, False) or not _qualified(backbone):
            continue
        try:
            if not _probe(backbone, api):
                continue
        except Exception:
            logger.debug("Singleton residual norm probe declined", exc_info=True)
            continue
        cls = type(backbone)
        call = _make_call(cls.__call__, api, cache_types)
        object.__setattr__(
            backbone, "__class__", type(cls.__name__, (cls,), {"__call__": call})
        )
        object.__setattr__(backbone, _TAG, True)
        count += 1
    if count:
        logger.info(
            "[qwen35_residual_norm] enrolled %d singleton plain backbones", count
        )
    return count
