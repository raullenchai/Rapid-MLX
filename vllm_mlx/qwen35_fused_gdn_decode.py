# SPDX-License-Identifier: Apache-2.0
"""Fuse Qwen3.5-family GatedDeltaNet single-token decode on Metal.

The input projections remain owned by :mod:`gdn_in_proj_fusion`.  This module
collapses the remaining causal convolution, Q/K normalization, recurrent
update, and gated RMSNorm into one launch.  The output projection stays on the
stock path.

Enrollment is model-local and fail-closed.  A real Metal probe must reproduce
the stock output, convolution cache, and FP32 recurrent state bit-for-bit
before any layer is tagged.  Prefill, batching, masks, training, sharding,
ragged caches, unknown geometry, and speculative verification stay stock.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from threading import Lock
from typing import Any, cast

import mlx.core as mx

from .kernels.qwen4_fused_gdn_decode import (
    fused_gdn_decode,
    fused_gdn_runtime_supported,
)

logger = logging.getLogger(__name__)

_NUM_KEY_HEADS = 16
_NUM_VALUE_HEADS = 32
_KEY_HEAD_DIM = 128
_VALUE_HEAD_DIM = 128
_CONV_KERNEL = 4
_KEY_DIM = _NUM_KEY_HEADS * _KEY_HEAD_DIM
_VALUE_DIM = _NUM_VALUE_HEADS * _VALUE_HEAD_DIM
_CONV_DIM = 2 * _KEY_DIM + _VALUE_DIM
_THREADGROUP_Y_CANDIDATES = (32, 16, 8, 4)
_TAG = "_rapid_qwen35_fused_gdn_decode"

_PROBE_LOCK = Lock()
_PROBE_COMPLETE = False
_PROBED_THREADGROUP_Y: int | None = None


def _bytes_equal(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def _stock_step(
    *,
    nn: Any,
    gated_delta_update: Callable[..., Any],
    conv: Any,
    qkv: mx.array,
    z: mx.array,
    beta: mx.array,
    alpha: mx.array,
    conv_state: mx.array,
    recurrent_state: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    norm_weight: mx.array,
    norm_eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv_state = mx.contiguous(conv_input[:, -(_CONV_KERNEL - 1) :, :])
    convolved = nn.silu(conv(conv_input))
    query, key, value = [
        part.reshape(1, 1, heads, dim)
        for part, heads, dim in zip(
            mx.split(convolved, [_KEY_DIM, 2 * _KEY_DIM], axis=-1),
            [_NUM_KEY_HEADS, _NUM_KEY_HEADS, _NUM_VALUE_HEADS],
            [_KEY_HEAD_DIM, _KEY_HEAD_DIM, _VALUE_HEAD_DIM],
            strict=True,
        )
    ]
    inv_scale = _KEY_HEAD_DIM**-0.5
    query = (inv_scale**2) * mx.fast.rms_norm(query, None, 1e-6)
    key = inv_scale * mx.fast.rms_norm(key, None, 1e-6)
    output, next_recurrent_state = gated_delta_update(
        query,
        key,
        value,
        alpha,
        beta,
        a_log,
        dt_bias,
        recurrent_state,
        None,
        use_kernel=True,
    )
    normalized = mx.fast.rms_norm(output, norm_weight, norm_eps)
    gate = nn.silu(
        z.reshape(1, 1, _NUM_VALUE_HEADS, _VALUE_HEAD_DIM).astype(mx.float32)
    )
    output = (gate * normalized.astype(mx.float32)).astype(qkv.dtype)
    return output.reshape(1, 1, _VALUE_DIM), next_conv_state, next_recurrent_state


def _probe_candidate(threadgroup_y: int) -> bool:
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    dtype = mx.bfloat16
    conv = nn.Conv1d(
        _CONV_DIM,
        _CONV_DIM,
        kernel_size=_CONV_KERNEL,
        groups=_CONV_DIM,
        bias=False,
    )
    conv.weight = (
        mx.random.normal((_CONV_DIM, _CONV_KERNEL, 1), key=mx.random.key(3501)) * 0.1
    ).astype(dtype)
    a_log = mx.random.normal((_NUM_VALUE_HEADS,), key=mx.random.key(3502)).astype(dtype)
    dt_bias = mx.random.normal((_NUM_VALUE_HEADS,), key=mx.random.key(3503)).astype(
        dtype
    )
    norm_weight = mx.random.normal((_VALUE_HEAD_DIM,), key=mx.random.key(3504)).astype(
        dtype
    )
    stock_conv = mx.zeros((1, _CONV_KERNEL - 1, _CONV_DIM), dtype=dtype)
    fused_conv = mx.array(stock_conv)
    stock_state = mx.zeros(
        (1, _NUM_VALUE_HEADS, _VALUE_HEAD_DIM, _KEY_HEAD_DIM), dtype=mx.float32
    )
    fused_state = mx.array(stock_state)

    for step in range(8):
        qkv = (
            mx.random.normal((1, 1, _CONV_DIM), key=mx.random.key(3600 + step)) * 0.3
        ).astype(dtype)
        z = (
            mx.random.normal((1, 1, _VALUE_DIM), key=mx.random.key(3700 + step)) * 2.0
        ).astype(dtype)
        # Pin the sigmoid edge where fast and precise forms are known to differ.
        z = mx.concatenate([mx.array([[[-6.84375]]], dtype=dtype), z[..., 1:]], -1)
        beta = mx.random.normal(
            (1, 1, _NUM_VALUE_HEADS), key=mx.random.key(3800 + step)
        ).astype(dtype)
        alpha = mx.random.normal(
            (1, 1, _NUM_VALUE_HEADS), key=mx.random.key(3900 + step)
        ).astype(dtype)
        stock_output, stock_conv, stock_state = _stock_step(
            nn=nn,
            gated_delta_update=gated_delta_update,
            conv=conv,
            qkv=qkv,
            z=z,
            beta=beta,
            alpha=alpha,
            conv_state=stock_conv,
            recurrent_state=stock_state,
            a_log=a_log,
            dt_bias=dt_bias,
            norm_weight=norm_weight,
            norm_eps=1e-6,
        )
        fused_output, fused_conv, fused_state = fused_gdn_decode(
            qkv,
            z,
            beta,
            alpha,
            fused_conv,
            conv.weight,
            a_log,
            dt_bias,
            fused_state,
            norm_weight,
            1e-6,
            threadgroup_y=threadgroup_y,
            num_key_heads=_NUM_KEY_HEADS,
            num_value_heads=_NUM_VALUE_HEADS,
            key_head_dim=_KEY_HEAD_DIM,
            value_head_dim=_VALUE_HEAD_DIM,
            conv_kernel=_CONV_KERNEL,
            qwen35_semantics=True,
        )
        mx.eval(
            stock_output,
            fused_output,
            stock_conv,
            fused_conv,
            stock_state,
            fused_state,
        )
        pairs = (
            (stock_output, fused_output),
            (stock_conv, fused_conv),
            (stock_state, fused_state),
        )
        equal = tuple(_bytes_equal(left, right) for left, right in pairs)
        if not all(equal):
            logger.debug(
                "Qwen3.5 fused GDN probe mismatch: ty=%d step=%d "
                "output=%s conv=%s state=%s max_abs=%s",
                threadgroup_y,
                step,
                equal[0],
                equal[1],
                equal[2],
                tuple(
                    float(mx.max(mx.abs(left - right)).item()) for left, right in pairs
                ),
            )
            return False
    return True


def probe_qwen35_fused_gdn_decode() -> int | None:
    """Return a bit-exact supported threadgroup geometry, once per process."""
    global _PROBE_COMPLETE, _PROBED_THREADGROUP_Y
    if _PROBE_COMPLETE:
        return _PROBED_THREADGROUP_Y
    with _PROBE_LOCK:
        if _PROBE_COMPLETE:
            return _PROBED_THREADGROUP_Y
        if fused_gdn_runtime_supported():
            for candidate in _THREADGROUP_Y_CANDIDATES:
                try:
                    if _probe_candidate(candidate):
                        _PROBED_THREADGROUP_Y = candidate
                        break
                except Exception:
                    # This is an optional launch-count optimization.  Any
                    # compile, runtime, or dependency drift must leave model
                    # loading on the stock implementation.
                    logger.debug(
                        "Qwen3.5 fused GDN probe candidate failed: ty=%d",
                        candidate,
                        exc_info=True,
                    )
                    continue
        _PROBE_COMPLETE = True
        return _PROBED_THREADGROUP_Y


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _eligible(self: Any, inputs: mx.array, mask: Any, cache: Any) -> bool:
    try:
        return bool(
            getattr(self, _TAG, False)
            and inputs.shape == (1, 1, self.hidden_size)
            and inputs.dtype == mx.bfloat16
            and mask is None
            and cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and getattr(cache, "lengths", None) is None
            and getattr(self, "sharding_group", None) is None
            and not self.training
            and hasattr(self, "in_proj_fused")
            and _shape(cache[0]) == (1, _CONV_KERNEL - 1, _CONV_DIM)
            and cache[0].dtype == mx.bfloat16
            and _shape(cache[1])
            == (1, _NUM_VALUE_HEADS, _VALUE_HEAD_DIM, _KEY_HEAD_DIM)
            and cache[1].dtype == mx.float32
        )
    except Exception:
        return False


def _patch_class(gdn_class: type) -> None:
    if getattr(gdn_class, "_rapid_qwen35_fused_gdn_patched", False):
        return
    original = cast(Callable[..., mx.array], gdn_class.__call__)

    def patched(self: Any, inputs: mx.array, mask: Any = None, cache: Any = None):
        if not _eligible(self, inputs, mask, cache):
            return original(self, inputs, mask, cache)
        try:
            from .gdn_in_proj_fusion import _fused_projections

            qkv, z, beta, alpha = _fused_projections(self, inputs)
            output, conv_state, recurrent_state = fused_gdn_decode(
                qkv,
                z,
                beta,
                alpha,
                cache[0],
                self.conv1d.weight,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.norm.weight,
                self.norm.eps,
                threadgroup_y=self._rapid_qwen35_fused_gdn_threadgroup_y,
                num_key_heads=_NUM_KEY_HEADS,
                num_value_heads=_NUM_VALUE_HEADS,
                key_head_dim=_KEY_HEAD_DIM,
                value_head_dim=_VALUE_HEAD_DIM,
                conv_kernel=_CONV_KERNEL,
                qwen35_semantics=True,
            )
        except Exception:
            return original(self, inputs, mask, cache)
        cache[0] = conv_state
        cache[1] = recurrent_state
        cache.advance(1)
        return self.out_proj(output)

    dynamic_class = cast(Any, gdn_class)
    dynamic_class.__call__ = patched
    dynamic_class._rapid_qwen35_fused_gdn_patched = True
    dynamic_class._rapid_qwen35_fused_gdn_original_call = original


def _structurally_eligible(layer: Any) -> bool:
    try:
        return bool(
            (
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_k_dim,
                layer.head_v_dim,
                layer.conv_kernel_size,
            )
            == (
                _NUM_KEY_HEADS,
                _NUM_VALUE_HEADS,
                _KEY_HEAD_DIM,
                _VALUE_HEAD_DIM,
                _CONV_KERNEL,
            )
            and layer.hidden_size == 2048
            and hasattr(layer, "in_proj_fused")
            and _shape(layer.conv1d.weight) == (_CONV_DIM, _CONV_KERNEL, 1)
            and _shape(layer.A_log) == (_NUM_VALUE_HEADS,)
            and _shape(layer.dt_bias) == (_NUM_VALUE_HEADS,)
            and _shape(layer.norm.weight) == (_VALUE_HEAD_DIM,)
        )
    except Exception:
        return False


def install_qwen35_fused_gdn_decode(model: Any) -> int:
    """Enroll exact Qwen3.5-family GDN layers and return their count."""
    if os.environ.get("RAPID_MLX_QWEN35_FUSED_GDN_DECODE", "1") == "0":
        return 0
    try:
        from mlx_lm.models.qwen3_5 import GatedDeltaNet

        layers = [
            module
            for _, module in model.named_modules()
            if type(module) is GatedDeltaNet and _structurally_eligible(module)
        ]
    except Exception:
        return 0
    if not layers:
        return 0
    try:
        threadgroup_y = probe_qwen35_fused_gdn_decode()
    except Exception:
        logger.warning(
            "Qwen3.5 fused GDN probe raised; using stock decode",
            exc_info=True,
        )
        return 0
    if threadgroup_y is None:
        logger.warning("Qwen3.5 fused GDN parity probe failed; using stock decode")
        return 0
    _patch_class(GatedDeltaNet)
    for layer in layers:
        setattr(layer, _TAG, True)
        layer._rapid_qwen35_fused_gdn_threadgroup_y = threadgroup_y
    logger.info("[qwen35_gdn] fused decode enrolled: %d GDN layers", len(layers))
    return len(layers)


__all__ = [
    "install_qwen35_fused_gdn_decode",
    "probe_qwen35_fused_gdn_decode",
]
