# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT

import math
import os
import threading
from functools import partial, wraps
from typing import Any, cast

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import sum_gradients

from ... import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

from rapid_mlx.models.inkling2.compat import (
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
    expert_routing,
    scalar,
)
from rapid_mlx.models.inkling2.config import TextArgs
from rapid_mlx.models.inkling2.kernels import (
    attend_clip_decode,
    attention_mask_decode,
    depth_attention,
    message_mix_decode,
    message_norm_decode,
    relative_attention,
    route_topk_decode,
    route_weights_decode,
    short_convolution_decode,
)

__all__ = [
    "Attention",
    "BlockAttentionResidual",
    "DecoderLayer",
    "DynamicShortConvolution",
    "Experts",
    "FullAttention",
    "FullDecoderLayer",
    "L2Clip",
    "MessageGate",
    "MessageInput",
    "RelativeLogits",
    "ScaledL2Clip",
    "ShortConvolution",
    "SlidingAttention",
    "SlidingDecoderLayer",
    "SoftcappedSwiGLU",
    "SparseMoeBlock",
    "TopKRouter",
    "unweighted_rms_norm",
]


_DECODE_PACK_LOCK = threading.RLock()
_ROUTER_SCORE_EPS = 1e-20
_DECODE_FUSIONS = {"attend_clip": True}


def _block_decode_only(fn):
    """Compile stateless module methods for fast single-token decode.

    The live state dictionary supplies weights as compile inputs, including
    weights replaced by loading or quantization after module construction.
    Python configuration attributes (scaling factors, eps, activation choices)
    and kernel/backend choices are fixed at trace time per shape. Changing them
    after warm-up requires clearing the cached _block_compiled_* attributes;
    no current serving path mutates these configuration attributes.
    Only call this on methods without cache access or Python state mutation.
    """
    name = f"_block_compiled_{fn.__name__}"

    @wraps(fn)
    def call(self, hidden_states, *args):
        if not (
            hidden_states.shape[1] == 1 and _fast_decode_enabled() and _use_kernels()
        ):
            return fn(self, hidden_states, *args)
        compiled = getattr(self, name, None)
        if compiled is None:
            # Each batch size gets its own trace to avoid stale broadcast
            # shapes from shapeless compilation.
            compiled = mx.compile(partial(fn, self), inputs=self.state, shapeless=False)
            # Wrapper setup happens outside the traced function. Private
            # attributes are excluded from nn.Module's parameter traversal.
            setattr(self, name, compiled)
        return compiled(hidden_states, *args)

    return call


def _fast_decode_enabled() -> bool:
    return os.environ.get("RAPID_MLX_INKLING2_FAST_DECODE", "1") != "0"


def _decode_only(fn):
    """Compile stateless helpers only for (B, 1, ...) decode inputs.

    Keep the Python function available through __wrapped__ for parity tests.
    No module, weight or cache is captured in the compiled closure. Helpers
    must support shapeless compilation as server batch sizes change at runtime.
    """
    compiled = mx.compile(fn, shapeless=True)

    @wraps(fn)
    def call(x, *args, **kwargs):
        if x.shape[1] == 1 and _fast_decode_enabled():
            return compiled(x, *args, **kwargs)
        return fn(x, *args, **kwargs)

    return call


def _use_kernels() -> bool:
    return bool(mx.default_device() == mx.gpu and mx.metal.is_available())


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


def _norm_cast(x, weight, eps, dtype, decode=False):
    if decode and _fast_decode_enabled() and _use_kernels():
        out = message_norm_decode(x, None, dtype, weight, eps)
        if out is not None:
            return out
    # Dtype objects are not valid mx.compile arguments. Cast outside the graph.
    return _norm(x, weight, eps).astype(dtype)


@_decode_only
def _norm(x, weight, eps):
    return mx.fast.rms_norm(x, weight, eps)


@_decode_only
def _value_norm(x, eps):
    return unweighted_rms_norm(x.astype(mx.float32), eps).astype(x.dtype)


@_decode_only
def _scale_states(x, scale):
    return (x.astype(mx.float32) * scale).astype(x.dtype)


@_decode_only
def _gate_attention(x, gate):
    return (x.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(
        gate.dtype
    )


def _route_weights(scores, bias, k, route_scale):
    if scores.shape[1] == 1 and _fast_decode_enabled() and _use_kernels():
        indices = mx.argpartition(_routing_order(scores, bias), kth=k - 1, axis=-1)[
            ..., :k
        ]
        return indices, route_weights_decode(scores, indices, route_scale)
    return _route_weights_reference(scores, bias, k, route_scale)


@_decode_only
def _routing_order(scores, bias):
    return -(scores + bias)


def _route_weights_reference(scores, bias, k, route_scale):
    # Not shape-agnostically compiled: argpartition's traced output shape is
    # reused across batch sizes, which returned batch-1 routes for batch 2.
    # This is the non-kernel fallback, so plain eager ops cost nothing on GPU.
    indices = mx.take(
        mx.argpartition(-(scores + bias), kth=k - 1, axis=-1),
        mx.arange(k),
        axis=-1,
    )
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_SCORE_EPS)
    return indices, weights * route_scale


@_decode_only
def _combine_experts(states, weights):
    return (states.astype(mx.float32) * weights[..., None]).sum(axis=-2)


def _message_input(states, message, dtype, decode=False):
    if decode and _fast_decode_enabled() and _use_kernels():
        out = message_norm_decode(states, message, dtype)
        if out is not None:
            return out
    return _normalized_message_input(states, message).astype(dtype)


@_decode_only
def _normalized_message_input(states, message):
    return mx.concatenate(
        [unweighted_rms_norm(states), unweighted_rms_norm(message)], axis=-1
    )


@_decode_only
def _message_mix(states, message, forget, gain):
    return (1 - mx.tanh(forget)) * states + mx.tanh(gain) * message


@_decode_only
def _residual_output(states, residual):
    return (residual + states.astype(mx.float32)).astype(states.dtype)


@partial(mx.compile, shapeless=True)
def _decode_log_scale(offset, floor, alpha):
    return 1.0 + alpha * mx.log(
        mx.maximum((offset.astype(mx.float32) + 1) / floor, 1.0)
    )


@partial(mx.compile, shapeless=True)
def _l2_clip(
    hidden_states: mx.array, scale: mx.array | float, threshold: float
) -> mx.array:
    states = hidden_states.astype(mx.float32) * scale
    norm = mx.sqrt((states * states).sum(axis=-1, keepdims=True))
    scaled = states * mx.minimum(threshold / (norm + 1e-6), 1.0)
    return scaled.astype(hidden_states.dtype)


@partial(mx.compile, shapeless=True)
def _softcapped_swiglu(
    gate: mx.array, up: mx.array, gate_softcap: float, up_softcap: float
) -> mx.array:
    input_dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate_capped = mx.tanh(gate / gate_softcap) * gate_softcap
    up_capped = mx.tanh(up / up_softcap) * up_softcap
    return (gate_capped * mx.sigmoid(gate) * up_capped).astype(input_dtype)


class L2Clip(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, 1.0, self.threshold)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, head_dim))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, self.scale, self.threshold)


class RelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states: mx.array) -> mx.array:
        return relative_states @ self.proj.astype(relative_states.dtype)


def _conv_input(
    hidden_states: mx.array,
    conv_state: mx.array | None,
    conv_mask: mx.array | None,
    kernel_size: int,
) -> mx.array:
    hidden_states = hidden_states.astype(mx.float32)
    if conv_mask is not None:
        hidden_states = mx.where(conv_mask[..., None], hidden_states, 0)
    if conv_state is None:
        shape = (hidden_states.shape[0], kernel_size - 1, hidden_states.shape[-1])
        conv_state = mx.zeros(shape, mx.float32)
    return mx.concatenate([conv_state, hidden_states], axis=1)


def _conv_state(window: mx.array, lengths: mx.array | None, n: int) -> mx.array:
    if window.shape[1] == n:
        # Decode kernels already selected and copied the retained state,
        # including rows with no valid input token.
        return window
    if lengths is None:
        pos = mx.arange(window.shape[1] - n, window.shape[1])[None, :, None]
    else:
        ends = mx.clip(lengths, 0, window.shape[1] - n)
        pos = (ends[:, None] + mx.arange(n))[..., None]
    # A gather copies, so the state does not keep the input in memory.
    return mx.take_along_axis(window, pos, axis=1)


class ShortConvolution(nn.Module):
    def __init__(self, dims: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dims, dims, kernel_size, groups=dims, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
        *,
        lengths: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        K = self.kernel_size
        if hidden_states.shape[1] == 1 and _fast_decode_enabled() and _use_kernels():
            return short_convolution_decode(
                hidden_states,
                conv_state,
                self.conv1d.weight,
                conv_mask,
                K,
                lengths=lengths,
            )
        residual = hidden_states.astype(mx.float32)
        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        # As the reference, the conv runs in the weight dtype.
        out = self.conv1d(window.astype(self.conv1d.weight.dtype))
        out = (out.astype(mx.float32) + residual) / math.sqrt(K + 1)
        return out.astype(hidden_states.dtype), window


class DynamicShortConvolution(nn.Module):
    def __init__(self, dims: int, input_dims: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, dims))
        self.weight_proj = nn.Linear(input_dims, num_heads * kernel_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        dynamic_input: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
        *,
        lengths: mx.array | None = None,
        dynamic: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        B, L, C = hidden_states.shape
        K, H = self.kernel_size, self.num_heads
        dynamic = self.weight_proj(dynamic_input) if dynamic is None else dynamic
        if L == 1 and _fast_decode_enabled() and _use_kernels():
            return short_convolution_decode(
                hidden_states,
                conv_state,
                self.static_weight,
                conv_mask,
                K,
                dynamic=dynamic,
                num_heads=H,
                lengths=lengths,
            )
        dynamic_weights = dynamic.astype(mx.float32)
        dynamic_weights = dynamic_weights.reshape(B, L, H, 1, K)
        static_weights = self.static_weight.astype(mx.float32).T.reshape(H, -1, K)
        weights = static_weights + dynamic_weights

        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        windows = mx.stack(
            [window[:, K - 1 - lag : K - 1 - lag + L] for lag in range(K)], axis=-1
        )
        windows = windows.reshape(B, L, H, -1, K)
        out = (windows * weights).sum(axis=-1).reshape(B, L, C)
        out = (out + hidden_states.astype(mx.float32)) / math.sqrt(2 * K + 1)
        return out.astype(hidden_states.dtype), window


def _positions(offset: int | mx.array, length: int) -> mx.array:
    pos = mx.arange(length)
    if isinstance(offset, mx.array):
        return cast(mx.array, offset[:, None] + pos)
    return pos + offset


def _key_positions(
    cache: Any, num_keys: int, num_queries: int, end: int | mx.array
) -> mx.array:
    k = mx.arange(num_keys)
    if num_queries == 1 and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):
        # A decode step writes in place, so the keys can be rotated.
        k = (k - cache._idx) % num_keys
    if isinstance(end, mx.array):
        return cast(mx.array, (end - num_keys)[:, None] + k)
    return k + (end - num_keys)


def _attention_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    position_bias: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    query_positions: mx.array,
    key_positions: mx.array,
) -> mx.array:
    B, Hq, L, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    R = position_bias.shape[-1]
    distance = query_positions[..., :, None] - key_positions[..., None, :]
    visible = (distance >= 0) & (key_positions[..., None, :] >= 0)
    if window:
        visible = visible & (distance < window)
    if distance.ndim == 3:
        distance, visible = distance[:, None], visible[:, None]
    bias = mx.take_along_axis(
        position_bias.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    bias = mx.where(distance < R, bias.astype(mx.float32), 0.0)
    attention_mask = mx.where(visible, bias, -mx.inf).reshape(B, Hkv, -1, L, S)

    query = query.astype(mx.float32).reshape(B, Hkv, -1, L, D)
    key = key.astype(mx.float32)[:, :, None]
    attn_weights = (query @ key.swapaxes(-1, -2)) * scale + attention_mask
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    attn_weights = mx.softmax(combined_logits, axis=-1)[..., :S]
    attn_output = attn_weights.astype(value.dtype) @ value[:, :, None]
    return attn_output.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


def _decode_attention_mask(position_bias, distance, visible, dtype, extent):
    return _decode_attention_bias(position_bias, distance, visible, extent).astype(
        dtype
    )


# Not shape-agnostically compiled: the traced broadcast reuses the first
# batch size it saw, so a later batch would receive a stale mask shape.
def _decode_attention_bias(position_bias, distance, visible, extent):
    bias = position_bias.transpose(0, 2, 1, 3)
    indices = mx.broadcast_arrays(
        mx.take(bias, mx.array([0]), axis=-1), mx.clip(distance, 0, extent - 1)
    )[1]
    bias = mx.take_along_axis(bias, indices, axis=-1)
    bias = mx.where(distance < extent, bias, mx.zeros_like(bias))
    return mx.where(visible, bias, -mx.inf)


def _packed_quantized_projections(owner, name, projections, x):
    """Pack once after loading; parameter slices share the packed allocation.

    Each pack variant gets a one-time evaluated matmul probe before publication.
    Construction or evaluated-probe failure disables the pack before publication,
    so it is never used and callers keep the unpacked projections. Real calls use
    stock quantized matmul on concatenated weights and return lazy outputs;
    invocation and later evaluation failures propagate without runtime fallback.
    """
    if any(type(p) is not nn.QuantizedLinear or "bias" in p for p in projections):
        return None
    first = projections[0]
    config = (first.group_size, first.bits, first.mode)
    fields = ("weight", "scales") + (
        ("biases",) if first.get("biases") is not None else ()
    )
    if any(
        (p.group_size, p.bits, p.mode) != config
        or (p.get("biases") is None) != (first.get("biases") is None)
        or any(
            p[f].dtype != first[f].dtype or p[f].shape[1:] != first[f].shape[1:]
            for f in fields
        )
        for p in projections
    ):
        return None
    sources = tuple(p[f] for p in projections for f in fields)
    attr = f"_decode_pack_{name}"
    cached = getattr(owner, attr, None)
    if cached is False:
        return None
    if (
        cached is None
        or cached[0] != config
        or any(a is not b for a, b in zip(sources, cached[1]))
    ):
        with _DECODE_PACK_LOCK:
            if getattr(owner, attr, None) is not cached:
                return _packed_quantized_projections(owner, name, projections, x)
            try:
                packed = {
                    f: mx.concatenate([p[f] for p in projections], axis=0)
                    for f in fields
                }
                mx.eval(*packed.values())
                mx.eval(
                    mx.quantized_matmul(
                        mx.zeros((1, 1, x.shape[-1]), dtype=x.dtype),
                        packed["weight"],
                        scales=packed["scales"],
                        biases=packed.get("biases"),
                        transpose=True,
                        group_size=config[0],
                        bits=config[1],
                        mode=config[2],
                    )
                )
                ends = []
                start = 0
                replacements = []
                for p in projections:
                    end = start + p.weight.shape[0]
                    replacements.extend(packed[f][start:end] for f in fields)
                    ends.append(end)
                    start = end
                mx.eval(replacements)
            except Exception:
                object.__setattr__(owner, attr, False)
                return None
            sources = tuple(replacements)
            for p, offset in zip(projections, range(0, len(sources), len(fields))):
                for i, f in enumerate(fields):
                    p[f] = sources[offset + i]
            cached = (config, sources, packed, ends[:-1])
            object.__setattr__(owner, attr, cached)
    output = mx.quantized_matmul(
        x,
        cached[2]["weight"],
        scales=cached[2]["scales"],
        biases=cached[2].get("biases"),
        transpose=True,
        group_size=config[0],
        bits=config[1],
        mode=config[2],
    )
    return mx.split(output, cached[3], axis=-1)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, *, sliding: bool):
        super().__init__()
        if sliding:
            self.num_heads = args.swa_num_attention_heads
            self.num_key_value_heads = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.sliding_window = args.sliding_window_size
            rel_extent = args.sliding_window_size
        else:
            self.num_heads = args.num_attention_heads
            self.num_key_value_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.sliding_window = 0
            rel_extent = args.rel_extent
        self.scaling = self.head_dim**-0.5

        dim = args.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.r_proj = nn.Linear(dim, self.num_heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, q_dim, bias=False)
        self.sink_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.rel_logits_proj = RelativeLogits(args.d_rel, rel_extent)
        self.q_l2_clip = ScaledL2Clip(
            self.num_heads, self.head_dim, args.qk_l2_clip_threshold
        )
        self.k_l2_clip = L2Clip(args.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(
            self.num_heads, args.d_rel, args.rel_l2_clip_threshold
        )
        self.eps = args.rms_norm_eps

    def _project_inputs(self, hidden, key, value, cache):
        primary = kv = kp = vp = None
        if (
            hidden.shape[:2] == (1, 1)
            and key.shape[:2] == value.shape[:2] == (1, 1)
            and cache is not None
            and cache[0] is not None
            and cache[0].offset > 0
            and _fast_decode_enabled()
            and _use_kernels()
        ):
            primary = _packed_quantized_projections(
                self,
                "query",
                (self.q_proj, self.r_proj, self.gate_proj, self.sink_proj),
                hidden,
            )
            if isinstance(self.k_sconv, DynamicShortConvolution):
                kp = _packed_quantized_projections(
                    self, "k", (self.k_proj, self.k_sconv.weight_proj), key
                )
                vp = _packed_quantized_projections(
                    self, "v", (self.v_proj, self.v_sconv.weight_proj), value
                )
            elif key is value:
                kv = _packed_quantized_projections(
                    self, "kv", (self.k_proj, self.v_proj), key
                )
        q, r, *gate_sinks = (
            primary
            if primary is not None
            else (self.q_proj(hidden), self.r_proj(hidden))
        )
        k = kp[0] if kp is not None else kv[0] if kv is not None else self.k_proj(key)
        v = vp[0] if vp is not None else kv[1] if kv is not None else self.v_proj(value)
        kw = kp[1] if kp is not None else None
        vw = vp[1] if vp is not None else None
        return q, k, v, r, gate_sinks, kw, vw

    def _attend(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        cache: Any | None,
        tau: mx.array | None = None,
        gate_sinks=None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        gate, sinks = gate_sinks or (
            self.gate_proj(hidden_states),
            self.sink_proj(hidden_states),
        )

        query_states = query_states.reshape(B, L, self.num_heads, -1)
        key_states = key_states.reshape(B, L, self.num_key_value_heads, -1)
        value_states = value_states.reshape(B, L, self.num_key_value_heads, -1)
        relative_states = relative_states.reshape(B, L, self.num_heads, -1)
        clipped = None
        if (
            L == 1
            and cache is not None
            and _fast_decode_enabled()
            and _use_kernels()
            and _DECODE_FUSIONS["attend_clip"]
        ):
            clipped = attend_clip_decode(
                query_states,
                key_states,
                relative_states,
                self.q_l2_clip.scale,
                self.rel_l2_clip.scale,
                self.q_l2_clip.threshold,
                self.rel_l2_clip.threshold,
                tau,
            )
        if clipped is None:
            query_states = self.q_l2_clip(query_states)
            key_states = self.k_l2_clip(key_states)
            relative_states = self.rel_l2_clip(relative_states)
            if tau is not None:
                query_states = _scale_states(query_states, tau)
                relative_states = _scale_states(relative_states, tau)
        else:
            query_states, key_states, relative_states = clipped
        value_states = _value_norm(value_states, self.eps)
        position_bias = self.rel_logits_proj(relative_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)
        if cache is not None:
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        attn_output = self._attention(
            query_states, key_states, value_states, position_bias, sinks, cache
        )

        attn_output = _gate_attention(attn_output.reshape(B, L, -1), gate)
        return self.o_proj(attn_output)

    def _attention(self, query, key, value, position_bias, sinks, cache) -> mx.array:
        B, _, L, _ = query.shape
        S = key.shape[2]
        end = cache.offset if cache is not None else L
        bf16 = query.dtype == key.dtype == value.dtype == mx.bfloat16
        if L > 1 and _use_kernels() and bf16:
            key_start = mx.broadcast_to(S - end, (B,))
            return relative_attention(
                query,
                key,
                value,
                position_bias,
                sinks,
                scale=self.scaling,
                window=self.sliding_window,
                key_start=key_start,
            )

        if L == 1 and _fast_decode_enabled() and _use_kernels():
            # SDPA's sink vector is shared across rows. Fold batch into heads
            # so each row retains its own learned sinks and GQA head grouping,
            # without appending/copying a dummy key and value to the KV bank.
            rotating = isinstance(cache, (RotatingKVCache, BatchRotatingKVCache))
            mask = attention_mask_decode(
                position_bias,
                end,
                cache._idx if rotating else 0,
                S,
                self.sliding_window,
                query.dtype,
                rotating=rotating,
            )
            Hq, Hkv, D = query.shape[1], key.shape[1], query.shape[-1]
            attn_output = mx.fast.scaled_dot_product_attention(
                query.reshape(1, B * Hq, 1, D),
                key.reshape(1, B * Hkv, S, D),
                value.reshape(1, B * Hkv, S, D),
                scale=self.scaling,
                mask=mask.reshape(1, B * Hq, 1, S),
                sinks=sinks.reshape(B * Hq).astype(query.dtype),
            )
            return attn_output.reshape(B, Hq, 1, D).transpose(0, 2, 1, 3)

        return _attention_reference(
            query,
            key,
            value,
            position_bias,
            sinks,
            scale=self.scaling,
            window=self.sliding_window,
            query_positions=_positions(end - L, L),
            key_positions=_key_positions(cache, S, L, end),
        )


class SlidingAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=True)
        dims = self.num_key_value_heads * self.head_dim
        self.k_sconv = ShortConvolution(dims, args.sconv_kernel_size)
        self.v_sconv = ShortConvolution(dims, args.sconv_kernel_size)

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list[mx.array | None],
        conv_mask: mx.array | None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        q, k, v, r, gate_sinks, _, _ = self._project_inputs(
            hidden_states, key_hidden_states, value_hidden_states, cache
        )
        key_states, conv_states[0] = self.k_sconv(
            k,
            conv_states[0],
            conv_mask,
            lengths=cache[1].lengths if cache is not None else None,
        )
        value_states, conv_states[1] = self.v_sconv(
            v,
            conv_states[1],
            conv_mask,
            lengths=cache[1].lengths if cache is not None else None,
        )
        return self._attend(
            hidden_states,
            q,
            key_states,
            value_states,
            r,
            kv_cache,
            gate_sinks=gate_sinks,
        )


class FullAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=False)
        dims = self.num_key_value_heads * self.head_dim
        conv_args = (args.hidden_size, self.num_key_value_heads, args.sconv_kernel_size)
        self.k_sconv = DynamicShortConvolution(dims, *conv_args)
        self.v_sconv = DynamicShortConvolution(dims, *conv_args)
        self.log_scaling_n_floor = args.log_scaling_n_floor
        self.log_scaling_alpha = args.log_scaling_alpha

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list[mx.array | None],
        conv_mask: mx.array | None,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        q, k, v, r, gate_sinks, kw, vw = self._project_inputs(
            hidden_states, key_hidden_states, value_hidden_states, cache
        )
        key_states, conv_states[0] = self.k_sconv(
            k,
            key_hidden_states,
            conv_states[0],
            conv_mask,
            lengths=cache[1].lengths if cache is not None else None,
            dynamic=kw,
        )
        value_states, conv_states[1] = self.v_sconv(
            v,
            value_hidden_states,
            conv_states[1],
            conv_mask,
            lengths=cache[1].lengths if cache is not None else None,
            dynamic=vw,
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        if hidden_states.shape[1] == 1 and _fast_decode_enabled():
            tau = _decode_log_scale(
                (
                    offset if isinstance(offset, mx.array) else scalar(offset, mx.int32)
                ).reshape(-1, 1),
                scalar(self.log_scaling_n_floor),
                scalar(self.log_scaling_alpha),
            )
        else:
            position_ids = _positions(offset, hidden_states.shape[1]).astype(mx.float32)
            tau = 1.0 + self.log_scaling_alpha * mx.log(
                mx.maximum((position_ids + 1) / self.log_scaling_n_floor, 1.0)
            )
        return self._attend(
            hidden_states,
            q,
            key_states,
            value_states,
            r,
            kv_cache,
            tau=tau[..., None, None],
            gate_sinks=gate_sinks,
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_softcap = args.mlp_gate_softcap
        self.up_softcap = args.mlp_up_softcap

    def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
        return _softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap)


class Experts(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        dims, hidden = args.hidden_size, args.intermediate_size
        num_experts = args.n_routed_experts
        # The rows alternate between gate and up, as in the checkpoint.
        self.gate_up_proj = SwitchLinear(dims, 2 * hidden, num_experts, bias=False)
        self.down_proj = SwitchLinear(hidden, dims, num_experts, bias=False)
        self.act_fn = SoftcappedSwiGLU(args)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        inv_order: mx.array | None = None
        idx = indices
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x = self.act_fn(gate_up[..., 0::2], gate_up[..., 1::2])
        x = self.down_proj(x, idx, sorted_indices=do_sort)
        if do_sort:
            assert inv_order is not None
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class TopKRouter(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.route_scale
        num_experts, dims = args.n_routed_experts, args.hidden_size
        self.weight = mx.zeros((num_experts, dims), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(
        self, hidden_states: mx.array, cached_decode: bool = False
    ) -> tuple[mx.array, mx.array]:
        logits = hidden_states.astype(mx.float32) @ self.weight.T
        if (
            cached_decode
            and hidden_states.shape[:2] == (1, 1)
            and _fast_decode_enabled()
            and _use_kernels()
            and expert_routing is None
            and 1 <= self.top_k <= logits.shape[-1] <= 128
            and logits.dtype == self.e_score_correction_bias.dtype == mx.float32
        ):
            routed = route_topk_decode(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
            )
            if routed is not None:
                return routed
        if (
            _use_kernels()
            and expert_routing is not None
            and (hidden_states.shape[1] != 1 or _fast_decode_enabled())
        ):
            return cast(
                tuple[mx.array, mx.array],
                expert_routing(
                    logits,
                    self.e_score_correction_bias,
                    top_k=self.top_k,
                    multiplier=1.0,
                    score_eps=_ROUTER_SCORE_EPS,
                    route_scale=self.routed_scaling_factor,
                ),
            )
        return cast(
            tuple[mx.array, mx.array],
            _route_weights(
                # Fast decode runs this sigmoid inside the compiled MoE block.
                # Metal routing tests guard identical compiled/eager expert
                # selection and bitwise weights, including top-k cutoff ties.
                mx.sigmoid(logits),
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
            ),
        )


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = TopKRouter(args)
        self.experts = Experts(args)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(
        self, hidden_states: mx.array, *, cached_decode: bool = False
    ) -> mx.array:
        group = self.sharding_group
        if group is not None:
            hidden_states = sum_gradients(group)(hidden_states)
        final_hidden_states = self._forward(hidden_states, cached_decode)
        if group is not None:
            final_hidden_states = mx.distributed.all_sum(
                final_hidden_states, group=group
            )
        return final_hidden_states.astype(hidden_states.dtype)

    @_block_decode_only
    def _forward(self, hidden_states, cached_decode=False):
        selected_experts, routing_weights = self.gate(hidden_states, cached_decode)
        expert_states = self.experts(hidden_states, selected_experts)
        return _combine_experts(expert_states, routing_weights)


class BlockAttentionResidual(nn.Module):
    def __init__(self, dims: int, num_queries: int = 1):
        super().__init__()
        shape = (dims,) if num_queries == 1 else (num_queries, dims)
        self.query = mx.zeros(shape, dtype=mx.float32)

    def __call__(
        self, block_states: mx.array, partial_block_states: mx.array
    ) -> mx.array:
        dims = partial_block_states.shape[-1]
        return _depth_attention(
            block_states, partial_block_states, self.query.reshape(-1, dims)
        )


def _depth_attention(
    block_states: mx.array, partial_block_states: mx.array, query: mx.array
) -> mx.array:
    query = query.astype(mx.float32)
    scale = partial_block_states.shape[-1] ** -0.5
    if _use_kernels():
        return depth_attention(block_states, partial_block_states, query, scale=scale)
    return _depth_attention_reference(partial_block_states, block_states, query, scale)


@_decode_only
def _depth_attention_reference(partial_block_states, block_states, query, scale):
    stacked_states = mx.concatenate([block_states, partial_block_states[None]])
    stacked_states = mx.moveaxis(stacked_states, 0, -2)
    weights = mx.softmax((stacked_states @ query.T) * scale, axis=-2, precise=True)
    return weights.swapaxes(-1, -2) @ stacked_states


class MessageGate(nn.Module):
    def __init__(self, dims: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dims, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, dims, bias=False)
        self.message_gate_proj = nn.Linear(rank, dims, bias=False)

    def __call__(
        self, hidden_states: mx.array, message: mx.array, decode=False
    ) -> mx.array:
        states = _message_input(
            hidden_states, message, self.down_proj.weight.dtype, decode
        )
        states = self.down_proj(states)
        if (
            hidden_states.shape[1] == 1
            and _fast_decode_enabled()
            and _use_kernels()
            and self.forget_gate_proj.weight.dtype
            == self.message_gate_proj.weight.dtype
            == mx.float32
            and "bias" not in self.forget_gate_proj
            and "bias" not in self.message_gate_proj
        ):
            return message_mix_decode(
                hidden_states,
                message,
                states,
                self.forget_gate_proj.weight,
                self.message_gate_proj.weight,
            )
        states = states.astype(mx.float32)
        return _message_mix(
            hidden_states,
            message,
            self.forget_gate_proj(states),
            self.message_gate_proj(states),
        )


class MessageInput(nn.Module):
    def __init__(self, dims: int, num_looped_layers: int):
        super().__init__()
        # The query, key, value and mlp queries of each looped layer.
        self.query = mx.zeros((num_looped_layers, 4, dims), mx.float32)

    def __call__(
        self, bank: tuple[mx.array, mx.array], loop_layer_idx: int
    ) -> mx.array:
        return _depth_attention(*bank, self.query[loop_layer_idx])


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int, *, num_queries: int):
        super().__init__()
        dims = args.hidden_size
        self.is_block_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        self.attention_input = (
            BlockAttentionResidual(dims, num_queries) if layer_idx > 0 else None
        )
        self.mlp_input = BlockAttentionResidual(dims)
        self.mlp = SparseMoeBlock(args)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        kernel_size = args.sconv_kernel_size
        self.post_attention_convolution = ShortConvolution(dims, kernel_size)
        self.post_mlp_convolution = ShortConvolution(dims, kernel_size)
        if layer_idx in args.loop_core:
            assert args.loop_config is not None
            rank = args.loop_config["gate_rank"]
            self.q_message_gate = MessageGate(dims, rank)
            self.k_message_gate = MessageGate(dims, rank)
            self.v_message_gate = MessageGate(dims, rank)
            self.mlp_message_gate = MessageGate(dims, rank)

    def _mix_messages(
        self,
        states: tuple[mx.array, mx.array, mx.array],
        messages: mx.array,
        decode=False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        q_states, k_states, v_states = states
        return (
            self.q_message_gate(q_states, messages[..., 0, :], decode),
            self.k_message_gate(k_states, messages[..., 1, :], decode),
            self.v_message_gate(v_states, messages[..., 2, :], decode),
        )

    @_block_decode_only
    def _mlp_states(
        self, hidden_states, block_states, messages, dtype_source, decode=False
    ):
        normed_states = unweighted_rms_norm(hidden_states)
        hidden_states = self.mlp_input(block_states, normed_states)[..., 0, :]
        if messages is not None:
            hidden_states = self.mlp_message_gate(
                hidden_states, messages[..., 3, :], decode
            )
        return _norm_cast(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            dtype_source.dtype,
            decode and messages is not None,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: mx.array | None,
        *,
        cache: Any | None = None,
        conv_mask: mx.array | None = None,
        messages: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        cached_decode = (
            hidden_states.shape[:2] == (1, 1)
            and cache is not None
            and isinstance(cache[0].offset, int)
            and cache[0].offset > 0
        )
        conv = cache[1] if cache is not None else None
        conv_states: list[mx.array | None] = (
            list(conv.cache) if conv is not None else [None] * 4
        )
        block_residual = hidden_states
        normed_residual = unweighted_rms_norm(block_residual.astype(mx.float32))

        attention_inputs = None
        if self.attention_input is not None:
            assert block_states is not None
            attention_inputs = self.attention_input(block_states, normed_residual)
        hidden_states = self._self_attention(
            hidden_states, attention_inputs, cache, conv_states, conv_mask, messages
        )
        hidden_states, conv_states[2] = self.post_attention_convolution(
            hidden_states,
            conv_states[2],
            conv_mask,
            lengths=conv.lengths if conv is not None else None,
        )

        residual = hidden_states = hidden_states.astype(mx.float32)
        if self.is_block_boundary:
            block = normed_residual[None]
            if block_states is None:
                block_states = block
            else:
                block_states = mx.concatenate([block_states, block])
        else:
            residual = hidden_states = block_residual.astype(mx.float32) + hidden_states

        assert block_states is not None
        hidden_states = self.mlp(
            self._mlp_states(
                hidden_states, block_states, messages, block_residual, cache is not None
            ),
            cached_decode=cached_decode,
        )
        hidden_states, conv_states[3] = self.post_mlp_convolution(
            hidden_states,
            conv_states[3],
            conv_mask,
            lengths=conv.lengths if conv is not None else None,
        )

        if conv is not None:
            n = self.post_mlp_convolution.kernel_size - 1
            for i, window in enumerate(conv_states):
                assert window is not None
                conv[i] = _conv_state(window, conv.lengths, n)
            conv.advance(block_residual.shape[1])
        return _residual_output(hidden_states, residual), block_states


class SlidingDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=1)
        self.self_attn = SlidingAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _self_attention(
        self,
        hidden_states: mx.array,
        attention_inputs: mx.array | None,
        cache: Any | None,
        conv_states: list[mx.array | None],
        conv_mask: mx.array | None,
        messages: mx.array | None,
    ) -> mx.array:
        return self.self_attn(
            *self._attention_states(
                hidden_states, attention_inputs, messages, cache is not None
            ),
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )

    @_block_decode_only
    def _attention_states(
        self, hidden_states, attention_inputs, messages, decode=False
    ):
        dtype = hidden_states.dtype
        if attention_inputs is not None:
            hidden_states = attention_inputs[..., 0, :]
        if messages is None:
            states: tuple[mx.array, ...] | list[mx.array] = (
                _norm_cast(
                    hidden_states,
                    self.input_layernorm.weight,
                    self.input_layernorm.eps,
                    dtype,
                ),
            ) * 3
        else:
            states = self._mix_messages((hidden_states,) * 3, messages, decode)
            states = [
                _norm_cast(
                    s,
                    self.input_layernorm.weight,
                    self.input_layernorm.eps,
                    dtype,
                    decode,
                )
                for s in states
            ]
        return tuple(states)


class FullDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=3)
        self.self_attn = FullAttention(args)
        eps = args.rms_norm_eps
        self.q_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.k_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.v_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def _self_attention(
        self,
        hidden_states: mx.array,
        attention_inputs: mx.array | None,
        cache: Any | None,
        conv_states: list[mx.array | None],
        conv_mask: mx.array | None,
        messages: mx.array | None,
    ) -> mx.array:
        return self.self_attn(
            *self._attention_states(
                hidden_states, attention_inputs, messages, cache is not None
            ),
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )

    @_block_decode_only
    def _attention_states(
        self, hidden_states, attention_inputs, messages, decode=False
    ):
        dtype = hidden_states.dtype
        if attention_inputs is None:
            query_states = key_states = value_states = hidden_states
        else:
            query_states, key_states, value_states = (
                attention_inputs[..., i, :] for i in range(3)
            )
        if messages is not None:
            query_states, key_states, value_states = self._mix_messages(
                (query_states, key_states, value_states), messages, decode
            )
        return (
            _norm_cast(
                query_states,
                self.q_input_layernorm.weight,
                self.q_input_layernorm.eps,
                dtype,
                decode and messages is not None,
            ),
            _norm_cast(
                key_states,
                self.k_input_layernorm.weight,
                self.k_input_layernorm.eps,
                dtype,
                decode and messages is not None,
            ),
            _norm_cast(
                value_states,
                self.v_input_layernorm.weight,
                self.v_input_layernorm.eps,
                dtype,
                decode and messages is not None,
            ),
        )
