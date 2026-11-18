# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Hy4 Flash equations; the caller supplies the array backend."""


def sigmoid(x, xp):
    return 1 / (1 + xp.exp(-x))


def softmax(x, axis, xp):
    x = xp.exp(x - xp.max(x, axis=axis, keepdims=True))
    return x / xp.sum(x, axis=axis, keepdims=True)


def rms(x, weight, eps, xp):
    x = x.astype(xp.float32)
    return x * (xp.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5 * weight


def hc_pre(x, weight, scale, base, eps, norm_eps, magnitude, xp):
    flat = x.reshape(*x.shape[:-2], -1).astype(xp.float32)
    mixes = (flat @ weight.T) * (
        xp.mean(flat * flat, axis=-1, keepdims=True) + norm_eps
    ) ** -0.5
    hc = x.shape[-2]
    pre = sigmoid(mixes[..., :hc] * scale[0] + base[:hc], xp) + eps
    post = magnitude * sigmoid(mixes[..., hc:] * scale[1] + base[hc:], xp) + eps
    return xp.sum(pre[..., None] * x, axis=-2).astype(x.dtype), post


def hc_post(y, residual, post, xp):
    return (
        residual.astype(xp.float32)
        + post[..., None] * y.astype(xp.float32)[..., None, :]
    ).astype(y.dtype)


def hc_head(x, weight, scale, base, eps, norm_eps, xp):
    flat = x.reshape(*x.shape[:-2], -1).astype(xp.float32)
    mixes = (flat @ weight.T) * (
        xp.mean(flat * flat, axis=-1, keepdims=True) + norm_eps
    ) ** -0.5
    pre = sigmoid(mixes * scale + base, xp) + eps
    return xp.sum(pre[..., None] * x, axis=-2).astype(x.dtype)


def route(logits, bias, top_k, normalize, factor, xp):
    scores = sigmoid(logits.astype(xp.float32), xp)
    indices = xp.argsort(scores + bias, axis=-1)[..., -top_k:]
    weights = xp.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / (xp.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return indices, weights * factor


def rope(x, positions, dims, theta, xp, *, indexer=False):
    """Adjacent-pair checkpoint rotation; IndexPool returns split-half layout."""
    inverse = theta ** (-xp.arange(0, dims, 2, dtype=xp.float32) / dims)
    frequency = positions[..., None] * inverse
    while frequency.ndim < x.ndim:
        frequency = xp.expand_dims(frequency, -2)
    pe = x[..., -dims:].astype(xp.float32)
    left, right = pe[..., ::2], pe[..., 1::2]
    a = left * xp.cos(frequency) - right * xp.sin(frequency)
    b = right * xp.cos(frequency) + left * xp.sin(frequency)
    if indexer:
        rotated = xp.concatenate((a, b), axis=-1)
    else:
        rotated = xp.stack((a, b), axis=-1).reshape(pe.shape)
    return xp.concatenate((x[..., :-dims], rotated.astype(x.dtype)), axis=-1)


def fwht_fp8(x, xp):
    """Normalized Hadamard followed by power-of-two scaled E4M3 rounding."""
    shape = x.shape
    width = shape[-1]
    x = x.astype(xp.float32)
    stride = 1
    while stride < width:
        pairs = x.reshape(*shape[:-1], -1, 2, stride)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        x = xp.stack((left + right, left - right), axis=-2).reshape(shape)
        stride *= 2
    x = x * width**-0.5
    exp2 = getattr(xp, "exp2", None) or (lambda v: xp.power(2.0, v))
    scale = exp2(
        xp.ceil(
            xp.log2(xp.maximum(xp.max(xp.abs(x), axis=-1, keepdims=True), 1e-4) / 448)
        )
    )
    value = xp.clip(x / scale, -448, 448)
    step = exp2(
        xp.maximum(xp.floor(xp.log2(xp.maximum(xp.abs(value), 2.0**-9))) - 3, -9)
    )
    return xp.round(value / step) * step * scale


def gated_pool(values, gates, ape, norm_weight, eps, xp):
    probabilities = softmax(gates.astype(xp.float32) + ape, -2, xp)
    return rms(
        xp.sum(values.astype(xp.float32) * probabilities, axis=-2), norm_weight, eps, xp
    )


def attention(q, k, v, allowed, sink, xp):
    """Sink contributes to the softmax denominator and has zero value."""
    logits = (
        q.astype(xp.float32) @ xp.swapaxes(k.astype(xp.float32), -2, -1)
    ) * q.shape[-1] ** -0.5
    logits = xp.where(allowed[:, None], logits, -float("inf"))
    sinks = xp.broadcast_to(sink[None, :, None, None], (*logits.shape[:-1], 1))
    probabilities = softmax(xp.concatenate((logits, sinks), axis=-1), -1, xp)[..., :-1]
    return (probabilities @ v.astype(xp.float32)).astype(q.dtype)


def sparse_mask(query, key, group_logits, ratio, budget, xp):
    """Recall completed groups plus the causal tail of the current group."""
    groups = xp.arange(group_logits.shape[-1])
    upper = query // ratio
    live = groups[None, None, :] < upper[..., None]
    scores = xp.where(live, group_logits, -float("inf"))
    indices = (
        xp.argsort(scores, axis=-1)[..., -min(budget - 1, scores.shape[-1]) :]
        if budget > 1
        else xp.zeros((*query.shape, 0), dtype=xp.int32)
    )
    selected = xp.any(key[:, None, :, None] // ratio == indices[:, :, None, :], axis=-1)
    history = selected & (key[:, None, :] // ratio < upper[..., None])
    tail = key[:, None, :] // ratio == upper[..., None]
    return (
        (history | tail)
        & (key[:, None, :] >= 0)
        & (key[:, None, :] <= query[..., None])
    )
