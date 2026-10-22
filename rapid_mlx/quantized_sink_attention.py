# SPDX-License-Identifier: Apache-2.0
"""Packed KV attention with a per-query-head softmax denominator logit."""

import mlx.core as mx


def quantized_sink_attention(
    queries,
    q_keys,
    q_values,
    *,
    scale,
    mask,
    sinks,
    group_size=64,
    bits=8,
):
    """Attend to affine quantized KV without materializing the dense history.

    Sinks contribute probability mass but no value. Masks use ordinary SDPA
    head axes, including batch-specific left padding and per-head masks.
    """
    batch, heads, length, dim = queries.shape
    kv_heads = q_keys[0].shape[-3]
    repeats = heads // kv_heads
    # Prefix views may retain capacity strides unsupported by packed kernels.
    q_keys = tuple(mx.contiguous(x) for x in q_keys)
    q_values = tuple(mx.contiguous(x) for x in q_values)
    q = queries * scale
    if repeats > 1:
        q = q.reshape(batch, kv_heads, repeats, length, dim)
        q_keys = tuple(mx.expand_dims(x, -3) for x in q_keys)
        q_values = tuple(mx.expand_dims(x, -3) for x in q_values)

    scores = mx.quantized_matmul(
        q, *q_keys, transpose=True, group_size=group_size, bits=bits
    ).reshape(batch, heads, length, -1)
    if mask is not None:
        if isinstance(mask, str):
            q_length, k_length = scores.shape[-2:]
            mask = (
                mx.arange(k_length - q_length, k_length)[:, None]
                >= mx.arange(k_length)[None, :]
            )
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    scores = scores.astype(mx.float32)
    sink_column = mx.broadcast_to(
        sinks.astype(mx.float32).reshape(1, heads, 1, 1),
        (batch, heads, length, 1),
    )
    probabilities = mx.softmax(
        mx.concatenate([scores, sink_column], axis=-1), axis=-1, precise=True
    )[..., :-1].astype(queries.dtype)
    if repeats > 1:
        probabilities = probabilities.reshape(batch, kv_heads, repeats, length, -1)
    return mx.quantized_matmul(
        probabilities,
        *q_values,
        transpose=False,
        group_size=group_size,
        bits=bits,
    ).reshape(batch, heads, length, -1)
