"""Independent-position decode. All weight projections run once for the batch.

Host position metadata controls shapes only. Device row indices and masks drive
ring writes, compression commits and sparse visibility without layer readbacks.
The chunk-general attention and compression implementations remain unchanged.
"""

import mlx.core as mx

from .fakequant import (
    fake_quant_fp4_e4m3,
    fake_quant_fp4_ue8m0,
    fake_quant_fp8_ue8m0,
)
from .layers import rope_tail
from .sparse_attention import sparse_attn

NEG_INF = float("-inf")


def row_rope(x, rd, cos, sin, positions, inverse=False):
    """Flatten row/token axes for the existing adjacent-pair RoPE operator."""
    flat = x.reshape(1, x.shape[0], *x.shape[2:])
    return rope_tail(flat, rd, cos[positions], sin[positions], inverse).reshape(x.shape)


def compressor_decode(compressor, x, positions, state):
    ratio = compressor.ratio
    if ratio == 1:
        return compressor.norm(compressor.wkv(x))
    rows = mx.arange(x.shape[0])
    slots = positions % ratio
    kv = compressor.wkv(x.astype(mx.float32))
    score = compressor.wgate(x.astype(mx.float32))
    # Completing groups must leave the saved partial buffer untouched, matching
    # single-stream state (including its unused, stale slots).
    complete = (slots == ratio - 1)[:, None, None]
    group_kv = mx.array(state.kv_state)
    group_score = mx.array(state.score_state)
    group_kv[rows, slots] = kv[:, 0]
    group_score[rows, slots] = score[:, 0]
    state.kv_state = mx.where(complete, state.kv_state, group_kv)
    state.score_state = mx.where(complete, state.score_state, group_score)
    pooled = mx.sum(group_kv * mx.softmax(group_score, axis=1), axis=1, keepdims=True)
    return compressor.norm(pooled.astype(x.dtype))


def commit(buffer, values, positions, ratio):
    """Scatter one latent per row, committing only rows closing a group."""
    rows = mx.arange(values.shape[0])
    groups = positions // ratio
    # A non-emitting row can already point one past the allocated last group.
    slots = mx.minimum(groups, buffer.shape[1] - 1)
    emit = ((positions + 1) % ratio == 0)[:, None]
    buffer[rows, slots] = mx.where(
        emit, values[:, 0].astype(buffer.dtype), buffer[rows, slots]
    )


def candidate_blocks(scores, counts, topk, size):
    width = scores.shape[-1]
    pad = (-width) % size
    padded = mx.pad(scores, ((0, 0), (0, 0), (0, pad)), constant_values=NEG_INF)
    blocks = padded.reshape(*scores.shape[:-1], -1, size).max(axis=-1)
    newest = (counts - 1) // size
    pin = mx.arange(blocks.shape[-1])[None, None, :] == newest[:, None, None]
    blocks = mx.where(pin & (counts[:, None, None] > 0), float("inf"), blocks)
    k = min(topk, blocks.shape[-1])
    selected = mx.argpartition(-blocks, k - 1, axis=-1)[..., :k]
    values = mx.take_along_axis(blocks, selected, axis=-1)
    keep = mx.put_along_axis(
        mx.zeros(blocks.shape, dtype=mx.bool_), selected, values > NEG_INF, axis=-1
    )
    return mx.repeat(keep, size, axis=-1)[..., :width]


def indexer_decode(indexer, x, qr, positions, offset, cos, sin, keys, counts, shared):
    b = x.shape[0]
    q = indexer.wq_b(qr).reshape(b, 1, indexer.n_heads, indexer.head_dim)
    q = row_rope(q, indexer.rope_head_dim, cos, sin, positions)
    q = fake_quant_fp4_ue8m0(q, 32)
    weights = indexer.weights_proj(x) * (indexer.softmax_scale * indexer.n_heads**-0.5)
    scores = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), keys.astype(mx.float32))
    scores = mx.sum(
        mx.maximum(scores, 0.0) * weights[..., None].astype(mx.float32), axis=2
    )
    scores = mx.where(
        mx.arange(keys.shape[1])[None, None, :] < counts[:, None, None], scores, NEG_INF
    )
    if indexer.is_candidate_source:
        shared.candidates = candidate_blocks(
            scores, counts, indexer.candidate_topk_blocks, indexer.candidate_block_size
        )
    elif indexer.uses_candidates and shared.candidates is not None:
        scores = mx.where(shared.candidates, scores, NEG_INF)
    k = min(indexer.index_topk, keys.shape[1])
    idx = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
    idx = mx.sort(idx, axis=-1)
    return mx.where(idx < counts[:, None, None], idx + offset, mx.array(-1, mx.int32))


def attention_decode(attn, x, cache, shared):
    b = x.shape[0]
    positions = cache.positions
    max_end = int(cache.offsets.max()) + 1
    cos, sin = attn._freqs(max_end)
    rd, width = attn.rope_head_dim, attn.window_size
    rows = mx.arange(b)
    lc = cache.layers[attn.layer_id]
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).reshape(b, 1, attn.n_heads, attn.head_dim)
    q = row_rope(q, rd, cos, sin, positions)
    kv = row_rope(attn.kv_norm(attn.wkv(x)), rd, cos, sin, positions)
    kv = fake_quant_fp8_ue8m0(kv, 32)

    wp = mx.minimum(positions, width)
    slots = (positions[:, None] - wp[:, None] + mx.arange(width)[None]) % width
    prev = lc.win_kv[rows[:, None], slots]
    kv_all = mx.concatenate([prev.astype(kv.dtype), kv], axis=1)
    idx = mx.maximum(wp - width + 1, 0)[:, None] + mx.arange(width)[None]
    idx = mx.where(idx < wp[:, None], idx, mx.where(idx == wp[:, None], width, -1))
    idxs = idx[:, None, :].astype(mx.int32)
    lc.win_kv[rows, positions % width] = kv[:, 0].astype(lc.dtype)
    offset = width + 1

    if attn.ratio:
        ratio = attn.ratio
        count = (positions + 1) // ratio
        nb = max_end // ratio
        latent = None
        if attn.is_kv_source:
            latent = compressor_decode(attn.compressor, x, positions, lc.comp_state)
            shared.kv_src_cache = lc
            lc.comp_counts = count
        group_positions = (positions // ratio) * ratio
        if attn.is_index_source:
            if attn.indexer.owns_k:
                key = attn.indexer.k_norm(attn.indexer.wk(latent))
                key = row_rope(key, rd, cos, sin, group_positions)
                key = fake_quant_fp4_ue8m0(key, 32)
                if lc.index_k.shape[1]:
                    commit(lc.index_k, key, positions, ratio)
                shared.index_src_cache = lc
            cidx = (
                indexer_decode(
                    attn.indexer,
                    x,
                    qr,
                    positions,
                    offset,
                    cos,
                    sin,
                    shared.index_src_cache.index_k[:, :nb],
                    count,
                    shared,
                )
                if nb
                else mx.zeros((b, 1, 0), dtype=mx.int32)
            )
            shared.topk_idxs = cidx
        else:
            cidx = shared.topk_idxs
        if latent is not None and lc.comp_kv.shape[1]:
            latent = row_rope(latent, rd, cos, sin, group_positions)
            latent = fake_quant_fp4_e4m3(latent, 16)
            commit(lc.comp_kv, latent, positions, ratio)
        if nb:
            kv_all = mx.concatenate(
                [kv_all, shared.kv_src_cache.comp_kv[:, :nb].astype(kv.dtype)], axis=1
            )
            idxs = mx.concatenate([idxs, cidx], axis=-1)

    sink = mx.zeros_like(attn.attn_sink) if attn._break_sink else attn.attn_sink
    output = sparse_attn(q, kv_all, sink, idxs, attn.softmax_scale)
    if not attn._break_rope_inverse:
        output = row_rope(output, rd, cos, sin, positions, inverse=True)
    output = output.reshape(b, 1, attn.n_groups, -1)
    output = attn.wo_a(output.astype(mx.float32)[..., None, :]).squeeze(-2)
    return attn.wo_b(output.reshape(b, 1, -1).astype(x.dtype))
