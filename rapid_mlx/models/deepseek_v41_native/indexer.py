"""The sparse-attention indexer — a second attention deciding what the first reads.

The eight ``index_source_layers`` own one. Only the four that are *also*
``kv_source_layers`` (2, 8, 14, 20) own index **keys** (``wk``/``k_norm`` + a key
cache): keys are derived from the compressor's pre-RoPE latent. Index sources
that are not kv sources (24, 28, 32, 36) score with their own ``wq_b`` /
``weights_proj`` against the *most recent owner's* key cache (layer 20's).

Two-level selection: layer 20 is the candidate source — it scores all latents,
keeps the ``candidate_topk_blocks`` best blocks of ``candidate_block_size`` (the
block holding the newest position is pinned in), and publishes the boolean mask.
Sources after it (24..36) mask their own scores with those candidates before the
final per-query top-``index_topk``.

Queries and keys are FP4 fake-quantized (blocks of 32, power-of-two scales), no
Hadamard rotation (a V4 feature V4.1 dropped). Scores are ReLU'd, then collapsed
over heads by ``weights_proj(x) * head_dim**-0.5 * n_heads**-0.5``.

Reference artifact worth knowing (documented in docs/upstream-notes.md): the
reference reads keys through a process-global pointer that the *last* owner set.
During decode, a ratio-2 owner whose group is incomplete therefore scores
against layer 20's keys. This port always reads the owner's own cache.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .fakequant import fake_quant_fp4_ue8m0
from .layers import RMSNorm, rope_tail

NEG_INF = float("-inf")
POS_INF = float("inf")


def select_candidate_blocks(
    scores: mx.array, lens: mx.array, topk_blocks: int, block_size: int
) -> mx.array:
    """Level one of the two-level top-k. scores [b, n, nb] with unreachable
    positions already at -inf; lens [n, 1] (positions visible per query).
    Returns a bool mask shaped like scores."""
    width = scores.shape[-1]
    pad = (-width) % block_size
    if pad:
        scores_p = mx.concatenate(
            [scores, mx.full((*scores.shape[:-1], pad), NEG_INF, dtype=scores.dtype)],
            axis=-1,
        )
    else:
        scores_p = scores
    blocks = scores_p.reshape(*scores.shape[:-1], -1, block_size).max(
        axis=-1
    )  # [b, n, NB]
    nb_blocks = blocks.shape[-1]

    # pin the block holding each query's newest position: it is only partly
    # filled and could otherwise be outscored by an older, full block
    last = (lens - 1) // block_size  # [n, 1]
    pin = mx.arange(nb_blocks)[None, :] == last  # [n, NB]
    blocks = mx.where(pin[None], POS_INF, blocks)

    k = min(topk_blocks, nb_blocks)
    top_idx = mx.argpartition(-blocks, k - 1, axis=-1)[..., :k]
    top_val = mx.take_along_axis(blocks, top_idx, axis=-1)
    keep = mx.zeros(blocks.shape, dtype=mx.bool_)
    keep = mx.put_along_axis(keep, top_idx, top_val > NEG_INF, axis=-1)
    return mx.repeat(keep, block_size, axis=-1)[..., :width]


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layers
        self.ratio = args.compress_ratio(layer_id)
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(args.dim, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(args.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, args.norm_eps)

    def publish_keys(self, latents: mx.array, start_pos: int, cos, sin, cache):
        """Turn this chunk's pre-RoPE latents into index keys and cache them.

        Must run before Attention overwrites the latents with their RoPE'd,
        quantized form. ``latents`` [b, g, head_dim] for groups g0.., where
        g0 = start_pos // ratio; a latent's rope position is its group's first
        token, g*ratio.
        """
        rd = self.rope_head_dim
        g0 = start_pos // self.ratio
        g = latents.shape[1]
        pos = (g0 + mx.arange(g)) * self.ratio
        k = self.k_norm(self.wk(latents))
        k = rope_tail(k, rd, cos[pos], sin[pos])
        k = fake_quant_fp4_ue8m0(k, 32)
        cache.index_k[: k.shape[0], g0 : g0 + g] = k

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        start_pos: int,
        offset: int,
        cos,
        sin,
        index_k: mx.array,
        shared,
    ) -> mx.array:
        """Score and pick top-k compressed positions for each query.

        x [b, n, dim] (post-attn-norm), qr [b, n, q_lora_rank],
        index_k [b, nb, head_dim] — the owner's key cache, already sliced to the
        nb = (start_pos+n)//ratio complete groups. Returns [b, n, k] int32
        indices into the concatenated window+compressed KV, -1 = masked.
        """
        bsz, n, _ = x.shape
        ratio, rd = self.ratio, self.rope_head_dim
        nb = index_k.shape[1]

        q = self.wq_b(qr).reshape(bsz, n, self.n_heads, self.head_dim)
        q = rope_tail(
            q, rd, cos[start_pos : start_pos + n], sin[start_pos : start_pos + n]
        )
        q = fake_quant_fp4_ue8m0(q, 32)

        w = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        scores = mx.einsum(
            "bshd,btd->bsht", q.astype(mx.float32), index_k.astype(mx.float32)
        )
        scores = mx.maximum(scores, 0.0) * w[..., None].astype(mx.float32)
        scores = mx.sum(scores, axis=2)  # [b, n, nb]

        # visibility: group j is visible to query i once the query passed its last token
        lens = ((start_pos + mx.arange(n) + 1) // ratio)[:, None]  # [n, 1]
        vis = mx.arange(nb)[None, :] < lens  # [n, nb]
        scores = mx.where(vis[None], scores, NEG_INF)

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                scores, lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates and shared.candidates is not None:
            scores = mx.where(shared.candidates, scores, NEG_INF)

        k = min(self.index_topk, nb)
        idx = mx.argpartition(-scores, k - 1, axis=-1)[..., :k].astype(mx.int32)
        idx = mx.sort(idx, axis=-1)  # position order
        visible = idx < lens.astype(mx.int32)[None]
        return mx.where(visible, idx + offset, mx.array(-1, mx.int32))
