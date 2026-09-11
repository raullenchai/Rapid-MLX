"""Learned KV compression — V4.1 pools ``ratio`` consecutive tokens into one latent.

Only the four ``kv_source_layers`` own a compressor; every other compressing
layer reads the source's cache. Two regimes:

* ``ratio > 1`` (layers 2..19, ratio 2): a learned softmax gate — ``wgate``
  scores each position, scores softmax across the group, KV vectors summed under
  those weights. Runs in fp32 (the checkpoint stores these projections wider).
  No overlap and no ``ape`` slot embedding — both were V4 features that V4.1
  dropped.
* ``ratio == 1`` (layer 20, feeding 20..39): a plain per-token projection —
  no gate, no pooling state, checkpoint-dtype compute.

The returned latent is **pre-RoPE and pre-quantization**: the indexer derives
its keys from exactly this form, so Attention applies the rope tail and the FP4
fake-quant afterwards, before writing the shared cache.

This implementation is chunk-general: a chunk starting at ``start_pos`` first
completes the carried partial group (``kv_state`` / ``score_state`` in the layer
cache), emits every group the chunk closes, and saves the new remainder. The
reference only implements full prefill and 1-token decode; both are special
cases of this path.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .layers import RMSNorm

NEG_INF = float("-inf")


class Compressor(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.ratio = args.compress_ratio(layer_id)
        self.head_dim = args.head_dim
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wkv = nn.Linear(args.dim, args.head_dim, bias=False)
        if self.ratio > 1:
            self.wgate = nn.Linear(args.dim, args.head_dim, bias=False)

    def __call__(self, x: mx.array, start_pos: int, comp_state) -> mx.array | None:
        """x [b, n, dim] at absolute positions [start_pos, start_pos+n).

        Returns the pre-RoPE latents [b, g, head_dim] for the ``g`` groups this
        chunk completes (group indices start_pos//ratio ..), or None when no
        group completes. ``comp_state`` carries the partial group across calls
        (unused at ratio 1).
        """
        ratio, dtype = self.ratio, x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))

        bsz, n, _ = x.shape
        xf = x.astype(mx.float32)
        # Call the modules rather than reading their physical weight arrays.
        # The 256 GB build retains these projections in affine 2-bit form;
        # direct matmul would interpret packed uint32 columns as logical input
        # features. The module call is identical for dense public builds.
        kv = self.wkv(xf)
        score = self.wgate(xf)

        m = start_pos % ratio  # carried tokens of the open group
        if m:
            kv = mx.concatenate([comp_state.kv_state[:bsz, :m], kv], axis=1)
            score = mx.concatenate([comp_state.score_state[:bsz, :m], score], axis=1)
        total = m + n
        g = total // ratio
        rem = total % ratio
        if rem:
            comp_state.kv_state[:bsz, :rem] = kv[:, total - rem :]
            comp_state.score_state[:bsz, :rem] = score[:, total - rem :]
            kv, score = kv[:, : total - rem], score[:, : total - rem]
        if g == 0:
            return None
        kv = kv.reshape(bsz, g, ratio, -1)
        score = score.reshape(bsz, g, ratio, -1)
        pooled = mx.sum(kv * mx.softmax(score, axis=2), axis=2)
        return self.norm(pooled.astype(dtype))


class CompressorState:
    """The open group's per-token kv/score rows, fp32 (mirrors the reference's
    ``kv_state`` / ``score_state`` buffers)."""

    def __init__(self, bsz: int, ratio: int, head_dim: int):
        self.kv_state = mx.zeros((bsz, ratio, head_dim), dtype=mx.float32)
        self.score_state = mx.full((bsz, ratio, head_dim), NEG_INF, dtype=mx.float32)
