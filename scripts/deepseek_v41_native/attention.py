"""V4.1 attention — MLA over a sliding window plus shared compressed positions.

One layer:

* queries through a low-rank bottleneck (``wq_a`` -> ``q_norm`` -> ``wq_b``),
  rope on the last 64 channels (no per-head q normalization — that was V4);
* a **single** shared 512-d KV vector per position (key and value are the same
  tensor), FP8 fake-quantized whole — rope tail included — before caching;
* every layer attends over the last ``window_size`` tokens; layers with
  ``compress_ratio > 0`` additionally attend over ``index_topk`` compressed
  positions. Only kv_source layers *produce* the compressed cache and only
  index_source layers *select* — everything else consumes through the per-forward
  ``shared`` state, exactly mirroring the reference's SharedAttentionRuntime;
* the attention output gets the rope **inverse** applied to its tail, then the
  grouped block-diagonal ``wo_a`` and the shared ``wo_b``. ``wo_a`` stays
  group-aware so affine packed weights can be used without reshaping bytes.

Rope: compressing layers use YaRN on ``compress_rope_theta``; pure sliding-window
layers use the base theta with YaRN off. One freqs table serves the queries, the
window KV, the compressed latents (at each group's first-token position), the
indexer, and the inverse on the way out.

The forward is chunk-general: full prefill, chunked prefill and 1-token decode
are the same code path.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from vllm_mlx import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.mla import MultiLinear  # noqa: E402

from .compressor import Compressor
from .config import ModelArgs
from .fakequant import fake_quant_fp4_e4m3, fake_quant_fp8_ue8m0
from .indexer import Indexer
from .layers import RMSNorm, precompute_freqs_cis, rope_tail
from .sparse_attention import sparse_attn


class GroupedOutputLinear(MultiLinear):
    """Grouped projection accepting logical and release-flat float weights."""

    def __init__(self, input_dims: int, output_dims: int, num_heads: int):
        super().__init__(input_dims, output_dims, num_heads)
        self.output_dims = output_dims
        self.num_heads = num_heads

    def __call__(self, x, transpose=True):
        weight = self.weight
        if weight.ndim == 2:
            weight = weight.reshape(self.num_heads, self.output_dims, -1)
            return x @ (weight.swapaxes(-1, -2) if transpose else weight)
        return super().__call__(x, transpose=transpose)


def window_idx_matrix(wp: int, n: int, window: int) -> mx.array:
    """[n, W_eff] window indices into concat([prev_window(wp), chunk(n)]).

    Query i (concat end position wp+i) sees the last ``window`` concat positions
    that exist; -1 pads. W_eff = min(window, wp+n).
    """
    w_eff = min(window, wp + n)
    end = mx.arange(n) + wp
    start = mx.maximum(end - window + 1, 0)
    idx = start[:, None] + mx.arange(w_eff)[None, :]
    return mx.where(idx > end[:, None], mx.array(-1, mx.int32), idx.astype(mx.int32))


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.ratio = args.compress_ratio(layer_id)
        self.eps = args.norm_eps
        self.softmax_scale = args.head_dim**-0.5

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # Modified by Rapid-MLX: retain the logical group axis so 2-bit affine
        # weights can be loaded without interpreting packed uint32 storage as
        # a dense matrix.
        self.wo_a = GroupedOutputLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.o_lora_rank,
            self.n_groups,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.is_kv_source = layer_id in args.kv_source_layers
        self.is_index_source = layer_id in args.index_source_layers
        if self.is_kv_source:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id)

        if self.ratio:
            orig_len, theta = args.original_seq_len, args.compress_rope_theta
        else:
            orig_len, theta = 0, args.rope_theta
        self._rope = (
            args.rope_head_dim,
            orig_len,
            theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self._cos = None
        self._sin = None

        # test hooks (negative controls)
        self._break_rope_inverse = False
        self._break_sink = False

    def _freqs(self, upto: int):
        if self._cos is None or self._cos.shape[0] < upto:
            rd, orig_len, theta, factor, bf, bs = self._rope
            self._cos, self._sin = precompute_freqs_cis(
                rd, max(upto, 64), orig_len, theta, factor, bf, bs
            )
        return self._cos, self._sin

    def __call__(self, x: mx.array, start_pos: int, cache, shared) -> mx.array:
        """x [b, n, dim] (post attn_norm), absolute positions [start_pos, start_pos+n)."""
        bsz, n, _ = x.shape
        rd = self.rope_head_dim
        end_pos = start_pos + n
        cos, sin = self._freqs(end_pos)
        c_q, s_q = cos[start_pos:end_pos], sin[start_pos:end_pos]

        # --- queries ---
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(bsz, n, self.n_heads, self.head_dim)
        q = rope_tail(q, rd, c_q, s_q)

        # --- window KV: rope tail, FP8 fake-quant over the whole vector ---
        kv = self.kv_norm(self.wkv(x))
        kv = rope_tail(kv, rd, c_q, s_q)
        kv = fake_quant_fp8_ue8m0(kv, 32)

        lc = cache.layers[self.layer_id]
        prev = lc.window_chrono(start_pos)  # [b, Wp, hd]
        wp = prev.shape[1]
        kv_all = mx.concatenate([prev.astype(kv.dtype), kv], axis=1) if wp else kv
        idxs = mx.broadcast_to(
            window_idx_matrix(wp, n, self.window_size)[None],
            (bsz, n, min(self.window_size, wp + n)),
        )
        lc.write_window(start_pos, kv)
        offset = wp + n  # compressed entries follow

        if self.ratio:
            src = shared.kv_src_cache  # the source layer's LayerCache
            compress_len = end_pos // self.ratio

            latents = None
            if self.is_kv_source:
                latents = self.compressor(x, start_pos, lc.comp_state)
                shared.kv_src_cache = src = lc

            # indexer runs on the pre-RoPE latents, before the cache write
            if self.is_index_source:
                if self.indexer.owns_k:
                    if latents is not None:
                        self.indexer.publish_keys(latents, start_pos, cos, sin, lc)
                    shared.index_src_cache = lc  # an owner always reads its own keys
                if compress_len == 0:
                    cidx = mx.zeros((bsz, n, 0), dtype=mx.int32)
                else:
                    index_k = shared.index_src_cache.index_k[:bsz, :compress_len]
                    cidx = self.indexer(
                        x, qr, start_pos, offset, cos, sin, index_k, shared
                    )
                shared.topk_idxs = cidx
            else:
                cidx = shared.topk_idxs

            if latents is not None:
                g0 = start_pos // self.ratio
                g = latents.shape[1]
                pos = (g0 + mx.arange(g)) * self.ratio  # group j at position j*ratio
                latents = rope_tail(latents, rd, cos[pos], sin[pos])
                latents = fake_quant_fp4_e4m3(latents, 16)
                lc.comp_kv[:bsz, g0 : g0 + g] = latents.astype(lc.dtype)

            if compress_len:
                comp = src.comp_kv[:bsz, :compress_len].astype(kv.dtype)
                kv_all = mx.concatenate([kv_all, comp], axis=1)
                idxs = mx.concatenate([idxs, cidx], axis=-1)

        sink = mx.zeros_like(self.attn_sink) if self._break_sink else self.attn_sink
        o = sparse_attn(q, kv_all, sink, idxs, self.softmax_scale)

        # --- inverse rope, grouped block-diagonal output LoRA ---
        if not self._break_rope_inverse:
            o = rope_tail(o, rd, c_q, s_q, inverse=True)
        o = o.reshape(bsz, n, self.n_groups, -1)
        # QuantizedMultiLinear treats the penultimate axis as matrix rows;
        # retain a singleton row so the group axis broadcasts against the
        # leading weight axis instead of being consumed as M.
        o = self.wo_a(o.astype(mx.float32)[..., None, :]).squeeze(-2)
        return self.wo_b(o.reshape(bsz, n, -1).astype(x.dtype))
