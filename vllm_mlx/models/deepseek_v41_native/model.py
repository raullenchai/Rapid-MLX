"""DeepSeek-V4.1 block and full model (text stack).

The block wires Hyper-Connections in V4.1's *staggered* form: the coefficient
set a sub-layer computes is consumed by the **next** sub-layer's collapse.
Attention collapses with the previous layer's FFN ``pre`` (identity one-hot at
the start), the FFN collapses with this layer's attention ``pre``, and the LM
head collapses with the last layer's FFN ``pre``.

Cross-layer sharing flows through a per-forward :class:`SharedState`, mirroring
the reference's process-global ``SharedAttentionRuntime``: kv sources publish
their compressed-KV cache and index-key cache, index sources publish their
top-k selection, the candidate source publishes its block mask, and every layer
in between consumes the most recent value. Layers run top-down, so every source
writes before its consumers read.

Engram layers apply their gated n-gram lookup to the hc-expanded stream
*before* the block runs, exactly as the reference does in ``Transformer.forward``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import Attention
from .cache import ModelCache
from .config import ModelArgs
from .engram import Engram, EngramHasher
from .hyper_connections import hc_mixes, hc_post, hc_pre, make_identity_pre_mix
from .layers import RMSNorm
from .moe import MoE


class SharedState:
    """What attention layers hand down the stack instead of recomputing.

    Fresh per forward; every source writes before its consumers read."""

    def __init__(self):
        self.kv_src_cache = None  # LayerCache of the most recent kv source
        self.index_src_cache = None  # LayerCache of the most recent index-key owner
        self.topk_idxs = None  # [b, n, k] from the most recent index source
        self.candidates = None  # [b, n, nb] bool from the candidate source


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps

        self.attn = Attention(layer_id, args)
        self.ffn = MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.hc_attn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_ffn_base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.hc_attn_scale = mx.zeros((3,), dtype=mx.float32)
        self.hc_ffn_scale = mx.zeros((3,), dtype=mx.float32)

        self.engram: Engram | None
        if layer_id in args.engram_layer_ids:
            self.engram = Engram(args, args.engram_layer_ids.index(layer_id))
        else:
            self.engram = None

    def __call__(self, x: mx.array, pre_mix: mx.array, start_pos: int, cache, shared):
        """x [b, s, hc, d]; pre_mix [b, s, hc] from the previous sub-layer.
        Returns (x, ffn_pre) — ffn_pre feeds the next layer (or the head)."""
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_iters,
            self.norm_eps,
            self.hc_eps,
        )
        h = hc_pre(x, pre_mix)
        h = self.attn(self.attn_norm(h), start_pos, cache, shared)
        x = hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_iters,
            self.norm_eps,
            self.hc_eps,
        )
        h = hc_pre(x, attn_pre)
        h = self.ffn(self.ffn_norm(h))
        x = hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class Model(nn.Module):
    """embed -> expand to hc copies -> blocks -> collapse (last ffn_pre) -> logits."""

    def __init__(self, args: ModelArgs, token_map=None):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [Block(i, args) for i in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        # A 40-layer V4.1 decode graph can exceed Metal's command-buffer
        # deadline on a 256 GiB host. Keep a correctness-first per-layer
        # boundary; qualification may raise this interval after proving the
        # larger graph stays below the watchdog.
        self.eval_interval = 1
        self.engram_hasher = None
        if args.engram_layer_ids and token_map is not None:
            self.set_token_map(token_map)
        # test hook: remap which source each consumer reads (negative control)
        self._break_sharing = False

    def set_token_map(self, token_map):
        self.engram_hasher = EngramHasher(self.args, token_map)

    def make_cache(
        self, bsz: int = 1, max_seq_len: int | None = None, dtype=mx.float32
    ) -> ModelCache:
        return ModelCache(self.args, bsz, max_seq_len, dtype)

    def __call__(
        self,
        input_ids: mx.array,
        cache: ModelCache,
        last_logit_only: bool = False,
        return_dspark_hidden: bool = False,
        enable_rollback: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """input_ids [b, n] continue the sequence at cache.offset. Advances the cache."""
        start_pos = cache.offset
        if enable_rollback:
            cache.begin_forward()
        elif cache.rollback_start is not None:
            cache.disable_rollback()
        b, n = input_ids.shape

        hashes = None
        if self.engram_hasher is not None:
            ids_np = np.array(input_ids, dtype=np.int64)
            hashes = self.engram_hasher(ids_np, start_pos, cache.engram_ids)
            hashes = mx.array(hashes)  # [b, n, n_engram_layers, cols]
        elif self.args.engram_layer_ids:
            raise RuntimeError(
                "model has engram layers but no token map — call "
                "set_token_map() (load.py builds it from the release tokenizer)"
            )

        h = self.embed(input_ids)
        h = mx.broadcast_to(h[:, :, None, :], (b, n, self.hc_mult, h.shape[-1]))

        pre_mix = make_identity_pre_mix(b, n, self.hc_mult)
        shared = SharedState()
        dspark_hiddens = {}
        for execution_index, layer in enumerate(self.layers, 1):
            layer_id = execution_index - 1
            if return_dspark_hidden and layer_id in self.args.dspark_target_layer_ids:
                dspark_hiddens[layer_id] = mx.mean(h, axis=2)
            if layer.engram is not None:
                assert hashes is not None
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index])
            if self._break_sharing and not layer.attn.is_kv_source and layer.attn.ratio:
                shared_use = SharedState()  # sever the link: consumers see nothing
                shared_use.kv_src_cache = shared.kv_src_cache
                shared_use.index_src_cache = shared.index_src_cache
                zero = (
                    mx.full(shared.topk_idxs.shape, -1, dtype=mx.int32)
                    if shared.topk_idxs is not None
                    else None
                )
                shared_use.topk_idxs = zero
                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared_use)
            else:
                h, pre_mix = layer(h, pre_mix, start_pos, cache, shared)
            if self.eval_interval and execution_index % self.eval_interval == 0:
                mx.eval(h, pre_mix)

        h = hc_pre(h, pre_mix)  # collapse with the last ffn_pre
        h = self.norm(h)
        if last_logit_only:
            h = h[:, -1:]
        logits = self.head(h.astype(mx.float32))  # fp32 logits, as the reference
        cache.offset = start_pos + n
        if return_dspark_hidden:
            missing = [
                layer_id
                for layer_id in self.args.dspark_target_layer_ids
                if layer_id not in dspark_hiddens
            ]
            if missing:
                raise RuntimeError("DSpark target hidden-state capture is incomplete")
            ordered = [
                dspark_hiddens[layer_id]
                for layer_id in self.args.dspark_target_layer_ids
            ]
            return logits, mx.concatenate(ordered, axis=-1)
        return logits
