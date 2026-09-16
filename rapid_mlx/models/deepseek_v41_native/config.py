"""Configuration for DeepSeek-V4.1-Flash (MLX port).

Accepts both config layouts that ship with the release:

* the inference-style flat ``config.json`` (``dim``, ``n_layers``,
  ``kv_source_layers`` ...), and
* the HF hub ``config.json`` with nested ``text_config`` (``hidden_size``,
  ``num_hidden_layers``, ``kv_source_layer_ids`` ...).

``compress_ratios`` in the release has ``n_layers + n_mtp_layers`` entries (43 =
40 + 3); only the first ``n_layers`` matter for this text-only runtime, but the
full tuple is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


@dataclass
class ModelArgs:
    vocab_size: int = 129280
    dim: int = 5120
    n_layers: int = 40
    moe_inter_dim: int = 2304

    # attention (MLA): one shared 512-d KV vector per position
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    norm_eps: float = 1e-20

    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # KV compression / cross-layer sharing.
    # compress_ratios: one per layer (0 = pure sliding window, 1 = one latent per
    # token, r>1 = r tokens pooled per latent). Only kv_source_layers own a
    # compressor + compressed-KV cache; every later layer with the same ratio
    # reads that cache. index_source_layers own an indexer (top-k selection);
    # layers in between reuse the most recent source's selection.
    compress_ratios: tuple[int, ...] = ()
    kv_source_layers: tuple[int, ...] = ()
    index_source_layers: tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0

    # candidate pre-filtering (two-level top-k); -1 disables
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0

    # rope / YaRN
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    max_seq_len: int = 1048576

    # sparse-attention indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # engram: n-gram hash tables added into the residual stream
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0

    # side-paths dropped for inference (kept for provenance)
    n_mtp_layers: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    vision_n_layers: int = 0

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    def compress_ratio(self, layer_id: int) -> int:
        if layer_id < len(self.compress_ratios):
            return int(self.compress_ratios[layer_id])
        return 0

    def kv_source_for(self, layer_id: int) -> int:
        """The layer whose compressed-KV cache layer ``layer_id`` reads.

        The reference threads a single mutable slot down the layer stack, so a
        consumer sees whatever the most recent source above it published.
        """
        srcs = [s for s in self.kv_source_layers if s <= layer_id]
        if not srcs:
            raise ValueError(
                f"layer {layer_id} compresses but has no kv source above it"
            )
        src = max(srcs)
        if self.compress_ratio(src) != self.compress_ratio(layer_id):
            raise ValueError(
                f"layer {layer_id} (ratio {self.compress_ratio(layer_id)}) would read "
                f"source {src} (ratio {self.compress_ratio(src)})"
            )
        return src

    def index_source_for(self, layer_id: int) -> int:
        srcs = [s for s in self.index_source_layers if s <= layer_id]
        if not srcs:
            raise ValueError(f"layer {layer_id} has no index source above it")
        return max(srcs)

    @classmethod
    def from_dict(cls, c: dict) -> ModelArgs:
        if "text_config" in c:  # HF hub layout: fold text_config over the top level
            top = dict(c)
            tc = top.pop("text_config")
            c = {**top, **tc}
        rope = _get(c, "rope_scaling", default={}) or {}
        return cls(
            vocab_size=_get(c, "vocab_size", default=129280),
            dim=_get(c, "hidden_size", "dim", default=5120),
            n_layers=_get(c, "num_hidden_layers", "n_layers", default=40),
            moe_inter_dim=_get(
                c, "moe_intermediate_size", "moe_inter_dim", default=2304
            ),
            n_heads=_get(c, "num_attention_heads", "n_heads", default=64),
            head_dim=_get(c, "head_dim", default=512),
            rope_head_dim=_get(c, "qk_rope_head_dim", "rope_head_dim", default=64),
            q_lora_rank=_get(c, "q_lora_rank", default=1280),
            o_lora_rank=_get(c, "o_lora_rank", default=1024),
            o_groups=_get(c, "o_groups", default=8),
            window_size=_get(c, "sliding_window", "window_size", default=128),
            norm_eps=_get(c, "rms_norm_eps", "norm_eps", default=1e-20),
            n_routed_experts=_get(c, "n_routed_experts", default=384),
            n_shared_experts=_get(c, "n_shared_experts", default=1),
            n_activated_experts=_get(
                c, "num_experts_per_tok", "n_activated_experts", default=6
            ),
            score_func=_get(c, "scoring_func", "score_func", default="sqrtsoftplus"),
            gate_temp=_get(c, "gate_temp", default=1.0),
            norm_topk_prob=_get(c, "norm_topk_prob", default=True),
            route_scale=_get(c, "routed_scaling_factor", "route_scale", default=1.5),
            swiglu_limit=_get(c, "swiglu_limit", default=0.0),
            compress_ratios=tuple(_get(c, "compress_ratios", default=()) or ()),
            kv_source_layers=tuple(
                _get(c, "kv_source_layer_ids", "kv_source_layers", default=()) or ()
            ),
            index_source_layers=tuple(
                _get(c, "index_source_layer_ids", "index_source_layers", default=())
                or ()
            ),
            compress_rope_theta=_get(c, "compress_rope_theta", default=160000.0),
            candidate_source_layer=_get(
                c, "candidate_source_layer_id", "candidate_source_layer", default=-1
            ),
            candidate_topk_blocks=_get(c, "candidate_topk_blocks", default=0),
            candidate_block_size=_get(c, "candidate_block_size", default=0),
            original_seq_len=_get(
                rope,
                "original_max_position_embeddings",
                default=_get(c, "original_seq_len", default=65536),
            ),
            rope_theta=_get(c, "rope_theta", default=10000.0),
            rope_factor=_get(
                rope, "factor", default=_get(c, "rope_factor", default=16.0)
            ),
            beta_fast=_get(rope, "beta_fast", default=_get(c, "beta_fast", default=32)),
            beta_slow=_get(rope, "beta_slow", default=_get(c, "beta_slow", default=1)),
            max_seq_len=_get(
                c, "max_position_embeddings", "max_seq_len", default=1048576
            ),
            index_n_heads=_get(c, "index_n_heads", default=32),
            index_head_dim=_get(c, "index_head_dim", default=128),
            index_topk=_get(c, "index_topk", default=512),
            hc_mult=_get(c, "hc_mult", default=4),
            hc_sinkhorn_iters=_get(c, "hc_sinkhorn_iters", default=20),
            hc_eps=_get(c, "hc_eps", default=1e-6),
            engram_layer_ids=tuple(_get(c, "engram_layer_ids", default=()) or ()),
            engram_num_embeddings=tuple(
                _get(c, "engram_num_embeddings", default=()) or ()
            ),
            engram_max_ngram_size=_get(c, "engram_max_ngram_size", default=4),
            engram_vocab_size=_get(c, "engram_vocab_size", default=0),
            engram_n_heads=_get(c, "engram_n_heads", default=0),
            engram_head_dim=_get(c, "engram_head_dim", default=0),
            engram_pad_id=_get(c, "engram_pad_token_id", "engram_pad_id", default=2),
            engram_compressed_vocab_size=_get(
                c, "engram_compressed_vocab_size", default=0
            ),
            n_mtp_layers=_get(c, "num_nextn_predict_layers", "n_mtp_layers", default=0),
            dspark_target_layer_ids=tuple(
                _get(c, "dspark_target_layer_ids", default=()) or ()
            ),
            vision_n_layers=0,  # text-only runtime
        )

    @property
    def raw(self) -> dict[str, Any]:
        return {"model_type": "deepseek_v41"}
