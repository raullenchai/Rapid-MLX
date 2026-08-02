# Copyright © 2026 Apple Inc.
# Vendored from spicyneuron/mlx-lm@df50b7a3 (_ds4 branch), including the
# DeepSeek V4 correctness series through 72c51e3. Keep this file and the
# deepseek_v4_{cache,hyper_connection,switch}.py companions in sync.

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten

# MUST run before any mlx_lm import: mlx_lm's package initializer captures a
# thread-local MLX stream, which is unusable on M5 single-stream GPUs (#404).
from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import CacheList, RotatingKVCache
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.pipeline import PipelineMixin

from .deepseek_v4_cache import DeepseekV4PoolingCache, PoolingCache
from .deepseek_v4_hyper_connection import HyperConnection, HyperHead, hc_expand
from .deepseek_v4_switch import FusedSwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    dspark_num_layers: int = 0
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    shared_experts = {k: mxfp8 for k, _ in flat_modules if ".ffn.shared_experts." in k}
    attn = {
        k: mxfp8 for k, _ in flat_modules if ".attn.w" in k or ".attn.indexer.wq" in k
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **shared_experts,
        **attn,
    }


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k].astype(
        mx.int32
    )
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _hash_expert_select(
    input_ids: mx.array,
    logits: mx.array,
    tid2eid: mx.array,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    inds = tid2eid[input_ids]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale
        self.attention_factor = 1.0

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")
            attention_factor = scaling_config.get("attention_factor")
            if attention_factor is not None:
                self.attention_factor = float(attention_factor)

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache = {}

    def _get_freqs(self, head_dim: int, inverse: bool):
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(
        self,
        x: mx.array,
        offset: Any = 0,
        inverse: bool = False,
    ) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        if self.attention_factor != 1.0:
            # NOPE channels precede the rotary channels in DeepSeek-V4.
            x = mx.concatenate(
                [
                    x[..., : -self.dims],
                    self.attention_factor * x[..., -self.dims :],
                ],
                axis=-1,
            )
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


def _apply_score_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None and pool_mask is None:
        return None

    if pool_mask is not None:
        if pool_mask.ndim == 2:
            pool_mask = pool_mask[None, None]
        elif pool_mask.ndim == 3:
            pool_mask = pool_mask[:, None]

    if mask is None:
        B, H, L, P = pool_mask.shape
        mask = mx.ones((B, H, L, N - P), dtype=mx.bool_)
    elif mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape

    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    else:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))

    if mask.dtype != mx.bool_ and pool_mask.dtype == mx.bool_:
        pool_mask = mx.where(
            pool_mask,
            mx.array(0, dtype=mask.dtype),
            mx.array(mx.finfo(mask.dtype).min, dtype=mask.dtype),
        )
    else:
        pool_mask = pool_mask.astype(mask.dtype)

    full_mask = mx.concatenate([mask, pool_mask], axis=-1)

    return full_mask


def _align_local_mask(mask: Optional[mx.array], local_len: int):
    if mask is None:
        return None

    current_len = mask.shape[-1]
    if current_len == local_len:
        return mask
    if current_len > local_len:
        return mask[..., -local_len:]

    pad_shape = (*mask.shape[:-1], local_len - current_len)
    if mask.dtype == mx.bool_:
        pad = mx.ones(pad_shape, dtype=mask.dtype)
    else:
        pad = mx.zeros(pad_shape, dtype=mask.dtype)
    return mx.concatenate([pad, mask], axis=-1)


def _pooled_mask(
    L: int,
    pooled_len: int,
    compress_ratio: int,
    offset: Union[int, mx.array] = 0,
):
    pool_idx = mx.arange(pooled_len)
    query_idx = mx.arange(1, L + 1)

    if isinstance(offset, mx.array) and offset.ndim > 0:
        query_idx = offset[:, None] + query_idx[None]
        return pool_idx[None, None] < query_idx[..., None] // compress_ratio

    query_idx = offset + query_idx
    return pool_idx[None] < query_idx[:, None] // compress_ratio


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, ape, head_dim):
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


@mx.compile
def _overlap_compress_kv(kv, gate, ape, head_dim, prior_kv, prior_gate):
    D = kv.shape[-1]

    # Overlap gates are stored before positional bias; add APE to both paths.
    gate = gate + ape.astype(gate.dtype)
    prior_gate = prior_gate + ape[:, : D // 2].astype(prior_gate.dtype)

    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([prior_kv[:, None], kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)

    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([prior_gate[:, None], gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)

    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    B, H, L, D = q.shape
    idx = topk[:, None, :, :, None]
    pooled = mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )

    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    pooled_sq = pooled.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = x @ self.weight.T

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds, weights = _hash_expert_select(
                input_ids,
                logits,
                self.tid2eid,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )
        else:
            inds, weights = _expert_select(
                logits,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )

        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = FusedSwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            swiglu_limit=config.swiglu_limit,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Compressor(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ) -> mx.array:
        B, _, _ = x.shape
        kv = self.wkv(x)
        gate = self.wgate(x)
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))
            if self.overlap:
                if pool_cache is None:
                    prior_kv = mx.zeros(
                        (B, self.compress_ratio, self.head_dim), dtype=kv.dtype
                    )
                    prior_gate = mx.full(
                        (B, self.compress_ratio, self.head_dim),
                        -mx.inf,
                        dtype=gate.dtype,
                    )
                else:
                    prior_kv, prior_gate = pool_cache.update_overlap_state(kv, gate)
                new_pooled = _overlap_compress_kv(
                    kv,
                    gate,
                    self.ape,
                    self.head_dim,
                    prior_kv,
                    prior_gate,
                )
            else:
                new_pooled = _simple_compress_kv(kv, gate, self.ape, self.head_dim)
            new_pooled = self.norm(new_pooled)
            new_pooled = self.rope(
                new_pooled[:, None],
                offset=pool_base,
            ).squeeze(1)

        if pool_cache is not None:
            new_pooled = pool_cache.update_and_fetch(new_pooled)

        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        position_rope: DeepseekV4RoPE,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None

        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = position_rope(q, offset)

        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        pmask = (
            pool_cache.make_mask(L, offset)
            if pool_cache is not None
            else _pooled_mask(
                L, pooled.shape[1], self.compressor.compress_ratio, offset
            )
        )
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None],
                scores,
                mx.finfo(scores.dtype).min,
            )
        k = min(self.index_topk, pooled.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


class LocalAttention(nn.Module):
    """DeepSeek V4 attention with no KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = 0
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        mask = _align_local_mask(mask, kv.shape[2])

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class CompressedAttention(nn.Module):
    """DeepSeek V4 attention with pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # Compressed layers use Yarn-scaled RoPE
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        mask = _align_local_mask(mask, kv.shape[2])

        # Pool tokens into compressed KV and concatenate with local KV
        pooled = self.compressor(x, pool_cache, offset)
        pooled_mask = None
        if pooled.shape[1] > 0:
            pooled_mask = (
                pool_cache.make_mask(L, offset)
                if pool_cache is not None
                else _pooled_mask(L, pooled.shape[1], self.compress_ratio, offset)
            )
            kv = mx.concatenate([kv, pooled[:, None]], axis=2)

        mask = _extend_mask(mask, pooled_mask, kv.shape[2])

        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=local_cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class SparseCompressedAttention(nn.Module):
    """DeepSeek V4 attention with sparse indexed pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        self.indexer = Indexer(config, self.compress_ratio)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        comp_cache = cache[1] if cache is not None else None
        idx_cache = cache[2] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q_residual = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(self.wkv(x)).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))
        mask = _align_local_mask(mask, kv.shape[2])

        pooled = self.compressor(x, comp_cache, offset)
        pmask = (
            comp_cache.make_mask(L, offset)
            if comp_cache is not None
            else (
                _pooled_mask(L, pooled.shape[1], self.compress_ratio, offset)
                if pooled.shape[1] > 0
                else None
            )
        )
        topk = self.indexer(x, q_residual, self.rope, idx_cache, offset)
        sinks = self.attn_sink.astype(q.dtype)

        # Local attention
        if pooled.shape[1] == 0:
            out = scaled_dot_product_attention(
                q,
                kv,
                kv,
                cache=local_cache,
                scale=self.scale,
                mask=mask,
                sinks=sinks,
            )

        # Compressed attention
        elif pooled.shape[1] <= self.indexer.index_topk:
            full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
            mask = _extend_mask(mask, pmask, full_kv.shape[2])
            out = scaled_dot_product_attention(
                q,
                full_kv,
                full_kv,
                cache=local_cache,
                scale=self.scale,
                mask=mask,
                sinks=sinks,
            )

        # Sparse compressed attention
        else:
            sparse_mask = None
            if pmask is not None:
                sparse_mask = mx.take_along_axis(
                    pmask[None] if pmask.ndim == 2 else pmask,
                    topk,
                    axis=2,
                )[:, None]
            out = _sparse_pooled_attention(
                q,
                kv,
                pooled,
                topk,
                mask,
                sparse_mask,
                self.scale,
                sinks,
            )

        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


def v4_attention_factory(config: ModelArgs, layer_idx: int) -> nn.Module:
    """Instantiate the appropriate attention module for a given layer."""
    ratio = config.compress_ratios[layer_idx]
    if ratio == 0:
        return LocalAttention(config, layer_idx)
    if ratio == 128:
        return CompressedAttention(config, layer_idx)
    return SparseCompressedAttention(config, layer_idx)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = v4_attention_factory(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        *,
        return_dspark_hidden: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        dspark_hiddens = []
        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)
            global_idx = layer.attn.layer_idx
            if return_dspark_hidden and global_idx in self.args.dspark_target_layer_ids:
                dspark_hiddens.append(h.mean(axis=2))

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, CacheList):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        output = self.norm(self.hc_head(h))
        if return_dspark_hidden:
            if len(dspark_hiddens) != len(self.args.dspark_target_layer_ids):
                raise RuntimeError(
                    "DSpark hidden-state capture is not available with the "
                    "current pipeline partition"
                )
            return output, mx.concatenate(dspark_hiddens, axis=-1)
        return output


class DSparkAttention(LocalAttention):
    """DSpark local attention: each draft slot sees target KV plus itself."""

    def update_main(self, main_x: mx.array, cache: RotatingKVCache) -> mx.array:
        offset = cache.offset
        kv = self.kv_norm(self.wkv(main_x)).reshape(
            main_x.shape[0], 1, main_x.shape[1], self.head_dim
        )
        kv = self.rope(kv, offset)
        return cache.update_and_fetch(
            kv, mx.zeros((*kv.shape[:-1], 0), dtype=kv.dtype)
        )[0]

    def __call__(
        self,
        x: mx.array,
        main_x: mx.array,
        cache: RotatingKVCache,
        *,
        update_main: bool = True,
    ) -> mx.array:
        B, K, _ = x.shape
        main_kv = self.update_main(main_x, cache) if update_main else cache.keys
        if main_kv is None:
            raise RuntimeError("DSpark target KV cache is empty")
        draft_offset = cache.offset

        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(B, K, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, draft_offset)

        draft_kv = self.kv_norm(self.wkv(x)).reshape(B, 1, K, self.head_dim)
        draft_kv = self.rope(draft_kv, draft_offset)

        # Flatten draft slots into the batch dimension. This gives every
        # semi-autoregressive slot the same target window plus only its own KV.
        main_len = main_kv.shape[2]
        expanded_main = mx.broadcast_to(
            main_kv[:, None], (B, K, 1, main_len, self.head_dim)
        ).reshape(B * K, 1, main_len, self.head_dim)
        own_kv = draft_kv.transpose(0, 2, 1, 3).reshape(B * K, 1, 1, self.head_dim)
        verify_kv = mx.concatenate([expanded_main, own_kv], axis=2)
        flat_q = q.transpose(0, 2, 1, 3).reshape(B * K, self.n_heads, 1, self.head_dim)
        out = scaled_dot_product_attention(
            flat_q,
            verify_kv,
            verify_kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(flat_q.dtype),
        )
        out = out.reshape(B, K, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = self.rope(out, draft_offset, inverse=True)
        out = out.reshape(B, self.o_groups, -1, K, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(out)


class DSparkMarkovHead(nn.Module):
    def __init__(self, config: ModelArgs):
        self.markov_w1 = nn.Embedding(config.vocab_size, config.dspark_markov_rank)
        self.markov_w2 = nn.Linear(
            config.dspark_markov_rank, config.vocab_size, bias=False
        )

    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        embedding = self.markov_w1(token_ids)
        return self.markov_w2(embedding), embedding


class DSparkBlock(nn.Module):
    def __init__(self, config: ModelArgs, stage_idx: int):
        self.stage_idx = stage_idx
        self.attn = DSparkAttention(config, config.num_hidden_layers + stage_idx)
        self.ffn = DeepseekV4MoE(config, config.num_hidden_layers + stage_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        if stage_idx == 0:
            self.main_proj = nn.Linear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                bias=False,
            )
            self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if stage_idx == config.dspark_num_layers - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(config)
            self.hc_head = HyperHead(config)

    def __call__(
        self,
        h: mx.array,
        main_x: mx.array,
        cache: RotatingKVCache,
        input_ids: mx.array,
        *,
        update_main: bool = True,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), main_x, cache, update_main=update_main)
        h = hc_expand(x, residual, post, comb)
        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = [DSparkBlock(config, idx) for idx in range(config.dspark_num_layers)]
        self._last_dspark_hidden: mx.array | None = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        *,
        return_dspark_hidden: bool = False,
    ):
        capture_dspark = return_dspark_hidden or bool(self.mtp)
        output = self.model(inputs, cache, return_dspark_hidden=capture_dspark)
        if capture_dspark:
            hidden, dspark_hidden = output
            self._last_dspark_hidden = dspark_hidden
            logits = self.lm_head(hidden)
            if return_dspark_hidden:
                return logits, dspark_hidden
            return logits
        if return_dspark_hidden:
            raise AssertionError("unreachable DSpark capture branch")
        return self.lm_head(output)

    def make_dspark_cache(self) -> list[RotatingKVCache]:
        return [RotatingKVCache(max_size=self.args.sliding_window) for _ in self.mtp]

    def dspark_forward(
        self,
        anchor_ids: mx.array,
        main_hidden: mx.array,
        cache: list[RotatingKVCache],
        max_draft_tokens: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array] | None:
        if not self.mtp:
            return None
        first = self.mtp[0]
        main_x = first.main_norm(first.main_proj(main_hidden))
        initial_prefill = cache[0].offset == 0
        for block, layer_cache in zip(self.mtp, cache):
            block.attn.update_main(main_x, layer_cache)
        if initial_prefill:
            return None

        K = self.args.dspark_block_size
        if max_draft_tokens is not None:
            K = max(1, min(K, int(max_draft_tokens)))
        draft_ids = mx.full(
            (anchor_ids.shape[0], K),
            self.args.dspark_noise_token_id,
            dtype=anchor_ids.dtype,
        )
        draft_ids[:, 0] = anchor_ids[:, -1]
        h = self.model.embed_tokens(draft_ids)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[-1]),
        )
        for block, layer_cache in zip(self.mtp, cache):
            h = block(
                h,
                main_x[:, -1:],
                layer_cache,
                draft_ids,
                update_main=False,
            )

        last = self.mtp[-1]
        collapsed = last.hc_head(h)
        draft_logits = self.lm_head(last.norm(collapsed))
        output_ids = mx.zeros((anchor_ids.shape[0], K + 1), dtype=anchor_ids.dtype)
        output_ids[:, 0] = anchor_ids[:, -1]
        for idx in range(K):
            bias, _ = last.markov_head(output_ids[:, idx])
            draft_logits[:, idx] = draft_logits[:, idx] + bias
            output_ids[:, idx + 1] = mx.argmax(draft_logits[:, idx], axis=-1)
        return output_ids, draft_logits

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                # local + compressor pool + indexer pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        DeepseekV4PoolingCache(ratio),
                        DeepseekV4PoolingCache(ratio),
                    )
                )
            else:
                # local + compressor pool
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp.") and not self.mtp:
                continue
            if ".confidence_head." in k:
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            nk = nk.replace(".hc_head_fn", ".hc_head.fn")
            nk = nk.replace(".hc_head_base", ".hc_head.base")
            nk = nk.replace(".hc_head_scale", ".hc_head.scale")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        for stage_idx in range(len(self.mtp)):
            prefix = f"mtp.{stage_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"mtp.{stage_idx}.ffn.switch_mlp.{dst}.{suffix}"] = (
                            mx.stack(stacked)
                        )

        # Fuse routed expert gate/up projections by concatenating output rows.
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.switch_mlp"
            for suffix in ("weight", "scales"):
                gate_key = f"{prefix}.gate_proj.{suffix}"
                up_key = f"{prefix}.up_proj.{suffix}"
                if gate_key in weights and up_key in weights:
                    weights[gate_key] = mx.concatenate(
                        [weights.pop(gate_key), weights.pop(up_key)], axis=1
                    )
        for stage_idx in range(len(self.mtp)):
            prefix = f"mtp.{stage_idx}.ffn.switch_mlp"
            for suffix in ("weight", "scales"):
                gate_key = f"{prefix}.gate_proj.{suffix}"
                up_key = f"{prefix}.up_proj.{suffix}"
                if gate_key in weights and up_key in weights:
                    weights[gate_key] = mx.concatenate(
                        [weights.pop(gate_key), weights.pop(up_key)], axis=1
                    )

        # Reshape wo_a from nn.Linear (2D) to MultiLinear (3D) for all layers
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )
        for stage_idx in range(len(self.mtp)):
            prefix = f"mtp.{stage_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.attn.sharding_group = group
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b,
                "all-to-sharded",
                segments=self.args.o_groups,
                group=group,
            )
            shard_inplace(layer.attn.wo_a, "sharded-to-all", group=group)
            layer.attn.attn_sink = mx.split(layer.attn.attn_sink, N)[rank]
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.switch_mlp.gate_proj,
                "all-to-sharded",
                segments=2,
                group=group,
            )
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            layer.ffn.switch_mlp.hidden_dims //= N
