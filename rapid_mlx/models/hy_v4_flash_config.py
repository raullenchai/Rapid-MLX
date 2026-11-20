# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text configuration and frozen checkpoint contracts for Hy4 Flash."""

import re
from dataclasses import dataclass, field, fields

import numpy as np


def checkpoint_quantization(config):
    """Read outer quantization metadata first, then nested text metadata."""
    text = config.get("text_config") or {}
    return (
        config.get("quantization")
        or config.get("quantization_config")
        or text.get("quantization")
        or text.get("quantization_config")
        or {}
    )


@dataclass
class ModelArgs:
    model_type: str = "hy_v4_flash"
    vocab_size: int = 120832
    hidden_size: int = 3072
    intermediate_size: int = 13824
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    swa_num_attention_heads: int = 64
    swa_qk_nope_head_dim: int = 192
    swa_qk_rope_head_dim: int = 64
    swa_v_head_dim: int = 256
    sliding_window: int = 512
    layer_types: list[str] = field(default_factory=list)
    mlp_layer_types: list[str] = field(default_factory=list)
    first_k_dense_replace: int = 1
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    num_experts_per_tok: int = 16
    moe_intermediate_size: int = 1536
    routed_expert_hidden_size: int = 1536
    routed_scaling_factor: float = 3.998
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    sandwich_norm: bool = True
    dsa_skip_attn_post_norm: bool = False
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    rms_norm_eps: float = 1e-5
    rope_parameters: dict = field(default_factory=dict)
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_group_topk: int = 512
    compressed_indexer_num_attention_heads: int = 16
    compressed_indexer_head_dim: int = 128
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_layer_head_vocab_sizes: dict = field(default_factory=dict)
    engram_max_ngram_size: int = 3
    engram_n_head_per_ngram: int = 8
    engram_n_embed_per_ngram: int = 768
    engram_pad_id: int = 120002
    engram_seed: int = 0
    enable_lm_head_fp32: bool = True
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, config):
        text = config.get("text_config", config)
        for name, expected in (
            ("enable_ihc", True),
            ("use_mla", True),
            ("use_dsa", True),
            ("use_compressed_indexer", True),
            ("compressed_indexer_reuse_fine_indexer", False),
            ("full_nope", False),
            ("gate_lora_rank", None),
            ("o_lora_rank", None),
            ("gate_groups", 1),
            ("o_groups", 1),
            ("hidden_act", "silu"),
            ("gated_mla", True),
            ("learnable_sink", True),
            ("qk_norm", True),
            ("compressed_indexer_skip_fine_select", True),
            ("moe_router_use_sigmoid", True),
            ("moe_router_enable_expert_bias", True),
            ("gating_type", "elementwise"),
            ("compressed_indexer_key_pooling", "gated"),
        ):
            if text.get(name, expected) != expected:
                raise ValueError(f"Unsupported Hy4 Flash setting: {name}")
        args = cls(**{f.name: text[f.name] for f in fields(cls) if f.name in text})
        for parameters in args.rope_parameters.values():
            if parameters.get("rope_type", "default") != "default":
                raise ValueError("Hy4 Flash requires default RoPE")
        if not args.layer_types:
            args.layer_types = ["deepseek_sparse_attention"] * args.num_hidden_layers
        if not args.mlp_layer_types:
            args.mlp_layer_types = [
                "dense" if i < args.first_k_dense_replace else "sparse"
                for i in range(args.num_hidden_layers)
            ]
        if len(args.layer_types) != args.num_hidden_layers or any(
            t not in ("sliding_attention", "deepseek_sparse_attention")
            for t in args.layer_types
        ):
            raise ValueError("Invalid Hy4 Flash attention layer_types")
        if len(args.mlp_layer_types) != args.num_hidden_layers or any(
            t not in ("dense", "sparse") for t in args.mlp_layer_types
        ):
            raise ValueError("Invalid Hy4 Flash mlp_layer_types")
        if not 1 <= args.num_experts_per_tok <= args.n_routed_experts:
            raise ValueError("Invalid Hy4 Flash expert top-k")
        if (
            args.sliding_window < 1
            or args.compressed_indexer_compress_ratio < 1
            or args.compressed_indexer_group_topk < 1
        ):
            raise ValueError("Invalid Hy4 Flash window or indexer budget")
        if args.hc_mult < 2:
            raise ValueError("Hy4 Flash requires multiple residual streams")
        if any(i < 1 or i > args.num_hidden_layers for i in args.engram_layer_ids):
            raise ValueError("Invalid one-based Engram layer id")
        return args


def checkpoint_key(name: str) -> str | None:
    """Normalize text weight names; auxiliary towers are deliberately omitted."""
    name = name.removeprefix("language_model.")
    if name.startswith("model.model."):
        name = name.removeprefix("model.")
    if name.startswith(("vit.", "vision_tower.", "model.dfly_layers.", "dfly_layers.")):
        return None
    name = re.sub(
        r"model\.layers\.(\d+)\.engram\.",
        lambda m: f"model.layers.{int(m[1]) - 1}.engram_layer.",
        name,
    )
    name = name.replace(".compressed_indexer.", ".indexer.")
    name = name.replace(".gate.e_score_correction_bias", ".expert_bias")
    if name.endswith((".hc_fn", ".hc_head_fn")):
        name += ".weight"
    if ".mlp.experts." in name:
        name = name.replace(".mlp.experts.", ".mlp.switch_mlp.")
        name = re.sub(
            r"(\.(?:gate_up_proj|down_proj))(_scale_inv)?$",
            lambda m: m[1] + ".weight" + (m[2] or ""),
            name,
        )
    return name


def engram_hash(input_ids, history, lookup, primes, layer_id, args):
    """Frozen lookup + odd int64 multipliers + XOR, with request lookback."""
    lookup = validate_lookup(lookup, args.vocab_size)
    compressed = int(lookup.max()) + 1
    token_map = np.concatenate((lookup, np.arange(len(lookup), args.vocab_size)))
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    multipliers = (
        np.random.default_rng(args.engram_seed + 10007 * layer_id).integers(
            0, bound, size=args.engram_max_ngram_size, dtype=np.int64
        )
        * 2
        + 1
    )
    ids = np.asarray(input_ids, dtype=np.int64)
    prior = np.asarray(history, dtype=np.int64)
    all_ids = np.concatenate((prior, ids), axis=1)
    positions = np.arange(prior.shape[1], all_ids.shape[1])
    blocked = np.zeros(ids.shape, dtype=bool)
    rolling = np.zeros(ids.shape, dtype=np.int64)
    hashes = []
    offset = 0
    for shift in range(args.engram_max_ngram_size):
        source = all_ids[:, np.maximum(positions - shift, 0)]
        blocked |= (positions < shift) | (source < 0) | (source >= args.vocab_size)
        mapped = token_map[np.clip(source, 0, args.vocab_size - 1)]
        mapped = np.where(blocked, token_map[args.engram_pad_id], mapped)
        # Signed overflow is part of the checkpoint hash contract.
        rolling = np.bitwise_xor(rolling, mapped.astype(np.int64) * multipliers[shift])
        if shift:
            for prime in primes[shift - 1]:
                hashes.append(rolling % prime + offset)
                offset += prime
    result = np.stack(hashes, axis=-1)
    return np.where(
        ((ids < 0) | (ids >= args.vocab_size) | (ids == args.engram_pad_id))[..., None],
        -1,
        result,
    ).astype(np.int32)


def dequantize_fp8(raw, scales, block=(128, 128)):
    """Decode E4M3FN bytes and multiplicative block (or embedding-row) scales."""
    raw = np.asarray(raw, dtype=np.uint8)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    decoded = np.where(
        exponent == 0,
        mantissa * 2.0**-9,
        (1 + mantissa / 8) * np.exp2(exponent.astype(np.float32) - 7),
    )
    decoded = np.where(raw & 128, -decoded, decoded).astype(np.float32)
    if np.any((exponent == 15) & (mantissa == 7)):
        raise ValueError("Non-finite FP8 checkpoint weight")
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape == raw.shape[:-1]:
        expanded = scales[..., None]
    else:
        rows, cols = raw.shape[-2:]
        expected = (
            *raw.shape[:-2],
            (rows + block[0] - 1) // block[0],
            (cols + block[1] - 1) // block[1],
        )
        if scales.shape != expected:
            raise ValueError("Invalid FP8 block scale shape")
        expanded = np.repeat(np.repeat(scales, block[0], axis=-2), block[1], axis=-1)[
            ..., :rows, :cols
        ]
    return decoded * expanded


def quantization_policy(path, module, *, group_size=64, bits=4):
    """Keep small fidelity-sensitive parameters dense and hash tables at g32."""
    if path == "lm_head" or path.endswith(
        (
            ".mlp.gate",
            ".hc_fn",
            ".hc_head_fn",
            ".key_value_projection",
            ".key_gate_projection",
        )
    ):
        return False
    if ".engram_layer." in path:
        if path.endswith(".embedding"):
            return {"group_size": 32, "bits": bits, "mode": "affine"}
        return False
    return {"group_size": group_size, "bits": bits, "mode": "affine"}


def validate_lookup(lookup, vocab_size):
    lookup = np.asarray(lookup)
    if (
        lookup.ndim != 1
        or lookup.dtype.kind not in "iu"
        or not 0 < len(lookup) <= vocab_size
        or np.any(lookup < 0)
        or np.any(lookup > np.iinfo(np.int32).max)
    ):
        raise ValueError("Invalid frozen Engram token map")
    return lookup


def parameter_shapes(a):
    """Compute resident parameter shapes without allocating model weights."""
    h, c = a.hidden_size, a.hc_mult
    shapes = {
        "model.embed_tokens.weight": (a.vocab_size, h),
        "model.norm.weight": (h,),
        "model.hc_head.hc_head_fn.weight": (c, c * h),
        "model.hc_head.hc_head_scale": (1,),
        "model.hc_head.hc_head_base": (c,),
        "lm_head.weight": (a.vocab_size, h),
    }
    if a.tie_word_embeddings:
        shapes.pop("lm_head.weight")

    def mlp(prefix, hidden, intermediate):
        for proj in ("gate_proj", "up_proj"):
            shapes[prefix + proj + ".weight"] = (intermediate, hidden)
        shapes[prefix + "down_proj.weight"] = (hidden, intermediate)

    for i in range(a.num_hidden_layers):
        p = f"model.layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[p + norm + ".weight"] = (h,)
        for hc in ("hc_attn_layer", "hc_mlp_layer"):
            base = p + hc + ".hc_pre."
            shapes[base + "hc_fn.weight"] = (2 * c, c * h)
            shapes[base + "hc_scale"] = (2,)
            shapes[base + "hc_base"] = (2 * c,)
        window = a.layer_types[i] == "sliding_attention"
        heads = a.swa_num_attention_heads if window else a.num_attention_heads
        nope = a.swa_qk_nope_head_dim if window else a.qk_nope_head_dim
        rope = a.swa_qk_rope_head_dim if window else a.qk_rope_head_dim
        v = a.swa_v_head_dim if window else a.v_head_dim
        attn = p + "self_attn."
        for name, shape in {
            "q_a_proj.weight": (a.q_lora_rank, h),
            "q_a_layernorm.weight": (a.q_lora_rank,),
            "q_b_proj.weight": (heads * (nope + rope), a.q_lora_rank),
            "kv_a_proj_with_mqa.weight": (a.kv_lora_rank + rope, h),
            "kv_a_layernorm.weight": (a.kv_lora_rank,),
            "kv_b_proj.weight": (heads * (nope + v), a.kv_lora_rank),
            "linear_gate.weight": (heads * v, h),
            "o_proj.weight": (h, heads * v),
            "learnable_sink_param": (heads,),
        }.items():
            shapes[attn + name] = shape
        if a.sandwich_norm and (window or not a.dsa_skip_attn_post_norm):
            shapes[attn + "post_proj_norm.weight"] = (h,)
        if not window:
            d, nh = (
                a.compressed_indexer_head_dim,
                a.compressed_indexer_num_attention_heads,
            )
            for name, shape in {
                "query_projection.weight": (nh * d, a.q_lora_rank),
                "head_weight_projection.weight": (nh, h),
                "token_key_projection.weight": (d, h),
                "token_key_norm.weight": (d,),
                "key_value_projection.weight": (d, d),
                "key_gate_projection.weight": (d, d),
                "key_norm.weight": (d,),
                "intra_group_ape": (a.compressed_indexer_compress_ratio, d),
            }.items():
                shapes[attn + "indexer." + name] = shape
        m = p + "mlp."
        if a.mlp_layer_types[i] == "dense":
            mlp(m, h, a.intermediate_size)
            if a.sandwich_norm:
                shapes[m + "post_fc2_norm.weight"] = (h,)
        else:
            e, r, mid = (
                a.n_routed_experts,
                a.routed_expert_hidden_size,
                a.moe_intermediate_size,
            )
            shapes[m + "gate.weight"] = (e, h)
            shapes[m + "expert_bias"] = (e,)
            shapes[m + "routed_expert_down_proj.weight"] = (r, h)
            shapes[m + "routed_expert_up_proj.weight"] = (h, r)
            for proj in ("gate_proj", "up_proj"):
                shapes[m + "switch_mlp." + proj + ".weight"] = (e, mid, r)
            shapes[m + "switch_mlp.down_proj.weight"] = (e, r, mid)
            if a.n_shared_experts:
                mlp(m + "shared_experts.", h, mid * a.n_shared_experts)
            if a.sandwich_norm:
                shapes[m + "combine_output_norm.weight"] = (h,)
        if i + 1 in a.engram_layer_ids:
            sizes = [
                v
                for order in a.engram_layer_head_vocab_sizes[str(i + 1)]
                for v in order
            ]
            d = a.engram_n_embed_per_ngram // a.engram_n_head_per_ngram
            width = len(sizes) * d
            base = p + "engram_layer."
            shapes[base + "lookup_table"] = (a.vocab_size,)
            shapes[base + "multi_head_embedding.embedding.weight"] = (sum(sizes), d)
            shapes[base + "unified_pre_gamma"] = (h,)
            for j in range(len(sizes)):
                shapes[base + f"lookup_per_head_norms.{j}.weight"] = (d,)
            for proj in ("value_proj", *(f"key_projs.{j}" for j in range(c))):
                shapes[base + proj + ".weight"] = (h, width)
                shapes[base + proj + ".bias"] = (h,)
            for j in range(c):
                shapes[base + f"unified_per_hc_scales.{j}"] = (h,)
                for norm in ("norm_key", "norm_query"):
                    shapes[base + f"{norm}.{j}.weight"] = (h,)
    return shapes
