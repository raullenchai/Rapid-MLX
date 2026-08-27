#!/usr/bin/env python3
"""Generate a SMALL SYNTHETIC Qwen3.8-Flash-Next-style shard set.

Same tensor-name/dtype/shape *pattern* as the real checkpoint, scaled down so
the converter can be verified end-to-end in seconds without any real weights.

Covers every tensor class the converter's manifest classification must handle
(per Vector's review): quantisable matrices, PLE/embed tables, plus the
COPY-as-fp classes — 1-D norms, ``A_log`` safegate params, buffer/aux dims,
non-group-size-divisible widths.

Produces ``<out>/{model-0000X-of-0000N.safetensors, model.safetensors.index.json,
config.json}``. Never run against real weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def build(
    out: Path, hidden: int = 2560, inter: int = 640, num_experts: int = 8
) -> None:
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing non-empty fixture dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    weight_map: dict[str, str] = {}
    shards: list[dict[str, np.ndarray | mx.array]] = []
    total = 5

    def put(tensors: dict[str, np.ndarray]):
        shard_name = f"model-{len(shards) + 1:05d}-of-{total:05d}.safetensors"
        for k in tensors:
            weight_map[k] = shard_name
        shards.append(tensors)

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    # Shard 1: MoE expert matrices (quantisable, 2-D divisible).
    moe = {}
    for e in range(num_experts):
        pre = f"model.language_model.layers.0.mlp.experts.{e}"
        moe[f"{pre}.down_proj.weight"] = f32(inter, hidden)
        moe[f"{pre}.gate_proj.weight"] = f32(hidden, inter)
        moe[f"{pre}.up_proj.weight"] = f32(hidden, inter)
    put(moe)

    # Shard 2: dense attention + a 1-D RMS norm (COPY), and an A_log buffer
    # (COPY — shape [heads]).
    dense = {}
    dense["model.language_model.embed_tokens.weight"] = f32(64, hidden)
    dense["model.language_model.layers.0.self_attn.q_proj.weight"] = f32(hidden, hidden)
    dense["model.language_model.layers.0.self_attn.o_proj.weight"] = f32(hidden, hidden)
    dense["model.language_model.layers.0.mlp.gate.weight"] = f32(512, hidden)
    dense["model.language_model.layers.0.mlp.shared_expert_gate.weight"] = f32(
        1, hidden
    )
    dense["model.language_model.norm.weight"] = mx.array(f32(hidden)).astype(
        mx.bfloat16
    )
    dense["model.language_model.layers.0.linear_attn.A_log"] = f32(2)
    put(dense)

    # Shard 3: exact PLE shard names and width 160 exercise q4-g32.
    ple = {}
    for i in range(2):
        name = (
            "model.language_model.layers.1.ple.ple_embedding."
            f"ngram_embedding.shard_{i}.weight"
        )
        ple[name] = f32(32, 160).astype(np.float16)
    ple["model.language_model.layers.1.ple.ple_embedding.layer_multipliers"] = f32(8)
    put(ple)

    # Shard 4: a non-divisible-by-group-size 2-D tensor (COPY) and a true
    # 1-D bias (COPY), plus an int aux buffer.
    misc = {}
    misc["model.language_model.layers.0.mlp.gate.bias"] = f32(inter)
    misc["projector.non_divisible.weight"] = f32(5, 100)  # 100 % 32 != 0 -> copy
    misc["model.language_model.rotary_emb.inv_freq"] = np.zeros(
        hidden // 2, dtype=np.float32
    )  # 1-D copy
    misc["model.language_model.layers.0.attention.num_buckets"] = np.array(
        [64], dtype=np.int32
    )  # int buffer copy
    put(misc)

    # Shard 5: dense logits head (2-D divisible, quantize).
    put({"lm_head.weight": f32(64, hidden)})

    for tensors, slot in zip(shards, range(1, total + 1)):
        mx.save_safetensors(
            str(out / f"model-{slot:05d}-of-{total:05d}.safetensors"),
            {name: mx.array(value) for name, value in tensors.items()},
        )

    index = {
        "metadata": {"total_size": sum(t.nbytes for sh in shards for t in sh.values())},
        "weight_map": weight_map,
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "hidden_size": hidden,
                "num_experts": num_experts,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": inter,
                "ple_embed_dim": hidden,
            },
            indent=2,
        )
    )
    print(f"fixture written to {out}: {total} shards, {len(weight_map)} weights")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic-kit")
    build(out)
