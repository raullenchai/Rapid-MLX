#!/usr/bin/env python3
"""Paired real-checkpoint benchmark for the Qwen3.5 MoE router fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "Implement an LRU cache in Python with get and put in O(1). Explain "
    "invariants and include tests. Be detailed."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local snapshot path")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup_pairs < 1 or args.max_tokens < 1:
        parser.error("pairs, warmup-pairs, and max-tokens must be positive")
    return args


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode("ascii")).hexdigest()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"--model must be a cached local snapshot: {args.model}")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from vllm_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
    from vllm_mlx.moe_fusion import fuse_gate_up
    from vllm_mlx.qwen35_moe_router import _TAG, install_qwen35_moe_router

    model, tokenizer = load(model_path)
    existing_fusions = {
        "moe_gate_up_layers": fuse_gate_up(model),
        "gdn_projection_layers": fuse_gdn_in_proj(model),
    }
    enrolled = install_qwen35_moe_router(model)
    blocks = [
        module
        for _, module in model.named_modules()
        if bool(getattr(module, _TAG, False))
    ]
    if not enrolled or len(blocks) != enrolled:
        raise RuntimeError("the target did not enroll in the fused Qwen MoE router")

    sampler = make_sampler(temp=0.0)

    def run(enabled: bool) -> dict[str, Any]:
        for block in blocks:
            setattr(block, _TAG, enabled)
        mx.random.seed(0)
        tokens = []
        last = None
        for response in stream_generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        ):
            tokens.append(int(response.token))
            last = response
        if last is None:
            raise RuntimeError("generation emitted no tokens")
        return {
            "enabled": enabled,
            "generation_tps": float(last.generation_tps),
            "tokens": len(tokens),
            "token_sha256": token_digest(tokens),
        }

    # Compile both graphs before measurement. A/B/B/A also balances which path
    # runs first while the host reaches its sustained thermal state.
    for _ in range(args.warmup_pairs):
        for enabled in (False, True, True, False):
            run(enabled)

    pairs = []
    hashes = set()
    for pair_index in range(args.pairs):
        order = (False, True) if pair_index % 2 == 0 else (True, False)
        samples = {}
        for enabled in order:
            sample = run(enabled)
            samples[enabled] = sample
            hashes.add(sample["token_sha256"])
        stock = samples[False]["generation_tps"]
        fused = samples[True]["generation_tps"]
        pairs.append(
            {
                "pair": pair_index,
                "stock_tps": stock,
                "fused_tps": fused,
                "speedup": fused / stock,
                "stock_hash": samples[False]["token_sha256"],
                "fused_hash": samples[True]["token_sha256"],
            }
        )

    ratios = [pair["speedup"] for pair in pairs]
    result = {
        "model": str(model_path),
        "machine": platform.machine(),
        "max_tokens": args.max_tokens,
        "prompt_tokens": len(tokenizer.encode(args.prompt)),
        "pairs": pairs,
        "summary": {
            "median_speedup": statistics.median(ratios),
            "mean_speedup": statistics.mean(ratios),
            "positive_pairs": sum(ratio > 1.0 for ratio in ratios),
            "token_hashes": len(hashes),
        },
        "existing_fusions": existing_fusions,
        "router_layers": enrolled,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if len(hashes) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
