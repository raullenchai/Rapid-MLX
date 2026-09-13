#!/usr/bin/env python3
"""Paired speed and token-parity gate for Qwen3.5-family fused GDN decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

PROMPTS = {
    "coding": "Implement a correct Python LRU cache with O(1) get and put, plus tests.",
    "creative": "Write a vivid 180-word scene about a lighthouse keeper hearing music below the ice.",
    "reasoning": "A shop marks an item up 25%, then discounts it 20%. Explain the net percentage change.",
    "json": 'Return only compact JSON with keys "prime" and "explanation" for whether 221 is prime.',
    "tool": "You have a weather(location) tool. State the exact tool call needed for weather in Kyoto.",
}


def _digest(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode("ascii")).hexdigest()


def _coefficient_of_variation(samples: list[float]) -> float:
    mean = statistics.mean(samples)
    return statistics.pstdev(samples) / mean if mean > 0.0 else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--quality-max-tokens", type=int, default=512)
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        parser.error("--model must be a cached local snapshot")
    if args.pairs < 1 or min(args.max_tokens, args.quality_max_tokens) < 32:
        parser.error("pairs must be positive and token limits must be at least 32")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from vllm_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
    from vllm_mlx.moe_fusion import fuse_gate_up
    from vllm_mlx.qwen35_fused_gdn_decode import (
        _TAG,
        install_qwen35_fused_gdn_decode,
    )
    from vllm_mlx.qwen35_moe_router import install_qwen35_moe_router

    model, tokenizer = load(model_path)
    existing = {
        "gate_up": fuse_gate_up(model),
        "gdn_projection": fuse_gdn_in_proj(model),
        "moe_router": install_qwen35_moe_router(model),
    }
    enrolled = install_qwen35_fused_gdn_decode(model)
    layers = [m for _, m in model.named_modules() if getattr(m, _TAG, False)]
    if enrolled == 0 or len(layers) != enrolled:
        raise RuntimeError("checkpoint did not enroll in fused GDN decode")
    sampler = make_sampler(temp=0.0)

    def render(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def run(
        enabled: bool, prompt: str, *, max_tokens: int = args.max_tokens
    ) -> dict[str, object]:
        for layer in layers:
            setattr(layer, _TAG, enabled)
        mx.random.seed(0)
        responses = list(
            stream_generate(
                model,
                tokenizer,
                render(prompt),
                max_tokens=max_tokens,
                sampler=sampler,
            )
        )
        tokens = [int(response.token) for response in responses]
        return {
            "tps": float(responses[-1].generation_tps),
            "tokens": tokens,
            "hash": _digest(tokens),
        }

    for enabled in (False, True, True, False):
        run(enabled, PROMPTS["coding"])

    quality = {}
    for name, prompt in PROMPTS.items():
        stock = run(False, prompt, max_tokens=args.quality_max_tokens)
        fused = run(True, prompt, max_tokens=args.quality_max_tokens)
        quality[name] = {
            "tokens": len(stock["tokens"]),
            "same_tokens": stock["tokens"] == fused["tokens"],
            "stock_hash": stock["hash"],
            "fused_hash": fused["hash"],
        }

    pairs = []
    for index in range(args.pairs):
        order = (False, True) if index % 2 == 0 else (True, False)
        samples = {enabled: run(enabled, PROMPTS["coding"]) for enabled in order}
        stock_tps = float(samples[False]["tps"])
        fused_tps = float(samples[True]["tps"])
        pairs.append(
            {
                "stock_tps": stock_tps,
                "fused_tps": fused_tps,
                "speedup": fused_tps / stock_tps,
            }
        )
    ratios = [sample["speedup"] for sample in pairs]
    stock_samples = [sample["stock_tps"] for sample in pairs]
    fused_samples = [sample["fused_tps"] for sample in pairs]
    stock_cv = _coefficient_of_variation(stock_samples)
    fused_cv = _coefficient_of_variation(fused_samples)
    positive_pairs = sum(ratio > 1.0 for ratio in ratios)
    performance_valid = (
        stock_cv <= 0.05
        and fused_cv <= 0.05
        and positive_pairs >= max(1, args.pairs - 1)
    )
    result = {
        "model": str(model_path),
        "machine": platform.machine(),
        "max_tokens": args.max_tokens,
        "quality_max_tokens": args.quality_max_tokens,
        "existing_fusions": existing,
        "gdn_layers": enrolled,
        "quality": quality,
        "pairs": pairs,
        "summary": {
            "median_speedup": statistics.median(ratios),
            "mean_speedup": statistics.mean(ratios),
            "positive_pairs": positive_pairs,
            "stock_tps_cv": stock_cv,
            "fused_tps_cv": fused_cv,
            "performance_valid": performance_valid,
            "quality_cases_exact": sum(
                bool(case["same_tokens"]) for case in quality.values()
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["summary"]["quality_cases_exact"] != len(PROMPTS):
        return 1
    return 0 if performance_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
