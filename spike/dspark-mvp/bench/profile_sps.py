# SPDX-License-Identifier: Apache-2.0
"""Profile target-model steps-per-second as a function of batch tokens B.

For Algorithm 1 we need SPS(B) — the steps/sec the target can deliver
when verifying B tokens in one forward pass. We approximate the
production "decode-then-verify" cost by running plain forward passes
of varying B token counts against a warm KV cache.

Output: spike/dspark-mvp/sps_table.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.utils import load
from mlx_lm.models.cache import make_prompt_cache


def warm_cache(lm, prompt_ids: mx.array) -> list:
    cache = make_prompt_cache(lm)
    out = lm(prompt_ids[None], cache=cache)
    mx.eval(out)
    return cache


def measure_sps(lm, cache, B: int, n_steps: int = 20) -> float:
    """Run n_steps decode steps with B tokens per step; return steps/sec."""
    # Use a synthetic sequence — content doesn't matter for timing once cache is warm.
    seq = mx.zeros((1, B), dtype=mx.int32) + 100  # dummy tokens (avoid 0 which may be special)

    # Warmup
    for _ in range(3):
        out = lm(seq, cache=cache)
        mx.eval(out)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        out = lm(seq, cache=cache)
        mx.eval(out)
    elapsed = time.perf_counter() - t0
    return n_steps / elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="mlx-community/Qwen3.5-9B-4bit")
    p.add_argument(
        "--Bs",
        nargs="+",
        type=int,
        default=[1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    )
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--out", default="spike/dspark-mvp/sps_table.json")
    p.add_argument(
        "--prompt-len",
        type=int,
        default=256,
        help="Length of the warm-up prompt fed before timing (sets cache depth).",
    )
    args = p.parse_args()

    print(f"[sps] loading {args.target}")
    model, _ = load(args.target)
    model.eval()
    lm = model.language_model

    # Warm a realistic-depth prompt (so cache attention has meaningful work).
    prompt = mx.zeros((args.prompt_len,), dtype=mx.int32) + 100
    cache = warm_cache(lm, prompt)
    print(f"[sps] cache warmed at depth={args.prompt_len}")

    results = []
    for B in args.Bs:
        sps = measure_sps(lm, cache, B, n_steps=args.n_steps)
        tok_per_sec = sps * B
        results.append({"B": B, "sps": sps, "tok_per_sec": tok_per_sec})
        print(f"[sps] B={B:4d}  sps={sps:7.2f}  tok/s={tok_per_sec:7.1f}")

    # Fit a monotone non-increasing envelope (DSpark paper assumes
    # "smoothly decaying hardware capacity curve" — Section 3.2.2). Use
    # the isotonic-from-left descending step that's the largest plateau
    # consistent with the noisy measurements.
    smoothed_sps = []
    running_max = float("inf")
    for r in results:
        running_max = min(running_max, r["sps"])
        smoothed_sps.append(running_max)
    # Forward pass: cap each SPS at the cumulative min seen so far.
    smoothed = []
    for r, s in zip(results, smoothed_sps):
        smoothed.append({"B": r["B"], "sps": s, "tok_per_sec": s * r["B"]})
    print()
    print("[sps] smoothed (monotone non-increasing envelope):")
    for s in smoothed:
        print(f"[sps]   B={s['B']:4d}  sps={s['sps']:7.2f}  tok/s={s['tok_per_sec']:7.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "target": args.target,
                "prompt_depth": args.prompt_len,
                "n_steps": args.n_steps,
                "results": results,
                "results_smoothed": smoothed,
            },
            indent=2,
        )
    )
    print(f"[sps] wrote {args.out}")


if __name__ == "__main__":
    main()
