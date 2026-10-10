#!/usr/bin/env python3
"""Qualify stock/blocked GDN prefill and fused decode on a resident model.

Uses fresh prompt caches for every request. Reports actual engagement, token
digests, raw paired observations and process peak memory; never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.model.is_dir() or args.rounds < 2 or args.max_tokens < 32:
        parser.error("require a local snapshot, >=2 rounds and >=32 output tokens")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.models import gated_delta as gd
    from mlx_lm.sample_utils import make_sampler

    from rapid_mlx import gdn_prefill as prefill
    from rapid_mlx.gdn_in_proj_fusion import fuse_gdn_in_proj
    from rapid_mlx.moe_fusion import fuse_gate_up
    from rapid_mlx.patches.qwen3_5_eager_dispatch import install_qwen3_5_eager_dispatch
    from rapid_mlx.qwen35_fused_gdn_decode import _TAG, install_qwen35_fused_gdn_decode
    from rapid_mlx.qwen35_moe_router import install_qwen35_moe_router

    model, tokenizer = load(str(args.model))
    install_qwen3_5_eager_dispatch()
    fusion = {
        "gate_up": fuse_gate_up(model),
        "gdn_input": fuse_gdn_in_proj(model),
        "router": install_qwen35_moe_router(model),
        "gdn_decode": install_qwen35_fused_gdn_decode(model),
    }
    layers = [m for _, m in model.named_modules() if getattr(m, _TAG, False)]
    if not layers or not prefill.install():
        raise RuntimeError("requested optimization did not enroll")
    fast, stock = gd.gated_delta_kernel, prefill._original_kernel
    original_blocked = prefill.gated_delta_blocked_seq
    calls = 0

    def counted(*a, **kw):
        nonlocal calls
        calls += 1
        return original_blocked(*a, **kw)

    prefill.gated_delta_blocked_seq = counted
    sampler = make_sampler(temp=0)
    arms = {
        "stock": (False, False),
        "prefill": (True, False),
        "prefill_and_fused_decode": (True, True),
    }

    def prompt(target: int) -> list[int]:
        unit = "The local service processes ordered records and preserves exact identifiers. "
        content = unit * target
        low, high = 0, len(content)
        while low < high:
            mid = (low + high + 1) // 2
            ids = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": content[:mid]
                        + "\nSummarize the document in detail.",
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(ids) <= target:
                low = mid
            else:
                high = mid - 1
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": content[:low] + "\nSummarize the document in detail.",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def run(ids: list[int], arm: str, budget: int) -> dict:
        nonlocal calls
        use_prefill, use_decode = arms[arm]
        gd.gated_delta_kernel = fast if use_prefill else stock
        for layer in layers:
            setattr(layer, _TAG, use_decode)
        calls = 0
        mx.random.seed(0)
        started = time.perf_counter()
        outputs = list(
            stream_generate(
                model,
                tokenizer,
                ids,
                max_tokens=budget,
                sampler=sampler,
                prefill_step_size=2048,
            )
        )
        mx.synchronize()
        tokens = [int(o.token) for o in outputs]
        last = outputs[-1]
        if use_prefill and not calls:
            raise RuntimeError("prefill arm silently ran stock")
        return {
            "arm": arm,
            "prompt_tokens": len(ids),
            "generated_tokens": len(tokens),
            "prompt_tps": float(last.prompt_tps),
            "decode_tps": float(last.generation_tps),
            "elapsed_s": time.perf_counter() - started,
            "blocked_calls": calls,
            "fused_decode_layers": len(layers) if use_decode else 0,
            "token_sha256": hashlib.sha256(
                b"".join(t.to_bytes(4, "little") for t in tokens)
            ).hexdigest(),
        }

    result = {
        "model": str(args.model),
        "versions": {
            p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "transformers"]
        },
        "fusion": fusion,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "prefill_step_size": 2048,
        "rows": [],
        "summary": [],
    }
    for length in args.lengths:
        ids = prompt(length)
        for arm in arms:
            run(ids, arm, min(args.max_tokens, 32))
        for rd in range(args.rounds):
            order = list(arms) if rd % 2 == 0 else list(reversed(arms))
            pair = []
            for arm in order:
                row = run(ids, arm, args.max_tokens)
                row.update(target_length=length, round=rd)
                pair.append(row)
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
            exact = len({r["token_sha256"] for r in pair}) == 1
            result["summary"].append(
                {"length": length, "round": rd, "all_tokens_exact": exact}
            )
    result["peak_gib"] = mx.get_peak_memory() / 2**30
    result["all_tokens_exact"] = all(r["all_tokens_exact"] for r in result["summary"])
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "event": "summary",
                "all_tokens_exact": result["all_tokens_exact"],
                "peak_gib": result["peak_gib"],
            }
        ),
        flush=True,
    )
    if not result["all_tokens_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
