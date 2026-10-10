#!/usr/bin/env python3
"""Warm alternating native-MTP qualification with exact emitted token IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def emitted_token_ids(results):
    """Prefer the terminal receipt; streaming repeats the final token."""
    if not results:
        raise RuntimeError("empty generation stream")
    terminal = results[-1]
    ids = getattr(terminal, "token_ids", None)
    if ids is not None:
        tokens = [int(t) for t in ids]
        if len(tokens) != terminal.generation_tokens:
            raise RuntimeError("terminal token receipt has the wrong length")
        return tokens
    tokens = []
    for row in results:
        count = row.generation_tokens
        if count == len(tokens):
            continue
        if count != len(tokens) + 1 or row.token is None:
            raise RuntimeError("stream did not expose a contiguous token sequence")
        tokens.append(int(row.token))
    return tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--drafter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    args = ap.parse_args()
    if not args.target.is_dir() or not args.drafter.is_dir() or args.rounds < 2:
        ap.error("local target/drafter and >=2 rounds required")
    import mlx.core as mx
    from mlx_vlm import load, stream_generate

    from rapid_mlx.models.mlx_vlm_vendored.speculative.drafters import load_drafter
    from scripts.benchmark_qwen36_native_mtp import PROMPTS, _render

    model, processor = load(str(args.target))
    drafter, kind = load_drafter(str(args.drafter), kind="mtp")
    if kind != "mtp":
        raise RuntimeError("wrong drafter kind")
    drafter.bind(model)
    result = {
        "target": str(args.target),
        "drafter": str(args.drafter),
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "canonical_preload_workaround": False,
        "rows": [],
        "pairs": [],
    }

    def run(prompt, mode, budget):
        for key in ["accept_lens", "draft_lens"]:
            value = getattr(drafter, key, None)
            if isinstance(value, list):
                value.clear()
        kw = {"max_tokens": budget, "temperature": 0.0, "verbose": False}
        if mode == "mtp":
            kw.update(draft_model=drafter, draft_kind="mtp", draft_block_size=3)
        mx.random.seed(0)
        out = list(stream_generate(model, processor, prompt, **kw))
        mx.synchronize()
        tokens = emitted_token_ids(out)
        if not tokens:
            raise RuntimeError("stream did not expose token IDs")
        last = out[-1]
        accepted = (
            list(getattr(drafter, "accept_lens", []) or []) if mode == "mtp" else []
        )
        drafted = (
            list(getattr(drafter, "draft_lens", []) or []) if mode == "mtp" else []
        )
        if mode == "mtp" and not accepted:
            raise RuntimeError("MTP arm did not engage")
        return {
            "mode": mode,
            "tokens": len(tokens),
            "reported_tokens": last.generation_tokens,
            "decode_tps": float(last.generation_tps),
            "prompt_tps": float(last.prompt_tps),
            "token_sha256": hashlib.sha256(
                b"".join(t.to_bytes(4, "little") for t in tokens)
            ).hexdigest(),
            "text_sha256": hashlib.sha256(
                "".join(r.text for r in out).encode()
            ).hexdigest(),
            "accepted_drafts": sum(accepted),
            "drafted_tokens": sum(drafted),
            "speculative_rounds": len(accepted),
            "acceptance": sum(accepted) / sum(drafted) if sum(drafted) else None,
        }

    warm = _render(processor, PROMPTS["coding"])
    for mode in ["ar", "mtp", "mtp", "ar"]:
        run(warm, mode, 32)
    for case, text in PROMPTS.items():
        prompt = _render(processor, text)
        for rd in range(args.rounds):
            pair = {}
            for mode in ["ar", "mtp"] if rd % 2 == 0 else ["mtp", "ar"]:
                row = run(prompt, mode, args.max_tokens)
                row.update(case=case, round=rd)
                pair[mode] = row
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
            result["pairs"].append(
                {
                    "case": case,
                    "round": rd,
                    "token_exact": pair["ar"]["token_sha256"]
                    == pair["mtp"]["token_sha256"],
                    "text_exact": pair["ar"]["text_sha256"]
                    == pair["mtp"]["text_sha256"],
                    "speedup": pair["mtp"]["decode_tps"] / pair["ar"]["decode_tps"],
                }
            )
    result["summary"] = {
        "pairs": len(result["pairs"]),
        "all_tokens_exact": all(r["token_exact"] for r in result["pairs"]),
        "paired_median_speedup": statistics.median(
            r["speedup"] for r in result["pairs"]
        ),
        "positive_pairs": sum(r["speedup"] > 1 for r in result["pairs"]),
        "peak_gib": mx.get_peak_memory() / 2**30,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]), flush=True)
    if not result["summary"]["all_tokens_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
