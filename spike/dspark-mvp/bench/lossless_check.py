# SPDX-License-Identifier: Apache-2.0
"""Lossless byte-identical check at temp=0.

For Phase A we need to confirm two properties:

  1. Fixed-gamma DFlash produces the same token sequence as a vanilla
     no-spec-decode greedy run.
  2. Algorithm-1-truncated DFlash also produces the same sequence
     (truncating low-confidence draft positions cannot violate the
     lossless contract — every emitted token is target's greedy argmax).

We run plain mlx-lm-style greedy decode + the DFlash trace from
trace_dflash.py + a replay that truncates the DFlash trace per
Algorithm 1, then compare byte-by-byte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_vlm.utils import load


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm1 import SPSTable, schedule_prefix_lengths  # noqa: E402


def vanilla_greedy(lm, prompt_ids: mx.array, max_tokens: int) -> list[int]:
    """Plain decode at temp=0 — the reference sequence."""
    cache = make_prompt_cache(lm)
    out = lm(prompt_ids[None], cache=cache)
    tok = int(mx.argmax(out.logits[0, -1], axis=-1).item())
    emitted = [tok]
    while len(emitted) < max_tokens:
        x = mx.array([[tok]], dtype=prompt_ids.dtype)
        out = lm(x, cache=cache)
        tok = int(mx.argmax(out.logits[0, -1], axis=-1).item())
        emitted.append(tok)
    return emitted


def replay_fixed_gamma(trace: dict) -> list[int]:
    """Reconstruct the fixed-gamma DFlash output sequence from a trace."""
    out = [trace["first_token"]]
    for r in trace["rounds"]:
        n_draft = r["block_size"] - 1
        accepted = r["accepted"]
        d = r["draft_tokens"]
        t = r["target_argmax"]
        if accepted < n_draft:
            out.extend(d[:accepted] + [t[accepted]])
        else:
            out.extend(d[:] + [t[n_draft]])
    return out


def replay_algo1(trace: dict, sps_table: SPSTable, gamma_cap: int) -> list[int]:
    """Reconstruct the Algorithm-1-truncated DFlash output.

    For every round, run Algorithm 1 with R=1 over the confidence row.
    If the scheduler chooses ell* < n_draft, we *truncate* the verify
    block at position ell* — the bonus is the target argmax at position
    ell* (which is t_list[ell*]). Truncation only DROPS the accepted
    suffix beyond ell*; the lossless contract holds because every
    emitted token is target's greedy argmax (either a matched-by-draft
    accepted token, or the target-argmax bonus at the truncation cut).
    """
    out = [trace["first_token"]]
    for r in trace["rounds"]:
        n_draft = r["block_size"] - 1
        conf = r["confidence"][:n_draft]
        if gamma_cap < n_draft:
            conf = conf[:gamma_cap]
        ell_star, _ = schedule_prefix_lengths([conf], sps_table)
        ell = min(ell_star[0], len(conf))
        # Actual draft acceptance up to ell:
        accepted_in_window = 0
        for k in range(ell):
            if r["draft_tokens"][k] == r["target_argmax"][k]:
                accepted_in_window += 1
            else:
                break
        d = r["draft_tokens"]
        t = r["target_argmax"]
        if accepted_in_window < ell:
            # Partial accept within truncated window
            out.extend(d[:accepted_in_window] + [t[accepted_in_window]])
        elif ell < n_draft:
            # Truncated: emit all ell accepted draft tokens + target's argmax at ell
            out.extend(d[:ell] + [t[ell]])
        elif accepted_in_window < n_draft:
            # Full window admitted; first mismatch beyond window
            out.extend(d[:accepted_in_window] + [t[accepted_in_window]])
        else:
            out.extend(d[:] + [t[n_draft]])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="mlx-community/Qwen3.5-9B-4bit")
    p.add_argument(
        "--trace", default="spike/dspark-mvp/bench/trace_qwen3.5-9b-4bit.json"
    )
    p.add_argument("--sps", default="spike/dspark-mvp/sps_table.json")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--gamma-cap", type=int, default=5)
    p.add_argument(
        "--out",
        default="spike/dspark-mvp/bench/lossless_report.json",
    )
    args = p.parse_args()

    trace_blob = json.loads(Path(args.trace).read_text())
    sps_blob = json.loads(Path(args.sps).read_text())
    sps_table = SPSTable(
        batch_sizes=tuple(r["B"] for r in sps_blob["results"]),
        sps=tuple(r["sps"] for r in sps_blob["results"]),
    )

    print(f"[lossless] loading {args.target}")
    model, processor = load(args.target)
    model.eval()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    lm = model.language_model

    report = []
    all_ok_fixed = True
    all_ok_algo1 = True
    for trace in trace_blob["traces"]:
        prompt = trace["prompt"]
        ids = tokenizer.encode(prompt)
        ids_arr = mx.array(ids, dtype=mx.int32)
        ref = vanilla_greedy(lm, ids_arr, args.max_tokens)
        fixed = replay_fixed_gamma(trace)[: args.max_tokens]
        algo1 = replay_algo1(trace, sps_table, args.gamma_cap)[: args.max_tokens]

        # Find first divergence (if any)
        def first_div(a, b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return i
            return len(a) if len(a) != len(b) else -1

        ref_short = ref[: min(len(ref), len(fixed), len(algo1))]
        fixed_short = fixed[: len(ref_short)]
        algo1_short = algo1[: len(ref_short)]
        d_fixed = first_div(ref_short, fixed_short)
        d_algo1 = first_div(ref_short, algo1_short)
        ok_fixed = d_fixed < 0
        ok_algo1 = d_algo1 < 0
        all_ok_fixed = all_ok_fixed and ok_fixed
        all_ok_algo1 = all_ok_algo1 and ok_algo1
        print(
            f"[lossless] prompt='{prompt[:40]}...'  "
            f"len(ref)={len(ref_short)}  "
            f"fixed={'OK' if ok_fixed else f'DIV@{d_fixed}'}  "
            f"algo1={'OK' if ok_algo1 else f'DIV@{d_algo1}'}"
        )
        report.append(
            {
                "prompt": prompt[:80],
                "n_compared": len(ref_short),
                "fixed_gamma_ok": ok_fixed,
                "fixed_gamma_div_at": None if ok_fixed else d_fixed,
                "algo1_ok": ok_algo1,
                "algo1_div_at": None if ok_algo1 else d_algo1,
            }
        )

    summary = {
        "all_fixed_ok": all_ok_fixed,
        "all_algo1_ok": all_ok_algo1,
        "per_prompt": report,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(
        f"[lossless] OVERALL: fixed={'PASS' if all_ok_fixed else 'FAIL'}  "
        f"algo1={'PASS' if all_ok_algo1 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
