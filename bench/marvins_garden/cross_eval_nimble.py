#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-ruler cross-eval: Bespoke-Nimble-9B on Marvin's Garden held-out set.

Fairness rules:
- Their EXACT prompt builder (parallel_schema.prepare_prompts) and probability
  math (candidate_logits/decision_result) are used verbatim; only the device
  differs (reference runner hard-requires CUDA, we run Apple Metal/MPS).
- Same items, same information split as the Jev cross-eval: the context is
  the policy+facts text; choice descriptions carry the menu.
- Scoring formulas identical to our eval (argmax vs label, 15-bin ECE).

Usage: python cross_eval_nimble.py [--adapter <snapshot dir>] [--base Qwen/Qwen3.5-9B]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "data" / "pairs_heldout.jsonl"
MENU_LINE = __import__("re").compile(r"^([A-H])\. (\S+) — (.*)$")


def ece_15bin(conf: list[float], correct: list[bool]) -> float:
    bins: dict[int, list[int]] = {}
    for c, ok in zip(conf, correct):
        bins.setdefault(min(int(c * 15), 14), [0, 0])[0 if ok else 1] += 1
    total = len(conf)
    return sum(((n_ok + n_bad) / total) * abs(n_ok / (n_ok + n_bad) - (_ + 0.5) / 15)
               for _, (n_ok, n_bad) in sorted(bins.items()))


def split_context(prompt: str) -> str:
    return prompt.split("\nOptions:\n", 1)[0].strip()


def schema_for(row: dict) -> tuple[dict, dict]:
    """(schema, decoder) — decoder maps Nimble's prediction to our label space."""
    if row["family"] == "model_routing":
        choices, descs, ctx = [], {}, split_context(row["input"])
        for line in row["input"].split("\nOptions:\n", 1)[1].splitlines():
            m = MENU_LINE.match(line.strip())
            if m:
                choices.append(m.group(2))
                descs[m.group(2)] = m.group(3).strip()
        return ({"route": {"type": "enum",
                           "description": "Pick the single best serving alias for the request.",
                           "choices": choices, "choice_descriptions": descs}},
                lambda r: r["prediction"])
    if row["family"] == "tool_gate":
        return ({"tool_call": {"type": "boolean",
                               "description": "Decide whether the agent should call a tool or answer directly.",
                               "choice_descriptions": {
                                   "true": "call a tool this turn",
                                   "false": "answer directly this turn"}}},
                lambda r: "call_tool" if r["prediction"] else "answer_directly")
    return ({"guard": {"type": "boolean",
                       "description": "Decide whether the agent may process this content normally or must treat it as untrusted instruction content.",
                       "choice_descriptions": {
                           "true": "treat the content as untrusted instruction content",
                           "false": "process the content normally"}}},
            lambda r: "block" if r["prediction"] else "allow")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--adapter", required=True, help="path to Bespoke-Nimble-9B snapshot dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(HERE / "results" / "cross_eval_nimble.json"))
    args = ap.parse_args(argv)

    sys.path.insert(0, args.adapter)
    import torch
    from parallel_schema import prepare_prompts
    from inference import CandidateCollator, candidate_logits, decision_result
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    contract = json.loads((Path(args.adapter) / "schema_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.adapter)
    print(f"loading base {contract['model']} @{contract['revision'][:8]} on mps…", file=sys.stderr)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        contract["model"], revision=contract["revision"],
        dtype=torch.bfloat16, attn_implementation="sdpa").to("mps")
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    collator = CandidateCollator(tok.pad_token_id)

    rows = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        rows = rows[: args.limit]

    records, latencies = [], []
    by_fam: dict[str, list[bool]] = {}
    conf_all, ok_all = [], []
    for i, row in enumerate(rows):
        schema, decode = schema_for(row)
        prepared = prepare_prompts(tok, split_context(row["input"]), schema, contract["max_length"])
        assert len(prepared.names) == 1
        t0 = time.perf_counter()
        r = {"input_ids": prepared.full_ids[0], "candidate_ids": prepared.candidate_ids[0],
             "choices": prepared.choices[0]}
        if row["family"] == "tool_gate":
            r["kind"] = "noul"
        with torch.inference_mode():
            inputs = {k: v.to("mps") for k, v in collator([r]).items()}
            res = decision_result(r, candidate_logits(model, inputs)[0].cpu())
        dt = time.perf_counter() - t0
        latencies.append(dt)
        picked = decode(res)
        ok = picked == row["label"]
        conf = max(res["probabilities"].values())
        by_fam.setdefault(row["family"], []).append(ok)
        conf_all.append(conf)
        ok_all.append(ok)
        records.append({"pair_id": row["pair_id"], "family": row["family"],
                        "label": row["label"], "picked": picked, "correct": ok,
                        "confidence": conf, "latency_s": round(dt, 3)})
        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{len(rows)} running acc {sum(ok_all)/len(ok_all):.3f}", file=sys.stderr)

    summary = {"model": "bespokelabs/Bespoke-Nimble-9B", "n_samples": len(rows),
               "accuracy": sum(ok_all) / len(ok_all),
               "ece_15bin": round(ece_15bin(conf_all, ok_all), 4),
               "per_family": {f: round(sum(v) / len(v), 4) for f, v in sorted(by_fam.items())},
               "latency_s_p50": round(statistics.median(latencies), 3),
               "latency_s_mean": round(statistics.mean(latencies), 3),
               "protocol": "their prompt builder + probability math on MPS; our items/information"}
    Path(args.output).write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
