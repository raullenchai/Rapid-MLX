#!/usr/bin/env python3
"""Same-ruler cross-eval: run TypeSafe Jev on Marvin's Garden held-out set.

Protocol (documented in docs/engineering/performance/2026-09-19-marvins-garden-mvp-signal.md):
- Same items: bench/marvins_garden/data/pairs_heldout.jsonl (192 samples).
- Same information: the model sees the identical policy + facts text. The
  letter menu ("Options: A. name — desc") is translated into Jev's native
  `choice` criteria {name: description}; the state carries the rest verbatim.
- Scoring: argmax choice vs label, confidence = selected probability,
  ECE 15-bin, per-family accuracy — identical formulas to eval_label_readout.
- Latency is wall-clock per API request (includes network RTT to the hosted
  endpoint); Marvin's latency is local inference. Not hardware-comparable —
  recorded for order-of-magnitude context only.

Usage: JEVAI_KEY=... python cross_eval_jev.py [--model jev-latest] [--limit N]
The key is read from the environment only and never written to disk by this
script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "data" / "pairs_heldout.jsonl"
MENU_LINE = re.compile(r"^([A-H])\. (\S+) — (.*)$")


def split_input(prompt: str) -> tuple[str, dict[str, str], str | None]:
    """Split a rendered prompt into (state, {choice: description}, task_line)."""
    if "\nOptions:\n" in prompt:
        head, menu = prompt.split("\nOptions:\n", 1)
    else:
        head, menu = prompt, ""
    criteria: dict[str, str] = {}
    for line in menu.splitlines():
        m = MENU_LINE.match(line.strip())
        if m:
            criteria[m.group(2)] = m.group(3).strip()
    task_line = None
    for line in head.splitlines():
        low = line.lower()
        if low.startswith(("pick the", "decide whether", "decide if", "decide the")):
            task_line = line.strip()
            break
    return head.strip(), criteria, task_line


def build_question(row: dict) -> dict:
    _, criteria, task_line = split_input(row["input"])
    if row["family"] == "model_routing":
        instructions = task_line or "Pick the single best serving alias for the request."
    elif row["family"] == "tool_gate":
        instructions = task_line or "Decide whether the agent should call a tool or answer directly."
    else:
        instructions = (
            task_line
            or "Decide whether the agent may process this content normally or must block it."
        )
    if not criteria:
        # binary rows may lack a letter menu; use the candidates bare
        criteria = {c: "" for c in row["candidates"]}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def call_jev(model: str, state: str, question: dict, key: str, retries: int = 4) -> tuple[dict, float]:
    payload = {"model": model, "state": state, "questions": {"q": question}}
    body = json.dumps(payload).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                "https://api.typesafe.ai/v1/systemone",
                data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.load(resp)
            return out, time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Jev API failed after {retries} attempts: {last_err}")


def ece_15bin(conf: list[float], correct: list[bool]) -> float:
    """15-bin expected calibration error (same formula as eval_label_readout)."""
    bins: dict[int, list[int]] = {}
    for c, ok in zip(conf, correct):
        bins.setdefault(min(int(c * 15), 14), [0, 0])[0 if ok else 1] += 1
    total = len(conf)
    e = 0.0
    for _, (n_ok, n_bad) in sorted(bins.items()):
        acc = n_ok / (n_ok + n_bad)
        center = (_ + 0.5) / 15
        e += ((n_ok + n_bad) / total) * abs(acc - center)
    return e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default=str(HERE / "results" / "cross_eval_jev.json"))
    args = ap.parse_args(argv)

    key = os.environ.get("JEVAI_KEY")
    if not key:
        print("error: set JEVAI_KEY (never hardcode keys)", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in open(args.pairs)]
    if args.limit:
        rows = rows[: args.limit]

    def work(row: dict):
        state, _, _ = split_input(row["input"])
        q = build_question(row)
        out, dt = call_jev(args.model, state, q, key)
        ans = out["answers"]["q"]
        probs = ans.get("probabilities") or {}
        pick = ans.get("choice")
        conf = float(ans.get("confidence") or probs.get(pick, 0.0))
        usage = out.get("usage") or {}
        return row, pick, conf, probs, dt, usage

    results, latencies = [], []
    in_tok = out_tok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, pick, conf, probs, dt, usage in pool.map(work, rows):
            ok = pick == row["label"]
            latencies.append(dt)
            results.append({
                "pair_id": row["pair_id"], "family": row["family"],
                "label": row["label"], "picked": pick, "correct": ok,
                "confidence": conf, "latency_s": round(dt, 3),
            })
            u = usage
            in_tok += u.get("input_tokens", 0)
            out_tok += u.get("output_tokens", 0)

    by_fam: dict[str, list[bool]] = {}
    conf_all, ok_all = [], []
    for r in results:
        by_fam.setdefault(r["family"], []).append(r["correct"])
        conf_all.append(r["confidence"])
        ok_all.append(r["correct"])
    acc = sum(ok_all) / len(ok_all)
    summary = {
        "model": args.model,
        "n_samples": len(results),
        "accuracy": acc,
        "ece_15bin": round(ece_15bin(conf_all, ok_all), 4),
        "per_family": {f: round(sum(v) / len(v), 4) for f, v in sorted(by_fam.items())},
        "latency_s_p50": round(statistics.median(latencies), 3),
        "latency_s_p95": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 3),
        "latency_s_mean": round(statistics.mean(latencies), 3),
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "protocol": "same items+information; letter menu -> native choice criteria",
    }
    Path(args.output).write_text(json.dumps({"summary": summary, "records": results}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
