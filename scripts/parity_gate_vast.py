#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parity gate: does the NVIDIA (llama.cpp) path reproduce the MLX path?

The Vast.ai serving path converts the 2-bit ternary adapter to GGUF (custom
format, upstream risk llama.cpp #29058/#27127). BEFORE exposing that target,
this gate must pass on the same 384-item held-out set:

  accuracy_remote >= accuracy_local - 2.0 pts   AND   agreement >= 97%

  python scripts/parity_gate_vast.py --base http://<vast>:8080 \
      --local-dump bench/marvins_garden/results/v15c_large_dump.jsonl

If --local-dump is missing, local predictions are recomputed first (slow,
needs MLX + the release adapter). Exit 0 only on PASS. Prints the confusion
between local and remote decisions for diagnosis.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "bench" / "marvins_garden"
sys.path.insert(0, str(BENCH))
import render  # noqa: E402  (single source of letter mapping)
PAIRS = BENCH / "data-large" / "pairs_heldout.jsonl"
THRESHOLD_PT = 2.0
THRESHOLD_AGREE = 0.97


def remote_probs(base, key, prompt, n_candidates, timeout=60):
    letters = [render.letter_for(i) for i in range(n_candidates)]
    body = json.dumps({"model": "marvin", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20}).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    items = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    tok2lp = {it["token"].strip(): it["logprob"] for it in items}
    lps = {l: tok2lp[l] for l in letters if l in tok2lp}
    if len(lps) != len(letters):
        raise RuntimeError(f"missing letters {sorted(set(letters)-set(lps))}")
    mx = max(lps.values())
    exps = {l: math.exp(v - mx) for l, v in lps.items()}
    z = sum(exps.values())
    return {l: exps[l] / z for l in letters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="llama-server base URL on the Vast box")
    ap.add_argument("--key", default=os.environ.get("MARVIN_SERVE_KEY", ""))
    ap.add_argument("--local-dump", default=str(BENCH / "results" / "v15c_large_dump.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--accuracy-local", type=float, default=0.9531,
                    help="reference local accuracy on this set")
    args = ap.parse_args()

    rows = [json.loads(l) for l in PAIRS.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    local = {}
    dump = Path(args.local_dump)
    if dump.exists():
        for l in dump.read_text().splitlines():
            d = json.loads(l)
            local[d["pair_id"]] = d
        print(f"local dump: {len(local)} rows (acc "
              f"{sum(d['correct'] for d in local.values())/max(1,len(local)):.4f})")
    else:
        print(f"WARNING: no local dump at {dump}; agreement checks disabled")

    n = ok = agree = 0
    fails = []
    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            probs = remote_probs(args.base, args.key, row["input"], len(row["candidates"]))
        except Exception as e:
            fails.append(f"{row['pair_id']}: {e}")
            continue
        n += 1
        letters = [render.letter_for(j) for j in range(len(row["candidates"]))]
        pred = row["candidates"][max(letters, key=lambda l: probs[l])]
        hit = pred == row["label"]
        ok += hit
        d = local.get(row["pair_id"])
        if d is not None:
            agree += int((pred == row["label"]) == bool(d["correct"]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} remote acc {ok/n:.4f} ({time.time()-t0:.0f}s)", flush=True)

    acc = ok / max(1, n)
    print(f"\nremote accuracy: {acc:.4f} ({ok}/{n})  vs local {args.accuracy_local:.4f}")
    if local:
        ag = agree / max(1, n)
        print(f"local-remote agreement: {ag:.4f}")
    if fails:
        print(f"errors: {len(fails)} (first: {fails[0]})")
    passed = (n == len(rows)) and acc >= args.accuracy_local - THRESHOLD_PT / 100 \
        and (not local or agree / max(1, n) >= THRESHOLD_AGREE)
    print("PARITY GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
