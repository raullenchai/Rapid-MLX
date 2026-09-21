#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OOD/abstention analysis for the routing adapter (v15c).

Inputs: two per-sample dumps from eval_label_readout.py --dump
  - in-distribution dump (routing/tool_gate/injection_guard heldout)
  - OOD probe dump (spire prompts scored by the routing adapter)

Reports:
  1. confidence distributions (in-dist vs OOD)
  2. risk-coverage curve on the in-dist dump (abstain by low confidence)
  3. OOD detection at candidate thresholds: fraction of OOD refused vs
     fraction of in-dist refused (the tradeoff an abstention rule buys)

Caveat (per external review): the OOD probe here is FAR from the business
distribution (a game); near-OOD probes (ambiguous intent, missing candidates)
still need to be minted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def risk_coverage(rows: list[dict], coverages=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)) -> None:
    ranked = sorted(rows, key=lambda r: -float(r["confidence"]))
    n = len(ranked)
    print("  coverage -> residual error rate (abstain lowest-confidence first)")
    for c in coverages:
        k = max(1, int(n * c))
        kept = ranked[:k]
        err = 1 - sum(1 for r in kept if r["correct"]) / k
        print(f"    {c:>5.0%} -> {err:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indist", type=Path, required=True)
    ap.add_argument("--ood", type=Path, required=True)
    args = ap.parse_args()

    ind, ood = load(args.indist), load(args.ood)
    for name, rows in (("in-dist", ind), ("OOD(spire)", ood)):
        confs = sorted(float(r["confidence"]) for r in rows)
        n = len(confs)
        print(f"{name}: n={n} conf p10={confs[n//10]:.3f} p50={confs[n//2]:.3f} "
              f"p90={confs[9*n//10]:.3f} acc={sum(1 for r in rows if r['correct'])/n:.3f}")

    print("\n[risk-coverage on in-dist]")
    risk_coverage(ind)

    print("\n[OOD detection: threshold -> %OOD refused | %in-dist refused]")
    # rank-safe: sweep thresholds on the pooled confidence
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for t in thresholds:
        ood_ref = sum(1 for r in ood if float(r["confidence"]) < t) / len(ood)
        ind_ref = sum(1 for r in ind if float(r["confidence"]) < t) / len(ind)
        print(f"    conf<{t:.2f}: OOD refused {ood_ref:>5.1%} | in-dist refused {ind_ref:>5.1%}")

    # AUROC (OOD lower confidence than in-dist) via rank statistic
    all_pairs = [(float(r["confidence"]), 0) for r in ind] + [(float(r["confidence"]), 1) for r in ood]
    ranked = sorted(all_pairs, reverse=True)  # high conf first
    pos = sum(1 for _, y in ranked if y == 1)
    neg = len(ranked) - pos
    if pos and neg:
        rank_sum = sum(i + 1 for i, (_, y) in enumerate(ranked) if y == 1)
        auroc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
        print(f"\nAUROC(conf separates OOD from in-dist): {auroc:.3f} "
              f"(0.5=no separation, 1.0=perfect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
