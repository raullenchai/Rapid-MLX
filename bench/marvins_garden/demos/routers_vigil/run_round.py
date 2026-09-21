#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Play one full round with the real model and write events.jsonl.

  python web/run_round.py --out docs/demo/vigil_round.jsonl [--seed 7] [--n 26]

This is the recording source for the gameplay video: every decision is the
real model's single forward pass.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import server  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/demo/vigil_round.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=26)
    ap.add_argument("--model", default="/Volumes/NVMe-4T/huggingface/hub/models--prism-ml--"
                    "Ternary-Bonsai-27B-mlx-2bit/snapshots/70f75f3ad081ab840a42f3304c02c27e7f89bfb7")
    ap.add_argument("--adapter", default=str(Path(__file__).resolve().parents[2]
                                             / "adapters" / "release" / "marvins-garden-v15c"))
    args = ap.parse_args(argv)

    import mlx_lm
    print("loading model…", file=sys.stderr)
    server.STATE["model"], server.STATE["tokenizer"] = mlx_lm.load(args.model, adapter_path=args.adapter)

    events = server.run_round(args.seed, args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    acc = sum(e["correct"] for e in events) / len(events)
    by_fam: dict[str, list[bool]] = {}
    for e in events:
        by_fam.setdefault(e["kind"], []).append(e["correct"])
    ms = [e["decision"]["ms"] for e in events]
    breaches = sum(1 for e in events if not e["correct"])
    print(json.dumps({
        "events": len(events), "accuracy": round(acc, 4), "breaches": breaches,
        "per_family": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_fam.items())},
        "ms_p50": round(statistics.median(ms)), "out": str(out),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
