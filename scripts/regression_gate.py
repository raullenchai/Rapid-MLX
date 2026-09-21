#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regression gate for Marvin's Garden adapters.

Reads one or more eval JSONs (eval_label_readout.py --output) and fails
(non-zero exit) if any family accuracy or the macro average drops below the
production floor, or if calibration degrades. Wire into training so that
"add a lane" can never silently regress an existing one:

  python scripts/regression_gate.py \
      --eval results/eval_marvin_v15c.json [--eval ...] \
      [--floor model_routing=0.90 --floor tool_gate=0.90 \
       --floor injection_guard=0.97 --floor macro=0.92] \
      [--max-ece 0.08]

Exit codes: 0 pass, 1 gate failure, 2 usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FLOORS = {
    "model_routing": 0.90,
    "tool_gate": 0.90,
    "injection_guard": 0.97,
    "macro": 0.92,
}
DEFAULT_MAX_ECE = 0.08


def parse_floors(pairs: list[str]) -> dict[str, float]:
    floors = {}
    for p in pairs:
        name, _, val = p.partition("=")
        floors[name.strip()] = float(val)
    return floors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", action="append", required=True, type=Path)
    ap.add_argument("--floor", action="append", default=[],
                    help="family=threshold (repeatable); macro supported")
    ap.add_argument("--max-ece", type=float, default=DEFAULT_MAX_ECE)
    args = ap.parse_args()

    try:
        floors = {**DEFAULT_FLOORS, **parse_floors(args.floor)}
    except ValueError as e:
        print(f"bad --floor: {e}", file=sys.stderr)
        return 2

    failures: list[str] = []
    for path in args.eval:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"FAIL {path}: unreadable ({e})", file=sys.stderr)
            return 2
        families = data.get("per_family") or {}
        accs = {}
        for fam, spec in sorted(families.items()):
            acc = float(spec["accuracy"])
            accs[fam] = acc
            floor = floors.get(fam)
            mark = "ok" if floor is None or acc >= floor else "FAIL"
            line = f"{path.name}: {fam:<18} acc={acc:.4f} (n={spec.get('n', '?')}) floor={floor} {mark}"
            print(line)
            if mark == "FAIL":
                failures.append(line)
        if accs:
            macro = sum(accs.values()) / len(accs)
            mf = floors.get("macro")
            mark = "ok" if mf is None or macro >= mf else "FAIL"
            print(f"{path.name}: {'macro':<18} acc={macro:.4f} floor={mf} {mark}")
            if mark == "FAIL":
                failures.append(f"macro {path.name}")
        ece = data.get("ece_15bin")
        if ece is not None:
            mark = "ok" if ece <= args.max_ece else "FAIL"
            print(f"{path.name}: {'ece_15bin':<18} {ece:.4f} max={args.max_ece} {mark}")
            if mark == "FAIL":
                failures.append(f"ece {path.name}")

    if failures:
        print(f"\nGATE FAILED ({len(failures)} violation(s))")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
