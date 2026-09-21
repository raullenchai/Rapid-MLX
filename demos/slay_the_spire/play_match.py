#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Headless match recorder: Marvin plays full battles, every decision logged.

Writes a match JSONL (one line per decision: state before + decision after)
used by replay_video.py to render the deliverable video without a browser.

  MARVIN_ADAPTER=... python play_match.py --battles 3 --out match.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "bench" / "marvins_garden"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH))
sys.path.insert(0, "/tmp/sts_lightspeed/build")

import slaythespire as sts  # noqa: E402
import server as srv  # noqa: E402  (reuses Session)
import marvin_spire  # noqa: E402


def play_battle(seed: int, out) -> dict:
    sess = srv.Session(seed=seed)
    srv.NAMER.learn_from(sess.bc)
    steps = 0
    while sess.bc.outcome == 0 and steps < 60:
        state = sess.snapshot()
        entries = gs_semantic(sess)
        result = sess.decide_and_apply()
        if "decision" not in result:
            break
        out.write(json.dumps({"seed": seed, "step": steps, "state": state,
                              "decision": result["decision"],
                              "probabilities": result["probabilities"]},
                             ensure_ascii=False) + "\n")
        steps += 1
    snap = sess.snapshot()
    summary = {"seed": seed, "steps": steps, "victory": snap["victory"],
               "hp_left": snap["player"]["hp"], "stats": sess.stats()}
    out.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    return summary


def gs_semantic(sess):
    import generate_spire as gs
    return gs.semantic_actions(sess.bc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=777)
    ap.add_argument("--out", default=str(HERE / "match.jsonl"))
    args = ap.parse_args()
    marvin_spire.load()
    wins = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for b in range(args.battles):
            s = play_battle(args.seed0 + b, out)
            wins += int(s["victory"])
            print(f"battle {b}: {'WIN' if s['victory'] else 'LOSS'} "
                  f"hp {s['hp_left']} · {s['steps']} decisions · "
                  f"oracle agreement {s['stats']['oracle_agreement']}")
    print(f"{wins}/{args.battles} victories → {args.out}")
