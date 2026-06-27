# SPDX-License-Identifier: Apache-2.0
"""Phase A throughput frontier replay.

Given:
  - a real DFlash trace (per-round confidence + per-round accepted)
  - a profiled SPS(B) table

we compute, at each spec-decode round, the throughput of:
  - FIXED gamma = block_size - 1  (DFlash baseline used by the paper)
  - DYNAMIC gamma per Algorithm 1, with the captured TV-distance
    confidence as the oracle estimator

For a synthetic concurrency of R independent requests, we sample R
rounds from the trace pool (with replacement), run Algorithm 1 over the
joint confidence matrix, and report:

  - aggregate tok/sec  = sum_r(emitted_r) / round_wall_clock
  - per-user TPS       = mean(emitted_r) / round_wall_clock
  - round_wall_clock   = 1 / SPS(B_chosen)

Output: spike/dspark-mvp/bench/frontier.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithm1 import SPSTable, schedule_prefix_lengths  # noqa: E402


def expected_accepts(conf: list[float], ell: int) -> float:
    """Expected number of accepted draft tokens given verification length ell.

    Under the paper's lossless property: tau_r = 1 + sum_{i<=ell_r} a_{r,i}
    where a_{r,i} = prod_{j<=i} c_{r,j}. The +1 is the always-emitted
    bonus token.
    """
    if ell == 0:
        return 1.0
    prefix = 1.0
    acc = 1.0
    for j in range(min(ell, len(conf))):
        prefix *= conf[j]
        acc += prefix
    return acc


def emitted_under_schedule(rounds: list[dict], ell_per_round: list[int]) -> int:
    """Sum of emitted tokens across rounds for a given schedule.

    Each round emits (accepted_within_window + 1) tokens where the +1 is the
    bonus token at the truncation cut.
    """
    total = 0
    for r, ell in zip(rounds, ell_per_round):
        accepted_in_window = 0
        for k in range(ell):
            if r["draft_tokens"][k] == r["target_argmax"][k]:
                accepted_in_window += 1
            else:
                break
        total += accepted_in_window + 1
    return total


def fixed_gamma_schedule(rounds: list[dict], gamma: int) -> list[int]:
    return [min(gamma, r["block_size"] - 1) for r in rounds]


def algo1_schedule(rounds: list[dict], sps_table: SPSTable, gamma_cap: int) -> list[int]:
    """Run Algorithm 1 INDEPENDENTLY per round at R=1.

    For single-stream (R=1) we re-run Algorithm 1 per round on its
    captured confidence row.
    """
    schedule = []
    for r in rounds:
        n_draft = r["block_size"] - 1
        conf = r["confidence"][:n_draft]
        if gamma_cap < n_draft:
            conf = conf[:gamma_cap]
        ell, _ = schedule_prefix_lengths([conf], sps_table)
        schedule.append(min(ell[0], len(conf)))
    return schedule


def batched_frontier(
    rounds_pool: list[dict],
    sps_table: SPSTable,
    R: int,
    gamma_cap: int,
    n_trials: int = 200,
    seed: int = 42,
) -> dict:
    """Simulate R-concurrent requests by sampling rounds from the pool.

    Per trial:
      - sample R rounds from the pool (with replacement)
      - compute B for each schedule:
          fixed_B = R * (1 + gamma)
          algo1_B = R + sum(ell_r)
      - compute step time = 1 / SPS(B)
      - compute emitted_r per request, sum/mean across R, divide by step time

    Returns aggregate tok/sec, per-user TPS, and Algorithm-1 win ratios.
    """
    rng = random.Random(seed)
    fixed_tps_aggs = []
    fixed_tps_per_user = []
    algo1_tps_aggs = []
    algo1_tps_per_user = []
    ell_chosen_distribution = []

    for _ in range(n_trials):
        sample = [rng.choice(rounds_pool) for _ in range(R)]

        # ---- Fixed gamma -------------------------------------------
        fixed_gamma = min(gamma_cap, sample[0]["block_size"] - 1)
        fixed_emitted = [
            emitted_under_schedule([r], [fixed_gamma]) for r in sample
        ]
        fixed_B = R * (1 + fixed_gamma)
        fixed_step_sec = 1.0 / sps_table(fixed_B)
        fixed_agg = sum(fixed_emitted) / fixed_step_sec
        fixed_per_user = statistics.mean(fixed_emitted) / fixed_step_sec

        # ---- Algorithm 1 -------------------------------------------
        conf_matrix = []
        for r in sample:
            row = r["confidence"][: r["block_size"] - 1]
            if gamma_cap < len(row):
                row = row[:gamma_cap]
            conf_matrix.append(row)
        ell_star, theta_best = schedule_prefix_lengths(conf_matrix, sps_table)
        algo1_emitted = [
            emitted_under_schedule([r], [ell])
            for r, ell in zip(sample, ell_star)
        ]
        algo1_B = R + sum(ell_star)
        algo1_step_sec = 1.0 / sps_table(algo1_B)
        algo1_agg = sum(algo1_emitted) / algo1_step_sec
        algo1_per_user = statistics.mean(algo1_emitted) / algo1_step_sec

        fixed_tps_aggs.append(fixed_agg)
        fixed_tps_per_user.append(fixed_per_user)
        algo1_tps_aggs.append(algo1_agg)
        algo1_tps_per_user.append(algo1_per_user)
        ell_chosen_distribution.extend(ell_star)

    return {
        "R": R,
        "gamma_cap": gamma_cap,
        "n_trials": n_trials,
        "fixed_agg_tps_mean": statistics.mean(fixed_tps_aggs),
        "fixed_agg_tps_median": statistics.median(fixed_tps_aggs),
        "fixed_per_user_tps_mean": statistics.mean(fixed_tps_per_user),
        "algo1_agg_tps_mean": statistics.mean(algo1_tps_aggs),
        "algo1_agg_tps_median": statistics.median(algo1_tps_aggs),
        "algo1_per_user_tps_mean": statistics.mean(algo1_tps_per_user),
        "delta_agg_pct": 100.0
        * (statistics.mean(algo1_tps_aggs) - statistics.mean(fixed_tps_aggs))
        / statistics.mean(fixed_tps_aggs),
        "delta_per_user_pct": 100.0
        * (statistics.mean(algo1_tps_per_user) - statistics.mean(fixed_tps_per_user))
        / statistics.mean(fixed_tps_per_user),
        "ell_chosen_mean": statistics.mean(ell_chosen_distribution),
        "ell_chosen_median": statistics.median(ell_chosen_distribution),
        "ell_chosen_p25": sorted(ell_chosen_distribution)[
            len(ell_chosen_distribution) // 4
        ],
        "ell_chosen_p75": sorted(ell_chosen_distribution)[
            3 * len(ell_chosen_distribution) // 4
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trace",
        default="spike/dspark-mvp/bench/trace_qwen3.5-9b-4bit.json",
    )
    p.add_argument("--sps", default="spike/dspark-mvp/sps_table.json")
    p.add_argument("--Rs", nargs="+", type=int, default=[1, 10, 50, 100])
    p.add_argument("--gamma-caps", nargs="+", type=int, default=[5, 8, 15])
    p.add_argument("--n-trials", type=int, default=300)
    p.add_argument(
        "--out",
        default="spike/dspark-mvp/bench/frontier.json",
    )
    args = p.parse_args()

    trace_blob = json.loads(Path(args.trace).read_text())
    sps_blob = json.loads(Path(args.sps).read_text())
    # Use the smoothed (monotone non-increasing) envelope per the
    # paper's assumption — Section 3.2.2 "smoothly decaying hardware
    # capacity curve". The raw measurements are noisy enough on
    # MLX/Metal small-B forwards that an unsmoothed SPS table breaks
    # Algorithm 1's marginal-throughput logic.
    rows = sps_blob.get("results_smoothed", sps_blob["results"])
    sps_table = SPSTable(
        batch_sizes=tuple(r["B"] for r in rows),
        sps=tuple(r["sps"] for r in rows),
    )

    # Pool of rounds across all traced prompts
    rounds_pool: list[dict] = []
    for t in trace_blob["traces"]:
        rounds_pool.extend(t["rounds"])
    print(f"[bench] rounds_pool size: {len(rounds_pool)}")

    # Single-stream (R=1) baseline summary — uses REAL trace order
    print()
    print("=== Single-stream (R=1) summary ===")
    for gamma_cap in args.gamma_caps:
        for t in trace_blob["traces"]:
            rounds = t["rounds"]
            fixed_sched = fixed_gamma_schedule(rounds, gamma_cap)
            algo1_sched = algo1_schedule(rounds, sps_table, gamma_cap)
            fixed_emit = emitted_under_schedule(rounds, fixed_sched)
            algo1_emit = emitted_under_schedule(rounds, algo1_sched)
            # Mean B for the round
            fixed_B_mean = statistics.mean(1 + e for e in fixed_sched)
            algo1_B_mean = statistics.mean(1 + e for e in algo1_sched)
            fixed_step_mean = statistics.mean(1.0 / sps_table(1 + e) for e in fixed_sched)
            algo1_step_mean = statistics.mean(1.0 / sps_table(1 + e) for e in algo1_sched)
            fixed_tok_per_sec = fixed_emit / (fixed_step_mean * len(rounds))
            algo1_tok_per_sec = algo1_emit / (algo1_step_mean * len(rounds))
            print(
                f"[bench] gamma_cap={gamma_cap}  "
                f"prompt='{t['prompt'][:30]}...'  "
                f"fixed: {fixed_emit} tok ({fixed_tok_per_sec:.1f}/s)  "
                f"algo1: {algo1_emit} tok ({algo1_tok_per_sec:.1f}/s)  "
                f"Δagg={100*(algo1_tok_per_sec-fixed_tok_per_sec)/fixed_tok_per_sec:+.1f}%  "
                f"ell_mean={statistics.mean(algo1_sched):.1f}"
            )

    # Batched frontier across (R, gamma_cap)
    print()
    print("=== Batched frontier (simulated) ===")
    frontier = []
    for R in args.Rs:
        for gamma_cap in args.gamma_caps:
            row = batched_frontier(
                rounds_pool, sps_table, R, gamma_cap, n_trials=args.n_trials
            )
            frontier.append(row)
            print(
                f"[bench] R={R:3d}  gamma_cap={gamma_cap}  "
                f"fixed_agg_tps={row['fixed_agg_tps_mean']:.1f}  "
                f"algo1_agg_tps={row['algo1_agg_tps_mean']:.1f}  "
                f"Δagg={row['delta_agg_pct']:+.2f}%  "
                f"Δper_user={row['delta_per_user_pct']:+.2f}%  "
                f"ell_med={row['ell_chosen_median']}"
            )

    Path(args.out).write_text(json.dumps({
        "sps_source": args.sps,
        "trace_source": args.trace,
        "frontier": frontier,
    }, indent=2))
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
