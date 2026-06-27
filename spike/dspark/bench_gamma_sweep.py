# SPDX-License-Identifier: Apache-2.0
"""Follow-up to bench_overhead.py — γ sweep on measured M3 Ultra SPS.

After bench_overhead.py revealed that the scheduler is NOT a no-op at
R=1 on the measured SPS curve (chooses ell_1*=2 over the default
gamma=8), this script extends to DFlash's actual default block size
(gamma=16, per vllm_mlx/spec_decode/dflash/__init__.py:103) and several
gammas in between, plus runs a "static gamma retune" comparison.

The point of the comparison: if globally setting gamma=2 captures the
same throughput as the dynamic scheduler at R=1, the scheduler adds
nothing over a one-line config tune.

Output: spike/dspark/results_gamma_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spike.dspark.algorithm import schedule


def _decaying_confidences(gamma: int, c0: float, decay: float) -> list[float]:
    """Approximate parallel-drafter Figure-2 conditional acceptance:
    c_k = max(0.4, c0 - decay*k).
    """
    return [max(0.4, c0 - decay * k) for k in range(gamma)]


PROFILES = {
    # (c0, decay) calibrated to roughly match paper Figure 2 DFlash
    # curves for each domain on Qwen3-4B target.
    "chat": (0.72, 0.025),
    "code": (0.87, 0.013),
    "math": (0.88, 0.010),
}


def _make_sps(profile_path: Path):
    data = json.loads(profile_path.read_text())
    sps_profile = {int(k): v for k, v in data["sps_profile"].items()}
    xs = sorted(sps_profile.keys())
    ys = [sps_profile[x] for x in xs]

    def _sps(b: int) -> float:
        if b <= xs[0]:
            return ys[0]
        if b >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= b <= xs[i + 1]:
                w = (b - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] * (1 - w) + ys[i + 1] * w
        return ys[-1]

    return _sps, sps_profile


def _theta_at(confs: list[float], ell: int, sps) -> float:
    """Closed-form throughput if we force ell admissions for a single
    request: tau* = 1 + sum_{j<=ell} a_j; B = 1 + ell.
    """
    running = 1.0
    tau = 1.0
    for j in range(ell):
        running *= confs[j]
        tau += running
    return tau * sps(1 + ell)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sps-source",
        type=Path,
        default=Path("spike/dspark/results.json"),
        help="JSON file with measured SPS profile (from bench_overhead.py)",
    )
    p.add_argument("--out", type=Path, default=Path("spike/dspark/results_gamma_sweep.json"))
    args = p.parse_args(argv)

    sps_callable, sps_profile = _make_sps(args.sps_source)

    out: dict = {
        "sps_profile_source": str(args.sps_source),
        "sps_profile": sps_profile,
        "comparisons": {},
    }

    # For each profile and gamma, compute:
    #   (a) static-gamma throughput (verify all gamma positions)
    #   (b) dynamic scheduler throughput (Algorithm 1 picks ell*)
    #   (c) "best static gamma" — the global gamma that maximizes theta
    #       over the range [0, 16].
    for prof_name, (c0, decay) in PROFILES.items():
        prof_results = {}
        for gamma in (4, 8, 12, 16, 24):
            confs = _decaying_confidences(gamma, c0, decay)
            res = schedule([confs], sps_callable)
            ell_dyn = res.lengths[0]
            theta_dyn = res.theta
            theta_static = _theta_at(confs, gamma, sps_callable)
            # delta vs static gamma (DFlash today)
            delta_pct = (
                100.0 * (theta_dyn - theta_static) / theta_static
                if theta_static > 0
                else float("nan")
            )
            prof_results[f"gamma={gamma}"] = {
                "static_ell": gamma,
                "static_theta_tok_per_s": theta_static,
                "dynamic_ell": ell_dyn,
                "dynamic_theta_tok_per_s": theta_dyn,
                "delta_pct": delta_pct,
            }

        # Best static gamma — equivalent to a one-line config tune.
        # Search [0, 16] for the gamma that maximizes theta if we always
        # verify the full block.
        best_static_ell = 0
        best_static_theta = sps_callable(1) * 1.0
        for ell_try in range(0, 17):
            confs_try = _decaying_confidences(max(ell_try, 1), c0, decay)
            theta_try = _theta_at(confs_try, ell_try, sps_callable)
            if theta_try > best_static_theta:
                best_static_theta = theta_try
                best_static_ell = ell_try
        prof_results["best_static_gamma_tune"] = {
            "ell": best_static_ell,
            "theta_tok_per_s": best_static_theta,
        }

        out["comparisons"][prof_name] = prof_results

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("results written to", args.out)

    # Pretty headline
    print("\n=== R=1 throughput on measured M3 Ultra SPS ===")
    print(f"{'profile':>8} {'gamma':>6} {'static':>10} {'dynamic':>10} {'delta':>8}")
    for prof_name, prof_results in out["comparisons"].items():
        for gamma_key, vals in prof_results.items():
            if not gamma_key.startswith("gamma="):
                continue
            print(
                f"{prof_name:>8} {gamma_key[6:]:>6} "
                f"{vals['static_theta_tok_per_s']:>10.2f} "
                f"{vals['dynamic_theta_tok_per_s']:>10.2f} "
                f"{vals['delta_pct']:>+7.1f}%"
            )

    print("\n=== Best STATIC gamma per profile (one-line config tune) ===")
    for prof_name, prof_results in out["comparisons"].items():
        b = prof_results["best_static_gamma_tune"]
        print(
            f"  {prof_name}: gamma*={b['ell']:>2d}, theta={b['theta_tok_per_s']:.2f} tok/s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
