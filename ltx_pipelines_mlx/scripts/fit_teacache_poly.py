"""Robust polyfit for TeaCache calibration data.

Tries polyfit degrees 1..max_degree on the saved (delta_in, delta_res_video)
pairs, and picks the **lowest stable** degree. A fit is "stable" when the
polynomial is monotone-non-decreasing AND non-negative across the observed
delta range. Among stable fits, the lowest degree wins (Occam's razor).

If no degree produces a stable fit, falls back to the lowest-RMSE candidate
and flags ``diagnostics["fallback"] = True`` so the caller knows the
calibration data is too noisy / too sparse to trust.

Usage:
    python -m ltx_pipelines_mlx.scripts.fit_teacache_poly \\
        regression_tests/ltx2_calibration.json \\
        --max-degree 4 \\
        --out fit_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FitResult:
    coefficients: list[float]
    degree: int
    diagnostics: dict


def _evaluate_candidate(
    xs: Sequence[float],
    ys: Sequence[float],
    deg: int,
    grid: np.ndarray,
) -> dict:
    """Fit one polynomial degree and characterize its stability."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coeffs = np.polyfit(xs, ys, deg=deg)
    poly = np.poly1d(coeffs)

    grid_values = poly(grid)
    sample_predictions = poly(np.asarray(xs))
    rmse = float(np.sqrt(np.mean((np.asarray(ys) - sample_predictions) ** 2)))

    diffs = np.diff(grid_values)
    is_monotone = bool(np.all(diffs >= -1e-9))
    is_positive = bool(np.all(grid_values >= -1e-9))

    return {
        "degree": deg,
        "coefficients": coeffs.tolist(),
        "monotone": is_monotone,
        "positive": is_positive,
        "rmse": rmse,
    }


def fit_robust(
    delta_in: Sequence[float],
    delta_out: Sequence[float],
    max_degree: int = 4,
) -> FitResult:
    """Pick the lowest polyfit degree that's monotone-positive on observed range.

    Args:
        delta_in: List of input L1 deltas (from calibration).
        delta_out: Corresponding output L1 deltas.
        max_degree: Highest degree to try (clamped to ``len(delta_in) - 1``).

    Returns:
        ``FitResult`` with the chosen coefficients (highest-degree-first per
        ``numpy.poly1d`` convention), the chosen degree, and a diagnostics
        dict containing per-degree evaluations.

    Raises:
        ValueError: If ``delta_in`` is empty.
    """
    if not delta_in:
        raise ValueError("empty delta_in — no calibration data to fit")
    if len(delta_in) != len(delta_out):
        raise ValueError(f"delta_in and delta_out length mismatch: {len(delta_in)} vs {len(delta_out)}")

    effective_max = min(max_degree, max(1, len(delta_in) - 1))
    grid = np.linspace(min(delta_in), max(delta_in), 100)

    candidates = [_evaluate_candidate(delta_in, delta_out, deg, grid) for deg in range(1, effective_max + 1)]

    stable = [c for c in candidates if c["monotone"] and c["positive"]]
    fallback = not stable
    chosen = stable[0] if stable else min(candidates, key=lambda c: c["rmse"])

    return FitResult(
        coefficients=chosen["coefficients"],
        degree=chosen["degree"],
        diagnostics={
            "candidates": candidates,
            "chosen": chosen,
            "delta_range": [float(min(delta_in)), float(max(delta_in))],
            "num_points": len(delta_in),
            "fallback": fallback,
        },
    )


def _format_paste_snippet(coeffs: list[float], thresh: float) -> str:
    lines = ["LTX2_TEACACHE_COEFFICIENTS = ["]
    for c in coeffs:
        lines.append(f"    {c!r},")
    lines.append("]")
    lines.append(f"LTX2_TEACACHE_THRESH = {thresh}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_json",
        type=Path,
        help="Calibration JSON (must contain a `deltas` key with delta_in / delta_res_video)",
    )
    parser.add_argument("--max-degree", type=int, default=4, help="Highest polyfit degree to try")
    parser.add_argument(
        "--rel-l1-thresh",
        type=float,
        default=None,
        help="Override default rel_l1_thresh in the paste snippet (defaults to value in JSON)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the fit result JSON")
    args = parser.parse_args()

    if not args.input_json.exists():
        print(f"input not found: {args.input_json}", file=sys.stderr)
        return 2

    payload = json.loads(args.input_json.read_text())
    deltas = payload.get("deltas")
    if not deltas or "delta_in" not in deltas or "delta_res_video" not in deltas:
        print(
            f"input JSON {args.input_json} missing `deltas` block (likely an older "
            "calibration run). Re-run scripts/calibrate_teacache.py with the patched "
            "version to capture raw deltas.",
            file=sys.stderr,
        )
        return 2

    result = fit_robust(
        delta_in=deltas["delta_in"],
        delta_out=deltas["delta_res_video"],
        max_degree=args.max_degree,
    )

    print(f"\nFit result for {args.input_json.name} ({result.diagnostics['num_points']} points):")
    print(f"  delta range: [{result.diagnostics['delta_range'][0]:.4f}, {result.diagnostics['delta_range'][1]:.4f}]\n")
    print("  Per-degree evaluation:")
    for c in result.diagnostics["candidates"]:
        marker = " ← chosen" if c["degree"] == result.degree else ""
        print(
            f"    deg {c['degree']}: monotone={c['monotone']!s:5}, "
            f"positive={c['positive']!s:5}, rmse={c['rmse']:.5f}{marker}"
        )

    if result.diagnostics["fallback"]:
        print(
            "\n  ⚠  No stable fit on the observed delta range. Falling back to lowest "
            "RMSE — coefficients may produce erratic skip decisions. Consider:"
        )
        print("     - more calibration prompts (5-10 instead of 3)")
        print("     - rerun on a fresh host (perf degradation skews late-step deltas)")
        print("     - reviewing input data for outliers")

    thresh = args.rel_l1_thresh if args.rel_l1_thresh is not None else payload.get("rel_l1_thresh", 0.15)

    print("\nPaste into ltx_pipelines_mlx/ti2vid_two_stages.py:\n")
    print(_format_paste_snippet(result.coefficients, thresh))

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "coefficients": result.coefficients,
                    "degree": result.degree,
                    "rel_l1_thresh": thresh,
                    "diagnostics": result.diagnostics,
                },
                indent=2,
            )
        )
        print(f"\n  Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
