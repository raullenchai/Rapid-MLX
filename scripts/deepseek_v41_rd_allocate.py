#!/usr/bin/env python3
"""Lagrangian expert rate-distortion allocation; budgets use decimal GB."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_contrastive import contrastive
from scripts.deepseek_v41_rd_common import BITS, bank_plan
from scripts.deepseek_v41_rd_layer_allocate import PRESETS, layer_allocate


def allocate(
    sensitivity, errors, costs, fixed_bytes, budget_bytes, min_kept=288, topk=6
):
    s, q, c = np.asarray(sensitivity), np.asarray(errors), np.asarray(costs)
    if s.ndim != 2 or q.shape != (*s.shape, 4) or c.shape != q.shape:
        raise ValueError(
            "expected sensitivity [L,E], errors/costs [L,E,4] for 0,2,3,4 bits"
        )
    if (
        not np.isfinite(s).all()
        or not np.isfinite(q).all()
        or not np.isfinite(c).all()
        or np.any(s < 0)
        or np.any(q < 0)
        or np.any(c < 0)
        or not np.all(q[..., 0] == 1)
        or np.any(c[..., 0] != 0)
        or np.any(np.diff(c, axis=-1) <= 0)
        or np.any(c != np.floor(c))
    ):
        raise ValueError(
            "invalid sensitivity, errors, or strictly increasing integer byte costs"
        )
    if (
        not 1 <= topk <= min_kept <= s.shape[1]
        or not np.isfinite(fixed_bytes)
        or not np.isfinite(budget_bytes)
        or fixed_bytes < 0
        or budget_bytes < 0
        or fixed_bytes != int(fixed_bytes)
        or budget_bytes != int(budget_bytes)
    ):
        raise ValueError("invalid layer guard or byte budget")
    c = c.astype(np.int64)
    d = s[..., None] * q
    remaining = int(budget_bytes) - int(fixed_bytes)
    minimum = np.sort(c[..., 1], axis=1)[:, :min_kept].sum()
    if remaining < minimum:
        raise ValueError(
            f"infeasible budget: fixed + minimum experts = {fixed_bytes + minimum} bytes"
        )

    def solve(lam):
        objective = d + lam * c
        positive = np.argmin(objective[..., 1:], axis=-1) + 1
        delta = (
            np.take_along_axis(objective, positive[..., None], -1)[..., 0]
            - objective[..., 0]
        )
        keep = delta < 0
        for layer in range(len(s)):
            keep[layer, np.argsort(delta[layer], kind="stable")[:min_kept]] = True
        state = np.where(keep, positive, 0)
        size = int(np.take_along_axis(c, state[..., None], -1).sum())
        distortion = float(np.take_along_axis(d, state[..., None], -1).sum())
        return state, size, distortion

    lo, hi = 0.0, 1.0
    while solve(hi)[1] > remaining:
        hi *= 2
    best, used, loss = solve(hi)
    lower = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        state, size, distortion = solve(mid)
        lower = max(lower, distortion + mid * (size - remaining))
        if size > remaining:
            lo = mid
        else:
            hi = mid
            best, used, loss = state, size, distortion
    # Repair discrete/tied steps using marginal distortion reduction per byte.
    # This is a feasible integer heuristic; report its dual gap, never claim
    # arbitrary multiple-choice knapsack optimality from greedy ratios alone.
    heap = []

    def edges(layer_id, e):
        old = int(best[layer_id, e])
        for new in range(old + 1, 4):
            benefit = d[layer_id, e, old] - d[layer_id, e, new]
            size = int(c[layer_id, e, new] - c[layer_id, e, old])
            if benefit > 0:
                heapq.heappush(
                    heap, (-benefit / size, layer_id, e, old, new, size, benefit)
                )

    for layer_id, e in np.ndindex(s.shape):
        edges(layer_id, e)
    while heap:
        _, layer_id, e, old, new, size, benefit = heapq.heappop(heap)
        if best[layer_id, e] != old or used + size > remaining:
            continue
        best[layer_id, e] = new
        used += size
        loss -= benefit
        edges(layer_id, e)
    return {
        "schema_version": 1,
        "bits_order": BITS.tolist(),
        "budget_bytes": int(budget_bytes),
        "fixed_non_expert_bytes": int(fixed_bytes),
        "resident_tensor_bytes": int(fixed_bytes + used),
        "predicted_distortion": max(0.0, float(loss)),
        "dual_lower_bound": lower,
        "duality_gap": max(0.0, float(loss - lower)),
        "lambda": hi,
        "min_kept": min_kept,
        "topk": topk,
        "shared_experts": "unchanged",
        "layers": {
            str(layer_id): {
                **bank_plan(BITS[best[layer_id]], topk),
                "bit_widths": BITS[best[layer_id]].tolist(),
            }
            for layer_id in range(len(s))
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--errors", type=Path, required=True)
    p.add_argument(
        "--costs",
        type=Path,
        required=True,
        help="JSON with expert_bytes [L,E,4], fixed_non_expert_bytes",
    )
    p.add_argument("--preset", choices=PRESETS)
    p.add_argument(
        "--granularity",
        choices=("layer", "projection", "legacy-expert"),
        default="layer",
    )
    p.add_argument(
        "--slice-roles",
        type=Path,
        help="JSON mapping calibrated slice names to hot/cold/reference",
    )
    p.add_argument("--hot-threshold", type=float, default=0.25)
    p.add_argument("--cold-ratio", type=float, default=4.0)
    p.add_argument("--max-prune-fraction", type=float)
    p.add_argument("--rd-fallback", action="store_true")
    p.add_argument(
        "--source", type=Path, help="source headers for exact component breakdown"
    )
    p.add_argument("--score", choices=["reap", "man", "msan"], default="reap")
    p.add_argument(
        "--budgets-gb", type=float, nargs="+", default=[260, 240, 220, 200, 185]
    )
    p.add_argument("--min-kept", type=int, default=288)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--min-split-overlap", type=float, default=0.9)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if not 0 <= a.min_split_overlap <= 1:
        raise ValueError("split-overlap threshold must be between zero and one")
    with (
        np.load(a.scores, allow_pickle=False) as scores,
        np.load(a.errors, allow_pickle=False) as errors,
    ):
        overlaps = scores["split_keep_overlap"]
        if not np.isfinite(overlaps).all() or np.min(overlaps) < a.min_split_overlap:
            raise ValueError(
                "unstable calibration keep sets; increase calibration tokens"
            )
        if not np.all(errors["completed"]):
            raise ValueError("quantization-error artifact is incomplete")
        sm, qm = (
            json.loads(str(scores["metadata"])),
            json.loads(str(errors["metadata"])),
        )
        if sm["model_revision"] != qm["model_revision"]:
            raise ValueError("model revisions do not match")
        if sm["routing_topk"] != a.topk:
            raise ValueError("top-k must match the calibrated model")
        if qm["group_size"] != 64 or qm["bits_order"] != BITS.tolist():
            raise ValueError(
                "quantization error format must be affine group-64, bits 0/2/3/4"
            )
        cost = json.loads(a.costs.read_text())
        if cost["model_revision"] != sm["model_revision"]:
            raise ValueError("cost inventory model revision does not match")
        if cost["group_size"] != 64:
            raise ValueError("cost inventory must use group-64")
        weight_costs = errors["expert_weight_bytes"]
        expert_costs = np.asarray(cost["expert_bytes"])
        if expert_costs.shape != weight_costs.shape:
            raise ValueError("cost and error expert shapes disagree")
        for layer_id in range(len(expert_costs)):
            row_bytes = cost["router_row_bytes_by_layer"][str(layer_id)]
            expected = weight_costs[layer_id].copy()
            expected[:, 1:] += row_bytes
            if not np.array_equal(expert_costs[layer_id], expected):
                raise ValueError("inventory costs disagree with measured MLX payloads")
        a.output_dir.mkdir(parents=True, exist_ok=True)
        selection = None
        if a.granularity != "legacy-expert":
            if a.score != "reap" or a.slice_roles is None or a.source is None:
                raise ValueError(
                    "contrastive layer allocation needs --score reap, --slice-roles and --source"
                )
            roles = json.loads(a.slice_roles.read_text())
            selection = contrastive(
                scores["reap_halves"],
                scores["counts_halves_by_slice"],
                sm["slice_names"],
                roles,
                a.hot_threshold,
                a.cold_ratio,
            )
        budgets = (
            [PRESETS[a.preset]["budget_bytes"] / 1e9] if a.preset else a.budgets_gb
        )
        for gb in budgets:
            if selection is None:
                if a.preset:
                    raise ValueError(
                        "release presets require contrastive layer/projection granularity"
                    )
                result = allocate(
                    scores["robust_" + a.score],
                    errors["q"],
                    cost["expert_bytes"],
                    cost["fixed_non_expert_bytes"],
                    round(gb * 1e9),
                    a.min_kept,
                    a.topk,
                )
            else:
                preset = PRESETS.get(a.preset, PRESETS["T256"])
                mtp_bytes = cost.get("mtp_bytes", 0) if a.preset == "T256" else 0
                result = layer_allocate(
                    selection,
                    errors["q"],
                    cost["expert_bytes"],
                    cost["fixed_non_expert_bytes"] + mtp_bytes,
                    round(gb * 1e9),
                    preset["max_prune_fraction"]
                    if a.max_prune_fraction is None
                    else a.max_prune_fraction,
                    preset["fallback"] or a.rd_fallback,
                    a.topk,
                    errors["projection_q"] if a.granularity == "projection" else None,
                )
                result["keep_mtp"] = a.preset == "T256"
                from scripts.deepseek_v41_rd_build import layout_report, plan
                from scripts.repack_deepseek_v41_native import CheckpointIndex

                config = json.loads((a.source / "config.json").read_text())
                entries, _, report = plan(CheckpointIndex(a.source), config, result)
                result["byte_breakdown"] = layout_report(entries, report)
                if report["resident_tensor_bytes"] != result["resident_tensor_bytes"]:
                    raise ValueError(
                        "allocation cost inventory disagrees with release conversion policy; regenerate --release-policy inventory"
                    )
                result["contrastive_thresholds"] = {
                    "hot_threshold": a.hot_threshold,
                    "cold_ratio": a.cold_ratio,
                    "min_cold_routes_per_half": 2,
                    "ratio_tolerance": 0.5,
                    "slice_roles": roles,
                }
            result["provenance"] = {"scores": sm, "errors": qm, "score": a.score}
            (a.output_dir / f"allocation-{gb:g}GB.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
