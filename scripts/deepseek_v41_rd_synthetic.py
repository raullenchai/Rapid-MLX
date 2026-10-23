#!/usr/bin/env python3
"""Reproducible assumed capacity examples, NOT calibrated release allocations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_contrastive import contrastive
from scripts.deepseek_v41_rd_allocate import publish_allocations
from scripts.deepseek_v41_rd_build import layout_report, mtp_byte_costs, plan
from scripts.deepseek_v41_rd_layer_allocate import PRESETS, layer_allocate
from scripts.repack_deepseek_v41_native import CheckpointIndex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--costs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    config = json.loads((a.source / "config.json").read_text())
    cost = json.loads(a.costs.read_text())
    costs = np.asarray(cost["expert_bytes"])
    layers, experts = costs.shape[:2]
    # Deliberately known specialist population: 24 cold specialists, 176 weak
    # hot experts, 184 strongly protected hot experts in each 384-expert layer.
    if experts != 384:
        raise ValueError("this illustrative fixture assumes 384 experts")
    r = np.ones((4, 2, layers, experts))
    r[:2, :, :, :200] = 0.001
    r[2, :, :, :24] = 8.0
    r[3, :, :, :24] = 0.1
    # Retain cross-layer sensitivity magnitudes when choosing precision.
    r *= np.linspace(0.5, 1.5, layers)[None, None, :, None]
    counts = np.full(r.shape, 8)
    roles = {
        "chat": "hot",
        "code": "hot",
        "cold_en": "cold",
        "general_reference_en": "reference",
    }
    selection = contrastive(r, counts, list(roles), roles)
    q = np.broadcast_to([1.0, 0.35, 0.12, 0.04], costs.shape)
    pq = np.broadcast_to(
        [[0.07, 0.025, 0.008], [0.07, 0.025, 0.008], [0.2, 0.035, 0.008]],
        (*costs.shape[:2], 3, 3),
    )
    index = CheckpointIndex(a.source)
    draft_costs = mtp_byte_costs(index, config)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "reference": True,
        "layers": {
            str(i): {"kept_expert_ids": list(range(experts)), "bits": 16}
            for i in range(layers)
        },
    }
    entries, _, report = plan(index, config, reference)
    reference["byte_breakdown"] = layout_report(entries, report)
    outputs = [(a.output_dir / "reference-plan.json", reference)]
    for target, preset in PRESETS.items():
        fixed = cost["fixed_non_expert_bytes"] + (
            draft_costs["4"] if target == "T256" else 0
        )
        result = layer_allocate(
            selection,
            q,
            costs,
            fixed,
            preset["budget_bytes"],
            preset["max_prune_fraction"],
            preset["fallback"],
            6,
            pq,
            allow_hot_pruning=preset["allow_hot_pruning"],
        )
        result["keep_mtp"] = target == "T256"
        result["mtp_expert_bits"] = 4
        result["mtp_bytes_by_expert_bits"] = draft_costs
        result["synthetic_only"] = True
        result["assumptions"] = (
            "24 stable cold specialists, 176 other weak hot experts, 184 protected per layer; all routed. Isolated q gate/up=[.07,.025,.008], down=[.2,.035,.008]. Not measured."
        )
        entries, _, report = plan(index, config, result)
        result["byte_breakdown"] = layout_report(entries, report)
        if report["resident_tensor_bytes"] != result["resident_tensor_bytes"]:
            raise ValueError("synthetic allocation and output bytes disagree")
        outputs.append((a.output_dir / f"{target}-synthetic.json", result))
        print(
            json.dumps(
                {
                    "target": target,
                    "contrastive_by_layer": selection["candidate_count_by_layer"],
                    "kept_experts": sum(
                        len(s["kept_expert_ids"]) for s in result["layers"].values()
                    ),
                    **result["byte_breakdown"],
                },
                indent=2,
            )
        )

    publish_allocations(outputs)


if __name__ == "__main__":
    main()
