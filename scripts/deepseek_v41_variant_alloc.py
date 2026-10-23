#!/usr/bin/env python3
"""Override layer projection widths for a sensitivity sweep (no budget constraint)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_v41_rd_layer_allocate import routed_and_router_bytes

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def variant_allocation(base, cost, overrides):
    result = copy.deepcopy(base)
    costs = np.asarray(cost["expert_bytes"])
    if (
        costs.ndim != 3
        or costs.shape[-1] != 4
        or set(base["layers"]) != {str(i) for i in range(len(costs))}
        or not np.isfinite(costs).all()
        or np.any(costs != np.floor(costs))
        or np.any(costs[..., 0] != 0)
        or np.any(np.diff(costs, axis=-1) <= 0)
        or base.get("reference")
    ):
        raise ValueError("invalid layer cost inventory or allocation")
    for override in overrides:
        try:
            layers, widths = override.split("=")
            bits = tuple(int(b) for b in widths.split("/"))
            ids = [int(i) for i in layers.split(",")]
        except ValueError as exc:
            raise ValueError("expected LAYERS=g/u/d") from exc
        if len(bits) != 3 or any(b not in (2, 3, 4) for b in bits) or not ids:
            raise ValueError("expected three widths from 2/3/4")
        for layer in ids:
            if str(layer) not in result["layers"]:
                raise ValueError(f"unknown layer: {layer}")
            spec = result["layers"][str(layer)]
            spec["projection_bits"] = dict(zip(PROJECTIONS, bits))
            spec.pop("bits", None)
    used = 0
    for layer, spec in result["layers"].items():
        kept = spec["kept_expert_ids"]
        if (
            not kept
            or kept != sorted(set(kept))
            or any(type(i) is not int or not 0 <= i < costs.shape[1] for i in kept)
        ):
            raise ValueError("invalid kept expert IDs")
        widths = spec.get(
            "projection_bits", dict.fromkeys(PROJECTIONS, spec.get("bits"))
        )
        if set(widths) != set(PROJECTIONS) or any(
            type(b) is not int or b not in (2, 3, 4) for b in widths.values()
        ):
            raise ValueError("invalid projection bits")
        spec["projection_bits"] = widths
        size = routed_and_router_bytes(
            costs[int(layer)], kept, [widths[p] for p in PROJECTIONS]
        )
        spec["routed_and_router_bytes"] = size
        used += size
    # The allocation's fixed total includes its selected draft precision. Older
    # inventories price only the former draft policy, so retain the base's total.
    fixed = cost["fixed_non_expert_bytes"] + (
        base["mtp_bytes"] if base.get("keep_mtp") else 0
    )
    if fixed != base["fixed_non_expert_bytes"]:
        raise ValueError("base fixed bytes disagree with cost inventory/draft policy")
    resident = int(fixed + used)
    delta = resident - base["resident_tensor_bytes"]
    result["resident_tensor_bytes"] = resident
    result["budget_bytes"] = None
    result["granularity"] = "projection"
    result["variant"] = {
        "base_budget_bytes": base.get("budget_bytes"),
        "base_allocation_sha256": hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "overrides": list(overrides),
        "resident_byte_delta": delta,
    }
    # These optimality estimates describe the base solve, not the forced variant.
    for field in ("predicted_distortion", "dual_lower_bound", "duality_gap"):
        result.pop(field, None)
    if "byte_breakdown" in result:
        breakdown = result["byte_breakdown"]
        breakdown["components"]["routed_experts"] += delta
        breakdown["resident_tensor_bytes"] = resident
        breakdown["total_payload_bytes"] += delta
        # Headers require a source plan; let the builder recompute shard layout.
        for field in (
            "safetensors_container_bytes",
            "total_safetensors_file_bytes",
            "largest_resident_shard_payload",
        ):
            breakdown.pop(field, None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--costs", type=Path, required=True)
    p.add_argument("--set-bits", action="append", required=True, metavar="LAYERS=g/u/d")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = variant_allocation(
        json.loads(a.base.read_text()), json.loads(a.costs.read_text()), a.set_bits
    )
    result["variant"]["costs_sha256"] = hashlib.sha256(a.costs.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Resident GB delta: {result['variant']['resident_byte_delta'] / 1e9:+.9f}")


if __name__ == "__main__":
    main()
