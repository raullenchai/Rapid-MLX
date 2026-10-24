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
from scripts.deepseek_v41_rd_allocate import inventory_digest, publish_allocations
from scripts.deepseek_v41_rd_layer_allocate import (
    checked_byte_costs,
    routed_and_router_bytes,
    router_row_bytes,
)

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def price_layers(layers, costs):
    pricing = {}
    for layer, spec in layers.items():
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
        size = routed_and_router_bytes(
            costs[int(layer)], kept, [widths[p] for p in PROJECTIONS]
        )
        pricing[layer] = (widths, size)
    return pricing


def variant_allocation(base, cost, overrides):
    if base.get("provenance", {}).get("costs_sha256") != inventory_digest(cost):
        raise ValueError(
            "cost inventory does not match base provenance; regenerate base allocation"
        )
    result = copy.deepcopy(base)
    costs = np.asarray(cost["expert_bytes"])
    if (
        costs.ndim != 3
        or costs.shape[-1] != 4
        or set(base["layers"]) != {str(i) for i in range(len(costs))}
        or not np.isfinite(costs).all()
        or np.any(costs != np.floor(costs))
        or np.any(costs[..., 0] != 0)
        or np.any(costs[..., 1:] <= costs[..., :-1])
        or base.get("reference")
    ):
        raise ValueError("invalid layer cost inventory or allocation")
    costs = checked_byte_costs(costs)
    base_pricing = price_layers(base["layers"], costs)
    if any(
        type(base.get(key)) is not int
        for key in ("fixed_non_expert_bytes", "resident_tensor_bytes")
    ) or (base.get("keep_mtp") and type(base.get("mtp_bytes")) is not int):
        raise ValueError(
            "base allocation requires integral fixed/resident/draft byte totals"
        )
    fixed = cost["fixed_non_expert_bytes"] + (
        base["mtp_bytes"] if base.get("keep_mtp") else 0
    )
    if fixed != base["fixed_non_expert_bytes"]:
        raise ValueError("base fixed bytes disagree with cost inventory/draft policy")
    base_routed = sum(size for _, size in base_pricing.values())
    if base["resident_tensor_bytes"] != fixed + base_routed:
        raise ValueError("base resident bytes disagree with layer specifications")
    for layer, (_, size) in base_pricing.items():
        if base["layers"][layer].get("routed_and_router_bytes", size) != size:
            raise ValueError("base layer bytes disagree with layer specifications")
    if "byte_breakdown" in base:
        breakdown = base["byte_breakdown"]
        components = breakdown["components"]
        router = sum(
            len(spec["kept_expert_ids"]) * router_row_bytes(costs[int(layer)])
            for layer, spec in base["layers"].items()
        )
        if (
            any(type(value) is not int or value < 0 for value in components.values())
            or components.get("router", 0) != router
            or components.get("routed_experts", 0) != base_routed - router
            or sum(
                value
                for name, value in components.items()
                if name not in ("router", "routed_experts", "engram_ssd")
            )
            != fixed
            or breakdown["resident_tensor_bytes"] != fixed + base_routed
            or breakdown["total_payload_bytes"] != sum(components.values())
            or breakdown.get("shared_engram_bytes", components.get("engram_ssd", 0))
            != components.get("engram_ssd", 0)
        ):
            raise ValueError(
                "base byte breakdown disagrees with allocation and inventory"
            )
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
    pricing = price_layers(result["layers"], costs)
    for layer, (widths, size) in pricing.items():
        result["layers"][layer]["projection_bits"] = widths
        result["layers"][layer]["routed_and_router_bytes"] = size
    used = sum(size for _, size in pricing.values())
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
    result["variant"]["costs_sha256"] = result["provenance"]["costs_sha256"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    publish_allocations([(a.output, result)])
    print(f"Resident GB delta: {result['variant']['resident_byte_delta'] / 1e9:+.9f}")


if __name__ == "__main__":
    main()
