#!/usr/bin/env python3
"""Build the offline decode snapshot from an explicit public leaderboard export.

Use --input with a JSON array of runs. Network acquisition is deliberately a
release/operator step, never an import, model-list or Desktop runtime action.
Only allowlisted aggregate fields leave this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_snapshot(runs: list[dict], refreshed_at: str) -> dict:
    aliases = json.loads((ROOT / "rapid_mlx/aliases.json").read_text())
    groups: dict[tuple, dict] = defaultdict(dict)
    for run in runs:
        if run.get("task_type") != "text_generation" or run.get("protocol") != {
            "id": "rapid-community-speed",
            "version": 2,
        }:
            continue
        machine = run.get("machine", {})
        chip = machine.get("chip", "").removeprefix("Apple ")
        ram = machine.get("memory_gib")
        if (
            not re.fullmatch(r"M[1-9][0-9]*(?: Pro| Max| Ultra)?", chip)
            or not isinstance(ram, (int, float))
            or not math.isfinite(ram)
            or ram <= 0
        ):
            continue
        method = run.get("execution", {}).get("speculative_decoding", {}).get("method")
        if method not in ("none", "mtp"):
            continue
        values = [
            case["decode_tps"]
            for case in run.get("cases", [])
            if case.get("case_id") in ("pp512-tg128", "pp2048-tg512")
            and isinstance(case.get("decode_tps"), (float, int))
            and math.isfinite(case["decode_tps"])
            and case["decode_tps"] > 0
        ]
        identifier = run.get("submission_id")
        if not values or not identifier:
            continue
        key = (run.get("model", {}).get("repo_id"), chip, method)
        groups[key][identifier] = (statistics.median(values), ram)
    policy = json.loads((ROOT / "rapid_mlx/model_recommendations.json").read_text())
    working_ram = {
        pick["alias"]: pick["footprint_mib"] / 1024 / 0.8
        for tier in policy["tiers"]
        for pick in tier["picks"]
    }
    models = {}
    for alias, profile in sorted(aliases.items()):
        if profile.get("modality", "text") not in ("text", "vision"):
            continue
        repo = profile["hf_path"]
        cohorts = []
        for (model, chip, method), observations in sorted(groups.items()):
            if model != repo:
                continue
            values = sorted(v[0] for v in observations.values())
            q = (
                statistics.quantiles(values, n=4, method="inclusive")
                if len(values) > 1
                else values * 3
            )
            cohorts.append(
                {
                    "chip": chip,
                    "method": method,
                    "n": len(values),
                    "ram_min_gib": min(v[1] for v in observations.values()),
                    "ram_max_gib": max(v[1] for v in observations.values()),
                    "lower": round(q[0], 2),
                    "median": round(statistics.median(values), 2),
                    "upper": round(q[2], 2),
                }
            )
        # A declared draft is not an automatic preset until qualified. Other
        # backend-owned presets need their own compatible execution cohort.
        default_method = "none"
        if (
            profile.get("tensorfold_mtp")
            or profile.get("tensorfold_kernel")
            or profile.get("dflash_backend") == "tensorfold"
        ):
            default_method = "unknown"
        elif profile.get("mtp_default_enabled", True) and (
            profile.get("supports_native_mtp")
            or (
                profile.get("mtp_draft_model")
                and profile.get("mtp_continuous_batching_tier") == "verified"
            )
        ):
            default_method = "mtp"
        models[alias] = {
            "repo_id": repo,
            "default_method": default_method,
            "min_ram_gib": max(
                profile.get("min_memory_gb", 0), working_ram.get(alias, 0)
            ),
            "cohorts": cohorts,
        }
    return {
        "schema_version": 1,
        "refreshed_at": refreshed_at,
        "protocol": "rapid-community-speed v2; median of short/long cases per distinct submission; interquartile range",
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--refreshed-at",
        required=True,
        help="UTC date of the complete export (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "rapid_mlx/decode_expectations.json"
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            build_snapshot(json.loads(args.input.read_text()), args.refreshed_at),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
