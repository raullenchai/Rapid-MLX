#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Distill local paired comparison evidence into reviewable profile entries."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rapid_mlx import __version__
from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator
from rapid_mlx.community_bench.comparison import arm_statistics, speed_winners
from rapid_mlx.tuning.profiles import runtime_context


def cell_key(run: dict) -> str:
    ctx = run["tuning"]["context"]
    primary = [c for c in run["model"]["components"] if c["role"] == "primary"]
    language = {
        key: value
        for key, value in run["execution"]["task"]["language"].items()
        if key != "speculative_decoding"
    }
    return json.dumps(
        [
            ctx["hardware"],
            ctx["model"],
            ctx["mlx"],
            ctx["macos"],
            ctx["profiles"],
            run["execution"]["runtime"],
            primary,
            run["workload"],
            ctx["mode"],
            ctx["width"],
            ctx["batch"],
            run["execution"]["resources"],
            language,
        ],
        sort_keys=True,
    )


def effective_configuration(run: dict) -> dict:
    ctx = run["tuning"]["context"]
    knobs = {
        name: decision["value"]
        for name, decision in ctx["knobs"].items()
        if decision["source"] != "capability"
    }
    knobs.update(run["tuning"]["overrides"])
    return {"mode": ctx["mode"], "knobs": knobs}


def quality_passes(runs: list[dict], defaults: list[dict]) -> bool:
    outputs = [r["tuning"]["outputs"] for r in defaults + runs]
    complete = all(
        d for output in outputs for hashes in output.values() for d in hashes
    )
    canonical = {json.dumps(output, sort_keys=True) for output in outputs}
    # Equality must include repeatability inside each case as well as across loads.
    stable = all(
        len(set(hashes)) == 1 for output in outputs for hashes in output.values()
    )
    if complete and stable and len(canonical) == 1:
        return True
    for run in runs:
        quality = run["tuning"].get("quality", {})
        if quality.get("n", 0) < 100 or "questions_digest" not in quality:
            return False
        for baseline in ("plain", "default"):
            pair = quality.get(baseline, {})
            if abs(pair.get("delta_pp", 100)) > 3 or pair.get("paired_p", 0) < 0.05:
                return False
    return True


def distill(
    runs: list[dict],
    *,
    as_of: datetime,
    max_age_days: int = 30,
    rapid_version: str = __version__,
) -> dict:
    as_of = as_of.replace(tzinfo=timezone.utc) if as_of.tzinfo is None else as_of
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejected = []
    validator = BenchmarkRunValidator()
    for run in runs:
        validator.validate(run)
        if (
            "tuning" not in run
            or "experiment" not in run
            or run["outcome"]["status"] != "completed"
            or run["workload"]["task_type"] != "text_generation"
        ):
            rejected.append(
                {"run_id": run["run_id"], "reason": "paired tuning evidence missing"}
            )
            continue
        age = (
            as_of - datetime.fromisoformat(run["completed_at"].replace("Z", "+00:00"))
        ).total_seconds() / 86400
        ctx = run["tuning"]["context"]
        runtime = run["execution"]["runtime"]
        conditions = [
            run["machine"][key] for key in ("conditions_before", "conditions_after")
        ]
        hw = ctx["hardware"]
        observed = run["machine"]["profile"]
        expected = runtime_context(
            {"device_name": observed["chip"]},
            library_version=ctx["mlx"],
            macos=ctx["macos"],
        ).hardware
        model = ctx["model"]
        eligible = (
            0 <= age <= max_age_days
            and runtime["rapid_mlx"] == rapid_version
            and runtime["rapid_mlx"] == ctx["rapid_mlx"]
            and runtime["mlx"] == ctx["mlx"]
            and ctx["macos"] == run["machine"]["os"]["version"]
            and ctx["mlx"] != "unknown"
            and ctx["macos"] != "unknown"
            and all(
                hw.get(k) is not None
                for k in ("gpu_generation", "gpu_cores", "memory_size")
            )
            and hw["gpu_generation"] == expected.gpu_generation
            and hw["tier"] == expected.tier
            and hw["gpu_cores"] == observed["gpu_cores"]
            and hw["memory_size"] == observed["memory_gib"] * (1 << 30)
            and model["family"] != "unknown"
            and model["quant_mode"] != "unknown"
            and all(
                model[key] is not None and model[key] > 0
                for key in (
                    "hidden_size",
                    "intermediate_size",
                    "num_experts",
                    "top_k",
                    "bits",
                    "group_size",
                )
            )
            and hw["tier"] != "unknown"
            and ctx["width"] == ctx["batch"] == 1
            and all(
                m["decode_duration_ms"] > 0 and m["output_tokens"] > 1
                for m in run["measurements"]
            )
            and all(
                c["power_source"] == "ac"
                and c["thermal_state"] == "nominal"
                and c["memory_pressure"] == "normal"
                and c["low_power_mode"] is False
                for c in conditions
            )
        )
        if not eligible:
            rejected.append(
                {
                    "run_id": run["run_id"],
                    "reason": "freshness, facts, or operating conditions gate",
                }
            )
            continue
        groups[(cell_key(run), run["experiment"]["group_id"])].append(run)
    candidates: dict[str, list[dict]] = defaultdict(list)
    for (cell, session), group in groups.items():
        defaults = [r for r in group if r["experiment"]["arm_id"] == "default"]
        if not defaults:
            rejected.append({"session": session, "reason": "default arm missing"})
            continue
        # Each repetition is an independently loaded runner, with one unique run ID.
        if len({r["run_id"] for r in group}) != len(group):
            rejected.append({"session": session, "reason": "duplicate run"})
            continue
        baseline_config = effective_configuration(defaults[0])
        if any(effective_configuration(run) != baseline_config for run in defaults):
            rejected.append(
                {"session": session, "reason": "default configuration changed"}
            )
            continue
        winners = speed_winners(group)
        accepted = []
        for arm in winners:
            evidence = [r for r in group if r["experiment"]["arm_id"] == arm]
            if not quality_passes(evidence, defaults):
                continue
            candidate_config = effective_configuration(evidence[0])
            if candidate_config == baseline_config or any(
                effective_configuration(run) != candidate_config for run in evidence
            ):
                continue
            score = statistics.mean(
                (m["output_tokens"] - 1) * 1000 / m["decode_duration_ms"]
                for r in evidence
                for m in r["measurements"]
            )
            accepted.append((score, arm, evidence))
        if not accepted:
            rejected.append(
                {
                    "session": session,
                    "reason": "no arm passes repetition, speed, and output/quality gates",
                }
            )
            continue
        _, arm, evidence = max(accepted, key=lambda item: (item[0], item[1]))
        run = evidence[0]
        ctx = run["tuning"]["context"]
        hw = ctx["hardware"]
        model = ctx["model"]
        runtime = run["execution"]["runtime"]
        knobs = effective_configuration(run)["knobs"]
        statistics_by_arm = arm_statistics(defaults + evidence)
        entry: dict = {
            "id": "community-" + hashlib.sha256(cell.encode()).hexdigest()[:16],
            "hardware": [
                {
                    "gpu_generation": hw["gpu_generation"],
                    "tier": hw["tier"],
                    "gpu_cores": [hw["gpu_cores"], hw["gpu_cores"]],
                    "memory_size": [hw["memory_size"], hw["memory_size"]],
                }
            ],
            "software": {
                "mlx": "===" + ctx["mlx"],
                "macos": "===" + ctx["macos"],
                "rapid_mlx": "===" + runtime["rapid_mlx"],
            },
            "model": {
                "family": model["family"],
                "geometry": [
                    model[k]
                    for k in (
                        "hidden_size",
                        "intermediate_size",
                        "num_experts",
                        "top_k",
                        "bits",
                        "group_size",
                    )
                ],
                "quant_mode": model["quant_mode"],
                "attention_kinds": model["attention_kinds"],
            },
            "workload": {
                "mode": ctx["mode"],
                "width": ctx["width"],
                "batch": ctx["batch"],
            },
            "knobs": knobs,
            "provenance": {
                "evidence": "community-comparison:" + session,
                "hardware_scope": "Measured exact core and memory cell; review workload and model generalization before merging.",
                "date": run["completed_at"][:10],
                "repetitions": {"default": len(defaults), arm: len(evidence)},
                "decision": "Paired local comparison passed freshness, conditions, output/quality and speed gates.",
                "quality": [r["tuning"].get("quality") for r in evidence],
                "mtp_tps": {
                    name: {
                        case: [row[0] for row in rows] for case, rows in cases.items()
                    }
                    for name, cases in statistics_by_arm.items()
                },
                "mtp_ranges_tps": {
                    name: {
                        case: [min(row[2] for row in rows), max(row[3] for row in rows)]
                        for case, rows in cases.items()
                    }
                    for name, cases in statistics_by_arm.items()
                },
            },
        }
        if "rapid_mlx_revision" in runtime:
            entry["provenance"]["commit"] = runtime["rapid_mlx_revision"]
        candidates[cell].append(entry)
    entries = []
    for cell_entries in candidates.values():
        if len({json.dumps(e["knobs"], sort_keys=True) for e in cell_entries}) == 1:
            entries.append(cell_entries[0])
        else:
            rejected.append(
                {
                    "cell": cell_entries[0]["id"],
                    "reason": "conflicting sessions require review",
                }
            )
    return {"entries": entries, "rejected": rejected}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--rapid-version", default=__version__)
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    args = parser.parse_args(argv)
    runs = []
    for path in args.files:
        document = json.loads(path.read_text())
        runs.extend(
            document.get("configurations", [document, *document.get("paired_runs", [])])
        )
    result = distill(
        runs,
        as_of=args.as_of or datetime.now(timezone.utc),
        max_age_days=args.max_age_days,
        rapid_version=args.rapid_version,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
