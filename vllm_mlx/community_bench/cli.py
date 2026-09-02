# SPDX-License-Identifier: Apache-2.0
"""CLI surface for the local-first Community Benchmark workspace."""

from __future__ import annotations

import json
import sys
from typing import Any

from .local_runner import LocalBenchmarkError, run_local
from .workspace import LocalRunArchive, benchmark_catalog, plan_for_alias


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _print_failure(args, exc: Exception) -> int:
    run = exc.run if isinstance(exc, LocalBenchmarkError) else None
    saved = exc.saved if isinstance(exc, LocalBenchmarkError) else False
    if args.json:
        payload = {"error": str(exc), "saved": saved}
        if run is not None:
            payload["run"] = run
        print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
    elif saved and run is not None:
        print(
            f"Benchmark failed; local outcome saved as {run['run_id']}: {exc}",
            file=sys.stderr,
        )
    elif run is not None:
        print(
            f"Benchmark failed; local outcome could not be saved: {exc}",
            file=sys.stderr,
        )
    else:
        print(f"Benchmark command failed: {exc}", file=sys.stderr)
    return 1


def benchmark_command(args) -> int:
    action = args.benchmark_action
    archive = LocalRunArchive.default()
    try:
        if action == "catalog":
            value = benchmark_catalog(memory_gib=args.memory_gib)
        elif action == "plan":
            value = plan_for_alias(args.benchmark_model)
        elif action == "run":
            value = run_local(
                args.benchmark_model,
                archive=archive,
                inherit_process_group=getattr(args, "inherit_process_group", False),
            )
        elif action == "results":
            value = {
                "schema_version": 1,
                "runs": archive.list(limit=getattr(args, "limit", None)),
            }
        elif action == "inspect":
            value = archive.get(args.run_id)
        else:  # pragma: no cover - argparse owns this invariant
            raise ValueError(f"unknown benchmark action {action!r}")
    except Exception as exc:
        return _print_failure(args, exc)

    if args.json:
        _print_json(value)
    elif action == "catalog":
        print("Community Benchmark models (local by default)\n")
        for model in value["models"]:
            marker = "★" if model["focus"] else " "
            fit = model["memory_fit"].replace("_", " ")
            print(f" {marker} {model['alias']:<32} {model['task_type']:<18} {fit}")
        print("\nRun: rapid-mlx benchmark run <model>")
    elif action == "plan":
        model = value["model"]
        print(f"Model:    {model['alias']}")
        print(f"Task:     {model['task_type']}")
        print(
            f"Protocol: {model['protocol_id']} v{value['workload']['protocol_version']}"
        )
        print("Storage:  local only (no upload)")
    elif action == "results":
        runs = value["runs"]
        if not runs:
            print("No local benchmark results yet.")
        for run in runs:
            print(
                f"{run['run_id']}  {run['workload']['task_type']:<18} "
                f"{run['outcome']['status']:<10} {run['completed_at']}"
            )
    elif action == "inspect":
        _print_json(value)
    else:  # run
        print(f"Saved local result {value['run_id']}")
        print("Nothing was uploaded.")
    return 0


__all__ = ["benchmark_command"]
