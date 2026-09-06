# SPDX-License-Identifier: Apache-2.0
"""CLI surface for the local-first Community Benchmark workspace."""

from __future__ import annotations

import json
import statistics
import sys
from typing import Any
from urllib.parse import quote

from .atomic_upload import preview_run, upload_run
from .hardware import host_memory_gib
from .local_runner import LocalBenchmarkError, run_local
from .workspace import LocalRunArchive, benchmark_catalog, plan_for_alias

_LEADERBOARD_URL = "https://rapidmlx.com/leaderboard"
_CONTRIBUTOR_BASE_URL = f"{_LEADERBOARD_URL}/contributors"


def _contributor_profile(receipt: dict[str, Any]) -> tuple[str, str] | None:
    """Return the server-assigned public identity and its board URL."""

    contributor = receipt.get("contributor")
    if not isinstance(contributor, dict):
        return None
    name, tag = contributor.get("name"), contributor.get("tag")
    if not isinstance(name, str) or not name or not isinstance(tag, str) or not tag:
        return None
    slug = quote(f"{name}-{tag}", safe="-")
    return f"{name} ·{tag}", f"{_CONTRIBUTOR_BASE_URL}/{slug}"


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _run_model_label(run: dict[str, Any]) -> str:
    """Human label for an archived run: the primary component's repo id.

    Only primary components are considered when any is present, regardless
    of position; other components are consulted only when no component is
    marked primary.
    """

    model = run.get("model")
    components = model.get("components") if isinstance(model, dict) else None
    if not isinstance(components, list):
        return "-"
    typed = [component for component in components if isinstance(component, dict)]
    primary = [c for c in typed if c.get("role") == "primary"]
    # A primary component is authoritative: if one exists but carries no
    # usable label, never attribute the run to an auxiliary (draft) model.
    for component in primary or typed:
        source = component.get("source")
        if isinstance(source, dict):
            repo_id = source.get("repo_id")
            if isinstance(repo_id, str):
                return repo_id
        manifest = component.get("artifact")
        if isinstance(manifest, dict):
            path = manifest.get("path")
            if isinstance(path, str):
                return path
    return "-"


def summarize_measurements(run: dict[str, Any]) -> list[str]:
    """One line per case with the numbers a user actually wants to see.

    Text cases report median decode throughput (``(output_tokens - 1) /
    decode_duration``, the board's formula) and time-to-first-token; image
    and video cases report the median wall time. Medians over completed rounds
    only, matching how the board aggregates. Returns ``[]`` when the run has
    no completed measurements so callers can fall back to the raw record.
    """

    measurements = run.get("measurements")
    if not isinstance(measurements, list):
        return []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for sample in measurements:
        if not isinstance(sample, dict) or not sample.get("completed"):
            continue
        case_id = sample.get("case_id")
        if isinstance(case_id, str):
            by_case.setdefault(case_id, []).append(sample)
    lines: list[str] = []
    for case_id, samples in by_case.items():
        # Same formula as the board (rapidmlx.com) and the standardized
        # ``bench`` runner, applied literally: ``(output_tokens - 1) /
        # decode_window`` — the first token lands at ``ttft_ms`` (llama.cpp
        # ``tg`` / vLLM TPOT semantics). A one-token sample therefore reads
        # 0.0 tok/s here exactly as it does on the board; only samples with
        # no output tokens carry no rate at all. Printing ``N / window``
        # made the CLI read 45.8 tok/s for a run the site showed as 45.5.
        decode = [
            (s["output_tokens"] - 1) / (s["decode_duration_ms"] / 1000.0)
            for s in samples
            if isinstance(s.get("output_tokens"), int | float)
            and s["output_tokens"] >= 1
            and isinstance(s.get("decode_duration_ms"), int | float)
            and s["decode_duration_ms"] > 0
        ]
        ttft = [
            s["ttft_ms"] for s in samples if isinstance(s.get("ttft_ms"), int | float)
        ]
        total = [
            s["total_duration_ms"]
            for s in samples
            if isinstance(s.get("total_duration_ms"), int | float)
        ]
        if decode:
            line = f"  {case_id:<16} {statistics.median(decode):7.1f} tok/s decode"
            if ttft:
                line += f"   TTFT {statistics.median(ttft):6.0f} ms"
        elif total:
            line = f"  {case_id:<16} {statistics.median(total) / 1000.0:7.2f} s per run"
        else:
            continue
        lines.append(f"{line}   ({len(samples)} rounds)")
    return lines


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
            memory_gib = args.memory_gib
            memory_source = "override"
            if memory_gib is None:
                memory_gib = host_memory_gib()
                memory_source = "host"
            value = benchmark_catalog(memory_gib=memory_gib)
            # ``memory_gib`` is what the fit column was computed against;
            # ``memory_source`` says whether that is this Mac's probed
            # unified memory or a planning value the user typed.
            value["memory_gib"] = memory_gib
            value["memory_source"] = memory_source
        elif action == "plan":
            value = plan_for_alias(args.benchmark_model)
        elif action == "run":
            value = run_local(
                args.benchmark_model,
                archive=archive,
                inherit_process_group=getattr(args, "inherit_process_group", False),
            )
        elif action == "results":
            runs = archive.list(limit=getattr(args, "limit", None))
            value = {
                "schema_version": 1,
                "runs": runs,
                "receipts": {
                    run["run_id"]: receipt
                    for run in runs
                    if (receipt := archive.receipt(run["run_id"])) is not None
                },
            }
        elif action == "inspect":
            value = archive.get(args.run_id)
        elif action == "share":
            is_preview = getattr(args, "preview", False)
            if args.json and not args.yes and not is_preview:
                raise ValueError("benchmark share --json requires --yes")
            run = archive.get(args.run_id)
            if is_preview:
                preview = preview_run(run)
                value = {"schema_version": 1, **preview}
            else:
                acceptance = upload_run(
                    run,
                    assume_yes=args.yes,
                    approved_install_id=getattr(args, "install_id", None),
                    approved_payload_digest=getattr(args, "payload_digest", None),
                    approved_body_digest=getattr(args, "body_digest", None),
                    approved_target=getattr(args, "target", None),
                )
                if acceptance is None:
                    value = {"schema_version": 1, "uploaded": False, "cancelled": True}
                else:
                    receipt = acceptance.receipt
                    receipt_saved = True
                    try:
                        archive.save_receipt(receipt, install_id=acceptance.install_id)
                    except (OSError, UnicodeError, ValueError):
                        receipt_saved = False
                    value = {
                        "schema_version": 1,
                        "uploaded": True,
                        "receipt_saved": receipt_saved,
                        "receipt": receipt,
                    }
        else:  # pragma: no cover - argparse owns this invariant
            raise ValueError(f"unknown benchmark action {action!r}")
    except Exception as exc:
        return _print_failure(args, exc)

    if args.json:
        _print_json(value)
    elif action == "catalog":
        print("Community Benchmark models (local by default)")
        memory_gib = value.get("memory_gib")
        if isinstance(memory_gib, int) and value.get("memory_source") == "override":
            print(
                f"Fit column assumes {memory_gib} GB (--memory-gib), "
                "not this Mac's memory\n"
            )
        elif isinstance(memory_gib, int):
            print(f"This Mac: {memory_gib} GB unified memory (fit column below)\n")
        else:
            print(
                "This Mac: memory unknown; pass --memory-gib to fill the fit column\n"
            )
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
        print("Storage:  local; upload requires a separate share command and consent")
    elif action == "results":
        runs = value["runs"]
        if not runs:
            print("No local benchmark results yet.")
        for run in runs:
            print(
                f"{run['run_id']}  {run['workload']['task_type']:<18} "
                f"{run['outcome']['status']:<10} {run['completed_at']}  "
                f"{_run_model_label(run)}"
            )
    elif action == "inspect":
        _print_json(value)
    elif action == "share":
        if getattr(args, "preview", False):
            _print_json(value)
        elif value["uploaded"]:
            receipt = value["receipt"]
            suffix = " (already uploaded)" if receipt["already_exists"] else ""
            print(f"Accepted benchmark {receipt['submission_id']}{suffix}.")
            if profile := _contributor_profile(receipt):
                identity, url = profile
                print(f"You contributed as {identity}.")
                print(f"View your contributions: {url}")
            else:
                print(f"View Community Benchmark: {_LEADERBOARD_URL}")
            if not value["receipt_saved"]:
                print(
                    "Warning: the upload succeeded but its local receipt could not be saved."
                )
        else:
            print("Upload cancelled. Nothing was sent.")
    else:  # run
        print(f"Saved local result {value['run_id']}")
        for line in summarize_measurements(value):
            print(line)
        print("Nothing was uploaded.")
        print(f"Share it: rapid-mlx benchmark share {value['run_id']}")
    return 0


__all__ = ["benchmark_command"]
