# SPDX-License-Identifier: Apache-2.0
"""CLI surface for the local-first Community Benchmark workspace."""

from __future__ import annotations

import json
import os
import signal
import statistics
import sys
import threading
from datetime import datetime
from typing import Any
from urllib.parse import quote

from ..byom.preflight import _gb, mlx_recommended_working_set_bytes
from ..leaderboard_links import run_url
from .acceleration import (
    acceleration_summary,
    preflight_benchmark_acceleration,
    request_from_args,
)
from .atomic_upload import preview_run, upload_run
from .hardware import host_memory_gib
from .local_runner import (
    LocalBenchmarkError,
    _is_deepseek_v41_benchmark_target,
    refuse_if_memory_does_not_fit,
    run_local,
)
from .publication import PublicationRefused
from .workspace import (
    LocalRunArchive,
    ProvenanceUnreadable,
    benchmark_catalog,
    describe_case,
    plan_for_alias,
)

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


def _progress_to_stderr(line: str) -> None:
    """Human progress goes to stderr so stdout stays a clean result stream.

    Presentation must never abort a benchmark: a closed or broken stderr
    (``2> >(head -n 3)``) is swallowed rather than raised into the runner.
    """

    try:
        print(line, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


#: Record-separator prefix for machine-readable progress lines under ``--json``.
#: A caller that opts in with ``--progress`` (the Desktop app) can then tell
#: progress from the failure document, which is emitted UNtagged on stderr and
#: surfaced verbatim on a non-zero exit — so the plain ``--json`` contract
#: (nothing on stderr) is preserved for callers that do not pass ``--progress``.
PROGRESS_TAG = "\x1e"


def _tagged_progress_to_stderr(line: str) -> None:
    """Like :func:`_progress_to_stderr` but prefixes each line with
    :data:`PROGRESS_TAG` so a ``--json`` consumer can separate live progress
    from the untagged failure document."""

    try:
        print(f"{PROGRESS_TAG}{line}", file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _tagged_event_to_stderr(payload: dict[str, Any]) -> None:
    """Emit one structured progress event on the machine-readable stream.

    Same ``PROGRESS_TAG`` as the prose lines, but the body is compact JSON.
    Prose never begins with ``{``, so a consumer separates the two on the
    first character without a second marker or a second stream.
    """

    try:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        print(f"{PROGRESS_TAG}{line}", file=sys.stderr, flush=True)
    except (OSError, ValueError, TypeError):
        pass


def _local_time(timestamp: Any) -> str:
    """Render an archived UTC timestamp in the user's local clock.

    Falls back to the raw value when it is not an ISO-8601 instant so a
    hand-edited archive still lists.
    """

    if not isinstance(timestamp, str):
        return "-"
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M local")


def _estimate_bytes(model: dict[str, Any]) -> int | None:
    estimate = model.get("estimated_memory_bytes")
    if isinstance(estimate, int):
        return estimate
    if "estimated_memory_bytes" not in model:
        estimate = model.get("estimated_memory_gb")
        if isinstance(estimate, (int, float)):
            return int(estimate * (1 << 30))
    return None


def _estimate_text(model: dict[str, Any]) -> str | None:
    estimate = _estimate_bytes(model)
    return _gb(estimate) if estimate is not None else None


def _profile_over_budget(model: dict[str, Any], budget_gib: Any) -> bool:
    estimate = _estimate_bytes(model)
    return (
        model.get("memory_fit_basis") == "memory_floor"
        and model.get("memory_fit") == "fits"
        and estimate is not None
        and isinstance(budget_gib, (int, float))
        and estimate > budget_gib * (1 << 30)
    )


def _memory_line(model: dict[str, Any], memory_gib: Any, budget_gib: Any) -> str:
    estimate = _estimate_text(model)
    fit = model.get("memory_fit")
    if estimate is None:
        if fit not in ("fits", "does_not_fit"):
            return "unknown (no estimate for this model)"
        line = "no working-set estimate"
    else:
        line = f"~{estimate} working set"
    if (
        estimate is not None
        and model.get("memory_estimate_source") == "weights_estimate"
    ):
        line += " (estimated from weight size)"
    floor = model.get("memory_floor_gib")
    if model.get("memory_fit_basis") == "memory_floor" and isinstance(
        floor, (int, float)
    ):
        if (
            fit == "does_not_fit"
            and estimate is not None
            and isinstance(memory_gib, (int, float))
            and memory_gib >= floor
            and isinstance(budget_gib, (int, float))
        ):
            budget_text = _gb(int(budget_gib * (1 << 30)))
            return f"{line}; does NOT fit this Mac (weights exceed the GPU limit of about {budget_text})"
        verdict = "fits" if fit == "fits" else "does NOT fit"
        qualification = "qualified for Macs" if fit == "fits" else "requires a Mac"
        explanation = ""
        if _profile_over_budget(model, budget_gib):
            estimate_label = (
                "the weight-size estimate"
                if model.get("memory_estimate_source") == "weights_estimate"
                else "this estimate"
            )
            explanation = f" — for this model the tested profile decides the fit, not {estimate_label}"
        return (
            f"{line}; {verdict} this Mac "
            f"(profile {qualification} with {floor:g} GB or more{explanation})"
        )
    if isinstance(memory_gib, int) and fit in ("fits", "does_not_fit"):
        verdict = "fits" if fit == "fits" else "does NOT fit"
        line += f"; {verdict} this Mac"
        if isinstance(budget_gib, (int, float)):
            budget_text = _gb(int(budget_gib * (1 << 30)))
            ram_text = _gb(memory_gib * (1 << 30))
            line += f" (GPU can use about {budget_text} of {ram_text})"
        else:
            line += f" ({_gb(memory_gib * (1 << 30))})"
    return line


def _print_catalog(value: dict[str, Any], *, show_all: bool) -> None:
    print("Community Benchmark models (local by default)")
    memory_gib = value.get("memory_gib")
    budget = value.get("memory_budget_gib")
    if isinstance(memory_gib, int):
        ram_text = _gb(memory_gib * (1 << 30))
        prefix = (
            f"Assuming {ram_text} unified memory (--memory-gib)"
            if value.get("memory_source") == "override"
            else f"This Mac: {ram_text} unified memory"
        )
        if isinstance(budget, (int, float)):
            prefix += f"; the GPU can use about {_gb(int(budget * (1 << 30)))}"
        print(prefix + " (fit column below)\n")
    else:
        print("This Mac: memory unknown; pass --memory-gib to fill the fit column\n")

    starred_sources: set[Any] = set()

    def row(model: dict[str, Any]) -> str:
        marker = "★" if model.get("focus") else " "
        fit = str(model.get("memory_fit", "unknown")).replace("_", " ")
        floor = model.get("memory_floor_gib")
        if model.get("memory_fit_basis") == "memory_floor" and isinstance(
            floor, (int, float)
        ):
            if (
                fit == "does not fit"
                and isinstance(memory_gib, (int, float))
                and memory_gib >= floor
                and isinstance(budget, (int, float))
            ):
                fit += " (GPU limit)"
            else:
                fit += f" ({floor:g} GB+ profile)"
                if _profile_over_budget(model, budget):
                    fit += "*"
                    starred_sources.add(model.get("memory_estimate_source"))
        estimate = _estimate_text(model)
        memory = f"~{estimate}" if estimate is not None else "?"
        downloaded = "  downloaded" if model.get("model_cached") is True else ""
        return (
            f" {marker} {model['alias']:<32} {model['task_type']:<18} "
            f"{memory:>7}  {fit}{downloaded}"
        )

    # Recommended = focus models verified to fit this Mac. With no memory
    # figure every fit is unknown; focus models are still listed first, but
    # the heading says the fit was not checked rather than promising one.
    memory_known = isinstance(memory_gib, int)
    recommended: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for model in value["models"]:
        fit = model.get("memory_fit")
        if model.get("focus") and (
            fit == "fits" if memory_known else fit != "does_not_fit"
        ):
            recommended.append(model)
        else:
            rest.append(model)
    if memory_known:
        print("Recommended (★ focus models that fit this Mac)")
    else:
        print("Recommended (★ focus models; memory fit unknown, see --memory-gib)")
    if recommended:
        for model in recommended:
            print(row(model))
    elif memory_known:
        print("   (none of the focus models fit this Mac's memory)")
    else:
        print("   (no focus models in this catalog)")
    if show_all:
        print("\nAll other models")
        for model in rest:
            print(row(model))
    else:
        fitting = sum(1 for m in rest if m.get("memory_fit") == "fits")
        verdict = f"{fitting} fit this Mac" if memory_known else "fit unknown"
        print(
            f"\n{len(rest)} more models have a registered protocol "
            f"({verdict}); list them with rapid-mlx benchmark catalog --all"
        )
    if starred_sources:
        if starred_sources == {"weights_estimate"}:
            print(
                "\n* Fit comes from the model's tested memory profile; the size estimate is computed from weight size and is not used for this model."
            )
        else:
            print(
                "\n* Fit comes from the model's tested memory profile; the size estimate shown is not used for this model."
            )
    print("\nRun: rapid-mlx benchmark run <model>")


def _print_plan(value: dict[str, Any]) -> None:
    model = value["model"]
    workload = value.get("workload", {})
    repo = model.get("repo_id")
    print(f"Model:    {model['alias']}" + (f"  ({repo})" if repo else ""))
    print(f"Task:     {model['task_type']}")
    print(f"Protocol: {model['protocol_id']} v{workload.get('protocol_version', '?')}")
    cases = workload.get("cases", [])
    if cases:
        print(f"Cases:    {len(cases)} (run in this order)")
        for case in cases:
            print(f"  {describe_case(case)}")
    memory_line = _memory_line(
        model, value.get("memory_gib"), value.get("memory_budget_gib")
    )
    print(f"Memory:   {memory_line}")
    cached = value.get("model_cached")
    if cached is True:
        print("Download: already in the local Hugging Face cache; nothing to fetch")
    elif cached is False:
        print(
            "Download: not cached yet; `benchmark run` downloads it from "
            "Hugging Face first"
        )
    elif "model_cached" in value:
        print(
            "Download: could not verify the local cache; `benchmark run` "
            "downloads anything that is missing"
        )
    runtime = model.get("runtime")
    if isinstance(runtime, dict):
        status = runtime.get("status")
        message = runtime.get("message")
        if status == "ready":
            print("Runtime:  ready")
        elif isinstance(message, str) and message:
            print(f"Runtime:  {status} — {message}")
        else:
            print(f"Runtime:  {status}")
    print("Storage:  local; upload requires a separate share command and consent")
    print("Nothing is uploaded by this command or by `benchmark run`.")


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
    message = str(exc).removeprefix("error: ")
    if args.json:
        payload = {"error": message, "saved": saved}
        # A refusal is a decision, not a malfunction. Flagged so a GUI can say
        # so in a calm voice instead of presenting it as a crash.
        if isinstance(exc, PublicationRefused):
            payload["refused"] = True
        if run is not None:
            payload["run"] = run
        print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
    elif saved and run is not None:
        print(
            f"Benchmark failed; local outcome saved as {run['run_id']}: {message}",
            file=sys.stderr,
        )
    elif run is not None:
        print(
            f"Benchmark failed; local outcome could not be saved: {message}",
            file=sys.stderr,
        )
    else:
        print(f"Benchmark command failed: {message}", file=sys.stderr)
    return 1


def benchmark_command(args) -> int:
    if args.benchmark_action != "run":
        return _benchmark_command(args)

    from ..cli import (
        LOG_LEVEL_ENV_VAR,
        _configure_bootstrap_logging,
        _resolve_serve_log_level,
    )

    flag_level = getattr(args, "log_level", None)
    level = _resolve_serve_log_level(flag_level, "WARNING")
    _configure_bootstrap_logging(level)
    echo_var = "RAPID_MLX_BENCH_SERVER_LOG_ECHO"
    previous = {key: os.environ.get(key) for key in (LOG_LEVEL_ENV_VAR, echo_var)}
    try:
        os.environ[LOG_LEVEL_ENV_VAR] = level
        if (
            flag_level is not None or previous[LOG_LEVEL_ENV_VAR] is not None
        ) and not args.json:
            os.environ[echo_var] = "1"
        else:
            os.environ.pop(echo_var, None)
        return _benchmark_command(args)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _benchmark_command(args) -> int:
    action = args.benchmark_action
    archive = LocalRunArchive.default()
    try:
        if action == "catalog":
            memory_gib = args.memory_gib
            memory_source = "override"
            if memory_gib is None:
                memory_gib = host_memory_gib()
                memory_source = "host"
            value = benchmark_catalog(
                memory_gib=memory_gib,
                check_cache=True,
                recommended_working_set_bytes=(
                    mlx_recommended_working_set_bytes()
                    if memory_source == "host"
                    else None
                ),
            )
            # ``memory_gib`` is what the fit column was computed against;
            # ``memory_source`` says whether that is this Mac's probed
            # unified memory or a planning value the user typed.
            value["memory_gib"] = memory_gib
            value["memory_source"] = memory_source
        elif action == "plan":
            value = plan_for_alias(
                args.benchmark_model,
                memory_gib=host_memory_gib(),
                recommended_working_set_bytes=mlx_recommended_working_set_bytes(),
                check_cache=True,
            )
        elif action == "run":
            # Progress is for a human watching a terminal. Under --json stderr
            # stays reserved for the failure document the Desktop app surfaces
            # verbatim on a non-zero exit — UNLESS the caller opts in with
            # --progress, which streams RS-tagged progress the Desktop can tell
            # apart from that document and render as a live bar + ETA.
            acceleration = request_from_args(args)
            if not acceleration.baseline and (
                acceleration.overrides or acceleration.flags
            ):
                try:
                    plan = plan_for_alias(args.benchmark_model)
                except (LocalBenchmarkError, KeyError, ValueError):
                    plan = None
                if plan is not None:
                    refuse_if_memory_does_not_fit(
                        args.benchmark_model, plan["model"]["repo_id"]
                    )
                if (
                    plan is not None
                    and plan["model"]["task_type"] == "text_generation"
                    and not _is_deepseek_v41_benchmark_target(plan["model"]["repo_id"])
                ):
                    preflight_benchmark_acceleration(args.benchmark_model, acceleration)
            event_sink = None
            if not args.json:
                progress_sink = _progress_to_stderr
            elif getattr(args, "progress", False):
                progress_sink = _tagged_progress_to_stderr
                # Structured events go only to machine consumers. A terminal
                # user gets prose; a GUI gets the protocol's real case list.
                event_sink = _tagged_event_to_stderr
            else:
                progress_sink = None
            received_signal = signal.SIGINT

            def cancel_on_sigterm(signum, frame):
                nonlocal received_signal
                received_signal = signum
                raise KeyboardInterrupt

            main_thread = threading.current_thread() is threading.main_thread()
            previous_handler = None
            if main_thread:
                previous_handler = signal.signal(signal.SIGTERM, cancel_on_sigterm)
            try:
                try:
                    value = run_local(
                        args.benchmark_model,
                        archive=archive,
                        inherit_process_group=getattr(
                            args, "inherit_process_group", False
                        ),
                        progress=progress_sink,
                        events=event_sink,
                        acceleration=acceleration,
                    )
                finally:
                    if main_thread:
                        signal.signal(signal.SIGTERM, previous_handler)
            except KeyboardInterrupt:
                if args.json:
                    print(
                        json.dumps(
                            {
                                "error": "benchmark cancelled",
                                "cancelled": True,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Benchmark cancelled; benchmark processes were stopped.",
                        file=sys.stderr,
                    )
                return 128 + received_signal
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
            # The build that produced THIS run, not whatever built the app
            # that is reading it. A benchmark measured by a modified tree
            # stays unpublishable after a clean rebuild.
            try:
                run_provenance = archive.provenance(args.run_id)
            except ProvenanceUnreadable as exc:
                # Corruption is refused, not ignored. Raised as a refusal so
                # the caller gets the same calm `refused: true` document a
                # dirty build gets, rather than a stack trace.
                raise PublicationRefused(
                    f"{exc}. This result's build provenance cannot be "
                    "verified, so it cannot be published. The result is saved "
                    "on this Mac and can be inspected; re-run the benchmark to "
                    "record it again."
                ) from exc
            if is_preview:
                preview = preview_run(run, provenance=run_provenance)
                value = {"schema_version": 1, **preview}
            else:
                acceptance = upload_run(
                    run,
                    assume_yes=args.yes,
                    provenance=run_provenance,
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
        _print_catalog(value, show_all=getattr(args, "all", False))
    elif action == "plan":
        _print_plan(value)
    elif action == "results":
        runs = value["runs"]
        if not runs:
            print("No local benchmark results yet.")
        for run in runs:
            kind = (
                run.get("execution", {})
                .get("task", {})
                .get("language", {})
                .get("runtime", {})
                .get("kind")
            )
            print(
                f"{run['run_id']}  {run['workload']['task_type']:<18} "
                f"{run['outcome']['status']:<10} "
                f"{_local_time(run.get('completed_at'))}  "
                f"{_run_model_label(run)}" + (f"  ({kind})" if kind else "")
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
            if link := run_url(receipt["submission_id"]):
                print(f"See it on the leaderboard: {link}")
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
        task = value.get("execution", {}).get("task", {})
        if value.get("workload", {}).get("task_type") == "text_generation" and task.get(
            "language"
        ):
            print(f"Acceleration: {acceleration_summary(task['language'])}")
        for line in summarize_measurements(value):
            print(line)
        print("Nothing was uploaded.")
        print(f"Share it: rapid-mlx benchmark share {value['run_id']}")
    return 0


__all__ = ["benchmark_command"]
