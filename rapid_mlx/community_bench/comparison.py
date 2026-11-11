# SPDX-License-Identifier: Apache-2.0
"""Local paired measurements using registered candidate values."""

from __future__ import annotations

import json
import os
import secrets
import statistics
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from uuid import uuid4

from rapid_mlx import tuning
from rapid_mlx.tuning.mtp import _expert_quantization, resolve_max_depth

from .acceleration import AccelerationRequest


def capture_tuning(
    model, language: dict, result, *, config=None, explicit_depth=False
) -> dict:
    """Snapshot installed facts and effective depth before unloading the model."""
    import mlx.core as mx

    model = getattr(model, "language_model", model)
    args = getattr(model, "args", None) or getattr(model, "config", None)
    profile = tuning.model_profile(
        config or (vars(args) if hasattr(args, "__dict__") else {})
    )
    if profile.num_experts:
        quant = _expert_quantization(model)
        profile = replace(
            profile,
            bits=quant[0] if quant else None,
            group_size=quant[1] if quant else None,
            quant_mode=quant[2] if quant else "unknown",
        )
    spec = language["speculative_decoding"]
    mode = "mtp" if spec["method"] == "mtp" else "plain"
    has_head = mode == "mtp" or (
        getattr(model, "mtp", None) is not None
        and callable(getattr(model, "mtp_forward", None))
        and callable(getattr(model, "make_mtp_cache", None))
    )
    context = tuning.runtime_context(
        mx.device_info(), model=profile, mode=mode, has_mtp_head=has_head
    )
    depth = None
    if mode == "mtp":
        depth = resolve_max_depth(
            model, spec["max_draft_tokens"], explicit=explicit_depth
        )
        depth = min(depth, getattr(model, "mtp_max_speculative_tokens", depth))
    return {
        "context": json.loads(json.dumps(tuning.dump(context))),
        "overrides": {"mtp_max_depth": depth} if mode == "mtp" else {},
        "outputs": {
            case: [r.output_digest for r in bucket.rounds_raw]
            for case, bucket in (
                ("pp512-tg128", result.short),
                ("pp2048-tg512", result.long),
            )
        },
    }


def candidate_values(context: dict, *, salt: int = 0):
    """Single-knob adjacent registry values.

    Rotate knobs (rather than flattened neighbors) so two-sided integer knobs
    do not receive twice the community coverage of boolean knobs.
    """
    ctx = tuning.TuningContext(
        tuning.HardwareProfile(**context["hardware"]),
        context["mlx"],
        context["macos"],
        tuning.ModelProfile(
            **{
                **context["model"],
                "attention_kinds": tuple(context["model"]["attention_kinds"]),
            }
        ),
        mode=context["mode"],
        width=context["width"],
        batch=context["batch"],
        has_mtp_head=context["has_mtp_head"],
        rapid_mlx=context["rapid_mlx"],
    )
    neighbors = []
    for name, knob in tuning.KNOBS.items():
        decision = context["knobs"][name]
        if (
            not knob.scope(ctx)
            or knob.requirement(ctx)
            or decision["source"] == "capability"
        ):
            continue
        value = decision["value"]
        values = knob.candidates or (
            (False, True) if type(knob.default) is bool else ()
        )
        normalized = [list(v) if isinstance(v, tuple) else v for v in values]
        if value not in normalized:
            continue
        index = normalized.index(value)
        adjacent = [values[i] for i in (index - 1, index + 1) if 0 <= i < len(values)]
        if name == "mtp_max_depth":
            from rapid_mlx.cli import _resolve_mtp_depth_for_model

            allowed = []
            for depth in adjacent:
                try:
                    _resolve_mtp_depth_for_model(ctx.model.family, depth, explicit=True)
                except ValueError:
                    continue
                allowed.append(depth)
            adjacent = allowed
        if adjacent:
            neighbors.append((name, adjacent))
    if not neighbors:
        return
    offset = salt % len(neighbors)
    neighbors = neighbors[offset:] + neighbors[:offset]
    for name, values in neighbors:
        for value in values:
            yield {name: value}


@contextmanager
def configuration_environment(values):
    """Pass temporary knob values into the fresh benchmark worker only."""
    key = "RAPID_MLX_BENCHMARK_KNOBS"
    previous = os.environ.get(key)
    try:
        if values:
            os.environ[key] = json.dumps(values)
        else:
            os.environ.pop(key, None)
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _configuration_worker(alias, options, sender, lifeline):
    from . import local_runner

    try:
        local_runner._enter_worker_lifetime(lifeline, sender=sender)
        run = local_runner.run_local(
            alias,
            **options,
            progress=lambda message: sender.send(("progress", message)),
            events=lambda event: sender.send(("event", event)),
        )
        sender.send(("ok", run))
    except BaseException as exc:
        sender.send(
            (
                "ok",
                {
                    "failure": {
                        "message": str(exc),
                        "run": getattr(exc, "run", None),
                        "saved": getattr(exc, "saved", False),
                    }
                },
            )
        )
    finally:
        sender.close()


def measure_configuration(alias, *, progress=None, events=None, **options):
    """Fresh interpreter: import-time kernel choices must follow each cell."""
    from .local_runner import LocalBenchmarkError, _run_detached_worker

    def message(kind, value):
        sink = progress if kind == "progress" else events
        if sink is not None:
            sink(value)

    result = _run_detached_worker(
        _configuration_worker,
        (alias, options),
        timeout_s=86400,
        phase="configuration measurement",
        on_message=message,
    )
    if "failure" in result:
        failure = result["failure"]
        raise LocalBenchmarkError(
            failure["message"], failure["run"], saved=failure["saved"]
        )
    return result


def planning_context(alias: str, request: AccelerationRequest) -> dict | None:
    """Resolve metadata and launch settings without loading model weights."""
    from rapid_mlx.cli import _resolve_checkpoint_config
    from rapid_mlx.model_aliases import resolve_profile, resolve_subfolder

    from .acceleration import resolve_benchmark_acceleration
    from .workspace import plan_for_alias

    plan = plan_for_alias(alias)
    if plan["model"]["task_type"] != "text_generation":
        return None
    launch, args = resolve_benchmark_acceleration(alias, request)
    if launch.lane != "engine":
        return None
    config = _resolve_checkpoint_config(alias, resolve_profile(alias))
    if not config:
        from pathlib import Path

        from rapid_mlx.model_downloads import hf_hub_download

        path = hf_hub_download(
            plan["model"]["repo_id"], "config.json", subfolder=resolve_subfolder(alias)
        )
        config = json.loads(Path(path).read_text())
    import mlx.core as mx

    mode = "mtp" if launch.scheduler_kwargs.get("spec_decode") == "mtp" else "plain"
    context: dict = json.loads(
        json.dumps(
            tuning.dump(
                tuning.runtime_context(
                    mx.device_info(),
                    config=config,
                    mode=mode,
                    has_mtp_head=mode == "mtp",
                )
            )
        )
    )
    if mode == "mtp":
        from rapid_mlx.cli import _resolve_mtp_depth_for_model

        decision = context["knobs"]["mtp_max_depth"]
        explicit = getattr(args, "_speculative_config_explicit", False)
        if explicit or decision["source"] in {"default", "capability"}:
            decision.update(
                value=_resolve_mtp_depth_for_model(
                    context["model"]["family"],
                    launch.scheduler_kwargs.get("mtp_max_k", 1),
                    explicit=explicit,
                ),
                source="default",
            )
    return context


def run_comparison(
    alias: str,
    *,
    configs: int,
    repetitions: int = 3,
    archive=None,
    acceleration: AccelerationRequest | None = None,
    salt: int | None = None,
    current_runner=None,
    **kwargs,
) -> dict:
    from itertools import islice

    from .local_runner import run_local
    from .workspace import LocalRunArchive

    run_local = current_runner or run_local
    if not 1 <= configs <= 16:
        raise ValueError("--configs must be between 1 and 16")
    if configs == 1:
        return run_local(alias, archive=archive, acceleration=acceleration, **kwargs)
    if not 3 <= repetitions <= 10:
        raise ValueError("--repetitions must be between 3 and 10")
    request = acceleration or AccelerationRequest()
    destination = archive or LocalRunArchive.default()
    group = str(uuid4())
    salt = secrets.randbelow(2147483648) if salt is None else salt
    runs: list[dict[str, Any]] = []
    progress = kwargs.get("progress")

    def announce(message):
        if progress is not None:
            progress(message)

    context = planning_context(alias, request) or {}
    candidates = (
        list(islice(candidate_values(context, salt=salt), configs - 1))
        if context
        else []
    )
    arms = [("default", {}), *[(f"nearby-{i}", v) for i, v in enumerate(candidates, 1)]]
    count = len(arms)
    actual_repetitions = repetitions if candidates else 1
    if not candidates:
        announce("No tunable settings apply here; ran a single benchmark instead.")
    elif count < configs:
        announce(
            f"Only {len(candidates)} single-change neighbors are valid for this model/mode (requested {configs - 1})"
        )
    announce(f"Plan: {count} configs × {actual_repetitions} repetitions")
    changes = []
    for arm, values in arms[1:]:
        name, value = next(iter(values.items()))
        decision = context["knobs"][name]
        change = f"{name}: {decision['value']} → {value} (current source: {decision['source'].split(':')[0]})"
        changes.append({"arm_id": arm, "change": change})
        announce(change)
    announce("estimated time: shown after the first repetition")
    # Alternate forward/reverse traversals: two arms yield ABBA blocks.
    schedule = [
        (rep, index, *arms[index])
        for rep in range(actual_repetitions)
        for index in (range(count) if rep % 2 == 0 else reversed(range(count)))
    ]
    manifest: dict[str, Any] = {
        "group_id": group,
        "status": "running",
        "plan": {
            "configs": count,
            "repetitions": actual_repetitions,
            "changes": changes,
            "order_strategy": "balanced_rotation",
        },
        "runs": [
            {
                "arm_id": arm,
                "repetition": rep + 1,
                "status": "pending",
                "saved_paths": [],
            }
            for rep, _, arm, _ in schedule
        ],
    }

    def save_manifest():
        return destination._atomic_save(
            destination.root / "experiments", group, manifest
        )

    manifest_path = save_manifest()
    started = time.monotonic()
    active = None
    try:
        for sequence, (rep, index, arm, values) in enumerate(schedule):
            active = manifest["runs"][sequence]
            active["status"] = "running"
            save_manifest()
            announce(
                f"config {index + 1}/{count} · repetition {rep + 1}/{actual_repetitions}"
            )
            arm_request = request
            if "mtp_max_depth" in values:
                overrides = dict(request.overrides)
                original = json.loads(overrides.get("speculative_config", "{}"))
                original.update(
                    method="mtp", num_speculative_tokens=values["mtp_max_depth"]
                )
                overrides["speculative_config"] = json.dumps(original)
                arm_request = replace(request, overrides=overrides)
            with configuration_environment(values):
                runner = run_local if sequence == 0 else measure_configuration
                run = runner(
                    alias,
                    archive=destination,
                    acceleration=arm_request,
                    tuning_observation=True,
                    **kwargs,
                )
            if candidates:
                run["tuning"]["selection"] = {
                    "salt": salt,
                    "changed_knobs": list(values),
                    "requested_configs": configs,
                }
                run["experiment"] = {
                    "group_id": group,
                    "arm_id": arm,
                    "role": "baseline" if arm == "default" else "candidate",
                    "sequence_index": sequence + 1,
                    "order_strategy": "balanced_rotation",
                    "assignment_seed": salt,
                    "varied_fields": [
                        "/tuning/context/knobs/" + name for name in values
                    ]
                    or ["/tuning/context/knobs"],
                }
            path = (
                destination.save(run)
                if candidates
                else destination.runs_dir / (run["run_id"] + ".json")
            )
            runs.append(run)
            active.update(
                status="completed"
                if run["outcome"]["status"] == "completed"
                else "failed",
                run_id=run["run_id"],
                saved_paths=[str(path)],
            )
            save_manifest()
            if active["status"] == "failed":
                raise ValueError(f"Benchmark run {run['run_id']} failed")
            if sequence == 0 and candidates:
                observed = {
                    name: decision["value"]
                    for name, decision in run["tuning"]["context"]["knobs"].items()
                }
                observed.update(run["tuning"]["overrides"])
                if any(
                    observed[name] != context["knobs"][name]["value"]
                    for values in candidates
                    for name in values
                ):
                    raise ValueError(
                        "Loaded settings differ from the comparison plan; the current run is saved. Re-run with explicit settings."
                    )
            if sequence == 0:
                remaining = (time.monotonic() - started) * (len(schedule) - 1)
                announce(
                    f"estimated time remaining: {remaining * 0.8:.0f}–{remaining * 1.5:.0f} seconds"
                )
        if candidates:
            runs[0]["paired_runs"] = json.loads(json.dumps(runs[1:]))
            destination.save(runs[0])
        manifest["status"] = "completed"
        save_manifest()
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        if active is not None and active["status"] == "running":
            active["status"] = "failed"
            failed_run = getattr(exc, "run", None)
            if failed_run and getattr(exc, "saved", False):
                active.update(
                    run_id=failed_run["run_id"],
                    saved_paths=[
                        str(destination.runs_dir / (failed_run["run_id"] + ".json"))
                    ],
                )
        save_manifest()
        completed = sum(item["status"] == "completed" for item in manifest["runs"])
        message = (
            f"Experiment {group}: {completed} completed runs; manifest: {manifest_path}"
        )
        if progress is None:
            import sys

            print(message, file=sys.stderr)
        else:
            announce(message)
        raise
    if not candidates:
        return runs[0]
    return {
        "schema_version": 1,
        "configurations": runs,
        "requested_configs": configs,
        "measured_configs": count,
        "verdicts": verdicts(runs),
    }


def arm_statistics(runs: list[dict]) -> dict:
    """Use per-run case medians as independent fresh-load repetitions."""
    arms: dict[str, dict[str, list]] = {}
    for run in runs:
        arm = run["experiment"]["arm_id"]
        for case in run["workload"]["cases"]:
            samples = [
                m
                for m in run["measurements"]
                if m["case_id"] == case["case_id"]
                and m["decode_duration_ms"] > 0
                and m["output_tokens"] > 1
            ]
            if not samples:
                # No decode-rate evidence: count it as a missing repetition.
                continue
            rates = [
                (m["output_tokens"] - 1) * 1000 / m["decode_duration_ms"]
                for m in samples
            ]
            row = arms.setdefault(arm, {}).setdefault(case["case_id"], [])
            row.append(
                (
                    statistics.median(rates),
                    statistics.median(m["ttft_ms"] for m in samples),
                    min(rates),
                    max(rates),
                )
            )
    return arms


VERDICT_LABELS = {
    "faster": "faster (reliable)",
    "no_reliable_gain": "no reliable gain",
    "outputs_differ": "outputs differ — not comparable",
    "incomplete": "incomplete",
    "quality_failed": "quality check failed",
}


def outputs_equal(runs: list[dict]) -> bool:
    """Require complete, stable per-case digests, including within each load."""
    canonical = set()
    for run in runs:
        outputs = run["tuning"]["outputs"]
        cases = run["workload"]["cases"]
        if set(outputs) != {c["case_id"] for c in cases}:
            return False
        for case in cases:
            hashes = outputs[case["case_id"]]
            if (
                len(hashes) != case["measured_rounds"]
                or not hashes
                or not all(hashes)
                or len(set(hashes)) != 1
            ):
                return False
        canonical.add(json.dumps(outputs, sort_keys=True))
    return len(canonical) == 1


def quality_passes(runs: list[dict]) -> bool:
    for run in runs:
        quality = run["tuning"].get("quality")
        if quality is None:
            continue
        if quality.get("n", 0) < 100 or not quality.get("questions_digest"):
            return False
        for baseline in ("plain", "default"):
            pair = quality.get(baseline, {})
            if abs(pair.get("delta_pp", 100)) > 3 or pair.get("paired_p", 0) < 0.05:
                return False
    return True


def arm_verdict(
    runs: list[dict], arm: str, *, min_repetitions: int = 3
) -> tuple[str, str]:
    """One ordered eligibility and speed decision for every comparison consumer."""
    min_repetitions = max(3, min_repetitions)
    evidence = [r for r in runs if r["experiment"]["arm_id"] in {"default", arm}]
    stats = arm_statistics(evidence)
    expected = {c["case_id"] for r in evidence for c in r["workload"]["cases"]}
    if (
        not expected
        or any(
            {c["case_id"] for c in r["workload"]["cases"]} != expected
            or {
                m["case_id"]
                for m in r["measurements"]
                if m["decode_duration_ms"] > 0 and m["output_tokens"] > 1
            }
            != expected
            for r in evidence
        )
        or any(
            set(stats.get(name, {})) != expected
            or any(len(stats[name][case]) < min_repetitions for case in expected)
            for name in ("default", arm)
        )
        or any(r["outcome"]["status"] != "completed" for r in evidence)
    ):
        return "incomplete", "incomplete"
    if not outputs_equal(evidence):
        return "outputs_differ", "outputs_differ"
    if not quality_passes(evidence):
        return "quality_failed", "quality_failed"
    default, candidate = stats["default"], stats[arm]
    if any(
        statistics.median(r[0] for r in candidate[c])
        < 1.02 * statistics.median(r[0] for r in default[c])
        for c in expected
    ):
        return "no_reliable_gain", "below_threshold"
    if any(
        min(r[2] for r in candidate[c]) <= max(r[3] for r in default[c])
        for c in expected
    ):
        return "no_reliable_gain", "overlapping_ranges"
    if any(
        statistics.median(r[1] for r in candidate[c])
        > 1.05 * statistics.median(r[1] for r in default[c])
        for c in expected
    ):
        return "no_reliable_gain", "ttft_regressed"
    return "faster", "faster"


def verdicts(runs: list[dict]) -> dict:
    return {
        arm: dict(zip(("label", "reason_code"), arm_verdict(runs, arm)))
        for arm in dict.fromkeys(r["experiment"]["arm_id"] for r in runs)
        if arm != "default"
    }


def speed_winners(runs: list[dict], *, min_repetitions: int = 3) -> list[str]:
    return [
        arm
        for arm in dict.fromkeys(r["experiment"]["arm_id"] for r in runs)
        if arm != "default"
        and arm_verdict(runs, arm, min_repetitions=min_repetitions)[0] == "faster"
    ]


def summary_lines(runs: list[dict]) -> list[str]:
    lines = [
        "Your current config vs nearby configs (local observation)",
        "Results are a local observation; this run does not change your configuration.",
        "Config       Case                tok/s median [min, max]     TTFT ms    reps",
    ]
    for arm, cases in arm_statistics(runs).items():
        for case, rows in cases.items():
            rates = [r[0] for r in rows]
            lines.append(
                f"{('current' if arm == 'default' else arm):<12} {case:<19} {statistics.median(rates):7.2f} [{min(r[2] for r in rows):.2f}, {max(r[3] for r in rows):.2f}]  {statistics.median(r[1] for r in rows):9.2f} {len(rows):7}"
            )
    for arm, verdict in verdicts(runs).items():
        lines.append(
            f"{arm}: {VERDICT_LABELS[verdict['label']]} ({verdict['reason_code']})"
        )
    winners = speed_winners(runs)
    lines.append(
        "Local speed-gate winners (>=2%, non-overlapping, >=3 reps per arm): "
        + (", ".join(winners) or "none")
    )
    return lines
