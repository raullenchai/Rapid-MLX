# SPDX-License-Identifier: Apache-2.0
"""Local paired measurements using registered candidate values."""

from __future__ import annotations

import json
import os
import secrets
import statistics
from contextlib import contextmanager
from dataclasses import replace
from itertools import combinations, product
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
    """Adjacent registry values, ordered by the number of changed knobs.

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
    for distance in range(1, len(neighbors) + 1):
        for selected in combinations(neighbors, distance):
            for values in product(*(values for _, values in selected)):
                yield dict(zip((name for name, _ in selected), values))


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


def run_comparison(
    alias: str,
    *,
    configs: int = 2,
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

    def measure(arm, values):
        arm_request = request
        if "mtp_max_depth" in values:
            # Explicit serve caps take precedence over policy; replace only that
            # cap while preserving the user's installed head and other options.
            overrides = dict(request.overrides)
            original = json.loads(overrides.get("speculative_config", "{}"))
            original.update(
                method="mtp", num_speculative_tokens=values["mtp_max_depth"]
            )
            overrides["speculative_config"] = json.dumps(original)
            arm_request = replace(request, overrides=overrides)
        with configuration_environment(values):
            run = measure_configuration(
                alias,
                archive=destination,
                acceleration=arm_request,
                tuning_observation=True,
                **kwargs,
            )
        return record(run, arm, values)

    def record(run, arm, values):
        run["tuning"]["selection"] = {
            "salt": salt,
            "changed_knobs": list(values),
            "requested_configs": configs,
        }
        run["experiment"] = {
            "group_id": group,
            "arm_id": arm,
            "role": "baseline" if arm == "default" else "candidate",
            "sequence_index": len(runs) + 1,
            "order_strategy": "alternating",
            "assignment_seed": salt,
            "varied_fields": ["/tuning/context/knobs/" + name for name in values]
            or ["/tuning/context/knobs"],
        }
        destination.save(run)
        runs.append(run)
        return run

    # Discover applicability through the legacy measurement itself. No new
    # process is needed for media, unsupported runtimes, or cells without a
    # neighbor, and parent-process loader/preflight behavior remains intact.
    first: dict = run_local(
        alias,
        archive=destination,
        acceleration=request,
        tuning_observation=True,
        **kwargs,
    )
    if "tuning" not in first:
        return first
    context = json.loads(json.dumps(first["tuning"]["context"]))
    # Effective explicit/clamped depth is the user's current value too.
    for name, value in first["tuning"]["overrides"].items():
        context["knobs"][name]["value"] = value
    candidates = list(islice(candidate_values(context, salt=salt), configs - 1))
    if not candidates:
        return first
    record(first, "default", {})
    arms = [("default", {}), *[(f"nearby-{i}", v) for i, v in enumerate(candidates, 1)]]
    for repetition in range(repetitions):
        offset = repetition % len(arms)
        for arm, values in arms[offset:] + arms[:offset]:
            if repetition == 0 and arm == "default":
                continue
            measure(arm, values)
    first["paired_runs"] = json.loads(json.dumps(runs[1:]))
    destination.save(first)
    return {
        "schema_version": 1,
        "configurations": runs,
        "requested_configs": configs,
        "measured_configs": len(arms),
    }


def arm_statistics(runs: list[dict]) -> dict:
    """Use per-run case medians as independent fresh-load repetitions."""
    arms: dict[str, dict[str, list]] = {}
    for run in runs:
        arm = run["experiment"]["arm_id"]
        for case in run["workload"]["cases"]:
            samples = [
                m for m in run["measurements"] if m["case_id"] == case["case_id"]
            ]
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


def speed_winners(runs: list[dict], *, min_repetitions: int = 3) -> list[str]:
    stats = arm_statistics(runs)
    default = stats["default"]
    winners = []
    for arm, cases in stats.items():
        if arm == "default" or cases.keys() != default.keys():
            continue
        if all(
            len(rows) >= min_repetitions
            and len(default[case]) >= min_repetitions
            and statistics.median(r[0] for r in rows)
            >= 1.02 * statistics.median(r[0] for r in default[case])
            and min(r[2] for r in rows) > max(r[3] for r in default[case])
            for case, rows in cases.items()
        ):
            winners.append(arm)
    return winners


def summary_lines(runs: list[dict]) -> list[str]:
    lines = [
        "Your current config vs nearby configs (local observation)",
        "Config       Case                tok/s median [min, max]     TTFT ms    reps    greedy",
    ]
    baseline = {
        json.dumps(r["tuning"]["outputs"], sort_keys=True)
        for r in runs
        if r["experiment"]["arm_id"] == "default"
    }
    for arm, cases in arm_statistics(runs).items():
        outputs = {
            json.dumps(r["tuning"]["outputs"], sort_keys=True)
            for r in runs
            if r["experiment"]["arm_id"] == arm
        }
        equality = "equal" if len(baseline) == 1 and outputs == baseline else "diverged"
        for case, rows in cases.items():
            rates = [r[0] for r in rows]
            lines.append(
                f"{('current' if arm == 'default' else arm):<12} {case:<19} {statistics.median(rates):7.2f} [{min(r[2] for r in rows):.2f}, {max(r[3] for r in rows):.2f}]  {statistics.median(r[1] for r in rows):9.2f} {len(rows):7}    {equality}"
            )
    stats = arm_statistics(runs)
    for arm, cases in stats.items():
        if arm != "default":
            gains = [
                100
                * (
                    statistics.median(r[0] for r in rows)
                    / statistics.median(r[0] for r in stats["default"][case])
                    - 1
                )
                for case, rows in cases.items()
            ]
            if min(gains) > 0:
                lines.append(
                    f"Nearby config {arm} was faster by {min(gains):.1f}% or more across cases on this machine (local observation; submit to help tune defaults)."
                )
    winners = speed_winners(runs)
    lines.append(
        "Local speed-gate winners (>=2%, non-overlapping, >=3 reps per arm): "
        + (", ".join(winners) or "none")
    )
    return lines
