# SPDX-License-Identifier: Apache-2.0
"""Local paired measurements using registered candidate values."""

from __future__ import annotations

import json
import statistics
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


def candidate_requests(context: dict, request: AccelerationRequest):
    """V1 varies installed-head MTP only, preserving every other serve option."""
    ctx = tuning.TuningContext(
        tuning.HardwareProfile(**context["hardware"]),
        context["mlx"],
        context["macos"],
        tuning.ModelProfile(**context["model"]),
        mode="mtp",
        width=1,
        batch=1,
        has_mtp_head=context["has_mtp_head"],
        rapid_mlx=context["rapid_mlx"],
    )
    from rapid_mlx.cli import _resolve_mtp_depth_for_model

    knob = tuning.KNOBS["mtp_max_depth"]
    if not knob.scope(ctx) or knob.requirement(ctx):
        return []
    common = dict(request.overrides)
    # These flags are replaced by each arm's explicit speculative configuration.
    for key in (
        "speculative_config",
        "spec_decode",
        "no_spec_decode",
        "mtp",
        "mtp_max_k",
    ):
        common.pop(key, None)
    original = json.loads(request.overrides.get("speculative_config", "{}"))
    mtp = {"method": "mtp"}
    if original.get("model"):
        mtp["model"] = original["model"]
    arms = [
        ("mtp-off", AccelerationRequest(overrides={**common, "no_spec_decode": True}))
    ]
    for depth in knob.candidates:
        try:
            _resolve_mtp_depth_for_model(ctx.model.family, depth, explicit=True)
        except ValueError:
            continue
        config = {**mtp, "num_speculative_tokens": depth}
        arms.append(
            (
                f"mtp-{depth}",
                AccelerationRequest(
                    overrides={**common, "speculative_config": json.dumps(config)},
                ),
            )
        )
    return arms


def run_sweep(
    alias: str,
    *,
    repetitions: int = 3,
    archive=None,
    acceleration: AccelerationRequest | None = None,
    **kwargs,
) -> dict:
    from .local_runner import run_local
    from .workspace import LocalRunArchive

    if not 1 <= repetitions <= 10:
        raise ValueError("--sweep-reps must be between 1 and 10")
    request = acceleration or AccelerationRequest()
    if request.baseline:
        raise ValueError("--sweep requires the serving default; remove --baseline")
    destination = archive or LocalRunArchive.default()
    group = str(uuid4())
    runs: list[dict[str, Any]] = []

    def measure(arm, arm_request):
        run = run_local(
            alias,
            archive=destination,
            acceleration=arm_request,
            tuning_observation=True,
            **kwargs,
        )
        if "tuning" not in run:
            raise ValueError(
                "--sweep requires a native text runtime with tuning observations"
            )
        run["experiment"] = {
            "group_id": group,
            "arm_id": arm,
            "role": "baseline" if arm == "default" else "candidate",
            "sequence_index": len(runs) + 1,
            "order_strategy": "alternating",
            "varied_fields": ["/execution/task/language/speculative_decoding"],
        }
        destination.save(run)
        runs.append(run)
        return run

    first = measure("default", request)
    candidates = candidate_requests(first["tuning"]["context"], request)
    # Rotate the arm order each repetition, including the already-measured default.
    arms = [("default", request), *candidates]
    for repetition in range(repetitions):
        offset = repetition % len(arms)
        ordered = arms[offset:] + arms[:offset]
        for arm, arm_request in ordered:
            if repetition == 0 and arm == "default":
                continue
            measure(arm, arm_request)
    return {"schema_version": 1, "sweep": runs}


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
        "Local observation",
        "Arm          Case                tok/s median [min, max]     TTFT ms    reps    greedy",
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
                f"{arm:<12} {case:<19} {statistics.median(rates):7.2f} [{min(r[2] for r in rows):.2f}, {max(r[3] for r in rows):.2f}]  {statistics.median(r[1] for r in rows):9.2f} {len(rows):7}    {equality}"
            )
    winners = speed_winners(runs)
    lines.append(
        "Local speed-gate winners (>=2%, non-overlapping, >=3 reps per arm): "
        + (", ".join(winners) or "none")
    )
    return lines
