# SPDX-License-Identifier: Apache-2.0
"""Serve acceleration selection and effective benchmark metadata."""

import argparse
import contextlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rapid_mlx.cli_parser import BENCHMARK_ACCELERATION_FLAGS, build_parser


@dataclass(frozen=True)
class AccelerationRequest:
    baseline: bool = False
    overrides: Mapping[str, Any] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    @property
    def is_default(self) -> bool:
        return not self.baseline and not self.overrides


class AccelerationConfigError(ValueError):
    """An acceleration configuration cannot be benchmarked."""


def request_from_args(args: argparse.Namespace) -> AccelerationRequest:
    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    serve = subparsers.choices["serve"]
    overrides = {}
    flags = []
    for action in serve._actions:
        option = next(
            (s for s in action.option_strings if s in BENCHMARK_ACCELERATION_FLAGS),
            None,
        )
        if option and hasattr(args, action.dest):
            overrides[action.dest] = getattr(args, action.dest)
            flags.append(option)
    return AccelerationRequest(
        bool(getattr(args, "baseline", False)), overrides, tuple(flags)
    )


def validate_request(
    request: AccelerationRequest, task_type: str, *, is_fixed_runtime: bool
) -> None:
    if request.baseline and request.overrides:
        raise AccelerationConfigError(
            "--baseline cannot be combined with " + ", ".join(request.flags)
        )
    if task_type != "text_generation" and not request.is_default:
        raise AccelerationConfigError(
            "acceleration flags apply to text-generation benchmarks only"
        )
    if is_fixed_runtime and not request.is_default:
        raise AccelerationConfigError(
            "This model always runs its qualified runtime; acceleration flags and --baseline cannot change it."
        )
    if "speculative_config" in request.overrides:
        try:
            config = json.loads(request.overrides["speculative_config"])
        except (TypeError, ValueError):
            return
        if isinstance(config, dict):
            unknown = sorted(config.keys() - {"method", "num_speculative_tokens"})
            if unknown:
                raise AccelerationConfigError(
                    f"--speculative-config keys {', '.join(unknown)} change speculative "
                    "decoding in ways the benchmark result cannot record; the benchmark "
                    "accepts only method and num_speculative_tokens (the model's serve "
                    "defaults apply to the rest)."
                )


def build_serve_args(alias: str, request: AccelerationRequest) -> argparse.Namespace:
    from rapid_mlx.local_model_path import resolved_model_ref
    from rapid_mlx.model_aliases import resolve_model

    args = build_parser().parse_args(["serve", alias])
    # CLI dispatch uses resolved_model_ref and resolve_model before serve_command;
    # serve_command makes no further model changes before normalization.
    args.model = resolved_model_ref(args.model)
    resolved = resolve_model(args.model)
    if resolved != args.model:
        args._original_alias = args.model
        args.model = resolved
    for dest, value in request.overrides.items():
        setattr(args, dest, value)
    args._prefill_step_size_explicit = "prefill_step_size" in request.overrides
    return args


@contextlib.contextmanager
def _capture_preflight_exit():
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            yield
    except SystemExit as exc:
        raise AccelerationConfigError(output.getvalue().strip() or str(exc)) from None


def _refuse_unsupported(alias: str, lane: str, suffix: bool) -> None:
    if lane != "engine":
        lane = lane.replace("_", " ")
        raise AccelerationConfigError(
            f"`serve {alias}` runs this model on its dedicated {lane} runtime, which the Community Benchmark cannot measure yet. Re-run with --baseline to measure it without accelerations."
        )
    if suffix:
        raise AccelerationConfigError(
            "Suffix decoding cannot be benchmarked because this workload repeats the same prompt across warmup and measured rounds."
        )


def preflight_benchmark_acceleration(alias: str, request: AccelerationRequest) -> None:
    from rapid_mlx.cli import serve_acceleration_preflights, serve_runtime_lane

    with _capture_preflight_exit():
        args = build_serve_args(alias, request)
        serve_acceleration_preflights(args)
        lane = serve_runtime_lane(args)
    _refuse_unsupported(
        alias,
        lane,
        getattr(args, "spec_decode", None) == "suffix"
        or getattr(args, "suffix_decoding", False),
    )


def resolve_benchmark_acceleration(alias: str, request: AccelerationRequest):
    from rapid_mlx.cli import resolve_serve_acceleration

    with _capture_preflight_exit():
        serve_args = build_serve_args(alias, request)
        plan = resolve_serve_acceleration(serve_args)
    _refuse_unsupported(
        alias,
        plan.lane,
        plan.scheduler_kwargs.get("spec_decode") == "suffix"
        or bool(plan.scheduler_kwargs.get("enable_suffix_decoding")),
    )
    return plan, serve_args


def effective_language(
    scheduler_config,
    *,
    runtime_method: str | None,
    draft_identity_digest: str | None,
    context_length,
    live_kv_quant: tuple[int, int] | None,
) -> dict:
    if runtime_method == "mtp":
        spec = {"method": "mtp", "max_draft_tokens": scheduler_config.mtp_max_k}
        if scheduler_config.mtp_sidecar:
            if not draft_identity_digest:
                raise AccelerationConfigError(
                    "MTP sidecar requires a draft model identity digest"
                )
            spec.update(
                assistant_source="external_model",
                draft_model_identity_digest=draft_identity_digest,
            )
        else:
            spec["assistant_source"] = "embedded_head"
    elif runtime_method is None:
        spec = {"method": "none"}
    else:
        raise AccelerationConfigError(
            f"the engine installed {runtime_method} speculative decoding, which the "
            "benchmark result cannot record yet; re-run with --baseline or --no-spec-decode"
        )
    # KV comes from the live decode cache (_live_kv_quant). TurboQuant and
    # min-quantize-tokens compress only prefix-cache store entries; the
    # benchmark disables that cache, so they never quantize a measured round.
    if live_kv_quant is not None:
        group, bits = live_kv_quant
        kv = {
            "mode": "quantized",
            "dtype": f"q{bits}" if bits in (2, 3, 4, 6, 8) else "other",
            "bits_x2": 2 * bits,
            "group_size": group,
            "scheme": "affine",
        }
    else:
        kv = {"mode": "unknown", "dtype": "unknown"}
    language = {
        "speculative_decoding": spec,
        "kv_cache": kv,
        "prefix_cache_enabled": bool(scheduler_config.enable_prefix_cache),
        "prefill_backend": "gpu",
        "prefill_chunk_size": scheduler_config.prefill_step_size,
        "context_length": context_length,
    }
    pflash = scheduler_config.pflash_config
    if pflash is not None and pflash.mode != "off":
        language["pflash"] = {
            "mode": pflash.mode,
            "threshold": int(pflash.threshold),
            "keep_ratio_millionths": round(pflash.keep_ratio * 1_000_000),
        }
    return language


def acceleration_summary(language: Mapping[str, Any]) -> str:
    parts = []
    spec = language["speculative_decoding"]
    if spec["method"] == "mtp":
        suffix = (
            " (external draft)"
            if spec.get("assistant_source") == "external_model"
            else ""
        )
        parts.append(f"MTP k={spec['max_draft_tokens']}{suffix}")
    kv = language["kv_cache"]
    if kv["mode"] == "quantized":
        parts.append(f"KV {kv['dtype']} ({kv.get('scheme', 'quantized')})")
    if language.get("prefix_cache_enabled"):
        parts.append("prefix cache")
    if not parts and "pflash" not in language:
        parts.append("no accelerations")
    parts.append(f"prefill chunk {language['prefill_chunk_size']}")
    if "pflash" in language:
        parts.append(f"PFlash {language['pflash']['mode']}")
    return " · ".join(parts)
