# SPDX-License-Identifier: Apache-2.0
"""Small, measured policy layer. Execution sites retain correctness probes."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

from .._env import is_falsey, is_truthy
from .capabilities import requirement_reason
from .defaults import MEASURED_DEFAULTS
from .profiles import (  # noqa: F401 -- public construction surface
    HardwareProfile,
    ModelProfile,
    TuningContext,
    device_key,
    mlx_version,
    model_profile,
    runtime_context,
)

QWEN35_AFFINE4 = (2048, 512, 256, 8, 4, 64)


def _expert(c):
    return (
        c.model.family
        in {
            "qwen35_moe",
            "qwen3_5_moe",
            "qwen3_6_moe",
            "qwen3_5_moe_text",
            "qwen3_6_moe_text",
        }
        and c.model.geometry == QWEN35_AFFINE4
        and c.batch == 1
        and c.mode in {"plain", "mtp"}
    )


@dataclass(frozen=True)
class Knob:
    scope: Callable[[TuningContext], bool]
    requirement: Callable[[TuningContext], str | None]
    env: str | None = None
    default: object = False


def _knob(name, scope, env=None, default=False):
    return Knob(scope, lambda c: requirement_reason(name, c), env, default)


KNOBS = {
    "mtp_max_depth": _knob(
        "mtp_max_depth",
        lambda c: c.mode == "mtp" and c.batch == 1,
        "RAPID_MLX_MTP_MAX_DEPTH",
        default=1,
    ),
    "qwen35_affine4_bank_views": _knob(
        "qwen35_affine4_bank_views",
        lambda c: _expert(c) and c.width == 1,
        "RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS",
    ),
    "qwen35_residual_norm": _knob(
        "qwen35_residual_norm",
        lambda c: _expert(c) and c.width == 1 and c.mode == "plain",
        "RAPID_MLX_QWEN35_RESIDUAL_NORM",
    ),
    "qwen35_affine4_rows": _knob(
        "qwen35_affine4_rows",
        lambda c: _expert(c) and 1 <= c.width <= 9,
        default=(1, 1),
    ),
    "affine8_attention": _knob(
        "affine8_attention", lambda c: c.batch == 1, "RAPID_MLX_AFFINE8_ATTENTION"
    ),
    "qsa_stage1": _knob(
        "qsa_stage1", lambda c: c.batch == 1 and c.width >= 64, "RAPID_MLX_QSA_STAGE1"
    ),
    "qsa_indexed_splitk": _knob(
        "qsa_indexed_splitk",
        lambda c: c.batch == 1 and c.width in {1, 3},
        "RAPID_MLX_QSA_INDEXED_SPLITK",
    ),
    "lane_tensor_units": _knob("lane_tensor_units", lambda c: True, default=True),
    "lane_simd_probe": Knob(
        lambda c: True,
        lambda c: (
            "known pipeline limits"
            if c.hardware.gpu_generation is not None and c.hardware.gpu_generation >= 15
            else None
        ),
        default=True,
    ),
}


@dataclass(frozen=True)
class Decision:
    value: object
    source: str
    reason: str | None = None


def local_calibration(ctx: TuningContext) -> Mapping[str, object]:
    """Reserved precedence slot: no files, probes, or calibration in this MVP."""
    return {}


def _valid(knob, value):
    if type(knob.default) is int:
        return type(value) is int and 1 <= value <= 4
    return (
        type(value) is bool if type(knob.default) is bool else value in ((1, 1), (2, 4))
    )


def _score(entry, ctx):
    hw = ctx.hardware
    # Never borrow across generations, tiers, or pinned software versions.
    if any(getattr(hw, k) != entry[k] for k in ("gpu_generation", "tier")):
        return None
    if any(getattr(ctx, k) != entry[k] for k in ("mlx", "macos") if k in entry):
        return None
    # Exact feature matches win; otherwise choose nearest known cores/memory
    # within the qualified generation+tier. Missing features add no evidence.
    distances = [
        abs(getattr(hw, k) - entry[k]) / max(entry[k], 1)
        for k in ("gpu_cores", "memory_size")
        if getattr(hw, k) is not None and k in entry
    ]
    return (len(distances), -sum(distances))


def resolve(
    name: str, ctx: TuningContext, *, environ: Mapping[str, str] | None = None
) -> Decision:
    knob = KNOBS[name]
    reason = knob.requirement(ctx)
    if reason or not knob.scope(ctx):
        return Decision(
            False if type(knob.default) is bool else knob.default,
            "blocked",
            reason or "outside knob scope",
        )
    override = (
        (os.environ if environ is None else environ).get(knob.env) if knob.env else None
    )
    if type(knob.default) is int:
        try:
            depth = int(cast(str, override))
        except (ValueError, TypeError):
            depth = None
        if _valid(knob, depth):
            return Decision(depth, "env")
    elif is_falsey(override):
        return Decision(False, "env")
    elif is_truthy(override):
        return Decision(True, "env")
    local = local_calibration(ctx).get(name)
    if local is not None and _valid(knob, local):
        return Decision(local, "local-calibration")
    candidates = [
        (score, e)
        for e in MEASURED_DEFAULTS
        if name in e["knobs"] and (score := _score(e, ctx)) is not None
    ]
    if candidates:
        best = max(score for score, _ in candidates)
        entries = [e for score, e in candidates if score == best]
        values = [e["knobs"][name] for e in entries]
        if all(v == values[0] for v in values) and _valid(knob, values[0]):
            return Decision(values[0], "table:" + min(e["id"] for e in entries))
    return Decision(knob.default, "default")


def dump(ctx: TuningContext) -> dict:
    """Pure identifier-free diagnostic data; no raw device names/config/env."""
    return {
        **asdict(ctx),
        "knobs": {name: asdict(resolve(name, ctx)) for name in KNOBS},
    }


def enabled(
    name: str,
    info: Mapping,
    *,
    family: str,
    geometry: tuple[int, ...],
    mode: str,
    width: int,
    library_version: str | None = None,
) -> bool:
    """Adapter for already-qualified round-13 callers; no per-token probes."""
    if len(geometry) != 6:
        return False
    ctx = runtime_context(
        info,
        model=ModelProfile(family, *geometry),
        library_version=library_version,
        mode=mode,
        width=width,
    )
    return bool(resolve(name, ctx).value)
