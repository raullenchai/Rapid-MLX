# SPDX-License-Identifier: Apache-2.0
"""Rapid Lattice: small, measured policy layer. Execution sites retain correctness probes."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

from .._env import is_falsey, is_truthy
from .capabilities import requirement_reason
from .defaults import builtin_profile, matches, overlay_profile
from .profiles import (  # noqa: F401 -- public construction surface
    HardwareProfile,
    LatticeContext,
    ModelProfile,
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
    scope: Callable[[LatticeContext], bool]
    requirement: Callable[[LatticeContext], str | None]
    env: str | None = None
    default: object = False
    profile_requirement: bool = False
    candidates: tuple[object, ...] = ()

    def valid(self, value):
        if self.default == ():
            return (
                isinstance(value, (tuple, list))
                and 3 <= len(value) <= 6
                and all(
                    type(v) in (int, float) and abs(v) <= sys.float_info.max
                    for v in value
                )
                and all(v > 0 for v in value[:-1])
                and value[-1] >= 0
            )
        if type(self.default) is int:
            if self.default == 0:
                return type(value) is int and value >= 0
            return type(value) is int and 1 <= value <= 4
        if type(self.default) is bool:
            return type(value) is bool
        return (
            isinstance(value, (tuple, list))
            and len(value) == 2
            and all(type(v) is int for v in value)
            and tuple(value) in ((1, 1), (2, 4))
        )


def _knob(
    name, scope, env=None, default=False, profile_requirement=False, candidates=()
):
    return Knob(
        scope,
        lambda c: requirement_reason(name, c),
        env,
        default,
        profile_requirement,
        candidates,
    )


KNOBS = {
    "mtp_gdn_fusion": _knob(
        "mtp_gdn_fusion",
        lambda c: c.mode == "mtp" and c.batch == 1 and c.width == 1,
    ),
    "mtp_prefill_block_size": _knob(
        "mtp_prefill_block_size",
        lambda c: c.mode == "mtp" and c.batch == 1 and c.width == 1,
        default=0,
    ),
    "mtp_forward_costs_ms": _knob(
        "mtp_forward_costs_ms",
        lambda c: c.mode == "mtp" and c.batch == 1 and c.width == 1,
        default=(),
    ),
    "mtp_max_depth": _knob(
        "mtp_max_depth",
        lambda c: c.mode == "mtp" and c.batch == 1,
        "RAPID_MLX_MTP_MAX_DEPTH",
        default=1,
        candidates=(1, 2, 3, 4),
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
        candidates=((1, 1), (2, 4)),
    ),
    "affine8_attention": _knob(
        "affine8_attention",
        lambda c: c.batch == 1,
        "RAPID_MLX_AFFINE8_ATTENTION",
        profile_requirement=True,
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
    profile_version: str | None = None


def local_calibration(ctx: LatticeContext) -> Mapping[str, object]:
    """Reserved precedence slot: no files, probes, or calibration in this MVP."""
    return {}


def resolve(
    name: str, ctx: LatticeContext, *, environ: Mapping[str, str] | None = None
) -> Decision:
    knob = KNOBS[name]
    reason = knob.requirement(ctx)
    if knob.profile_requirement and not any(
        name in cell.get("capabilities", ()) and matches(cell, ctx)
        for cell in builtin_profile().cells
    ):
        reason = "unqualified runtime"
    if reason or not knob.scope(ctx):
        return Decision(
            False if type(knob.default) is bool else knob.default,
            "capability",
            reason or "outside knob scope",
        )
    environment = os.environ if environ is None else environ
    try:
        values = json.loads(environment.get("RAPID_MLX_BENCHMARK_KNOBS", "{}"))
        value = values.get(name) if isinstance(values, dict) else None
        if knob.valid(value):
            return Decision(value, "benchmark")
    except (ValueError, TypeError):
        pass
    override = (
        (os.environ if environ is None else environ).get(knob.env) if knob.env else None
    )
    if type(knob.default) is int:
        try:
            depth = int(cast(str, override))
        except (ValueError, TypeError):
            depth = None
        if knob.valid(depth):
            return Decision(depth, "env")
    elif is_falsey(override):
        return Decision(False, "env")
    elif is_truthy(override):
        return Decision(True, "env")
    local = local_calibration(ctx).get(name)
    if local is not None and knob.valid(local):
        return Decision(local, "local")
    for source, profile in (
        ("overlay", overlay_profile()),
        ("builtin", builtin_profile()),
    ):
        cells = [e for e in profile.cells if name in e["knobs"] and matches(e, ctx)]
        if cells:
            values = [e["knobs"][name] for e in cells]
            if all(v == values[0] for v in values):
                return Decision(
                    values[0],
                    source + ":" + min(e["id"] for e in cells),
                    profile_version=profile.version,
                )
            # Ambiguous cells never silently choose one conflicting result.
            return Decision(
                knob.default,
                "default",
                "conflicting lattice profile cells",
                profile.version,
            )
    return Decision(knob.default, "default")


def dump(ctx: LatticeContext) -> dict:
    """Identifier-free diagnostics; profile versions accompany selected cells."""
    return {
        **asdict(ctx),
        "profiles": {
            "builtin": builtin_profile().version,
            "overlay": overlay_profile().version,
        },
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
