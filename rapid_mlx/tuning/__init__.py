# SPDX-License-Identifier: Apache-2.0
"""Measured performance policy; callers retain all correctness qualification.

Importing this module does not import MLX. Decisions are deliberately separate
from the output/cache probes that authorize executing an optimization.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version

from .._env import is_falsey, is_truthy
from .defaults import MEASURED_DEFAULTS

QWEN35_AFFINE4 = (2048, 512, 256, 8, 4, 64)


@dataclass(frozen=True)
class Knob:
    description: str
    env: str
    family: str
    geometry: tuple[int, ...]
    modes: tuple[str, ...]
    widths: tuple[int, ...]
    default: bool = False


KNOBS = {
    "qwen35_affine4_bank_views": Knob(
        "Allocation-backed singleton expert-bank bindings",
        "RAPID_MLX_QWEN35_AFFINE4_BANK_VIEWS",
        "qwen35_moe",
        QWEN35_AFFINE4,
        ("plain", "mtp"),
        (1,),
    ),
    "qwen35_residual_norm": Knob(
        "Exact residual/RMSNorm carry in a qualified 40-layer BF16 backbone",
        "RAPID_MLX_QWEN35_RESIDUAL_NORM",
        "qwen35_moe",
        QWEN35_AFFINE4,
        ("plain",),
        (1,),
    ),
}


@cache
def _parse_device_name(name: str) -> str:
    match = re.fullmatch(r"Apple (M[1-9][0-9]*)(?: (Pro|Max|Ultra))?", name.strip())
    return " ".join(part for part in match.groups() if part) if match else "unknown"


def device_key(info: Mapping[str, object]) -> str:
    """Preserve generation and variant; never guess from core count or memory."""
    name = info.get("device_name")
    return _parse_device_name(name) if isinstance(name, str) else "unknown"


@cache
def mlx_version() -> str:
    """Library identity without importing MLX, including on non-Mac hosts."""
    try:
        return version("mlx")
    except PackageNotFoundError:
        return "unknown"


def enabled(
    name: str,
    info: Mapping[str, object],
    *,
    family: str,
    geometry: tuple[int, ...],
    mode: str,
    width: int,
    library_version: str | None = None,
) -> bool:
    """Scope, then env override > exact device/library table > safe default.

    Overrides request an attempt, never bypass scope or a caller's probes.
    Empty/unrecognized environment values behave like an unset override.
    """
    knob = KNOBS[name]
    if (
        family != knob.family
        or geometry != knob.geometry
        or mode not in knob.modes
        or width not in knob.widths
    ):
        return False
    override = os.environ.get(knob.env)
    if is_falsey(override):
        return False
    if is_truthy(override):
        return True
    key = (
        device_key(info),
        library_version if library_version is not None else mlx_version(),
    )
    return MEASURED_DEFAULTS.get(key, {}).get("knobs", {}).get(name, knob.default)
