# SPDX-License-Identifier: Apache-2.0
"""Strict data-only profiles. Invalid documents/cells never select a value."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .profiles import LatticeContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    version: str | None = None
    cells: tuple[dict[str, Any], ...] = ()


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValueError("missing required keys")
    if value.keys() - set(required) - set(optional):
        raise ValueError("unknown key")


def _label(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", value):
        raise ValueError("invalid identifier")


def _integer(value):
    if type(value) is not int or value < 1:
        raise ValueError("expected positive integer")


def _cell(cell, knobs):
    _keys(
        cell,
        ("id", "knobs", "provenance"),
        ("hardware", "software", "model", "workload", "capabilities"),
    )
    _label(cell["id"])
    _keys(
        cell["provenance"],
        ("evidence",),
        (
            "gpu_cores",
            "memory_size",
            "date",
            "commit",
            "plain_tps",
            "mtp_tps",
            "repetitions",
            "plain_arms_tps",
            "decision",
            "plain_ranges_tps",
            "mtp_ranges_tps",
            "quality",
            "hardware_scope",
        ),
    )
    if (
        not isinstance(cell["provenance"], dict)
        or not isinstance(cell["provenance"].get("evidence"), str)
        or not cell["provenance"]["evidence"]
    ):
        raise ValueError("provenance requires evidence")
    if "hardware" in cell and (
        not isinstance(cell["hardware"], list) or not cell["hardware"]
    ):
        raise ValueError("hardware set required")
    for hw in cell.get("hardware", []):
        _keys(hw, ("gpu_generation", "tier"), ("gpu_cores", "memory_size"))
        _integer(hw["gpu_generation"])
        if hw["tier"] not in ("base", "Pro", "Max", "Ultra"):
            raise ValueError("unknown tier")
        for key in ("gpu_cores", "memory_size"):
            if key in hw:
                bounds = hw[key]
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError("expected inclusive range")
                for bound in bounds:
                    _integer(bound)
                if bounds[0] > bounds[1]:
                    raise ValueError("inverted range")
    software = cell.get("software", {})
    _keys(software, (), ("mlx", "macos", "rapid_mlx"))
    for constraint in software.values():
        if not isinstance(constraint, str) or not constraint:
            raise ValueError("expected software constraint")
        SpecifierSet(constraint)
    model = cell.get("model", {})
    _keys(model, (), ("family", "geometry", "quant_mode", "attention_kinds"))
    for key in ("family", "quant_mode"):
        if key in model:
            _label(model[key])
    if "geometry" in model:
        geometry = model["geometry"]
        if isinstance(geometry, dict):
            _keys(
                geometry,
                (),
                (
                    "hidden_size",
                    "intermediate_size",
                    "num_experts",
                    "top_k",
                    "bits",
                    "group_size",
                ),
            )
            dimensions = geometry.values()
        elif isinstance(geometry, list) and len(geometry) == 6:
            dimensions = geometry
        else:
            raise ValueError("expected six dimensions or named geometry facts")
        for dimension in dimensions:
            _integer(dimension)
    if "attention_kinds" in model:
        if not isinstance(model["attention_kinds"], list):
            raise ValueError("expected attention kinds")
        for kind in model["attention_kinds"]:
            _label(kind)
    workload = cell.get("workload", {})
    _keys(workload, (), ("mode", "width", "batch", "flattened_rows"))
    if "mode" in workload:
        _label(workload["mode"])
    for key in ("width", "batch"):
        if key in workload:
            _integer(workload[key])
    if "flattened_rows" in workload:
        bounds = workload["flattened_rows"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(v) is not int or v < 0 for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("expected inclusive nonnegative row range")
    if not isinstance(cell["knobs"], dict) or not cell["knobs"]:
        raise ValueError("empty knob set")
    for name, value in cell["knobs"].items():
        if name not in knobs or not knobs[name].valid(value):
            raise ValueError("unknown knob or invalid value")
        if isinstance(value, list):
            cell["knobs"][name] = tuple(value)
    capabilities = cell.get("capabilities", [])
    if not isinstance(capabilities, list) or any(
        name not in knobs or not knobs[name].profile_requirement
        for name in capabilities
    ):
        raise ValueError("unknown capability")
    return cell


def load_profile(path, knobs) -> Profile:
    """Read a packaged resource or user file; reject cells independently."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _keys(data, ("schema_version", "profile_version", "generated_at", "cells"))
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported schema")
        _label(data["profile_version"])
        datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        if not isinstance(data["cells"], list):
            raise ValueError("expected cells")
    except (OSError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        logger.debug("Rejected lattice profile: %s", exc)
        return Profile()
    cells = []
    ids = [e.get("id") for e in data["cells"] if isinstance(e, dict)]
    for cell in data["cells"]:
        try:
            cell = _cell(cell, knobs)
            if ids.count(cell["id"]) != 1:
                raise ValueError("duplicate cell id")
        except (ValueError, TypeError, InvalidSpecifier) as exc:
            logger.debug("Rejected lattice cell: %s", exc)
            continue
        cells.append(cell)
    return Profile(data["profile_version"], tuple(cells))


@cache
def builtin_profile() -> Profile:
    from . import KNOBS

    return load_profile(
        files("rapid_mlx.lattice").joinpath("profiles/builtin.json"), KNOBS
    )


def overlay_path() -> Path:
    """Same user config directory as user-aliases.json."""
    return Path.home() / ".config" / "rapid-mlx" / "lattice-profile.json"


@cache
def overlay_profile() -> Profile:
    from . import KNOBS

    return load_profile(overlay_path(), KNOBS)


def matches(cell, ctx: LatticeContext) -> bool:
    """Only declared sets match; absent pinned facts never supply evidence."""
    if "hardware" in cell and not any(
        ctx.hardware.gpu_generation == hw["gpu_generation"]
        and ctx.hardware.tier == hw["tier"]
        and all(
            getattr(ctx.hardware, key) is not None
            and bounds[0] <= getattr(ctx.hardware, key) <= bounds[1]
            for key, bounds in hw.items()
            if key in {"gpu_cores", "memory_size"}
        )
        for hw in cell["hardware"]
    ):
        return False
    for key, constraint in cell.get("software", {}).items():
        try:
            if Version(getattr(ctx, key)) not in SpecifierSet(constraint):
                return False
        except InvalidVersion:
            return False
    for key, value in cell.get("model", {}).items():
        if key == "geometry" and isinstance(value, dict):
            if any(
                getattr(ctx.model, fact) != expected for fact, expected in value.items()
            ):
                return False
        elif getattr(ctx.model, key) != (
            tuple(value) if isinstance(value, list) else value
        ):
            return False
    workload = cell.get("workload", {})
    if "flattened_rows" in workload:
        low, high = workload["flattened_rows"]
        if ctx.flattened_rows is None or not low <= ctx.flattened_rows <= high:
            return False
    return all(
        getattr(ctx, key) == value
        for key, value in workload.items()
        if key != "flattened_rows"
    )
