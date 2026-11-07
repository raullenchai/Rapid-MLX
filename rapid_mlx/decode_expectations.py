"""Offline, configuration-specific decode expectations shared with Desktop.

A reference range describes submitted benchmarks, not a service-level promise.
Sparse or incompatible samples remain unavailable; download size is not an
active-weight measurement and cannot support a promised bandwidth bound.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def snapshot() -> dict[str, Any]:
    payload: object = json.loads(
        Path(__file__).with_name("decode_expectations.json").read_text()
    )
    if not isinstance(payload, dict):
        raise ValueError("decode expectation snapshot must be a JSON object")
    return payload


def chip_name(brand: str) -> str | None:
    value = brand.removeprefix("Apple ").strip()
    return value if re.fullmatch(r"M[1-9][0-9]*(?: Pro| Max| Ultra)?", value) else None


def detected_chip() -> str | None:
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return chip_name(brand)


def expectation(
    alias: str, chip: str | None, ram_gib: float, *, repo_id: str | None = None
) -> dict[str, Any]:
    """Resolve the catalog default; MTP-off data is secondary context only."""
    data = snapshot()
    result: dict[str, Any] = {
        "status": "unavailable",
        "chip": chip,
        "refreshed_at": data["refreshed_at"],
    }
    model = data["models"].get(alias)
    if (
        not model
        or not chip
        or not math.isfinite(ram_gib)
        or ram_gib <= 0
        or repo_id != model["repo_id"]
    ):
        return result
    result["method"] = model["default_method"]
    if ram_gib < model["min_ram_gib"]:
        result["status"] = "insufficient_ram"
        return result
    cohorts = [
        row
        for row in model["cohorts"]
        if row["chip"] == chip
        and row["ram_min_gib"] <= ram_gib <= row["ram_max_gib"]
        and row["n"] >= 3
    ]
    for row in cohorts:
        if (
            row["method"] == model["default_method"]
            and model["default_config_key"] is not None
            and row["config_key"] == model["default_config_key"]
        ):
            return {**result, **row, "status": "measured"}
    baseline = next(
        (
            row
            for row in cohorts
            if row["method"] == "none" and row["config_key"] == "none"
        ),
        None,
    )
    if baseline:
        result["baseline"] = baseline
    return result


def summary(value: dict[str, Any], *, compact: bool = False) -> str:
    """One display contract for models, doctor and cross-language vectors."""
    method = (
        "MTP on"
        if value.get("method") == "mtp"
        else "MTP off"
        if value.get("method") == "none"
        else "configuration unavailable"
    )
    if value["status"] == "measured":
        feel = (
            "smooth"
            if value["median"] >= 30
            else "usable"
            if value["median"] >= 10
            else "tight"
        )
        if compact:
            return (
                f"{value['lower']:.0f}–{value['upper']:.0f} tok/s · {method} · {feel}"
            )
        return f"{value['lower']:.0f}–{value['upper']:.0f} tok/s measured · {method} · {feel} (n={value['n']}, {value['ram_min_gib']:g}–{value['ram_max_gib']:g} GB sampled)"
    if value["status"] == "insufficient_ram":
        return "Expected decode unavailable: below catalog RAM requirement"
    text = "Expected decode range unavailable"
    if "method" in value:
        text += f" · {method}"
    if not compact and (baseline := value.get("baseline")):
        text += f"; MTP-off reference {baseline['lower']:.0f}–{baseline['upper']:.0f} tok/s (n={baseline['n']}, {baseline['ram_min_gib']:g}–{baseline['ram_max_gib']:g} GB sampled)"
    return text
