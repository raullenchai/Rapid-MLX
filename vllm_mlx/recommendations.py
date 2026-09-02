"""Shared RAM-tier model recommendations for CLI and Rapid Desktop."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from .catalog.legacy import load_product_recommendation_policy


@dataclass(frozen=True)
class Recommendation:
    role: str
    alias: str
    footprint_gb: float
    capability_pct: int
    tokens_per_sec: float | None
    launch_flags: tuple[str, ...]
    caveat: str | None = None


@dataclass(frozen=True)
class RecommendationTier:
    floor_gb: int
    picks: tuple[Recommendation, Recommendation]


@lru_cache(maxsize=1)
def load_recommendation_tiers() -> tuple[RecommendationTier, ...]:
    """Decode the validated atomic policy into the stable public API."""

    payload = load_product_recommendation_policy()
    if payload.get("task_type") != "text_generation":
        raise ValueError("default recommendation policy must target text_generation")
    if payload.get("machine_dimension") != "physical_memory_mib":
        raise ValueError("unsupported recommendation machine dimension")

    limitation_copy = {
        "not_for_coding": "Not for coding",
        "basic_chat": "Basic chat",
    }

    tiers: list[RecommendationTier] = []
    for raw_tier in payload["tiers"]:
        picks_list: list[Recommendation] = []
        for raw in raw_tier["picks"]:
            if raw.get("execution_preset_id") is not None:
                raise ValueError(
                    "recommendation execution presets are not supported by the "
                    "legacy serve-flag API"
                )
            limitations = raw.get("limitation_ids", [])
            unknown = set(limitations) - limitation_copy.keys()
            if unknown:
                raise ValueError(
                    f"unknown recommendation limitation IDs: {sorted(unknown)}"
                )
            if len(limitations) > 1:
                raise ValueError(
                    "recommendation display supports at most one limitation ID"
                )
            score_value = raw.get("capability_score_x100")
            if score_value is None:
                raise ValueError(
                    "recommendation display requires capability_score_x100"
                )
            score = int(score_value)
            if score % 100:
                raise ValueError(
                    "capability score cannot be represented as a whole percent"
                )
            picks_list.append(
                Recommendation(
                    role=raw["role"],
                    alias=raw["alias"],
                    footprint_gb=round(int(raw["footprint_mib"]) / 1024, 1),
                    capability_pct=score // 100,
                    tokens_per_sec=(
                        None
                        if raw.get("decode_tokens_per_second_x100") is None
                        else int(raw["decode_tokens_per_second_x100"]) / 100
                    ),
                    launch_flags=(),
                    caveat=(limitation_copy[limitations[0]] if limitations else None),
                )
            )
        picks = tuple(picks_list)
        if len(picks) != 2 or [pick.role for pick in picks] != ["smart", "fast"]:
            raise ValueError(
                f"RAM tier {raw_tier['minimum_memory_mib']} must contain smart + fast picks"
            )
        floor_mib = int(raw_tier["minimum_memory_mib"])
        if floor_mib % 1024:
            raise ValueError("Desktop RAM tier floors must be whole GiB values")
        tiers.append(RecommendationTier(floor_mib // 1024, picks))
    if not tiers or any(
        current.floor_gb <= previous.floor_gb
        for previous, current in zip(tiers, tiers[1:])
    ):
        raise ValueError("recommendation tiers must have strictly increasing floors")
    return tuple(tiers)


def physical_ram_gb() -> float:
    """Return physical RAM on macOS, or zero when the probe is unavailable."""
    try:
        output = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        # Match Rapid Desktop's MacHardware.physicalRAMGB exactly. Apple sells
        # RAM in GiB-sized tiers and the Swift side divides by 2^30; decimal GB
        # here would select a different tier near an 18/24/48 GB boundary.
        return int(output) / float(1 << 30)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def recommendation_tier(ram_gb: float) -> RecommendationTier:
    tiers = load_recommendation_tiers()
    chosen = tiers[0]
    for tier in tiers:
        if ram_gb >= tier.floor_gb:
            chosen = tier
    return chosen


def recommendation_footprint_gb(alias: str) -> float | None:
    """Return the one catalog working-set footprint for ``alias``.

    Picks repeat across RAM tiers, but their footprint may not drift by tier:
    the number describes the model's complete serve-process working set. A
    custom/unmeasured alias returns ``None`` so the caller can retain its
    conservative fallback.
    """
    wanted = alias.casefold()
    found: float | None = None
    for tier in load_recommendation_tiers():
        for pick in tier.picks:
            if pick.alias.casefold() != wanted:
                continue
            if found is not None and found != pick.footprint_gb:
                raise ValueError(
                    f"conflicting recommendation footprints for {alias!r}: "
                    f"{found} and {pick.footprint_gb}"
                )
            found = pick.footprint_gb
    return found


def is_recommended_alias(alias: str, ram_gb: float) -> bool:
    """Whether ``alias`` is a curated pick supported by this host's RAM."""
    wanted = alias.casefold()
    return any(
        tier.floor_gb <= ram_gb
        and any(pick.alias.casefold() == wanted for pick in tier.picks)
        for tier in load_recommendation_tiers()
    )


def recommendation_payload(ram_gb: float) -> dict[str, Any]:
    tier = recommendation_tier(ram_gb)
    return {
        "schema_version": 1,
        "physical_ram_gb": round(ram_gb, 1),
        "tier_floor_gb": tier.floor_gb,
        "picks": [
            {**asdict(pick), "launch_flags": list(pick.launch_flags)}
            for pick in tier.picks
        ],
    }
