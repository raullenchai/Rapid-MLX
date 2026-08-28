"""Shared RAM-tier model recommendations for CLI and Rapid Desktop."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Any


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


def load_recommendation_tiers() -> tuple[RecommendationTier, ...]:
    resource = files("vllm_mlx").joinpath("model_recommendations.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported model recommendation schema")

    tiers: list[RecommendationTier] = []
    for raw_tier in payload["tiers"]:
        picks = tuple(
            Recommendation(
                role=raw["role"],
                alias=raw["alias"],
                footprint_gb=float(raw["footprint_gb"]),
                capability_pct=int(raw["capability_pct"]),
                tokens_per_sec=(
                    None
                    if raw.get("tokens_per_sec") is None
                    else float(raw["tokens_per_sec"])
                ),
                launch_flags=tuple(raw.get("launch_flags", ())),
                caveat=raw.get("caveat"),
            )
            for raw in raw_tier["picks"]
        )
        if len(picks) != 2 or [pick.role for pick in picks] != ["smart", "fast"]:
            raise ValueError(
                f"RAM tier {raw_tier['floor_gb']} must contain smart + fast picks"
            )
        tiers.append(RecommendationTier(int(raw_tier["floor_gb"]), picks))
    if not tiers or list(tiers) != sorted(tiers, key=lambda tier: tier.floor_gb):
        raise ValueError("recommendation tiers must be sorted by floor_gb")
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
