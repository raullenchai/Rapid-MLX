# SPDX-License-Identifier: Apache-2.0
"""Reviewed configurations, including combinations; gains are not additive."""

from typing import NotRequired, TypedDict


class MeasuredDefault(TypedDict):
    id: str
    gpu_generation: int
    tier: str
    knobs: dict[str, bool | tuple[int, int]]
    evidence: str
    mlx: NotRequired[str]
    macos: NotRequired[str]
    gpu_cores: NotRequired[int]
    memory_size: NotRequired[int]
    date: NotRequired[str]
    commit: NotRequired[str]
    plain_tps: NotRequired[tuple[float, float]]
    mtp_tps: NotRequired[tuple[float, float]]
    repetitions: NotRequired[dict[str, int]]
    plain_arms_tps: NotRequired[dict[str, float]]
    decision: NotRequired[str]
    plain_ranges_tps: NotRequired[tuple[tuple[float, float], tuple[float, float]]]


MEASURED_DEFAULTS: list[MeasuredDefault] = [
    {
        "id": "singleton-15-ultra",
        "gpu_generation": 15,
        "tier": "Ultra",
        "mlx": "0.32.4",
        "gpu_cores": 80,
        "memory_size": 274877906944,
        "knobs": {"qwen35_affine4_bank_views": True, "qwen35_residual_norm": True},
        "date": "2026-11-09",
        "commit": "75444b54d95db2ad96db630c8fd6afd7333f2ced",
        "evidence": "docs/engineering/performance/2026-11-09-m3-singleton-backbone.md",
        "plain_tps": (161.93, 178.21),
        "mtp_tps": (168.68, 169.17),
        "repetitions": {"plain": 3, "mtp": 6},
    },
    {
        "id": "singleton-16-pro",
        "gpu_generation": 16,
        "tier": "Pro",
        "mlx": "0.32.4",
        "gpu_cores": 16,
        "memory_size": 51539607552,
        "knobs": {"qwen35_affine4_bank_views": False, "qwen35_residual_norm": False},
        "date": "2026-11-09",
        "commit": "e79576e4bd94a19f3e299fb86d70d1dad1565996",
        "evidence": "docs/engineering/performance/2026-11-09-m3-singleton-backbone.md",
        "plain_tps": (100.29, 101.56),
        "mtp_tps": (103.18, 103.17),
        "plain_arms_tps": {
            "off": 100.29,
            "bank_views": 100.19,
            "residual_norm": 101.42,
            "both": 101.56,
        },
        "repetitions": {"plain": 3, "mtp": 3},
        "decision": "Pair stays off: plain +1.27%, below the 2% gate.",
    },
    {
        "id": "singleton-17-max",
        "gpu_generation": 17,
        "tier": "Max",
        "mlx": "0.32.4",
        "gpu_cores": 40,
        "memory_size": 38654705664,
        "knobs": {"qwen35_affine4_bank_views": True, "qwen35_residual_norm": True},
        "date": "2026-11-09",
        "commit": "e79576e4bd94a19f3e299fb86d70d1dad1565996",
        "evidence": "docs/engineering/performance/2026-11-09-m3-singleton-backbone.md",
        "plain_tps": (167.27, 171.81),
        "mtp_tps": (180.03, 179.93),
        "plain_arms_tps": {
            "off": 167.27,
            "bank_views": 168.94,
            "residual_norm": 169.06,
            "both": 171.81,
        },
        "repetitions": {"plain": 3, "mtp": 3},
        "decision": "Enable the measured pair: plain +2.71%, n=3 non-overlapping ranges (167.07–167.46 "
        "vs 171.72–172.13); MTP -0.06% within spread. No isolated gain claimed.",
        "plain_ranges_tps": ((167.07, 167.46), (171.72, 172.13)),
    },
    {
        "id": "affine4-m4-pro-rows",
        "gpu_generation": 16,
        "tier": "Pro",
        "knobs": {"qwen35_affine4_rows": (2, 4)},
        "evidence": "docs/engineering/performance/fused-affine4-moe.md",
        "decision": "Preserve previously qualified row tiles; historical gate had no software pin.",
    },
    {
        "id": "affine8-m3-ultra",
        "gpu_generation": 15,
        "tier": "Ultra",
        "mlx": "0.32.3",
        "macos": "26.5.2",
        "knobs": {"affine8_attention": True},
        "evidence": "docs/engineering/performance/affine8-attention-m3-ultra.json",
    },
]
