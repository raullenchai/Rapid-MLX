# SPDX-License-Identifier: Apache-2.0
"""Reviewed measurements, keyed by exact device variant and MLX version.

The evidence describes configurations, not independent additive knob gains.
"""

from typing import TypedDict


class Measurement(TypedDict, total=False):
    knobs: dict[str, bool]
    date: str
    commit: str
    evidence: str
    plain_tps: tuple[float, float]
    mtp_tps: tuple[float, float]
    repetitions: dict[str, int]
    plain_arms_tps: dict[str, float]
    decision: str


MEASURED_DEFAULTS: dict[tuple[str, str], Measurement] = {
    ("M3 Ultra", "0.32.4"): {
        "knobs": {
            "qwen35_affine4_bank_views": True,
            "qwen35_residual_norm": True,
        },
        "date": "2026-11-09",
        "commit": "75444b54d95db2ad96db630c8fd6afd7333f2ced",
        "evidence": "docs/engineering/performance/2026-11-09-m3-singleton-backbone.md",
        "plain_tps": (161.93, 178.21),
        "mtp_tps": (168.68, 169.17),
        "repetitions": {"plain": 3, "mtp": 6},
    },
    ("M4 Pro", "0.32.4"): {
        "knobs": {
            "qwen35_affine4_bank_views": False,
            "qwen35_residual_norm": False,
        },
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
        "decision": "Measured off: each isolated plain gain is below 2%; combined gain alone does not qualify either knob.",
    },
    ("M5 Max", "0.32.4"): {
        "knobs": {
            "qwen35_affine4_bank_views": False,
            "qwen35_residual_norm": False,
        },
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
        "decision": "Measured off: each isolated plain gain is below 2%; combined gain alone does not qualify either knob.",
    },
}
