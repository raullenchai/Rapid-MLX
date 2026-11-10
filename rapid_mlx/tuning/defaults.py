# SPDX-License-Identifier: Apache-2.0
"""Reviewed measurements, keyed by exact device variant and MLX version.

The evidence describes configurations, not independent additive knob gains.
"""

MEASURED_DEFAULTS = {
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
}
