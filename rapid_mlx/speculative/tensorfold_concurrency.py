# SPDX-License-Identifier: Apache-2.0
"""Explicit concurrency for profiles qualified on the pinned TensorFold runtime."""

from __future__ import annotations

import os
import warnings

QUALIFIED_PROFILES = frozenset(
    {"nemotron-3.5-lightning-tensorfold", "qwen3.8-27b-tensorfold"}
)
LANES_ENV = "RAPID_MLX_TENSORFOLD_LANES"


def request_lanes(profile_id: str, value: int | str | None = None) -> int:
    if value is None:
        value = os.environ.get(LANES_ENV, "1")
        if LANES_ENV in os.environ:
            warnings.warn(
                f"{LANES_ENV} is deprecated; use --tensorfold-lanes instead",
                UserWarning,
                stacklevel=2,
            )
    if isinstance(value, bool) or str(value) not in {"1", "2", "4", "8"}:
        raise ValueError(f"{LANES_ENV} must be 1, 2, 4 or 8")
    lanes = int(value)
    if lanes > 1 and profile_id not in QUALIFIED_PROFILES:
        raise ValueError(
            f"{profile_id} has not qualified concurrent TensorFold decoding; "
            f"unset {LANES_ENV} or set it to 1"
        )
    return lanes
