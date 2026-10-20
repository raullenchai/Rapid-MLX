"""Opt-in request concurrency for the native V4.1 server (no MLX import)."""

from __future__ import annotations

import os

LANES_ENV = "RAPID_MLX_DSV41_LANES"


def request_lanes() -> int:
    value = os.environ.get(LANES_ENV, "1")
    if value not in ("1", "2", "4", "8"):
        raise ValueError(f"{LANES_ENV} must be one of 1, 2, 4, 8")
    return int(value)
