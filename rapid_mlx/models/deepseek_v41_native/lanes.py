"""Explicit request concurrency for the native V4.1 server (no MLX import)."""

from __future__ import annotations

import os

LANES_ENV = "RAPID_MLX_DSV41_LANES"


def request_lanes(value: int = 1) -> int:
    if LANES_ENV in os.environ:
        raise ValueError(
            f"{LANES_ENV} is retired; unset it and use --dsv41-lanes 1|2|4|8 instead"
        )
    if type(value) is not int or value not in (1, 2, 4, 8):
        raise ValueError("--dsv41-lanes must be one of 1, 2, 4, 8")
    return value


def serve_lanes(value: int, model_name: str | None) -> int:
    from .artifacts import is_product_target

    lanes = request_lanes(value)
    if lanes != 1 and not is_product_target(model_name or ""):
        raise ValueError("--dsv41-lanes requires a native DeepSeek V4.1 model")
    return lanes
