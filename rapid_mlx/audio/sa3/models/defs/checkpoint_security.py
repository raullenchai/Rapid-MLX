# SPDX-License-Identifier: Apache-2.0
"""Safe-by-default loading for legacy SA3 PyTorch checkpoints."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

from rapid_mlx._env import env_truthy

logger = logging.getLogger(__name__)

_UNSAFE_ENV = "RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE"


def load_torch_checkpoint(path: str | Path) -> Any:
    """Load tensors safely, requiring explicit consent for pickle fallback."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        allowed = env_truthy(_UNSAFE_ENV)
        if not allowed:
            raise RuntimeError(
                f"Refusing unsafe pickle fallback for SA3 checkpoint {path}. "
                f"Use a tensor-only checkpoint, or set {_UNSAFE_ENV}=1 only "
                "if you trust this checkpoint's source."
            ) from exc
        logger.warning(
            "Unsafe SA3 pickle fallback explicitly enabled for %s via %s; "
            "checkpoint loading may execute arbitrary Python code.",
            path,
            _UNSAFE_ENV,
        )
        return torch.load(path, map_location="cpu", weights_only=False)
