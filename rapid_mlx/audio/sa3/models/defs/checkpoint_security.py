# SPDX-License-Identifier: Apache-2.0
"""Safe-by-default loading for legacy SA3 PyTorch checkpoints."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import tempfile
import pickle
from pathlib import Path
from typing import Any

from rapid_mlx._env import env_truthy
from rapid_mlx.utils.torch_checkpoint import load_tensor_checkpoint

logger = logging.getLogger(__name__)

_UNSAFE_ENV = "RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE"


def load_torch_checkpoint(path: str | Path) -> Any:
    """Load tensors safely; legacy objects require consent bound to file bytes."""
    import torch

    try:
        return load_tensor_checkpoint(path)
    except pickle.UnpicklingError as exc:
        approved_digest = os.environ.get("RAPID_MLX_SA3_TRUSTED_SHA256", "").lower()
        allowed = env_truthy(_UNSAFE_ENV) and re.fullmatch(
            r"[0-9a-f]{64}", approved_digest
        )
        if not allowed:
            raise RuntimeError(
                f"Refusing unsafe pickle fallback for SA3 checkpoint {path}. "
                f"Use a tensor-only checkpoint, or set {_UNSAFE_ENV}=1 only "
                "together with RAPID_MLX_SA3_TRUSTED_SHA256 set to the SHA-256 "
                "of the specific file you have reviewed."
            ) from exc
        logger.warning(
            "Unsafe SA3 pickle fallback explicitly enabled for %s via %s; "
            "checkpoint loading may execute arbitrary Python code.",
            path,
            _UNSAFE_ENV,
        )
        # Hash the exact private copy that will be deserialized. A mutable source
        # path or cache symlink cannot switch bytes between approval and load.
        with tempfile.TemporaryFile() as approved:
            digest = hashlib.sha256()
            with Path(path).open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    approved.write(chunk)
            if not hmac.compare_digest(digest.hexdigest(), approved_digest):
                raise RuntimeError(
                    "SA3 checkpoint does not match the explicitly trusted SHA-256"
                )
            approved.seek(0)
            return torch.load(approved, map_location="cpu", weights_only=False)
