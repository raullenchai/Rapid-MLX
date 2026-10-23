# SPDX-License-Identifier: Apache-2.0
"""Restricted legacy tensor loading on a runtime with patched unpickling."""

from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version


def load_tensor_checkpoint(path: str | Path) -> Any:
    """Require patched PyTorch and explicit tensor-only deserialization.

    GHSA-63cw-57p8-fm3p affects weights_only itself through PyTorch 2.9.1.
    Passing True also prevents TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD from weakening
    this call. Never retry with unrestricted pickle here.
    """
    import torch

    try:
        installed = Version(str(getattr(torch, "__version__", "unknown")))
    except InvalidVersion as exc:
        raise RuntimeError(
            "Cannot verify PyTorch version for checkpoint loading"
        ) from exc
    if installed < Version("2.10.0") or installed.is_prerelease:
        raise RuntimeError(
            "Legacy tensor checkpoint loading requires stable PyTorch >=2.10.0 "
            "for patched weights_only deserialization; use safetensors or upgrade"
        )
    return torch.load(path, map_location="cpu", weights_only=True)
