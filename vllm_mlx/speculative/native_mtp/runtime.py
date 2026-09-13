# SPDX-License-Identifier: Apache-2.0
"""Lazy runtime bridge for the qualified native MTP drafter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any

logger = logging.getLogger(__name__)
QUALIFIED_MLX_VLM_VERSION = "0.6.17"


def have_runtime() -> bool:
    """Return whether the exact runtime qualified by this backend is installed."""

    try:
        return (
            find_spec("mlx_vlm.speculative.drafters") is not None
            and version("mlx-vlm") == QUALIFIED_MLX_VLM_VERSION
        )
    except Exception:  # noqa: BLE001 - a broken optional stack must fail closed
        return False


@dataclass
class NativeMTPRuntime:
    drafter: Any
    drafter_repo: str
    target_revision: str
    drafter_revision: str
    block_size: int
    kind: str = "mtp"
    algorithm: str = "mtp"

    def reset_accept_lens(self) -> None:
        for name in ("accept_lens", "draft_lens"):
            values = getattr(self.drafter, name, None)
            if isinstance(values, list):
                values.clear()

    def accept_lens_snapshot(self) -> list[int]:
        values = getattr(self.drafter, "accept_lens", None)
        return list(values) if isinstance(values, list) else []


def load_runtime(
    drafter_repo: str,
    *,
    target_revision: str,
    drafter_revision: str,
    block_size: int,
) -> NativeMTPRuntime:
    """Load one immutable Qwen3.5-family MTP sidecar through mlx-vlm."""

    try:
        from mlx_vlm.speculative.drafters import load_drafter
        from mlx_vlm.utils import get_model_path
    except ImportError as exc:
        raise RuntimeError(
            "native MTP requires "
            f"mlx-vlm {QUALIFIED_MLX_VLM_VERSION}; install rapid-mlx[mtp]"
        ) from exc

    source = str(get_model_path(drafter_repo, revision=drafter_revision))
    drafter, kind = load_drafter(source, kind="mtp")
    model_type = getattr(getattr(drafter, "config", None), "model_type", None)
    if kind != "mtp" or model_type != "qwen3_5_mtp":
        raise RuntimeError(
            "native MTP sidecar architecture mismatch: expected qwen3_5_mtp"
        )
    configured_block = int(getattr(drafter.config, "block_size", 0) or 0)
    if configured_block != block_size:
        raise RuntimeError(
            "native MTP sidecar block-size mismatch: "
            f"expected {block_size}, got {configured_block}"
        )
    logger.info(
        "Loaded native MTP sidecar %s@%s (block=%d)",
        drafter_repo,
        drafter_revision[:12],
        block_size,
    )
    return NativeMTPRuntime(
        drafter=drafter,
        drafter_repo=drafter_repo,
        target_revision=target_revision,
        drafter_revision=drafter_revision,
        block_size=block_size,
    )


__all__ = [
    "NativeMTPRuntime",
    "QUALIFIED_MLX_VLM_VERSION",
    "have_runtime",
    "load_runtime",
]
