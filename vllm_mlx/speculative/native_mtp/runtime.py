# SPDX-License-Identifier: Apache-2.0
"""Lazy runtime bridge for the qualified native MTP drafter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import version
from importlib.util import find_spec
from typing import Any

logger = logging.getLogger(__name__)
QUALIFIED_MLX_VLM_VERSION = "0.7.1"
GLM_CACHE_RUNTIME_LABEL = "cache-owned GLM runtime"


def have_glm_cache_runtime() -> bool:
    """Probe the exact cache/model seams Rapid's GLM transaction requires."""
    try:
        from .glm5_compat import (
            _is_stateless_drafter,
            install_glm5_mtp_compatibility,
        )

        install_glm5_mtp_compatibility()
        from mlx_vlm.generate import ar
        from mlx_vlm.models.cache import ArraysCache, PoolingCache
        from mlx_vlm.models.glm5_next import language
        from mlx_vlm.speculative.drafters import load_drafter  # noqa: F401
        from mlx_vlm.speculative.drafters.glm5_next_mtp import (  # noqa: F401
            Glm5NextMTPDraftModel,
        )

        from vllm_mlx.patches.glm5_next_runtime import (
            _has_native_glm5_next_runtime,
        )

        cache_methods = (
            "start_speculation",
            "validate_speculation",
            "commit_speculation",
            "abort_speculation",
        )
        return (
            _has_native_glm5_next_runtime(language)
            and _is_stateless_drafter(Glm5NextMTPDraftModel)
            and all(hasattr(ArraysCache, name) for name in cache_methods)
            and all(hasattr(PoolingCache, name) for name in cache_methods)
            and all(
                hasattr(ar, name)
                for name in (
                    "generate_step",
                    "SpeculativePrefill",
                    "run_speculative_rounds",
                    "speculative_prefill_kwargs",
                )
            )
        )
    except Exception:  # noqa: BLE001 - optional runtime must fail closed
        return False


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
    model_type: str = "qwen3_5_mtp"
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
    expected_model_type: str = "qwen3_5_mtp",
) -> NativeMTPRuntime:
    """Load one qualified MTP sidecar through mlx-vlm's compatible loader."""

    try:
        from mlx_vlm.speculative.drafters import load_drafter
        from mlx_vlm.utils import get_model_path
    except ImportError as exc:
        raise RuntimeError(
            "native MTP requires "
            f"mlx-vlm {QUALIFIED_MLX_VLM_VERSION}; install rapid-mlx[mtp]"
        ) from exc

    # Fail before resolving the sidecar path: ``get_model_path`` may download
    # gigabytes, and a structurally incompatible runtime can never use them.
    if expected_model_type == "glm5_next_mtp" and not have_glm_cache_runtime():
        raise RuntimeError(
            f"GLM native MTP requires the qualified {GLM_CACHE_RUNTIME_LABEL}"
        )

    source = str(get_model_path(drafter_repo, revision=drafter_revision))
    drafter, kind = load_drafter(source, kind="mtp")
    model_type = getattr(getattr(drafter, "config", None), "model_type", None)
    if kind != "mtp" or model_type != expected_model_type:
        raise RuntimeError(
            f"native MTP sidecar architecture mismatch: expected {expected_model_type}"
        )
    configured_block = int(getattr(drafter.config, "block_size", 0) or 0)
    if configured_block != block_size:
        raise RuntimeError(
            "native MTP sidecar block-size mismatch: "
            f"expected {block_size}, got {configured_block}"
        )
    if expected_model_type == "glm5_next_mtp":
        from .transaction import install_generation_hooks

        install_generation_hooks()
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
        model_type=expected_model_type,
    )


__all__ = [
    "NativeMTPRuntime",
    "QUALIFIED_MLX_VLM_VERSION",
    "GLM_CACHE_RUNTIME_LABEL",
    "have_runtime",
    "have_glm_cache_runtime",
    "load_runtime",
]
