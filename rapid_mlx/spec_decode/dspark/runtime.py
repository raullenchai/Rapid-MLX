# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm bridge for the qualified LFM companion DSpark runtime."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .artifacts import CompanionDSparkArtifacts
from .eligibility import CompanionDSparkPair, validate_companion_artifacts

QUALIFIED_MLX_VLM_VERSION = "0.7.2"


def have_runtime() -> bool:
    try:
        return version("mlx-vlm") == QUALIFIED_MLX_VLM_VERSION
    except PackageNotFoundError:
        return False


@dataclass
class CompanionDSparkRuntime:
    drafter: Any
    drafter_repo: str
    target_revision: str
    drafter_revision: str
    num_speculative_tokens: int
    draft_block_size: int
    kind: str = "dflash"
    algorithm: str = "dspark"

    def reset_accept_lens(self) -> None:
        values = getattr(self.drafter, "accept_lens", None)
        if isinstance(values, list):
            values.clear()

    def accept_lens_snapshot(self) -> list[int]:
        values = getattr(self.drafter, "accept_lens", None)
        return list(values) if isinstance(values, list) else []


def load_runtime(
    pair: CompanionDSparkPair,
    artifacts: CompanionDSparkArtifacts,
) -> tuple[Any, Any, CompanionDSparkRuntime]:
    """Load and attach the qualified target/drafter pair or fail startup."""

    if not have_runtime():
        raise RuntimeError(
            "LFM companion DSpark requires exactly mlx-vlm "
            f"{QUALIFIED_MLX_VLM_VERSION}; install the qualified vision runtime"
        )
    validate_companion_artifacts(
        pair,
        target_path=artifacts.target_path,
        drafter_path=artifacts.drafter_path,
    )

    from mlx_vlm import load
    from mlx_vlm.speculative.drafters import (
        load_drafter,
        validate_drafter_compatibility,
    )

    model, processor = load(artifacts.target_path)
    drafter, kind = load_drafter(artifacts.drafter_path, kind=None)
    if kind != "dflash":
        raise RuntimeError(
            "LFM companion DSpark runtime mismatch: upstream drafter did not "
            "select its required speculative walk"
        )
    validate_drafter_compatibility(model, drafter, kind)
    runtime = CompanionDSparkRuntime(
        drafter=drafter,
        drafter_repo=pair.drafter_repo,
        target_revision=pair.target_revision,
        drafter_revision=pair.drafter_revision,
        num_speculative_tokens=pair.num_speculative_tokens,
        draft_block_size=pair.draft_block_size,
        kind=kind,
    )
    return model, processor, runtime


__all__ = [
    "CompanionDSparkRuntime",
    "QUALIFIED_MLX_VLM_VERSION",
    "have_runtime",
    "load_runtime",
]
