# SPDX-License-Identifier: Apache-2.0
"""Fail-closed qualification for companion-model DSpark serving."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CompanionDSparkError(ValueError):
    """Raised before model load when the requested pair is not qualified."""


@dataclass(frozen=True)
class CompanionDSparkPair:
    target_repo: str
    target_revision: str
    drafter_repo: str
    drafter_revision: str
    num_speculative_tokens: int
    draft_block_size: int


LFM25_VL_3B = CompanionDSparkPair(
    target_repo="LiquidAI/LFM2.5-VL-3B",
    target_revision="35a118d938ce6d123ac2d371649f24a8efb69058",
    drafter_repo="LiquidAI/LFM2.5-VL-3B-DSpark",
    drafter_revision="af77e9306a26e8625fde74d2a3051ab6d21bd955",
    # Public ``num_speculative_tokens`` counts proposals. mlx-vlm's
    # ``draft_block_size`` includes the anchor token, so seven proposals are
    # expressed as an internal width of eight.
    num_speculative_tokens=7,
    draft_block_size=8,
)


def resolve_companion_dspark_pair(
    *, target_repo: str, drafter_repo: str | None, num_speculative_tokens: int
) -> CompanionDSparkPair:
    """Return the one immutable companion pair qualified by Rapid-MLX."""

    pair = LFM25_VL_3B
    if target_repo != pair.target_repo:
        raise CompanionDSparkError(
            "companion DSpark is currently qualified only for "
            f"{pair.target_repo!r}; got target {target_repo!r}"
        )
    if drafter_repo != pair.drafter_repo:
        raise CompanionDSparkError(
            f"{pair.target_repo} requires companion drafter {pair.drafter_repo!r}"
        )
    if num_speculative_tokens != pair.num_speculative_tokens:
        raise CompanionDSparkError(
            f"{pair.target_repo} companion DSpark requires "
            f"num_speculative_tokens={pair.num_speculative_tokens} "
            f"(mlx-vlm draft_block_size={pair.draft_block_size})"
        )
    return pair


def _read_config(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "config.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CompanionDSparkError(
            f"cannot read qualified DSpark artifact config: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CompanionDSparkError(f"invalid DSpark artifact config: {path}")
    return value


def validate_companion_artifacts(
    pair: CompanionDSparkPair, *, target_path: str | Path, drafter_path: str | Path
) -> None:
    """Validate the immutable pair's structural ABI before loading weights."""

    target = _read_config(target_path)
    text = target.get("text_config")
    if not isinstance(text, dict):
        text = {}
    target_shape = (
        target.get("model_type"),
        target.get("dtype"),
        text.get("num_hidden_layers"),
        text.get("hidden_size"),
        text.get("vocab_size"),
    )
    if target_shape != ("lfm2_vl", "bfloat16", 30, 2048, 128000):
        raise CompanionDSparkError(
            "qualified DSpark target ABI mismatch: expected "
            "BF16 lfm2_vl/30 layers/hidden 2048/vocab 128000"
        )

    draft = _read_config(drafter_path)
    architectures = draft.get("architectures")
    dflash = draft.get("dflash_config")
    if not isinstance(architectures, list):
        architectures = []
    if not isinstance(dflash, dict):
        dflash = {}
    try:
        configured_block = int(draft.get("block_size", 0))
        target_layers = tuple(int(v) for v in dflash.get("target_layer_ids", ()))
    except (TypeError, ValueError):
        configured_block = 0
        target_layers = ()
    valid = (
        "Lfm2DSparkDraftModel" in architectures
        and draft.get("dtype") == "bfloat16"
        and draft.get("hidden_size") == 2048
        and draft.get("vocab_size") == 128000
        and configured_block >= pair.draft_block_size
        and dflash.get("num_target_layers") == 30
        and target_layers == (2, 9, 17, 21, 27)
        and draft.get("markov_rank") == 256
    )
    if not valid:
        raise CompanionDSparkError("qualified LFM2.5-VL-3B DSpark drafter ABI mismatch")


__all__ = [
    "CompanionDSparkPair",
    "CompanionDSparkError",
    "LFM25_VL_3B",
    "resolve_companion_dspark_pair",
    "validate_companion_artifacts",
]
