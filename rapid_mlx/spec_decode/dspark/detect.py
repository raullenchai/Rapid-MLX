"""Fail-closed detection for DeepSeek V4 DSpark checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DSparkMetadata:
    num_layers: int
    block_size: int
    noise_token_id: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int


_REQUIRED_TENSORS = frozenset(
    {
        "mtp.0.main_proj.weight",
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    }
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def detect_dspark_metadata(model_path: str | Path) -> DSparkMetadata | None:
    """Return DSpark metadata only for a complete local checkpoint.

    DeepSeek's Hugging Face config currently carries only the legacy
    ``num_nextn_predict_layers`` field. The authoritative DSpark geometry is
    shipped in ``inference/config.json``; tensor presence is verified against
    the safetensors index so stripped conversions fail closed.
    """

    root = Path(model_path)
    config = _read_json(root / "config.json")
    inference = _read_json(root / "inference" / "config.json")
    index = _read_json(root / "model.safetensors.index.json")
    if config is None or config.get("model_type") != "deepseek_v4":
        return None
    if inference is None or index is None:
        return None
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not _REQUIRED_TENSORS.issubset(weight_map):
        return None

    try:
        num_layers = int(inference["n_mtp_layers"])
        block_size = int(inference["dspark_block_size"])
        noise_token_id = int(inference["dspark_noise_token_id"])
        target_layer_ids = tuple(int(v) for v in inference["dspark_target_layer_ids"])
        markov_rank = int(inference["dspark_markov_rank"])
    except (KeyError, TypeError, ValueError):
        return None
    if num_layers != 3 or block_size <= 0 or not target_layer_ids or markov_rank <= 0:
        return None
    return DSparkMetadata(
        num_layers=num_layers,
        block_size=block_size,
        noise_token_id=noise_token_id,
        target_layer_ids=target_layer_ids,
        markov_rank=markov_rank,
    )
