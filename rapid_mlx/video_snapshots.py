# SPDX-License-Identifier: Apache-2.0
"""Resolve video data before handing local paths to the bundled runtime."""

import re
from pathlib import Path


def resolve_model_dir(model: str | Path) -> Path:
    """Accept explicit local data or verify a catalog-pinned Hub snapshot."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()

    from huggingface_hub.constants import HF_HUB_OFFLINE

    from rapid_mlx import model_downloads
    from rapid_mlx.model_aliases import resolve_hub_model, resolve_revision
    from rapid_mlx.model_integrity import verify_snapshot

    revision = resolve_revision(str(model))
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            f"Video model or LoRA {model!s} has no approved immutable revision; "
            "use a pinned catalog identity or an existing local path."
        )
    repo = resolve_hub_model(str(model))
    snapshot = Path(model_downloads.snapshot_download(repo, revision=revision))
    verify_snapshot(
        repo,
        snapshot,
        revision=revision,
        local_files_only=HF_HUB_OFFLINE or model_downloads.disabled(),
    )
    return snapshot.resolve()


def resolve_lora_path(value: str) -> str:
    """Resolve exactly one tensor file, refusing ambiguous LoRA repositories."""
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    directory = resolve_model_dir(value)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {directory}")
    if len(files) != 1:
        raise ValueError(
            f"Ambiguous LoRA snapshot {directory}; specify one local .safetensors file."
        )
    return str(files[0])
