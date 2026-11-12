# SPDX-License-Identifier: Apache-2.0
"""Server-owned image LoRA library: resolve request names to adapter files."""

import os
from pathlib import Path

from ..api.image_loras import (
    LORA_ADAPTER,
    LORA_FAMILIES,
    ImageLora,
    ImageLoras,
    LoraKey,
)

__all__ = [
    "LORA_ADAPTER",
    "LORA_FAMILIES",
    "ImageLora",
    "ImageLoras",
    "LoraKey",
    "list_loras",
    "resolve_loras",
]


def _library() -> Path:
    return (
        Path(os.environ.get("RAPID_MLX_LORA_DIR", "~/.rapid-mlx/loras"))
        .expanduser()
        .resolve()
    )


def _resolve_file(library: Path, name: str) -> Path:
    path = (library / f"{name}.safetensors").resolve()
    if not path.is_relative_to(library) or not path.is_file():
        raise ValueError(f"LoRA '{name}' is not a regular file inside the LoRA library")
    return path


def list_loras() -> list[dict]:
    """List direct safetensors files, excluding unsafe names and escaping symlinks."""
    library = _library()
    data = []
    for entry in sorted(library.glob("*.safetensors"), key=lambda path: path.stem):
        try:
            name = ImageLora(name=entry.stem).name
            path = _resolve_file(library, name)
            data.append(
                {"id": name, "object": "image.lora", "bytes": path.stat().st_size}
            )
        except (ValueError, OSError, RuntimeError):
            continue
    return data


def resolve_loras(loras: ImageLoras | None) -> LoraKey:
    """Validate requests and resolve again at load time, including file mtimes."""
    entries = LORA_ADAPTER.validate_python(loras or [])
    library = _library()
    key = []
    for entry in entries:
        try:
            path = _resolve_file(library, entry.name)
            key.append((str(path), path.stat().st_mtime_ns, entry.scale))
        except (ValueError, OSError, RuntimeError) as exc:
            available = [item["id"] for item in list_loras()][:20]
            raise ValueError(
                f"Unknown or unavailable LoRA '{entry.name}'. Available LoRAs (up to 20): {available}"
            ) from exc
    return tuple(key)
