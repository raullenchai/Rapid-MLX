# SPDX-License-Identifier: Apache-2.0
"""Name-only image LoRA requests and the server-owned adapter library."""

import os
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, TypeAdapter, field_validator


class ImageLora(BaseModel):
    name: str = Field(min_length=1)
    scale: float = Field(default=1.0, ge=-2.0, le=2.0, allow_inf_nan=False)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if any(part in name for part in ("/", "\\", "..", "\0")):
            raise ValueError("LoRA name must be a filename stem, never a path")
        return name


def _unique_loras(loras: list[ImageLora]) -> list[ImageLora]:
    if len({lora.name for lora in loras}) != len(loras):
        raise ValueError("Duplicate LoRA names are not allowed")
    return loras


ImageLoras = Annotated[
    list[ImageLora], Field(max_length=4), AfterValidator(_unique_loras)
]
LORA_ADAPTER = TypeAdapter(ImageLoras)
LoraKey = tuple[tuple[str, int, float], ...]
LORA_FAMILIES = frozenset(
    {
        "flux2-klein",
        "flux3-klein",
        "flux-schnell",
        "flux-dev",
        "qwen-image",
        "qwen-image-edit",
        "qwen-image-2.1",
        "z-image",
    }
)


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
