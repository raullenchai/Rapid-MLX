# SPDX-License-Identifier: Apache-2.0
"""Name-only image LoRA request schema (import-light: no image runtime)."""

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
