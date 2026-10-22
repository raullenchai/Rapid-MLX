"""Read HDR LoRA metadata from safetensors.

MLX-native port of the HDR-detection helpers from upstream
``ltx_pipelines.hdr_ic_lora``. Auto-detects whether a LoRA was trained
with HDR LogC3 inputs by inspecting safetensors metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import safetensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HdrLoraConfig:
    """Explicit HDR LoRA parameters.

    Read from LoRA safetensors metadata by :func:`read_hdr_lora_config`,
    or constructed manually for testing.

    Attributes:
        hdr_transform: Name of the HDR compression transform. Currently
            only ``"logc3"`` is supported.
        reference_downscale_factor: Spatial downscale factor for the
            reference video conditioning (matches the LoRA's training
            recipe). 1 = no downscale.
    """

    hdr_transform: str = "logc3"
    reference_downscale_factor: int = 1


def read_hdr_lora_config(lora_path: str) -> HdrLoraConfig | None:
    """Read HDR config from LoRA safetensors metadata.

    Returns ``None`` when the LoRA has no HDR metadata (i.e. it's a
    standard non-HDR LoRA).

    Detection rule: the LoRA's metadata must contain either
    ``hdr_transform`` (preferred, names the transform) or
    ``use_hdr_transform`` (legacy boolean). Empty / missing values
    treated as non-HDR.

    Args:
        lora_path: Path to the LoRA safetensors file.

    Returns:
        :class:`HdrLoraConfig` if HDR metadata present, else ``None``.
    """
    try:
        with safetensors.safe_open(lora_path, framework="numpy") as f:
            metadata = f.metadata() or {}
    except (OSError, ValueError) as e:
        logger.warning("Failed to read metadata from LoRA file %r: %s", lora_path, e)
        return None

    hdr_transform = metadata.get("hdr_transform", "")
    has_hdr = bool(hdr_transform or metadata.get("use_hdr_transform"))
    if not has_hdr:
        return None

    transform = hdr_transform if hdr_transform and hdr_transform != "true" else "logc3"
    scale = int(metadata.get("reference_downscale_factor", 1))
    return HdrLoraConfig(hdr_transform=transform, reference_downscale_factor=scale)
