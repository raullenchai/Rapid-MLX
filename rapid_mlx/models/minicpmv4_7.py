# SPDX-License-Identifier: Apache-2.0
"""Lazy MiniCPM-V 4.7 binding for the pinned multimodal loader."""

import importlib
import sys


def install_minicpmv4_7() -> None:
    """Reuse the architecture binding seam; importing also installs the processor."""
    package = importlib.import_module(
        "rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7"
    )
    sys.modules["mlx_vlm.models.minicpmv4_7"] = package
    parent = importlib.import_module("mlx_vlm.models")
    vars(parent)["minicpmv4_7"] = package
    # Upstream's 4.7 image-message registry entry is absent from 0.7.2.
    # Native serving renders ordered image/video parts through the processor;
    # the pinned prompt helper also needs this entry for text/image callers.
    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

    MODEL_CONFIG["minicpmv4_7"] = MessageFormat.LIST_WITH_IMAGE_FIRST
