# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT

"""Inkling 2.

Text-only runtime; see NOTICE for provenance and port differences.
"""

from rapid_mlx.models.inkling2.config import ModelArgs
from rapid_mlx.models.inkling2.model import Model

__all__ = [
    "Model",
    "ModelArgs",
]
