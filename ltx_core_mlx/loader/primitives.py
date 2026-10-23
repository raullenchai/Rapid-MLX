"""Core types for model weight loading and LoRA operations.

Ported from ltx-core/src/ltx_core/loader/primitives.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx

from ltx_core_mlx.loader.sd_ops import SDOps


@dataclass(frozen=True)
class StateDict:
    """Container for a model state dictionary.

    Attributes:
        sd: Dictionary mapping parameter names to arrays.
        size: Total memory footprint in bytes.
        dtype: Set of array dtypes present.
    """

    sd: dict[str, mx.array]
    size: int
    dtype: set[mx.Dtype]

    def footprint(self) -> int:
        return self.size


class LoraPathStrengthAndSDOps(NamedTuple):
    """LoRA path, strength, and key remapping operations."""

    path: str
    strength: float
    sd_ops: SDOps


class LoraStateDictWithStrength(NamedTuple):
    """LoRA state dict paired with its strength multiplier."""

    state_dict: StateDict
    strength: float
