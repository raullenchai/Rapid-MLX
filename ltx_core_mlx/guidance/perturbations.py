"""STG (Spatio-Temporal Guidance) perturbation system.

Ported from ltx-core/src/ltx_core/guidance/perturbations.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mlx.core as mx


class PerturbationType(Enum):
    """Types of attention perturbations for STG."""

    SKIP_A2V_CROSS_ATTN = "skip_a2v_cross_attn"
    SKIP_V2A_CROSS_ATTN = "skip_v2a_cross_attn"
    SKIP_VIDEO_SELF_ATTN = "skip_video_self_attn"
    SKIP_AUDIO_SELF_ATTN = "skip_audio_self_attn"


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation specifying which attention type to skip and in which blocks."""

    type: PerturbationType
    blocks: list[int] | None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False
        if self.blocks is None:
            return True
        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration holding a list of perturbations for a single sample."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False
        return any(p.is_perturbed(perturbation_type, block) for p in self.perturbations)

    @staticmethod
    def empty() -> PerturbationConfig:
        return PerturbationConfig([])


@dataclass(frozen=True)
class BatchedPerturbationConfig:
    """Perturbation configurations for a batch, with utilities for generating attention masks."""

    perturbations: list[PerturbationConfig]

    def mask(self, perturbation_type: PerturbationType, block: int) -> mx.array:
        """Generate a batch mask: 1.0 for unperturbed, 0.0 for perturbed samples."""
        values = [0.0 if p.is_perturbed(perturbation_type, block) else 1.0 for p in self.perturbations]
        return mx.array(values, dtype=mx.bfloat16)

    def mask_like(self, perturbation_type: PerturbationType, block: int, values: mx.array) -> mx.array:
        """Generate a mask broadcastable to values shape."""
        m = self.mask(perturbation_type, block)
        # Reshape to (B, 1, 1, ...) for broadcasting
        shape = [m.shape[0]] + [1] * (values.ndim - 1)
        return m.reshape(shape)

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return any(p.is_perturbed(perturbation_type, block) for p in self.perturbations)

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return all(p.is_perturbed(perturbation_type, block) for p in self.perturbations)

    @staticmethod
    def empty(batch_size: int) -> BatchedPerturbationConfig:
        return BatchedPerturbationConfig([PerturbationConfig.empty() for _ in range(batch_size)])
