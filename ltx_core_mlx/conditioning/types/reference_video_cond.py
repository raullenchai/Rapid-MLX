"""VideoConditionByReferenceLatent — IC-LoRA style reference conditioning.

Ported from ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState


class VideoConditionByReferenceLatent:
    """Condition generation by appending reference latent with scaled positions.

    Used for IC-LoRA: a reference image/video latent is appended to the sequence.
    The downscale_factor scales spatial positions to match target coordinate space
    (e.g., 2 = half-resolution reference for full-resolution target).

    Args:
        reference_latent: Reference latent tokens, (B, Nr, C).
        reference_positions: Positions for reference tokens, (B, Nr, num_axes).
        downscale_factor: Target/reference resolution ratio. Spatial positions
            (height, width) are multiplied by this. Default 1 (no scaling).
        strength: Conditioning strength. 1.0 = preserved, 0.0 = denoised.
    """

    def __init__(
        self,
        reference_latent: mx.array,
        reference_positions: mx.array | None = None,
        downscale_factor: int = 1,
        strength: float = 1.0,
    ):
        self.reference_latent = reference_latent
        self.reference_positions = reference_positions
        self.downscale_factor = downscale_factor
        self.strength = strength

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Apply reference conditioning by appending tokens."""
        num_ref = self.reference_latent.shape[1]
        mask_value = 1.0 - self.strength

        new_latent = mx.concatenate([state.latent, self.reference_latent], axis=1)
        new_clean = mx.concatenate([state.clean_latent, self.reference_latent], axis=1)

        ref_mask = mx.full((state.denoise_mask.shape[0], num_ref, 1), mask_value)
        new_mask = mx.concatenate([state.denoise_mask, ref_mask], axis=1)

        # Extend positions with optional spatial scaling
        new_positions = state.positions
        if state.positions is not None and self.reference_positions is not None:
            ref_pos = self.reference_positions
            if self.downscale_factor != 1:
                # Scale spatial axes only (height=axis 1, width=axis 2), not temporal (axis 0)
                scale = mx.array([1.0, float(self.downscale_factor), float(self.downscale_factor)])
                ref_pos = ref_pos * scale[None, None, :]
            new_positions = mx.concatenate([state.positions, ref_pos], axis=1)

        # Build attention mask — num_noisy_tokens must be the ORIGINAL target
        # token count (F*H*W), not state.latent.shape[1] which may include
        # previously appended conditioning tokens.
        F, H, W = spatial_dims
        num_noisy = F * H * W
        new_attn_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_noisy,
            num_new_tokens=num_ref,
            batch_size=state.latent.shape[0],
        )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=new_positions,
            attention_mask=new_attn_mask,
        )
