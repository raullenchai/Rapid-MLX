"""AudioConditionByReferenceLatent — audio reference conditioning (lipdub).

Ported from upstream ``ltx_core/conditioning/types/reference_audio_cond.py``.
Audio-side mirror of :class:`VideoConditionByReferenceLatent`: appends
reference audio tokens after the target audio sequence so they stay clean
while the target audio is denoised.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState


class AudioConditionByReferenceLatent:
    """Append patchified reference audio tokens after the target audio sequence.

    Mirrors :class:`ltx_core_mlx.conditioning.types.reference_video_cond.VideoConditionByReferenceLatent`
    but for the audio modality. Reference tokens stay clean (mask=0) so the target
    audio tokens in positions ``[0, num_noisy_tokens)`` get denoised normally.

    Args:
        patchified: Patchified reference audio latent ``(B, T_ref, C)``.
        positions: RoPE positions for reference tokens. Upstream uses
            ``(B, 1, T_ref, 2)`` (interval-style); we use ``(B, T_ref, num_axes)``
            (point-style) to match the rest of the MLX port.
        strength: ``1.0`` keeps reference clean; ``0.0`` would fully denoise it.
    """

    def __init__(
        self,
        patchified: mx.array,
        positions: mx.array,
        strength: float = 1.0,
    ) -> None:
        self.patchified = patchified
        self.positions = positions.astype(mx.float32)
        self.strength = strength

    def apply(self, state: LatentState, num_noisy_tokens: int) -> LatentState:
        """Apply by appending reference tokens.

        Args:
            state: Current audio latent state.
            num_noisy_tokens: Token count of the original target audio sequence
                (before any prior appends). Used to build the attention mask.

        Returns:
            Updated LatentState with reference tokens appended.
        """
        tokens = self.patchified
        num_ref = tokens.shape[1]
        mask_value = 1.0 - self.strength

        new_latent = mx.concatenate([state.latent, tokens], axis=1)
        new_clean = mx.concatenate([state.clean_latent, tokens], axis=1)

        ref_mask = mx.full((state.denoise_mask.shape[0], num_ref, 1), mask_value)
        new_mask = mx.concatenate([state.denoise_mask, ref_mask], axis=1)

        new_positions = state.positions
        if state.positions is not None:
            new_positions = mx.concatenate([state.positions, self.positions], axis=1)

        new_attn_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_noisy_tokens,
            num_new_tokens=num_ref,
            batch_size=tokens.shape[0],
        )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=new_positions,
            attention_mask=new_attn_mask,
        )


__all__ = ["AudioConditionByReferenceLatent"]
