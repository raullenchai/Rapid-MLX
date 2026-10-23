"""Wrapper conditioning item that adds attention masking to any inner conditioning.

Ported from ltx-core/src/ltx_core/conditioning/types/attention_strength_wrapper.py
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState


class ConditioningItemAttentionStrengthWrapper:
    """Wraps a conditioning item to add an attention mask for its tokens.

    Separates the attention-masking concern from the underlying conditioning
    logic (token layout, positional encoding, denoise strength). The inner
    conditioning item appends tokens to the latent sequence as usual, and this
    wrapper then builds or updates the self-attention mask so that the newly
    added tokens interact with the noisy tokens according to attention_mask.

    Args:
        conditioning: Any conditioning item with an apply() method.
        attention_mask: Per-token attention weight controlling how strongly the
            new conditioning tokens attend to/from noisy tokens. Can be a
            scalar (float) applied uniformly, or a tensor of shape (B, M)
            for spatial control. Values in [0, 1].
    """

    def __init__(
        self,
        conditioning: object,
        attention_mask: float | mx.array,
    ):
        self.conditioning = conditioning
        self.attention_mask = attention_mask

    def apply(
        self,
        state: LatentState,
        spatial_dims: tuple[int, int, int],
    ) -> LatentState:
        """Apply inner conditioning, then build the attention mask for its tokens."""
        original_state = state

        # Inner conditioning appends tokens
        new_state = self.conditioning.apply(state, spatial_dims)

        num_new_tokens = new_state.latent.shape[1] - original_state.latent.shape[1]
        if num_new_tokens == 0:
            return new_state

        # Convert scalar attention_mask to the format expected by update_attention_mask
        attn_mask = self.attention_mask

        # Build the attention mask — num_noisy_tokens must be the ORIGINAL
        # target token count (F*H*W), not original_state.latent.shape[1]
        # which may include previously appended conditioning tokens.
        F, H, W = spatial_dims
        num_noisy = F * H * W
        new_attention_mask = update_attention_mask(
            latent_state=original_state,
            attention_mask=attn_mask,
            num_noisy_tokens=num_noisy,
            num_new_tokens=num_new_tokens,
            batch_size=new_state.latent.shape[0],
        )

        return LatentState(
            latent=new_state.latent,
            clean_latent=new_state.clean_latent,
            denoise_mask=new_state.denoise_mask,
            positions=new_state.positions,
            attention_mask=new_attention_mask,
        )
