"""VideoConditionByKeyframeIndex — appended tokens with attention mask.

Ported from ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py

Each keyframe is a single conditioning item with a single frame_idx.
Multiple keyframes are applied sequentially, each appending its tokens
and incrementally building the attention mask.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.utils.positions import VIDEO_SPATIAL_SCALE, VIDEO_TEMPORAL_SCALE


def _compute_keyframe_positions(
    frame_idx: int,
    height: int,
    width: int,
    frame_rate: float,
    num_pixel_frames: int = 1,
) -> mx.array:
    """Compute positions for a single keyframe, matching the reference.

    The reference computes positions for a single-frame (1, H, W) shape,
    then offsets the temporal axis by frame_idx BEFORE dividing by frame_rate.
    Causal fix is only applied for frame_idx == 0.

    This differs from extracting positions from the full video grid because
    non-zero keyframes intentionally omit the causal fix on their single-
    frame positions, producing an offset temporal coordinate that the model
    was trained to interpret as conditioning reference positions.

    Args:
        frame_idx: Pixel frame index (0-based). E.g., for a 97-frame video
            the last frame is 96. NOT a latent frame index.
        height: Latent height H.
        width: Latent width W.
        frame_rate: Frame rate.
        num_pixel_frames: Number of pixel frames the keyframe latent encodes.
            For single-frame keyframes (default), the temporal range is
            narrowed to [start, start+1) instead of the VAE-scaled width.
            Matches Lightricks/LTX-2 PR #192 (commit a2c3f24): without this,
            non-zero keyframes occupy a temporal range 8x wider than they
            represent, which leaves a gap between the last generated latent
            frame and the end keyframe.

    Returns:
        Positions (1, H*W, 3) float32.
    """
    # Temporal: compute pixel coords for a single frame, then offset by frame_idx.
    # Reference: get_pixel_coords on single-frame shape, then += frame_idx, then /= frame_rate.
    if frame_idx == 0:
        # With causal fix: pixel range [max(0, 0*8+1-8), 0*8+1] = [0, 1)
        t_start = 0.0
        t_end = 1.0
    else:
        # Without causal fix: pixel range [0*8, 1*8) = [0, 8)
        t_start = 0.0
        t_end = float(VIDEO_TEMPORAL_SCALE)

    # Single-frame keyframes occupy exactly 1 pixel-frame of temporal width.
    if num_pixel_frames == 1:
        t_end = t_start + 1.0

    t_mid = (t_start + t_end) / 2.0
    # Add pixel frame_idx offset then divide by frame_rate
    t_mid = (t_mid + frame_idx) / frame_rate

    # Spatial: same as regular positions — pixel midpoints
    h_mids = mx.arange(height).astype(mx.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0
    w_mids = mx.arange(width).astype(mx.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0

    # Build 2D grid for single frame: (H, W, 3)
    h_grid = mx.repeat(h_mids[:, None], width, axis=1)
    w_grid = mx.repeat(w_mids[None, :], height, axis=0)
    t_grid = mx.full((height, width), t_mid)

    positions = mx.stack([t_grid, h_grid, w_grid], axis=-1).reshape(1, -1, 3)
    return positions.astype(mx.float32)


class VideoConditionByKeyframeIndex:
    """Condition generation by appending a single keyframe's tokens.

    Matches the reference: one instance per keyframe, applied sequentially.
    Each call to apply() appends this keyframe's tokens and extends the
    attention mask so that the new tokens attend to noisy tokens but not
    to tokens from previously appended keyframes.

    Args:
        frame_idx: Pixel frame index for this keyframe (0-based). E.g. for a
            97-frame video the last frame is 96. NOT a latent frame index.
        keyframe_latent: Clean latent for this keyframe, (B, H*W, C).
        spatial_dims: (F, H, W) latent spatial dimensions.
        frame_rate: Frame rate for position computation.
        strength: Conditioning strength. 1.0 = fully preserved.
        num_pixel_frames: Number of pixel frames the keyframe latent encodes.
            Defaults to 1 (the typical case). See ``_compute_keyframe_positions``.
    """

    def __init__(
        self,
        frame_idx: int,
        keyframe_latent: mx.array,
        spatial_dims: tuple[int, int, int],
        frame_rate: float,
        strength: float = 1.0,
        num_pixel_frames: int = 1,
    ):
        self.frame_idx = frame_idx
        self.keyframe_latent = keyframe_latent
        self.strength = strength

        # Compute positions matching reference: single-frame positions with
        # frame_idx offset, NOT extracted from the full video grid.
        _, H, W = spatial_dims
        self.keyframe_positions = _compute_keyframe_positions(
            frame_idx, H, W, frame_rate, num_pixel_frames=num_pixel_frames
        )

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Apply keyframe conditioning by appending tokens.

        Args:
            state: Current latent state (may already have prior keyframes appended).
            spatial_dims: (F, H, W) — used for num_noisy_tokens computation.

        Returns:
            Updated LatentState with this keyframe's tokens appended.
        """
        F, H, W = spatial_dims
        num_kf = self.keyframe_latent.shape[1]
        mask_value = 1.0 - self.strength

        new_latent = mx.concatenate([state.latent, self.keyframe_latent], axis=1)
        new_clean = mx.concatenate([state.clean_latent, self.keyframe_latent], axis=1)

        kf_mask = mx.full((state.denoise_mask.shape[0], num_kf, 1), mask_value)
        new_mask = mx.concatenate([state.denoise_mask, kf_mask], axis=1)

        # Extend positions
        new_positions = state.positions
        if state.positions is not None:
            new_positions = mx.concatenate([state.positions, self.keyframe_positions], axis=1)

        # Build attention mask incrementally — num_noisy_tokens is always
        # the original generation token count, not the current sequence length
        num_noisy = F * H * W
        new_attn_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_noisy,
            num_new_tokens=num_kf,
            batch_size=state.latent.shape[0],
        )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=new_positions,
            attention_mask=new_attn_mask,
        )
