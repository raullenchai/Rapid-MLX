"""Core latent conditioning — LatentState, masks, and noise management.

Ported from ltx-core/src/ltx_core/conditioning/latent.py
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class LatentState:
    """Generation state for diffusion.

    Attributes:
        latent: Noisy latent being denoised, (B, N, C).
        clean_latent: Original clean latent for conditioning, (B, N, C).
        denoise_mask: Per-token mask: 1.0 = denoise (generate), 0.0 = preserve.
        positions: Positional indices (B, N, num_axes) or None.
        attention_mask: Self-attention mask (B, N, N) with values in [0,1], or None.
    """

    latent: mx.array
    clean_latent: mx.array
    denoise_mask: mx.array
    positions: mx.array | None = None
    attention_mask: mx.array | None = None


class VideoConditionByLatentIndex:
    """Replace tokens at specific frame indices with conditioning latent.

    Used for Image-to-Video: the first frame's latent tokens are
    set as clean (preserved) while the rest are generated.

    Args:
        frame_indices: List of frame indices to condition.
        clean_latent: Clean latent tokens for those frames, (B, num_indices * tokens_per_frame, C).
        strength: Conditioning strength. 1.0 = fully preserved (mask=0),
            0.0 = fully denoised (mask=1). Default 1.0.
    """

    def __init__(
        self,
        frame_indices: list[int],
        clean_latent: mx.array,
        strength: float = 1.0,
    ):
        self.frame_indices = frame_indices
        self.clean_latent = clean_latent
        self.strength = strength

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        """Apply conditioning: replace latent, clean_latent, and mask at frame indices."""
        _F, H, W = spatial_dims
        tokens_per_frame = H * W

        new_latent = state.latent
        new_clean = state.clean_latent
        new_mask = state.denoise_mask
        mask_value = 1.0 - self.strength

        for i, frame_idx in enumerate(self.frame_indices):
            start = frame_idx * tokens_per_frame
            end = start + tokens_per_frame
            src_start = i * tokens_per_frame
            src_end = src_start + tokens_per_frame

            if src_end > self.clean_latent.shape[1]:
                break

            frame_tokens = self.clean_latent[:, src_start:src_end, :]

            # Update latent, clean_latent, and mask together
            new_latent = mx.concatenate([new_latent[:, :start, :], frame_tokens, new_latent[:, end:, :]], axis=1)
            new_clean = mx.concatenate([new_clean[:, :start, :], frame_tokens, new_clean[:, end:, :]], axis=1)
            frame_mask = mx.full((state.denoise_mask.shape[0], tokens_per_frame, 1), mask_value)
            new_mask = mx.concatenate([new_mask[:, :start, :], frame_mask, new_mask[:, end:, :]], axis=1)

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=state.positions,
            attention_mask=state.attention_mask,
        )


class TemporalRegionMask:
    """Create a denoise mask for a specific time region.

    Used for Retake: marks a time segment for regeneration while
    preserving the rest.

    Args:
        start_frame: First frame to regenerate (inclusive).
        end_frame: Last frame to regenerate (exclusive).
    """

    def __init__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def create_mask(self, num_frames: int, tokens_per_frame: int) -> mx.array:
        """Create denoise mask.

        Args:
            num_frames: Total number of frames.
            tokens_per_frame: H * W tokens per frame.

        Returns:
            Mask of shape (1, num_frames * tokens_per_frame, 1).
        """
        mask = mx.zeros((1, num_frames * tokens_per_frame, 1))
        start = self.start_frame * tokens_per_frame
        end = min(self.end_frame * tokens_per_frame, num_frames * tokens_per_frame)
        # Set region to denoise
        mask = mask.at[:, start:end, :].add(mx.ones((1, end - start, 1)))
        return mask


def create_initial_state(
    shape: tuple[int, ...],
    seed: int,
    clean_latent: mx.array | None = None,
    positions: mx.array | None = None,
) -> LatentState:
    """Create initial latent state with pure noise.

    .. note::
        Pipelines should prefer
        :func:`ltx_pipelines_mlx.utils.helpers.create_noised_state`
        with ``initial_latent=None`` and ``legacy_scalar_blend=True`` —
        it produces a bit-equivalent state and aligns with the upstream
        ``init → cond → noise`` orchestration. This primitive is kept
        for low-level testing and for ``retake`` / ``extend`` pipelines
        that bypass the canonical helper.

    Args:
        shape: (B, N, C) shape for the latent.
        seed: Random seed.
        clean_latent: Optional clean latent for conditioning.

    Returns:
        Initial LatentState with noise and full denoise mask.
    """
    mx.random.seed(seed)
    noise = mx.random.normal(shape).astype(mx.bfloat16)

    if clean_latent is None:
        clean_latent = mx.zeros(shape, dtype=mx.bfloat16)

    denoise_mask = mx.ones((shape[0], shape[1], 1), dtype=mx.bfloat16)

    return LatentState(latent=noise, clean_latent=clean_latent, denoise_mask=denoise_mask, positions=positions)


def apply_conditioning(
    state: LatentState,
    conditions: list,
    spatial_dims: tuple[int, int, int],
) -> LatentState:
    """Apply a list of conditioning items to the state.

    .. note::
        Pipelines should prefer
        :func:`ltx_pipelines_mlx.utils.helpers.state_with_conditionings`,
        which is identical in semantics and lives next to the rest of
        the upstream-isomorphic orchestration helpers. This primitive
        is kept for low-level testing.

    Args:
        state: Current latent state.
        conditions: List of conditioning items with .apply() method.
        spatial_dims: (F, H, W) spatial dimensions.

    Returns:
        Modified state.
    """
    for condition in conditions:
        state = condition.apply(state, spatial_dims)
    return state


def apply_denoise_mask(
    x0: mx.array,
    clean_latent: mx.array,
    denoise_mask: mx.array,
) -> mx.array:
    """Blend predicted x0 with clean latent using denoise mask.

    Where mask = 1.0 (generate): use x0 prediction.
    Where mask = 0.0 (preserve): use clean_latent.

    Args:
        x0: Predicted clean sample, (B, N, C).
        clean_latent: Known clean latent, (B, N, C).
        denoise_mask: Blend mask, (B, N, 1).

    Returns:
        Blended result.
    """
    return x0 * denoise_mask + clean_latent * (1.0 - denoise_mask)


def noise_latent_state(
    state: LatentState,
    sigma: float,
    seed: int,
) -> LatentState:
    """Add noise to a latent state respecting the denoise mask.

    Matches reference GaussianNoiser: interpolates between noise and clean
    based on denoise_mask * noise_scale (sigma).

    Preserved regions (mask=0): keep clean_latent.
    Generated regions (mask=1): latent = noise * sigma + clean * (1 - sigma).

    Args:
        state: Current latent state (with clean_latent and denoise_mask set).
        sigma: Noise scale (typically 1.0 for full noise, or start_sigma for stage 2).
        seed: Random seed for reproducible noise.

    Returns:
        New LatentState with properly noised latent.
    """
    mx.random.seed(seed)
    noise = mx.random.normal(state.clean_latent.shape).astype(state.clean_latent.dtype)
    scaled_mask = state.denoise_mask * sigma
    latent = noise * scaled_mask + state.clean_latent * (1.0 - scaled_mask)
    return LatentState(
        latent=latent,
        clean_latent=state.clean_latent,
        denoise_mask=state.denoise_mask,
        positions=state.positions,
        attention_mask=state.attention_mask,
    )


def add_noise_with_state(
    state: LatentState,
    sigma: mx.array,
) -> mx.array:
    """Add noise to state respecting denoise mask.

    Timesteps for preserved regions are set to 0 (no noise).

    Args:
        state: Current latent state.
        sigma: Noise level, (B,) or scalar.

    Returns:
        Noisy latent, (B, N, C).
    """
    # Effective sigma: 0 where preserved, sigma where generating
    if sigma.ndim == 0:
        sigma = sigma[None]
    effective_sigma = sigma[:, None, None] * state.denoise_mask

    # x_t = clean + sigma * noise_direction
    noise_direction = state.latent - state.clean_latent
    return state.clean_latent + effective_sigma * noise_direction
