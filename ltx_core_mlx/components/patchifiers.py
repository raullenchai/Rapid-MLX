"""Patchification utilities for video and audio latents.

Ported from ltx-core/src/ltx_core/components/patchifiers.py
"""

from __future__ import annotations

import mlx.core as mx


class VideoLatentPatchifier:
    """Converts between spatial video latents and flat patch sequences.

    Video VAE produces latents of shape (B, C, F, H, W).
    The patchifier reshapes them to (B, F*H*W, C) for transformer input,
    and back again for decoding.

    Args:
        patch_size_t: Temporal patch size (default 1).
        patch_size_h: Height patch size (default 1).
        patch_size_w: Width patch size (default 1).
    """

    def __init__(self, patch_size_t: int = 1, patch_size_h: int = 1, patch_size_w: int = 1):
        self.patch_size_t = patch_size_t
        self.patch_size_h = patch_size_h
        self.patch_size_w = patch_size_w

    def patchify(self, latent: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
        """Convert (B, C, F, H, W) -> (B, N, C) and return spatial dims.

        Args:
            latent: Video latent of shape (B, C, F, H, W).

        Returns:
            Tuple of (tokens, (F, H, W)) where tokens is (B, F*H*W, C).
        """
        B, C, F, H, W = latent.shape
        # (B, C, F, H, W) -> (B, F, H, W, C) -> (B, F*H*W, C)
        tokens = latent.transpose(0, 2, 3, 4, 1).reshape(B, F * H * W, C)
        return tokens, (F, H, W)

    def unpatchify(self, tokens: mx.array, spatial_dims: tuple[int, int, int]) -> mx.array:
        """Convert (B, N, C) -> (B, C, F, H, W).

        Args:
            tokens: Flat tokens of shape (B, F*H*W, C).
            spatial_dims: Tuple of (F, H, W).

        Returns:
            Video latent of shape (B, C, F, H, W).
        """
        F, H, W = spatial_dims
        B, _N, C = tokens.shape
        # (B, F*H*W, C) -> (B, F, H, W, C) -> (B, C, F, H, W)
        return tokens.reshape(B, F, H, W, C).transpose(0, 4, 1, 2, 3)


class AudioPatchifier:
    """Converts between audio latents and flat patch sequences.

    Audio latents have shape (B, 8, T, 16) from the audio VAE.
    Flattened to (B, T, 128) for transformer.
    """

    def patchify(self, latent: mx.array) -> tuple[mx.array, int]:
        """Convert (B, 8, T, 16) -> (B, T, 128).

        Args:
            latent: Audio latent of shape (B, 8, T, 16).

        Returns:
            Tuple of (tokens, T) where tokens is (B, T, 128).
        """
        B, C1, T, C2 = latent.shape
        # (B, 8, T, 16) -> (B, T, 8, 16) -> (B, T, 128)
        tokens = latent.transpose(0, 2, 1, 3).reshape(B, T, C1 * C2)
        return tokens, T

    def unpatchify(self, tokens: mx.array, _time_dim: int | None = None) -> mx.array:
        """Convert (B, T, 128) -> (B, 8, T, 16).

        Args:
            tokens: Flat tokens of shape (B, T, 128).
            _time_dim: Unused (kept for API symmetry).

        Returns:
            Audio latent of shape (B, 8, T, 16).
        """
        B, T, _C = tokens.shape
        # (B, T, 128) -> (B, T, 8, 16) -> (B, 8, T, 16)
        return tokens.reshape(B, T, 8, 16).transpose(0, 2, 1, 3)


def compute_video_latent_shape(
    num_frames: int,
    height: int,
    width: int,
    temporal_compression: int = 8,
    spatial_compression: int = 32,
) -> tuple[int, int, int]:
    """Compute the latent spatial dimensions after VAE encoding.

    Args:
        num_frames: Number of video frames.
        height: Video height in pixels.
        width: Video width in pixels.
        temporal_compression: VAE temporal compression factor.
        spatial_compression: VAE spatial compression factor.

    Returns:
        Tuple of (F', H', W') latent dimensions.
    """
    F = (num_frames + temporal_compression - 1) // temporal_compression
    H = height // spatial_compression
    W = width // spatial_compression
    return F, H, W
