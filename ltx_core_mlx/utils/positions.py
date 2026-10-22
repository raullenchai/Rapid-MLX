"""Position computation for video and audio tokens.

Video positions are 3D (time, height, width) in pixel-space with causal fix, time/frame_rate.
Audio positions are 1D (time) in real-time seconds.
"""

from __future__ import annotations

import mlx.core as mx

# VAE scale factors
VIDEO_TEMPORAL_SCALE = 8
VIDEO_SPATIAL_SCALE = 32
AUDIO_DOWNSAMPLE_FACTOR = 4
AUDIO_HOP_LENGTH = 160
AUDIO_SAMPLE_RATE = 16000
AUDIO_LATENTS_PER_SECOND = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH / AUDIO_DOWNSAMPLE_FACTOR  # 25.0


def compute_audio_token_count(num_video_frames: int, frame_rate: float = 24.0) -> int:
    """Compute the number of audio latent tokens for a given video length.

    Args:
        num_video_frames: Number of pixel frames.
        frame_rate: Video frame rate.

    Returns:
        Number of audio tokens.
    """
    duration = num_video_frames / frame_rate
    return round(duration * AUDIO_LATENTS_PER_SECOND)


def compute_video_positions(
    num_frames: int,
    height: int,
    width: int,
    frame_rate: float = 24.0,
) -> mx.array:
    """Compute 3D positions for video tokens in pixel-space with causal fix.

    Matches reference: get_pixel_coords(causal_fix=True) -> midpoints -> temporal/frame_rate.

    Args:
        num_frames: Number of latent frames F.
        height: Latent height H.
        width: Latent width W.
        frame_rate: Video frame rate (default 24.0).

    Returns:
        positions: (1, F*H*W, 3) float32 positions [time, height, width].
    """
    # Temporal: pixel coords with causal fix
    # latent frame i -> pixel range [max(0, i*8 + 1 - 8), (i+1)*8 + 1 - 8]
    #                             = [max(0, (i-1)*8 + 1), i*8 + 1]
    # Midpoint = (start + end) / 2, then divide by frame_rate
    idx = mx.arange(num_frames).astype(mx.float32)
    f_starts = mx.maximum(idx * VIDEO_TEMPORAL_SCALE + 1 - VIDEO_TEMPORAL_SCALE, 0.0)
    f_ends = mx.maximum((idx + 1) * VIDEO_TEMPORAL_SCALE + 1 - VIDEO_TEMPORAL_SCALE, 0.0)
    f_mids = (f_starts + f_ends) / 2.0 / frame_rate

    # Spatial: simple pixel midpoints (no causal fix)
    # latent h -> pixel range [h*32, (h+1)*32], midpoint = h*32 + 16
    h_mids = mx.arange(height).astype(mx.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0
    w_mids = mx.arange(width).astype(mx.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0

    # Build 3D meshgrid
    f_grid = mx.repeat(mx.repeat(f_mids[:, None, None], height, axis=1), width, axis=2)
    h_grid = mx.repeat(mx.repeat(h_mids[None, :, None], num_frames, axis=0), width, axis=2)
    w_grid = mx.repeat(mx.repeat(w_mids[None, None, :], num_frames, axis=0), height, axis=1)

    # Stack and flatten: (F, H, W, 3) -> (1, F*H*W, 3)
    positions = mx.stack([f_grid, h_grid, w_grid], axis=-1).reshape(-1, 3)
    return positions[None, :, :].astype(mx.float32)


def compute_audio_positions(
    num_tokens: int,
) -> mx.array:
    """Compute 1D positions for audio tokens in real-time seconds.

    Matches reference _compute_audio_timings with causal=True.

    Args:
        num_tokens: Number of audio tokens T.

    Returns:
        positions: (1, T, 1) float32 positions in seconds.
    """
    # Reference: _get_audio_latent_time_in_sec(idx, shift=0):
    #   audio_mel_frame = idx * downsample_factor
    #   causal: audio_mel_frame = (audio_mel_frame + 1 - downsample_factor).clip(min=0)
    #   time = audio_mel_frame * hop_length / sample_rate
    # For token i: start = _get(i), end = _get(i+1)
    # Midpoint = (start + end) / 2
    idx = mx.arange(num_tokens).astype(mx.float32)
    starts = (
        mx.maximum(idx * AUDIO_DOWNSAMPLE_FACTOR + 1 - AUDIO_DOWNSAMPLE_FACTOR, 0.0)
        * AUDIO_HOP_LENGTH
        / AUDIO_SAMPLE_RATE
    )
    ends = (
        mx.maximum((idx + 1) * AUDIO_DOWNSAMPLE_FACTOR + 1 - AUDIO_DOWNSAMPLE_FACTOR, 0.0)
        * AUDIO_HOP_LENGTH
        / AUDIO_SAMPLE_RATE
    )
    mids = (starts + ends) / 2.0

    return mids[None, :, None]


def compute_keyframes_mask(
    num_latent_frames: int,
    height: int,
    width: int,
    num_appended_tokens: int = 0,
    batch: int = 1,
) -> mx.array:
    """Build the LTX-2.5 keyframes mask marking the first latent frame's tokens.

    The keyframe absolute-position embedding is applied to video tokens whose
    latent encodes a *single standalone pixel frame* — the target's first
    latent frame (the video encoder is causal, so latent frame 0 covers pixel
    frame 0 only) plus any generated keyframe slots. Appended conditioning
    tokens (image-conditioning keyframes / references) are explicitly NOT
    marked, matching ComfyUI ``keyframes_abs_pos_mask`` (guide tokens are
    excluded) and ltx-core ``keyframes_mask`` semantics.

    Args:
        num_latent_frames: Number of latent frames F (the target's first
            latent frame, i.e. tokens ``[0, H*W)``, is marked).
        height: Latent height H.
        width: Latent width W.
        num_appended_tokens: Extra tokens appended after the generation
            tokens (keyframes / references); left unmarked.
        batch: Batch size.

    Returns:
        Mask ``(B, num_latent_frames * H * W + num_appended_tokens, 1)``
        of zeros/ones (1 marks single-pixel-frame tokens).
    """
    tokens_per_frame = height * width
    num_tokens = num_latent_frames * tokens_per_frame + num_appended_tokens
    mask = mx.zeros((batch, num_tokens, 1))
    mask[:, :tokens_per_frame, :] = 1.0
    return mask
