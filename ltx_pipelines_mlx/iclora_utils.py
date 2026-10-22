"""Shared IC-LoRA helpers: LoRA metadata, mask downsampling, reference-video conditioning.

Mirrors upstream ``ltx_pipelines.iclora_utils``: used by ``ic_lora`` and ``lipdub``
(video reference path only). LipDub audio helpers live in ``lipdub.py``.

API divergence vs upstream: ``downsample_mask_video_to_latent`` and
``append_ic_lora_reference_video_conditionings`` take latent dims as
``(F, H, W)`` tuples instead of ``VideoLatentShape`` (which we don't model
as a dataclass). Same math.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.positions import compute_video_positions
from ltx_core_mlx.utils.video import load_video_frames_normalized

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009 -- security hook flags mx.eval pattern


def read_lora_reference_downscale_factor(lora_path: str) -> int:
    """Read ``reference_downscale_factor`` from LoRA safetensors metadata (default 1)."""
    try:
        with safe_open(lora_path, framework="numpy") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logger.warning("Failed to read metadata from LoRA file '%s': %s", lora_path, e)
        return 1


def downsample_mask_video_to_latent(
    mask: mx.array,
    target_f: int,
    target_h: int,
    target_w: int,
) -> mx.array:
    """Downsample a pixel-space mask video to flattened latent token weights.

    Causal temporal downsampling: first frame is kept separately (temporal
    scale factor = 1 for that frame), remaining frames are area-averaged by
    the VAE's temporal scale factor.

    Args:
        mask: Pixel-space mask ``(B, 1, F_pixel, H_pixel, W_pixel)`` in ``[0, 1]``.
        target_f: Target latent temporal dim.
        target_h: Target latent spatial height.
        target_w: Target latent spatial width.

    Returns:
        Flattened latent-space mask ``(B, F_lat * H_lat * W_lat)``.
    """
    mask_np = np.array(mask)
    b, _, f_pix, _h_pix, _w_pix = mask_np.shape

    from PIL import Image as PILImage

    spatial_down = np.zeros((b, 1, f_pix, target_h, target_w), dtype=np.float32)
    for bi in range(b):
        for fi in range(f_pix):
            frame = mask_np[bi, 0, fi]
            img = PILImage.fromarray((frame * 255).astype(np.uint8))
            img = img.resize((target_w, target_h), PILImage.Resampling.BOX)
            spatial_down[bi, 0, fi] = np.array(img).astype(np.float32) / 255.0

    first_frame = spatial_down[:, :, :1, :, :]

    if f_pix > 1 and target_f > 1:
        t = (f_pix - 1) // (target_f - 1)
        assert (f_pix - 1) % (target_f - 1) == 0, (
            f"Pixel frames ({f_pix}) not compatible with latent frames ({target_f}): "
            f"(f_pix - 1) must be divisible by (target_f - 1)"
        )
        rest = spatial_down[:, :, 1:, :, :]
        rest = rest.reshape(b, 1, target_f - 1, t, target_h, target_w)
        rest = rest.mean(axis=3)
        latent_mask = np.concatenate([first_frame, rest], axis=2)
    else:
        latent_mask = first_frame

    latent_mask = latent_mask.reshape(b, target_f * target_h * target_w)
    return mx.array(latent_mask)


def append_ic_lora_reference_video_conditionings(
    conditionings: list,
    video_conditioning: list[tuple[str, float]],
    *,
    height: int,
    width: int,
    num_frames: int,
    video_encoder,
    reference_downscale_factor: int,
    conditioning_attention_strength: float = 1.0,
    conditioning_attention_mask: mx.array | None = None,
) -> None:
    """Append :class:`VideoConditionByReferenceLatent` items for each reference path.

    Mirrors upstream ``ltx_pipelines.iclora_utils.append_ic_lora_reference_video_conditionings``,
    minus the ``tiling_config`` arg (our VAE encoder doesn't expose a tiled-encode
    entry point; tiled video VAE happens at the decoder side via streaming).
    """
    scale = reference_downscale_factor
    if scale != 1 and (height % scale != 0 or width % scale != 0):
        raise ValueError(
            f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
        )

    _, ref_H_lat, ref_W_lat = compute_video_latent_shape(num_frames, height // scale, width // scale)
    ref_height = ref_H_lat * 32
    ref_width = ref_W_lat * 32

    for video_path, strength in video_conditioning:
        # The video VAE encoder requires a (1 + 8k)-frame input. Source files
        # produced by LTX itself are saved 8k-trimmed (``_decode_and_save_video``
        # drops the leading frame), so feeding them through unchanged fails the
        # encoder's ``space_to_depth`` reshape. Probe the source, clamp to the
        # caller's target, and round down to the nearest (1 + 8k). Mirrors
        # ``RetakePipeline._encode_source_video``.
        info = probe_video_info(video_path)
        max_frames = min(num_frames, info.num_frames)
        k = max(1, (max_frames - 1) // 8)
        vae_compatible_frames = 1 + k * 8
        video = load_video_frames_normalized(video_path, ref_height, ref_width, vae_compatible_frames)
        video = (video * 2.0 - 1.0).astype(mx.bfloat16)
        encoded_video = video_encoder.encode(video)
        _mx_eval(encoded_video)

        ref_F = encoded_video.shape[2]
        ref_H = encoded_video.shape[3]
        ref_W = encoded_video.shape[4]
        ref_positions = compute_video_positions(ref_F, ref_H, ref_W)
        ref_tokens = encoded_video.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)

        if conditioning_attention_mask is not None:
            latent_mask = downsample_mask_video_to_latent(
                mask=conditioning_attention_mask,
                target_f=ref_F,
                target_h=ref_H,
                target_w=ref_W,
            )
            attn_mask = latent_mask * conditioning_attention_strength
        elif conditioning_attention_strength < 1.0:
            attn_mask = conditioning_attention_strength
        else:
            attn_mask = None

        cond = VideoConditionByReferenceLatent(
            reference_latent=ref_tokens,
            reference_positions=ref_positions,
            downscale_factor=scale,
            strength=strength,
        )
        if attn_mask is not None:
            cond = ConditioningItemAttentionStrengthWrapper(
                conditioning=cond,
                attention_mask=attn_mask,
            )
        conditionings.append(cond)


__all__ = [
    "append_ic_lora_reference_video_conditionings",
    "downsample_mask_video_to_latent",
    "read_lora_reference_downscale_factor",
]
