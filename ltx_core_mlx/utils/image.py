"""Image and video preparation utilities for VAE encoding.

Image-side helpers (CRF round-trip, resize+crop, normalize) live upstream-iso
in :mod:`ltx_pipelines_mlx.utils.media_io`. The two thin shims here keep the
historic ``prepare_image_for_encoding`` import path working — they delegate
to ``media_io.load_image_and_preprocess`` so the actual logic isn't
duplicated. Pipelines and new code should import from
:mod:`ltx_pipelines_mlx.utils.media_io` directly to match upstream paths.

This module also keeps :func:`load_video_frames`, the multi-frame ffmpeg
loader for video conditioning. Audio-side decode is in
:mod:`ltx_core_mlx.utils.audio`; ffmpeg discovery in
:mod:`ltx_core_mlx.utils.ffmpeg`.
"""

from __future__ import annotations

import subprocess

import mlx.core as mx
import numpy as np
from PIL import Image

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


def prepare_image_for_encoding(
    image: Image.Image | str,
    height: int,
    width: int,
    crf: int = 33,
) -> mx.array:
    """Legacy alias for :func:`ltx_pipelines_mlx.utils.media_io.load_image_and_preprocess`.

    Kept as a thin import-stable shim. New call sites should import from
    ``ltx_pipelines_mlx.utils.media_io`` directly to match upstream's
    ``ltx_pipelines.utils.media_io.load_image_and_preprocess`` path.

    Behavior identical to the upstream pipeline:

    1. (str path) decode → uint8 RGB array.
       (PIL.Image input) bypass decode, use directly.
    2. H.264 round-trip at ``crf`` (default 33; pass ``crf=0`` to skip).
    3. Aspect-preserving resize + center crop to ``(height, width)``.
    4. Normalize ``[0, 1] → [-1, 1]``, ``HWC → BCHW``, bfloat16.

    Returns:
        mx.array of shape ``(1, 3, H, W)`` in ``[-1, 1]``, bfloat16.
    """
    # Imported lazily to avoid circular: media_io lives in ltx-pipelines-mlx.
    from ltx_pipelines_mlx.utils.media_io import (
        load_image_and_preprocess,
        preprocess,
        resize_and_center_crop,
    )

    if isinstance(image, str):
        return load_image_and_preprocess(image, height, width, crf=crf)

    # PIL.Image was passed directly — replicate the same pipeline manually
    # since load_image_and_preprocess takes a path. This branch is mostly
    # used by tests and a few internal call sites that already have a PIL
    # object in hand.
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    if crf and crf > 0:
        arr = preprocess(arr, crf=crf)
    cropped = resize_and_center_crop(arr, height, width)

    f = np.asarray(cropped, dtype=np.float32) / 255.0
    f = f * 2.0 - 1.0
    tensor = mx.array(f).transpose(2, 0, 1)[None, ...]
    return tensor.astype(mx.bfloat16)


def load_video_frames(
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> mx.array:
    """Load video frames via ffmpeg as a tensor for VAE encoding.

    Args:
        video_path: Path to the video file.
        height: Frame height in pixels.
        width: Frame width in pixels.
        num_frames: Number of frames to read.

    Returns:
        Video tensor of shape (1, 3, F, H, W) in [-1, 1] range, bfloat16.

    Raises:
        RuntimeError: If ffmpeg fails to read the video.
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-i",
        video_path,
        "-vframes",
        str(num_frames),
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to read video: {result.stderr.decode()}")

    raw = result.stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    # Normalize to [-1, 1]
    frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
    # FHWC -> BCFHW: (F, H, W, 3) -> (3, F, H, W) -> (1, 3, F, H, W)
    tensor = mx.array(frames).transpose(3, 0, 1, 2)[None, ...]
    return tensor.astype(mx.bfloat16)
