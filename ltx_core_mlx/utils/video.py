"""Video loading utilities using ffmpeg subprocess."""

from __future__ import annotations

import subprocess

import mlx.core as mx
import numpy as np

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg


def load_video_frames_normalized(
    path: str,
    height: int,
    width: int,
    max_frames: int,
    fps: float | None = None,
) -> mx.array:
    """Load video frames from a file using ffmpeg.

    Decodes the video, rescales to (height, width), and returns normalised
    frames in [0, 1] as a (1, 3, F, H, W) bfloat16 tensor.

    For VAE encoding ([-1, 1] range), use :func:`load_video_for_encoding` instead.

    Args:
        path: Path to the video file.
        height: Target frame height.
        width: Target frame width.
        max_frames: Maximum number of frames to decode.
        fps: If given, resample the video to this frame rate before extracting
            frames. When ``None`` the native frame rate is used.

    Returns:
        mx.array of shape (1, 3, F, H, W) in [0, 1] range, bfloat16.

    Raises:
        RuntimeError: If ffmpeg decoding fails.
    """
    ffmpeg = find_ffmpeg()

    # Build the video filter chain.
    vf_parts: list[str] = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    vf_parts.append(f"scale={width}:{height}")
    vf_filter = ",".join(vf_parts)

    cmd = [
        ffmpeg,
        "-i",
        path,
        "-vf",
        vf_filter,
        "-frames:v",
        str(max_frames),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decoding failed: {result.stderr.decode()}")

    raw = result.stdout
    frame_bytes = height * width * 3
    num_frames = len(raw) // frame_bytes
    if num_frames == 0:
        raise RuntimeError(f"No frames decoded from {path}")

    # (F, H, W, 3) uint8 -> float32 [0, 1]
    arr = np.frombuffer(raw[: num_frames * frame_bytes], dtype=np.uint8)
    arr = arr.reshape(num_frames, height, width, 3).astype(np.float32) / 255.0

    # (F, H, W, 3) -> (F, 3, H, W) -> (1, 3, F, H, W)
    tensor = mx.array(arr).transpose(0, 3, 1, 2)  # (F, 3, H, W)
    tensor = tensor.transpose(1, 0, 2, 3)[None, ...]  # (1, 3, F, H, W)
    return tensor.astype(mx.bfloat16)


def load_video_for_encoding(
    path: str,
    height: int,
    width: int,
    max_frames: int,
) -> mx.array:
    """Load video frames normalised to [-1, 1] for VAE encoding.

    This is a convenience wrapper around :func:`load_video_frames` that
    rescales the pixel values from [0, 1] to [-1, 1].

    Args:
        path: Path to the video file.
        height: Target frame height.
        width: Target frame width.
        max_frames: Maximum number of frames to decode.

    Returns:
        mx.array of shape (1, 3, F, H, W) in [-1, 1] range, bfloat16.
    """
    frames = load_video_frames_normalized(path, height, width, max_frames)
    return (frames * 2.0 - 1.0).astype(mx.bfloat16)
