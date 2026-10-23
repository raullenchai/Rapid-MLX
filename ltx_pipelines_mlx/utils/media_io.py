"""Image / video / audio I/O — mirrors upstream ``ltx_pipelines.utils.media_io``.

Ports the public API surface of upstream's ``media_io.py`` so pipelines can
import the same names from the same path. The implementation uses ffmpeg
subprocess pipes (instead of upstream's PyAV) — same I/O behavior, no extra
runtime dependency.

Public names match upstream verbatim:

- ``DEFAULT_IMAGE_CRF`` — default H.264 CRF for I2V image preprocessing.
- ``decode_image`` — load an image file as ``numpy.ndarray`` (HWC, uint8).
- ``encode_single_frame`` — encode one RGB frame to H.264 mp4 bytes.
- ``decode_single_frame`` — decode the first frame of a buffer back to RGB.
- ``preprocess`` — round-trip an image through libx264 at a given CRF.
- ``resize_and_center_crop`` — aspect-preserving resize + center crop.
- ``to_vae_range`` / ``from_vae_range`` — ``[0, 1] ↔ [-1, 1]`` shifts.
- ``load_image_and_preprocess`` — full I2V image pipeline (decode → CRF
  round-trip → resize/crop → normalize → MLX tensor).

The legacy alias ``prepare_image_for_encoding`` (kept in
:mod:`ltx_core_mlx.utils.image`) now delegates here so call sites that
haven't migrated to the upstream-named API keep working.
"""

from __future__ import annotations

import math
import subprocess
from io import BytesIO
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from ltx_core_mlx.utils.ffmpeg import find_ffmpeg

# Upstream-verbatim default (LTX-2.3 port). Used by ``ImageConditioningInput.crf`` and by
# ``load_image_and_preprocess``. Round-tripping the input image through
# libx264 at this CRF brings it close to the LTX-2 training distribution
# (which is built from real video frames carrying H.264 compression
# artefacts), preventing the model from over-reacting to pristine
# PNG/JPEG textures during I2V conditioning.
DEFAULT_IMAGE_CRF = 33

# Official LTX-2.5 default. The 2.5 reference pipelines preprocess I2V
# images at CRF 18 (inheriting ``DEFAULT_IMAGE_CRF = 33`` from the 2.3 port
# over-compresses 2.5 inputs and hurts I2V quality).
DEFAULT_IMAGE_CRF_V25 = 18


def default_image_crf(model_version: str) -> int:
    """Return the I2V image CRF default for a model version.

    ``"2.5.0"`` (and any ``2.5*`` version) → 18 (official 2.5);
    anything else (the 2.3 port) → 33 (upstream-verbatim).
    """
    if model_version.startswith("2.5"):
        return DEFAULT_IMAGE_CRF_V25
    return DEFAULT_IMAGE_CRF


def to_vae_range(x: mx.array) -> mx.array:
    """Shift ``[0, 1]`` pixels to the VAE's expected ``[-1, 1]`` range."""
    return x * 2.0 - 1.0


def from_vae_range(z: mx.array) -> mx.array:
    """Inverse of :func:`to_vae_range`: ``[-1, 1]`` → ``[0, 1]``."""
    return (z + 1.0) / 2.0


def decode_image(image_path: str) -> np.ndarray:
    """Load an image file as an ``HxWx3`` uint8 ``np.ndarray`` (RGB).

    Mirrors upstream's signature; uses PIL under the hood (upstream uses
    cv2 / pyav). RGB-converts so the array is always ``HWC, uint8``.
    """
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def encode_single_frame(
    output_file: BytesIO | str,
    image_array: np.ndarray,
    crf: float,
) -> None:
    """Encode a single RGB frame to a 1-frame H.264 mp4.

    Mirrors upstream's PyAV-based implementation using an ffmpeg subprocess
    pipeline. Output goes to ``output_file`` (``BytesIO`` for in-memory or a
    path string for disk). Even-pixel padding is handled internally; the
    caller must crop back if the original dimensions were odd.

    Args:
        output_file: Destination — either a ``BytesIO`` (preferred, in-memory)
            or a filesystem path string.
        image_array: ``HxWx3`` uint8 array.
        crf: H.264 CRF (0 = lossless, higher = more compression). Upstream
            default is 33.
    """
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"encode_single_frame expects HxWx3 RGB, got {image_array.shape}")

    height, width, _ = image_array.shape
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    if (pad_w, pad_h) != (width, height):
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        padded[:height, :width, :] = image_array
        image_array = padded

    raw = image_array.tobytes()
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{pad_w}x{pad_h}",
        "-r",
        "1",
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-frames:v",
        "1",
    ]

    if isinstance(output_file, BytesIO):
        cmd += ["-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"]
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode_single_frame failed: {proc.stderr.decode(errors='ignore')}")
        output_file.write(proc.stdout)
    else:
        cmd += [str(output_file)]
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode_single_frame failed: {proc.stderr.decode(errors='ignore')}")


def decode_single_frame(video_file: BytesIO | str) -> np.ndarray:
    """Decode the first frame of a video buffer/file back to ``HxWx3`` RGB.

    Companion to :func:`encode_single_frame`. Returns the original odd-pixel
    dimensions are NOT inferred here — caller is responsible for cropping
    back if it padded for encoding.
    """
    ffmpeg = find_ffmpeg()
    if isinstance(video_file, BytesIO):
        in_data = video_file.getvalue()
        in_arg = "pipe:0"
    else:
        in_data = None
        in_arg = str(video_file)

    # Probe size by asking ffmpeg for raw RGB; we read frame data then deduce shape.
    # For pipe input we need to ask ffprobe-equivalent first; simpler: do a probe.
    if in_data is not None:
        # Probe via ffmpeg -i (writes metadata to stderr). Workaround: use
        # ffprobe through a temp file would be cleaner; here we trust the
        # caller and infer dimensions from the encoded raw buffer length
        # after a first decode pass.
        cmd_probe = [ffmpeg, "-i", in_arg, "-f", "null", "-"]
        probe = subprocess.run(cmd_probe, input=in_data, capture_output=True, timeout=30)
        # Parse "Stream ... NxM" from stderr.
        size = _parse_size_from_stderr(probe.stderr.decode(errors="ignore"))
    else:
        cmd_probe = [ffmpeg, "-i", in_arg, "-f", "null", "-"]
        probe = subprocess.run(cmd_probe, capture_output=True, timeout=30)
        size = _parse_size_from_stderr(probe.stderr.decode(errors="ignore"))

    if size is None:
        raise RuntimeError("decode_single_frame: could not determine frame size from ffmpeg probe")
    width, height = size

    cmd = [
        ffmpeg,
        "-i",
        in_arg,
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=in_data, capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode_single_frame failed: {proc.stderr.decode(errors='ignore')}")

    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    return arr.reshape(height, width, 3).copy()


def _parse_size_from_stderr(stderr: str) -> tuple[int, int] | None:
    """Extract (width, height) from ffmpeg's ``-i ...`` stderr summary."""
    import re

    # e.g. "Stream #0:0(und): Video: h264 ..., yuv420p, 1280x720, ..."
    m = re.search(r",\s*(\d{2,5})x(\d{2,5})", stderr)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def preprocess(image: np.ndarray, crf: float = DEFAULT_IMAGE_CRF) -> np.ndarray:
    """Round-trip an ``HxWx3`` uint8 RGB array through libx264 at the given CRF.

    Mirrors upstream verbatim: encode → decode → return decoded RGB.
    ``crf == 0`` is a passthrough.
    """
    if crf == 0:
        return image
    h, w, _ = image.shape
    pad_w = w + (w & 1)
    pad_h = h + (h & 1)

    with BytesIO() as buf:
        encode_single_frame(buf, image, crf)
        encoded_bytes = buf.getvalue()
    decoded = decode_single_frame(BytesIO(encoded_bytes))

    # Crop back to original odd dimensions if padding was applied.
    if (pad_w, pad_h) != (w, h):
        decoded = decoded[:h, :w, :]
    return decoded


def resize_and_center_crop(
    image: Image.Image | np.ndarray,
    height: int,
    width: int,
) -> Image.Image:
    """Aspect-preserving resize then center-crop to ``(height, width)``.

    Returns a PIL Image. Accepts a PIL Image or a uint8 numpy HWC array.
    Mirrors upstream's ``resize_and_center_crop`` semantics.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, mode="RGB")
    src_w, src_h = image.size
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    crop_left = (new_w - width) // 2
    crop_top = (new_h - height) // 2
    return image.crop((crop_left, crop_top, crop_left + width, crop_top + height))


def load_image_and_preprocess(
    image_path: str | Path,
    height: int,
    width: int,
    crf: int = DEFAULT_IMAGE_CRF,
) -> mx.array:
    """Full I2V image pipeline (upstream-iso).

    Pipeline (upstream verbatim):
        1. :func:`decode_image` — load PNG/JPEG → HxWx3 uint8.
        2. :func:`preprocess` — H.264 round-trip at ``crf``.
        3. :func:`resize_and_center_crop` — fit to target H/W.
        4. ``[0, 1] → [-1, 1]`` (:func:`to_vae_range`) + HWC→BCHW + bfloat16.

    Mirrors upstream's ``load_image_and_preprocess`` signature; the upstream
    ``dtype`` / ``device`` arguments are dropped (MLX uses bfloat16 + unified
    memory, no device choice).

    Returns:
        ``mx.array`` of shape ``(1, 3, H, W)`` in ``[-1, 1]``, bfloat16.
    """
    if isinstance(image_path, Path):
        image_path = str(image_path)
    arr = decode_image(image_path)
    if crf and crf > 0:
        arr = preprocess(arr, crf=crf)
    image = resize_and_center_crop(arr, height, width)

    # HWC uint8 → float32 → [-1, 1]
    f = np.asarray(image, dtype=np.float32) / 255.0
    f = f * 2.0 - 1.0
    # HWC → CHW → BCHW
    tensor = mx.array(f).transpose(2, 0, 1)[None, ...]
    return tensor.astype(mx.bfloat16)


__all__ = [
    "DEFAULT_IMAGE_CRF",
    "DEFAULT_IMAGE_CRF_V25",
    "decode_image",
    "decode_single_frame",
    "default_image_crf",
    "encode_single_frame",
    "from_vae_range",
    "load_image_and_preprocess",
    "preprocess",
    "resize_and_center_crop",
    "to_vae_range",
]
