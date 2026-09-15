"""Small shared utilities: dtypes, seeding, logging, image <-> array."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from PIL import Image

__all__ = [
    "DTYPES",
    "as_dtype",
    "get_logger",
    "seed_everything",
    "prepare_image",
    "to_pil",
    "to_array",
]

# Canonical string <-> mlx dtype mapping used in configs and CLI args.
DTYPES: dict[str, mx.Dtype] = {
    "float32": mx.float32,
    "fp32": mx.float32,
    "float16": mx.float16,
    "fp16": mx.float16,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
}


def as_dtype(dtype: str | mx.Dtype | None) -> mx.Dtype | None:
    """Resolve a dtype given as a string alias or an ``mx.Dtype``."""
    if dtype is None or isinstance(dtype, mx.Dtype):
        return dtype
    try:
        return DTYPES[dtype.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown dtype {dtype!r}. Choose from {sorted(DTYPES)}."
        ) from exc


_LOG_LEVEL = os.environ.get("MLX_DIFFUSION_LOG", "INFO").upper()


def get_logger(name: str = "mlx_diffuser") -> logging.Logger:
    """Return a configured library logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> mx.array:
    """Seed MLX's global RNG and return a fresh key for explicit use."""
    mx.random.seed(seed)
    return mx.random.key(seed)


def to_array(image: Image.Image | np.ndarray, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Convert a PIL image or HWC uint8/float array to a CHW-free [-1, 1] array.

    Returns shape ``(H, W, C)`` in ``[-1, 1]`` (MLX/`channels-last` convention).
    """
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 127.5 - 1.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return mx.array(arr).astype(dtype)


def prepare_image(
    image,
    *,
    height: int,
    width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Prepare a path, PIL image, NumPy array, or MLX array for image conditioning.

    Paths and PIL images are converted to RGB and resized with Lanczos. Array inputs
    must already have the requested spatial size and be ``HWC`` or ``BHWC``. Integer
    arrays are normalized to ``[-1, 1]``; floating-point arrays are assumed to
    already use that range.
    """
    import numpy as np
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )
    elif isinstance(image, Image.Image):
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

    if isinstance(image, Image.Image):
        array = to_array(image, dtype=dtype)[None]
    elif isinstance(image, mx.array):
        array = image
        if array.dtype == mx.uint8:
            array = array.astype(mx.float32) / 127.5 - 1.0
        else:
            array = array.astype(dtype)
    else:
        np_image = np.asarray(image)
        if np.issubdtype(np_image.dtype, np.integer):
            np_image = np_image.astype(np.float32) / 127.5 - 1.0
        array = mx.array(np_image).astype(dtype)

    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(
            f"image must have shape (H, W, C) or (B, H, W, C), got {array.shape}."
        )
    if array.shape[1:3] != (height, width):
        raise ValueError(
            f"array image must already be {height}x{width}; got {array.shape[1]}x{array.shape[2]}."
        )
    if array.shape[-1] != 3:
        raise ValueError(f"image must have 3 RGB channels, got {array.shape[-1]}.")
    return array


def to_pil(array: mx.array) -> Image.Image:
    """Convert a single ``(H, W, C)`` array in ``[-1, 1]`` to a PIL image."""
    import numpy as np
    from PIL import Image

    arr = mx.clip((array + 1.0) * 127.5 + 0.5, 0, 255).astype(mx.uint8)
    np_arr = np.array(arr)
    if np_arr.shape[-1] == 1:
        np_arr = np_arr[..., 0]
    return Image.fromarray(np_arr)
