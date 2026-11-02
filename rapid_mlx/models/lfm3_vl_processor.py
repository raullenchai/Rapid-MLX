# SPDX-License-Identifier: Apache-2.0
"""Image preprocessing primitives for LFM3-VL.

The patch layout follows the checkpoint's image processor.  The serving pixel
limit is deliberately lower than the checkpoint's unconstrained default.
"""

import math

import numpy as np
from PIL import Image

IMAGE_TOKEN_ID = 125030
VIDEO_TOKEN_ID = 125029
DEFAULT_MAX_PIXELS = 1_048_576


def smart_resize(
    height, width, factor=32, min_pixels=65_536, max_pixels=DEFAULT_MAX_PIXELS
):
    if min(height, width) <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("Invalid image dimensions or pixel limits")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio exceeds 200")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def image_grid(height, width, *, min_pixels=65_536, max_pixels=DEFAULT_MAX_PIXELS):
    """Return the patch grid from dimensions, without decoding image pixels."""
    h, w = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    if h * w > max_pixels:
        raise ValueError("Resized image exceeds the pixel budget")
    return (1, h // 16, w // 16)


def preprocess_images(images, *, min_pixels=65_536, max_pixels=DEFAULT_MAX_PIXELS):
    """Return ordered NumPy patches and THW grids for PIL images."""
    patches, grids = [], []
    for image in images:
        _, gh, gw = image_grid(
            image.height, image.width, min_pixels=min_pixels, max_pixels=max_pixels
        )
        h, w = gh * 16, gw * 16
        rgb = image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
        pixels = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        pixels = pixels.reshape(3, gh // 2, 2, 16, gw // 2, 2, 16)
        pixels = pixels.transpose(1, 4, 2, 5, 0, 3, 6)
        pixels = np.repeat(pixels[..., None, :, :], 2, axis=5)
        patches.append(pixels.reshape(gh * gw, 1536))
        grids.append((1, gh, gw))
    return (
        np.concatenate(patches) if patches else np.empty((0, 1536), dtype=np.float32),
        np.asarray(grids, dtype=np.int32).reshape(-1, 3),
    )


def expand_image_tokens(token_ids, grids):
    """Expand one image marker per grid, preserving image order."""
    grids = iter(grids)
    result = []
    for token in token_ids:
        if token == VIDEO_TOKEN_ID:
            raise ValueError("Video is not supported by LFM3-VL")
        if token == IMAGE_TOKEN_ID:
            try:
                t, h, w = next(grids)
            except StopIteration as exc:
                raise ValueError("More image markers than images") from exc
            if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
                raise ValueError("Invalid image grid")
            result.extend([IMAGE_TOKEN_ID] * (int(h) * int(w) // 4))
        else:
            result.append(token)
    if next(grids, None) is not None:
        raise ValueError("More images than image markers")
    return result


class LFM3VLProcessor:
    """Tokenizer facade and ordered image preparation for the media lane."""

    native_media_backend = "lfm3_vl"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def expand_image_prompt(
        self, prompt, image_sizes, *, min_pixels=65_536, max_pixels=None
    ):
        """Expand image markers using only ordered (width, height) sizes."""
        token_ids = (
            self.tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str)
            else list(prompt)
        )
        max_pixels = min(int(max_pixels or DEFAULT_MAX_PIXELS), DEFAULT_MAX_PIXELS)
        grids = [
            image_grid(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
            for w, h in image_sizes
        ]
        return expand_image_tokens(token_ids, grids)

    def prepare_inputs(
        self, prompt, images=(), *, min_pixels=65_536, max_pixels=DEFAULT_MAX_PIXELS
    ):
        import mlx.core as mx

        if max_pixels is None:
            max_pixels = DEFAULT_MAX_PIXELS
        max_pixels = min(int(max_pixels), DEFAULT_MAX_PIXELS)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        patches, grids = preprocess_images(
            images, min_pixels=min_pixels, max_pixels=max_pixels
        )
        expanded = expand_image_tokens(token_ids, grids)
        return {
            "input_ids": mx.array([expanded]),
            "pixel_values": mx.array(patches) if len(grids) else None,
            "image_grid_thw": mx.array(grids) if len(grids) else None,
            "attention_mask": mx.ones((1, len(expanded)), dtype=mx.int32),
        }
