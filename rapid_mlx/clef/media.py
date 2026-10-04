# SPDX-License-Identifier: Apache-2.0
"""Bounded, local-only media decoding for Clef's decision API."""

from __future__ import annotations

import base64
import binascii
import io
import warnings

_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_PIXELS = 16_777_216
_MAX_TOTAL_PIXELS = 16_777_216
_ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}


def _decode_image(value: str, pixel_budget: list[int]):
    from PIL import Image

    if not isinstance(value, str) or not value.startswith("data:"):
        raise ValueError("Clef media must be a base64 image data URL")
    header, separator, payload = value.partition(",")
    if not separator or not header.endswith(";base64"):
        raise ValueError("Clef media must be a base64 image data URL")
    mime = header[5:-7]
    if mime not in _ALLOWED_MIME:
        raise ValueError("Clef media must be PNG, JPEG, or WebP")
    if len(payload) > (_MAX_IMAGE_BYTES * 4 // 3) + 4:
        raise ValueError("Clef image exceeds the 4 MiB encoded limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image data") from exc
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError("Clef image exceeds the 4 MiB encoded limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                pixels = opened.width * opened.height
                if pixels > _MAX_PIXELS:
                    raise ValueError("Clef image exceeds the 16 MP pixel limit")
                if pixels > pixel_budget[0]:
                    raise ValueError("Clef media exceeds the 16 MP total pixel limit")
                if opened.format.lower() != mime.split("/", 1)[1]:
                    raise ValueError(
                        "Clef image format does not match the data URL MIME"
                    )
                pixel_budget[0] -= pixels
                return opened.convert("RGB")
    except (
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("invalid or oversized Clef image") from exc


def decode_media(images: list[str] | None, videos: list[list[str]] | None):
    """Decode media under one pixel budget shared by images and video frames."""
    images = images or []
    videos = videos or []
    if len(images) > 8:
        raise ValueError("Clef accepts at most 8 images")
    if len(videos) > 2 or sum(map(len, videos)) > 32:
        raise ValueError("Clef accepts at most 2 videos and 32 frames total")
    if any(len(frames) < 2 for frames in videos):
        raise ValueError("Clef videos must contain at least two frames")
    pixel_budget = [_MAX_TOTAL_PIXELS]
    decoded_images = [_decode_image(value, pixel_budget) for value in images]
    decoded_videos = []
    if videos:
        import numpy as np

        decoded_videos = [
            [np.asarray(_decode_image(frame, pixel_budget)) for frame in frames]
            for frames in videos
        ]
    return decoded_images, decoded_videos


def decode_images(values: list[str]):
    return decode_media(values, None)[0]


def decode_videos(values: list[list[str]]):
    return decode_media(None, values)[1]
