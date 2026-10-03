# SPDX-License-Identifier: Apache-2.0
"""Bounded, local-only media decoding for Clef's decision API."""

from __future__ import annotations

import base64
import binascii
import io
import warnings

_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_PIXELS = 16_777_216
_ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}


def _decode_image(value: str):
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
                if opened.width * opened.height > _MAX_PIXELS:
                    raise ValueError("Clef image exceeds the 16 MP pixel limit")
                if opened.format.lower() not in {"png", "jpeg", "webp"}:
                    raise ValueError("Clef image format does not match the allowlist")
                return opened.convert("RGB")
    except (
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("invalid or oversized Clef image") from exc


def decode_images(values: list[str]):
    if len(values) > 8:
        raise ValueError("Clef accepts at most 8 images")
    return [_decode_image(value) for value in values]


def decode_videos(values: list[list[str]]):
    import numpy as np

    if len(values) > 2 or sum(map(len, values)) > 32:
        raise ValueError("Clef accepts at most 2 videos and 32 frames total")
    if any(len(frames) < 2 for frames in values):
        raise ValueError("Clef videos must contain at least two frames")
    return [[np.asarray(_decode_image(frame)) for frame in frames] for frames in values]
