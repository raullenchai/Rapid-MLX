# SPDX-License-Identifier: Apache-2.0
"""Rapid Canonical JSON v1 (RCJ-1) encoding and content digests."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            raise ValueError("RCJ-1 integers must be within the JSON safe range")
        return value
    if isinstance(value, float):
        raise TypeError("RCJ-1 forbids floating-point values")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("RCJ-1 object keys must be strings")
            if not key.isascii():
                raise TypeError("RCJ-1 object keys must be ASCII")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError("RCJ-1 key collision after NFC normalization")
            normalized[normalized_key] = _normalize(child)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        return [_normalize(child) for child in value]
    raise TypeError(f"RCJ-1 cannot encode {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode ``value`` according to the repository's RCJ-1 contract."""

    return json.dumps(
        _normalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def rcj_digest(value: Any) -> str:
    """Return the lowercase SHA-256 content address for an RCJ-1 value."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"
