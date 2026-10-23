# SPDX-License-Identifier: Apache-2.0
"""Fingerprints for removable fields in shared adapter settings.

The receipt contains no credentials. Missing receipts deliberately confer no
ownership: a provider name or a loopback URL alone is not evidence of a Rapid
write. Reconfigured fields are retained when they no longer match the receipt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import _common


def _receipt(path: Path) -> Path:
    return path.with_name(path.name + ".rapid-mlx-owned.json")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def record(path: Path, fields: dict[str, Any]) -> None:
    _common.atomic_write_json(
        _receipt(path), {key: _fingerprint(value) for key, value in fields.items()}
    )


def remove_fields(path: Path, fields: dict[str, Any], keys: tuple[str, ...]) -> bool:
    receipt = _common.load_json_lenient(_receipt(path))
    if not isinstance(receipt, dict):
        raise ValueError("Rapid settings ownership receipt must contain an object")
    removed = False
    for key in keys:
        if key in fields and receipt.get(key) == _fingerprint(fields[key]):
            del fields[key]
            removed = True
    return removed


def forget(path: Path) -> None:
    """Retire ownership after removal so later user writes cannot match it."""
    _receipt(path).unlink(missing_ok=True)
