# SPDX-License-Identifier: Apache-2.0
"""Fingerprints for removable fields in shared adapter settings.

Receipts are private metadata: their fingerprints include credential-derived
values. They use the same owner-only permissions as the credential-bearing
configuration and its backups, and must not be published. Fingerprints support
value comparison, not password storage or authentication.

Missing receipts deliberately confer no ownership: a provider name or a loopback
URL alone is not evidence of a Rapid write. Reconfigured fields are retained
when they no longer match the receipt.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import _common


def _receipt(path: Path) -> Path:
    return path.with_name(path.name + ".rapid-mlx-owned.json")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _snapshot(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _restore(path: Path, previous: bytes | None) -> None:
    if _snapshot(path) == previous:
        return
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _common.atomic_write_text(path, previous.decode("utf-8"))


def _commit(
    path: Path, write_config: Callable[[], None], finish_receipt: Callable[[], None]
) -> None:
    # Keep the lock file's inode stable across transactions. Removing it would
    # let a new caller bypass a waiter still holding the old inode's lock.
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".rapid-mlx.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        previous = _snapshot(path)
        receipt_path = _receipt(path)
        previous_receipt = _snapshot(receipt_path)
        try:
            write_config()
            finish_receipt()
        except BaseException:
            _restore(path, previous)
            _restore(receipt_path, previous_receipt)
            raise
    finally:
        os.close(descriptor)


def write(path: Path, fields: dict[str, Any], write_config: Callable[[], None]) -> None:
    """Publish settings and their receipt, restoring settings if the receipt fails."""
    fingerprints = {key: _fingerprint(value) for key, value in fields.items()}
    _commit(
        path,
        write_config,
        lambda: _common.atomic_write_json(_receipt(path), fingerprints),
    )


def remove(path: Path, write_config: Callable[[], None]) -> None:
    """Commit removal with receipt retirement, rolling back if retirement fails."""
    _commit(path, write_config, lambda: forget(path))


def remove_fields(path: Path, fields: dict[str, Any], keys: tuple[str, ...]) -> bool:
    receipt = _common.load_json_lenient(_receipt(path))
    if not isinstance(receipt, dict):
        raise ValueError("Rapid settings ownership receipt must contain an object")
    removed = False
    for key in keys:
        if key not in fields:
            continue
        try:
            fingerprint = _fingerprint(fields[key])
        except (TypeError, ValueError):
            # YAML user entries may contain dates or mixed-key mappings.
            # Values Rapid could not have recorded are never ours to remove.
            continue
        if receipt.get(key) == fingerprint:
            del fields[key]
            removed = True
    return removed


def forget(path: Path) -> None:
    """Retire ownership after removal so later user writes cannot match it."""
    _receipt(path).unlink(missing_ok=True)
