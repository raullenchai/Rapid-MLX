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
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


_locks = threading.local()


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Serialize a complete operation, including nested transaction helpers."""
    canonical = path.resolve()
    identity = (os.getpid(), canonical)
    held: set[tuple[int, Path]] = getattr(_locks, "held", set())
    _locks.held = held
    if identity in held:
        yield
        return
    canonical.parent.mkdir(parents=True, exist_ok=True)
    lock_path = canonical.with_name(canonical.name + ".rapid-mlx.lock")
    # Retain the inode so a new caller cannot bypass an existing waiter.
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        held.add(identity)
        try:
            yield
        finally:
            held.remove(identity)
    finally:
        os.close(descriptor)


def _commit(
    path: Path, write_config: Callable[[], None], finish_receipt: Callable[[], None]
) -> None:
    with locked(path):
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


def _read_receipt(path: Path) -> dict[str, Any]:
    receipt = _common.load_json_lenient(_receipt(path))
    if not isinstance(receipt, dict):
        raise ValueError("Rapid settings ownership receipt must contain an object")
    return receipt


def write(
    path: Path,
    fields: dict[str, Any],
    write_config: Callable[[], None],
    *,
    before_fields: dict[str, Any],
    claim_changes: bool = True,
) -> None:
    """Own introduced/changed values and preserve only verified prior ownership.

    Callers pass the fields read under the operation lock. Atomic entries can
    disable claim_changes so an update to a pre-existing user entry never
    authorizes deleting that entire entry.
    """
    with locked(path):
        before = before_fields
        previous = _read_receipt(path)
        fingerprints = {}
        for key, value in fields.items():
            was_owned = False
            if key in before:
                try:
                    was_owned = previous.get(key) == _fingerprint(before[key])
                except (TypeError, ValueError):
                    pass
            if (
                key not in before
                or was_owned
                or (claim_changes and before[key] != value)
            ):
                fingerprints[key] = _fingerprint(value)
        _commit(
            path,
            write_config,
            lambda: _common.atomic_write_json(_receipt(path), fingerprints),
        )


def remove(path: Path, write_config: Callable[[], None]) -> None:
    """Commit removal with receipt retirement, rolling back if retirement fails."""
    _commit(path, write_config, lambda: forget(path))


def remove_fields(path: Path, fields: dict[str, Any], keys: tuple[str, ...]) -> bool:
    receipt = _read_receipt(path)
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
    with locked(path):
        _receipt(path).unlink(missing_ok=True)
