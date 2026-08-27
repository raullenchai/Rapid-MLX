# SPDX-License-Identifier: Apache-2.0
"""Atomic rollback admission for composite generation caches."""

from __future__ import annotations

import copy
import threading
from collections.abc import Iterable
from typing import Any

_TRIM_LOCK = threading.RLock()


def _leaf_caches(cache: Any):
    children = getattr(cache, "caches", None)
    if children is None:
        yield cache
        return
    for child in children:
        yield from _leaf_caches(child)


def _checkpoint(cache: Any):
    capture = getattr(cache, "trim_checkpoint", None)
    if callable(capture):
        return True, capture()
    # Cache.trim() is a logical-cursor operation: backing tensor payloads stay
    # allocated for reuse. Preserve array references and copy mutable cursor
    # containers so a failed transaction can restore the pre-trim view.
    state = {}
    for name, value in vars(cache).items():
        state[name] = (
            copy.copy(value) if isinstance(value, (dict, list, set, tuple)) else value
        )
    return False, state


def _restore(cache: Any, checkpoint) -> None:
    custom, state = checkpoint
    if custom:
        cache.restore_trim_checkpoint(state)
        return
    vars(cache).clear()
    vars(cache).update(state)


def can_trim(cache: Any, n: int) -> bool:
    """Return whether ``cache.trim(n)`` can commit without partial mutation."""
    if n < 0:
        return False
    children = getattr(cache, "caches", None)
    if children is not None:
        return all(can_trim(child, n) for child in children)
    amount_check = getattr(cache, "can_trim", None)
    if callable(amount_check):
        return bool(amount_check(n))
    can_undo = getattr(cache, "_can_undo", None)
    if callable(can_undo) and can_undo(n):
        return True
    check = getattr(cache, "is_trimmable", None)
    size = getattr(cache, "size", None)
    if not (callable(check) and check() and callable(size)):
        return False
    logical_size = size()
    try:
        return int(logical_size) >= n
    except (TypeError, ValueError):
        return False


def trim_all(caches: Iterable[Any], n: int) -> bool:
    """Preflight and transactionally trim every composite cache leaf."""
    if n <= 0:
        return n == 0
    leaves = [leaf for cache in caches for leaf in _leaf_caches(cache)]
    with _TRIM_LOCK:
        if not all(can_trim(cache, n) for cache in leaves):
            return False
        checkpoints = [(cache, _checkpoint(cache)) for cache in leaves]
        try:
            for cache in leaves:
                if cache.trim(n) != n:
                    raise RuntimeError("cache returned a short trim")
        except Exception:
            for cache, checkpoint in reversed(checkpoints):
                _restore(cache, checkpoint)
            return False
        return True
