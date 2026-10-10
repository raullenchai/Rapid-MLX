# SPDX-License-Identifier: Apache-2.0
"""Retire the selected pool model's legacy automatic prompt snapshots."""

from __future__ import annotations

import hashlib
import shutil
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace


def _known_namespaces(model: str, safe_name: str) -> set[str]:
    from rapid_mlx.kv_cache_dtype import KV_CACHE_DTYPES
    from rapid_mlx.runtime.cache import _semantic_cache_identity

    # Derive full fingerprints, never infer ownership from a lossy path prefix.
    # Include the original pre-semantic namespace used by older installations.
    identities = [model]
    for dtype in KV_CACHE_DTYPES:
        cfg = SimpleNamespace(engine=None, kv_cache_dtype=dtype)
        identities.append(_semantic_cache_identity(cfg, model))
    return {
        safe_name + "--" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        for identity in identities
    }


def clear_legacy_prompt_cache(model: str) -> int:
    from rapid_mlx.runtime.cache import _exclusive_cache_lock

    root = Path.home()
    for component in (".cache", "rapid-mlx", "prefix_cache"):
        root = root / component
        if root.is_symlink():
            raise OSError("refusing a symlink in the prompt-cache root path")
    safe_name = (
        model.replace("/", "--").replace("\\", "--").replace("..", "--").lstrip(".")
    ) or "default"
    known = _known_namespaces(model, safe_name)
    # Lock before discovery, including namespaces a writer has not created yet.
    # The existing lock protocol also creates an initially absent cache root.
    with ExitStack() as locks:
        for name in sorted(known):
            acquired = locks.enter_context(
                _exclusive_cache_lock(str(root / name), operation="pool cleanup")
            )
            if not acquired:
                raise OSError(
                    "model prompt cache is in use; stop its other server first"
                )
        return _clear_locked_snapshots(root, known, safe_name)


def _clear_locked_snapshots(root: Path, known: set[str], safe_name: str) -> int:
    # Unhashed names have no recoverable ownership: even model and .model
    # collide, and model.new can name a model or an interrupted transaction.
    # Refuse every unhashed candidate rather than erase another model.
    candidates: set[str] = set()
    for path in root.iterdir():
        name = path.name
        # Exact matching precedes transaction-suffix interpretation, since a
        # model identifier may itself end in .new or .old.
        if name in known:
            candidates.add(name)
            continue
        if name == safe_name:
            raise OSError(
                "legacy prompt-cache ownership is ambiguous; stop other servers "
                "and review/remove the selected model's old snapshots manually"
            )
        base = name
        for suffix in (".new", ".old"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        if base in known:
            candidates.add(base)
            continue
        digest = base.removeprefix(safe_name + "--")
        if base == safe_name or (
            base.startswith(safe_name + "--")
            and len(digest) == 16
            and all(c in "0123456789abcdef" for c in digest)
        ):
            raise OSError(
                "legacy prompt-cache ownership is ambiguous; stop other servers "
                "and review/remove the selected model's old snapshots manually"
            )
    removed = 0
    for name in sorted(candidates):
        for suffix in ("", ".new", ".old"):
            path = root / (name + suffix)
            if path.is_symlink():
                raise OSError("refusing a symlink in the model prompt-cache namespace")
            if path.exists():
                shutil.rmtree(path)
                removed += 1
    return removed
