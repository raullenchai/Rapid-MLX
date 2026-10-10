# SPDX-License-Identifier: Apache-2.0
"""Retire the selected pool model's legacy automatic prompt snapshots."""

from __future__ import annotations

import shutil
from pathlib import Path


def clear_legacy_prompt_cache(model: str) -> int:
    from rapid_mlx.runtime.cache import _exclusive_cache_lock

    root = Path.home() / ".cache" / "rapid-mlx" / "prefix_cache"
    if root.is_symlink():
        raise OSError("refusing a symlink as the prompt-cache root")
    safe_name = (
        model.replace("/", "--").replace("\\", "--").replace("..", "--").lstrip(".")
    ) or "default"
    if not root.exists():
        return 0
    # Exact legacy name or the semantic revision namespace, including interrupted
    # .new/.old snapshots. Never glob with an operator-supplied model string.
    namespaces = set()
    for path in root.iterdir():
        name = path.name
        for suffix in (".new", ".old"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        digest = name.removeprefix(safe_name + "--")
        if name == safe_name or (
            name.startswith(safe_name + "--")
            and len(digest) == 16
            and all(c in "0123456789abcdef" for c in digest)
        ):
            namespaces.add(name)
    removed = 0
    for name in sorted(namespaces):
        with _exclusive_cache_lock(
            str(root / name), operation="pool cleanup"
        ) as acquired:
            if not acquired:
                raise OSError(
                    "model prompt cache is in use; stop its other server first"
                )
            for suffix in ("", ".new", ".old"):
                path = root / (name + suffix)
                if path.is_symlink():
                    raise OSError(
                        "refusing a symlink in the model prompt-cache namespace"
                    )
                if path.exists():
                    shutil.rmtree(path)
                    removed += 1
    return removed
