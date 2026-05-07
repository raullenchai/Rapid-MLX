# SPDX-License-Identifier: Apache-2.0
"""Model alias registry — maps short names to HuggingFace paths."""

import difflib
import json
import os

_aliases: dict[str, str] | None = None


def _load() -> dict[str, str]:
    global _aliases
    if _aliases is None:
        path = os.path.join(os.path.dirname(__file__), "aliases.json")
        with open(path) as f:
            _aliases = json.load(f)
    return _aliases


def resolve_model(name: str) -> str:
    """Resolve a model alias to its full HuggingFace path.

    If name contains '/' it's already a full path — pass through.
    If a local file/directory with the name exists, prefer that.
    If name matches an alias, return the mapped HF path.
    Otherwise return unchanged.
    """
    if "/" in name:
        return name
    if os.path.exists(name):
        return name
    aliases = _load()
    return aliases.get(name, name)


def list_aliases() -> dict[str, str]:
    """Return all available aliases."""
    return dict(_load())


def suggest_similar(name: str, n: int = 3, cutoff: float = 0.5) -> list[str]:
    """Return up to `n` aliases similar to `name` for typo suggestions."""
    return difflib.get_close_matches(name, _load().keys(), n=n, cutoff=cutoff)
