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


def _family_prefix(name: str) -> str:
    """Strip trailing size/quant tokens to get the model-family prefix.

    ``deepseek-v4-27b`` → ``deepseek-v4`` (drop ``27b``)
    ``qwen3.5-122b-8bit`` → ``qwen3.5`` (drop ``8bit`` then ``122b``)
    ``hermes`` → ``hermes`` (single token, no change)

    Used to keep typo suggestions inside the same family — ``deepseek-v4-27b``
    suggests ``deepseek-v4-flash``, not ``deepseek-r1-32b``.
    """
    parts = name.split("-")
    while parts:
        tail = parts[-1]
        if not tail:
            break
        # size token (``27b``, ``1.5b``), quant token (``8bit``, ``mxfp4``),
        # or pure-digit version segment.
        if tail[-1].lower() == "b" or "bit" in tail.lower() or tail.isdigit():
            parts.pop()
            continue
        break
    return "-".join(parts)


def suggest_similar(name: str, n: int = 3, cutoff: float = 0.5) -> list[str]:
    """Return up to ``n`` aliases similar to ``name`` for typo suggestions.

    Family-aware: only suggests aliases that share a family prefix with the
    input. This keeps the wrong-family bait-and-switch (typing
    ``deepseek-v4-27b`` and being told ``deepseek-r1-32b``) from happening,
    and also prevents legitimate single-segment HuggingFace IDs like
    ``gpt2`` or ``bert-base-uncased`` from spuriously matching.

    Behaviour:
    - Multi-segment family (``fam`` contains ``-``): match aliases that
      start with ``fam-`` or are exactly ``fam``.
    - Single-segment family (``hermes``, ``qwen3.5``, ``gpt2``): match
      aliases whose name starts with ``fam``. Requires ``fam`` to be at
      least 3 chars to avoid one- or two-letter false positives.
    - No same-family match: return ``[]`` (let the loader's 404 handle it).
    """
    fam = _family_prefix(name)
    if not fam:
        return []
    aliases = list(_load().keys())
    if "-" in fam:
        same_fam = [a for a in aliases if a.startswith(fam + "-") or a == fam]
    else:
        if len(fam) < 3:
            return []
        same_fam = [a for a in aliases if a.startswith(fam)]
    if not same_fam:
        return []
    return difflib.get_close_matches(name, same_fam, n=n, cutoff=cutoff)
