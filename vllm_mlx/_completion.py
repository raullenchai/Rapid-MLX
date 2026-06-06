# SPDX-License-Identifier: Apache-2.0
"""Shell completion helpers for the rapid-mlx CLI.

Used by ``argcomplete`` to suggest model aliases when the user
tab-completes on subcommands like ``rapid-mlx chat gemma-4-<TAB>``.

Kept free of heavy imports so the completion handler stays snappy:
the only file read here is ``aliases.json`` (~10 KB), small enough
to JSON-decode on every keystroke without measurable lag and small
enough that re-reading per call means a freshly-edited
``aliases.json`` tab-completes the same shell session.

Wired in ``vllm_mlx/cli.py`` and ``vllm_mlx/share/cli.py`` via::

    arg = parser.add_argument("model", ...)
    arg.completer = alias_completer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ALIASES_PATH = Path(__file__).parent / "aliases.json"


def _load_alias_names() -> list[str]:
    """Return sorted alias keys from ``aliases.json`` (empty list on error).

    Tab-completion must never raise — a missing or corrupt aliases.json
    should degrade gracefully to "no suggestions" rather than crashing
    the user's shell.
    """
    try:
        with _ALIASES_PATH.open("rb") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    return sorted(data.keys())


def alias_completer(prefix: str = "", **_: Any) -> list[str]:
    """Argcomplete callback: aliases matching ``prefix``.

    Returns the full sorted alias list when prefix is empty (user
    typed nothing yet, hit Tab) — the shell collapses to the longest
    common prefix and re-prompts on a second Tab, which is the
    standard behavior.
    """
    names = _load_alias_names()
    if not prefix:
        return names
    return [n for n in names if n.startswith(prefix)]


def alias_csv_completer(prefix: str = "", **_: Any) -> list[str]:
    """Comma-separated-list variant for ``doctor --models a,b,c``.

    The user-visible prefix at completion time contains everything
    typed for this flag — e.g. ``qwen3.5-4b,gem`` when partway through
    the second entry. We split on the last comma, complete only the
    trailing token against alias names, and re-attach the prefix so
    the shell inserts the full value correctly.
    """
    head, sep, tail = prefix.rpartition(",")
    pool = _load_alias_names()
    matches = [n for n in pool if n.startswith(tail)]
    if not sep:
        return matches
    return [f"{head},{m}" for m in matches]
