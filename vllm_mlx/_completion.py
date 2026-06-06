# SPDX-License-Identifier: Apache-2.0
"""Shell completion helpers for the rapid-mlx CLI.

Used by ``argcomplete`` to suggest model aliases when the user
tab-completes on subcommands like ``rapid-mlx chat gemma-4-<TAB>``.

Two layers of defense for the completion hot path:

1. **mtime-keyed cache** so a hot Tab burst doesn't re-parse the JSON
   on every keystroke. The cache key is ``(mtime, size)`` so an
   editor-save re-load is automatic — no manual invalidation needed.
2. **Size cap + control-char strip** so an oversized or hostile
   ``aliases.json`` can't (a) stall every Tab on JSON decode or
   (b) leak protocol-separator bytes into argcomplete's line-oriented
   stdout IPC. The wheel-shipped file is well under the cap and only
   uses ASCII identifier characters, so this is purely defense in
   depth against a supply-chain swap or a hand-edited dev copy.

Wired in ``vllm_mlx/cli.py`` and ``vllm_mlx/share/cli.py`` via::

    arg = parser.add_argument("model", ...)
    arg.completer = alias_completer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ALIASES_PATH = Path(__file__).parent / "aliases.json"

# Hard cap on the JSON we will parse. The wheel-shipped file is ~10 KB
# (73 aliases as of 0.6.77). 1 MB is ~100× headroom while still tiny
# enough to fail closed on a hostile multi-GB file before we burn the
# user's shell on a JSON decode.
_MAX_ALIASES_BYTES = 1_000_000

# Cache of (mtime, size, sorted_names). One-entry cache is enough: the
# completer is invoked from a single shell process per Tab burst.
_CACHE: tuple[float, int, list[str]] | None = None


def _is_safe_alias_name(name: object) -> bool:
    """Reject names that would corrupt argcomplete's line-oriented IPC.

    Argcomplete writes completions to fd 8 separated by either ``\\v``
    or ``\\n`` (depending on ``_ARGCOMPLETE_IFS``). A key containing
    those bytes would split into multiple bogus suggestions; a key
    containing control characters could mis-render in the user's
    terminal or corrupt the shell completion buffer. Require printable
    non-whitespace characters only — every legitimate alias today
    matches.
    """
    if not isinstance(name, str) or not name:
        return False
    return all(c.isprintable() and not c.isspace() for c in name)


def _load_alias_names() -> list[str]:
    """Return sorted, safety-filtered alias keys from ``aliases.json``.

    Tab-completion must never raise — a missing, oversized, or corrupt
    ``aliases.json`` should degrade gracefully to "no suggestions"
    rather than crashing the user's shell. The mtime+size cache lets
    a burst of Tabs hit a hot path instead of re-decoding JSON each
    time.
    """
    global _CACHE
    try:
        stat = _ALIASES_PATH.stat()
    except OSError:
        return []
    if stat.st_size > _MAX_ALIASES_BYTES:
        return []
    cache = _CACHE
    if cache is not None and cache[0] == stat.st_mtime and cache[1] == stat.st_size:
        return cache[2]
    try:
        with _ALIASES_PATH.open("rb") as f:
            raw = f.read(_MAX_ALIASES_BYTES + 1)
    except OSError:
        return []
    if len(raw) > _MAX_ALIASES_BYTES:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        # ``json.loads`` recurses on nested objects/arrays. A small but
        # deeply-nested file (e.g. ``[[[[...]]]]``) raises
        # ``RecursionError``, which is technically not a ``JSONDecodeError``
        # and would otherwise leak as a traceback into the user's shell.
        return []
    if not isinstance(data, dict):
        return []
    names = sorted(k for k in data if _is_safe_alias_name(k))
    _CACHE = (stat.st_mtime, stat.st_size, names)
    return names


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
    the second entry. We split on the last comma, strip whitespace
    around the tail so ``--models a, gem<TAB>`` works the same as
    ``--models a,gem<TAB>`` (the runtime ``split + strip`` accepts
    both forms; the completer should match that contract), complete
    only the stripped tail against alias names, and re-attach the
    head prefix so the shell inserts the full value correctly.
    """
    head, sep, tail = prefix.rpartition(",")
    tail = tail.lstrip()
    pool = _load_alias_names()
    matches = [n for n in pool if n.startswith(tail)]
    if not sep:
        return matches
    return [f"{head},{m}" for m in matches]
