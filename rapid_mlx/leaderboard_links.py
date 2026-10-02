# SPDX-License-Identifier: Apache-2.0
"""Links into the community leaderboard (rapidmlx.com/leaderboard).

The leaderboard has one page with URL-state views:

* ``?mac=<chip>-<memory>`` — what runs well on one Mac class, e.g.
  ``?mac=m4-pro-48`` (the engine's smart/fast picks plus every community
  measurement on that exact chip and memory size);
* ``?run=<submission id>`` — one shared benchmark run, in context.

``install.sh`` builds the same ``?mac=`` slug in shell (``leaderboard_mac_url``);
``tests/test_leaderboard_links.py`` keeps the two in agreement. Nothing here
performs network I/O; only the chip/memory probe reads ``sysctl``.
"""

from __future__ import annotations

import re

LEADERBOARD_URL = "https://rapidmlx.com/leaderboard"

_RUN_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# Same grammar as install.sh's leaderboard_mac_url: the whole lower-cased
# brand string must be "apple m<gen>[ pro|max|ultra]" — anything else
# (Intel, a VM's "Apple M2 Max (Virtual)", stray tokens) gets the bare board.
_BRAND = re.compile(r"^apple m([0-9]+)( (pro|max|ultra))?$")


def mac_slug(chip: str | None, memory_gib: int | None) -> str | None:
    """``("Apple M4 Pro", 48)`` -> ``"m4-pro-48"``; ``None`` when unknown.

    Memory must be a positive whole number of GiB, as ``sysctl hw.memsize``
    reports for every Apple Silicon Mac; anything else is not guessed at.
    """
    if not isinstance(chip, str):
        return None
    if (
        isinstance(memory_gib, bool)
        or not isinstance(memory_gib, int)
        or memory_gib <= 0
    ):
        return None
    match = _BRAND.match(chip.lower())
    if not match:
        return None
    parts = [f"m{match.group(1)}"]
    if match.group(3):
        parts.append(match.group(3))
    parts.append(str(memory_gib))
    return "-".join(parts)


def mac_url(chip: str | None, memory_gib: int | None) -> str:
    """The ``?mac=`` view for this Mac, or the board itself when unknown."""
    slug = mac_slug(chip, memory_gib)
    return f"{LEADERBOARD_URL}?mac={slug}" if slug else LEADERBOARD_URL


def run_url(submission_id: object) -> str | None:
    """The ``?run=`` view for an accepted submission id, else ``None``."""
    if not isinstance(submission_id, str):
        return None
    value = submission_id.strip().lower()
    return f"{LEADERBOARD_URL}?run={value}" if _RUN_ID.match(value) else None


def this_mac_url(memory_gib: int | None = None) -> str:
    """``mac_url`` for the host, degrading to the board on any probe failure.

    Pass ``memory_gib`` when the caller already measured it, so only the chip
    is probed.
    """
    from rapid_mlx.community_bench import hardware

    try:
        chip = hardware._chip()
    except RuntimeError:
        chip = None
    if memory_gib is None:
        memory_gib = hardware.host_memory_gib()
    return mac_url(chip, memory_gib)
