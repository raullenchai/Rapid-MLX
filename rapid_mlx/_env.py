# SPDX-License-Identifier: Apache-2.0
"""Shared parsing for boolean ``RAPID_MLX_*`` environment switches."""

from __future__ import annotations

import os

TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})
FALSEY_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled"})


def is_truthy(value: str | None) -> bool:
    """Return True if ``value`` is one of ``TRUTHY_VALUES`` (any case, stripped)."""
    return value is not None and value.strip().lower() in TRUTHY_VALUES


def is_falsey(value: str | None) -> bool:
    """Return True if ``value`` is one of ``FALSEY_VALUES`` (any case, stripped)."""
    return value is not None and value.strip().lower() in FALSEY_VALUES


def env_truthy(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy value.

    Unset, empty, falsey, and unrecognised values are all False, so callers
    use this for opt-in switches.
    """
    return is_truthy(os.environ.get(name))


def env_falsey(name: str) -> bool:
    """Return True if env var ``name`` is set to a falsey value.

    Unset, empty, truthy, and unrecognised values are all False, so callers
    use ``not env_falsey(...)`` for switches that are on by default.
    """
    return is_falsey(os.environ.get(name))
