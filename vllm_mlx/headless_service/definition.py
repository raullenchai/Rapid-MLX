# SPDX-License-Identifier: Apache-2.0
"""Inspection helpers for an installed LaunchDaemon definition."""

from __future__ import annotations

from pathlib import Path

from .common import DEFAULT_LABEL, home_for_user


def installed_plist(label: str = DEFAULT_LABEL) -> dict | None:
    from .install import _plist_path
    from .plist import parse_plist

    path = _plist_path(label)
    if not path.is_file():
        return None
    try:
        return parse_plist(path.read_bytes())
    except Exception:
        return None


def installed_identity(label: str = DEFAULT_LABEL) -> tuple[str, Path, Path] | None:
    """Return ``(user, home, config_path)`` from the installed plist."""
    plist = installed_plist(label)
    if not plist:
        return None
    user = plist.get("UserName")
    if not isinstance(user, str) or not user:
        return None
    environment = plist.get("EnvironmentVariables") or {}
    raw_home = environment.get("HOME")
    home = Path(raw_home) if isinstance(raw_home, str) else home_for_user(user)
    if home is None:
        return None
    argv = plist.get("ProgramArguments") or []
    raw_config = None
    for index, token in enumerate(argv[:-1]):
        if token == "--config":
            raw_config = argv[index + 1]
            break
    if isinstance(raw_config, str):
        config = Path(raw_config)
    else:
        from .config import config_path

        config = config_path(home, label)
    return user, home, config
