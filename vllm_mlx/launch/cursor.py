# SPDX-License-Identifier: Apache-2.0
"""Cursor editor launch adapter for publicly reachable HTTPS endpoints.

Cursor routes BYOK requests through its own backend, so this adapter cannot
point Cursor directly at a server bound to the user's localhost. The launch
dispatcher enforces that distinction and only calls this adapter when the
operator explicitly supplies a public HTTPS ``--server-url`` (for example, a
tunnel forwarding to Rapid-MLX).
"""

from __future__ import annotations

from pathlib import Path

from . import _common

_CONFIG_DIR_MAC = Path.home() / "Library" / "Application Support" / "Cursor" / "User"
_CONFIG_DIR_LINUX = Path.home() / ".config" / "Cursor" / "User"
_SETTINGS_FILENAME = "settings.json"


def _candidate_dirs() -> list[Path]:
    """Return per-OS Cursor user-settings directories in priority order."""
    return [_CONFIG_DIR_MAC, _CONFIG_DIR_LINUX]


def detect() -> bool:
    """Return whether Cursor appears to be installed."""
    if _common.mac_app_installed("Cursor"):
        return True
    if _common.which("cursor") is not None:
        return True
    return any(directory.exists() for directory in _candidate_dirs())


def current_config_path() -> Path | None:
    """Return Cursor's ``settings.json`` path for this platform."""
    for directory in _candidate_dirs():
        if directory.exists():
            return directory / _SETTINGS_FILENAME
    return _CONFIG_DIR_MAC / _SETTINGS_FILENAME


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Point Cursor at a public HTTPS endpoint forwarding to Rapid-MLX.

    The caller is responsible for rejecting local/private endpoints. All
    unrelated Cursor settings are preserved.
    """
    path = config_path or current_config_path()
    assert path is not None

    existing = _common.load_json_lenient(path)
    _common.backup_existing(path)

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    existing["cursor.aiprovider.openai.baseUrl"] = base_url
    existing["cursor.aiprovider.openai.apiKey"] = api_key
    existing["cursor.aiprovider.openai.model"] = model

    _common.atomic_write_json(path, existing)
    return path
