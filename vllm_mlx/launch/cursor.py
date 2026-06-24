# SPDX-License-Identifier: Apache-2.0
"""Cursor editor launch adapter.

Cursor's "AI Settings" panel for OpenAI-compatible providers is fed via
the standard VS Code-style ``settings.json``. The relevant keys are
under the ``cursor.aiprovider.*`` namespace (the same family Cursor
exposes for the OpenAI base-URL override in its settings UI).

Cursor lives at slightly different paths than vanilla VS Code — its
support dir is ``~/Library/Application Support/Cursor`` on macOS, NOT
``Code``. The settings file shape (top-level dotted keys → string
values) is identical.
"""

from __future__ import annotations

from pathlib import Path

from . import _common

# Cursor's per-OS user settings dir. We probe both macOS (Apple
# Silicon-first; the rapid-mlx target platform) and Linux (a small but
# growing fraction of Cursor users since the official Linux build
# shipped). Windows isn't supported by rapid-mlx and so isn't probed.
_CONFIG_DIR_MAC = (
    Path.home() / "Library" / "Application Support" / "Cursor" / "User"
)
_CONFIG_DIR_LINUX = Path.home() / ".config" / "Cursor" / "User"

_SETTINGS_FILENAME = "settings.json"


def _candidate_dirs() -> list[Path]:
    """Per-OS Cursor user-settings dirs in priority order."""
    return [_CONFIG_DIR_MAC, _CONFIG_DIR_LINUX]


def detect() -> bool:
    """Return True when Cursor appears to be installed.

    Three signals — any one is sufficient:

    * The ``Cursor.app`` bundle exists under ``/Applications`` or
      ``~/Applications`` (macOS).
    * The ``cursor`` CLI shim is on PATH (Cursor's "Install 'cursor'
      command in PATH" action drops a shim that mirrors VS Code's
      ``code`` command).
    * The Cursor user-settings dir already exists (the editor has
      been opened at least once).

    Multi-signal detection avoids false negatives on Linux installs
    where the ``.app`` bundle check doesn't apply.
    """
    if _common.mac_app_installed("Cursor"):
        return True
    if _common.which("cursor") is not None:
        return True
    return any(d.exists() for d in _candidate_dirs())


def current_config_path() -> Path | None:
    """Return Cursor's ``settings.json`` path.

    Picks the macOS path if Cursor's macOS dir is present (or doesn't
    exist on either OS — in which case the macOS canonical path is the
    one we mkdir into, matching what Cursor would create on first
    launch), otherwise falls back to the Linux path.
    """
    for d in _candidate_dirs():
        if d.exists():
            return d / _SETTINGS_FILENAME
    # No existing Cursor dir — return the macOS canonical path as the
    # creation target. detect() returns False in this case so the
    # dispatcher prints a "Cursor not detected" message before we get
    # here unless --force is in play (today: never).
    return _CONFIG_DIR_MAC / _SETTINGS_FILENAME


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Patch Cursor's ``settings.json`` to point at the local rapid-mlx
    OpenAI-compatible server.

    Keys we own — Cursor reads dotted top-level keys exactly like VS
    Code, so we set them as flat string keys (NOT a nested object):

    * ``cursor.aiprovider.openai.baseUrl`` → ``<server_url>/v1``
    * ``cursor.aiprovider.openai.apiKey`` → ``<api_key>``
    * ``cursor.aiprovider.openai.model`` → ``<model>``

    These keys mirror what the Cursor "OpenAI API" settings panel
    writes when the user clicks through the UI. Pasted-in values from
    that panel and values written here round-trip — Cursor doesn't
    distinguish.

    All other Cursor settings (theme, keybindings, every other dotted
    key) round-trip untouched.
    """
    path = config_path or current_config_path()
    assert path is not None

    existing = _common.load_json_lenient(path)
    _common.backup_existing(path)

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    existing["cursor.aiprovider.openai.baseUrl"] = base_url
    existing["cursor.aiprovider.openai.apiKey"] = api_key
    existing["cursor.aiprovider.openai.model"] = model

    _common.atomic_write_json(path, existing)
    return path
