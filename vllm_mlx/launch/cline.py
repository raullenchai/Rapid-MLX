# SPDX-License-Identifier: Apache-2.0
"""Cline (VS Code extension) launch adapter.

Cline lives under VS Code's global storage as a single
``cline_mcp_settings.json``. The relevant keys are the OpenAI-compatible
provider settings — Cline routes traffic at ``openAiBaseUrl`` and
authenticates with ``openAiApiKey`` when ``apiProvider`` is
``"openai"``. Pointing those three at our local server is all that
``rapid-mlx launch cline`` has to do.

Cline's exact config schema has churned a few times across releases; we
preserve every existing key and only touch the four we know we own,
which means a config from a future Cline release still round-trips
cleanly (the unknown keys come back out untouched on the next save).
"""

from __future__ import annotations

from pathlib import Path

from . import _common

# VS Code extension id ("publisher.name"). Cline's stable id is
# ``saoudrizwan.claude-dev`` (the project predates the rename to
# "Cline" and the extension id never changed). The settings file
# lives under VS Code's globalStorage tree for that extension.
_EXTENSION_ID = "saoudrizwan.claude-dev"

# The settings filename — same shape across VS Code Stable, Insiders,
# and VSCodium. We probe all three install roots in priority order so a
# user on VSCodium isn't penalised for not running upstream VS Code.
_SETTINGS_FILENAME = "cline_mcp_settings.json"


def _candidate_settings_roots() -> list[Path]:
    """Per-OS list of VS Code (and forks') ``User/globalStorage`` roots.

    Order is "most likely first" so :func:`current_config_path` returns
    the canonical Stable path when multiple installs coexist. macOS
    paths come first because that's the platform rapid-mlx targets
    (Apple Silicon); Linux paths follow so CI / dev containers still
    detect a configured Cline install.
    """
    home = Path.home()
    return [
        # macOS — VS Code Stable, Insiders, VSCodium.
        home / "Library/Application Support/Code/User/globalStorage",
        home / "Library/Application Support/Code - Insiders/User/globalStorage",
        home / "Library/Application Support/VSCodium/User/globalStorage",
        # Linux — same three flavours under ~/.config.
        home / ".config/Code/User/globalStorage",
        home / ".config/Code - Insiders/User/globalStorage",
        home / ".config/VSCodium/User/globalStorage",
    ]


def detect() -> bool:
    """Return True when a VS Code-family install has the Cline extension
    materialised on disk.

    We check for the per-extension ``globalStorage`` directory rather
    than the editor binary alone — a user can have ``code`` on their
    PATH without having installed Cline, in which case ``launch cline``
    has nothing useful to do.
    """
    return current_config_path() is not None


def current_config_path() -> Path | None:
    """Return the canonical Cline settings path, or ``None`` if Cline
    isn't installed.

    "Installed" means *either*:

    * the settings file already exists (Cline has been opened at least
      once and wrote its initial config), OR
    * the extension's ``globalStorage`` dir exists (Cline is installed
      but hasn't created the MCP settings file yet — we'll create it).

    If neither condition holds for any VS Code flavour, return None and
    the launch dispatcher prints a "Cline not detected — install it
    from the VS Code marketplace" hint.
    """
    for root in _candidate_settings_roots():
        ext_dir = root / _EXTENSION_ID / "settings"
        candidate = ext_dir / _SETTINGS_FILENAME
        # Prefer a fully-materialised file. If the dir exists but the
        # file doesn't, treat as installed-but-uninitialised and
        # return the canonical path so we can create the file.
        if candidate.exists():
            return candidate
        if ext_dir.exists():
            return candidate
    return None


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Patch Cline's ``cline_mcp_settings.json`` to route at the local
    rapid-mlx OpenAI-compatible server.

    Keys we own:

    * ``apiProvider`` → ``"openai"``
    * ``openAiBaseUrl`` → ``<server_url>/v1``
    * ``openAiApiKey`` → ``<api_key>`` (default: ``"sk-noop"``,
      since rapid-mlx defaults to no-auth on loopback)
    * ``openAiModelId`` → ``<model>``

    Every other key in the existing file is preserved verbatim — the
    user's MCP tool list, custom instructions, ratelimit prefs all
    survive a relaunch.

    The ``config_path`` arg is a test/dry-run hook; production callers
    let :func:`current_config_path` resolve it. Returns the path so the
    CLI can print "✓ Patched Cline config at <path>".
    """
    path = config_path or current_config_path()
    if path is None:
        raise FileNotFoundError(
            "Cline does not appear to be installed (no globalStorage dir found). "
            "Install Cline from the VS Code marketplace and try again."
        )

    existing = _common.load_json_lenient(path)
    _common.backup_existing(path)

    # ``server_url`` may or may not include the ``/v1`` suffix — match
    # what the user typed: if they said ``http://127.0.0.1:8000`` we
    # add ``/v1`` (Cline expects an OpenAI-compatible *base* URL that
    # ends in ``/v1``); if they already passed ``/v1`` we leave it
    # alone. Avoids accidentally producing ``/v1/v1``.
    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    existing["apiProvider"] = "openai"
    existing["openAiBaseUrl"] = base_url
    existing["openAiApiKey"] = api_key
    existing["openAiModelId"] = model

    _common.atomic_write_json(path, existing)
    return path
