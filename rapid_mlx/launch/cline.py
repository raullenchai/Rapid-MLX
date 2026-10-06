# SPDX-License-Identifier: Apache-2.0
"""Cline (Cline CLI and the VS Code extension) launch adapter.

Where Cline keeps its model provider
------------------------------------
Current Cline (CLI 3.x, VS Code extension 4.x) shares one data directory,
``~/.cline/data`` (``$CLINE_DATA_DIR``, else ``$CLINE_DIR/data``). Provider
settings live in ``settings/providers.json``, owned by the Cline SDK's
``ProviderSettingsManager``:

.. code-block:: json

   {
     "version": 1,
     "lastUsedProvider": "openai-compatible",
     "modes": {},
     "providers": {
       "openai-compatible": {
         "settings": {"provider": "openai-compatible", "apiKey": "...",
                      "model": "...", "baseUrl": "http://127.0.0.1:8000/v1"},
         "updatedAt": "2026-10-06T18:08:54.038Z",
         "tokenSource": "manual"
       }
     }
   }

That is exactly what ``cline auth -p openai -b <url> -k <key> -m <model>``
writes, and the Cline CLI reads it on every run. The VS Code extension uses
it as well, but its *selected* provider comes from its own UI state first
(``globalState.json``, which a running VS Code keeps in memory and rewrites),
so an extension that already has a provider picked must be switched in its
settings panel — we print the exact steps rather than editing live editor
state.

``cline_mcp_settings.json`` is Cline's MCP-server list; it never held model
provider settings, and this adapter does not touch it.

Cline treats a ``providers.json`` that fails its schema as empty and would
overwrite it on the next save, so we refuse to modify one we cannot parse
and keep the shape strict (version 1, ISO ``updatedAt`` in UTC with ``Z``).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import _common

# The OpenAI-compatible provider id in the Cline SDK (``cline auth -p openai``
# stores it under this id).
_PROVIDER_ID = "openai-compatible"

# Legacy VS Code extension id ("publisher.name"); the extension's
# ``globalStorage`` dir still exists for installs that predate the shared
# ``~/.cline`` directory, so it remains a detection signal.
_EXTENSION_ID = "saoudrizwan.claude-dev"


def _cline_data_dir() -> Path:
    """Resolve Cline's data dir the way the Cline SDK does
    (``resolveClineDataDir``): ``$CLINE_DATA_DIR``, else
    ``$CLINE_DIR/data``, else ``~/.cline/data``."""
    data_dir = os.environ.get("CLINE_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser()
    cline_dir = os.environ.get("CLINE_DIR", "").strip()
    root = Path(cline_dir).expanduser() if cline_dir else Path.home() / ".cline"
    return root / "data"


def _candidate_settings_roots() -> list[Path]:
    """VS Code (and forks') ``User/globalStorage`` roots, macOS first."""
    home = Path.home()
    return [
        home / "Library/Application Support/Code/User/globalStorage",
        home / "Library/Application Support/Code - Insiders/User/globalStorage",
        home / "Library/Application Support/VSCodium/User/globalStorage",
        home / ".config/Code/User/globalStorage",
        home / ".config/Code - Insiders/User/globalStorage",
        home / ".config/VSCodium/User/globalStorage",
    ]


def detect() -> bool:
    """Return True when the Cline CLI or the VS Code extension is present.

    Signals: ``cline`` on PATH, Cline's data directory, or the extension's
    ``globalStorage`` directory in a VS Code-family editor.
    """
    if _common.which("cline"):
        return True
    if _cline_data_dir().exists():
        return True
    return any((root / _EXTENSION_ID).exists() for root in _candidate_settings_roots())


def current_config_path() -> Path | None:
    """Return ``<cline data dir>/settings/providers.json``."""
    return _cline_data_dir() / "settings" / "providers.json"


def load_providers(path: Path) -> dict[str, Any]:
    """Read ``providers.json``; ``{}`` when missing or blank.

    Raises ``ValueError`` for anything Cline's schema would reject, so we
    never rewrite (and thereby wipe) a file we do not understand.
    """
    try:
        data = _common.load_json_lenient(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if data == {}:
        return {}
    if not _matches_cline_schema(data):
        raise ValueError(
            f"{path} is not a Cline providers file (version 1) this version of "
            "rapid-mlx understands; configure Cline with "
            "`cline auth -p openai -b <url> -k <key> -m <model>` instead"
        )
    return data


_TOKEN_SOURCES = {"manual", "oauth", "migration"}


def _matches_cline_schema(data: object) -> bool:
    """The structural part of Cline's ``StoredProviderSettingsSchema`` (zod)
    that a rewrite would carry forward: anything failing it is read by Cline
    as an empty file and would be lost on its next save."""
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    if type(version) is not int or version != 1:  # ``True == 1`` in Python
        return False
    if "lastUsedProvider" in data and not (
        isinstance(data["lastUsedProvider"], str) and data["lastUsedProvider"]
    ):
        return False
    for optional_mapping in ("modes", "repairs"):
        if optional_mapping in data and not isinstance(data[optional_mapping], dict):
            return False
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False
    for entry in providers.values():
        if not isinstance(entry, dict):
            return False
        settings = entry.get("settings")
        if not isinstance(settings, dict) or not isinstance(
            settings.get("provider"), str
        ):
            return False
        if not isinstance(entry.get("updatedAt"), str):
            return False
        if entry.get("tokenSource", "manual") not in _TOKEN_SOURCES:
            return False
    return True


def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _base_url(server_url: str) -> str:
    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return base_url


def patched_config(
    existing: dict[str, Any],
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    *,
    now: str | None = None,
) -> dict[str, Any]:
    """Return the ``providers.json`` we would write, without touching disk.

    Mirrors ``ProviderSettingsManager.saveProviderSettings``: the
    ``openai-compatible`` entry gets our base URL / key / model (other
    fields in that entry, e.g. custom headers, are kept), becomes
    ``lastUsedProvider``, and every other provider is preserved. When the
    entry already matches, the input is returned unchanged so a re-run is a
    no-op (no new ``updatedAt``).
    """
    # A new file follows Cline's own key order.
    result = (
        dict(existing)
        if existing
        else {
            "version": 1,
            "lastUsedProvider": _PROVIDER_ID,
            "modes": {},
            "providers": {},
        }
    )
    result.setdefault("modes", {})
    providers = dict(result.get("providers") or {})
    previous = providers.get(_PROVIDER_ID)
    previous = previous if isinstance(previous, dict) else {}
    previous_settings = previous.get("settings")
    previous_settings = previous_settings if isinstance(previous_settings, dict) else {}
    settings = {
        **previous_settings,
        "provider": _PROVIDER_ID,
        "apiKey": api_key,
        "model": model,
        "baseUrl": _base_url(server_url),
    }
    if (
        settings == previous_settings
        and result.get("lastUsedProvider") == _PROVIDER_ID
        and result == existing
    ):
        return existing
    providers[_PROVIDER_ID] = {
        "settings": settings,
        "updatedAt": now or _now_iso(),
        "tokenSource": previous.get("tokenSource", "manual"),
    }
    result["providers"] = providers
    result["lastUsedProvider"] = _PROVIDER_ID
    return result


def post_setup_notes(server_url: str, model: str, api_key: str | None) -> list[str]:
    """Lines ``rapid-mlx launch cline`` prints after configuring Cline,
    including the exact settings-panel steps for the VS Code extension."""
    key = "your RAPID_MLX_API_KEY value" if api_key else "any value, e.g. sk-noop"
    return [
        'Cline CLI: ready — run `cline "<task>"`.',
        "Cline in VS Code: if Cline already has a provider selected, open Cline "
        "> Settings (gear icon) > API Configuration and set:",
        "    API Provider: OpenAI Compatible",
        f"    Base URL:     {_base_url(server_url)}",
        f"    API Key:      {key}",
        f"    Model ID:     {model}",
    ]


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Point Cline's ``openai-compatible`` provider at the rapid-mlx server
    and make it the last-used provider.

    Idempotent (an already-matching file is not rewritten or backed up);
    otherwise the existing file is backed up to ``providers.json.bak.<ts>``
    and replaced atomically with owner-only permissions, as Cline does.
    """
    path = config_path or current_config_path()
    assert path is not None
    # Write through a symlinked providers.json (e.g. into a dotfiles repo)
    # instead of replacing the link with a regular file.
    path = path.resolve()
    existing = load_providers(path)
    updated = patched_config(existing, server_url, model, api_key)
    if updated is existing and path.exists():
        return path
    _common.backup_existing(path)
    _common.atomic_write_json(path, updated)
    return path


def preview(
    server_url: str, model: str, api_key: str = "sk-noop"
) -> tuple[Path, str, tuple[str, ...]]:
    """``(path, redacted diff, notes)`` for ``launch --dry-run``; no writes."""
    path = current_config_path()
    assert path is not None
    existing = load_providers(path)
    updated = patched_config(existing, server_url, model, api_key)
    if updated is existing and path.exists():
        return path, "", ()

    def render(data: dict[str, Any]) -> str:
        if not data:
            return ""
        return json.dumps(_common.redact_secrets(data), indent=2)

    return path, _common.unified_diff(render(existing), render(updated), path), ()
