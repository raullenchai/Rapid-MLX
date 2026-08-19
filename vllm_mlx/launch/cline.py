# SPDX-License-Identifier: Apache-2.0
"""Cline (VS Code extension) launch adapter.

Cline's provider configuration does NOT live in VS Code's
``globalStorage`` — that tree only holds ``cline_mcp_settings.json``,
which is the MCP *server* list and nothing else. Writing
``openAiBaseUrl`` there (as this adapter used to) is a silent no-op:
Cline never reads those keys back, so ``rapid-mlx launch cline``
appeared to succeed while the extension kept talking to whatever
provider it was already on.

The real store is a file-backed tree under ``~/.cline/data`` (override
chain: ``CLINE_DATA_DIR`` → ``$CLINE_DIR/data`` → ``~/.cline/data``).
Verified against Cline 4.1.10 by reading the shipped bundles.

Complicating things, the extension ships **two** bundles and decides
which to activate at runtime from a remote feature flag
(``ext-sdk-bundle-rollout``), cached in globalStorage. A user can be
flipped between them by a server-side rollout with no local change, so
we write BOTH shapes and let whichever bundle wins read its own:

* ``legacy`` — a shim that reimplements VS Code's globalState/secrets
  API over plain JSON files:

  - ``~/.cline/data/globalState.json`` — ``planModeApiProvider`` /
    ``actModeApiProvider`` set to ``"openai"``, plus ``openAiBaseUrl``
    and the per-mode ``{plan,act}ModeOpenAiModelId``.
  - ``~/.cline/data/secrets.json`` (0600) — ``openAiApiKey``.

* ``next`` — a single ``~/.cline/data/settings/providers.json`` keyed
  by provider id, where a custom OpenAI-compatible endpoint is the
  ``openai-compatible`` provider.

Both writes preserve every key we don't own, so a user's task history,
auto-approve settings, MCP list and other providers survive a relaunch.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from . import _common

# Provider id both bundles agree on for "an OpenAI-compatible endpoint
# that isn't api.openai.com". The next bundle also accepts ``openai``
# but aliases it straight to this, so we write the canonical spelling.
_PROVIDER_ID = "openai-compatible"

# VS Code extension id ("publisher.name"). Cline's stable id is
# ``saoudrizwan.claude-dev`` (the project predates the rename to
# "Cline" and the extension id never changed). We no longer *write*
# anything under globalStorage, but its per-extension directory is
# still the most reliable "is Cline installed" signal.
_EXTENSION_ID = "saoudrizwan.claude-dev"


def _data_dir() -> Path:
    """Resolve Cline's data root the same way both bundles do.

    ``CLINE_DATA_DIR`` wins outright; otherwise the root is
    ``$CLINE_DIR/data`` falling back to ``~/.cline/data``. Honouring
    the env vars matters for users who relocate the tree off a synced
    home directory — patching the default path would leave their real
    config untouched.
    """
    explicit = (os.environ.get("CLINE_DATA_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    cline_dir = (os.environ.get("CLINE_DIR") or "").strip()
    root = Path(cline_dir) if cline_dir else Path.home() / ".cline"
    return root / "data"


def _candidate_extension_dirs() -> list[Path]:
    """Per-OS list of the Cline extension dir inside each VS Code fork.

    Order is "most likely first". macOS paths come first because that's
    the platform rapid-mlx targets (Apple Silicon); Linux paths follow
    so CI / dev containers still detect a configured Cline install.
    """
    home = Path.home()
    return [
        # The extension itself, installed by any VS Code flavour.
        home / ".vscode/extensions",
        home / ".vscode-insiders/extensions",
        home / ".vscode-oss/extensions",
        # macOS — VS Code Stable, Insiders, VSCodium globalStorage.
        home / "Library/Application Support/Code/User/globalStorage",
        home / "Library/Application Support/Code - Insiders/User/globalStorage",
        home / "Library/Application Support/VSCodium/User/globalStorage",
        # Linux — same three flavours under ~/.config.
        home / ".config/Code/User/globalStorage",
        home / ".config/Code - Insiders/User/globalStorage",
        home / ".config/VSCodium/User/globalStorage",
    ]


def detect() -> bool:
    """Return True when Cline appears installed on this machine.

    Two independent signals, either of which is sufficient:

    * ``~/.cline/data`` exists — Cline has run at least once, so the
      store we patch is definitely the one it reads.
    * a VS Code-family tree contains the extension (either the unpacked
      ``saoudrizwan.claude-dev-<version>`` directory or its
      globalStorage) — Cline is installed but may never have been
      opened, in which case we create the data tree ourselves.
    """
    if _data_dir().exists():
        return True
    for root in _candidate_extension_dirs():
        if not root.is_dir():
            continue
        # globalStorage uses the bare id; the extensions dir suffixes a
        # version, so match on the prefix.
        if (root / _EXTENSION_ID).exists():
            return True
        try:
            if any(p.name.startswith(_EXTENSION_ID + "-") for p in root.iterdir()):
                return True
        except OSError:
            continue
    return False


def current_config_path() -> Path | None:
    """Return the path we report as "the" Cline config, or None.

    We touch three files; the launch CLI's ``--dry-run`` output and the
    "Patched cline config at <path>" line want a single representative,
    so we name ``globalState.json`` — the one the currently-default
    (legacy) bundle actually reads. Returns a path even when nothing
    exists yet, provided :func:`detect` says Cline is installed, since
    we create the tree on write.
    """
    if not detect():
        return None
    return _data_dir() / "globalState.json"


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Point Cline at the local rapid-mlx OpenAI-compatible server.

    Writes all three files both bundles could read (see the module
    docstring) so a remote rollout flipping the user between bundles
    doesn't silently unconfigure them.

    The ``config_path`` arg is a test/dry-run hook naming the data
    *directory*'s ``globalState.json``; production callers let
    :func:`current_config_path` resolve it. Returns that path so the
    CLI can print "Patched cline config at <path>".
    """
    global_state_path = config_path or current_config_path()
    if global_state_path is None:
        raise FileNotFoundError(
            "Cline does not appear to be installed (no ~/.cline/data and no "
            "VS Code extension found). Install Cline from the VS Code "
            "marketplace, open it once, and try again."
        )
    data_dir = global_state_path.parent

    # ``server_url`` may or may not include the ``/v1`` suffix — match
    # what the user typed: if they said ``http://127.0.0.1:8000`` we
    # add ``/v1`` (Cline expects an OpenAI-compatible *base* URL that
    # ends in ``/v1``); if they already passed ``/v1`` we leave it
    # alone. Avoids accidentally producing ``/v1/v1``.
    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    # Parse every document before changing any of them. Cline can switch
    # between these two bundle shapes remotely, so a malformed providers.json
    # must not leave the already-written legacy files pointing at Rapid while
    # the command reports failure (and vice versa).
    state = _load_mapping(data_dir / "globalState.json")
    secrets = _load_mapping(data_dir / "secrets.json")
    providers = _load_mapping(data_dir / "settings" / "providers.json")

    _write_legacy(data_dir, base_url, model, api_key, state, secrets)
    _write_next(data_dir, base_url, model, api_key, providers)
    return global_state_path


def _load_mapping(path: Path) -> dict:
    """Load one Cline document and reject non-object JSON before any write."""
    value = _common.load_json_lenient(path)
    if not isinstance(value, dict):
        raise ValueError(f"Cline config at {path} must contain a JSON object")
    return value


def _write_legacy(
    data_dir: Path,
    base_url: str,
    model: str,
    api_key: str,
    state: dict,
    secrets: dict,
) -> None:
    """Patch the legacy bundle's globalState.json + secrets.json.

    Cline keeps *separate* provider selections for Plan and Act mode, so
    both ``planMode*`` and ``actMode*`` have to be set or the user gets
    rapid-mlx in one mode and their old provider in the other.

    ``welcomeViewCompleted`` is forced true: a fresh install leaves it
    unset and the onboarding pane sits over the chat, hiding the
    configuration we just wrote and inviting the user to pick a cloud
    provider (which would overwrite it).
    """
    path = data_dir / "globalState.json"
    _common.backup_existing(path)

    state["planModeApiProvider"] = "openai"
    state["actModeApiProvider"] = "openai"
    state["openAiBaseUrl"] = base_url
    state["planModeOpenAiModelId"] = model
    state["actModeOpenAiModelId"] = model
    state["welcomeViewCompleted"] = True

    _common.atomic_write_json(path, state)

    # The API key is a secret in Cline's model, stored in a separate
    # 0600 file. ``atomic_write_json`` creates via mkstemp, which is
    # already 0600, so the mode matches what Cline itself writes.
    secrets_path = data_dir / "secrets.json"
    _common.backup_existing(secrets_path)
    secrets["openAiApiKey"] = api_key
    _common.atomic_write_json(secrets_path, secrets)


def _write_next(
    data_dir: Path, base_url: str, model: str, api_key: str, doc: dict
) -> None:
    """Patch the next bundle's settings/providers.json.

    The file is validated with a strict zod schema on read: ``version``
    must be the literal ``1`` and ``updatedAt`` must parse as RFC3339,
    or Cline discards the *whole* file and falls back to defaults —
    taking the user's other providers with it. So we pin both, and we
    only ever add/replace our own provider entry.
    """
    path = data_dir / "settings" / "providers.json"
    _common.backup_existing(path)

    doc["version"] = 1
    providers = doc.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    existing = providers.get(_PROVIDER_ID)
    settings = existing.get("settings", {}) if isinstance(existing, dict) else {}
    if not isinstance(settings, dict):
        settings = {}
    settings.update(
        {
            "provider": _PROVIDER_ID,
            "apiKey": api_key,
            "model": model,
            "baseUrl": base_url,
        }
    )
    providers[_PROVIDER_ID] = {
        "settings": settings,
        # ``timespec="seconds"`` because zod's ``.datetime()`` rejects
        # the 6-digit microsecond precision Python emits by default.
        "updatedAt": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "tokenSource": "manual",
    }
    doc["providers"] = providers
    doc["lastUsedProvider"] = _PROVIDER_ID

    _common.atomic_write_json(path, doc)
