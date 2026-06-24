# SPDX-License-Identifier: Apache-2.0
"""Claude Code CLI launch adapter.

Unlike the VS Code-extension clients (Cline, Continue), the Claude Code
CLI ships as a Node-based command-line tool (``claude``) that reads its
config from ``~/.config/claude/settings.json`` and also honours a small
set of environment variables — most importantly
``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY``.

rapid-mlx already serves the Anthropic ``/v1/messages`` shape (see the
README "Claude Code" section), so the launch step is simply:

1. Write a ``settings.json`` that points Claude Code at our local
   ``http://127.0.0.1:8000`` (NOT ``/v1`` — Anthropic SDK appends the
   ``/v1/messages`` path itself; double-slashing yields a 404).
2. Carry the model id so a fresh Claude Code session uses our loaded
   model rather than the SDK default.

Detection probes both an installed ``claude`` binary on PATH AND a
``~/.claude`` state dir — either alone is good enough since some
package managers (npm-global, brew, asdf) install the binary outside
the standard ``~/.config`` tree before the first run.
"""

from __future__ import annotations

from pathlib import Path

from . import _common

# Per-user state dir Anthropic's CLI creates on first run. We probe for
# it as a "claude has been launched at least once" signal — a fallback
# when the binary lives outside our PATH visibility (e.g. inside a
# Volta or asdf shim that's not in the parent shell's PATH).
_CLAUDE_STATE_DIR = Path.home() / ".claude"

# Canonical settings location. Anthropic's docs document this path on
# both macOS and Linux; the CLI mkdir's it on first launch.
_CONFIG_DIR = Path.home() / ".config" / "claude"
_CONFIG_FILENAME = "settings.json"


def detect() -> bool:
    """Return True when Claude Code CLI is plausibly installed.

    Three independent signals — any one is sufficient:

    * The ``claude`` executable resolves on PATH.
    * ``~/.claude`` exists (the CLI's state dir from a previous run).
    * ``~/.config/claude/`` exists (the config dir, mkdir'd on first
      launch on freshly installed boxes).

    Multiple signals exist because users install the CLI through wildly
    different package managers (npm-global, brew, official installer)
    and each one writes to a different binary path. A single-signal
    check would false-negative on every box where the binary's outside
    our PATH visibility.
    """
    if _common.which("claude") is not None:
        return True
    if _CLAUDE_STATE_DIR.exists():
        return True
    if _CONFIG_DIR.exists():
        return True
    return False


def current_config_path() -> Path | None:
    """Return ``~/.config/claude/settings.json``.

    Unlike Cline (where we *might* refuse to write when the extension
    isn't installed), Claude Code's settings file is safe to create
    speculatively — the CLI mkdir's the same path on first run, so
    pre-creating it is identical to letting the CLI do it. We always
    return the canonical path; the launch dispatcher prints a "Claude
    Code not detected" hint when :func:`detect` is False but proceeds
    with the patch on user override (``--force``, not yet implemented;
    today we just patch on best-effort).
    """
    return _CONFIG_DIR / _CONFIG_FILENAME


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Patch ``~/.config/claude/settings.json`` to route at the local
    rapid-mlx Anthropic-compatible endpoint.

    Keys we own (all under the top-level ``env`` object, which Anthropic
    documents as the recommended way to feed Claude Code env vars from
    the settings file rather than the parent shell):

    * ``env.ANTHROPIC_BASE_URL`` → ``<server_url>`` (no ``/v1`` —
      Anthropic SDK joins paths itself; ``http://127.0.0.1:8000`` is
      correct, ``http://127.0.0.1:8000/v1`` produces 404 on
      ``/v1/v1/messages``).
    * ``env.ANTHROPIC_API_KEY`` → ``<api_key>``
    * ``env.ANTHROPIC_MODEL`` → ``<model>`` (Claude Code reads this as
      the default model id; the rapid-mlx server accepts any
      ``claude-*`` model name and routes to the actually-loaded engine,
      so this is informational rather than enforced).

    All other keys (``permissions``, ``apiKeyHelper``, ``mcp_servers``,
    custom shortcuts) round-trip untouched.
    """
    path = config_path or current_config_path()
    assert path is not None  # current_config_path never returns None here

    existing = _common.load_json_lenient(path)
    _common.backup_existing(path)

    # Strip a trailing ``/v1`` if the user pasted one — the Anthropic
    # SDK appends ``/v1/messages`` itself, and ``/v1/v1/messages`` is a
    # 404. This is the inverse of Cline's behaviour (which *requires*
    # the ``/v1`` suffix), so we explicitly normalise here rather than
    # share the logic.
    base_url = server_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]

    env = dict(existing.get("env", {}))
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_API_KEY"] = api_key
    env["ANTHROPIC_MODEL"] = model
    existing["env"] = env

    _common.atomic_write_json(path, existing)
    return path
