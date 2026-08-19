# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx launch <client>`` — one-shot bootstrap.

Detects whether the named client (Cline or the Claude Code CLI) is
installed on this machine, then writes/patches the client's local
config so it routes traffic at the local rapid-mlx OpenAI-compatible
server (default ``http://127.0.0.1:8000/v1``). Optionally spawns
``rapid-mlx serve`` in the background so a user goes from a fresh
install to "Cline talking to my Mac" in one command.

Membership in :data:`ADAPTERS` is a factual claim: *this client reads a
documented config file, and writing that file makes the client use our
server*. Two clients were removed for failing it:

* **Cursor** — its provider settings are fields inside one large
  reactive-storage JSON blob in ``state.vscdb`` plus a macOS keychain
  entry, and the ``cursor.aiprovider.*`` keys the adapter used to write
  do not exist in Cursor's schema at all. It is now an adapter profile
  (``rapid-mlx agents cursor``) that prints GUI setup steps.
* **Continue.dev** — the upstream project went read-only on
  2026-06-19. Kilo Code (``rapid-mlx agents kilo-code``) is the
  maintained successor and speaks the same OpenAI-compatible API.

The implementation lives in per-client modules so each adapter's
config-shape knowledge stays narrow:

* :mod:`vllm_mlx.launch.cline` — Cline VS Code extension
* :mod:`vllm_mlx.launch.claude_code` — Claude Code CLI (Anthropic SDK)

All adapters expose the same surface (:func:`detect`,
:func:`current_config_path`, :func:`write_or_patch_config`) so the
top-level ``launch`` dispatcher in :mod:`vllm_mlx.launch.cli` can route
to them via a single registry. See ``cli.py`` in this package for the
argparse wiring and the ``--start-server`` background-serve handling.

See GitHub issue #566 for motivation (the Ollama ``ollama launch
cline`` shape we're copying — same OpenAI-compatible plumbing, same
one-verb UX).
"""

from . import claude_code, cline

# Registry consumed by ``vllm_mlx.launch.cli`` — order is the
# display order in ``rapid-mlx launch list``. Keys are the
# user-facing client names accepted on the CLI (kebab-case so
# ``claude-code`` matches the client's common command name).
ADAPTERS: dict[str, object] = {
    "cline": cline,
    "claude-code": claude_code,
}

__all__ = ["ADAPTERS", "claude_code", "cline"]
