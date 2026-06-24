# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx launch <client>`` — one-shot bootstrap.

Detects whether the named client (Cline, Claude Code CLI, Continue,
Cursor) is installed on this machine, then writes/patches the client's
local config so it routes traffic at the local rapid-mlx OpenAI-
compatible server (default ``http://127.0.0.1:8000/v1``). Optionally
spawns ``rapid-mlx serve`` in the background so a user goes from a
fresh install to "Cline talking to my Mac" in one command.

The implementation lives in per-client modules so each adapter's
config-shape knowledge stays narrow:

* :mod:`vllm_mlx.launch.cline` — Cline VS Code extension
* :mod:`vllm_mlx.launch.claude_code` — Claude Code CLI (Anthropic SDK)
* :mod:`vllm_mlx.launch.continue_dev` — Continue.dev VS Code/JetBrains
* :mod:`vllm_mlx.launch.cursor` — Cursor editor

All adapters expose the same surface (:func:`detect`,
:func:`current_config_path`, :func:`write_or_patch_config`) so the
top-level ``launch`` dispatcher in :mod:`vllm_mlx.launch.cli` can route
to them via a single registry. See ``cli.py`` in this package for the
argparse wiring and the ``--start-server`` background-serve handling.

See GitHub issue #566 for motivation (the Ollama ``ollama launch
cline`` shape we're copying — same OpenAI-compatible plumbing, same
one-verb UX).
"""

from . import claude_code, cline, continue_dev, cursor

# Registry consumed by ``vllm_mlx.launch.cli`` — order is the
# display order in ``rapid-mlx launch list``. Keys are the
# user-facing client names accepted on the CLI (kebab-case so
# ``claude-code`` matches Cline's blog post / Cursor's settings panel
# shape; ``continue`` would collide with the Python keyword so we use
# ``continue-dev``).
ADAPTERS: dict[str, object] = {
    "cline": cline,
    "claude-code": claude_code,
    "continue-dev": continue_dev,
    "cursor": cursor,
}

__all__ = ["ADAPTERS", "claude_code", "cline", "continue_dev", "cursor"]
