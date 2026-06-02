# SPDX-License-Identifier: Apache-2.0
"""Security banner shown before printing share URL+key.

We want this loud enough that nobody accidentally tweets a screenshot
of their share URL and bearer key.
"""

from __future__ import annotations

import json
import shlex
import sys


def _supports_color() -> bool:
    return sys.stdout.isatty() and not sys.platform.startswith("win")


def render(
    url: str,
    api_key: str,
    model: str,
    subdomain: str,
    chat_frontend: str | None,
) -> str:
    red = "\033[1;31m" if _supports_color() else ""
    yellow = "\033[1;33m" if _supports_color() else ""
    reset = "\033[0m" if _supports_color() else ""
    bold = "\033[1m" if _supports_color() else ""

    # Quote every interpolated value before it lands in the copy-paste
    # curl line. The key NEVER appears in argv: it goes into an env var
    # the user exports first, so shell history doesn't capture it inline.
    # (Bonus: most shells ignore leading-space lines via HISTCONTROL, so
    # the export itself can also stay out of history — documented below.)
    safe_url = shlex.quote(f"{url}/v1/chat/completions")
    safe_body = shlex.quote(
        json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    )
    # One-click chat link. ``#k=<sub>.<key>`` — the configured chat
    # frontend's splash splits on the ``.`` delimiter, derives
    # ``https://<sub>.rapidmlx.com``, and shows it for confirmation before
    # sending the key. The combined token sits in the URL fragment so it
    # never reaches the server log; the splash immediately clears it via
    # history.replaceState.
    #
    # Codex round 1 BLOCKING: an earlier ``<sub>-<key>`` format was
    # ambiguous because the relay's subdomain charset permits ``-`` —
    # ``foo-bar-<key>`` would parse to subdomain ``foo`` + key
    # ``bar-<key>``. ``.`` is forbidden in both the subdomain charset
    # (``[a-z0-9-]``) and the API key (hex from ``secrets.token_hex``),
    # so it cleanly separates the two even for hyphenated subdomains.
    #
    # ``chat_frontend`` is None when the user opted out via
    # ``--chat-frontend ""`` (e.g. pointing at OpenWebUI which doesn't
    # implement the splash protocol). In that case we omit the Chat:
    # line entirely — the URL + Key lines below are all the user needs
    # to wire up an arbitrary OpenAI-compatible frontend by hand.
    if chat_frontend:
        chat_link = f"{chat_frontend}/#k={subdomain}.{api_key}"
        chat_line = f"  {bold}Chat:{reset}   {yellow}{chat_link}{reset}\n"
    else:
        chat_line = ""

    return (
        f"\n{red}╔══════════════════════════════════════════════════════════════════╗{reset}\n"
        f"{red}║  ⚠  PUBLIC INTERNET — read this before sharing                   ║{reset}\n"
        f"{red}╠══════════════════════════════════════════════════════════════════╣{reset}\n"
        f"{red}║{reset} rapid-mlx share is now exposing this machine to the public      {red} ║{reset}\n"
        f"{red}║{reset} internet. Anyone who has the API key below can:                {red} ║{reset}\n"
        f"{red}║{reset}   • use your compute (free inference on your bill)              {red} ║{reset}\n"
        f"{red}║{reset}   • see every prompt and response that goes through              {red}║{reset}\n"
        f"{red}║{reset}                                                                  {red}║{reset}\n"
        f"{red}║{reset} Do NOT screenshot, paste, or commit this key. Ctrl-C stops it.  {red} ║{reset}\n"
        f"{red}╚══════════════════════════════════════════════════════════════════╝{reset}\n"
        f"\n"
        f"  {bold}Model:{reset}  {model}\n"
        f"{chat_line}"
        f"  {bold}URL:{reset}    {url}\n"
        f"  {bold}Key:{reset}    {yellow}{api_key}{reset}\n"
        f"\n"
        f"  Test it (key stays out of shell history via env-var):\n"
        f"    export RAPID_MLX_SHARE_KEY={yellow}<paste-key>{reset}\n"
        f"    curl -sS {safe_url} \\\n"
        f'      -H "Authorization: Bearer $RAPID_MLX_SHARE_KEY" \\\n'
        f"      -H 'Content-Type: application/json' \\\n"
        f"      -d {safe_body}\n"
        f"\n"
        f"  Press Ctrl-C to stop sharing.\n"
    )
