# SPDX-License-Identifier: Apache-2.0
"""Security banner shown before printing share URL+key.

We want this loud enough that nobody accidentally tweets a screenshot
of their share URL and bearer key.
"""

from __future__ import annotations

import sys


def _supports_color() -> bool:
    return sys.stdout.isatty() and not sys.platform.startswith("win")


def render(url: str, api_key: str, model: str) -> str:
    red = "\033[1;31m" if _supports_color() else ""
    yellow = "\033[1;33m" if _supports_color() else ""
    reset = "\033[0m" if _supports_color() else ""
    bold = "\033[1m" if _supports_color() else ""

    return (
        f"\n{red}╔══════════════════════════════════════════════════════════════════╗{reset}\n"
        f"{red}║  ⚠  PUBLIC INTERNET — read this before sharing                   ║{reset}\n"
        f"{red}╠══════════════════════════════════════════════════════════════════╣{reset}\n"
        f"{red}║{reset} rapid-mlx share is now exposing this Mac to the public internet.{red} ║{reset}\n"
        f"{red}║{reset} Anyone who has the API key below can:                          {red}  ║{reset}\n"
        f"{red}║{reset}   • use your compute (free inference on your bill)              {red} ║{reset}\n"
        f"{red}║{reset}   • see every prompt and response that goes through              {red}║{reset}\n"
        f"{red}║{reset}                                                                  {red}║{reset}\n"
        f"{red}║{reset} Do NOT screenshot, paste, or commit this key. Ctrl-C stops it.  {red} ║{reset}\n"
        f"{red}╚══════════════════════════════════════════════════════════════════╝{reset}\n"
        f"\n"
        f"  {bold}Model:{reset}  {model}\n"
        f"  {bold}URL:{reset}    {url}\n"
        f"  {bold}Key:{reset}    {yellow}{api_key}{reset}\n"
        f"\n"
        f"  Test it:\n"
        f"    curl -sS {url}/v1/chat/completions \\\n"
        f"      -H 'Authorization: Bearer {api_key}' \\\n"
        f"      -H 'Content-Type: application/json' \\\n"
        f"      -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'\n"
        f"\n"
        f"  Press Ctrl-C to stop sharing.\n"
    )
