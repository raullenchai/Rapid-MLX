# SPDX-License-Identifier: Apache-2.0
"""Owner-provided tool labels shared by terminal and Desktop clients."""

from urllib.parse import urlsplit


def tool_progress(name: str, args: dict) -> str:
    path = str(args.get("path") or ".")[:200]
    pattern = str(args.get("pattern", ""))[:120]
    if name == "read_file":
        return f"Reading {path}…"
    if name == "grep":
        return f"Searching for '{pattern}' in {path}…"
    if name == "glob":
        return f"Finding '{pattern}' in {path}…"
    if name == "delegate_task":
        return "Delegating tasks…"
    if name == "weather":
        return (
            f"Checking weather for {str(args.get('location', 'your location'))[:120]}…"
        )
    if name == "web_search":
        return f"Searching the web: {str(args.get('query', ''))[:150]}…"
    if name == "web_fetch":
        try:
            host = urlsplit(str(args.get("url", ""))).hostname or "web page"
        except ValueError:
            host = "web page"
        return f"Reading {host[:160]}…"
    return f"{name}…"
