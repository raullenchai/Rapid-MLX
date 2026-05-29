# SPDX-License-Identifier: Apache-2.0
"""System-prompt canonicalization helpers."""

from __future__ import annotations

import re

_STRIPPERS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "anthropic_billing_header",
        re.compile(r"(?im)^x-anthropic-billing-header:[^\n]*(?:\n|$)"),
        "",
    ),
)


def canonicalize_system_prompt(text: str | None) -> str | None:
    """Remove known non-semantic volatile lines from system prompt text."""
    if text is None:
        return None

    for _name, pattern, replacement in _STRIPPERS:
        text = pattern.sub(replacement, text)
    return text


def canonicalize_system_messages(messages: list[dict]) -> list[dict]:
    """Canonicalize string content on system-role messages without mutation."""
    canonicalized: list[dict] = []
    changed = False
    for message in messages:
        if message.get("role") != "system":
            canonicalized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, str):
            canonicalized.append(message)
            continue

        next_content = canonicalize_system_prompt(content)
        if next_content == content:
            canonicalized.append(message)
            continue

        changed = True
        next_message = message.copy()
        next_message["content"] = next_content
        canonicalized.append(next_message)

    return canonicalized if changed else messages
