# SPDX-License-Identifier: Apache-2.0
"""Bounded text for terminal display and client identity capture."""


def unsafe(char):
    code = ord(char)
    return (
        code < 32
        or 127 <= code <= 159
        or 0x202A <= code <= 0x202E
        or 0x2066 <= code <= 0x2069
    )


def display(value, limit=256):
    text = "".join(
        (f"\\x{ord(c):02x}" if ord(c) <= 255 else f"\\u{ord(c):04x}")
        if unsafe(c)
        else c
        for c in str(value)[:limit]
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."


def client_identity(value):
    return "".join(c for c in value if not unsafe(c))[:256]
