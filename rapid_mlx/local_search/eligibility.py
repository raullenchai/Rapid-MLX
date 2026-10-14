# SPDX-License-Identifier: Apache-2.0
"""Deterministic semantic eligibility; excluded content remains searchable in FTS."""

import re
from pathlib import Path

ELIGIBILITY_VERSION = "1"
# Data/configuration exports are literal lookups, irrespective of their content.
KEYWORD_ONLY_SUFFIXES = frozenset(
    {
        ".json",
        ".jsonl",
        ".ndjson",
        ".csv",
        ".tsv",
        ".log",
        ".xml",
        ".lock",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".plist",
        ".map",
    }
)


def semantic_format(path: Path) -> bool:
    return path.suffix.lower() not in KEYWORD_ONLY_SUFFIXES


def low_information(text: str) -> bool:
    """Conservative shape checks, without a dictionary or language model.

    Detect dominant long ASCII encodings, compact JSON-shaped data, and long
    diverse runs of tiny ASCII words without punctuation. Ordinary source syntax
    and non-ASCII letters cannot trigger the tiny-word rule. These checks detect
    obvious noise, not all meaningless text; short/ambiguous text is retained.
    """
    if len(text) < 128:
        return False
    lines = text.splitlines()
    encoded = (
        "".join(lines)
        if len(lines) > 1 and all(len(line) >= 64 for line in lines[:-1])
        else text
    )
    blobs = re.findall(
        r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}(?![A-Za-z0-9+/])", encoded
    )
    if sum(map(len, blobs)) > len(text) / 2:
        return True
    compact = re.sub(r"\s+", "", text)
    # Quoted values/keys and JSON delimiters dominate; normal minified source
    # has operators and unquoted identifiers and does not meet this condition.
    data = re.findall(
        r'"(?:[^"\\]|\\.)*"|[\[\]{},:]|\b(?:-?\d+(?:\.\d+)?|true|false|null)\b', compact
    )
    if len(compact) >= 256 and len(compact) > 0.98 * len(text) and ":" in compact:
        if sum(map(len, data)) >= 0.9 * len(compact):
            return True
    if re.fullmatch(r"[A-Za-z\s]+", text):
        words = text.lower().split()
        if len(words) >= 32:
            return (
                sum(len(w) <= 3 for w in words) >= 0.8 * len(words)
                and sum(len(w) <= 2 for w in words) >= 0.5 * len(words)
                and len(set(words)) >= 0.8 * len(words)
            )
    return False


def semantic_chunk(path: Path, text: str) -> bool:
    return semantic_format(path) and not low_information(text)
