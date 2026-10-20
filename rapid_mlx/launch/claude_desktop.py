# SPDX-License-Identifier: Apache-2.0
"""Opt-in Claude Desktop gateway mapping; never touches the client app."""

from __future__ import annotations

import json
from pathlib import Path

from rapid_mlx.agents.isolated import (
    _private_write,
    connection_dir,
    save_port,
    stable_key,
    stable_port,
)

TIERS = ("opus", "sonnet", "haiku")


def mapping_path() -> Path:
    return connection_dir() / "claude-desktop.json"


def mappings() -> dict[str, str]:
    """Return only a complete, explicitly enabled tier assignment."""
    path = mapping_path()
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or set(data) != set(TIERS):
        return {}
    if not all(isinstance(data[tier], str) and data[tier] for tier in TIERS):
        return {}
    return data


def resolve_model(requested: str) -> str:
    """Resolve only documented bare tiers and Rapid's discovered tier IDs."""
    for tier, model_id in mappings().items():
        if requested in (tier, f"claude-{tier}-rapid-mlx"):
            return model_id
    return requested


def configure(
    model: str,
    *,
    opus: str | None = None,
    sonnet: str | None = None,
    haiku: str | None = None,
    port: int | None = None,
) -> tuple[str, str]:
    """Save Rapid's mapping and return the in-app gateway values."""
    choices = {
        "opus": opus or model,
        "sonnet": sonnet or model,
        "haiku": haiku or model,
    }
    key = stable_key()
    assert key is not None
    if port is not None:
        save_port(port)
    _private_write(mapping_path(), json.dumps(choices, sort_keys=True) + "\n")
    return f"http://127.0.0.1:{port or stable_port()}", key
