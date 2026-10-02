"""Validated source of truth for repositories intentionally absent from R2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UnmirroredEntry:
    hf_path: str
    reason: str
    since: str


def _catalog_hf_paths(main_aliases_path: Path, audio_aliases_path: Path) -> set[str]:
    paths: set[str] = set()
    for path, key in (
        (main_aliases_path, "hf_path"),
        (audio_aliases_path, "hf_id"),
    ):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        for profile in payload.values():
            if not isinstance(profile, dict):
                continue
            hf_path = profile.get(key)
            if isinstance(hf_path, str):
                paths.add(hf_path)
    return paths


def load_unmirrored(
    path: Path, main_aliases_path: Path, audio_aliases_path: Path
) -> dict[str, UnmirroredEntry]:
    """Load and fully validate the intentional-unmirror registry."""
    payload: Any = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "entries"}:
        raise ValueError(
            f"{path} must contain exactly the keys schema_version and entries"
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError(f"{path} schema_version must be 1")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError(f"{path} entries must be a JSON array")

    known_hf_paths = _catalog_hf_paths(main_aliases_path, audio_aliases_path)
    entries: dict[str, UnmirroredEntry] = {}
    expected_keys = {"hf_path", "reason", "since"}
    for index, raw_entry in enumerate(raw_entries):
        label = f"{path} entries[{index}]"
        if not isinstance(raw_entry, dict) or set(raw_entry) != expected_keys:
            raise ValueError(
                f"{label} must contain exactly the keys hf_path, reason, and since"
            )
        values = {key: raw_entry[key] for key in expected_keys}
        for key, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
        hf_path = values["hf_path"]
        if hf_path in entries:
            raise ValueError(f"{label} has duplicate hf_path {hf_path!r}")
        since = values["since"]
        try:
            parsed_since = date.fromisoformat(since)
        except ValueError as error:
            raise ValueError(
                f"{label}.since must be an ISO date (YYYY-MM-DD)"
            ) from error
        if parsed_since.isoformat() != since:
            raise ValueError(f"{label}.since must be an ISO date (YYYY-MM-DD)")
        if hf_path not in known_hf_paths:
            raise ValueError(
                f"{label}.hf_path {hf_path!r} is not present in the alias catalogs"
            )
        entries[hf_path] = UnmirroredEntry(
            hf_path=hf_path,
            reason=values["reason"],
            since=since,
        )
    return entries
