# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx models --json`` — machine-readable output that replaces
scraping the fixed-width text table (which the desktop app and the DSH
provider both otherwise parse by column).

Structure, not values: aliases and cache contents change per release and per
machine, so the tests pin the SHAPE (keys, types, section split) and that the
command emits a single valid JSON document with no banner leaking onto stdout.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from vllm_mlx.cli import (
    _available_models_json_payload,
    _cached_models_json_payload,
    models_command,
)


def test_available_payload_shape() -> None:
    payload = _available_models_json_payload()
    assert set(payload) == {"text", "audio", "video", "image"}
    assert all(isinstance(payload[k], list) for k in payload)
    # There is always at least one text alias in the registry.
    assert payload["text"], "expected at least one text alias"
    entry = payload["text"][0]
    for key in (
        "alias",
        "hf_path",
        "size_bytes",
        "tool_call_parser",
        "reasoning_parser",
        "is_hybrid",
        "is_moe",
        "supports_spec_decode",
        "modality",
    ):
        assert key in entry, f"text entry missing {key!r}"
    assert isinstance(entry["is_hybrid"], bool)
    assert isinstance(entry["is_moe"], bool)
    assert entry["size_bytes"] is None or isinstance(entry["size_bytes"], int)


def test_available_sections_are_split_by_modality() -> None:
    payload = _available_models_json_payload()
    # No text alias should carry a generative modality, and vice versa.
    assert all(e["modality"] not in ("video-gen", "image-gen") for e in payload["text"])
    assert all(e["modality"] == "video-gen" for e in payload["video"])
    assert all(e["modality"] == "image-gen" for e in payload["image"])
    assert all(e["modality"] == "audio" for e in payload["audio"])


def test_cached_payload_shape() -> None:
    payload = _cached_models_json_payload()
    assert set(payload) == {"cached", "count", "total_bytes"}
    assert isinstance(payload["cached"], list)
    assert payload["count"] == len(payload["cached"])
    assert isinstance(payload["total_bytes"], int)
    # Biggest-first ordering.
    sizes = [m["size_bytes"] for m in payload["cached"]]
    assert sizes == sorted(sizes, reverse=True)
    for m in payload["cached"]:
        assert set(m) >= {"alias", "repo", "size_bytes", "state", "external"}
        assert m["state"] in {"ok", "unmapped", "incomplete", "external"}
        # alias is only meaningful for a runnable, registry-mapped entry.
        if m["state"] != "ok":
            assert m["alias"] is None


def test_command_emits_single_valid_json_available(capfd) -> None:
    models_command(SimpleNamespace(cached=False, json=True))
    out = capfd.readouterr().out
    doc = json.loads(out)  # raises if a banner leaked onto stdout
    assert "text" in doc


def test_command_emits_single_valid_json_cached(capfd) -> None:
    models_command(SimpleNamespace(cached=True, json=True))
    out = capfd.readouterr().out
    doc = json.loads(out)
    assert "cached" in doc and "total_bytes" in doc
