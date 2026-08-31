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
    assert set(payload) == {"text", "audio", "video", "image", "atomic"}
    assert all(isinstance(payload[k], list) for k in ("text", "audio", "video", "image"))
    assert set(payload["atomic"]) == {
        "snapshot",
        "recommendation_policies",
        "shadow_report",
    }
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
        "supports_native_mtp",
        "mtp_draft_model",
        "mtp_speculative_tokens",
        "modality",
        "video_modes",
        "min_memory_gb",
        "is_builtin",
        "is_text_only",
    ):
        assert key in entry, f"text entry missing {key!r}"
    assert isinstance(entry["is_hybrid"], bool)
    assert isinstance(entry["is_moe"], bool)
    assert isinstance(entry["supports_native_mtp"], bool)
    assert isinstance(entry["is_builtin"], bool)
    assert isinstance(entry["is_text_only"], bool)
    assert entry["size_bytes"] is None or isinstance(entry["size_bytes"], int)
    assert entry["video_modes"] == []
    assert entry["min_memory_gb"] is None or isinstance(
        entry["min_memory_gb"], (int, float)
    )


def test_atomic_projection_preserves_legacy_aliases_and_is_bounded() -> None:
    payload = _available_models_json_payload()
    atomic = payload["atomic"]
    snapshot = atomic["snapshot"]

    assert atomic["shadow_report"]["equivalent"] is True
    assert {entry["alias"] for entry in snapshot["aliases"]} == {
        entry["alias"]
        for bucket in ("text", "audio", "video", "image")
        for entry in payload[bucket]
    }
    assert snapshot["catalog_digest"].startswith("sha256:")
    # Desktop caps subprocess stdout at 1 MiB. Keep healthy headroom while the
    # shadow envelope contains both legacy buckets and the atomic graph.
    assert len(json.dumps(payload, separators=(",", ":")).encode()) < 768 * 1024


def test_available_sections_are_split_by_modality() -> None:
    payload = _available_models_json_payload()
    # No text alias should carry a generative modality, and vice versa.
    assert all(e["modality"] not in ("video-gen", "image-gen") for e in payload["text"])
    assert all(e["modality"] == "video-gen" for e in payload["video"])
    assert all(e["modality"] == "image-gen" for e in payload["image"])
    assert all(e["modality"] == "audio" for e in payload["audio"])


def test_video_entries_expose_pre_serve_modes_and_memory_floor() -> None:
    payload = _available_models_json_payload()
    expected_modes = {
        "cogvideox-fun-5b-q4": ["text-to-video"],
        "cogvideox-fun-5b-q8": ["text-to-video"],
        "cogvideox-fun-5b-bf16": ["text-to-video"],
        "wan2.2-ti2v-5b-q8": ["text-to-video", "image-to-video"],
        "wan2.2-ti2v-5b-bf16": ["text-to-video", "image-to-video"],
        "wan2.2-i2v-a14b-q8": ["image-to-video"],
        "wan2.2-t2v-a14b-bf16": ["text-to-video"],
        "ltx-2.3-mlx-q4": ["text-to-video", "image-to-video"],
        "ltx-2.5-mlx-q8": ["text-to-video", "image-to-video"],
    }
    by_alias = {entry["alias"]: entry for entry in payload["video"]}

    assert set(by_alias) == set(expected_modes)
    for alias, modes in expected_modes.items():
        assert by_alias[alias]["video_modes"] == modes
        assert by_alias[alias]["min_memory_gb"] > 0

    assert all(entry["video_modes"] == [] for entry in payload["text"])
    assert all(entry["video_modes"] == [] for entry in payload["image"])


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


def test_atomic_shadow_failure_preserves_legacy_discovery(monkeypatch) -> None:
    import vllm_mlx.audio.registry as audio_registry

    def broken_audio_registry():
        raise ValueError("simulated optional registry failure")

    monkeypatch.setattr(audio_registry, "list_audio_aliases", broken_audio_registry)
    payload = _available_models_json_payload()

    assert payload["text"]
    assert payload["audio"] == []
    assert "atomic" not in payload
