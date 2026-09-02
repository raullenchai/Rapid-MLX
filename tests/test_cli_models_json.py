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
    assert all(
        isinstance(payload[k], list) for k in ("text", "audio", "video", "image")
    )
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
        "wan2.1-t2v-1.3b-bf16": ["text-to-video"],
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
        assert set(m) >= {
            "alias",
            "repo",
            "subfolder",
            "size_bytes",
            "state",
            "external",
        }
        assert m["state"] in {"ok", "unmapped", "incomplete", "external"}
        # alias is only meaningful for a runnable, registry-mapped entry.
        if m["state"] != "ok":
            assert m["alias"] is None
            assert m["subfolder"] is None


def test_cached_payload_reports_every_complete_alias_subfolder(monkeypatch) -> None:
    import vllm_mlx.cli as cli
    import vllm_mlx.model_aliases as aliases

    profiles = {
        "nested-4bit": SimpleNamespace(hf_path="org/multi-quant", subfolder="4bit"),
        "nested-8bit": SimpleNamespace(hf_path="org/multi-quant", subfolder="8bit"),
    }
    monkeypatch.setattr(aliases, "list_profiles", lambda: profiles)
    monkeypatch.setattr(aliases, "resolve_subfolder", lambda _repo: "4bit")
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [("org/multi-quant", 1024, 0.0)],
    )
    monkeypatch.setattr(cli, "_scan_external_model_dirs", lambda: [])
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda _repo: True)
    monkeypatch.setattr(
        cli,
        "_cached_subfolder_size",
        lambda _repo, subfolder: {"4bit": 400, "8bit": 800}.get(subfolder),
    )

    rows = cli._cached_models_json_payload()["cached"]
    assert [(row["alias"], row["subfolder"], row["size_bytes"]) for row in rows] == [
        ("nested-8bit", "8bit", 800),
        ("nested-4bit", "4bit", 400),
    ]
    assert all(row["repo"] == "org/multi-quant" for row in rows)


def test_cached_subfolder_size_accepts_one_complete_unreferenced_snapshot(
    monkeypatch, tmp_path
) -> None:
    import huggingface_hub.constants as hub_constants

    import vllm_mlx.cli as cli

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    repo_root = tmp_path / "models--org--multi-quant"
    checkpoint = repo_root / "snapshots" / "abc123" / "4bit"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    weights = b"complete weights"
    (checkpoint / "model.safetensors").write_bytes(weights)

    size = cli._cached_subfolder_size("org/multi-quant", "4bit")
    assert size is not None
    assert size >= len(weights)
    assert cli._cached_subfolder_size("org/multi-quant", "8bit") is None

    (repo_root / "snapshots" / "other").mkdir()
    assert cli._cached_subfolder_size("org/multi-quant", "4bit") is None


def test_cached_subfolder_size_fails_closed_for_invalid_current_snapshots(
    monkeypatch, tmp_path
) -> None:
    import os

    import huggingface_hub.constants as hub_constants

    import vllm_mlx.cli as cli

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    repo_root = tmp_path / "models--org--multi-quant"
    snapshot = repo_root / "snapshots" / "abc123"
    refs = repo_root / "refs"
    snapshot.mkdir(parents=True)
    refs.mkdir()
    (refs / "main").write_text("abc123", encoding="utf-8")

    assert cli._cached_subfolder_size("org/multi-quant", "../escape") is None
    assert cli._cached_subfolder_size("org/multi-quant", "4bit") is None

    outside = tmp_path / "outside"
    outside.mkdir()
    (snapshot / "4bit").symlink_to(outside, target_is_directory=True)
    assert cli._cached_subfolder_size("org/multi-quant", "4bit") is None
    (snapshot / "4bit").unlink()
    checkpoint = snapshot / "4bit"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    def broken_walk(*_args, **_kwargs):
        raise OSError("cache disappeared")
        yield  # pragma: no cover

    monkeypatch.setattr(os, "walk", broken_walk)
    assert cli._cached_subfolder_size("org/multi-quant", "4bit") is None


def test_cached_payload_reconciles_root_duplicates_external_and_unknown(
    monkeypatch,
) -> None:
    import vllm_mlx.cli as cli
    import vllm_mlx.model_aliases as aliases

    profiles = {
        "root-a": SimpleNamespace(hf_path="org/root", subfolder=None),
        "root-b": SimpleNamespace(hf_path="org/root", subfolder=None),
    }
    monkeypatch.setattr(aliases, "list_profiles", lambda: profiles)
    monkeypatch.setattr(aliases, "resolve_subfolder", lambda _repo: None)
    monkeypatch.setattr(
        cli,
        "_scan_hf_cache_models",
        lambda: [
            ("org/root", 100, 1.0),
            ("org/unmapped", 80, 0.0),
            ("org/incomplete", 60, 0.0),
        ],
    )
    monkeypatch.setattr(
        cli, "_scan_external_model_dirs", lambda: [("local/model", 40, 0.0)]
    )
    monkeypatch.setattr(
        cli,
        "_cache_entry_is_runnable",
        lambda repo: repo != "org/incomplete",
    )

    rows = cli._cached_models_json_payload()["cached"]
    assert [(row["repo"], row["alias"], row["state"]) for row in rows] == [
        ("org/root", "root-a", "ok"),
        ("org/unmapped", None, "unmapped"),
        ("org/incomplete", None, "incomplete"),
        ("local/model", None, "external"),
    ]


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
