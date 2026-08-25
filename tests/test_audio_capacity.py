"""Capacity provenance for audio roles (#2305).

Issue #2305 forbids inferring capacity from model names or hashes. These tests
pin the three legitimate sources and, most importantly, that the name-parsing
heuristic used for arbitrary text checkpoints is NOT reachable from this path.
"""

from __future__ import annotations

import json
import os

from vllm_mlx.runtime.audio_capacity import (
    AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES,
    resolve_audio_role_capacity,
)


def test_registered_alias_is_sized_from_the_catalog_manifest():
    from vllm_mlx.audio.registry import resolve_audio_alias
    from vllm_mlx.model_sizes import size_bytes

    capacity = resolve_audio_role_capacity("whisper-large-v3")

    entry = resolve_audio_alias("whisper-large-v3")
    assert capacity.capacity_source == "manifest"
    assert capacity.hf_id == entry.hf_id
    assert capacity.weight_bytes == size_bytes(entry.hf_id)
    assert capacity.reserved_bytes == (
        capacity.weight_bytes + AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES
    )


def test_every_registered_audio_alias_resolves_to_a_known_capacity():
    # The registry is the only place a new audio model lands. If one is added
    # without regenerating model_sizes.json, every load of it becomes an
    # unbudgeted admission — catch that here rather than in production.
    from vllm_mlx.audio.registry import list_audio_aliases

    unknown = [
        entry.alias
        for entry in list_audio_aliases()
        if not resolve_audio_role_capacity(entry.alias).is_known
    ]
    assert unknown == []


def test_hf_id_resolves_the_same_way_as_its_short_alias():
    by_alias = resolve_audio_role_capacity("kokoro")
    by_repo = resolve_audio_role_capacity(by_alias.hf_id)

    assert by_repo == by_alias


def test_unlisted_repo_falls_back_to_measuring_the_cached_snapshot(
    tmp_path, monkeypatch
):
    repo_id = "someone/unlisted-asr"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "abc123"
    blobs = repo_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("abc123")

    blob = blobs / "deadbeef"
    blob.write_bytes(b"x" * 4096)
    # HF snapshots are symlink trees into blobs/; one blob can back several
    # entries and must only be counted once.
    os.symlink(blob, snapshot / "weights.npz")
    os.symlink(blob, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(json.dumps({"a": 1}))

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 4096 + len(json.dumps({"a": 1}))
    assert capacity.reserved_bytes == (
        capacity.weight_bytes + AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES
    )


def test_uncached_unlisted_repo_reports_unknown_rather_than_guessing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity("someone/never-downloaded")

    assert capacity.capacity_source == "unknown"
    assert capacity.is_known is False
    # Zero, and the manager turns that into a rejection: a zero charge would
    # otherwise skip the ceiling check entirely.
    assert capacity.reserved_bytes == 0
    assert capacity.weight_bytes is None


def test_snapshot_without_refs_main_still_measures_the_cache(tmp_path, monkeypatch):
    # A manually populated or partially repaired cache can lack refs/main.
    # Refusing to size it would reject the load outright, so fall back to the
    # largest snapshot on disk rather than reporting "unknown".
    repo_id = "someone/no-refs"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.npz").write_bytes(b"x" * 2048)

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 2048


def test_capacity_is_never_inferred_from_a_parameter_count_in_the_name(
    tmp_path, monkeypatch
):
    # ``estimate_model_bytes`` would happily size this from "7b" + "4bit".
    # #2305 forbids that for audio roles; the resolver must say "unknown".
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity("fictional/asr-7b-4bit")

    assert capacity.capacity_source == "unknown"
    assert capacity.reserved_bytes == 0
