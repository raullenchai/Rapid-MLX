"""Capacity provenance for audio roles (#2305).

Issue #2305 forbids inferring capacity from model names or hashes. These tests
pin the three legitimate sources and, most importantly, that the name-parsing
heuristic used for arbitrary text checkpoints is NOT reachable from this path.
"""

from __future__ import annotations

import json
import os

import pytest

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


def test_complete_snapshot_without_index_is_measured(tmp_path, monkeypatch):
    # Single-file (non-sharded) repos are complete with one non-empty
    # ``model*.safetensors`` and must still be sizable.
    repo_id = "someone/single-file-asr"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    (snapshot / "model.safetensors").write_bytes(b"x" * 2048)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 2048 + 2


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


def test_partial_snapshot_is_not_reported_as_cached(tmp_path, monkeypatch):
    """A half-downloaded repo must not size to its fragment (#2305 follow-up).

    An interrupted pull leaves config.json plus some shards. Counting those
    bytes reported a few hundred KiB as the model's footprint, admission
    reserved ~512 MiB of overhead, and then the loader downloaded the missing
    multi-GiB weights against that reservation.
    """

    repo_id = "someone/interrupted-asr"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    # Index promises two shards; only one landed before the pull died.
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1024)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "unknown"
    assert capacity.reserved_bytes == 0


def test_snapshot_without_refs_main_is_not_measured(tmp_path, monkeypatch):
    """Only the revision the loader resolves may be measured.

    Without refs/main there is no way to know which snapshot snapshot_download
    would open, so an unrelated stale one must not stand in for it.
    """

    repo_id = "someone/no-refs"
    snapshot = tmp_path / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "old"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x" * 4096)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    assert resolve_audio_role_capacity(repo_id).capacity_source == "unknown"


def test_runtime_overhead_does_not_claim_to_bound_decoded_audio():
    """The 25 MB upload cap bounds compressed bytes, not decoded samples.

    25 MB of 6 kbps Opus is ~9.7 hours, which is ~2.1 GiB of float32 at 16 kHz
    — far past the per-role allowance. The request-level duration limit is what
    actually bounds it, so assert that limit keeps the buffer within budget.
    """

    from vllm_mlx.runtime.audio_capacity import MAX_TRANSCRIPTION_SECONDS

    float32_at_16k = MAX_TRANSCRIPTION_SECONDS * 16_000 * 4
    assert float32_at_16k < AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES

    # And the cap the upload limit alone would permit does NOT fit, which is
    # why the duration bound has to exist.
    opus_6kbps_seconds = (25 * 1024 * 1024) / (6_000 / 8)
    assert opus_6kbps_seconds * 16_000 * 4 > AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES


def test_overlong_audio_is_rejected_before_the_engine_allocates(tmp_path):
    """Duration, not compressed size, is what bounds the decoded buffer."""

    import wave

    from fastapi import HTTPException

    from vllm_mlx.routes.audio import _audio_request_bytes
    from vllm_mlx.runtime.audio_capacity import MAX_TRANSCRIPTION_SECONDS

    def write_short_wav(path, rate=8000):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(b"\0\0")
        return path

    short = write_short_wav(tmp_path / "short.wav")
    # A tiny file charges a tiny amount rather than raising.
    assert _audio_request_bytes(str(short)) < 1024 * 1024

    long_path = tmp_path / "long.wav"
    rate = 8000
    frames = int((MAX_TRANSCRIPTION_SECONDS + 3600) * rate)
    with wave.open(str(long_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0")
    # Rewrite the frame count in the header rather than materializing hours of
    # silence on disk.
    with open(long_path, "r+b") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(40)
        handle.write((frames * 2).to_bytes(4, "little"))
        handle.seek(4)
        handle.write((size - 8).to_bytes(4, "little"))

    with pytest.raises(HTTPException) as excinfo:
        _audio_request_bytes(str(long_path))
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail["error"]["code"] == "audio_too_long"


def test_unreadable_container_is_charged_worst_case_not_refused(tmp_path):
    """A format the metadata probe cannot parse must still be bounded.

    Refusing outright would reject containers the mlx-audio backend decodes
    fine (it accepts more formats than libsndfile), turning a memory guard into
    a compatibility regression. Charging the worst case the file size permits
    keeps the bound sound without that cost.
    """

    from vllm_mlx.routes.audio import _audio_request_bytes
    from vllm_mlx.runtime.audio_capacity import (
        transcription_buffer_bytes,
        worst_case_duration_seconds,
    )

    opaque = tmp_path / "mystery.bin"
    opaque.write_bytes(b"\x00" * 4096)

    charged = _audio_request_bytes(str(opaque))

    assert charged == transcription_buffer_bytes(worst_case_duration_seconds(4096))
    # Non-zero: an unmeasurable file must not be free.
    assert charged > 0


def test_speech_input_is_length_bounded():
    """TTS output size grows with input length, so input must be bounded."""

    from pydantic import ValidationError

    from vllm_mlx.api.models import AudioSpeechRequest
    from vllm_mlx.runtime.audio_capacity import MAX_SPEECH_INPUT_CHARACTERS

    assert AudioSpeechRequest(input="x" * MAX_SPEECH_INPUT_CHARACTERS)
    with pytest.raises(ValidationError):
        AudioSpeechRequest(input="x" * (MAX_SPEECH_INPUT_CHARACTERS + 1))


def test_speech_buffer_scales_with_speed_not_just_characters():
    """20k chars is not a memory bound on its own — speed multiplies it."""

    from vllm_mlx.runtime.audio_capacity import (
        MAX_SPEECH_INPUT_CHARACTERS,
        speech_buffer_bytes,
    )

    at_normal = speech_buffer_bytes(MAX_SPEECH_INPUT_CHARACTERS, speed=1.0)
    at_slowest = speech_buffer_bytes(MAX_SPEECH_INPUT_CHARACTERS, speed=0.25)

    # speed=0.25 quadruples the duration of identical text.
    assert at_slowest == pytest.approx(at_normal * 4, rel=0.01)
    # And the worst case is multi-GiB, which is why it must be charged rather
    # than assumed to fit inside a fixed per-role allowance.
    assert at_slowest > 4 * 1024**3
    assert at_slowest > AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES


def test_custom_npz_repo_is_measured_without_registry_membership(tmp_path, monkeypatch):
    """The routes accept any org/repo, so completeness must follow the layout.

    An unregistered but complete config.json + weights.npz repo previously fell
    through to the text/split probes, resolved to "unknown", and 507'd under a
    ceiling even though it was fully downloaded.
    """

    repo_id = "someone/custom-asr-npz"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    (snapshot / "config.json").write_text("{}")
    (snapshot / "weights.npz").write_bytes(b"x" * 8192)

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 8192 + 2
