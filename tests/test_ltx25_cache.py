# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 cache admission uses the files opened by its bundled runtime."""

import json

import pytest

from rapid_mlx import _download_gate as gate
from rapid_mlx import cli, model_metadata

REPO = "MrMofer/ltx-2.5-mlx-q8"
CHECKPOINT_FILES = (
    "config.json",
    "embedded_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/tokenizer.json",
    "text_encoder/tokenizer_config.json",
    "transformer-distilled.safetensors",
    "transformer-dev.safetensors",
    "ltx-2.5-22b-distilled-lora-450.safetensors",
    "connector.safetensors",
    "vae_encoder.safetensors",
    "vae_decoder.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2.safetensors",
    "temporal_upscaler_x2.safetensors",
    "duration_head.safetensors",
)


def _cache(tmp_path, monkeypatch):
    root = tmp_path / "models--MrMofer--ltx-2.5-mlx-q8"
    snap = root / "snapshots" / "revision"
    snap.mkdir(parents=True)
    for name in CHECKPOINT_FILES:
        path = snap / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"{}" if name.endswith(".json") else b"weights")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("revision")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    import huggingface_hub.file_download as fd

    monkeypatch.setattr(fd, "HF_HUB_CACHE", str(tmp_path), raising=False)
    return snap


def test_complete_ltx25_cache_serves_offline_without_stub_notice(tmp_path, monkeypatch):
    snap = _cache(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert cli._cache_runnability(REPO) is True
    assert gate.weightless_stub_notice(REPO) is None
    assert (
        model_metadata._complete_cached_checkpoint(snap.parents[1], snap, REPO) == snap
    )
    cli._ensure_model_downloaded(REPO)


def test_complete_ltx25_cache_without_main_ref_is_not_admitted(tmp_path, monkeypatch):
    snap = _cache(tmp_path, monkeypatch)
    (snap.parents[1] / "refs" / "main").unlink()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert cli._cache_runnability(REPO) is False
    with pytest.raises(SystemExit, match="1"):
        cli._ensure_model_downloaded(REPO)


@pytest.mark.parametrize(
    "missing", ["transformer-distilled.safetensors", "text_encoder/model.safetensors"]
)
def test_partial_ltx25_cache_is_not_runnable(tmp_path, monkeypatch, missing):
    snap = _cache(tmp_path, monkeypatch)
    (snap / missing).unlink()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert cli._cache_runnability(REPO) is False
    assert gate.weightless_stub_notice(REPO) is not None
    assert (
        model_metadata._complete_cached_checkpoint(snap.parents[1], snap, REPO) is None
    )
    with pytest.raises(SystemExit, match="1"):
        cli._ensure_model_downloaded(REPO)


@pytest.mark.parametrize(
    "preferred",
    [
        "transformer.safetensors",
        "transformer-distilled-1.1.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
    ],
)
def test_empty_preferred_ltx25_weight_is_incomplete(tmp_path, monkeypatch, preferred):
    snap = _cache(tmp_path, monkeypatch)
    (snap / preferred).touch()

    assert cli._cache_runnability(REPO) is False
    assert gate.weightless_stub_notice(REPO) is not None


@pytest.mark.parametrize(
    "name", [name for name in CHECKPOINT_FILES if name.endswith(".json")]
)
def test_truncated_ltx25_json_is_incomplete(tmp_path, monkeypatch, name):
    snap = _cache(tmp_path, monkeypatch)
    (snap / name).write_text("{")

    assert cli._cache_runnability(REPO) is False
    assert gate.weightless_stub_notice(REPO) is not None


def test_text_cache_behavior_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "models--publisher--text"
    snap = root / "snapshots" / "revision"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("revision")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    import huggingface_hub.file_download as fd

    monkeypatch.setattr(fd, "HF_HUB_CACHE", str(tmp_path), raising=False)

    assert cli._cache_runnability("publisher/text") is False
    assert gate.is_weightless_stub("publisher/text") is True
    (snap / "model.safetensors").write_bytes(b"weights")
    assert cli._cache_runnability("publisher/text") is True
    assert gate.is_weightless_stub("publisher/text") is False


def test_ltx25_main_ref_must_select_complete_snapshot(tmp_path, monkeypatch):
    snap = _cache(tmp_path, monkeypatch)
    partial = snap.parent / "new-revision"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    (snap.parents[1] / "refs" / "main").write_text(partial.name)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    assert cli._cache_runnability(REPO) is False
    assert gate.weightless_stub_notice(REPO) is not None
    with pytest.raises(SystemExit, match="1"):
        cli._ensure_model_downloaded(REPO)


def test_complete_ltx25_cache_online_with_preferred_weights(tmp_path, monkeypatch):
    snap = _cache(tmp_path, monkeypatch)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    for name in (
        "transformer.safetensors",
        "transformer-distilled-1.1.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
    ):
        (snap / name).write_bytes(b"weights")

    assert cli._cache_runnability(REPO) is True
    assert gate.weightless_stub_notice(REPO) is None
    cli._ensure_model_downloaded(REPO)


def test_ltx25_tokenizer_larger_than_config_limit_is_admitted(tmp_path, monkeypatch):
    snap = _cache(tmp_path, monkeypatch)
    (snap / "text_encoder/tokenizer.json").write_text(
        json.dumps({"vocab": "x" * (2 * 1024 * 1024)})
    )
    assert cli._cache_runnability(REPO) is True
    assert gate.weightless_stub_notice(REPO) is None
