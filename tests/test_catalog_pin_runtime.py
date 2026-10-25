# SPDX-License-Identifier: Apache-2.0
"""Catalog pins reach download boundaries without fetching model bytes."""

import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx import model_aliases, model_downloads
from rapid_mlx.audio import registry


def test_all_catalog_aliases_resolve_exact_revision():
    root = Path(__file__).resolve().parents[1]
    repos = {}
    for file, field in [
        ("rapid_mlx/aliases.json", "hf_path"),
        ("rapid_mlx/audio/aliases.json", "hf_id"),
    ]:
        for alias, entry in json.loads((root / file).read_text()).items():
            if alias.startswith("_"):
                continue
            pin, repo = entry["revision"], entry[field]
            assert re.fullmatch("[0-9a-f]{40}", pin)
            assert model_aliases.resolve_revision(alias) == pin
            assert model_aliases.resolve_revision(repo) == pin
            assert repos.setdefault(repo.lower(), pin) == pin


@pytest.mark.parametrize("pin", ["main", "A" * 40, 123, []])
def test_audio_invalid_pin_rejected(pin):
    with pytest.raises(ValueError, match="40-hex"):
        registry._revision({"revision": pin})


@pytest.mark.parametrize(
    "name", ["kokoro", "mlx-community/Kokoro-82M-bf16", "prince-canuma/Kokoro-82M"]
)
@pytest.mark.parametrize("method", ["snapshot_download", "hf_hub_download"])
def test_audio_download_pin_reaches_hub(monkeypatch, name, method):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(
        huggingface_hub, method, lambda *a, **kw: calls.append((a, kw)) or "/snapshot"
    )
    monkeypatch.setattr(model_downloads, "disabled", lambda: False)
    args = (name,) if method == "snapshot_download" else (name, "config.json")
    getattr(model_downloads, method)(*args)
    assert calls[0][0][0] == model_aliases.resolve_hub_model(name)
    assert calls[0][1]["revision"] == registry.resolve_audio_revision(name)
    with pytest.raises(ValueError, match="conflicting revision"):
        getattr(model_downloads, method)(*args, revision="main")


def test_stt_implicit_load_uses_pinned_local_snapshot(monkeypatch, tmp_path):
    import huggingface_hub

    from rapid_mlx.audio.stt import STTEngine

    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: calls.append((a, kw)) or str(tmp_path),
    )
    utils = ModuleType("mlx_audio.stt.utils")
    loaded = []
    utils.load_model = lambda path: (
        loaded.append(path)
        or SimpleNamespace(config=SimpleNamespace(model_type="sensevoice"))
    )
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils)
    engine = STTEngine("mlx-community/SenseVoiceSmall")
    engine.load()
    assert loaded == [str(tmp_path)]
    assert calls[0][1]["revision"] == registry.resolve_audio_revision(engine.model_name)


def test_f5_loader_uses_shared_pin_gate(monkeypatch):
    from rapid_mlx.audio.tts import TTSEngine

    package = ModuleType("f5_tts_mlx")
    utils = ModuleType("f5_tts_mlx.utils")
    cfm = ModuleType("f5_tts_mlx.cfm")
    calls = []
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda repo: calls.append(repo) or "/snapshot",
    )
    cfm.F5TTS = SimpleNamespace(
        from_pretrained=lambda repo: utils.snapshot_download(repo)
    )
    package.utils = utils
    for name, module in [
        ("f5_tts_mlx", package),
        ("f5_tts_mlx.utils", utils),
        ("f5_tts_mlx.cfm", cfm),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.load()
    assert engine.model == "/snapshot"
    assert calls == ["lucasnewman/f5-tts-mlx"]


def test_kokoro_implicit_assets_use_pin_gate(monkeypatch, tmp_path):
    from rapid_mlx.audio import tts

    pipeline = ModuleType("mlx_audio.tts.models.kokoro.pipeline")
    kokoro = ModuleType("mlx_audio.tts.models.kokoro")
    kokoro.pipeline = pipeline
    generate = ModuleType("mlx_audio.tts.generate")
    generate.load_model = lambda path: path
    for name, module in [
        ("mlx_audio.tts.models.kokoro", kokoro),
        ("mlx_audio.tts.models.kokoro.pipeline", pipeline),
        ("mlx_audio.tts.generate", generate),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(tts, "_resolve_safe_tts_checkpoint", lambda name: tmp_path)
    engine = tts.TTSEngine("mlx-community/Kokoro-82M-bf16")
    engine.load()
    assert pipeline.snapshot_download is model_downloads.snapshot_download
    assert engine.model == tmp_path
