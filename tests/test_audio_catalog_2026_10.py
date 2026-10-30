# SPDX-License-Identifier: Apache-2.0
"""Catalog and offline engine contracts for the October 2026 audio models."""

import fnmatch
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rapid_mlx.audio import registry
from rapid_mlx.audio.tts import TTSEngine, detect_tts_family

ROOT = Path(__file__).resolve().parents[1]
GRANITE = "ibm-granite/granite-speech-5.0-470m-turboctc"
FIRERED = "mlx-community/FireRedASR2-AED-mlx"
NEMOTRON = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
VOX = "mlx-community/VoxCPM2-8bit"
MOSS = "mlx-community/MOSS-TTS-Nano-100M"
CODEC = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
PATTERNS = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
)


@pytest.mark.parametrize(
    "alias,repo,kind,family,languages",
    [
        (
            "granite-speech",
            GRANITE,
            "stt",
            "granite_speech5_ctc",
            "en,fr,de,es,pt,it,ja",
        ),
        (
            "granite-speech-5.0-470m-turboctc",
            GRANITE,
            "stt",
            "granite_speech5_ctc",
            "en,fr,de,es,pt,it,ja",
        ),
        ("firered-asr2", FIRERED, "stt", "fireredasr2", "zh,en"),
        ("fireredasr2-aed", FIRERED, "stt", "fireredasr2", "zh,en"),
        ("nemotron-asr", NEMOTRON, "stt", "nemotron_asr", "multilingual"),
        (
            "nemotron-3.5-asr-streaming-0.6b",
            NEMOTRON,
            "stt",
            "nemotron_asr",
            "multilingual",
        ),
        ("voxcpm2", VOX, "tts", "voxcpm", None),
        ("voxcpm2-8bit", VOX, "tts", "voxcpm", None),
        ("moss-tts-nano", MOSS, "tts", "moss_tts_nano", None),
    ],
)
def test_new_aliases(alias, repo, kind, family, languages):
    entry = registry.resolve_audio_alias(alias)
    assert entry is not None
    assert (entry.hf_id, entry.type, entry.family) == (repo, kind, family)
    assert re.fullmatch(r"[0-9a-f]{40}", entry.revision)
    assert entry.languages == languages
    if kind == "tts":
        assert entry.default_voice == "default"
    raw = json.loads((ROOT / "rapid_mlx/audio/aliases.json").read_text())[alias]
    assert "trust_remote_code" not in raw
    assert "model_file" not in raw


def test_codec_asset_and_catalog_notes():
    expected = registry.AudioRuntimeAsset(
        repo_id=CODEC,
        revision="6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
        allow_patterns=PATTERNS,
    )
    assert registry.runtime_assets_for("moss-tts-nano") == (expected,)
    assert registry.runtime_assets_for(MOSS) == (expected,)
    for pattern in expected.allow_patterns:
        assert not any(
            fnmatch.fnmatch(name, pattern)
            for name in ("modeling_moss_audio_tokenizer.py", "__init__.py")
        )
    notes = registry.resolve_audio_alias("chatterbox").notes
    assert "English" in notes and "Multilingual" not in notes


def test_family_detection():
    assert detect_tts_family(MOSS) == "moss_tts_nano"
    assert detect_tts_family(VOX) == "voxcpm"


def test_moss_generation_passes_only_codec_arguments(monkeypatch):
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.array = type("FakeArray", (), {})
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    calls = []

    class Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            yield type(
                "Result",
                (),
                {"audio": np.ones((4800, 2), dtype=np.float32), "sample_rate": 48000},
            )()

    engine = TTSEngine(MOSS)
    engine.model = Model()
    engine._loaded = True
    engine._moss_codec_path = "/local/codec"
    normal = engine.generate("hi", voice="default", speed=1.0, ref_audio=None)
    fast = engine.generate("hi", speed=1.25)
    streamed = list(engine.stream_generate("hi"))
    assert normal.sample_rate == 48000
    assert normal.audio.shape == (4800,)
    assert np.all(normal.audio == 1.0)
    assert fast.audio.ndim == 1 and len(fast.audio) < len(normal.audio)
    assert len(streamed) == 1 and streamed[0].audio.shape == (4800,)
    assert np.all(streamed[0].audio == 1.0)
    assert (
        calls
        == [
            {
                "text": "hi",
                "mode": "continuation",
                "audio_tokenizer_source": "/local/codec",
            }
        ]
        * 3
    )


def test_moss_load_resolves_codec(monkeypatch, tmp_path):
    from rapid_mlx import model_downloads
    from rapid_mlx.audio import tts

    fake_audio = types.ModuleType("mlx_audio")
    fake_tts = types.ModuleType("mlx_audio.tts")
    generate = types.ModuleType("mlx_audio.tts.generate")
    generate.load_model = lambda path: object()
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", generate)
    calls = []
    monkeypatch.setattr(model_downloads, "require_local", lambda name: None)
    monkeypatch.setattr(tts, "_resolve_safe_tts_checkpoint", lambda name: tmp_path)
    monkeypatch.setattr(generate, "load_model", lambda path: object())
    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda repo, **kwargs: calls.append((repo, kwargs)) or "/local/codec",
    )
    engine = TTSEngine(MOSS)
    engine.load()
    assert engine._loaded and engine._moss_codec_path == "/local/codec"
    assert calls == [(CODEC, {"allow_patterns": list(PATTERNS)})]
    monkeypatch.setattr(registry, "runtime_assets_for", lambda name: ())
    engine = TTSEngine(MOSS)
    with pytest.raises(RuntimeError, match="codec runtime asset"):
        engine.load()
    assert not engine._loaded


def test_model_sizes():
    sizes = json.loads((ROOT / "rapid_mlx/model_sizes.json").read_text())["sizes"]
    for repo in (GRANITE, FIRERED, NEMOTRON, VOX, MOSS):
        assert type(sizes[repo]) is int and sizes[repo] > 0
