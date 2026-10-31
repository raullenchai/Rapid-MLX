import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rapid_mlx.audio import registry


@pytest.mark.parametrize(
    ("model_type", "kind"),
    [("nemotron_asr", "stt"), ("WHISPER", "stt"), ("kokoro", "tts"), ("qwen3", None)],
)
def test_local_config(tmp_path, model_type, kind):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    entry = registry.resolve_local_audio_folder(str(tmp_path))
    assert (entry.type if entry else None) == kind
    if entry:
        assert entry.hf_id == str(tmp_path.resolve())
        assert entry.revision is None


def test_bad_config(tmp_path):
    assert registry.resolve_local_audio_folder(str(tmp_path)) is None
    (tmp_path / "config.json").write_text("{")
    assert registry.resolve_local_audio_folder(str(tmp_path)) is None


@pytest.mark.parametrize(
    ("name", "config", "expected"),
    [
        ("Kokoro-82M-bf16", {}, ("tts", "kokoro")),
        ("generic", {"architecture": "QWEN3-TTS"}, ("tts", "qwen3_tts")),
        ("qwen3-tts-0.6b", {}, ("tts", "qwen3_tts")),
        ("kokoro-whisper", {}, None),
        ("Kokoro-82M-bf16", {"model_type": "qwen3"}, None),
        ("kokoro", {"model_type": "qwen3", "architecture": "kokoro"}, None),
        ("generic", {"model_type": "NEMOTRON-ASR"}, ("stt", "nemotron_asr")),
        (
            "generic",
            {
                "model_type": "unknown",
                "architectures": ["WhisperForConditionalGeneration"],
            },
            ("stt", "whisper"),
        ),
        (
            "whisper-model",
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
            None,
        ),
    ],
)
def test_local_folder_resolution_order(tmp_path, name, config, expected):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config))
    entry = registry.resolve_local_audio_folder(str(folder) + "/")
    assert ((entry.type, entry.family) if entry else None) == expected


def test_local_folder_does_not_shadow_catalog_alias(tmp_path, monkeypatch):
    folder = tmp_path / "Whisper-Large-V3"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type":"whisper"}')
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    catalog = registry.resolve_audio_alias("whisper-large-v3")
    entry = registry.resolve_local_audio_folder(str(folder))
    registry.register_served_local_audio_entry(entry, str(folder))
    assert catalog is not None
    assert registry.resolve_audio_alias("WHISPER-LARGE-V3") == catalog
    assert registry.resolve_audio_alias(str(folder)) == entry


def test_route_local_and_default(tmp_path, monkeypatch):
    from rapid_mlx.routes import audio

    (tmp_path / "config.json").write_text('{"model_type":"nemotron_asr"}')
    entry = registry.resolve_local_audio_folder(str(tmp_path))
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    registry.register_served_local_audio_entry(entry)
    monkeypatch.setattr(
        "rapid_mlx.config.get_config",
        lambda: SimpleNamespace(
            model_alias=entry.hf_id, model_name="gateway", model_path=entry.hf_id
        ),
    )
    assert audio._resolve_stt_model(entry.alias) == entry.hf_id
    assert audio._resolve_stt_model("default") == entry.hf_id
    assert audio._resolve_stt_model(None) == entry.hf_id
    assert registry.resolve_audio_alias(entry.hf_id) == entry
    assert registry.is_audio_name(entry.alias)
    assert audio._resolve_stt_model("gateway") == entry.hf_id


def test_serve_local_audio(tmp_path, monkeypatch):
    import sys

    from rapid_mlx import cli
    from tests.test_audio_alias_registry import _make_serve_args

    (tmp_path / "config.json").write_text('{"model_type":"nemotron_asr"}')
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_resolve_api_key=lambda key: key),
    )
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8000)
    with (
        patch.object(cli, "_serve_audio_mode") as audio,
        patch.object(cli, "_ensure_model_downloaded") as text,
        patch.object(cli, "_run_optional_runtime_guard"),
    ):
        cli.serve_command(_make_serve_args(str(tmp_path)))
    assert audio.called
    assert not text.called


@pytest.mark.parametrize(
    ("name", "config", "kind", "family"),
    [
        ("stt-folder", {"model_type": "nemotron_asr"}, "stt", "nemotron_asr"),
        ("Kokoro-82M-bf16", {}, "tts", "kokoro"),
    ],
)
def test_serve_relative_local_audio(tmp_path, monkeypatch, name, config, kind, family):
    import sys

    from rapid_mlx import cli
    from tests.test_audio_alias_registry import _make_serve_args

    folder = tmp_path / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_resolve_api_key=lambda key: key),
    )
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kwargs: 8000)
    with (
        patch.object(cli, "_serve_audio_mode") as audio,
        patch.object(cli, "_ensure_model_downloaded") as text,
        patch.object(cli, "_run_optional_runtime_guard"),
    ):
        cli.serve_command(_make_serve_args(f"./{name}"))
    entry = audio.call_args.args[1]
    assert (entry.type, entry.family, entry.hf_id) == (
        kind,
        family,
        str(folder.resolve()),
    )
    assert not text.called
    assert registry.resolve_audio_alias(f"./{name}") == entry
    assert registry.resolve_audio_alias(str(folder.absolute())) == entry


@pytest.mark.parametrize("model", [None, "./stt-folder"])
def test_relative_transcription_request(tmp_path, monkeypatch, model):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.audio import probe
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio

    folder = tmp_path / "stt-folder"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type":"nemotron_asr"}')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    entry = registry.resolve_local_audio_folder("./stt-folder")
    registry.register_served_local_audio_entry(entry, "./stt-folder")
    cfg = get_config()
    monkeypatch.setattr(cfg, "model_alias", None)
    monkeypatch.setattr(cfg, "model_name", "./stt-folder")
    monkeypatch.setattr(cfg, "model_path", "./stt-folder")
    observed = []

    async def record_request(*, model, **kwargs):
        observed.append(audio._resolve_stt_model(model))
        return {"text": "hi"}

    monkeypatch.setattr(cfg, "api_key", None)
    monkeypatch.setattr(probe, "require_mlx_audio_stt", lambda: None)
    monkeypatch.setattr(audio, "_run_stt_request", record_request)
    app = FastAPI()
    app.include_router(audio.router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": model} if model else {},
            files={"file": ("tone.wav", b"audio", "audio/wav")},
        )
    assert response.status_code == 200, response.text
    assert observed == [str(folder.resolve())]


def test_text_folder_takes_text_path(tmp_path, monkeypatch):
    import sys

    from rapid_mlx import cli
    from tests.test_audio_alias_registry import _make_serve_args

    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_resolve_api_key=lambda key: key),
    )
    with (
        patch.object(cli, "_serve_audio_mode") as audio,
        patch.object(
            cli, "_ensure_model_downloaded", side_effect=RuntimeError("text path")
        ),
        pytest.raises(RuntimeError, match="text path"),
    ):
        cli.serve_command(_make_serve_args(str(tmp_path)))
    assert not audio.called


def test_main_skips_text_preflight_for_local_audio(tmp_path, monkeypatch):
    import sys

    from rapid_mlx import cli

    (tmp_path / "config.json").write_text('{"model_type":"nemotron_asr"}')
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", str(tmp_path)])
    with (
        patch.object(cli, "serve_command") as serve,
        patch(
            "rapid_mlx.byom.preflight.run_cli_preflight",
            side_effect=AssertionError("text preflight"),
        ),
    ):
        cli.main()
    assert serve.call_args.args[0].model == str(tmp_path)


def test_tts_local_custom_name_and_default(tmp_path, monkeypatch):
    from rapid_mlx.routes import audio

    (tmp_path / "config.json").write_text('{"model_type":"kokoro"}')
    entry = registry.resolve_local_audio_folder(str(tmp_path))
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    registry.register_served_local_audio_entry(entry)
    monkeypatch.setattr(
        "rapid_mlx.config.get_config",
        lambda: SimpleNamespace(
            model_alias="relative-name", model_name="gateway", model_path=entry.hf_id
        ),
    )
    assert audio._resolve_tts_model(entry.alias) == entry.hf_id
    assert audio._resolve_tts_model("gateway") == entry.hf_id
    assert audio._resolve_tts_model(None) == entry.hf_id


def test_local_stt_loader_skips_hub(tmp_path, monkeypatch):
    import sys

    from rapid_mlx import model_aliases, model_downloads
    from rapid_mlx.audio.stt import STTEngine
    from rapid_mlx.utils import model_file_guard

    def forbidden(*args, **kwargs):
        raise AssertionError("Hub resolution called for local directory")

    monkeypatch.setattr(model_downloads, "require_local", forbidden)
    monkeypatch.setattr(model_downloads, "snapshot_download", forbidden)
    monkeypatch.setattr(model_aliases, "resolve_revision", forbidden)
    monkeypatch.setattr(model_file_guard, "checked_stt_loader", lambda loader: loader)
    monkeypatch.setitem(sys.modules, "mlx_audio", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mlx_audio.stt.utils",
        SimpleNamespace(load_model=lambda path: SimpleNamespace(config=None)),
    )
    engine = STTEngine(str(tmp_path))
    engine.load()
    assert engine._loaded


def test_local_snapshot_voices(tmp_path, monkeypatch):
    import huggingface_hub

    from rapid_mlx.audio.tts import _list_snapshot_voices

    voices = tmp_path / "voices"
    voices.mkdir()
    for name in ("b.safetensors", "a.safetensors", "ignored.txt"):
        (voices / name).touch()
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *a, **k: pytest.fail("local voice lookup reached Hub"),
    )
    assert _list_snapshot_voices(str(tmp_path)) == ["a", "b"]


def test_local_snapshot_adapter_and_whisper_processor(tmp_path, monkeypatch):
    from rapid_mlx import model_downloads
    from rapid_mlx.audio.stt import _whisper_processor_source
    from rapid_mlx.audio.tts import (
        _resolve_indextts_snapshot,
        _scoped_snapshot_download,
    )

    monkeypatch.setattr(
        model_downloads,
        "snapshot_download",
        lambda *a, **k: pytest.fail("local snapshot reached Hub"),
    )
    module = SimpleNamespace(snapshot_download=None)
    with _scoped_snapshot_download(module):
        assert module.snapshot_download(repo_id=str(tmp_path)) == str(tmp_path)
    assert module.snapshot_download is None
    assert _resolve_indextts_snapshot(str(tmp_path)) == tmp_path
    assert _whisper_processor_source(str(tmp_path), 80, 51865) == "openai/whisper-base"
    (tmp_path / "preprocessor_config.json").write_text("{}")
    assert _whisper_processor_source(str(tmp_path), 80, 51865) == "openai/whisper-base"
    (tmp_path / "tokenizer.json").touch()
    assert _whisper_processor_source(str(tmp_path), 80, 51865) == str(tmp_path)


def test_local_speech_request_reaches_engine_without_hub(tmp_path, monkeypatch):
    import sys

    import huggingface_hub
    import numpy as np
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rapid_mlx.audio import tts
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio

    folder = tmp_path / "Kokoro-82M-bf16"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type":"kokoro"}')
    (folder / "voices").mkdir()
    (folder / "voices" / "af_heart.safetensors").touch()
    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    entry = registry.resolve_local_audio_folder(str(folder))
    registry.register_served_local_audio_entry(entry)
    cfg = get_config()
    monkeypatch.setattr(cfg, "model_alias", None)
    monkeypatch.setattr(cfg, "model_name", str(folder))
    monkeypatch.setattr(cfg, "model_path", str(folder))
    monkeypatch.setattr(cfg, "api_key", None)
    for name in ("try_to_load_from_cache", "snapshot_download"):
        monkeypatch.setattr(
            huggingface_hub,
            name,
            lambda *a, **k: pytest.fail("local speech request reached Hub"),
        )
    seen = []

    class FakeEngine:
        def __init__(self, model_name):
            self.model_name = model_name
            seen.append(model_name)

        def load(self):
            pass

        def generate(self, text, **kwargs):
            assert text == "Hello"
            assert kwargs["voice"] == "af_heart"
            return tts.AudioOutput(np.zeros(10, dtype=np.float32), 24000, 10 / 24000)

        def to_bytes(self, output, format):
            return b"RIFFfake"

    monkeypatch.setattr(tts, "TTSEngine", FakeEngine)
    monkeypatch.setattr(
        "rapid_mlx.runtime.audio_worker.run_audio_mlx_sync",
        lambda role, model, phase, fn, *a, **kw: fn(*a, **kw),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(_emit_audio_model_served_once=lambda *a: None),
    )
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_kokoro_runtime", lambda *a: None)
    monkeypatch.setattr(audio, "_tts_engine", None)
    app = FastAPI()
    app.include_router(audio.router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech", json={"input": "Hello", "response_format": "wav"}
        )
    assert response.status_code == 200, response.text
    assert seen == [str(folder.resolve())]
