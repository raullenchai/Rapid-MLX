"""Offline regressions for WAV lengths, local alignment and user aliases."""
# ruff: noqa: F401, F811

import json
import struct
import sys
from types import SimpleNamespace

import pytest

from tests.test_forced_alignment_route_hardening import (
    _audio_worker_without_mlx_init,
    _FakeAlignerEngine,
    _mount_audio_app,
    _post,
    _stub_aligner,
)


def _wav(data_size=48000, riff_size=48036, channels=1):
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, 16000, 32000 * channels, 2 * channels, 16
        )
        + b"data"
        + struct.pack("<I", data_size)
        + b"\0" * (48000 * channels)
    )


@pytest.mark.parametrize("size", [0xFFFFFFFF, 0xFFFFFF00, 192000, None])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("probe_duration", [None, 1.25])
def test_wav_duration_falls_through_for_oversized_headers(
    tmp_path, monkeypatch, size, channels, probe_duration
):
    from rapid_mlx.audio import stt
    from rapid_mlx.audio.stt import _audio_duration_seconds

    calls = []

    def info(audio_path):
        calls.append(audio_path)
        return SimpleNamespace(duration=probe_duration)

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(info=info) if probe_duration is not None else None,
    )
    monkeypatch.setattr(stt.shutil, "which", lambda _: None)
    path = tmp_path / "pipe.wav"
    path.write_bytes(
        _wav(size if size is not None else 48000 * channels, 0xFFFFFFFF, channels)
    )
    expected = probe_duration if size is not None else 1.5
    duration = _audio_duration_seconds(str(path), allow_decode=False)
    if expected is None:
        assert duration is None
    else:
        assert duration == pytest.approx(expected)
    assert calls == (
        [str(path)] if size is not None and probe_duration is not None else []
    )


def test_oversized_wav_duration_falls_back_to_decode(tmp_path, monkeypatch):
    from rapid_mlx.audio import stt

    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setattr(stt.shutil, "which", lambda _: None)
    calls = []

    def decode(audio_path):
        calls.append(audio_path)
        return 1.5

    monkeypatch.setattr(stt, "_decoded_audio_duration", decode)
    path = tmp_path / "pipe.wav"
    path.write_bytes(_wav(0xFFFFFFFF, 0xFFFFFFFF))
    assert stt._audio_duration_seconds(str(path), allow_decode=False) is None
    assert calls == []
    assert stt._audio_duration_seconds(str(path), allow_decode=True) == pytest.approx(
        1.5
    )
    assert calls == [str(path)]


def test_sentinel_wav_duration_does_not_count_trailing_metadata(tmp_path, monkeypatch):
    from rapid_mlx.audio import stt

    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setattr(stt.shutil, "which", lambda _: None)
    path = tmp_path / "metadata.wav"
    path.write_bytes(
        _wav(0xFFFFFFFF, 0xFFFFFFFF)
        + b"LIST"
        + struct.pack("<I", 16000)
        + b"\0" * 16000
    )
    assert stt._audio_duration_seconds(str(path), allow_decode=False) is None


def test_wav_duration_ignores_trailing_metadata(tmp_path):
    from rapid_mlx.audio.stt import _audio_duration_seconds

    path = tmp_path / "metadata.wav"
    path.write_bytes(_wav() + b"LIST" + struct.pack("<I", 16000) + b"\0" * 16000)
    assert _audio_duration_seconds(str(path)) == pytest.approx(1.5)


@pytest.fixture
def local_checkpoint(tmp_path_factory, monkeypatch):
    from rapid_mlx.audio import registry
    from rapid_mlx.config import get_config

    monkeypatch.setattr(registry, "_LOCAL_AUDIO_ENTRIES", {})
    root = tmp_path_factory.mktemp("checkpoint")

    def make(config):
        folder = root / "timing"
        folder.mkdir(exist_ok=True)
        (folder / "config.json").write_text(json.dumps(config))
        entry = registry.resolve_local_audio_folder(str(folder))
        assert entry is not None
        registry.register_served_local_audio_entry(entry)
        cfg = get_config()
        for key, value in dict(
            model_alias="gateway", model_name=str(folder), model_path=str(folder)
        ).items():
            monkeypatch.setattr(cfg, key, value)
        return str(folder)

    return make


_ALIGNER_CONFIGS = [
    {
        "model_type": "qwen3_asr",
        "thinker_config": {"model_type": "qwen3_forced_aligner"},
    },
    {"model_type": "qwen3_forced_aligner"},
]


@pytest.mark.parametrize("config", _ALIGNER_CONFIGS)
def test_neutral_local_aligner_engine(local_checkpoint, config, tmp_path):
    from rapid_mlx.audio.stt import STTEngine

    path = local_checkpoint(config)
    engine = STTEngine(path)
    engine._loaded = True
    engine.model = SimpleNamespace(generate=lambda **kwargs: SimpleNamespace(items=[]))
    audio = tmp_path / "clip.wav"
    audio.write_bytes(_wav())
    assert engine.align(str(audio), text="hello").duration == pytest.approx(1.5)
    with pytest.raises(ValueError, match="forced-alignment model"):
        engine.transcribe(str(audio))


@pytest.mark.parametrize("config", _ALIGNER_CONFIGS)
def test_neutral_local_aligner_route(_stub_aligner, local_checkpoint, config):
    path = local_checkpoint(config)
    client, restore = _mount_audio_app()
    try:
        response = _post(client, {"model": path, "text": "hello"})
    finally:
        restore()
    assert response.status_code == 200, response.text
    from rapid_mlx.routes import audio

    assert audio._aligner_engine.model_name == path


@pytest.mark.parametrize("model", [None, "   "])
def test_alignment_omitted_model_uses_served_local(
    _stub_aligner, local_checkpoint, model
):
    path = local_checkpoint(_ALIGNER_CONFIGS[0])
    data = {"text": "hello"}
    if model is not None:
        data["model"] = model
    client, restore = _mount_audio_app()
    try:
        response = _post(client, data)
    finally:
        restore()
    assert response.status_code == 200, response.text
    from rapid_mlx.routes import audio

    assert audio._aligner_engine.model_name == path


def test_ordinary_local_asr_rejects_alignment(_stub_aligner, local_checkpoint):
    path = local_checkpoint(
        {"model_type": "qwen3_asr", "thinker_config": {"model_type": "qwen3_asr"}}
    )
    client, restore = _mount_audio_app()
    try:
        response = _post(client, {"model": path, "text": "hello"})
    finally:
        restore()
    assert response.status_code == 400
    assert _FakeAlignerEngine.last_call is None


@pytest.fixture
def audio_user_alias(tmp_path, monkeypatch):
    from rapid_mlx.cli import (
        _resolve_audio_model_for_serve,
        alias_command,
        build_parser,
    )
    from rapid_mlx.model_aliases import resolve_model

    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(tmp_path / "aliases.json"))

    def make(name, target):
        args = build_parser().parse_args(["alias", "set", name, target])
        alias_command(args)
        resolved = resolve_model(name)
        assert resolved == target
        assert _resolve_audio_model_for_serve(resolved).hf_id == target
        return name

    return make


def test_user_whisper_alias_transcription(_stub_aligner, audio_user_alias, monkeypatch):
    from rapid_mlx.config import get_config

    target = "mlx-community/whisper-base-mlx"
    alias = audio_user_alias("gateway-stt", target)
    cfg = get_config()
    monkeypatch.setattr(cfg, "model_alias", alias)
    monkeypatch.setattr(cfg, "model_name", target)
    client, restore = _mount_audio_app()
    try:
        response = _post(client, {"model": alias})
    finally:
        restore()
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "recognised speech"


def test_user_tts_alias_resolves_target(audio_user_alias):
    from rapid_mlx.routes.audio import _resolve_tts_model

    target = "mlx-community/Kokoro-82M-bf16"
    alias = audio_user_alias("gateway-tts", target)
    assert _resolve_tts_model(alias) == target


@pytest.mark.parametrize("guard", ["translation", "timestamps"])
def test_user_asr_alias_keeps_capability_guards(audio_user_alias, guard):
    from fastapi import HTTPException

    from rapid_mlx.routes import audio

    alias = audio_user_alias("gateway-stt", audio.STT_MODEL_ALIASES["parakeet"])
    with pytest.raises(HTTPException) as error:
        if guard == "translation":
            audio._reject_non_whisper_for_translation(alias)
        else:
            audio._reject_word_timestamps_for_non_whisper(alias, ["word"])
    assert error.value.status_code == 400
