# SPDX-License-Identifier: Apache-2.0
"""IndexTTS zero-shot cloning: registry, engine, loader, and HTTP contract."""

from __future__ import annotations

import base64
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

INDEXTTS_REPO = "mlx-community/IndexTTS-1.5"
_REF_AUDIO = base64.b64encode(b"RIFF----WAVEfake-reference").decode()
_UNSET = object()


def test_registry_exposes_indextts_aliases():
    from vllm_mlx.audio.registry import resolve_audio_alias

    for alias in ("indextts", "indextts-1.5", INDEXTTS_REPO):
        entry = resolve_audio_alias(alias)
        assert entry is not None
        assert entry.hf_id == INDEXTTS_REPO
        assert entry.family == "indextts"
        assert entry.default_voice == "clone"


def test_engine_detects_both_indextts_spellings():
    from vllm_mlx.audio.tts import TTSEngine

    assert TTSEngine("mlx-community/IndexTTS-1.5")._model_family == "indextts"
    assert TTSEngine("org/index-tts-custom")._model_family == "indextts"


def test_engine_forwards_only_text_and_reference(monkeypatch):
    from vllm_mlx.audio.tts import TTSEngine

    calls: list[dict] = []
    decoded_reference = object()

    class _Result:
        audio = np.zeros(240, dtype=np.float32)
        sample_rate = 24000

    class _Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            yield _Result()

    engine = TTSEngine(INDEXTTS_REPO)
    engine.model = _Model()
    engine._loaded = True
    import mlx_audio.tts.generate as generate_mod

    monkeypatch.setattr(
        generate_mod,
        "load_audio",
        lambda path, sample_rate: decoded_reference,
    )

    output = engine.generate(
        "Clone this voice",
        voice="must-not-leak",
        speed=1.7,
        lang_code="z",
        ref_audio="/tmp/reference.wav",
        ref_text="not needed by IndexTTS",
    )

    assert calls == [{"text": "Clone this voice", "ref_audio": decoded_reference}]
    assert output.sample_rate == 24000


def test_engine_requires_reference_audio():
    from vllm_mlx.audio.tts import TTSEngine

    engine = TTSEngine(INDEXTTS_REPO)
    engine.model = object()
    engine._loaded = True
    with pytest.raises(ValueError, match="requires ref_audio"):
        engine.generate("No reference")


def test_loader_injects_tokenizer_without_mutating_config(monkeypatch, tmp_path):
    """Community checkpoints omit tokenizer_name; patch only in memory."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "indextts"}))
    (tmp_path / "tokenizer.model").write_bytes(b"tokenizer")

    seen: dict = {}

    class _Model:
        def __init__(self, config):
            seen["config"] = config

        def load_weights(self, weights, strict):
            seen["weights"] = weights
            seen["strict"] = strict

        def parameters(self):
            return ["params"]

        def eval(self):
            seen["eval"] = True

    import huggingface_hub
    import mlx.core as mx
    import mlx_audio.tts.models.indextts.indextts as indextts_mod

    def _unexpected_download(*args, **kwargs):
        raise AssertionError("local IndexTTS checkpoints must not hit Hugging Face")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _unexpected_download)
    monkeypatch.setattr(indextts_mod, "Model", _Model)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(mx, "load", lambda path: {"w": "value"})
    monkeypatch.setattr(mx, "eval", lambda params: seen.setdefault("mx_eval", params))

    from vllm_mlx.audio.tts import _load_indextts_model

    model = _load_indextts_model(str(tmp_path))

    assert isinstance(model, _Model)
    assert Path(seen["config"]["tokenizer_name"]) == tmp_path
    assert seen["weights"] == [("w", "value")]
    assert seen["strict"] is True
    assert seen["eval"] is True
    # The shared HF cache file remains byte-for-byte untouched.
    assert json.loads(config_path.read_text()) == {"model_type": "indextts"}


def _install_fake_mlx_audio(monkeypatch):
    fake = types.ModuleType("mlx_audio")
    fake.__path__ = []
    fake.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio", loader=None, is_package=True
    )
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_tts.__path__ = []
    fake_tts.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts", loader=None, is_package=True
    )
    monkeypatch.setitem(sys.modules, "mlx_audio", fake)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)


class _RecordingEngine:
    instances: list[_RecordingEngine] = []
    _real_to_bytes = None

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: list[dict] = []
        type(self).instances.append(self)

    def load(self):
        pass

    def generate(
        self,
        text,
        voice=_UNSET,
        speed=1.0,
        ref_audio=_UNSET,
        ref_text=_UNSET,
        **kwargs,
    ):
        from vllm_mlx.audio.tts import AudioOutput

        call = {"text": text, "speed": speed}
        if voice is not _UNSET:
            call["voice"] = voice
        if ref_audio is not _UNSET:
            call["ref_audio"] = ref_audio
            call["ref_exists_during_generate"] = Path(ref_audio).is_file()
        if ref_text is not _UNSET:
            call["ref_text"] = ref_text
        self.calls.append(call)
        return AudioOutput(
            audio=np.zeros(240, dtype=np.float32),
            sample_rate=24000,
            duration=0.01,
        )

    def to_bytes(self, audio, format="wav"):
        return type(self)._real_to_bytes(self, audio, format=format)


def _mount(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm_mlx.audio import probe as probe_mod
    from vllm_mlx.audio import tts as tts_mod
    from vllm_mlx.config import get_config
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import audio as audio_route

    _RecordingEngine.instances = []
    _RecordingEngine._real_to_bytes = tts_mod.TTSEngine.to_bytes
    _install_fake_mlx_audio(monkeypatch)
    monkeypatch.setattr(probe_mod, "require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(tts_mod, "TTSEngine", _RecordingEngine)
    monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _name: [])
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    app = FastAPI()
    app.include_router(audio_route.router)
    install_exception_handlers(app)
    monkeypatch.setattr(get_config(), "api_key", None)
    return TestClient(app)


def test_route_accepts_reference_without_transcript(monkeypatch):
    client = _mount(monkeypatch)
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "indextts",
            "input": "Clone this voice",
            "ref_audio": _REF_AUDIO,
        },
    )

    assert response.status_code == 200, response.text
    (engine,) = _RecordingEngine.instances
    assert engine.model_name == INDEXTTS_REPO
    (call,) = engine.calls
    assert call["ref_exists_during_generate"] is True
    assert "voice" not in call
    assert "ref_text" not in call


def test_route_rejects_indextts_without_reference(monkeypatch):
    client = _mount(monkeypatch)
    response = client.post(
        "/v1/audio/speech",
        json={"model": "indextts", "input": "No reference"},
    )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "missing_reference_audio"
    assert _RecordingEngine.instances == []


def test_model_rejects_reference_text_without_audio():
    from pydantic import ValidationError

    from vllm_mlx.api.models import AudioSpeechRequest

    with pytest.raises(ValidationError, match="ref_text requires ref_audio"):
        AudioSpeechRequest(
            model="indextts",
            input="Bad pair",
            ref_text="orphan transcript",
        )


# --------------------------------------------------------------------------- #
# Snapshot resolution — a warm cache must not touch the network
# --------------------------------------------------------------------------- #
def _seed_indextts_snapshot(root: Path, *, shards=("model.safetensors",), index=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    (root / "tokenizer.model").write_bytes(b"tok")
    for shard in shards:
        (root / shard).write_bytes(b"weights")
    if index is not None:
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return root


def _patch_snapshot_download(monkeypatch, *, cached, networked):
    """Route ``local_files_only`` at ``cached`` and everything else at ``networked``.

    ``cached=None`` models a cache the Hub client cannot resolve at all (no
    ``refs/main``), which is what an interrupted mirror pull leaves behind.
    """
    import huggingface_hub

    calls: list[bool] = []

    def _fake(model_name, allow_patterns=None, local_files_only=False, **kwargs):
        calls.append(local_files_only)
        if local_files_only:
            if cached is None:
                raise OSError("no resolvable local snapshot")
            return str(cached)
        return str(networked)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake)
    return calls


def test_warm_cache_resolves_without_a_network_call(monkeypatch, tmp_path):
    """A complete cached snapshot is used as-is — no online resolve at all.

    ``snapshot_download`` resolves ``main`` → sha through the Hub even when every
    file is already on disk, and that lookup carries no timeout, so it is the
    step that hangs a start on a hostile network.
    """
    from vllm_mlx.audio import tts

    cached = _seed_indextts_snapshot(tmp_path / "cached")
    calls = _patch_snapshot_download(
        monkeypatch, cached=cached, networked=tmp_path / "networked"
    )

    assert tts._resolve_indextts_snapshot(INDEXTTS_REPO) == cached
    assert calls == [True]  # the online resolve never ran


def test_partial_cache_falls_back_to_the_network(monkeypatch, tmp_path):
    """A snapshot missing an indexed shard must still complete its download.

    huggingface_hub only judges completeness when it holds a cached tree
    listing; a mirror-populated snapshot has none and is returned sight-unseen,
    so the missing shard has to be caught here.
    """
    from vllm_mlx.audio import tts

    cached = _seed_indextts_snapshot(
        tmp_path / "cached",
        shards=("0.safetensors",),
        index={"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}},
    )
    networked = tmp_path / "networked"
    calls = _patch_snapshot_download(monkeypatch, cached=cached, networked=networked)

    assert tts._resolve_indextts_snapshot(INDEXTTS_REPO) == networked
    assert calls == [True, False]


def test_unresolvable_cache_falls_back_to_the_network(monkeypatch, tmp_path):
    """An unresolvable cache is today's path, not a hard failure."""
    from vllm_mlx.audio import tts

    networked = tmp_path / "networked"
    calls = _patch_snapshot_download(monkeypatch, cached=None, networked=networked)

    assert tts._resolve_indextts_snapshot(INDEXTTS_REPO) == networked
    assert calls == [True, False]


def test_complete_sharded_cache_is_accepted(tmp_path):
    """Every shard the index names is present → the cache is usable."""
    from vllm_mlx.audio import tts

    snapshot = _seed_indextts_snapshot(
        tmp_path / "cached",
        shards=("0.safetensors", "1.safetensors"),
        index={"weight_map": {"a": "0.safetensors", "b": "1.safetensors"}},
    )

    assert tts._cached_snapshot_holds_indextts(snapshot) is True


@pytest.mark.parametrize("absent", ["config.json", "tokenizer.model"])
def test_cache_missing_a_required_file_is_rejected(tmp_path, absent):
    """The loader opens both of these directly; neither may be assumed."""
    from vllm_mlx.audio import tts

    snapshot = _seed_indextts_snapshot(tmp_path / "cached")
    (snapshot / absent).unlink()

    assert tts._cached_snapshot_holds_indextts(snapshot) is False
