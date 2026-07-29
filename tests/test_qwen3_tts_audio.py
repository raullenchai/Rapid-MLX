# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS (CustomVoice) audio support.

Adds Alibaba's Qwen3-TTS 1.7B CustomVoice to the TTS lane: multilingual
predefined speakers (Chinese: Vivian/Serena/Uncle_Fu/Dylan/Eric, English:
Ryan/Aiden) with emotion/style control via the OpenAI ``instructions``
field (mapped to the engine's ``instruct`` argument).

These tests are hermetic — no weights, no network. They fake the
``mlx_audio`` boundary and pin the contract:

* the ``qwen3-tts`` aliases resolve through the central registry to the
  CustomVoice bf16 repo with ``family="qwen3_tts"``;
* ``TTSEngine`` detects the family and its voice list is the documented
  speaker set;
* ``generate`` forwards ``instruct`` AND uses ``lang_code="auto"`` for
  Qwen3 (never Kokoro's single-letter ``"a"``), and does NOT leak
  ``instruct`` into families that have no emotion control;
* the route validates the speaker set, resolves the omitted / ``default``
  voice to the registry ``default_voice`` (``Serena``), and threads the
  ``instructions`` field down to the engine.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

CUSTOMVOICE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"


# ---------------------------------------------------------------------------
# A) Registry resolution
# ---------------------------------------------------------------------------


class TestQwen3TTSRegistry:
    @pytest.mark.parametrize(
        "alias,expected_hf_id",
        [
            ("qwen3-tts", CUSTOMVOICE_BF16),
            ("qwen3-tts-customvoice", CUSTOMVOICE_BF16),
            ("qwen3-tts-6bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"),
            ("qwen3-tts-4bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"),
            # Case-insensitive: docs/SDKs mix case off the upstream repo.
            ("Qwen3-TTS", CUSTOMVOICE_BF16),
        ],
    )
    def test_alias_resolves(self, alias, expected_hf_id):
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(alias)
        assert entry is not None, f"{alias!r} did not resolve in the registry"
        assert entry.type == "tts"
        assert entry.family == "qwen3_tts"
        assert entry.hf_id == expected_hf_id
        assert entry.default_voice == "Serena"

    def test_hf_id_reverse_lookup(self):
        """The full HF id maps back to the qwen3_tts entry so ``serve
        <hf-id>`` forks into audio mode like the short alias does."""
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(CUSTOMVOICE_BF16)
        assert entry is not None and entry.family == "qwen3_tts"

    def test_registered_default_voice_is_a_real_speaker(self):
        """The registry ``default_voice`` MUST be in the served voice list
        or the cold-start / voice-omitted path 400s on a value the server
        itself chose."""
        from vllm_mlx.audio.registry import resolve_audio_alias
        from vllm_mlx.audio.tts import QWEN3_TTS_VOICES

        entry = resolve_audio_alias("qwen3-tts")
        assert entry.default_voice in QWEN3_TTS_VOICES


# ---------------------------------------------------------------------------
# B) Engine — family detection, voice list, emotion-aware generate
# ---------------------------------------------------------------------------


class _CapturingModel:
    """Fake mlx_audio model: records the kwargs ``generate`` was called
    with and yields one result carrying a tiny audio buffer."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        import numpy as np

        self.calls.append(kwargs)
        result = types.SimpleNamespace(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000
        )
        return iter([result])


def _qwen3_engine():
    """A loaded ``TTSEngine`` for the Qwen3 CustomVoice repo whose model is
    the capturing fake (no weights, no network)."""
    from vllm_mlx.audio.tts import TTSEngine

    engine = TTSEngine(CUSTOMVOICE_BF16)
    engine.model = _CapturingModel()
    engine._loaded = True
    return engine


class TestQwen3TTSEngine:
    def test_family_detected(self):
        assert _qwen3_engine()._model_family == "qwen3_tts"

    def test_get_voices_is_speaker_set(self):
        from vllm_mlx.audio.tts import QWEN3_TTS_VOICES

        assert _qwen3_engine().get_voices() == list(QWEN3_TTS_VOICES)

    def test_generate_forwards_instruct_and_auto_lang(self):
        engine = _qwen3_engine()
        engine.generate("你好", voice="Serena", instruct="悬疑而低沉")
        (call,) = engine.model.calls
        assert call["voice"] == "Serena"
        assert call["instruct"] == "悬疑而低沉"
        # Qwen3 auto-detects language; forwarding Kokoro's "a" would
        # mis-hint it.
        assert call["lang_code"] == "auto"

    def test_generate_omits_instruct_when_absent(self):
        engine = _qwen3_engine()
        engine.generate("你好", voice="Serena")
        (call,) = engine.model.calls
        assert "instruct" not in call

    def test_instruct_not_leaked_to_kokoro_family(self):
        """A Kokoro engine handed ``instruct`` (e.g. a client sending
        ``instructions`` to a non-emotion model) must NOT forward it — the
        Kokoro path has no such kwarg and would raise, and its lang_code
        stays the single-letter form."""
        from vllm_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine.model = _CapturingModel()
        engine._loaded = True
        engine.generate("hello", voice="af_heart", instruct="cheerful")
        (call,) = engine.model.calls
        assert "instruct" not in call
        assert call["lang_code"] == "a"


# ---------------------------------------------------------------------------
# C) Route — voice validation + instructions plumbing
# ---------------------------------------------------------------------------


def _install_fake_mlx_audio(monkeypatch):
    """Minimal fake so the TTS-lane probe passes without the real extra."""
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
    """No-op TTSEngine stub that records the generate() kwargs and returns
    a real WAV via the actual encoder."""

    instances: list[_RecordingEngine] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generate_calls: list[dict] = []
        _RecordingEngine.instances.append(self)

    def load(self):
        pass

    def generate(self, text, voice="af_heart", speed=1.0, instruct=None):
        import numpy as np

        self.generate_calls.append(
            {"text": text, "voice": voice, "speed": speed, "instruct": instruct}
        )
        from vllm_mlx.audio.tts import AudioOutput

        return AudioOutput(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000, duration=0.01
        )

    # Bound to the REAL ``TTSEngine.to_bytes`` in ``_mount`` BEFORE the
    # monkeypatch rebinds ``tts_mod.TTSEngine`` to this stub — going
    # through the patched attribute would recurse forever.
    _real_to_bytes = None

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
    # Capture the REAL encoder before TTSEngine is rebound to the stub.
    _RecordingEngine._real_to_bytes = tts_mod.TTSEngine.to_bytes
    _install_fake_mlx_audio(monkeypatch)
    monkeypatch.setattr(probe_mod, "require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda: None)
    monkeypatch.setattr(tts_mod, "TTSEngine", _RecordingEngine)
    # Cold-start voice path: force snapshot enumeration to empty so the
    # route uses the static Qwen3 speaker list (matches a fresh install).
    monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _n: [])
    # Route the module-global engine cache through monkeypatch so teardown
    # restores the true original singleton — a direct assignment would leak
    # our stub into later audio tests and make them order-dependent.
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    app = FastAPI()
    app.include_router(audio_route.router)
    install_exception_handlers(app)
    cfg = get_config()
    # monkeypatch captures the REAL api_key and restores it on teardown;
    # setting it directly first would make monkeypatch record ``None`` as
    # the "original" and leak auth-disabled config into later tests.
    monkeypatch.setattr(cfg, "api_key", None)
    client = TestClient(app)
    return client


class TestQwen3TTSRoute:
    def test_speech_forwards_instructions_as_instruct(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "他被诸葛亮压了一辈子。",
                "voice": "Serena",
                "instructions": "悬疑而低沉，逐渐激昂。",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        (engine,) = _RecordingEngine.instances
        assert engine.model_name == CUSTOMVOICE_BF16
        (call,) = engine.generate_calls
        assert call["voice"] == "Serena"
        assert call["instruct"] == "悬疑而低沉，逐渐激昂。"

    def test_omitted_voice_resolves_to_registry_default(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "你好世界。"},
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        # Voice omitted → registry default_voice, not Kokoro's af_heart.
        assert call["voice"] == "Serena"

    def test_unknown_voice_rejected_with_speaker_list(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "hi",
                "voice": "af_heart",  # a Kokoro voice, invalid for qwen3
            },
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "invalid_voice"
        # The envelope should advertise a real Qwen3 speaker.
        assert "Serena" in body["error"]["message"]
