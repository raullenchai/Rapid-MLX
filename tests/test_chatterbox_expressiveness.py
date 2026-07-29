# SPDX-License-Identifier: Apache-2.0
"""Chatterbox expressiveness (``exaggeration``) + zero-shot cloning.

Wires Chatterbox's emotion/intensity knob end-to-end so a user can
de-flatten monotone narration over the OpenAI-compatible HTTP surface:

* ``AudioSpeechRequest`` gains a bounded ``exaggeration`` field
  (``0.0`` neutral → ``2.0`` very theatrical). Only the Chatterbox
  family honours it; every other TTS family ignores it — mirroring
  OpenAI's behaviour for styling on voices that don't support it, so a
  caller may send it unconditionally without a 400.
* ``TTSEngine.generate`` grows an ``exaggeration`` parameter and a
  dedicated Chatterbox branch that forwards ``exaggeration`` and
  ``ref_audio`` (zero-shot cloning) ONLY when set, and deliberately
  does NOT forward Kokoro-oriented ``voice``/``speed``/``lang_code``.
* ``/v1/audio/speech``'s ``create_speech`` threads the field into the
  engine's ``generate`` kwargs only when the caller sent it.

These tests are hermetic — no weights, no network. They fake the
``mlx_audio`` boundary and pin the contract at three layers: the
request model, the engine, and the route.

Context assumed TRUE (verified against the installed ``mlx_audio``):
``chatterbox.Model.generate`` carries a real ``exaggeration`` named
parameter AND ``**kwargs`` (as does ``mlx_audio.tts.generate_audio``),
and the same ``chatterbox.Model`` backs both the non-turbo and turbo
repos, so forwarding ``exaggeration`` never raises ``TypeError`` on
either variant.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

# The engine's ``generate`` path imports ``mlx.core`` and the whole
# audio lane transitively pulls numpy/mlx; skip cleanly on API-only
# runners that don't install the heavy deps.
pytest.importorskip("mlx.core")
pytest.importorskip("numpy")

CHATTERBOX_FP16 = "mlx-community/chatterbox-fp16"
CHATTERBOX_TURBO = "mlx-community/chatterbox-turbo-fp16"


_UNSET = object()


# ---------------------------------------------------------------------------
# A) Request model — bounded ``exaggeration`` field
# ---------------------------------------------------------------------------


class TestAudioSpeechRequestExaggeration:
    def test_defaults_to_none(self):
        from vllm_mlx.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(input="Hello world")
        assert req.exaggeration is None

    def test_accepts_in_range_value(self):
        from vllm_mlx.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(
            model="chatterbox", input="Big news today!", exaggeration=0.8
        )
        assert req.exaggeration == 0.8

    @pytest.mark.parametrize("value", [0.0, 2.0])
    def test_accepts_boundaries(self, value):
        from vllm_mlx.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(input="hi", exaggeration=value)
        assert req.exaggeration == value

    @pytest.mark.parametrize("value", [-0.1, 2.1, 5.0])
    def test_rejects_out_of_range(self, value):
        from pydantic import ValidationError

        from vllm_mlx.api.models import AudioSpeechRequest

        with pytest.raises(ValidationError):
            AudioSpeechRequest(input="hi", exaggeration=value)


# ---------------------------------------------------------------------------
# B) Engine — family detection + conditional forwarding
# ---------------------------------------------------------------------------


class _CapturingChatterbox:
    """Fake ``mlx_audio`` Chatterbox model with a DELIBERATELY STRICT
    ``generate`` signature: only ``text`` plus the two knobs the engine's
    chatterbox branch is permitted to forward (``exaggeration``,
    ``ref_audio``), each behind a sentinel default. If the engine leaks
    Kokoro-oriented ``voice``/``speed``/``lang_code`` into the call it
    raises ``TypeError`` here instead of silently passing the test."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, *, text, exaggeration=_UNSET, ref_audio=_UNSET):
        import numpy as np

        rec: dict = {"text": text}
        if exaggeration is not _UNSET:
            rec["exaggeration"] = exaggeration
        if ref_audio is not _UNSET:
            rec["ref_audio"] = ref_audio
        self.calls.append(rec)
        result = types.SimpleNamespace(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000
        )
        return iter([result])


class _CapturingKokoro:
    """Fake Kokoro-style model. Its explicit signature accepts the
    Kokoro call shape (``text``/``voice``/``speed``/``lang_code``) but NOT
    ``exaggeration`` — so a non-chatterbox family that erroneously
    forwarded the knob would raise ``TypeError`` here."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, *, text, voice=None, speed=1.0, lang_code=None):
        import numpy as np

        self.calls.append(
            {"text": text, "voice": voice, "speed": speed, "lang_code": lang_code}
        )
        result = types.SimpleNamespace(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000
        )
        return iter([result])


def _chatterbox_engine(model_name: str = CHATTERBOX_FP16):
    from vllm_mlx.audio.tts import TTSEngine

    engine = TTSEngine(model_name)
    engine.model = _CapturingChatterbox()
    engine._loaded = True
    return engine


class TestChatterboxEngine:
    @pytest.mark.parametrize("model_name", [CHATTERBOX_FP16, CHATTERBOX_TURBO])
    def test_family_detected(self, model_name):
        from vllm_mlx.audio.tts import TTSEngine

        # Both the non-turbo and turbo repos detect as the same family
        # (they load the same ``chatterbox.Model``), so the exaggeration
        # branch fires for both.
        assert TTSEngine(model_name)._model_family == "chatterbox"

    def test_forwards_exaggeration_when_set(self):
        engine = _chatterbox_engine()
        engine.generate("Big news today!", exaggeration=0.9)
        (call,) = engine.model.calls
        assert call["exaggeration"] == 0.9
        assert call["text"] == "Big news today!"

    def test_omits_exaggeration_when_absent(self):
        engine = _chatterbox_engine()
        engine.generate("neutral narration")
        (call,) = engine.model.calls
        # No knob → the model's own default holds (we never pass it).
        assert "exaggeration" not in call

    def test_does_not_forward_voice_speed_lang_code(self):
        """The route always hands the engine Kokoro's ``voice``/``speed``
        defaults; the chatterbox branch must NOT relay them (the strict
        fake would raise ``TypeError`` if it did)."""
        engine = _chatterbox_engine()
        engine.generate("hi", voice="af_heart", speed=1.5, lang_code="a")
        (call,) = engine.model.calls
        assert set(call) == {"text"}

    def test_forwards_ref_audio_for_cloning(self):
        engine = _chatterbox_engine()
        engine.generate("hi", ref_audio="/tmp/narrator.wav")
        (call,) = engine.model.calls
        assert call["ref_audio"] == "/tmp/narrator.wav"
        assert "exaggeration" not in call

    def test_forwards_both_exaggeration_and_ref_audio(self):
        engine = _chatterbox_engine()
        engine.generate("hi", exaggeration=0.7, ref_audio="/tmp/narrator.wav")
        (call,) = engine.model.calls
        assert call["exaggeration"] == 0.7
        assert call["ref_audio"] == "/tmp/narrator.wav"

    def test_exaggeration_not_leaked_to_kokoro_family(self):
        """A Kokoro engine handed ``exaggeration`` (e.g. a client sending
        the knob to a non-expressive model) must NOT forward it — the
        Kokoro path has no such kwarg and the strict fake would raise."""
        from vllm_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine.model = _CapturingKokoro()
        engine._loaded = True
        engine.generate("hello", voice="af_heart", exaggeration=0.9)
        (call,) = engine.model.calls
        assert "exaggeration" not in call
        # Kokoro keeps its single-letter lang_code — the branch is unchanged.
        assert call["lang_code"] == "a"


# ---------------------------------------------------------------------------
# C) Route — ``/v1/audio/speech`` threads ``exaggeration`` to the engine
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
    a real WAV via the actual encoder. ``exaggeration`` uses a sentinel
    default so a call is recorded as carrying it ONLY when the route
    actually forwarded it (proving conditional forwarding)."""

    instances: list[_RecordingEngine] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generate_calls: list[dict] = []
        _RecordingEngine.instances.append(self)

    def load(self):
        pass

    def generate(
        self,
        text,
        voice="af_heart",
        speed=1.0,
        instruct=None,
        exaggeration=_UNSET,
        ref_audio=None,
        ref_text=None,
    ):
        import numpy as np

        rec: dict = {"text": text, "voice": voice, "speed": speed}
        if exaggeration is not _UNSET:
            rec["exaggeration"] = exaggeration
        if ref_audio is not None:
            rec["ref_audio"] = ref_audio
        if ref_text is not None:
            rec["ref_text"] = ref_text
        self.generate_calls.append(rec)
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
    _RecordingEngine._real_to_bytes = tts_mod.TTSEngine.to_bytes
    _install_fake_mlx_audio(monkeypatch)
    monkeypatch.setattr(probe_mod, "require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda: None)
    monkeypatch.setattr(tts_mod, "TTSEngine", _RecordingEngine)
    # Force snapshot enumeration empty so the route uses the static
    # chatterbox voice list (``["default"]``) — matches a fresh install.
    monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _n: [])
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    app = FastAPI()
    app.include_router(audio_route.router)
    install_exception_handlers(app)
    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", None)
    return TestClient(app)


class TestChatterboxRoute:
    def test_forwards_exaggeration_to_engine(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "chatterbox",
                "input": "This changes everything!",
                "exaggeration": 0.85,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        (engine,) = _RecordingEngine.instances
        assert engine.model_name == CHATTERBOX_TURBO
        (call,) = engine.generate_calls
        assert call["exaggeration"] == 0.85

    def test_omits_exaggeration_when_absent(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "chatterbox", "input": "neutral narration"},
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        # Not sent → the route must NOT inject the kwarg, so the engine's
        # own default holds.
        assert "exaggeration" not in call

    def test_exaggeration_accepted_against_non_chatterbox_model(self, monkeypatch):
        """OpenAI parity: a styling field sent to a family that ignores it
        must NOT 400. The route forwards it; the engine drops it for
        non-chatterbox families. Here we only assert the route accepts it
        and threads it through (the stub records it regardless of family)."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "hello", "exaggeration": 0.5},
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        assert call["exaggeration"] == 0.5

    @pytest.mark.parametrize("value", [-0.5, 2.5])
    def test_out_of_range_exaggeration_returns_400_envelope(self, monkeypatch, value):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "chatterbox", "input": "hi", "exaggeration": value},
        )
        assert resp.status_code == 400, resp.text
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "exaggeration", err
        # No engine should have been constructed — validation fires before
        # any weight load.
        assert _RecordingEngine.instances == []


# ---------------------------------------------------------------------------
# D) Route — Chatterbox zero-shot cloning IS reachable via /v1/audio/speech
# ---------------------------------------------------------------------------

# A minimal valid base64 WAV payload (decodes to ``b"RIFF"``) — enough to
# pass ``_decode_tts_ref_audio``'s non-empty / valid-base64 checks without
# shipping a real clip. Same fixture the F5 wiring test uses.
_REF_AUDIO_B64 = "data:audio/wav;base64,UklGRg=="


class TestChatterboxCloningRoute:
    def test_ref_audio_clones_for_chatterbox_end_to_end(self, monkeypatch):
        """Codex #1 regression: the engine's chatterbox ``ref_audio`` branch
        must be reachable over HTTP, not just via the Python engine API.
        The route gate now admits the Chatterbox family; the reference clip
        is decoded to a temp file whose path reaches the engine."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "chatterbox",
                "input": "Cloned narrator speaking.",
                "ref_audio": _REF_AUDIO_B64,
                # The shared F5 validator requires the pair; Chatterbox
                # ignores the transcript but the caller still supplies it.
                "ref_text": "reference transcript",
            },
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        assert engine.model_name == CHATTERBOX_TURBO
        (call,) = engine.generate_calls
        # The engine received a decoded reference path (not the raw base64).
        assert call.get("ref_audio")
        assert not call["ref_audio"].startswith("data:")

    def test_ref_audio_rejected_for_non_cloning_family(self, monkeypatch):
        """A ``ref_audio`` aimed at a family with no cloning surface (Kokoro)
        is still a 400 ``unsupported_voice_cloning`` — the gate widened to
        Chatterbox, it did not open to everyone."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "hello",
                "ref_audio": _REF_AUDIO_B64,
                "ref_text": "reference transcript",
            },
        )
        assert resp.status_code == 400, resp.text
        err = resp.json()["error"]
        assert err["code"] == "unsupported_voice_cloning", err
        assert _RecordingEngine.instances == []

    def test_ref_audio_without_ref_text_rejected(self, monkeypatch):
        """The shared F5 pairing invariant still holds: ``ref_audio`` without
        ``ref_text`` is a validation 400 before any engine is built. (Fully
        transcript-free Chatterbox cloning would require relaxing the shared
        validator and is intentionally left to a follow-up.)"""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "chatterbox",
                "input": "hello",
                "ref_audio": _REF_AUDIO_B64,
            },
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert _RecordingEngine.instances == []
