# SPDX-License-Identifier: Apache-2.0
"""End-to-end ``POST /v1/audio/speech`` inline voice-cloning contract.

The Qwen3-TTS **Base** clone path (alias ``qwen3-tts-clone``) has to travel
the FULL route, not just the engine. Three route hazards are pinned here,
all hermetic (no weights, no network — the ``TTSEngine`` is stubbed and the
``mlx_audio`` probe is faked):

1. A clone-capable model (Qwen3-TTS Base / F5-TTS) with an inline
   ``ref_audio`` reference clip must SKIP the named-speaker allowlist and
   OMIT ``voice`` from the generate call — the Base repo's registry
   ``default_voice`` is the ``"clone"`` sentinel, which is NOT a member of
   the speaker allowlist, so running the allowlist would 400 the very
   request we mean to serve.
2. A Base repo WITHOUT a reference still 400s with an actionable message
   (Base cannot synthesize reference-free), while WITH a reference it is
   the correct target and must reach the engine.
3. A reference clip aimed at a non-clone model (Qwen3-TTS CustomVoice)
   is rejected up front rather than silently ignored.

Uses the same ``mlx_audio``-fake + stubbed-``TTSEngine`` harness as
``tests/test_qwen3_tts_audio.py``, with a clone-aware recording engine that
records exactly which kwargs the route forwarded so the "voice omitted"
contract can be asserted.
"""

from __future__ import annotations

import base64
import importlib.machinery
import sys
import types

import pytest

pytestmark = pytest.mark.requires_mlx

CLONE_BASE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
CUSTOMVOICE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"

# A tiny non-empty payload — the route only base64-decodes and size-bounds
# it (see ``_decode_tts_ref_audio``); the stubbed engine never opens it.
_REF_WAV_B64 = base64.b64encode(b"RIFF----WAVEfake-reference-clip").decode()
_UNSET = object()


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


class _CloneRecordingEngine:
    """TTSEngine stub that records generate() kwargs, using sentinel
    defaults so the test can distinguish "``voice`` was forwarded" from
    "``voice`` was omitted" — the crux of the inline-clone contract."""

    instances: list[_CloneRecordingEngine] = []
    _real_to_bytes = None

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generate_calls: list[dict] = []
        _CloneRecordingEngine.instances.append(self)

    def load(self):
        pass

    def generate(
        self,
        text,
        voice=_UNSET,
        speed=1.0,
        instruct=None,
        ref_audio=_UNSET,
        ref_text=_UNSET,
    ):
        import numpy as np

        rec: dict = {"text": text, "speed": speed, "instruct": instruct}
        if voice is not _UNSET:
            rec["voice"] = voice
        if ref_audio is not _UNSET:
            rec["ref_audio"] = ref_audio
        if ref_text is not _UNSET:
            rec["ref_text"] = ref_text
        self.generate_calls.append(rec)

        from vllm_mlx.audio.tts import AudioOutput

        return AudioOutput(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000, duration=0.01
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

    _CloneRecordingEngine.instances = []
    _CloneRecordingEngine._real_to_bytes = tts_mod.TTSEngine.to_bytes
    _install_fake_mlx_audio(monkeypatch)
    monkeypatch.setattr(probe_mod, "require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tts_mod, "TTSEngine", _CloneRecordingEngine)
    # Cold-start: force snapshot enumeration empty so the route uses the
    # static per-family voice list (matches a fresh install).
    monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _n: [])
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    app = FastAPI()
    app.include_router(audio_route.router)
    install_exception_handlers(app)
    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", None)
    return TestClient(app)


class TestInlineCloneRoute:
    def test_base_clone_omits_voice_and_forwards_reference(self, monkeypatch):
        """qwen3-tts-clone + inline ref_audio/ref_text, ``voice`` omitted:
        200, the engine receives the reference clip AND its transcript, and
        ``voice`` is NOT forwarded (the ``"clone"`` sentinel never reaches
        the engine, and no ``invalid_voice`` 400 is raised)."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts-clone",
                "input": "你好世界。",
                "ref_audio": _REF_WAV_B64,
                "ref_text": "这是参考音频。",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        (engine,) = _CloneRecordingEngine.instances
        assert engine.model_name == CLONE_BASE_BF16
        (call,) = engine.generate_calls
        # The clone surface reached the engine…
        assert call.get("ref_audio") is not None
        assert call["ref_text"] == "这是参考音频。"
        # …and ``voice`` was OMITTED from the generate call.
        assert "voice" not in call

    def test_base_clone_ignores_explicit_voice(self, monkeypatch):
        """Even an explicit named speaker is dropped on a clone request —
        the reference clip governs the timbre, so the route must neither
        400 on it nor forward it to the engine."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts-clone",
                "input": "你好世界。",
                "voice": "Serena",
                "ref_audio": _REF_WAV_B64,
                "ref_text": "这是参考音频。",
            },
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _CloneRecordingEngine.instances
        (call,) = engine.generate_calls
        assert "voice" not in call
        assert call.get("ref_audio") is not None

    def test_base_without_reference_400s_actionable(self, monkeypatch):
        """qwen3-tts-clone with NO reference: 400 ``unsupported_model_variant``
        (Base cannot synthesize reference-free). The message must point at
        BOTH remedies — supply ref_audio, or use a CustomVoice repo — and
        no engine is constructed."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts-clone", "input": "你好世界。"},
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "unsupported_model_variant"
        msg = body["error"]["message"]
        assert "ref_audio" in msg
        assert "CustomVoice" in msg
        assert _CloneRecordingEngine.instances == []

    def test_customvoice_with_reference_rejected(self, monkeypatch):
        """A reference clip aimed at the reference-free CustomVoice variant
        is rejected up front (``unsupported_voice_cloning``), not silently
        ignored — CustomVoice has no cloning surface."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",  # CustomVoice alias
                "input": "你好世界。",
                "ref_audio": _REF_WAV_B64,
                "ref_text": "这是参考音频。",
            },
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "unsupported_voice_cloning"
        assert _CloneRecordingEngine.instances == []

    def test_real_customvoice_not_swept_by_base_guard(self, monkeypatch):
        """Regression pin for the classification: the Base-reject guard is
        keyed on the repo-name tokens, so a real CustomVoice repo is NOT
        swept up by the Base guard even though both share the qwen3-tts
        prefix — it synthesizes normally (reference-free)."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "你好世界。", "voice": "Serena"},
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _CloneRecordingEngine.instances
        (call,) = engine.generate_calls
        assert call["voice"] == "Serena"


class TestF5InlineCloneRoute:
    def test_f5_clone_forwards_reference_and_omits_voice(self, monkeypatch):
        """F5-TTS is clone-capable too: an inline reference reaches the
        engine and ``voice`` is omitted (F5 has no named-speaker surface),
        proving the clone gate is not Qwen3-only."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "f5-tts-zh",
                "input": "你好世界。",
                "ref_audio": _REF_WAV_B64,
                "ref_text": "这是参考音频。",
            },
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _CloneRecordingEngine.instances
        (call,) = engine.generate_calls
        assert call.get("ref_audio") is not None
        assert call["ref_text"] == "这是参考音频。"
        assert "voice" not in call


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", True),
        ("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16", False),
        ("lucasnewman/f5-tts-mlx", True),
        # An org segment carrying ``customvoice`` must NOT suppress a real
        # Base repo (classification is on the repo NAME's tokens).
        ("customvoice-org/Qwen3-TTS-0.6B-Base", True),
        # An unrelated ``base`` in the ORG segment must NOT make a
        # CustomVoice repo look clone-capable.
        ("base-org/Qwen3-TTS-0.6B-CustomVoice", False),
        ("mlx-community/Kokoro-82M-bf16", False),
        # A bare ``f5`` token that is NOT ``f5-tts``/``f5_tts`` must NOT be
        # deemed clone-capable: TTSEngine._detect_family classifies it as
        # Kokoro and would drop ref_audio, so the gate must agree and let
        # the reference be rejected up front rather than silently ignored.
        ("org/f5-foo", False),
        ("org/f5", False),
    ],
)
def test_is_clone_capable_model_classification(model_name, expected):
    from vllm_mlx.routes.audio import _is_clone_capable_model

    assert _is_clone_capable_model(model_name) is expected


def test_clone_gate_matches_engine_family_for_f5_token():
    """Regression pin (codex round 2): the route clone gate and the engine
    family classifier must never disagree in the DANGEROUS direction —
    gate says clone-capable but engine drops ref_audio. For an ``f5`` token
    that is not ``f5-tts``/``f5_tts`` the engine detects the default
    (Kokoro) family, so the gate must NOT deem it clone-capable."""
    from vllm_mlx.audio.tts import TTSEngine
    from vllm_mlx.routes.audio import _is_clone_capable_model

    name = "org/f5-foo"
    assert _is_clone_capable_model(name) is False
    # Engine would NOT treat it as F5 (family falls back to kokoro), which
    # is exactly why the gate must reject the reference.
    assert TTSEngine(name)._model_family != "f5"
