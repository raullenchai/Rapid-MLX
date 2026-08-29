# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS *VoiceDesign* audio support.

VoiceDesign is the sibling of CustomVoice: it shares the ``qwen3_tts``
family (same mlx_audio loader + ``generate()`` entry) but has NO named
speakers. The whole voice — timbre, gender, age, accent, emotion, prosody —
is authored in natural language via the OpenAI ``instructions`` field
(mapped to mlx_audio's ``generate_voice_design`` ``instruct`` argument,
which is a MANDATORY positional with no default). ``voice`` is ignored.

These tests are hermetic — no weights, no network. They fake the
``mlx_audio`` boundary and pin the contract:

* the three ``qwen3-tts-voicedesign`` aliases resolve through the central
  registry to the mlx-community VoiceDesign repos with ``family="qwen3_tts"``
  and the ``describe`` sentinel as ``default_voice``;
* ``TTSEngine`` detects the family as ``qwen3_tts`` AND flags the checkpoint
  as VoiceDesign (``_is_qwen3_voicedesign``), while a CustomVoice checkpoint
  is NOT flagged;
* ``generate`` ALWAYS forwards ``instruct`` for VoiceDesign — the caller's
  value when supplied, else the neutral-narrator fallback — so the mandatory
  ``generate_voice_design`` arg is never missing (guarding the TypeError the
  pre-fix ``if instruct:`` path would raise deep in mlx_audio);
* the voice surface is the single ``describe`` sentinel (not the nine
  CustomVoice speakers) on both the engine (``get_voices``) and the route
  (``_allowed_voices_for``); the route resolves the omitted / ``describe``
  voice and threads ``instructions`` down to the engine.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

pytestmark = pytest.mark.requires_mlx

VOICEDESIGN_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
CUSTOMVOICE_BF16 = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"


# ---------------------------------------------------------------------------
# A) Registry resolution + family/variant detection
# ---------------------------------------------------------------------------


class TestVoiceDesignRegistry:
    @pytest.mark.parametrize(
        "alias,expected_hf_id",
        [
            ("qwen3-tts-voicedesign", VOICEDESIGN_BF16),
            (
                "qwen3-tts-voicedesign-8bit",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            ),
            (
                "qwen3-tts-voicedesign-4bit",
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
            ),
            # Case-insensitive: docs/SDKs mix case off the upstream repo id.
            ("Qwen3-TTS-VoiceDesign", VOICEDESIGN_BF16),
        ],
    )
    def test_alias_resolves(self, alias, expected_hf_id):
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(alias)
        assert entry is not None, f"{alias!r} did not resolve in the registry"
        assert entry.type == "tts"
        # VoiceDesign REUSES the qwen3_tts family (same loader) — the
        # CustomVoice/VoiceDesign split is a per-request generate distinction.
        assert entry.family == "qwen3_tts"
        assert entry.hf_id == expected_hf_id
        # No named speakers → the sentinel, not a CustomVoice speaker.
        assert entry.default_voice == "describe"

    def test_hf_id_reverse_lookup(self):
        """The full HF id maps back to the qwen3_tts entry so ``serve
        <hf-id>`` forks into audio mode like the short alias does."""
        from vllm_mlx.audio.registry import resolve_audio_alias

        entry = resolve_audio_alias(VOICEDESIGN_BF16)
        assert entry is not None and entry.family == "qwen3_tts"

    def test_default_voice_is_the_sentinel_not_a_speaker(self):
        """VoiceDesign's registry ``default_voice`` must be the ``describe``
        sentinel (its only allowed voice), NOT a CustomVoice speaker — else
        the voice-omitted / cold-start path would validate against a name the
        VoiceDesign surface no longer advertises."""
        from vllm_mlx.audio.registry import resolve_audio_alias
        from vllm_mlx.audio.tts import (
            QWEN3_TTS_VOICEDESIGN_VOICES,
            QWEN3_TTS_VOICES,
        )

        entry = resolve_audio_alias("qwen3-tts-voicedesign")
        assert entry.default_voice in QWEN3_TTS_VOICEDESIGN_VOICES
        assert entry.default_voice not in QWEN3_TTS_VOICES


# ---------------------------------------------------------------------------
# B) Engine — variant flag, voice surface, mandatory-instruct forwarding
# ---------------------------------------------------------------------------


class _VoiceDesignModel:
    """Fake mlx_audio VoiceDesign model. ``generate`` mirrors the real
    ``generate_voice_design`` contract where ``instruct`` is MANDATORY (no
    default): if the engine fails to forward it the call raises ``TypeError``
    here — exactly the failure the fix guards against — instead of silently
    passing. ``voice`` is accepted but ignored (VoiceDesign drops it)."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, *, text, instruct, voice=None, speed=1.0, lang_code=None):
        import numpy as np

        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang_code": lang_code,
                "instruct": instruct,
            }
        )
        result = types.SimpleNamespace(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000
        )
        return iter([result])


def _voicedesign_engine():
    """A loaded VoiceDesign ``TTSEngine`` whose model is the mandatory-instruct
    fake (no weights, no network)."""
    from vllm_mlx.audio.tts import TTSEngine

    engine = TTSEngine(VOICEDESIGN_BF16)
    engine.model = _VoiceDesignModel()
    engine._loaded = True
    return engine


class TestVoiceDesignEngine:
    def test_family_and_variant_flag(self):
        from vllm_mlx.audio.tts import TTSEngine

        vd = TTSEngine(VOICEDESIGN_BF16)
        assert vd._model_family == "qwen3_tts"
        assert vd._is_qwen3_voicedesign() is True

        # The CustomVoice sibling shares the family but is NOT flagged.
        cv = TTSEngine(CUSTOMVOICE_BF16)
        assert cv._model_family == "qwen3_tts"
        assert cv._is_qwen3_voicedesign() is False

    def test_customvoice_token_wins_over_namespace_voicedesign(self):
        """A ``voicedesign`` token in the ORG/namespace must NOT flip a
        CustomVoice checkpoint to VoiceDesign — the explicit ``customvoice``
        variant token wins. A naive whole-id ``voicedesign`` match would wrongly
        misclassify it (swap its voice surface / force a neutral instruct)."""
        from vllm_mlx.audio.tts import TTSEngine

        cv = TTSEngine("voicedesign-org/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
        assert cv._is_qwen3_voicedesign() is False
        assert cv.get_voices()[:1] == ["Vivian"]  # CustomVoice speaker set

    def test_registry_is_authoritative_for_known_aliases(self):
        """A registered alias / canonical HF id is classified off registry
        metadata (its clean canonical ``hf_id``), not off whatever local path
        it happens to be loaded from — so the supported inputs never depend on
        the name heuristic. Pin the module-level classifier directly."""
        from vllm_mlx.audio.tts import (
            is_qwen3_voicedesign_model as is_vd,
        )

        # Short aliases resolve to the canonical VoiceDesign / CustomVoice ids.
        assert is_vd("qwen3-tts-voicedesign") is True
        assert is_vd("qwen3-tts-voicedesign-4bit") is True
        assert is_vd("qwen3-tts") is False
        assert is_vd("qwen3-tts-customvoice") is False
        # Canonical HF ids (case-insensitive registry reverse-lookup).
        assert is_vd(VOICEDESIGN_BF16) is True
        assert is_vd(CUSTOMVOICE_BF16) is False

    def test_variant_token_read_from_repo_component_not_parent_dirs(self):
        """The variant token is read ONLY from the repo-NAME component, never a
        parent directory or org namespace. A VoiceDesign checkpoint under a
        ``customvoice`` parent dir (or vice versa) is classified by its own
        name, not the coincidental ancestor token."""
        from vllm_mlx.audio.tts import is_qwen3_voicedesign_model as is_vd

        # VoiceDesign repo under a 'customvoice' parent dir → VoiceDesign.
        assert is_vd("/srv/customvoice/Qwen3-TTS-VoiceDesign-bf16") is True
        # CustomVoice repo under a 'voicedesign' parent dir → CustomVoice.
        assert is_vd("/srv/voicedesign/Qwen3-TTS-CustomVoice-bf16") is False
        # Trailing separator must not blank out the final component.
        assert is_vd("/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/") is True
        assert is_vd("/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16/") is False

    def test_detects_voicedesign_in_hf_cache_snapshot_path(self):
        """A resolved HuggingFace cache path ends in ``/snapshots/<hash>`` — a
        basename-only check would see the opaque commit hash and misclassify the
        checkpoint. Detection is on the full identifier, so the VoiceDesign
        cache path is still recognised (and the CustomVoice one is not)."""
        from vllm_mlx.audio.tts import (
            QWEN3_TTS_VOICEDESIGN_VOICES,
            TTSEngine,
        )

        vd_cache = (
            "/home/u/.cache/huggingface/hub/"
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/"
            "snapshots/0123456789abcdef0123456789abcdef01234567"
        )
        cv_cache = (
            "/home/u/.cache/huggingface/hub/"
            "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16/"
            "snapshots/0123456789abcdef0123456789abcdef01234567"
        )
        vd = TTSEngine(vd_cache)
        assert vd._is_qwen3_voicedesign() is True
        assert vd.get_voices() == list(QWEN3_TTS_VOICEDESIGN_VOICES)
        cv = TTSEngine(cv_cache)
        assert cv._is_qwen3_voicedesign() is False
        assert cv.get_voices()[:1] == ["Vivian"]

    def test_engine_and_route_classify_consistently(self, monkeypatch):
        """The engine (voice surface + generate dispatch) and the route (voice
        allowlist) MUST agree on every id — a split classifier would let a
        request synthesize as VoiceDesign while being validated against the
        CustomVoice speakers, or vice versa. Pin both against the shared
        classifier across the tricky shapes: a VoiceDesign token in the ORG
        (``qwen3-tts-org/VoiceDesign-checkpoint``), a ``voicedesign`` token in
        the ORG of a CustomVoice repo (must stay CustomVoice), and resolved HF
        cache snapshot paths (basename is the commit hash)."""
        from vllm_mlx.audio import tts as tts_mod
        from vllm_mlx.audio.tts import (
            QWEN3_TTS_VOICEDESIGN_VOICES,
            QWEN3_TTS_VOICES,
            TTSEngine,
        )
        from vllm_mlx.routes.audio import _allowed_voices_for

        # Cold-start: force snapshot enumeration empty so the route validates
        # against the static per-family classifier (the shared helper) rather
        # than trying to stat a non-existent local snapshot dir.
        monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _n: [])

        cache = "/home/u/.cache/huggingface/hub/models--mlx-community--"
        snap = "/snapshots/0123456789abcdef0123456789abcdef01234567"
        cases = {
            VOICEDESIGN_BF16: QWEN3_TTS_VOICEDESIGN_VOICES,
            CUSTOMVOICE_BF16: QWEN3_TTS_VOICES,
            # qwen family matched on the full id (org token), VoiceDesign token
            # present, no customvoice token → VoiceDesign.
            "qwen3-tts-org/VoiceDesign-checkpoint": QWEN3_TTS_VOICEDESIGN_VOICES,
            # voicedesign in the org of a CustomVoice repo → customvoice wins.
            "voicedesign-org/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16": QWEN3_TTS_VOICES,
            # Resolved HF cache snapshot paths (basename is the commit hash).
            f"{cache}Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16{snap}": (
                QWEN3_TTS_VOICEDESIGN_VOICES
            ),
            f"{cache}Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16{snap}": QWEN3_TTS_VOICES,
        }
        for model_name, expected in cases.items():
            engine = TTSEngine(model_name)
            assert engine.get_voices() == list(expected), model_name
            assert _allowed_voices_for(model_name) == list(expected), model_name

    def test_get_voices_is_describe_sentinel(self):
        from vllm_mlx.audio.tts import (
            QWEN3_TTS_VOICEDESIGN_VOICES,
            QWEN3_TTS_VOICES,
        )

        voices = _voicedesign_engine().get_voices()
        assert voices == list(QWEN3_TTS_VOICEDESIGN_VOICES)
        # Not the CustomVoice speaker list — ``voice`` is meaningless here.
        assert "Serena" not in voices
        assert voices != list(QWEN3_TTS_VOICES)

    def test_generate_without_instruct_uses_fallback_and_does_not_raise(self):
        """The guard: a VoiceDesign generate with NO ``instruct`` must NOT
        crash. Pre-fix the ``if instruct:`` branch forwarded nothing, so the
        mandatory ``generate_voice_design`` arg was missing → TypeError deep
        in mlx_audio (the mandatory-arg fake reproduces that). The fix always
        forwards a description, falling back to the neutral narrator."""
        from vllm_mlx.audio.tts import QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT

        engine = _voicedesign_engine()
        # Must not raise despite no instruct supplied.
        engine.generate("讲一个故事。", voice="describe")
        (call,) = engine.model.calls
        assert call["instruct"] == QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT
        # Qwen3 auto-detects language; Kokoro's "a" would mis-hint it.
        assert call["lang_code"] == "auto"

    def test_generate_empty_instruct_uses_fallback(self):
        """An empty-string ``instruct`` (what a blank ``instructions`` field
        yields) is falsy and must also fall back — never forwarded as ''."""
        from vllm_mlx.audio.tts import QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT

        engine = _voicedesign_engine()
        engine.generate("讲一个故事。", voice="describe", instruct="")
        (call,) = engine.model.calls
        assert call["instruct"] == QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT

    def test_generate_forwards_caller_instruct(self):
        """When the caller supplies a description it is the PRIMARY control
        and is forwarded verbatim (not the fallback)."""
        engine = _voicedesign_engine()
        engine.generate(
            "讲一个故事。",
            voice="describe",
            instruct="a warm, low female narrator, calm and measured",
        )
        (call,) = engine.model.calls
        assert call["instruct"] == "a warm, low female narrator, calm and measured"
        assert call["lang_code"] == "auto"

    def test_voice_seed_is_deterministic_and_restores_mlx_rng(self):
        import mlx.core as mx
        import numpy as np

        globals()["categorical_sampling"] = lambda logits, temperature: (
            mx.random.categorical(logits * (1 / temperature))
        )

        class _RandomVoiceDesignModel:
            def generate(
                self, *, text, instruct, voice=None, speed=1.0, lang_code=None
            ):
                del text, instruct, voice, speed, lang_code
                logits = mx.zeros((32, 16))
                yield types.SimpleNamespace(
                    audio=globals()["categorical_sampling"](logits, 1.0).astype(
                        mx.float32
                    ),
                    sample_rate=24000,
                )

        engine = _voicedesign_engine()
        engine.model = _RandomVoiceDesignModel()
        before = [np.array(value) for value in mx.random.state]

        try:
            first = engine.generate(
                "第一句。", instruct="a warm narrator", voice_seed=20260731
            )
            second = engine.generate(
                "第二句。", instruct="a warm narrator", voice_seed=20260731
            )

            np.testing.assert_array_equal(first.audio, second.audio)
            for expected, actual in zip(before, mx.random.state, strict=True):
                np.testing.assert_array_equal(expected, np.array(actual))
        finally:
            globals().pop("categorical_sampling", None)


# ---------------------------------------------------------------------------
# C) Route — voice surface + instructions plumbing
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
    """No-op TTSEngine stub that records the generate() kwargs and returns a
    real WAV via the actual encoder."""

    instances: list[_RecordingEngine] = []

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generate_calls: list[dict] = []
        _RecordingEngine.instances.append(self)

    def load(self):
        pass

    def generate(
        self, text, voice="af_heart", speed=1.0, instruct=None, voice_seed=None
    ):
        import numpy as np

        self.generate_calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "instruct": instruct,
                "voice_seed": voice_seed,
            }
        )
        from vllm_mlx.audio.tts import AudioOutput

        return AudioOutput(
            audio=np.zeros(240, dtype=np.float32), sample_rate=24000, duration=0.01
        )

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
    monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda *a, **k: None)
    monkeypatch.setattr(tts_mod, "TTSEngine", _RecordingEngine)
    # Cold-start voice path: force snapshot enumeration to empty so the route
    # uses the static per-family list (matches a fresh install).
    monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _n: [])
    monkeypatch.setattr(audio_route, "_tts_engine", None)

    app = FastAPI()
    app.include_router(audio_route.router)
    install_exception_handlers(app)
    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", None)
    return TestClient(app)


class TestVoiceDesignRoute:
    @pytest.mark.parametrize("seed", [-1, 4_294_967_296, True, "7"])
    def test_voice_seed_schema_rejects_invalid_values(self, seed):
        from pydantic import ValidationError

        from vllm_mlx.api.models import AudioSpeechRequest

        with pytest.raises(ValidationError):
            AudioSpeechRequest(input="hello", voice_seed=seed)

    def test_voices_route_lists_describe_sentinel(self, monkeypatch):
        """``GET /v1/audio/voices`` for a VoiceDesign model advertises the
        ``describe`` sentinel — NOT the CustomVoice speakers."""
        client = _mount(monkeypatch)
        resp = client.get("/v1/audio/voices", params={"model": "qwen3-tts-voicedesign"})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"voices": ["describe"]}

    def test_speech_forwards_instructions_as_instruct(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts-voicedesign",
                "input": "他被诸葛亮压了一辈子。",
                "voice": "describe",
                "instructions": "a hoarse elderly male storyteller, slow and grave",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "audio/wav"
        (engine,) = _RecordingEngine.instances
        assert engine.model_name == VOICEDESIGN_BF16
        (call,) = engine.generate_calls
        assert call["instruct"] == "a hoarse elderly male storyteller, slow and grave"

    def test_speech_forwards_and_echoes_voice_seed(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts-voicedesign",
                "input": "第一章。",
                "instructions": "a warm, low narrator",
                "voice_seed": 8675309,
            },
        )

        assert resp.status_code == 200, resp.text
        assert resp.headers["x-voice-seed"] == "8675309"
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        assert call["voice_seed"] == 8675309

    def test_voice_seed_is_rejected_for_non_voicedesign_model(self, monkeypatch):
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello.",
                "voice": "Serena",
                "voice_seed": 7,
            },
        )

        assert resp.status_code == 400, resp.text
        error = resp.json()["error"]
        assert error["code"] == "unsupported_voice_seed"
        assert error["param"] == "voice_seed"
        assert _RecordingEngine.instances == []

    @pytest.mark.parametrize(
        "body",
        [
            {"model": "qwen3-tts-voicedesign", "input": "你好世界。"},
            {
                "model": "qwen3-tts-voicedesign",
                "input": "你好世界。",
                "voice": "default",
            },
            {
                "model": "qwen3-tts-voicedesign",
                "input": "你好世界。",
                "voice": "describe",
            },
        ],
        ids=["omitted", "explicit-default", "explicit-describe"],
    )
    def test_voice_omitted_or_sentinel_validates(self, monkeypatch, body):
        """The omitted / ``default`` / ``describe`` voice all pass validation
        (the registry default resolves to the ``describe`` sentinel, which is
        the sole allowed voice) rather than 400."""
        client = _mount(monkeypatch)
        resp = client.post("/v1/audio/speech", json=body)
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        assert call["voice"] == "describe"

    def test_customvoice_speaker_rejected_for_voicedesign(self, monkeypatch):
        """A CustomVoice speaker (``Serena``) is NOT a valid VoiceDesign voice:
        the surface is the ``describe`` sentinel, so the route 400s with the
        sentinel in the ``Available:`` preview instead of silently synthesizing
        with a meaningless speaker."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts-voicedesign",
                "input": "测试。",
                "voice": "Serena",
            },
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "invalid_voice"
        assert "describe" in body["error"]["message"]

    def test_org_prefix_does_not_misclassify_customvoice(self, monkeypatch):
        """A ``voicedesign`` token in the org segment must NOT collapse a
        CustomVoice repo's voice surface to the ``describe`` sentinel: the
        route classifies on the repo basename, so its speakers stay valid."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": "voicedesign-org/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                "input": "测试。",
                "voice": "Serena",
            },
        )
        # Serena is a valid CustomVoice speaker — accepted, not 400'd as if
        # the surface were VoiceDesign's ``describe`` sentinel.
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        assert call["voice"] == "Serena"

    def test_route_omits_instruct_when_no_instructions(self, monkeypatch):
        """A VoiceDesign request with no ``instructions`` is accepted (the field
        is optional) and the route forwards NO ``instruct`` to the engine — it
        is the ENGINE's ``generate`` that then substitutes the neutral-narrator
        fallback (covered end-to-end against the real ``TTSEngine.generate`` in
        ``TestVoiceDesignEngine.test_generate_without_instruct_...``). This test
        pins only the route contract: the field is not required and no empty
        instruct is fabricated at the route layer."""
        client = _mount(monkeypatch)
        resp = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts-voicedesign", "input": "你好世界。"},
        )
        assert resp.status_code == 200, resp.text
        (engine,) = _RecordingEngine.instances
        (call,) = engine.generate_calls
        # Route did not fabricate an instruct; engine-layer fallback owns it.
        assert call["instruct"] is None

    def test_end_to_end_fallback_reaches_model(self, monkeypatch):
        """The whole stack: route → REAL ``TTSEngine.generate`` → model. Only
        the mlx_audio model boundary is faked (a mandatory-``instruct`` model),
        so a no-``instructions`` VoiceDesign request must still succeed AND the
        model must actually receive the neutral-narrator fallback. Unlike the
        stub-engine tests, this fails if the production fallback is removed."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.audio import probe as probe_mod
        from vllm_mlx.audio import tts as tts_mod
        from vllm_mlx.config import get_config
        from vllm_mlx.middleware.exception_handlers import install_exception_handlers
        from vllm_mlx.routes import audio as audio_route

        # Fake the mlx_audio.tts.generate.load_model boundary to return a
        # mandatory-instruct VoiceDesign model (no weights, no network).
        captured = _VoiceDesignModel()
        fake_gen = types.ModuleType("mlx_audio.tts.generate")
        fake_gen.load_model = lambda _name: captured
        _install_fake_mlx_audio(monkeypatch)
        monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", fake_gen)
        monkeypatch.setattr(probe_mod, "require_mlx_audio_tts", lambda: None)
        monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda *a, **k: None)
        monkeypatch.setattr(tts_mod, "_list_snapshot_voices", lambda _n: [])
        monkeypatch.setattr(audio_route, "_tts_engine", None)

        app = FastAPI()
        app.include_router(audio_route.router)
        install_exception_handlers(app)
        monkeypatch.setattr(get_config(), "api_key", None)
        client = TestClient(app)

        resp = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts-voicedesign", "input": "你好世界。"},
        )
        assert resp.status_code == 200, resp.text
        (call,) = captured.calls
        assert call["instruct"] == tts_mod.QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT
        assert call["lang_code"] == "auto"
