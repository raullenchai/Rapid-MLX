# SPDX-License-Identifier: Apache-2.0
"""R11-B regression bundle — Bo's 0.8.12 audio dogfood follow-up.

Three findings:

* **R11-B-F2** — ``{"format":"mp3"}`` (the legacy OpenAI key) was
  silently dropped on ``/v1/audio/speech``; ``response_format``
  fell back to ``"wav"`` and the route returned HTTP 200 with
  ``Content-Type: audio/wav`` and RIFF/WAVE bytes. SDK clients
  copying older sample code (early ``openai-python`` < 1.0,
  Anthropic tutorials) had no way to tell their request was
  silently being downgraded. Fix: a ``model_validator(mode="before")``
  on :class:`vllm_mlx.api.models.AudioSpeechRequest` folds
  ``format`` into ``response_format`` when the latter isn't
  explicitly set. Explicit ``response_format`` always wins on
  conflict so a caller using both spellings (which itself is a
  client bug) never gets a silent override of intent.

* **R11-B-F3** — ``{"voice":"default"}`` (the obvious naive caller
  value) on kokoro / chatterbox / voxcpm was rejected by the
  voice-allowlist as ``invalid_voice``, even though the registry
  already advertises a ``default_voice`` for each entry — and the
  omitted-voice path (Pydantic default ``"af_heart"``) worked.
  The asymmetry was a UX trap. Fix: a
  :func:`vllm_mlx.routes.audio._resolve_default_voice_literal`
  pre-step maps ``voice="default"`` → ``entry.default_voice`` when
  the resolved model is registered.

* **R11-B-F4** — ``/v1/models`` for an audio-only alias advertised
  ``capabilities=["text"]`` and ``modality=null``. Drop-in OpenAI
  clients couldn't distinguish audio aliases from text models on
  the wire. Fix:
  :func:`vllm_mlx.routes.models._resolve_audio_entry` short-circuits
  audio aliases to ``capabilities=["audio.speech"]`` (TTS) or
  ``["audio.transcription"]`` (STT) and ``modality="audio"``.

Bo r11 evidence: /tmp/dogfood-0812/bo-r1.md F2 / F3 / F4.
"""

from __future__ import annotations

import sys
import types

import pytest

# ``vllm_mlx.routes.audio`` transitively imports ``mlx.core`` via the
# engine wiring. Linux CI runners don't install mlx, so a bare import
# raises ``ModuleNotFoundError`` and the rest of the file looks like a
# 13-test failure. Mirror the r8-A skip block so the file is a clean
# SKIP off-platform but still executes on Apple Silicon dev / CI.
pytest.importorskip(
    "mlx.core",
    reason="audio route imports transitively pull in mlx; "
    "test runs on Apple Silicon / dev, not Linux CI runners",
)
pytest.importorskip(
    "mlx_lm",
    reason="audio route imports transitively pull in mlx_lm; "
    "test runs on Apple Silicon / dev, not Linux CI runners",
)
pytest.importorskip(
    "multipart",
    reason="TestClient requires python-multipart on the form lanes",
)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — copied from test_audio_r8_a_bundle.py so this file stands alone
# with the same harness pattern.
# ---------------------------------------------------------------------------


def _install_fake_mlx_audio(monkeypatch):
    """Make ``importlib.util.find_spec("mlx_audio")`` succeed without
    the real package present (the route's TTS-lane probe gates on this)."""
    import importlib.machinery

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_mlx_audio.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio", loader=None, is_package=True
    )
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_tts.__path__ = []
    fake_tts.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts", loader=None, is_package=True
    )
    fake_tts_generate = types.ModuleType("mlx_audio.tts.generate")
    fake_tts_generate.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts.generate", loader=None
    )
    fake_tts_generate.load_model = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", fake_tts_generate)


def _mount_audio_app() -> tuple[TestClient, callable]:
    """Mount the audio router with the rapid-mlx exception handlers so the
    Pydantic validation errors surface as the OpenAI envelope (not the
    default FastAPI 422)."""
    from vllm_mlx.config import get_config
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import audio as audio_route

    app = FastAPI()
    app.include_router(audio_route.router)
    install_exception_handlers(app)
    cfg = get_config()
    saved = cfg.api_key
    cfg.api_key = None

    def _restore():
        cfg.api_key = saved

    return TestClient(app), _restore


def _stub_engine(monkeypatch, *, voice_observed=None, format_observed=None):
    """Install a no-op TTSEngine that records the voice & format passed
    to it and uses the REAL ``to_bytes`` so encoder smoke-checks still run.
    Mirrors :func:`tests.test_audio_r8_a_bundle._stub_engine` but exposes
    the format observation hook so F2 can prove the alias actually
    reached the encoder.
    """
    import numpy as np

    from vllm_mlx.audio import probe as probe_mod
    from vllm_mlx.audio import tts as tts_mod
    from vllm_mlx.routes import audio as audio_route

    observed_models: list[str] = []
    real_to_bytes = tts_mod.TTSEngine.to_bytes

    class _RecordingEngine:
        def __init__(self, model_name: str):
            observed_models.append(model_name)
            self.model_name = model_name

        def load(self):
            pass

        def generate(self, text, voice="af_heart", speed=1.0):
            if voice_observed is not None:
                voice_observed.append(voice)
            audio = (np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 0.3).astype(
                np.float32
            )
            return tts_mod.AudioOutput(audio=audio, sample_rate=24000, duration=1.0)

        def to_bytes(self, audio, format="wav"):
            if format_observed is not None:
                format_observed.append(format)
            return real_to_bytes(self, audio, format=format)

    monkeypatch.setattr(tts_mod, "TTSEngine", _RecordingEngine)
    monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda: None)
    _install_fake_mlx_audio(monkeypatch)
    audio_route._tts_engine = None
    return audio_route, observed_models


# ===========================================================================
# R11-B-F2 — ``format`` alias maps to ``response_format``
# ===========================================================================


class TestFormatAliasLegacyToResponseFormat:
    """The legacy ``format`` key (early ``openai-python`` < 1.0,
    Anthropic sample code) must fold into ``response_format`` so SDK
    clients get the codec they asked for. Pre-fix the key was silently
    dropped and ``response_format`` defaulted to ``"wav"`` → callers
    got HTTP 200 with RIFF/WAVE bytes mislabeled as the requested
    codec."""

    def test_format_alias_legacy_to_response_format(self, monkeypatch):
        """Bo's reproducer: ``{"format":"mp3"}`` on chatterbox must
        produce ``audio/mpeg`` bytes (not silently downgrade to WAV).
        Pre-fix this returned 200 with ``Content-Type: audio/wav``
        and a RIFF/WAVE body.
        """
        pytest.importorskip("soundfile")

        format_observed: list[str] = []
        audio_route, _ = _stub_engine(monkeypatch, format_observed=format_observed)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": "Hi",
                    "voice": "default",
                    # Legacy spelling — the OpenAI spec migrated to
                    # ``response_format`` but several SDK paths still
                    # emit the old key.
                    "format": "mp3",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        # MP3 encoder may not be present on every libsndfile build;
        # accept either:
        #   * 200 + ``audio/mpeg`` Content-Type + MP3 frame sync, OR
        #   * 400 ``invalid_response_format`` envelope.
        # Pre-fix the route returned 200 ``audio/wav`` — that exact
        # silent-downgrade shape must NEVER come back.
        ctype = r.headers.get("content-type", "").split(";")[0].strip()
        if r.status_code == 200:
            assert ctype == "audio/mpeg", (
                f"R11-B-F2 regression: ``format=mp3`` returned "
                f"Content-Type {ctype!r}, expected 'audio/mpeg'. "
                f"The legacy ``format`` key was silently dropped."
            )
            assert format_observed == ["mp3"], (
                f"R11-B-F2 regression: engine received format="
                f"{format_observed!r}, expected ['mp3']. The alias "
                f"did not reach the encoder."
            )
            body = r.content
            assert not body.startswith(b"RIFF"), (
                "R11-B-F2 regression: body starts with 'RIFF' — the "
                "WAV fallback is still active despite Content-Type "
                "claiming 'audio/mpeg'. The legacy ``format`` key was "
                "silently dropped and the route fell through to WAV."
            )
            # MP3 frame sync: 0xFFFB or 0xFFF3 (MPEG-1/2 Layer-3).
            assert body[:1] == b"\xff" and (body[1] & 0xE0) == 0xE0, (
                f"R11-B-F2 regression: body does not start with MP3 "
                f"frame sync, got {body[:4]!r}."
            )
        else:
            # Graceful 400 fallback when the encoder doesn't ship MP3.
            assert r.status_code == 400, (
                f"R11-B-F2 regression: ``format=mp3`` returned "
                f"unexpected status {r.status_code}. Body: {r.text[:500]}"
            )
            body = r.json()
            assert body["error"]["param"] == "response_format", body
            assert body["error"]["code"] == "invalid_response_format", body

    def test_explicit_response_format_wins_over_format_alias(self, monkeypatch):
        """When both ``response_format`` AND ``format`` are sent
        (itself a client bug) the spec-correct field must win. Never
        a silent override of explicit caller intent — the legacy
        alias is a back-compat hook, not a silent overwrite."""
        pytest.importorskip("soundfile")

        format_observed: list[str] = []
        audio_route, _ = _stub_engine(monkeypatch, format_observed=format_observed)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "af_heart",
                    "response_format": "wav",
                    # Conflicting legacy key — should be ignored.
                    "format": "mp3",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        ctype = r.headers.get("content-type", "").split(";")[0].strip()
        assert ctype == "audio/wav", (
            f"R11-B-F2 regression: explicit response_format=wav was "
            f"silently overridden by format=mp3 (Content-Type={ctype!r}). "
            f"Explicit caller intent must always win."
        )
        assert format_observed == ["wav"], (
            f"R11-B-F2 regression: engine received format="
            f"{format_observed!r}, expected ['wav']. The legacy alias "
            f"silently overrode the explicit field."
        )

    @pytest.mark.parametrize(
        "bad_legacy",
        [
            123,  # int
            True,  # bool
            [],  # list
            {},  # dict
        ],
    )
    def test_nonstring_format_alias_400s(self, monkeypatch, bad_legacy):
        """Codex r1 #1 BLOCKING (review-20260315-103736-152fd1.md): a
        non-string legacy ``format`` value (``{"format":123}``) MUST 400
        on ``response_format``, NOT silently fall through to the
        ``"wav"`` default. The before-validator now folds EVERY legacy
        value into ``response_format`` so the field-level type-check
        produces the same 400 envelope a wrong-typed
        ``response_format`` would emit. Pre-codex this passed through
        verbatim and ``response_format`` defaulted to ``"wav"`` — the
        exact silent-downgrade shape F-2 exists to prevent."""
        _install_fake_mlx_audio(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "af_heart",
                    "format": bad_legacy,
                },
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R11-B-F2 codex r1 regression: format={bad_legacy!r} "
            f"returned {r.status_code}, expected 400. Body: {r.text[:500]}"
        )
        body = r.json()
        # The envelope must surface ``response_format`` as the failing
        # field (NOT ``format``) so the caller learns the spec-correct
        # field name even when they used the legacy alias.
        assert body["error"]["param"] == "response_format", body
        assert body["error"]["type"] == "invalid_request_error", body

    def test_unknown_format_alias_still_400s(self, monkeypatch):
        """The alias only changes the SOURCE of the value, not the
        allowed-set contract. A legacy ``{"format":"jpeg"}`` (not in
        the supported codec list) MUST 400 with the same envelope an
        explicit ``response_format`` would emit."""
        _install_fake_mlx_audio(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "af_heart",
                    "format": "jpeg",
                },
            )
        finally:
            restore()

        assert r.status_code == 400, r.text
        body = r.json()
        assert body["error"]["type"] == "invalid_request_error", body
        # The validator surfaces the canonical field name in ``param``
        # so callers can fix their request payload without reading the
        # alias docs.
        assert body["error"]["param"] == "response_format", body


# ===========================================================================
# R11-B-F3 — ``voice="default"`` falls back to the registry default
# ===========================================================================


# (alias, expected_voice_substring)
# Each TTS family that ships a default_voice in aliases.json:
#   * kokoro → "af_heart" (the Pydantic default; the registry agrees)
#   * chatterbox → "default" (the engine's catch-all; mlx_audio
#     resolves it internally, no safetensors lookup)
#   * voxcpm → "default" (same shape as chatterbox)
_DEFAULT_VOICE_TARGETS = [
    ("kokoro", "af_heart"),
    ("chatterbox", "default"),
    ("voxcpm", "default"),
]


class TestVoiceDefaultFallsBackToRegistry:
    """The literal ``voice="default"`` (the obvious naive caller value)
    must resolve to the registry's ``default_voice`` for the resolved
    model. Pre-fix this was rejected by the kokoro allowlist as
    ``invalid_voice`` even though the registry already advertises
    ``default_voice="af_heart"`` for it."""

    @pytest.mark.parametrize("alias,expected_voice", _DEFAULT_VOICE_TARGETS)
    def test_voice_default_falls_back_to_registry(
        self, monkeypatch, alias, expected_voice
    ):
        voice_observed: list[str] = []
        audio_route, _ = _stub_engine(monkeypatch, voice_observed=voice_observed)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": alias,
                    "input": "Hi",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, (
            f"R11-B-F3 regression: ``voice='default'`` on {alias!r} "
            f"returned {r.status_code}, expected 200. Pre-fix this 400'd "
            f"on the kokoro allowlist even though the registry already "
            f"advertises ``default_voice``. Body: {r.text[:500]}"
        )
        assert voice_observed == [expected_voice], (
            f"R11-B-F3 regression: engine received voice="
            f"{voice_observed!r}, expected [{expected_voice!r}]. "
            f"``default`` was not substituted with the registry value."
        )

    def test_voice_omitted_still_uses_pydantic_default(self, monkeypatch):
        """Bo explicitly confirmed the omitted-voice path works today.
        Make sure the F-3 fix doesn't accidentally regress it: when
        ``voice`` is absent from the request, Pydantic populates the
        field default (``af_heart``) — the literal-resolution hook MUST
        NOT fire because the value isn't ``"default"``."""
        voice_observed: list[str] = []
        audio_route, _ = _stub_engine(monkeypatch, voice_observed=voice_observed)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voice_observed == ["af_heart"], (
            "Regression: omitted voice no longer resolves to the "
            f"Pydantic default 'af_heart'; saw {voice_observed!r}. "
            "The F-3 hook must not affect the omitted-voice path."
        )

    def test_voice_default_with_hf_id_resolves(self, monkeypatch):
        """The HF-id path (``model='mlx-community/Kokoro-82M-bf16'``)
        and the short-alias path (``model='kokoro'``) MUST resolve
        ``voice='default'`` to the same registry value. The reverse-
        HF-id lookup in :func:`resolve_audio_alias` powers this."""
        voice_observed: list[str] = []
        audio_route, _ = _stub_engine(monkeypatch, voice_observed=voice_observed)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "mlx-community/Kokoro-82M-bf16",
                    "input": "Hi",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voice_observed == ["af_heart"], (
            f"R11-B-F3 regression: HF-id path saw voice="
            f"{voice_observed!r}, expected ['af_heart']. The reverse-HF-id "
            f"lookup in resolve_audio_alias did not fire."
        )

    def test_invalid_voice_still_400s(self, monkeypatch):
        """The fallback only fires for the literal ``"default"`` — every
        other invalid voice (typo, drop-in OpenAI name) must still 400.
        Pin so the F-3 hook doesn't silently relax the allowlist
        contract."""
        _install_fake_mlx_audio(monkeypatch)
        _stub_engine(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hi",
                    "voice": "alloy",  # OpenAI default voice; not Kokoro
                    "response_format": "wav",
                },
            )
        finally:
            restore()

        assert r.status_code == 400, r.text
        body = r.json()
        assert body["error"]["code"] == "invalid_voice", body


# ===========================================================================
# R11-B-F4 — audio aliases advertise audio capability + modality
# ===========================================================================


def _mount_models_app(monkeypatch, **cfg_overrides):
    """Mount the models router with controlled config state. Mirrors
    :func:`tests.test_capabilities_field._mount_models_app`."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import models as models_route

    app = FastAPI()
    app.include_router(models_route.router)

    cfg = get_config()
    saved = {
        k: getattr(cfg, k, None)
        for k in (
            "model_name",
            "model_alias",
            "model_registry",
            "embedding_model_locked",
            "tool_call_parser",
            "api_key",
        )
    }
    cfg.model_registry = None
    cfg.api_key = None
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)

    import vllm_mlx.server as srv

    saved_srv = {
        "_embedding_model_locked": srv._embedding_model_locked,
        "_tool_call_parser": srv._tool_call_parser,
    }
    srv._embedding_model_locked = cfg_overrides.get("embedding_model_locked")
    srv._tool_call_parser = cfg_overrides.get("tool_call_parser")

    def _restore():
        for k, v in saved.items():
            setattr(cfg, k, v)
        for k, v in saved_srv.items():
            setattr(srv, k, v)

    return TestClient(app), _restore


# Each row: (alias, hf_id, expected_capability). Covers every audio
# family that ships in aliases.json — adding a new family should add a
# row here too so the wire-level capability advertisement stays uniform.
_AUDIO_ALIASES_FOR_CAP_CHECK = [
    # TTS aliases → ``audio.speech``.
    ("kokoro", "mlx-community/Kokoro-82M-bf16", "audio.speech"),
    ("chatterbox", "mlx-community/chatterbox-turbo-fp16", "audio.speech"),
    ("vibevoice", "mlx-community/VibeVoice-Realtime-0.5B-4bit", "audio.speech"),
    ("voxcpm", "mlx-community/VoxCPM1.5", "audio.speech"),
    ("dia", "mlx-community/Dia-1.6B-4bit", "audio.speech"),
    # STT aliases → ``audio.transcription``.
    ("whisper", "mlx-community/whisper-large-v3-mlx", "audio.transcription"),
    ("whisper-large-v3", "mlx-community/whisper-large-v3-mlx", "audio.transcription"),
    ("parakeet", "mlx-community/parakeet-tdt-0.6b-v2", "audio.transcription"),
]


class TestAudioAliasesHaveAudioCapabilities:
    """``/v1/models`` for an audio-only alias must advertise the audio
    capability + ``modality="audio"`` so drop-in OpenAI clients can
    route on the wire. Pre-fix every audio alias came back as
    ``capabilities=["text"]`` / ``modality=null``."""

    @pytest.mark.parametrize("alias,hf_id,expected_cap", _AUDIO_ALIASES_FOR_CAP_CHECK)
    def test_audio_aliases_have_audio_capabilities(
        self, monkeypatch, alias, hf_id, expected_cap
    ):
        """Both forms (alias + HF id) get the same audio shape on the
        wire."""
        client, restore = _mount_models_app(
            monkeypatch,
            model_name=hf_id,
            model_alias=alias,
        )
        try:
            r = client.get("/v1/models")
        finally:
            restore()

        assert r.status_code == 200, r.text
        body = r.json()
        ids_in_listing = {entry["id"] for entry in body["data"]}
        assert hf_id in ids_in_listing, (
            f"R11-B-F4 regression: HF id {hf_id!r} missing from "
            f"/v1/models listing. Body: {body}"
        )
        assert alias in ids_in_listing, (
            f"R11-B-F4 regression: short alias {alias!r} missing "
            f"from /v1/models listing. Body: {body}"
        )

        for entry in body["data"]:
            if entry["id"] not in (alias, hf_id):
                continue
            assert entry["modality"] == "audio", (
                f"R11-B-F4 regression: entry {entry['id']!r} reports "
                f"modality={entry['modality']!r}, expected 'audio'. "
                f"Pre-fix this was 'null' and drop-in OpenAI clients "
                f"couldn't tell audio aliases from text models."
            )
            assert expected_cap in entry["capabilities"], (
                f"R11-B-F4 regression: entry {entry['id']!r} missing "
                f"{expected_cap!r} in capabilities={entry['capabilities']!r}. "
                f"Pre-fix this was ['text'] and clients couldn't route "
                f"audio traffic correctly."
            )
            # ``text`` MUST NOT bleed into the audio entry — the
            # pre-fix tag was misleading and the new contract is a
            # clean audio-only capability set.
            assert "text" not in entry["capabilities"], (
                f"R11-B-F4 regression: audio entry {entry['id']!r} leaked "
                f"'text' tag: {entry['capabilities']!r}. Audio aliases "
                f"are NOT text models; the capability list must be "
                f"audio-only."
            )

    def test_retrieve_model_for_audio_alias_has_audio_capability(self, monkeypatch):
        """``GET /v1/models/{id}`` must agree with the listing — same
        shape for the same id, otherwise clients hitting the
        single-id endpoint to bootstrap state see a stale text
        envelope."""
        client, restore = _mount_models_app(
            monkeypatch,
            model_name="mlx-community/Kokoro-82M-bf16",
            model_alias="kokoro",
        )
        try:
            r = client.get("/v1/models/kokoro")
        finally:
            restore()

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["modality"] == "audio", body
        assert body["capabilities"] == ["audio.speech"], body
