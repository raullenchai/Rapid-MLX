# SPDX-License-Identifier: Apache-2.0
"""R7-C regression bundle — Bo's 0.8.8 dogfood follow-up to r6-C.

Three findings:

* **R7-H3** — ``/v1/audio/speech`` with Kokoro returned 500 ``No audio
  generated``. Root cause (verified by direct in-process probe) is the
  ``mlx_audio==0.4.4`` ``istftnet.SineGen`` regression: an off-by-one
  in the upsample interpolation produces ``noise_amp`` shape
  ``(1, 36600, 1)`` while ``sine_waves`` is ``(1, 36900, 9)`` —
  ``noise = noise_amp * mx.random.normal(sine_waves.shape)`` then
  raises ``[broadcast_shapes] ... cannot be broadcast``. ``mlx-audio
  ==0.4.3`` does NOT have the regression. Fix pins the dep to
  ``<0.4.4``. The catch-all also now logs the FULL traceback at
  ``exception`` level so future incidents are diagnosable from the
  operator log (the pre-fix log had only the leaf message).

* **R7-M8** — ``/v1/audio/speech`` with ``input=""`` (or no ``input``
  key, or whitespace-only) 500'd with ``No audio generated``. The
  route declared ``input: str = ""`` as a bare query parameter, so
  JSON bodies were silently dropped AND there was nowhere to attach a
  validation constraint. Fix binds a Pydantic
  :class:`vllm_mlx.api.models.AudioSpeechRequest` body model with
  ``input: str = Field(..., min_length=1)`` and a non-blank validator.
  The envelope handler (registered for ``/v1/audio/speech``) emits a
  400 ``invalid_request_error`` with ``param="input"``.

* **R7-M9** — ``/v1/audio/transcriptions`` with ``model="whisper"``
  (no size suffix) 404'd with ``model_not_found_error``. The chat lane
  resolves short aliases at request-time; STT didn't. Fix adds
  ``whisper`` (and ``whisper-1``) to ``STT_MODEL_ALIASES`` pointing at
  the largest supported variant. ``/v1/audio/transcriptions`` AND
  ``/v1/audio/translations`` both pick this up via the shared
  ``_run_stt_request`` helper.
"""

from __future__ import annotations

import io
import struct
import sys
import types
import wave

import pytest

# ``vllm_mlx.routes.audio`` transitively imports ``mlx.core`` via the
# engine wiring. Linux CI runners (``pr_validate``'s validate job) don't
# install mlx, so a bare import raises ``ModuleNotFoundError`` and 13
# unrelated tests look like regressions. ``importorskip`` short-circuits
# the file with a clean SKIP so it stays out of the negative-control set
# AND still executes anywhere with the right deps (Apple Silicon CI +
# any dev machine with ``[audio]`` installed). Mirrors the
# ``test_audio_upload_size_limit.py`` skip block — same trap.
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
    reason="TestClient(files=...) requires python-multipart; "
    "skip on minimal-deps runners (CI pr-validate)",
)

from fastapi import FastAPI  # noqa: E402 — keep skips at top
from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (copied from test_audio_routes_bundle.py so this file stands alone)
# ---------------------------------------------------------------------------


def _make_tone_wav(duration_s: float = 0.25, freq_hz: float = 440.0) -> bytes:
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        import math

        for i in range(n_samples):
            sample = int(8000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            w.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


def _install_fake_mlx_audio(monkeypatch):
    """Install a fake ``mlx_audio`` package so the shallow probe's
    ``find_spec`` succeeds without the real package present."""
    import importlib.machinery

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_mlx_audio.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio", loader=None, is_package=True
    )
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_stt.__path__ = []
    fake_stt.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.stt", loader=None, is_package=True
    )
    fake_stt_utils = types.ModuleType("mlx_audio.stt.utils")
    fake_stt_utils.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.stt.utils", loader=None
    )
    fake_stt_utils.load_model = lambda *_a, **_k: None
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
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_stt_utils)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", fake_tts_generate)


def _mount_audio_app() -> tuple[TestClient, callable]:
    """Mount the audio router on a bare FastAPI app with the rapid-mlx
    exception handlers wired in. The handlers are what turn the
    Pydantic validation error into the OpenAI envelope shape — without
    them the test would see the default FastAPI 422.
    """
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


# ---------------------------------------------------------------------------
# R7-M9 — STT short alias resolution
# ---------------------------------------------------------------------------


class TestSTTShortWhisperAlias:
    """``model="whisper"`` resolves through the alias map instead of
    404'ing — R-04 contract parity with the chat lane.
    """

    def test_whisper_short_alias_resolves(self):
        """The alias map MUST contain ``whisper`` → the largest Whisper
        repo so a bare ``model="whisper"`` request from drop-in OpenAI
        SDK code lands on the supported variant.
        """
        from vllm_mlx.routes.audio import STT_MODEL_ALIASES, _resolve_stt_model

        # The mapping must exist…
        assert "whisper" in STT_MODEL_ALIASES, (
            "R7-M9 regression: 'whisper' is not in STT_MODEL_ALIASES. "
            "Bo's r2-T6 finding required this short alias to resolve "
            "to the largest supported variant for drop-in OpenAI SDK "
            "compatibility."
        )
        # …and resolve via the shared helper (the same helper STT and
        # translations call).
        resolved = _resolve_stt_model("whisper")
        assert "whisper-large-v3" in resolved.lower(), (
            f"Expected `whisper` to resolve to the largest variant, got {resolved!r}"
        )

    def test_whisper_1_legacy_alias_resolves(self):
        """OpenAI's legacy ``whisper-1`` placeholder maps to the same
        Whisper variant so legacy SDKs don't 404."""
        from vllm_mlx.routes.audio import STT_MODEL_ALIASES, _resolve_stt_model

        assert "whisper-1" in STT_MODEL_ALIASES, (
            "Legacy ``whisper-1`` placeholder must be accepted; some "
            "OpenAI SDKs still emit it as the default."
        )
        assert "whisper-large-v3" in _resolve_stt_model("whisper-1").lower()

    def test_whisper_short_alias_through_transcriptions_route(self, monkeypatch):
        """End-to-end through ``/v1/audio/transcriptions``: a
        ``model="whisper"`` form field must NOT trip the 404
        ``model_not_found_error`` envelope. Pre-fix Bo saw 404; post-
        fix the route reaches the engine (we stub the engine so we
        observe the resolved model name).
        """
        from vllm_mlx.audio import stt as stt_mod
        from vllm_mlx.routes import audio as audio_route

        observed: list[str] = []

        class _RecordingEngine:
            def __init__(self, model_name: str):
                observed.append(model_name)
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, *_a, **_k):
                return types.SimpleNamespace(
                    text="hi", language="en", duration=0.1, segments=None
                )

        monkeypatch.setattr(stt_mod, "STTEngine", _RecordingEngine)
        _install_fake_mlx_audio(monkeypatch)

        audio_route._stt_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
            audio_route._stt_engine = None

        assert r.status_code == 200, (
            f"R7-M9 regression: model='whisper' returned {r.status_code} "
            f"(expected 200 after alias map resolves the short id). "
            f"Body: {r.text}"
        )
        # The engine MUST have been instantiated with the resolved HF
        # path, not the literal short alias — confirms the alias map
        # is wired into ``_resolve_stt_model``.
        assert observed, "STT engine was never instantiated"
        assert "whisper-large-v3" in observed[0].lower(), (
            f"Expected the alias map to resolve 'whisper' to the "
            f"largest variant, but the engine saw {observed[0]!r}"
        )

    def test_whisper_short_alias_through_translations_route(self, monkeypatch):
        """The same alias map is shared with ``/v1/audio/translations``
        via ``_run_stt_request`` so both routes resolve short aliases
        identically. Codex r6 NIT requires the model to remain a
        Whisper engine; ``whisper`` → whisper-large-v3 satisfies that.
        """
        from vllm_mlx.audio import stt as stt_mod
        from vllm_mlx.routes import audio as audio_route

        observed: list[str] = []

        class _RecordingEngine:
            def __init__(self, model_name: str):
                observed.append(model_name)
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, *_a, **_k):
                return types.SimpleNamespace(
                    text="hi", language="en", duration=0.1, segments=None
                )

        monkeypatch.setattr(stt_mod, "STTEngine", _RecordingEngine)
        _install_fake_mlx_audio(monkeypatch)

        audio_route._stt_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/translations",
                data={"model": "whisper"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
            audio_route._stt_engine = None

        assert r.status_code == 200, (
            f"R7-M9 regression on translations: model='whisper' "
            f"returned {r.status_code}, expected 200. Body: {r.text}"
        )
        assert observed and "whisper-large-v3" in observed[0].lower()


# ---------------------------------------------------------------------------
# R7-M8 — empty / blank ``input`` → 400 ``invalid_request_error``
# ---------------------------------------------------------------------------


class TestSpeechInputValidation:
    """Empty / missing / whitespace ``input`` raises the OpenAI 400
    envelope BEFORE the synthesis engine runs. Pre-fix every shape
    collapsed into the engine's ``No audio generated`` 500.
    """

    def test_empty_input_returns_400_envelope(self, monkeypatch):
        _install_fake_mlx_audio(monkeypatch)

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "", "voice": "af_heart"},
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R7-M8 regression: empty input returned {r.status_code} "
            f"(expected 400). Body: {r.text}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "input", err
        # The message must point at the ``input`` field so a caller
        # can fix the request without server-side investigation.
        assert "input" in err["message"].lower(), err

    def test_missing_input_key_returns_400_envelope(self, monkeypatch):
        """Pre-fix the route declared ``input: str = ""`` as a query
        parameter, so a JSON body without ``input`` silently fell back
        to the empty-string default. The bound Pydantic model now makes
        ``input`` required — a missing key MUST 400.
        """
        _install_fake_mlx_audio(monkeypatch)

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "voice": "af_heart"},
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R7-M8 regression: missing input key returned "
            f"{r.status_code} (expected 400). Body: {r.text}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "input", err

    def test_whitespace_only_input_returns_400_envelope(self, monkeypatch):
        """``min_length=1`` doesn't catch whitespace-only input because
        ``"   "`` is three characters. The model's custom validator
        rejects blank strings so the wire contract is "non-blank text"
        and the empty-phoneme 500 cannot fire.
        """
        _install_fake_mlx_audio(monkeypatch)

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "   ", "voice": "af_heart"},
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R7-M8 regression: whitespace input returned "
            f"{r.status_code} (expected 400). Body: {r.text}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "input", err
        # The message must mention non-empty so callers understand the
        # rejection rule (vs. a generic "validation error").
        msg = err["message"].lower()
        assert "non-empty" in msg or "non-blank" in msg or "blank" in msg, err


# ---------------------------------------------------------------------------
# R7-M8 — ``input`` reaches the synthesis engine when valid
# ---------------------------------------------------------------------------


class TestSpeechBodyHonored:
    """The Pydantic Body model means JSON requests actually reach the
    handler — pre-fix the bare query params silently dropped the JSON
    body so a non-empty ``input`` was still ``""`` at the engine."""

    def test_json_body_input_reaches_engine(self, monkeypatch):
        from vllm_mlx.audio import tts as tts_mod
        from vllm_mlx.audio.probe import require_kokoro_runtime  # noqa: F401
        from vllm_mlx.routes import audio as audio_route

        observed: list[str] = []

        class _NoopEngine:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def load(self):
                pass

            def generate(self, text: str, voice: str = "af_heart", speed: float = 1.0):
                observed.append(text)
                import numpy as np

                return types.SimpleNamespace(
                    audio=np.zeros(2400, dtype=np.float32),
                    sample_rate=24000,
                    duration=0.1,
                )

            def to_bytes(self, audio, format: str = "wav") -> bytes:
                return b"\x00" * 1234

        monkeypatch.setattr(tts_mod, "TTSEngine", _NoopEngine)
        # Skip the misaki gate — we're not testing that path here.
        from vllm_mlx.audio import probe as probe_mod

        monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda: None)
        _install_fake_mlx_audio(monkeypatch)

        audio_route._tts_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello world",
                    "voice": "af_heart",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert observed == ["hello world"], (
            f"R7-M8 regression: the JSON body ``input`` field did not "
            f"reach the engine. Observed text: {observed}. Pre-fix the "
            "route's bare query params silently dropped the body."
        )


# ---------------------------------------------------------------------------
# R7-H3 — catch-all logs full traceback + emits OpenAI envelope
# ---------------------------------------------------------------------------


class TestSpeechCatchAllShape:
    """When the synthesis engine raises a backend error (mlx_audio
    regression, weight load failure, etc.) the catch-all must:

    1. Log the FULL traceback via ``logger.exception`` so operators
       can root-cause from the log alone.
    2. Surface an OpenAI 500 envelope with ``type="api_error"`` /
       ``code="tts_generation_failed"`` so SDK error branches can
       pattern-match instead of seeing a bare-string ``detail``.
    """

    def test_engine_failure_emits_openai_envelope(self, monkeypatch, caplog):
        import logging

        from vllm_mlx.audio import tts as tts_mod
        from vllm_mlx.routes import audio as audio_route

        class _BoomEngine:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def load(self):
                pass

            def generate(self, *_a, **_k):
                # Simulate the R7-H3 root-cause shape verbatim — the
                # exact istftnet broadcast error mlx-audio 0.4.4 raised.
                raise ValueError(
                    "[broadcast_shapes] Shapes (1,36600,1) and "
                    "(1,36900,9) cannot be broadcast."
                )

            def to_bytes(self, *_a, **_k):
                return b""

        monkeypatch.setattr(tts_mod, "TTSEngine", _BoomEngine)
        from vllm_mlx.audio import probe as probe_mod

        monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda: None)
        _install_fake_mlx_audio(monkeypatch)

        audio_route._tts_engine = None

        client, restore = _mount_audio_app()
        with caplog.at_level(logging.ERROR, logger="rapid_mlx.routes.audio"):
            try:
                r = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "kokoro",
                        "input": "hello",
                        "voice": "af_heart",
                    },
                )
            finally:
                restore()
                audio_route._tts_engine = None

        assert r.status_code == 500, r.text
        body = r.json()
        err = body["error"]
        assert err["type"] == "api_error", err
        assert err["code"] == "tts_generation_failed", err

        # ``logger.exception`` writes the traceback into the
        # LogRecord.exc_info tuple — assert that's populated for at
        # least one record (it wasn't pre-fix; the legacy
        # ``logger.error(f"...: {e}")`` left exc_info=None and the
        # operator only saw the leaf message).
        had_traceback = any(
            "TTS generation failed" in rec.getMessage() and rec.exc_info is not None
            for rec in caplog.records
        )
        assert had_traceback, (
            "R7-H3 regression: the TTS catch-all must log the full "
            "traceback via ``logger.exception`` so operators can "
            "root-cause from the log. Found records: "
            f"{[(r.getMessage(), r.exc_info is not None) for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# R7-H3 — pyproject pins mlx-audio<0.4.4
# ---------------------------------------------------------------------------


class TestMlxAudioVersionPin:
    """The R7-H3 fix is upstream — mlx-audio 0.4.4 broke
    ``istftnet.SineGen``. Pin the dep below 0.4.4 in pyproject.toml so
    a fresh ``pip install rapid-mlx[audio]`` doesn't pull the broken
    release. The test parses pyproject.toml verbatim so a future
    contributor that loosens the bound trips CI.
    """

    def test_mlx_audio_upper_bound_pins_below_0_4_4(self):
        from pathlib import Path

        try:
            import tomllib  # 3.11+
        except ImportError:  # pragma: no cover — keep 3.10 fallback
            import tomli as tomllib  # type: ignore[import-not-found]

        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as f:
            cfg = tomllib.load(f)
        audio_deps = cfg["project"]["optional-dependencies"]["audio"]
        mlx_audio_specs = [d for d in audio_deps if d.startswith("mlx-audio")]
        assert len(mlx_audio_specs) == 1, (
            f"Expected exactly one mlx-audio pin, found {mlx_audio_specs}"
        )
        spec = mlx_audio_specs[0]
        # Both the floor AND the upper-bound matter. The floor is
        # historical; the upper-bound is the R7-H3 fix.
        assert "<0.4.4" in spec, (
            f"R7-H3 regression: mlx-audio must be pinned ``<0.4.4`` to "
            f"avoid the istftnet SineGen broadcast_shapes regression. "
            f"Current pin: {spec!r}"
        )


# ---------------------------------------------------------------------------
# R7-H3 — TTS alias map is module-level + shared helper
# ---------------------------------------------------------------------------


class TestTTSAliasResolver:
    """The TTS alias resolution rule lives in a module-level helper
    (``_resolve_tts_model``) so adding an engine lands in one place
    and tests can pin the contract without going through the handler.
    Mirrors the STT lane's ``_resolve_stt_model`` shape — R-04 contract
    parity across audio routes.
    """

    def test_tts_alias_map_includes_canonical_engines(self):
        from vllm_mlx.routes.audio import TTS_MODEL_ALIASES

        for alias in ("kokoro", "chatterbox", "vibevoice", "voxcpm"):
            assert alias in TTS_MODEL_ALIASES, (
                f"R7-H3 follow-up: TTS_MODEL_ALIASES is missing "
                f"canonical engine {alias!r}. The handler used to "
                "inline this map; promotion to module-level requires "
                "all pre-existing aliases stay."
            )

    def test_tts_default_alias_resolves(self):
        from vllm_mlx.routes.audio import _resolve_tts_model

        # ``None``, ``""``, and ``"default"`` all map to the default
        # alias's HF path — drop-in OpenAI SDK compatibility (R-03).
        for placeholder in (None, "", "default"):
            resolved = _resolve_tts_model(placeholder)
            assert "kokoro" in resolved.lower(), (
                f"_resolve_tts_model({placeholder!r}) must default to "
                f"the Kokoro alias, got {resolved!r}"
            )

    def test_tts_pass_through_for_full_hf_path(self):
        """A HuggingFace-shaped id passes through verbatim so callers
        can opt in to repos not in the alias map."""
        from vllm_mlx.routes.audio import _resolve_tts_model

        hf_path = "mlx-community/Kokoro-82M-bf16"
        assert _resolve_tts_model(hf_path) == hf_path
