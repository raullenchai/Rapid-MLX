# SPDX-License-Identifier: Apache-2.0
"""F-K-* regression tests — Karim's audio routes bundle (0.8.5 dogfood).

Four findings from Karim's r1/r2 probe runs:

* **F-K-WHISPER-500** — ``/v1/audio/transcriptions`` with
  ``model=whisper-large-v3`` 500'd because the mlx-community Whisper
  repos ship only ``weights.npz``/``config.json`` — no processor files
  — so ``mlx_audio.stt.models.whisper.Model.post_load_hook`` silently
  set ``_processor=None`` and the first transcribe raised
  ``ValueError: Processor not found``. The integration-layer fix loads
  the WhisperProcessor from the OpenAI counterpart repo.

* **F-K-KOKORO-MISAKI** — ``/v1/audio/speech`` with Kokoro 503'd
  because Kokoro's lazy ``misaki`` G2P import fails on installs without
  the ``[audio]`` extra. The fix gates the missing extra at the route
  boundary so the 503 fires BEFORE weight load.

* **F-K-TRANSLATIONS-MISSING** — ``/v1/audio/translations`` 404'd
  because the route was never registered. The fix mirrors the
  transcriptions route via a shared helper, forcing ``task="translate"``.

* **F-K-CAPABILITIES-OMIT-AUDIO** — the per-lane probe only checked
  that ``mlx_audio`` imports; a torn backend (Whisper without
  processor, Kokoro without misaki) still mounted the route then 5xx'd
  on first use. The fix adds an opt-in deep probe that runs a dry-run
  inference per lane and surfaces the verdict via ``/v1/models``.

Tests stub mlx_audio at the integration layer so they don't need real
weights and run fast in CI.
"""

from __future__ import annotations

import io
import struct
import sys
import types
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test fixtures: synthetic WAV + mlx_audio stub
# ---------------------------------------------------------------------------


def _make_tone_wav(duration_s: float = 0.25, freq_hz: float = 440.0) -> bytes:
    """Return a tiny mono 16-kHz WAV — small enough to keep test
    fixtures in-memory without disk I/O."""
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
    """Install a fake ``mlx_audio`` package + sub-modules in
    ``sys.modules`` so the shallow probe's ``find_spec("mlx_audio")``
    succeeds without crashing on a missing ``__spec__``. Used by tests
    that need the shallow lane probe to pass while exercising deeper
    behaviour (Kokoro-misaki gate, deep probe degraded state)."""
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
    fake_stt_utils.load_model = lambda *_args, **_kw: None
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_tts.__path__ = []
    fake_tts.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts", loader=None, is_package=True
    )
    fake_tts_generate = types.ModuleType("mlx_audio.tts.generate")
    fake_tts_generate.__spec__ = importlib.machinery.ModuleSpec(
        "mlx_audio.tts.generate", loader=None
    )
    fake_tts_generate.load_model = lambda *_args, **_kw: None
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_stt_utils)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.generate", fake_tts_generate)


@pytest.fixture
def _reset_audio_probe():
    """Clear cached probe verdicts + recorded lane statuses between tests."""
    from vllm_mlx.audio import probe

    probe._reset_probe_cache()
    yield
    probe._reset_probe_cache()


def _mount_audio_app() -> tuple[TestClient, callable]:
    """Mount the audio router on a bare FastAPI app, bypassing auth."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import audio as audio_route

    app = FastAPI()
    app.include_router(audio_route.router)
    cfg = get_config()
    saved = cfg.api_key
    cfg.api_key = None

    def _restore():
        cfg.api_key = saved

    return TestClient(app), _restore


def _mount_models_app() -> tuple[TestClient, callable]:
    """Mount the models router on a bare FastAPI app, bypassing auth."""
    from vllm_mlx.config import get_config
    from vllm_mlx.routes import models as models_route

    app = FastAPI()
    app.include_router(models_route.router)
    cfg = get_config()
    saved_api = cfg.api_key
    saved_name = cfg.model_name
    saved_alias = cfg.model_alias
    cfg.api_key = None
    cfg.model_name = "test-alias"
    cfg.model_alias = "test-alias"

    def _restore():
        cfg.api_key = saved_api
        cfg.model_name = saved_name
        cfg.model_alias = saved_alias

    return TestClient(app), _restore


class _FakeWhisperResult:
    text = "hello world"
    language = "en"
    segments = None


class _FakeWhisperModel:
    """Mimics the surface ``vllm_mlx.audio.stt.STTEngine`` touches.

    Pre-fix the real mlx_audio Whisper model had ``_processor=None`` at
    load time (because mlx-community repos lack processor files), so
    ``generate`` raised ``ValueError: Processor not found``. The fake
    mirrors that broken state until the integration layer attaches a
    real processor (the F-K-WHISPER-500 fix) — then ``generate``
    succeeds. This lets us validate the patch helper without needing
    real weights or a network round-trip to HuggingFace.
    """

    def __init__(self):
        self._processor = None  # The bug: processor wasn't attached.

    def generate(self, audio_path, **kwargs):
        if self._processor is None:
            raise ValueError(
                "Processor not found. Make sure the model was loaded "
                "with a HuggingFace processor."
            )
        return _FakeWhisperResult()


class _FakeParakeetResult:
    text = ""
    language = None
    segments = None


class _FakeParakeetModel:
    def generate(self, audio_path, **kwargs):
        return _FakeParakeetResult()


# ---------------------------------------------------------------------------
# F-K-WHISPER-500 — integration-layer processor patch
# ---------------------------------------------------------------------------


class TestWhisperProcessorPatch:
    """The STT engine attaches a WhisperProcessor from the OpenAI repo
    when mlx_audio's own post_load_hook left ``_processor=None``."""

    def test_processor_patched_when_post_load_hook_failed(
        self, monkeypatch, _reset_audio_probe
    ):
        """Simulate the F-K-WHISPER-500 shape: ``load_model`` returns
        a Whisper model with ``_processor=None``. After ``STTEngine.load``
        the processor should be attached (via the OpenAI counterpart
        repo) and ``generate`` should succeed.
        """
        from vllm_mlx.audio import stt as stt_mod

        fake_model = _FakeWhisperModel()
        sentinel_processor = object()

        def _fake_load_model(model_path, **kwargs):
            return fake_model

        # Stub the transformers WhisperProcessor.from_pretrained so the
        # test doesn't reach the network. We treat the returned
        # sentinel as opaque — only the *attachment* is under test.
        class _FakeProcessor:
            @staticmethod
            def from_pretrained(name):
                # Verify the integration layer requested the OpenAI repo,
                # not the broken mlx-community one. This is the
                # behavioral pin.
                assert name.startswith("openai/whisper"), (
                    f"Expected OpenAI counterpart, got {name!r}"
                )
                return sentinel_processor

        # Inject our fakes into the stt module's lazy-import sites.
        fake_mlx_audio_stt_utils = types.SimpleNamespace(load_model=_fake_load_model)
        monkeypatch.setitem(
            sys.modules, "mlx_audio.stt.utils", fake_mlx_audio_stt_utils
        )
        fake_transformers = types.SimpleNamespace(WhisperProcessor=_FakeProcessor)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        engine = stt_mod.STTEngine("mlx-community/whisper-large-v3-mlx")
        engine.load()

        assert engine._loaded is True
        assert fake_model._processor is sentinel_processor, (
            "F-K-WHISPER-500: STTEngine.load must attach a WhisperProcessor "
            "from the OpenAI repo when mlx_audio's post_load_hook left "
            "_processor=None. Pre-fix, _processor stayed None and "
            "transcribe() 500'd with `Processor not found`."
        )

        # Now transcribe should NOT raise.
        result = engine.transcribe("ignored-path.wav")
        assert result.text == "hello world"

    def test_processor_not_overwritten_when_already_present(
        self, monkeypatch, _reset_audio_probe
    ):
        """If mlx_audio's post_load_hook DID attach a processor (e.g.
        a future mlx-community upload ships processor files), the
        patch helper must be a no-op — overwriting would be incorrect."""
        from vllm_mlx.audio import stt as stt_mod

        existing_processor = object()
        fake_model = _FakeWhisperModel()
        fake_model._processor = existing_processor  # post_load_hook succeeded.

        def _fake_load_model(model_path, **kwargs):
            return fake_model

        class _FakeProcessor:
            calls = 0

            @staticmethod
            def from_pretrained(name):
                _FakeProcessor.calls += 1
                return object()

        fake_mlx_audio_stt_utils = types.SimpleNamespace(load_model=_fake_load_model)
        monkeypatch.setitem(
            sys.modules, "mlx_audio.stt.utils", fake_mlx_audio_stt_utils
        )
        fake_transformers = types.SimpleNamespace(WhisperProcessor=_FakeProcessor)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        engine = stt_mod.STTEngine("mlx-community/whisper-large-v3-mlx")
        engine.load()

        assert fake_model._processor is existing_processor, (
            "Patch helper must not overwrite a processor that mlx_audio "
            "already attached — that would mask a future fix upstream."
        )
        assert _FakeProcessor.calls == 0, (
            "Patch helper must not call WhisperProcessor.from_pretrained "
            "when a processor is already present."
        )

    def test_parakeet_skipped_by_patch(self, monkeypatch, _reset_audio_probe):
        """Parakeet engines don't use ``_processor`` — the patch helper
        must skip them (no spurious network fetch)."""
        from vllm_mlx.audio import stt as stt_mod

        fake_model = _FakeParakeetModel()

        def _fake_load_model(model_path, **kwargs):
            return fake_model

        class _FakeProcessor:
            calls = 0

            @staticmethod
            def from_pretrained(name):
                _FakeProcessor.calls += 1
                return object()

        fake_mlx_audio_stt_utils = types.SimpleNamespace(load_model=_fake_load_model)
        monkeypatch.setitem(
            sys.modules, "mlx_audio.stt.utils", fake_mlx_audio_stt_utils
        )
        fake_transformers = types.SimpleNamespace(WhisperProcessor=_FakeProcessor)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        engine = stt_mod.STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
        engine.load()

        assert _FakeProcessor.calls == 0, (
            "Parakeet path must not invoke the Whisper processor patch."
        )


# ---------------------------------------------------------------------------
# F-K-WHISPER-500 — transcriptions route degrades cleanly when patch fails
# ---------------------------------------------------------------------------


class TestTranscriptionsCleanEnvelopeOnProcessorFailure:
    """When the Whisper backend can't find a processor (e.g. the
    OpenAI fallback fetch failed too), the route must return a 503
    ``backend_unavailable`` envelope instead of a generic 500
    ``transcription_failed``.

    Pre-fix the user saw 500 with a vague body — same envelope as a
    genuinely-broken audio file — and couldn't tell whether to retry
    with a different model or re-encode the audio.
    """

    def test_processor_not_found_returns_clean_503(
        self, monkeypatch, _reset_audio_probe
    ):
        from vllm_mlx.routes import audio as audio_route

        # Force a Whisper model with _processor=None to slip past the
        # integration-layer patch (e.g. the OpenAI fetch failed).
        fake_model = _FakeWhisperModel()

        def _fake_load_model(model_path, **kwargs):
            return fake_model

        fake_mlx_audio_stt_utils = types.SimpleNamespace(load_model=_fake_load_model)
        monkeypatch.setitem(
            sys.modules, "mlx_audio.stt.utils", fake_mlx_audio_stt_utils
        )

        # Make the WhisperProcessor fetch fail so _processor stays None.
        class _BrokenProcessor:
            @staticmethod
            def from_pretrained(name):
                raise OSError("simulated network failure to openai/whisper-large-v3")

        fake_transformers = types.SimpleNamespace(WhisperProcessor=_BrokenProcessor)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        # Force-clear any module-level _stt_engine cached from a prior test.
        audio_route._stt_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
            audio_route._stt_engine = None

        assert r.status_code == 503, (
            f"Expected 503 backend_unavailable envelope when Whisper "
            f"processor is missing, got {r.status_code}: {r.text}"
        )
        body = r.json()
        assert "error" in body.get("detail", {}), body
        err = body["detail"]["error"]
        assert err["type"] == "backend_unavailable_error", err
        assert err["code"] == "backend_unavailable", err
        # Actionable hint mentions parakeet as fallback.
        assert "parakeet" in err["message"].lower(), err


# ---------------------------------------------------------------------------
# F-K-WHISPER-500 — transcriptions succeed end-to-end through the route
# ---------------------------------------------------------------------------


class TestTranscriptionsRouteEndToEnd:
    """Happy-path: a Whisper model with a patched processor produces
    a 200 ``{"text": ..., "language": ..., "duration": ...}`` from
    ``/v1/audio/transcriptions``."""

    def test_transcriptions_returns_200_with_text(
        self, monkeypatch, _reset_audio_probe
    ):
        from vllm_mlx.routes import audio as audio_route

        fake_model = _FakeWhisperModel()
        sentinel = object()

        def _fake_load_model(model_path, **kwargs):
            return fake_model

        class _FakeProcessor:
            @staticmethod
            def from_pretrained(name):
                return sentinel

        fake_mlx_audio_stt_utils = types.SimpleNamespace(load_model=_fake_load_model)
        monkeypatch.setitem(
            sys.modules, "mlx_audio.stt.utils", fake_mlx_audio_stt_utils
        )
        fake_transformers = types.SimpleNamespace(WhisperProcessor=_FakeProcessor)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        audio_route._stt_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper-large-v3"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
            audio_route._stt_engine = None

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"] == "hello world"
        assert body["language"] == "en"


# ---------------------------------------------------------------------------
# F-K-TRANSLATIONS-MISSING — route is registered + behaves like transcriptions
# ---------------------------------------------------------------------------


class TestTranslationsRoute:
    """/v1/audio/translations exists and mirrors transcriptions with
    ``task="translate"``. The wire-level OpenAI contract differs only
    in that ``language`` is not accepted (output is always English)."""

    def test_translations_route_registered(self):
        """The OpenAPI/router should advertise the route.

        Pre-fix this 404'd at the FastAPI routing layer because the
        decorator was never written.
        """
        from vllm_mlx.routes import audio as audio_route

        paths = {r.path for r in audio_route.router.routes}
        assert "/v1/audio/translations" in paths, (
            "F-K-TRANSLATIONS-MISSING regression: /v1/audio/translations "
            "is not registered. OpenAI spec requires it for Whisper "
            "translation-to-English."
        )

    def test_translations_returns_200_with_text(self, monkeypatch, _reset_audio_probe):
        from vllm_mlx.routes import audio as audio_route

        fake_model = _FakeWhisperModel()

        def _fake_load_model(model_path, **kwargs):
            return fake_model

        class _FakeProcessor:
            @staticmethod
            def from_pretrained(name):
                return object()

        # Capture the ``task`` kwarg the engine receives so we can pin
        # that the translations route forces ``translate``.
        observed_tasks: list[str] = []

        def _generate(audio_path, **kwargs):
            observed_tasks.append(kwargs.get("task"))
            return _FakeWhisperResult()

        fake_model.generate = _generate

        fake_mlx_audio_stt_utils = types.SimpleNamespace(load_model=_fake_load_model)
        monkeypatch.setitem(
            sys.modules, "mlx_audio.stt.utils", fake_mlx_audio_stt_utils
        )
        fake_transformers = types.SimpleNamespace(WhisperProcessor=_FakeProcessor)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        audio_route._stt_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/translations",
                data={"model": "whisper-large-v3"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
            audio_route._stt_engine = None

        assert r.status_code == 200, r.text
        body = r.json()
        assert "text" in body, body
        assert body["text"] == "hello world"

        # The engine MUST have received task="translate", not "transcribe".
        assert "translate" in observed_tasks, (
            "F-K-TRANSLATIONS-MISSING: /v1/audio/translations must pass "
            f"task='translate' to the STT engine. Saw: {observed_tasks}"
        )

    def test_translations_returns_404_for_unknown_model(
        self, monkeypatch, _reset_audio_probe
    ):
        """Model validation should fire on translations the same way it
        fires on transcriptions — unknown alias → clean 404."""
        from vllm_mlx.routes import audio as audio_route

        audio_route._stt_engine = None

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/translations",
                data={"model": "no-such-model"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
            audio_route._stt_engine = None

        assert r.status_code == 404, r.text
        body = r.json()
        err = body["detail"]["error"]
        assert err["type"] == "model_not_found_error", err


# ---------------------------------------------------------------------------
# F-K-KOKORO-MISAKI — clean 503 when misaki is missing
# ---------------------------------------------------------------------------


class TestKokoroMisakiGate:
    """When ``misaki`` is missing, /v1/audio/speech with a Kokoro model
    must return a clean 503 envelope BEFORE the weight load — the
    existing route's catch-all 503 fired only after the engine
    constructor raised ``ImportError``, which still pulled the 300 MB
    weights from HF on every failed attempt.
    """

    def test_kokoro_request_503s_cleanly_when_misaki_missing(
        self, monkeypatch, _reset_audio_probe
    ):
        # Pretend misaki isn't installed. ``find_spec("mlx_audio")``
        # may walk the spec of an already-loaded mlx_audio module —
        # if a previous test imported mlx_audio without preserving its
        # __spec__, find_spec raises. Filter our fake to only intercept
        # the misaki probe so the real mlx_audio path keeps working.
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def _fake_find_spec(name, *args, **kwargs):
            if name == "misaki":
                return None
            try:
                return real_find_spec(name, *args, **kwargs)
            except ValueError:
                # ``mlx_audio.__spec__ is not set`` — happens when a
                # test stub overwrote sys.modules['mlx_audio'] without
                # a spec. Return a sentinel spec so the shallow probe
                # treats mlx_audio as present.
                if name == "mlx_audio":
                    return object()
                raise

        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

        # Ensure mlx_audio appears available so the TTS-lane probe passes
        # — we want to exercise the Kokoro-specific gate, not the lane
        # gate. Stub the sub-module to satisfy the lazy import inside
        # the probe. ``__spec__`` is set so ``find_spec`` walks cleanly.
        _install_fake_mlx_audio(monkeypatch)

        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={"model": "kokoro", "input": "hello", "voice": "af_heart"},
            )
        finally:
            restore()

        assert r.status_code == 503, r.text
        body = r.json()
        # detail is either a string (legacy envelope shape) or a dict.
        detail = body.get("detail", "")
        if isinstance(detail, dict):
            detail = detail.get("error", {}).get("message", "")
        detail_lower = detail.lower()
        assert "misaki" in detail_lower, (
            f"503 envelope should mention misaki. Got: {detail}"
        )
        # Install hint is actionable.
        assert "rapid-mlx[audio]" in detail_lower or "pip install" in detail_lower, (
            f"503 envelope should include an install hint. Got: {detail}"
        )

    def test_non_kokoro_tts_does_not_require_misaki(
        self, monkeypatch, _reset_audio_probe
    ):
        """The Kokoro misaki gate must NOT trip for Chatterbox/etc. —
        ``misaki`` is Kokoro-specific. Pre-fix there was no per-family
        gate; making the dep check global would break Chatterbox-only
        installs."""
        import importlib.util

        from vllm_mlx.audio.probe import require_kokoro_runtime

        real_find_spec = importlib.util.find_spec

        def _fake_find_spec(name, *args, **kwargs):
            if name == "misaki":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

        # The Kokoro gate itself must raise.
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            require_kokoro_runtime()
        assert exc.value.status_code == 503

        # But the speech route's own Kokoro branch only fires when
        # ``"kokoro" in model_name.lower()`` — non-Kokoro paths skip
        # the misaki probe entirely. Source-pin this so a refactor
        # that drops the conditional gets caught.
        from pathlib import Path

        route_file = (
            Path(__file__).resolve().parents[1] / "vllm_mlx" / "routes" / "audio.py"
        )
        source = route_file.read_text()
        assert 'if "kokoro" in model_name.lower():' in source, (
            "Kokoro misaki gate must be conditional on the model family; "
            "global gating would 503 Chatterbox/VibeVoice/VoxCPM users."
        )


# ---------------------------------------------------------------------------
# F-K-CAPABILITIES-OMIT-AUDIO — deep probe surfaces degraded backends
# ---------------------------------------------------------------------------


class TestDeepProbeSurfacesDegradedLane:
    """The deep probe runs a dry-run inference per lane and records
    ``degraded`` when the dry-run raises. ``/v1/models`` surfaces the
    verdict via ``audio_lanes``.

    Pre-fix the per-lane probe only checked ``mlx_audio`` importability;
    a Whisper backend with a missing processor would pass the probe,
    mount the route, and 500 on every real request. The deep probe
    catches that shape at boot.
    """

    def test_dry_run_failure_marks_lane_degraded(self, monkeypatch, _reset_audio_probe):
        from vllm_mlx.audio import probe
        from vllm_mlx.audio import stt as stt_mod

        # Stub STTEngine to raise inside transcribe (mimics
        # F-K-WHISPER-500 — model loads, generate raises).
        class _BrokenEngine:
            def __init__(self, model_name):
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, *args, **kwargs):
                raise RuntimeError("simulated processor-missing failure")

        monkeypatch.setattr(stt_mod, "STTEngine", _BrokenEngine)

        # Pretend mlx_audio is available so the shallow probe passes.
        # Use real ``ModuleType`` instances so ``find_spec`` doesn't
        # raise on a missing ``__spec__``.
        _install_fake_mlx_audio(monkeypatch)

        status = probe.deep_probe_audio_lane("stt")
        assert status["status"] == "degraded", status
        assert "simulated processor-missing failure" in (status["reason"] or ""), status

    def test_dry_run_success_marks_lane_ok(self, monkeypatch, _reset_audio_probe):
        from vllm_mlx.audio import probe
        from vllm_mlx.audio import stt as stt_mod

        class _OKEngine:
            def __init__(self, model_name):
                self.model_name = model_name

            def load(self):
                pass

            def transcribe(self, *args, **kwargs):
                return types.SimpleNamespace(
                    text="", language=None, duration=None, segments=None
                )

        monkeypatch.setattr(stt_mod, "STTEngine", _OKEngine)

        _install_fake_mlx_audio(monkeypatch)

        status = probe.deep_probe_audio_lane("stt")
        assert status["status"] == "ok", status
        assert status["reason"] is None

    def test_models_endpoint_surfaces_lane_status(
        self, monkeypatch, _reset_audio_probe
    ):
        """/v1/models entries carry ``audio_lanes`` once the deep probe
        has recorded a verdict — pre-fix this was invisible."""
        from vllm_mlx.audio import probe

        probe._record_lane_status("stt", "degraded", "simulated")
        probe._record_lane_status("tts", "ok", None)

        client, restore = _mount_models_app()
        try:
            r = client.get("/v1/models")
        finally:
            restore()

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["data"], body
        entry = body["data"][0]
        assert "audio_lanes" in entry, entry
        assert entry["audio_lanes"] == {"stt": "degraded", "tts": "ok"}, entry

    def test_models_endpoint_omits_lane_status_when_probe_never_ran(
        self, _reset_audio_probe
    ):
        """When the deep probe is disabled (default), the field is
        ``None`` so the wire shape is unchanged for existing clients."""
        client, restore = _mount_models_app()
        try:
            r = client.get("/v1/models")
        finally:
            restore()

        assert r.status_code == 200, r.text
        body = r.json()
        entry = body["data"][0]
        assert entry["audio_lanes"] is None, entry


# ---------------------------------------------------------------------------
# Route registration: middleware guards the translations path too
# ---------------------------------------------------------------------------


class TestAudioBodyLimitCoversTranslations:
    """The 25 MB upload cap middleware must guard the new translations
    route — without it, an attacker can send 1 GB to translations and
    exhaust the worker."""

    def test_translations_in_guarded_paths(self):
        from vllm_mlx.routes import audio as audio_route

        guarded = audio_route.AudioBodyLimitMiddleware._GUARDED_PATHS
        assert "/v1/audio/translations" in guarded, (
            "F-K-TRANSLATIONS-MISSING regression: the AudioBodyLimitMiddleware "
            "must include /v1/audio/translations in _GUARDED_PATHS so the "
            "25 MB upload cap covers both transcription endpoints."
        )
