# SPDX-License-Identifier: Apache-2.0
"""R6-H2 (Eva 0.8.7 dogfood) — STT ``response_format`` must be honored.

Eva's r1 dogfood reproduced ``POST /v1/audio/transcriptions`` with
``response_format`` ∈ {json, text, srt, vtt, verbose_json}: every value
came back as a JSON envelope (Content-Type ``application/json``),
silently ignoring the field. The r6-C fix branches on the validated
value and produces the right Content-Type + body shape.

This test stubs the engine so the assertions run without weights.
"""

from __future__ import annotations

import io
import math
import struct
import sys
import types
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared fixtures — synthetic WAV + a fake STT engine that returns
# multi-segment transcript so SRT/VTT formatters have real cues to render.
# ---------------------------------------------------------------------------


def _make_tone_wav(duration_s: float = 0.25, freq_hz: float = 440.0) -> bytes:
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(8000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            w.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


class _FakeMultiSegmentResult:
    """Two-cue transcript so subtitle formatters have something real."""

    text = "hello world goodbye world"
    language = "en"
    duration = 4.5
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world"},
        {"start": 2.5, "end": 4.5, "text": "goodbye world"},
    ]


class _FakeEngine:
    """Mirrors the ``STTEngine`` surface the route depends on.

    Returns a ``_FakeMultiSegmentResult`` regardless of input so the
    formatter branches can be exercised deterministically.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self):
        pass

    def transcribe(self, audio_path, language=None, task="transcribe"):
        return _FakeMultiSegmentResult()


@pytest.fixture
def _stub_engine(monkeypatch):
    """Stub the STTEngine + mlx_audio probe so the route runs without weights."""
    # The probe path: pretend mlx_audio is installed.
    import importlib.machinery

    from vllm_mlx.audio import probe
    from vllm_mlx.routes import audio as audio_route

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
    fake_stt_utils.load_model = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_stt_utils)

    probe._reset_probe_cache()

    # Patch the STTEngine import inside the audio module so
    # ``_run_stt_request`` picks up our fake engine.
    fake_stt_module = types.SimpleNamespace(STTEngine=_FakeEngine)
    monkeypatch.setattr("vllm_mlx.audio.stt.STTEngine", _FakeEngine, raising=False)
    # The route lazily does ``from ..audio.stt import STTEngine`` so
    # patch the binding inside ``sys.modules`` too.
    audio_stt_mod = sys.modules.get("vllm_mlx.audio.stt")
    if audio_stt_mod is not None:
        monkeypatch.setattr(audio_stt_mod, "STTEngine", _FakeEngine)
    else:  # pragma: no cover — first-import fallback
        monkeypatch.setitem(sys.modules, "vllm_mlx.audio.stt", fake_stt_module)

    # Force-clear the route's module-level engine cache between tests.
    audio_route._stt_engine = None

    yield
    audio_route._stt_engine = None
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


def _post_transcription(client, response_format: str | None):
    """Issue a transcription POST with the given response_format."""
    data = {"model": "whisper-large-v3"}
    if response_format is not None:
        data["response_format"] = response_format
    return client.post(
        "/v1/audio/transcriptions",
        data=data,
        files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
    )


# ---------------------------------------------------------------------------
# Each format gets its own assertion: status 200, right Content-Type, body
# parseable.
# ---------------------------------------------------------------------------


class TestResponseFormatHonored:
    """The five OpenAI ``response_format`` values must each produce the
    right Content-Type + body. Pre-fix every value returned JSON."""

    def test_json_default(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, None)
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/json"), r.headers
        body = r.json()
        assert body["text"] == "hello world goodbye world"
        assert body["language"] == "en"

    def test_json_explicit(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, "json")
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["text"] == "hello world goodbye world"

    def test_text(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, "text")
        finally:
            restore()
        assert r.status_code == 200, r.text
        ctype = r.headers["content-type"]
        assert ctype.startswith("text/plain"), (
            f"response_format=text must yield text/plain, got {ctype!r}"
        )
        # PlainTextResponse returns the raw text body, not a JSON envelope.
        assert r.text == "hello world goodbye world", r.text

    def test_srt(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, "srt")
        finally:
            restore()
        assert r.status_code == 200, r.text
        ctype = r.headers["content-type"]
        assert ctype.startswith("text/srt") or ctype.startswith("text/plain"), (
            f"response_format=srt must yield a text/srt-shaped Content-Type, "
            f"got {ctype!r}"
        )
        body = r.text
        # SRT structure: index + timestamps in HH:MM:SS,mmm --> HH:MM:SS,mmm
        assert "1\n" in body, body
        assert "00:00:00,000 --> 00:00:02,000" in body, body
        assert "hello world" in body
        # Second cue must be present and use the SRT comma separator.
        assert "00:00:02,500 --> 00:00:04,500" in body, body
        assert "goodbye world" in body

    def test_vtt(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, "vtt")
        finally:
            restore()
        assert r.status_code == 200, r.text
        ctype = r.headers["content-type"]
        assert ctype.startswith("text/vtt") or ctype.startswith("text/plain"), (
            f"response_format=vtt must yield a text/vtt-shaped Content-Type, "
            f"got {ctype!r}"
        )
        body = r.text
        # WebVTT mandates the header line.
        assert body.startswith("WEBVTT"), body
        # VTT uses dot as the millisecond separator (vs SRT's comma).
        assert "00:00:00.000 --> 00:00:02.000" in body, body
        assert "hello world" in body
        # No comma timestamps — those are SRT-only.
        assert "00:00:00,000" not in body

    def test_verbose_json(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, "verbose_json")
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("application/json")
        body = r.json()
        # verbose_json adds segments + duration on top of the basic shape.
        assert body["text"] == "hello world goodbye world"
        assert body["language"] == "en"
        assert body["task"] == "transcribe"
        assert body["duration"] == 4.5
        assert isinstance(body["segments"], list)
        assert len(body["segments"]) == 2, body["segments"]
        first = body["segments"][0]
        assert first["start"] == 0.0
        assert first["end"] == 2.0
        assert first["text"] == "hello world"


class TestResponseFormatRejectsInvalid:
    """An unknown ``response_format`` value must 400 — not silently
    fall through to JSON."""

    def test_typo_returns_400_envelope(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = _post_transcription(client, "jsno")
        finally:
            restore()
        assert r.status_code == 400, (
            f"Expected 400 invalid_request_error for typo'd "
            f"response_format, got {r.status_code}: {r.text}"
        )
        body = r.json()
        err = body.get("detail", {}).get("error") or body.get("error")
        assert err is not None, body
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "response_format", err


class TestTranslationsResponseFormat:
    """The translations route mirrors transcriptions on the
    ``response_format`` contract — both share the helper. Spot-check
    a non-JSON value reaches the formatter (not the JSON fallthrough)."""

    def test_translations_text_format(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/translations",
                data={"model": "whisper-large-v3", "response_format": "text"},
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == "hello world goodbye world"

    def test_translations_verbose_json_advertises_translate_task(self, _stub_engine):
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/translations",
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                },
                files={"file": ("tone.wav", _make_tone_wav(), "audio/wav")},
            )
        finally:
            restore()
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["task"] == "translate", (
            "verbose_json from /v1/audio/translations must set task='translate' "
            "so callers can disambiguate from /v1/audio/transcriptions output."
        )


class TestSubtitleTimestampRollover:
    """Codex r1 BLOCKING regression: ``_format_srt_timestamp`` /
    ``_format_vtt_timestamp`` must carry the millisecond rollover
    (``59.9996`` → 60s) all the way up the hierarchy, never producing
    invalid components like ``00:00:60,000``.
    """

    @pytest.mark.parametrize(
        "seconds,expected_srt,expected_vtt",
        [
            # The reported rollover case — 0.9996 rounds to 1.000.
            (0.9996, "00:00:01,000", "00:00:01.000"),
            # 59.9996 used to overflow to 00:00:60,000 — invalid.
            (59.9996, "00:01:00,000", "00:01:00.000"),
            # 3599.9996 used to overflow to 00:59:60,000 — invalid.
            (3599.9996, "01:00:00,000", "01:00:00.000"),
            # Sanity checks: clean values produce clean output.
            (0.0, "00:00:00,000", "00:00:00.000"),
            (1.5, "00:00:01,500", "00:00:01.500"),
            (61.25, "00:01:01,250", "00:01:01.250"),
            (3723.0, "01:02:03,000", "01:02:03.000"),
            # Negative input clamps to zero (defective backend guard).
            (-1.5, "00:00:00,000", "00:00:00.000"),
        ],
    )
    def test_timestamp_rollover_carries_correctly(
        self, seconds, expected_srt, expected_vtt
    ):
        from vllm_mlx.routes.audio import (
            _format_srt_timestamp,
            _format_vtt_timestamp,
        )

        srt = _format_srt_timestamp(seconds)
        vtt = _format_vtt_timestamp(seconds)
        assert srt == expected_srt, (
            f"SRT rollover bug: {seconds!r} → {srt!r} (expected {expected_srt!r}). "
            "Codex r1 BLOCKING: seconds component must not exceed 59 — carries "
            "must propagate up to minutes/hours."
        )
        assert vtt == expected_vtt, (
            f"VTT rollover bug: {seconds!r} → {vtt!r} (expected {expected_vtt!r}). "
            "Same rollover rule applies to WebVTT."
        )

    def test_no_component_exceeds_documented_max(self):
        """Walk a fine-grained sweep around each carry boundary and
        confirm seconds, minutes never appear as ``60``.

        Belt-and-braces on top of the parametrised cases above — catches
        a future regression that special-cases a single boundary while
        leaving the adjacent one broken.
        """
        import re

        from vllm_mlx.routes.audio import (
            _format_srt_timestamp,
            _format_vtt_timestamp,
        )

        srt_re = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
        vtt_re = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")

        # Sweep each carry boundary at sub-ms precision so the rounding
        # behaviour is fully exercised.
        for base in (0, 1, 59, 60, 3599, 3600, 3601):
            for delta in (0.0, 0.0004, 0.0005, 0.0009, 0.0010, 0.9996, 0.9999):
                t = float(base) + delta
                srt = _format_srt_timestamp(t)
                vtt = _format_vtt_timestamp(t)
                m_srt = srt_re.fullmatch(srt)
                m_vtt = vtt_re.fullmatch(vtt)
                assert m_srt is not None, f"SRT shape broken at {t}: {srt!r}"
                assert m_vtt is not None, f"VTT shape broken at {t}: {vtt!r}"
                _h, m, s, _ms = m_srt.groups()
                assert int(m) < 60, f"SRT minutes >= 60 at {t}: {srt!r}"
                assert int(s) < 60, f"SRT seconds >= 60 at {t}: {srt!r}"
                _h, m, s, _ms = m_vtt.groups()
                assert int(m) < 60, f"VTT minutes >= 60 at {t}: {vtt!r}"
                assert int(s) < 60, f"VTT seconds >= 60 at {t}: {vtt!r}"
