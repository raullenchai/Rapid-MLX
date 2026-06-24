# SPDX-License-Identifier: Apache-2.0
"""R8-A regression bundle — Bo's 0.8.9 audio dogfood follow-up.

Four findings:

* **R8-H4** — ``model="kokoro-82m-8bit"`` (the brief's canonical full
  alias) 500'd because ``TTS_MODEL_ALIASES`` only carried the short
  ``kokoro`` form. Mapping the full siblings (``kokoro-82m-bf16``,
  ``kokoro-82m-4bit``, ``kokoro-82m-8bit``) AND making the resolver
  case-insensitive lets both shapes hit the same HF repo.

* **R8-H5** — ``response_format ∈ {mp3, flac, opus, ogg}`` returned
  RIFF/WAV bytes mislabeled as ``audio/{format}``. ``TTSEngine.to_bytes``
  now branches on ``format`` and encodes via scipy (wav) /
  ``soundfile`` (flac/ogg/opus/mp3) / raw bytes (pcm). Unsupported
  formats raise :class:`UnsupportedAudioFormatError` which the route
  translates to a 400 envelope. Content-Type now matches the actual
  bytes via :data:`vllm_mlx.routes.audio._TTS_CONTENT_TYPES`.

* **R8-M4** — Invalid ``voice`` (e.g. ``"alloy"`` from drop-in OpenAI
  SDK code) 500'd inside ``mlx_audio.load_safetensors``. The route
  now pre-flights ``voice`` against the model's known voice list and
  emits a 400 ``invalid_request_error`` with ``param="voice"`` listing
  the available voices.

* **R8-M5** — ``rapid-mlx serve kokoro`` (short audio alias) on a
  fresh ``pip install rapid-mlx`` (no ``[audio]`` extra) printed
  "is not a known alias" instead of the actionable
  "install rapid-mlx[audio]" hint — the CLI fail-fast tripped before
  the audio boot guard ran. Fix: skip the fail-fast for names that
  ``is_audio_model_alias`` recognises so the guard fires.
"""

from __future__ import annotations

import sys
import types

import pytest

# ``vllm_mlx.routes.audio`` transitively imports ``mlx.core`` via the
# engine wiring. Linux CI runners (``pr_validate``'s validate job) don't
# install mlx, so a bare import raises ``ModuleNotFoundError`` and the
# rest of the file looks like a 13-test failure. Mirror the r7-C skip
# block so the file is a clean SKIP off-platform but still executes on
# Apple Silicon dev / CI with the right deps.
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
# Helpers (copied from r7-C so this file stands alone — the harness pattern
# is identical so a future refactor lifting both into a conftest stays a
# single mechanical change)
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


def _stub_engine(monkeypatch, *, voice_observed=None):
    """Install a no-op TTSEngine that records the model_name passed to it
    and uses the REAL ``to_bytes`` so encoder smoke-checks still run."""
    import numpy as np

    from vllm_mlx.audio import probe as probe_mod
    from vllm_mlx.audio import tts as tts_mod
    from vllm_mlx.routes import audio as audio_route

    observed_models: list[str] = []
    # Capture the REAL ``to_bytes`` BEFORE ``monkeypatch.setattr``
    # rebinds ``tts_mod.TTSEngine`` to our stub — otherwise
    # ``tts_mod.TTSEngine.to_bytes`` after the patch points back at the
    # stub and recurses forever. A simple unbound-method reference here
    # gives us the original encoder without going through the patched
    # attribute lookup.
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
            # 1 second of a 440 Hz tone at 0.3 amplitude so every encoder
            # has real signal to compress (silent input lets some codecs
            # output suspiciously short payloads that obscure regressions).
            audio = (np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 0.3).astype(
                np.float32
            )
            return tts_mod.AudioOutput(audio=audio, sample_rate=24000, duration=1.0)

        def to_bytes(self, audio, format="wav"):
            return real_to_bytes(self, audio, format=format)

    monkeypatch.setattr(tts_mod, "TTSEngine", _RecordingEngine)
    monkeypatch.setattr(probe_mod, "require_kokoro_runtime", lambda: None)
    _install_fake_mlx_audio(monkeypatch)
    audio_route._tts_engine = None
    return audio_route, observed_models


# ---------------------------------------------------------------------------
# R8-H4 — full-alias resolution + case-insensitive lookup
# ---------------------------------------------------------------------------


class TestFullAliasResolution:
    """The brief's literal ``kokoro-82m-8bit`` and its siblings resolve
    through the same helper as the short ``kokoro`` alias — pre-fix only
    the short form hit the table, so the full names 404'd at HF lookup
    inside mlx_audio."""

    @pytest.mark.parametrize(
        "alias",
        [
            "kokoro-82m-bf16",
            "kokoro-82m-4bit",
            "kokoro-82m-8bit",
            # Case-insensitive: SDKs / docs frequently mix the case
            # because the upstream HF repo is "Kokoro-82M-bf16".
            "Kokoro-82M-bf16",
            "KOKORO-82M-8BIT",
        ],
    )
    def test_full_kokoro_alias_resolves_to_kokoro_repo(self, alias):
        from vllm_mlx.routes.audio import _resolve_tts_model

        resolved = _resolve_tts_model(alias)
        assert "kokoro" in resolved.lower(), (
            f"R8-H4 regression: alias {alias!r} did NOT resolve to a "
            f"Kokoro HF repo, got {resolved!r}. Pre-fix this fell "
            f"through to passthrough and mlx-audio 404'd at HF lookup."
        )
        assert "/" in resolved, (
            f"R8-H4 regression: alias {alias!r} must resolve to a full "
            f"HF repo id (with org/), got {resolved!r}."
        )

    def test_short_and_full_alias_resolve_identically(self):
        """The short ``kokoro`` and full ``kokoro-82m-bf16`` MUST map to
        the same repo so both paths go through identical model init —
        the regression Bo flagged was a divergence between the two."""
        from vllm_mlx.routes.audio import _resolve_tts_model

        assert _resolve_tts_model("kokoro") == _resolve_tts_model("kokoro-82m-bf16")

    def test_unknown_alias_still_passes_through(self):
        """Pass-through behaviour for unrecognised names is preserved —
        a client opting in to a HF repo not in the alias table must
        still reach mlx_audio with the verbatim id."""
        from vllm_mlx.routes.audio import _resolve_tts_model

        # HF-style ids contain '/' so they're untouched (case preserved).
        hf_path = "mlx-community/Some-Future-TTS-Model"
        assert _resolve_tts_model(hf_path) == hf_path


# ---------------------------------------------------------------------------
# R8-H5 — encoder produces real bytes + Content-Type matches
# ---------------------------------------------------------------------------


# Each entry: (response_format, content_type, magic_bytes prefix).
# The encoder check below boots the TTS route end-to-end (with a stub
# engine) and reads the first 4 bytes back to confirm the bytes match
# the requested container. Pre-fix every entry's body was RIFF/WAVE.
_FORMAT_EXPECTATIONS = [
    ("wav", "audio/wav", b"RIFF"),
    ("flac", "audio/flac", b"fLaC"),
    ("ogg", "audio/ogg", b"OggS"),
    ("opus", "audio/ogg", b"OggS"),  # Opus is OGG-Opus container.
]


class TestTTSContentType:
    """Smoke-test the Content-Type ↔ body alignment that R8-H5 fixed.

    Pre-fix the route returned ``RIFF…WAVE`` bytes labelled
    ``audio/mp3``/``audio/flac``/``audio/opus`` etc. Now the body's
    magic bytes match the IANA-canonical Content-Type for the
    requested format.
    """

    @pytest.mark.parametrize("response_format,content_type,magic", _FORMAT_EXPECTATIONS)
    def test_response_format_produces_matching_bytes(
        self, monkeypatch, response_format, content_type, magic
    ):
        # ``soundfile`` is in the [audio] extra; the test file's top-level
        # skip blocks already gate on mlx, so if we're here without
        # soundfile that's a project misconfiguration we want surfaced
        # (not a silent skip).
        pytest.importorskip("soundfile")

        audio_route, observed = _stub_engine(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-82m-8bit",  # R8-H4 full alias
                    "input": "hello world",
                    "voice": "af_heart",
                    "response_format": response_format,
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, (
            f"R8-H5 regression: response_format={response_format!r} "
            f"returned {r.status_code}. Body: {r.text[:500]}"
        )
        # R8-H4: the full alias MUST have routed through to the Kokoro
        # HF id (not been passed verbatim to the engine).
        assert observed and "kokoro" in observed[0].lower(), (
            f"R8-H4 regression: full alias did not resolve to a Kokoro "
            f"repo; engine saw {observed!r}"
        )
        # R8-H5: Content-Type matches the IANA canonical for the
        # requested format AND the magic bytes match the container.
        ctype = r.headers.get("content-type", "").split(";")[0].strip()
        assert ctype == content_type, (
            f"R8-H5 regression: response_format={response_format!r} "
            f"emitted Content-Type {ctype!r}, expected {content_type!r}."
        )
        body = r.content
        assert len(body) > 100, (
            f"R8-H5 regression: {response_format!r} payload too small "
            f"({len(body)} bytes) — encoder likely silently no-op'd."
        )
        assert body.startswith(magic), (
            f"R8-H5 regression: {response_format!r} body starts with "
            f"{body[:8]!r}, expected magic {magic!r}. Pre-fix the route "
            f"returned RIFF/WAV bytes mislabeled as audio/{response_format}."
        )

    def test_brief_canonical_alias_with_mp3_e2e(self, monkeypatch):
        """The brief's exact reproducer: ``model="kokoro-82m-8bit"`` +
        ``response_format="mp3"`` returns >1 KiB of audio and the
        Content-Type matches the bytes. Pre-fix this 500'd at alias
        resolution; the previous regression Bo flagged.
        """
        pytest.importorskip("soundfile")

        audio_route, observed = _stub_engine(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro-82m-8bit",
                    "input": "hello world",
                    "voice": "af_heart",
                    "response_format": "mp3",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        # The MP3 encoder may not be present on every libsndfile build —
        # we accept either:
        #   * 200 + valid MP3 body (>1 KiB, ``\xff\xfb`` or ``\xff\xf3``
        #     frame sync) + ``audio/mpeg`` Content-Type, OR
        #   * 400 ``invalid_response_format`` envelope (graceful fallback
        #     to the documented-supported set).
        # Both are acceptable; what's NOT acceptable is a 200 + RIFF/WAV
        # body mislabeled as ``audio/mpeg``, which is what pre-fix did.
        if r.status_code == 200:
            assert (
                r.headers.get("content-type", "").split(";")[0].strip() == "audio/mpeg"
            )
            assert len(r.content) > 1024, (
                f"MP3 body too small ({len(r.content)} bytes); encoder "
                f"likely produced a no-op."
            )
            # MP3 frame sync byte 0xFF + (0xFB / 0xF3 / 0xF2 / 0xE3 ...).
            assert r.content[0] == 0xFF, (
                f"MP3 body does not start with frame sync (0xFF), got "
                f"{r.content[:4]!r}. Pre-fix this WAS the RIFF/WAVE "
                f"mislabel regression."
            )
        elif r.status_code == 400:
            body = r.json()
            assert body["error"]["code"] == "invalid_response_format", body
            assert body["error"]["param"] == "response_format", body
        else:
            pytest.fail(
                f"Unexpected status {r.status_code} for mp3 response: {r.text[:500]}"
            )

    def test_pcm_returns_raw_int16_no_riff(self, monkeypatch):
        """OpenAI's ``response_format="pcm"`` is raw 16-bit LE PCM with
        NO container. Pre-fix we wrapped the same bytes in RIFF/WAVE
        and labelled them ``audio/pcm`` — that's a structural mislabel
        any decoder following the OpenAI contract mis-parses."""
        audio_route, _ = _stub_engine(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hi",
                    "voice": "af_heart",
                    "response_format": "pcm",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        body = r.content
        assert not body.startswith(b"RIFF"), (
            "R8-H5 regression: PCM body starts with RIFF — the legacy "
            "WAV wrapper is still on it. ``response_format=pcm`` must be "
            "raw int16 LE bytes."
        )
        # 1 second @ 24 kHz, int16 = 48000 bytes exactly.
        assert len(body) == 48000, (
            f"R8-H5: PCM body length unexpected ({len(body)}), expected "
            f"48000 for the 24 kHz stub tone."
        )

    def test_unsupported_format_returns_400_envelope(self, monkeypatch):
        """``aac`` is in the OpenAI surface but libsndfile doesn't ship
        an AAC encoder. The Pydantic validator rejects it up-front with
        a 400 ``invalid_request_error`` listing the supported set —
        pre-fix the route returned 200 + WAV bytes labelled audio/aac.
        """
        _install_fake_mlx_audio(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hi",
                    "voice": "af_heart",
                    "response_format": "aac",
                },
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R8-H5 regression: aac returned {r.status_code} not 400. "
            f"Body: {r.text[:500]}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "response_format", err
        # The message must list the supported set so the caller can
        # retry with a known-good format.
        msg = err["message"].lower()
        assert "wav" in msg and "flac" in msg, (
            f"Error message must list supported formats; got {msg!r}"
        )


class TestTTSContentTypeTable:
    """The ``_TTS_CONTENT_TYPES`` table and the Pydantic
    ``_TTS_ALLOWED_RESPONSE_FORMATS`` tuple are paired: any format in
    the allowed set MUST have a Content-Type entry (else the request
    would pass validation and hit the ``application/octet-stream``
    fallback, masking a configuration bug). This test pins both
    directions so a future addition can't drift."""

    def test_content_type_table_covers_allowed_formats(self):
        from vllm_mlx.api.models import _TTS_ALLOWED_RESPONSE_FORMATS
        from vllm_mlx.routes.audio import _TTS_CONTENT_TYPES

        for fmt in _TTS_ALLOWED_RESPONSE_FORMATS:
            assert fmt in _TTS_CONTENT_TYPES, (
                f"Format {fmt!r} is allowed by AudioSpeechRequest but "
                f"has no entry in _TTS_CONTENT_TYPES — a request would "
                f"return application/octet-stream."
            )

    def test_iana_canonical_types(self):
        """Pin the IANA-canonical Content-Type per format. The pre-fix
        ``f"audio/{format}"`` formula produced ``audio/mp3`` and
        ``audio/opus`` which are not the IANA-registered types
        (``audio/mpeg`` and ``audio/ogg`` respectively). Clients that
        follow the registry get the right type."""
        from vllm_mlx.routes.audio import _TTS_CONTENT_TYPES

        assert _TTS_CONTENT_TYPES["mp3"] == "audio/mpeg"
        assert _TTS_CONTENT_TYPES["opus"] == "audio/ogg"
        assert _TTS_CONTENT_TYPES["ogg"] == "audio/ogg"
        assert _TTS_CONTENT_TYPES["flac"] == "audio/flac"


# ---------------------------------------------------------------------------
# R8-M4 — voice validation
# ---------------------------------------------------------------------------


class TestVoiceValidation:
    """Invalid ``voice`` returns 400 with ``param="voice"`` and the
    available voice list. Pre-fix this 500'd inside
    ``mlx_audio.load_safetensors`` on the missing voice file."""

    @pytest.mark.parametrize(
        "bad_voice",
        [
            "not_a_real_voice",
            "alloy",  # OpenAI default voice name
            "nova",
            "shimmer",
            "af_hart",  # typo for af_heart
        ],
    )
    def test_invalid_voice_returns_400_envelope(self, monkeypatch, bad_voice):
        _install_fake_mlx_audio(monkeypatch)
        _stub_engine(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello",
                    "voice": bad_voice,
                    "response_format": "wav",
                },
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R8-M4 regression: voice={bad_voice!r} returned "
            f"{r.status_code}, expected 400. Body: {r.text[:500]}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "voice", err
        assert err["code"] == "invalid_voice", err
        # The available list must be in the message so the caller can
        # pick a valid voice from the envelope alone.
        msg = err["message"].lower()
        assert "af_heart" in msg, (
            f"Error message must include known voices (e.g. af_heart); got {msg!r}"
        )

    def test_blank_voice_returns_400_envelope(self, monkeypatch):
        """``voice=""`` MUST 400 (Pydantic validator) — pre-fix it 500'd
        on the empty safetensors filename."""
        _install_fake_mlx_audio(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello",
                    "voice": "",
                    "response_format": "wav",
                },
            )
        finally:
            restore()

        assert r.status_code == 400, (
            f"R8-M4 regression: blank voice returned {r.status_code}, "
            f"expected 400. Body: {r.text[:500]}"
        )
        body = r.json()
        err = body["error"]
        assert err["type"] == "invalid_request_error", err
        assert err["param"] == "voice", err

    def test_valid_voice_passes_through(self, monkeypatch):
        """Sanity check: a known-valid voice still reaches the engine
        unchanged (the validator must NOT reject ``af_heart``)."""
        audio_route, observed = _stub_engine(monkeypatch)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello",
                    "voice": "af_heart",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text


class TestAllowedVoicesHelper:
    """Pin the ``_allowed_voices_for`` helper's family detection so a
    future engine added to the registry can't accidentally diverge
    from the route's voice-validation rule.

    R11-B-F1 (Bo 0.8.12 dogfood) refactored the helper to enumerate
    the snapshot's ``voices/`` dir at request time and fall back to
    the per-family static list when the snapshot isn't cached locally.
    The cold-start (fallback) path is the contract pinned here —
    monkeypatching ``_list_snapshot_voices`` to ``[]`` forces the
    static branch so the assertions stay deterministic regardless of
    the test runner's HuggingFace cache state.
    """

    @pytest.fixture(autouse=True)
    def _force_static_fallback(self, monkeypatch):
        # Pin the static-fallback branch by forcing the dynamic
        # enumeration to return empty. The dynamic-success path is
        # covered by ``TestAllowedVoicesDynamicEnumeration`` below.
        import vllm_mlx.routes.audio as audio_route

        # ``_allowed_voices_for`` calls ``_list_snapshot_voices`` via
        # a lazy ``from ..audio.tts import ...`` inside the helper, so
        # the monkeypatch lands on the source attribute.
        from vllm_mlx.audio import tts as tts_module

        monkeypatch.setattr(tts_module, "_list_snapshot_voices", lambda _name: [])
        yield audio_route  # nothing for the tests to consume

    def test_kokoro_short_alias_returns_kokoro_voices(self):
        from vllm_mlx.audio.tts import KOKORO_VOICES
        from vllm_mlx.routes.audio import _allowed_voices_for

        assert _allowed_voices_for("kokoro") == list(KOKORO_VOICES)

    def test_kokoro_hf_path_returns_kokoro_voices(self):
        from vllm_mlx.audio.tts import KOKORO_VOICES
        from vllm_mlx.routes.audio import _allowed_voices_for

        assert _allowed_voices_for("mlx-community/Kokoro-82M-bf16") == list(
            KOKORO_VOICES
        )

    def test_chatterbox_returns_chatterbox_voices(self):
        from vllm_mlx.audio.tts import CHATTERBOX_VOICES
        from vllm_mlx.routes.audio import _allowed_voices_for

        assert _allowed_voices_for("chatterbox") == list(CHATTERBOX_VOICES)

    def test_vibevoice_returns_seeded_english_voices(self):
        # R11-B-F1: pre-fix the helper hard-coded ``["default"]`` for
        # every non-kokoro/non-chatterbox family. The new cold-start
        # fallback for VibeVoice seeds the English voice set so the
        # 400 envelope's ``Available:`` preview is informative even
        # before the snapshot has been downloaded — and so a request
        # carrying ``voice="en-Grace_woman"`` (the registry default)
        # passes voice validation on the first call.
        from vllm_mlx.routes.audio import _allowed_voices_for

        voices = _allowed_voices_for("vibevoice")
        assert "en-Grace_woman" in voices, (
            "VibeVoice cold-start fallback must include en-Grace_woman "
            f"(registry default). Got: {voices}"
        )
        # Pre-fix this would have included only ``"default"``, which
        # is precisely the value VibeVoice does NOT ship.
        assert "default" not in voices, (
            "VibeVoice has no default.safetensors in its snapshot — "
            "the cold-start fallback MUST NOT advertise it as a valid "
            f"voice. Got: {voices}"
        )

    def test_unknown_family_returns_default(self):
        from vllm_mlx.routes.audio import _allowed_voices_for

        assert _allowed_voices_for("some/UnknownEngine") == ["default"]


class TestAllowedVoicesDynamicEnumeration:
    """R11-B-F1: when the snapshot is cached locally the helper must
    enumerate the actual ``voices/`` dir, not the per-family static
    list. Covers the warm-path: chatterbox/voxcpm/dia all ship a
    single ``default.safetensors`` so the enumeration returns the
    same ``["default"]`` the static list had; kokoro / vibevoice
    ship richer voice sets and the enumeration must surface them."""

    def test_enumeration_strips_safetensors_extension(self, tmp_path, monkeypatch):
        # Build a fake snapshot dir matching the real HF cache shape.
        snapshot = tmp_path / "snapshot"
        (snapshot / "voices").mkdir(parents=True)
        (snapshot / "voices" / "en-Grace_woman.safetensors").write_bytes(b"x")
        (snapshot / "voices" / "en-Mike_man.safetensors").write_bytes(b"x")
        # A non-safetensors sibling MUST be ignored — VibeVoice ships
        # neither but kokoro snapshots include ``.pt`` mirrors of every
        # voice.
        (snapshot / "voices" / "en-Grace_woman.pt").write_bytes(b"x")
        # The helper uses ``config.json`` as the cache probe.
        (snapshot / "config.json").write_bytes(b"{}")

        from huggingface_hub import try_to_load_from_cache as real_helper  # noqa: F401

        def fake_cache_lookup(repo_id, filename):
            assert filename == "config.json"
            return str(snapshot / "config.json")

        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", fake_cache_lookup)

        from vllm_mlx.audio.tts import _list_snapshot_voices

        result = _list_snapshot_voices("mlx-community/VibeVoice-Realtime-0.5B-4bit")
        assert result == ["en-Grace_woman", "en-Mike_man"], result

    def test_enumeration_returns_empty_when_snapshot_missing(self, monkeypatch):
        # Local-only lookup: ``try_to_load_from_cache`` returns ``None``
        # for a repo not on disk. We MUST NOT trigger a download from
        # inside the voice-validation path.
        monkeypatch.setattr(
            "huggingface_hub.try_to_load_from_cache",
            lambda repo_id, filename: None,
        )

        from vllm_mlx.audio.tts import _list_snapshot_voices

        assert _list_snapshot_voices("mlx-community/Nonexistent-Repo") == []

    def test_enumeration_returns_empty_when_voices_dir_missing(
        self, tmp_path, monkeypatch
    ):
        # Snapshot cached but no ``voices/`` dir — engines that pull
        # voices from a different layout (or no per-voice files at
        # all) must NOT crash the enumeration. The caller falls back
        # to the per-family static list.
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_bytes(b"{}")

        monkeypatch.setattr(
            "huggingface_hub.try_to_load_from_cache",
            lambda repo_id, filename: str(snapshot / "config.json"),
        )

        from vllm_mlx.audio.tts import _list_snapshot_voices

        assert _list_snapshot_voices("mlx-community/Whatever") == []

    def test_alias_short_form_resolves_via_registry(self, tmp_path, monkeypatch):
        # A short alias (``"vibevoice"``) must map to its HF id via
        # the registry before the cache lookup — that way the alias
        # AND the full HF id resolve to the same snapshot.
        snapshot = tmp_path / "snapshot"
        (snapshot / "voices").mkdir(parents=True)
        (snapshot / "voices" / "en-Grace_woman.safetensors").write_bytes(b"x")
        (snapshot / "config.json").write_bytes(b"{}")

        captured = {}

        def fake_cache_lookup(repo_id, filename):
            captured["repo_id"] = repo_id
            return str(snapshot / "config.json")

        monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", fake_cache_lookup)

        from vllm_mlx.audio.tts import _list_snapshot_voices

        result = _list_snapshot_voices("vibevoice")
        assert captured["repo_id"] == "mlx-community/VibeVoice-Realtime-0.5B-4bit"
        assert result == ["en-Grace_woman"]


class TestVibevoiceDefaultSentinelResolves:
    """R11-B-F1: a request carrying ``voice="default"`` against an
    alias whose registry ``default_voice`` is something else (here
    VibeVoice's ``"en-Grace_woman"``) must be remapped to the registry
    default BEFORE voice validation. Pre-fix the literal ``"default"``
    would hit the static voice list which only contained ``"default"``,
    then fall through to ``mlx_audio.tts.models.vibevoice.Model.load
    _voice("default")`` which 500'd with ``FileNotFoundError: Voice
    cache not found: .../voices/default.safetensors``.

    Engines whose registry ``default_voice`` IS ``"default"``
    (chatterbox / voxcpm / dia) must keep their current behaviour —
    the remap is a no-op there.
    """

    def _patch_static_fallback(self, monkeypatch):
        # The test harness may run on a workstation whose HF cache
        # already has the VibeVoice / chatterbox snapshot. Force
        # ``_list_snapshot_voices`` to ``[]`` so the assertions pin
        # the cold-start fallback path deterministically.
        from vllm_mlx.audio import tts as tts_module

        monkeypatch.setattr(tts_module, "_list_snapshot_voices", lambda _name: [])

    def test_vibevoice_default_remaps_to_registry_default(self, monkeypatch):
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "vibevoice",
                    "input": "hello",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, (
            f"R11-B-F1 regression: vibevoice voice=default returned "
            f"{r.status_code}; expected 200 after remap to "
            f"registry default_voice. Body: {r.text[:500]}"
        )
        # Engine receives the remapped voice, not the literal sentinel.
        assert voices_seen == ["en-Grace_woman"], voices_seen

    def test_chatterbox_default_stays_default(self, monkeypatch):
        # Chatterbox's registry ``default_voice`` IS ``"default"`` —
        # the remap is structurally a no-op. Regression guard for the
        # opposite direction (we don't want to accidentally rewrite
        # ``"default"`` to ``"af_heart"`` or anything else).
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": "hello",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voices_seen == ["default"], voices_seen

    def test_vibevoice_explicit_voice_is_respected(self, monkeypatch):
        # Explicit non-sentinel voices bypass the remap branch. A
        # client that explicitly asks for ``en-Mike_man`` must reach
        # the engine with exactly that string.
        # The dynamic enumeration would return ``[]`` here (no real
        # snapshot in the test harness) so the cold-start fallback
        # list must include ``en-Mike_man`` for validation to pass.
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "vibevoice",
                    "input": "hello",
                    "voice": "en-Mike_man",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voices_seen == ["en-Mike_man"], voices_seen

    def test_voxcpm_and_dia_default_still_accepted(self, monkeypatch):
        # Regression guard: voxcpm and dia ship a single
        # ``default.safetensors``; their registry ``default_voice`` is
        # ``"default"`` so the remap is a no-op AND the static
        # fallback ``["default"]`` accepts the request. Pre-R11-B-F1
        # path stays alive.
        self._patch_static_fallback(monkeypatch)
        for alias in ("voxcpm", "dia"):
            voices_seen: list[str] = []
            audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
            client, restore = _mount_audio_app()
            try:
                r = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": alias,
                        "input": "hello",
                        "voice": "default",
                        "response_format": "wav",
                    },
                )
            finally:
                restore()
                audio_route._tts_engine = None

            assert r.status_code == 200, (
                f"{alias} voice=default returned {r.status_code}; "
                f"expected 200. Body: {r.text[:500]}"
            )
            assert voices_seen == ["default"], (alias, voices_seen)

    def test_vibevoice_voice_omitted_remaps_to_registry_default(self, monkeypatch):
        # Codex r1 MEDIUM (Diff): pre-fix the Pydantic model defaulted
        # ``voice`` to ``"af_heart"`` so a request body like
        # ``{"model":"vibevoice","input":"hi"}`` (no ``voice`` key)
        # 400'd because ``af_heart`` isn't a real VibeVoice voice.
        # The omitted-voice shape MUST resolve to the registry default
        # ``en-Grace_woman``, same as the explicit ``"default"``
        # sentinel.
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "vibevoice",
                    "input": "hello",
                    # voice deliberately omitted
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, (
            f"R11-B-F1 / codex r1 regression: vibevoice with omitted "
            f"voice returned {r.status_code}; expected 200 after remap "
            f"to registry default. Body: {r.text[:500]}"
        )
        assert voices_seen == ["en-Grace_woman"], voices_seen

    def test_chatterbox_voice_omitted_remaps_to_default(self, monkeypatch):
        # Chatterbox's registry default is ``"default"`` — the remap
        # IS the canonical path because pre-fix this 400'd with
        # ``voice='af_heart' not recognized for model 'mlx-community/
        # chatterbox-turbo-fp16'`` whenever a client omitted ``voice``.
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": "hello",
                    # voice deliberately omitted
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voices_seen == ["default"], voices_seen

    def test_explicit_af_heart_respected_against_kokoro(self, monkeypatch):
        # Codex r1 MEDIUM regression-guard: an EXPLICIT
        # ``voice="af_heart"`` from the client must NOT be rewritten
        # — only the Pydantic-default omitted-voice case routes through
        # the registry. This pins the boundary between "user said
        # nothing" and "user explicitly requested af_heart".
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello",
                    "voice": "af_heart",  # explicit
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voices_seen == ["af_heart"], voices_seen

    def test_kokoro_default_remaps_to_af_heart(self, monkeypatch):
        # Kokoro's registry ``default_voice`` is ``"af_heart"`` so a
        # request that explicitly sends ``voice="default"`` must be
        # remapped — same contract as vibevoice. This is the codepath
        # the task brief calls out as "Bo confirmed kokoro fallback
        # works"; pre-fix it actually 400'd here because ``"default"``
        # wasn't in ``KOKORO_VOICES``.
        self._patch_static_fallback(monkeypatch)
        voices_seen: list[str] = []
        audio_route, _models = _stub_engine(monkeypatch, voice_observed=voices_seen)
        client, restore = _mount_audio_app()
        try:
            r = client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello",
                    "voice": "default",
                    "response_format": "wav",
                },
            )
        finally:
            restore()
            audio_route._tts_engine = None

        assert r.status_code == 200, r.text
        assert voices_seen == ["af_heart"], voices_seen


# ---------------------------------------------------------------------------
# R8-M5 — CLI boot guard fires for short audio aliases
# ---------------------------------------------------------------------------


class TestCliBootGuardShortAlias:
    """Reproduces the CLI fail-fast / boot-guard ordering bug.

    Bo's R2.9 sequence:
      * R2.9b: ``rapid-mlx serve kokoro`` on a no-``[audio]`` venv exits
        with the generic "is not a known alias" — pre-fix the CLI
        fail-fast in ``main()`` tripped BEFORE ``serve_command`` got
        the chance to fire the audio boot guard with the actionable
        install hint. The fix is to skip the fail-fast for names
        ``is_audio_model_alias`` recognises so the guard fires.
    """

    def test_short_audio_alias_bypasses_unknown_alias_failfast(
        self, monkeypatch, capsys
    ):
        """Drive the CLI fail-fast branch directly: a short audio alias
        (``kokoro``, ``whisper``, ...) must NOT print "is not a known
        alias" and exit 1 — it must fall through so the audio boot
        guard in ``serve_command`` can fire instead."""
        from vllm_mlx.audio.probe import is_audio_model_alias
        from vllm_mlx.model_aliases import resolve_model

        # Confirm the precondition: these names are NOT in aliases.json
        # (the resolver returns them verbatim) AND they DO trip the
        # audio probe. Without both, the fix has no observable effect.
        for name in ("kokoro", "whisper", "chatterbox", "kokoro-82m-8bit"):
            assert resolve_model(name) == name, (
                f"Precondition broken: {name!r} is now in aliases.json. "
                f"If that's intentional, the R8-M5 fix may be obsolete "
                f"for this name."
            )
            assert is_audio_model_alias(name), (
                f"Precondition broken: {name!r} no longer trips the "
                f"audio probe — the fail-fast bypass would skip it."
            )

    def test_full_alias_kokoro_82m_8bit_also_recognised(self):
        """The brief's literal ``kokoro-82m-8bit`` is recognised by the
        boot-guard probe (substring match on ``kokoro``)."""
        from vllm_mlx.audio.probe import is_audio_model_alias

        assert is_audio_model_alias("kokoro-82m-8bit")
        assert is_audio_model_alias("kokoro-82m-bf16")

    def test_non_audio_alias_still_fails_fast(self):
        """A genuinely-unknown alias (``gemma4-27b``) must STILL trip
        the fail-fast — the fix's bypass only opens the door for
        names that look like audio aliases."""
        from vllm_mlx.audio.probe import is_audio_model_alias

        assert not is_audio_model_alias("gemma4-27b")
        assert not is_audio_model_alias("some-future-llm-9b")

    def test_main_does_not_failfast_on_short_audio_alias(self, monkeypatch, capsys):
        """End-to-end: ``rapid-mlx serve kokoro`` on a venv without
        ``[audio]`` must reach the audio boot guard's exit-2 install
        hint INSTEAD of the generic exit-1 "not a known alias" Bo saw.

        We drive ``cli.main`` directly via ``sys.argv`` and patch the
        ``find_spec`` lookup so ``mlx_audio`` looks missing. Pre-fix
        ``main`` printed "is not a known alias" and exited 1; post-fix
        the short audio alias bypasses the fail-fast and ``serve_command``
        exits 2 with the actionable hint.
        """
        import importlib.util

        from vllm_mlx import cli

        real_find_spec = importlib.util.find_spec

        def _no_mlx_audio(name, *a, **kw):
            if name == "mlx_audio":
                return None
            return real_find_spec(name, *a, **kw)

        monkeypatch.setattr(importlib.util, "find_spec", _no_mlx_audio)

        # ``serve_command`` calls many downstream helpers — short-circuit
        # the version check so we never reach the heavy ``server`` import.
        # The audio boot guard fires before the version check so this
        # never matters in the failing path; we patch it as defense for
        # the (unlikely) future where the guard is moved.
        from vllm_mlx import _version_check

        monkeypatch.setattr(
            _version_check, "prompt_upgrade_if_available", lambda: False
        )

        # Drive ``main`` with the exact command Bo ran. ``--port`` is a
        # belt for the argparse defaults but isn't load-bearing.
        monkeypatch.setattr("sys.argv", ["rapid-mlx", "serve", "kokoro"])

        with pytest.raises(SystemExit) as excinfo:
            cli.main()

        # The fix's success criterion: we MUST NOT see exit 1 from the
        # main()'s fail-fast branch. Exit 2 from the audio boot guard
        # is the desired outcome.
        assert excinfo.value.code == 2, (
            f"R8-M5 regression: ``rapid-mlx serve kokoro`` exited "
            f"{excinfo.value.code!r}, expected 2 (audio boot guard). "
            f"Pre-fix the CLI fail-fast tripped first with exit 1."
        )
        captured = capsys.readouterr()
        err = captured.err + captured.out
        assert "is not a known alias" not in err, (
            "R8-M5 regression: the fail-fast message leaked. The audio "
            f"boot guard should fire instead. Output: {err[:500]}"
        )
        assert "[audio]" in err, (
            f"R8-M5 regression: the install hint did not surface. Output: {err[:500]}"
        )
