# SPDX-License-Identifier: Apache-2.0
"""An omitted-``model`` TTS request must follow the SERVED model, not Kokoro.

Dogfood finding: ``rapid-mlx serve qwen3-tts-voicedesign`` answered a bare
``/v1/audio/voices`` with Kokoro's voices, and a bare ``/v1/audio/speech`` tried
to LOAD Kokoro — both contradicting the ``Audio mode: qwen3-tts-voicedesign``
banner the server printed at boot. The routes hard-coded ``DEFAULT_TTS_ALIAS``
for the omitted-model case instead of consulting the served model.

These pin :func:`_served_tts_default` and the default branch of
:func:`_resolve_tts_model`, which is the shared resolution both routes go
through. Hermetic: ``get_config`` is stubbed, so no server and no ``mlx_audio``.
"""

from types import SimpleNamespace

import pytest

from rapid_mlx.api.models import AudioSpeechRequest
from rapid_mlx.routes import audio


def _serve(monkeypatch, *, alias=None, name=None):
    """Stub the process config as if ``serve`` stamped these onto it."""
    monkeypatch.setattr(
        "rapid_mlx.config.get_config",
        lambda: SimpleNamespace(model_alias=alias, model_name=name),
    )


@pytest.fixture
def _audio_probes_ok(monkeypatch):
    """No-op the mlx_audio / Kokoro runtime probes so the route bodies run."""
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(
        "rapid_mlx.audio.probe.require_kokoro_runtime", lambda *a, **k: None
    )


def test_no_served_model_falls_back_to_kokoro(monkeypatch):
    """The API-only / text-only config: nothing served → unchanged default."""
    _serve(monkeypatch, alias=None, name=None)
    assert audio._served_tts_default() is None
    for placeholder in (None, "", "default"):
        assert "kokoro" in audio._resolve_tts_model(placeholder).lower()


def test_served_tts_alias_becomes_the_default(monkeypatch):
    """serve <tts-alias> → a bare request resolves to THAT model, not Kokoro."""
    _serve(monkeypatch, alias="qwen3-tts-voicedesign")
    assert audio._served_tts_default() == "qwen3-tts-voicedesign"
    resolved = audio._resolve_tts_model(None)
    assert resolved == audio._resolve_tts_model("qwen3-tts-voicedesign")
    assert "kokoro" not in resolved.lower()


def test_served_tts_by_hf_id_is_recognised(monkeypatch):
    """No alias hop (a full HF id was passed to serve): reverse lookup finds it."""
    hf = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    _serve(monkeypatch, name=hf)
    assert audio._served_tts_default() == hf
    assert audio._resolve_tts_model(None) == audio._resolve_tts_model(hf)
    assert "kokoro" not in audio._resolve_tts_model(None).lower()


def test_served_stt_model_does_not_hijack_the_tts_default(monkeypatch):
    """A served STT model is audio but not TTS — the TTS default stays Kokoro."""
    _serve(monkeypatch, alias="whisper-large-v3")
    assert audio._served_tts_default() is None
    assert "kokoro" in audio._resolve_tts_model(None).lower()


def test_enable_audio_on_a_chat_model_falls_back_to_kokoro(monkeypatch):
    """--enable-audio on a text model: the served alias isn't audio at all."""
    _serve(monkeypatch, alias="qwen3.5-4b-4bit")
    assert audio._served_tts_default() is None
    assert "kokoro" in audio._resolve_tts_model(None).lower()


def test_served_by_custom_name_still_finds_the_tts_model(monkeypatch):
    """``--served-model-name`` case: the alias is an opaque gateway name that
    resolves to nothing, but ``model_name`` carries a real TTS HF id. The
    default must check both, not short-circuit on the truthy alias."""
    hf = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    _serve(monkeypatch, alias="my-gateway-name", name=hf)
    assert audio._served_tts_default() == hf
    assert "kokoro" not in audio._resolve_tts_model(None).lower()


def test_explicit_model_is_never_overridden_by_the_served_default(monkeypatch):
    """An explicit request model always wins over the served-model default."""
    _serve(monkeypatch, alias="qwen3-tts-voicedesign")
    # Client explicitly asks for Kokoro on a VoiceDesign server → honoured.
    kokoro_id = audio.TTS_MODEL_ALIASES["kokoro"]
    assert audio._resolve_tts_model("kokoro") == kokoro_id
    assert "voicedesign" not in kokoro_id.lower()


# --- Route-level: the actual regression paths, not just the helpers ----------


@pytest.mark.asyncio
async def test_voices_route_omitted_model_uses_served(monkeypatch, _audio_probes_ok):
    """GET /v1/audio/voices with no ?model passes the SERVED model — not the
    literal "kokoro" — to the voice resolver; an explicit ?model is honoured."""
    _serve(monkeypatch, alias="qwen3-tts-voicedesign")
    seen: list[str] = []
    monkeypatch.setattr(audio, "_allowed_voices_for", lambda m: seen.append(m) or ["x"])
    await audio.list_voices(model=None)  # omitted → served
    await audio.list_voices(model="default")  # OpenAI placeholder → served too
    await audio.list_voices(model="kokoro")  # explicit → honoured
    assert seen == ["qwen3-tts-voicedesign", "qwen3-tts-voicedesign", "kokoro"]


@pytest.mark.asyncio
async def test_speech_route_omitted_model_resolves_to_served(
    monkeypatch, _audio_probes_ok
):
    """POST /v1/audio/speech with no ``model`` synthesises with the served
    model's HF id (via model_fields_set → None → served default), and an
    explicit ``model`` is honoured verbatim. Spies the blocking synth so no
    real engine loads; its first arg is the fully resolved model_name."""
    _serve(monkeypatch, alias="qwen3-tts-voicedesign")
    captured: list[str] = []
    monkeypatch.setattr(
        audio,
        "_generate_speech_blocking",
        lambda model_name, *a, **k: captured.append(model_name) or (b"RIFF0", 24000, 1),
    )

    await audio.create_speech(AudioSpeechRequest(input="hi"))
    assert captured[-1] == audio._resolve_tts_model("qwen3-tts-voicedesign")
    assert "voicedesign" in captured[-1].lower()

    await audio.create_speech(
        AudioSpeechRequest(model="kokoro", input="hi", voice="af_heart")
    )
    assert captured[-1] == audio.TTS_MODEL_ALIASES["kokoro"]
    assert "voicedesign" not in captured[-1].lower()
