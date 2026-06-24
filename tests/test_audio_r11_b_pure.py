# SPDX-License-Identifier: Apache-2.0
"""R11-B regression bundle — platform-independent coverage.

Codex review-20260315-103736 #8 (low BLOCKING): the main R11-B bundle
(``tests/test_audio_r11_b_bundle.py``) skips when ``mlx.core`` /
``mlx_lm`` is missing because it imports the audio route handler
which transitively pulls those in. That left pure
:class:`vllm_mlx.api.models.AudioSpeechRequest` and
:func:`vllm_mlx.routes.models._build_model_info` regressions uncovered
on Linux CI.

This file covers the two diff regressions that DON'T need mlx and
therefore run on every CI runner:

* **R11-B-F2 (model layer)** — the ``format`` → ``response_format``
  legacy alias on :class:`AudioSpeechRequest`. Tested directly against
  the Pydantic model so we never have to mount the route to prove the
  validator wiring.

* **R11-B-F4 (model card)** — the audio capability + modality
  short-circuit in :func:`vllm_mlx.routes.models._build_model_info`.
  Tested by calling the helper directly so we never have to mount
  the audio route either.

The full route-integration coverage stays in
``tests/test_audio_r11_b_bundle.py`` (gated on the platform skip).
This file is the cheap regression net for everyday CI.
"""

from __future__ import annotations

import pytest

# Intentionally NO ``pytest.importorskip("mlx.core")`` here — this file
# MUST execute on Linux CI without mlx installed. The imports below
# are scrubbed to mlx-free paths:
#   * ``vllm_mlx.api.models`` — pure Pydantic models, no mlx.
#   * ``vllm_mlx.routes.models`` — model-card builder for /v1/models;
#     no mlx import.
#   * ``vllm_mlx.audio.registry`` — JSON registry loader; no mlx.


# ===========================================================================
# R11-B-F2 — ``format`` → ``response_format`` alias at the model layer
# ===========================================================================


class TestFormatAliasModelLayer:
    """Validator regressions for the F-2 ``format`` legacy alias. Tested
    directly against :class:`AudioSpeechRequest` so the wiring at the
    Pydantic layer is locked down on every CI runner, not just the
    MLX-enabled ones."""

    def test_format_alias_folds_into_response_format(self):
        """``{"format":"mp3"}`` with no ``response_format`` → the model
        binds ``response_format="mp3"``. Pre-fix this fell through and
        Pydantic populated the field default ``"wav"`` — the silent-
        downgrade shape R11-B-F2 exists to prevent."""
        from vllm_mlx.api.models import AudioSpeechRequest

        r = AudioSpeechRequest(model="kokoro", input="Hi", voice="af_heart", format="mp3")
        assert r.response_format == "mp3"

    def test_explicit_response_format_beats_format_alias(self):
        """Both keys present → the spec-correct ``response_format`` wins.
        Never a silent override of explicit caller intent. The legacy
        alias is back-compat, NOT a silent overwrite."""
        from vllm_mlx.api.models import AudioSpeechRequest

        r = AudioSpeechRequest(
            model="kokoro",
            input="Hi",
            voice="af_heart",
            response_format="wav",
            format="mp3",
        )
        assert r.response_format == "wav"

    def test_format_alias_absent_keeps_pydantic_default(self):
        """``format`` absent → ``response_format`` falls back to the
        Pydantic default. Pin this so the F-2 hook can't accidentally
        clobber the default-resolution path."""
        from vllm_mlx.api.models import AudioSpeechRequest

        r = AudioSpeechRequest(model="kokoro", input="Hi", voice="af_heart")
        assert r.response_format == "wav"

    def test_unknown_format_alias_400s(self):
        """An unsupported codec sent via the legacy spelling (``format=
        "jpeg"``) MUST raise the same ValidationError an explicit
        ``response_format="jpeg"`` would. The alias only changes the
        SOURCE of the value, not the allowed-set contract."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import AudioSpeechRequest

        with pytest.raises(ValidationError) as exc_info:
            AudioSpeechRequest(
                model="kokoro", input="Hi", voice="af_heart", format="jpeg"
            )
        # The failing field name MUST be ``response_format`` (the
        # canonical field) so the wire envelope teaches the caller the
        # spec-correct name, not the legacy alias.
        errors = exc_info.value.errors()
        assert any("response_format" in str(e.get("loc", ())) for e in errors), errors

    @pytest.mark.parametrize(
        "bad_legacy",
        [
            123,  # int
            True,  # bool
            [],  # list
            {},  # dict
        ],
    )
    def test_nonstring_format_alias_400s(self, bad_legacy):
        """Codex r1 BLOCKING #1: non-string legacy values
        (``{"format":123}``) MUST 400 on ``response_format``, NOT
        silently fall through to ``"wav"``. The before-validator now
        folds EVERY non-None legacy value into ``response_format`` so
        Pydantic's type-check on the str field surfaces the same 400
        envelope a wrong-typed ``response_format`` would."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import AudioSpeechRequest

        with pytest.raises(ValidationError) as exc_info:
            AudioSpeechRequest(
                model="kokoro", input="Hi", voice="af_heart", format=bad_legacy
            )
        errors = exc_info.value.errors()
        # Must surface as a ``response_format`` failure so the caller
        # learns the spec-correct field name.
        assert any("response_format" in str(e.get("loc", ())) for e in errors), errors

    def test_none_format_alias_keeps_pydantic_default(self):
        """``{"format": null}`` is the JSON shape an SDK might emit when
        it explicitly clears the field; treat it as unset so the
        Pydantic default still wins."""
        from vllm_mlx.api.models import AudioSpeechRequest

        r = AudioSpeechRequest(
            model="kokoro", input="Hi", voice="af_heart", format=None
        )
        assert r.response_format == "wav"


# ===========================================================================
# R11-B-F4 — audio aliases advertise audio capability + modality
# ===========================================================================


# (alias, hf_id, expected_capability). Covers every TTS + STT family
# in aliases.json so an addition to the registry is a single-row diff
# here too. Same matrix as ``test_audio_r11_b_bundle`` but exercised
# via the helper directly to skip the audio-route import.
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


class TestAudioCapabilityShortCircuit:
    """Pin the ``_build_model_info`` short-circuit for audio aliases at
    the helper level so non-MLX CI catches a regression that would
    otherwise only surface from the full mounted route."""

    @pytest.mark.parametrize(
        "alias,hf_id,expected_cap", _AUDIO_ALIASES_FOR_CAP_CHECK
    )
    def test_audio_alias_has_audio_capability(self, alias, hf_id, expected_cap):
        """Both the short alias and HF id MUST return
        ``modality="audio"`` + the expected ``audio.<kind>`` capability.
        Reverse-HF-id lookup in ``resolve_audio_alias`` powers the
        HF-id half of this matrix."""
        from vllm_mlx.routes.models import _build_model_info

        for model_id in (alias, hf_id):
            info = _build_model_info(model_id)
            assert info.modality == "audio", (
                f"R11-B-F4 regression: {model_id!r} reports modality="
                f"{info.modality!r}, expected 'audio'."
            )
            assert info.capabilities == [expected_cap], (
                f"R11-B-F4 regression: {model_id!r} reports capabilities="
                f"{info.capabilities!r}, expected [{expected_cap!r}]."
            )
            # ``text`` MUST NOT leak — the audio entry is audio-only on
            # the wire. (The capabilities list-equality above also pins
            # this, but the explicit check makes the failure clearer.)
            assert "text" not in info.capabilities, (
                f"R11-B-F4 regression: {model_id!r} leaked 'text' tag."
            )

    def test_text_model_still_advertises_text(self):
        """Regression guard: a text-only model id MUST keep
        ``capabilities=["text"]`` — the audio short-circuit must only
        fire for registered audio aliases."""
        from vllm_mlx.routes.models import _build_model_info

        info = _build_model_info("mlx-community/Qwen3.5-4B-MLX-4bit")
        assert "text" in info.capabilities, info.capabilities
        # The TEXT model MUST NOT carry an audio capability tag — pin
        # so a future broadening of the audio short-circuit can't
        # accidentally paint audio onto chat models.
        for cap in info.capabilities:
            assert not cap.startswith("audio."), (
                f"text model leaked audio capability {cap!r}: {info.capabilities}"
            )


# ===========================================================================
# R11-B-F3 — ``voice="default"`` resolver helper (no engine needed)
# ===========================================================================


class TestResolveDefaultVoiceLiteral:
    """The ``_resolve_default_voice_literal`` helper is pure registry
    lookup — it never touches mlx — so we can pin it on every CI
    runner. The route-integration test in the main bundle covers the
    end-to-end wiring; this file pins the helper contract."""

    def test_default_literal_resolves_for_kokoro_short_alias(self):
        from vllm_mlx.routes.audio import _resolve_default_voice_literal

        # kokoro's registry default_voice is af_heart.
        assert _resolve_default_voice_literal("kokoro", "default") == "af_heart"

    def test_default_literal_resolves_for_kokoro_hf_id(self):
        from vllm_mlx.routes.audio import _resolve_default_voice_literal

        # Reverse HF-id lookup in resolve_audio_alias makes this work.
        assert (
            _resolve_default_voice_literal(
                "mlx-community/Kokoro-82M-bf16", "default"
            )
            == "af_heart"
        )

    def test_non_default_voice_passes_through(self):
        """The helper only fires for the literal ``"default"`` —
        everything else passes through unchanged. Pin so a future
        edit can't accidentally broaden the substitution."""
        from vllm_mlx.routes.audio import _resolve_default_voice_literal

        assert _resolve_default_voice_literal("kokoro", "af_heart") == "af_heart"
        assert _resolve_default_voice_literal("kokoro", "alloy") == "alloy"
        assert _resolve_default_voice_literal("kokoro", "") == ""

    def test_unknown_model_passes_default_through(self):
        """A model id the registry doesn't know about → the literal
        ``"default"`` passes through unchanged so the route's family
        detector (``_allowed_voices_for``) still owns the decision
        (it returns ``["default"]`` for unknown families)."""
        from vllm_mlx.routes.audio import _resolve_default_voice_literal

        assert (
            _resolve_default_voice_literal("mlx-community/Some-Future-TTS", "default")
            == "default"
        )

    def test_chatterbox_default_voice_resolves(self):
        """chatterbox's registry default_voice is the literal ``"default"``
        (the engine's catch-all). The helper still returns that value
        so the downstream allowlist accepts it without rejection."""
        from vllm_mlx.routes.audio import _resolve_default_voice_literal

        assert _resolve_default_voice_literal("chatterbox", "default") == "default"
