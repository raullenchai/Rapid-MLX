# SPDX-License-Identifier: Apache-2.0
"""
Speech-to-Text (STT) engine using mlx-audio.

Supports:
- Whisper (multilingual, 99+ languages)
- Parakeet (English-focused, fast)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default models
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"

# ---------------------------------------------------------------------------
# F-K-WHISPER-961: VAD pre-trim guard (see #961)
# ---------------------------------------------------------------------------
# kootenayalex reproduced ``STTEngine.transcribe`` on 12 s of digital
# silence returning ``"Thank you."`` — a canonical Whisper silence
# hallucination. The documented anti-hallucination guards inside
# ``mlx_audio.stt.models.whisper.Model.generate`` (``no_speech_threshold``,
# ``logprob_threshold``, ``hallucination_silence_threshold``) are inert
# in practice because on pure silence the model returns
# ``no_speech_prob ≈ 1e-11`` (or NaN on some chunked paths) AND
# ``avg_logprob ≈ -0.26`` — the AND-gate skip condition
# ``no_speech_prob > 0.6 AND avg_logprob < -1.0`` never fires. The
# upstream fix would require modifying ``mlx_audio`` (third-party, not
# vendored here); monkey-patching it is fragile across upstream bumps.
#
# The systematic remedy applied here — "if the audio has no speech,
# don't invoke Whisper on it" — is a strictly-stronger invariant than
# post-hoc hallucination detection. We use the bundled
# ``mlx_audio.vad.silero_vad`` model (already installable via the
# ``[audio]`` extra, no new dependency) as a cheap pre-filter:
#
#   * VAD reports no speech → return an empty ``TranscriptionResult``
#     without invoking Whisper. Cost: ~200 ms VAD vs. multi-second
#     Whisper decode → strict latency win on silent clips too.
#   * VAD reports speech → trim the input to
#     ``[first_speech.start - pad, last_speech.end + pad]`` and pass
#     the trimmed waveform to Whisper. Segment timestamps are then
#     offset by the trim start so callers still see absolute times.
#
# Gating (both must be true for the guard to run):
#   * The engine kwarg ``enable_vad_pretrim`` (default ``True``).
#   * The env override ``RAPID_MLX_STT_VAD_PRETRIM`` is not one of
#     ``{"0", "false", "no", "off"}`` (case-insensitive).
#
# If the VAD model fails to load (missing extras, network error, etc.),
# the guard logs a warning once and every subsequent call transparently
# falls back to the unmodified transcription path — no regression to
# pre-fix behaviour.
_VAD_MODEL_REPO = "mlx-community/silero-vad"
# Silero already applies ``speech_pad_ms=30`` internally; the extra
# 200 ms we add on each side gives Whisper a small ambient-context
# tail so an onset isn't clipped mid-phoneme.
_VAD_TRIM_PAD_SECONDS = 0.2
_VAD_SAMPLE_RATE = 16_000

# Load-once cache. Kept module-level (not instance-level) so multiple
# STTEngine instances share one VAD model — the Silero weights are
# ~1.7 MB but the mx.array upload isn't free either.
_VAD_MODEL_CACHE: Any | None = None
_VAD_LOAD_ATTEMPTED = False


# F-K-WHISPER-500: every mlx-community Whisper repo ships ONLY
# ``weights.npz`` + ``config.json`` — no ``preprocessor_config.json``,
# no tokenizer files. ``mlx_audio.stt.models.whisper.Model.post_load_hook``
# therefore swallows the ``WhisperProcessor.from_pretrained(model_path)``
# failure (warns), sets ``model._processor = None``, and the first
# transcription request 500s with ``ValueError: Processor not found``.
#
# Fix at the rapid-mlx integration layer: after ``load_model`` returns,
# if the model is Whisper and ``_processor`` is None, manually attach a
# ``WhisperProcessor`` loaded from the OpenAI counterpart repo (which
# ships every processor file the tokenizer wrapper needs). The fallback
# is keyed off the mlx-community alias so unknown repos still get an
# attempt at ``openai/whisper-large-v3`` (the v3 tokenizer matches all
# v3 mlx variants because the language vocabulary is identical).
#
# This is the smallest possible fix that doesn't require uploading
# processor files to the mlx-community side or pinning a specific
# mlx_audio version. It's belt-and-braces — if a future mlx-community
# upload ships processor files, the post_load_hook's own loader
# succeeds first and ``_processor`` is already non-None on entry to
# the patch helper below.
_WHISPER_PROCESSOR_SOURCE_MAP: dict[str, str] = {
    # mlx-community model id  →  openai counterpart that ships processor
    "mlx-community/whisper-large-v3-mlx": "openai/whisper-large-v3",
    "mlx-community/whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "mlx-community/whisper-medium-mlx": "openai/whisper-medium",
    "mlx-community/whisper-small-mlx": "openai/whisper-small",
    "mlx-community/whisper-base-mlx": "openai/whisper-base",
    "mlx-community/whisper-tiny-mlx": "openai/whisper-tiny",
}
_DEFAULT_WHISPER_PROCESSOR_FALLBACK = "openai/whisper-large-v3"


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list | None = None


@dataclass
class _VADTrimResult:
    """Outcome of a single VAD pre-trim call — see #961.

    Three shapes:

    * ``skipped=True`` — VAD is unavailable, disabled by env, or the
      audio decode failed. Caller must fall back to the unmodified
      transcription path (Whisper reads the original ``audio_path``).
    * ``skipped=False, has_speech=False`` — VAD ran and reported no
      speech at all. Caller must return an empty
      ``TranscriptionResult`` without invoking Whisper.
    * ``skipped=False, has_speech=True, waveform=..., offset_seconds=...``
      — VAD found speech. Caller must invoke Whisper on ``waveform``
      (mono 16 kHz) and add ``offset_seconds`` to every returned
      segment (and per-word) timestamp so callers still see absolute
      times relative to the original file.
    """

    skipped: bool = False
    has_speech: bool = False
    waveform: Any = None  # mx.array | np.ndarray, mono @ 16 kHz
    offset_seconds: float = 0.0
    sample_rate: int = _VAD_SAMPLE_RATE


def _vad_pretrim_disabled_by_env() -> bool:
    """Return ``True`` if ``RAPID_MLX_STT_VAD_PRETRIM`` opts out.

    Truthy defaults ("", "1", "true", "yes", "on") leave the guard on;
    only the explicit disable strings turn it off. This matches the
    ``env_truthy`` convention used elsewhere in the repo (see
    ``scripts/pr_validate/context.py``) — inverted here because the
    default is on.
    """
    val = os.environ.get("RAPID_MLX_STT_VAD_PRETRIM", "").strip().lower()
    return val in {"0", "false", "no", "off"}


def _get_vad_model() -> Any | None:
    """Return the shared Silero VAD model, loading it on first call.

    Returns ``None`` on any failure (import error, network hiccup,
    weight-file mismatch) so callers fall back to the unmodified
    transcription path without ever raising into ``transcribe()``.
    The failure is logged at WARNING once — subsequent calls short-
    circuit off ``_VAD_LOAD_ATTEMPTED`` and produce no more log spam.
    """
    global _VAD_MODEL_CACHE, _VAD_LOAD_ATTEMPTED
    if _VAD_MODEL_CACHE is not None:
        return _VAD_MODEL_CACHE
    if _VAD_LOAD_ATTEMPTED:
        return None
    _VAD_LOAD_ATTEMPTED = True
    try:
        from mlx_audio.vad import load as vad_load  # noqa: PLC0415
    except ImportError as e:
        logger.warning(
            "VAD pre-trim disabled: mlx_audio.vad not importable (%s). "
            "Install rapid-mlx[audio] to enable the anti-hallucination "
            "guard for pure-silence clips (#961).",
            e,
        )
        return None
    try:
        _VAD_MODEL_CACHE = vad_load(_VAD_MODEL_REPO)
    except Exception as e:  # noqa: BLE001
        # Network failure, weight mismatch, HF gate — log once + bail.
        logger.warning(
            "VAD pre-trim disabled: could not load %r: %s",
            _VAD_MODEL_REPO,
            e,
        )
        return None
    return _VAD_MODEL_CACHE


def _maybe_vad_trim(audio_path: str) -> _VADTrimResult:
    """Run Silero VAD, trim to the speech span, return a ``_VADTrimResult``.

    Never raises: any failure short-circuits to ``skipped=True`` so the
    caller falls back to unmodified transcription. The rationale is that
    the guard is best-effort — a hallucinated ``"Thank you."`` is a bug
    but a *harder* bug for the reporter than a raise-through crash.
    """
    if _vad_pretrim_disabled_by_env():
        return _VADTrimResult(skipped=True)

    vad = _get_vad_model()
    if vad is None:
        return _VADTrimResult(skipped=True)

    try:
        from mlx_audio.stt.utils import load_audio  # noqa: PLC0415
    except ImportError:
        return _VADTrimResult(skipped=True)

    try:
        waveform = load_audio(audio_path)  # mono float32 mx.array @ 16 kHz
    except Exception as e:  # noqa: BLE001
        # Decode failure — do NOT swallow. Return skipped so Whisper's
        # own file open surfaces the same error via the route's decode
        # classifier (see tests/test_stt_corrupted_file.py).
        logger.debug(
            "VAD pre-trim: audio load failed for %r, deferring to Whisper: %s",
            audio_path,
            e,
        )
        return _VADTrimResult(skipped=True)

    if getattr(waveform, "shape", (0,))[-1] == 0:
        # Empty audio → no speech by definition.
        return _VADTrimResult(skipped=False, has_speech=False)

    try:
        speech_ts = vad.get_speech_timestamps(
            waveform,
            sample_rate=_VAD_SAMPLE_RATE,
            return_seconds=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "VAD pre-trim: get_speech_timestamps failed on %r: %s — "
            "falling back to unmodified transcription.",
            audio_path,
            e,
        )
        return _VADTrimResult(skipped=True)

    if not speech_ts:
        return _VADTrimResult(skipped=False, has_speech=False)

    total_seconds = waveform.shape[-1] / _VAD_SAMPLE_RATE
    start_s = max(0.0, float(speech_ts[0]["start"]) - _VAD_TRIM_PAD_SECONDS)
    end_s = min(
        total_seconds,
        float(speech_ts[-1]["end"]) + _VAD_TRIM_PAD_SECONDS,
    )
    if end_s <= start_s:
        # Degenerate span (shouldn't happen with valid VAD output, but
        # be defensive). Treat as no speech to avoid pushing a zero-
        # length array into Whisper.
        return _VADTrimResult(skipped=False, has_speech=False)

    start_sample = int(round(start_s * _VAD_SAMPLE_RATE))
    end_sample = int(round(end_s * _VAD_SAMPLE_RATE))
    trimmed = waveform[start_sample:end_sample]

    return _VADTrimResult(
        skipped=False,
        has_speech=True,
        waveform=trimmed,
        offset_seconds=start_s,
        sample_rate=_VAD_SAMPLE_RATE,
    )


def _shift_segment_time(seg: Any, offset: float) -> None:
    """Add ``offset`` seconds to a segment's ``start``/``end`` (and
    per-word times when word-level timestamps are present).

    Mutates in place. Handles both dict-shaped segments (Whisper) and
    object-shaped segments (Parakeet / future engines). No-op when a
    key/attribute is missing or None.
    """
    if isinstance(seg, dict):
        for key in ("start", "end"):
            v = seg.get(key)
            if v is not None:
                seg[key] = float(v) + offset
        words = seg.get("words")
        if isinstance(words, list):
            for w in words:
                if not isinstance(w, dict):
                    continue
                for k in ("start", "end"):
                    v = w.get(k)
                    if v is not None:
                        w[k] = float(v) + offset
        return

    for k in ("start", "end"):
        if hasattr(seg, k):
            v = getattr(seg, k)
            if v is not None:
                setattr(seg, k, float(v) + offset)


class STTEngine:
    """
    Speech-to-Text engine supporting Whisper and Parakeet models.

    Usage:
        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        engine.load()
        result = engine.transcribe("audio.mp3")
        print(result.text)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        enable_vad_pretrim: bool = True,
    ):
        """
        Initialize STT engine.

        Args:
            model_name: HuggingFace model name. Supported:
                - mlx-community/whisper-large-v3-mlx (multilingual)
                - mlx-community/whisper-large-v3-turbo (fast)
                - mlx-community/whisper-medium-mlx
                - mlx-community/whisper-small-mlx
                - mlx-community/parakeet-tdt-0.6b-v2 (English, fastest)
                - mlx-community/parakeet-tdt-0.6b-v3
            enable_vad_pretrim: If ``True`` (default) and the engine
                is Whisper-shaped, run Silero VAD before Whisper. Pure-
                silence clips return ``text=""`` (guards #961), and
                clips with trailing/leading silence get trimmed to the
                speech span (Whisper hallucinates less on shorter
                inputs). Set to ``False`` to preserve pre-fix behaviour
                — or set the env ``RAPID_MLX_STT_VAD_PRETRIM=0`` for a
                run-wide override. Non-Whisper engines (Parakeet, etc.)
                are unaffected regardless of this flag.
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._is_parakeet = "parakeet" in model_name.lower()
        # Codex r6 BLOCKING: ``_ensure_whisper_processor`` previously
        # ran for every non-Parakeet engine, which would attach a
        # WhisperProcessor to Voxtral/other future STT engines whose
        # model object happens to expose a None-valued ``_processor``
        # attribute. Gate on a positive Whisper id check so the patch
        # only fires for actual Whisper backends — non-Whisper engines
        # surface their own load-time errors unmodified.
        self._is_whisper = "whisper" in model_name.lower()
        # F-K-WHISPER-961: VAD pre-trim guard. See top-of-file block
        # for the full rationale. Only applied to Whisper engines —
        # Parakeet/Canary/etc. have their own silence semantics and
        # are not covered by the reported issue.
        self._enable_vad_pretrim = enable_vad_pretrim

    def load(self) -> None:
        """Load the STT model."""
        if self._loaded:
            return

        try:
            from mlx_audio.stt.utils import load_model

            self.model = load_model(self.model_name)
            # F-K-WHISPER-500: patch up the missing WhisperProcessor
            # mlx-community Whisper repos don't ship. Runs AFTER
            # mlx_audio's own post_load_hook, so if that succeeded
            # (e.g. a future repo upload includes processor files)
            # this is a no-op.
            #
            # Codex r6 BLOCKING: gate POSITIVELY on the Whisper id
            # (``self._is_whisper``) rather than negatively on
            # Parakeet. Any future STT engine (Voxtral, Wav2Vec, etc.)
            # whose model object exposes ``_processor=None`` would
            # otherwise get a Whisper processor stapled on by mistake.
            # Belt-and-braces — even the helper double-checks model
            # type via ``hasattr/_processor`` before doing anything.
            if self._is_whisper:
                self._ensure_whisper_processor()
            self._loaded = True
            logger.info(f"STT model loaded: {self.model_name}")
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required for STT. Install with: pip install mlx-audio"
            ) from e

    def _ensure_whisper_processor(self) -> None:
        """Attach a ``WhisperProcessor`` if mlx_audio didn't.

        F-K-WHISPER-500: ``mlx_audio.stt.models.whisper.Model.post_load_hook``
        attempts ``WhisperProcessor.from_pretrained(model_path)`` and
        swallows the failure (warns, sets ``_processor=None``). Every
        ``mlx-community/whisper-*-mlx`` repo lacks processor files, so
        the first ``transcribe`` call 500s with ``ValueError: Processor
        not found``. Patch the gap by loading the processor from the
        OpenAI counterpart repo whose files are public.

        No-op when:
          * the model already has a non-None ``_processor`` (a future
            mlx-community upload could ship them);
          * the model object doesn't expose a ``_processor`` attribute
            (non-Whisper STT engines load through different code paths
            and either already have a tokenizer or surface their own
            error envelope on missing-processor);
          * ``transformers`` isn't installed (extremely unlikely
            because ``mlx_audio`` itself requires it, but we keep this
            tolerant rather than crash the load — the missing processor
            will surface as the same upstream ``ValueError`` at first
            inference, which the route now converts to a clean 503
            envelope).
        """
        # Non-Whisper engines (Parakeet, Voxtral, etc.) don't use
        # ``_processor`` — they ship their own tokenizer infrastructure.
        if self.model is None:
            return
        if not hasattr(self.model, "_processor"):
            return
        if getattr(self.model, "_processor", None) is not None:
            return

        # Pick the OpenAI counterpart by exact alias match, then fall
        # back to the v3-large processor (vocab is identical for every
        # v3 variant; this is the most permissive fallback).
        processor_source = _WHISPER_PROCESSOR_SOURCE_MAP.get(
            self.model_name, _DEFAULT_WHISPER_PROCESSOR_FALLBACK
        )
        try:
            from transformers import WhisperProcessor
        except ImportError:
            logger.warning(
                "transformers not installed; Whisper processor patch skipped — "
                "transcription will fail with the upstream `Processor not found`."
            )
            return

        try:
            processor = WhisperProcessor.from_pretrained(processor_source)
        except Exception as e:  # noqa: BLE001
            # Network failure, gated repo, unsupported revision — log
            # and bail. The upstream ValueError surfaces at first
            # transcribe; the route wraps it in a clean 5xx envelope.
            logger.warning(
                "WhisperProcessor.from_pretrained(%r) failed: %s — "
                "transcription will still fail until the processor is wired.",
                processor_source,
                e,
            )
            return

        self.model._processor = processor
        logger.info(
            "Attached WhisperProcessor from %r to %r (F-K-WHISPER-500 fix).",
            processor_source,
            self.model_name,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        # F-K-TRANSLATIONS-MISSING: ``task`` is forwarded by both
        # ``/v1/audio/transcriptions`` (``task="transcribe"``) and
        # ``/v1/audio/translations`` (``task="translate"``). For
        # Whisper engines the value flows through to ``model.generate``
        # so the underlying decoder emits English when translating
        # and source-language text when transcribing. Parakeet engines
        # ignore the kwarg (English-only). This kwarg was present
        # pre-bundle; the comment is here so the call sites are
        # discoverable from the function definition.
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language code (e.g., "en", "es"). Auto-detected if None.
            task: "transcribe" or "translate" (translate to English).
                Forwarded to ``model.generate`` for Whisper engines;
                ignored by Parakeet (which is English-only).

        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._loaded:
            self.load()

        audio_path = str(audio_path)

        # F-K-WHISPER-961: VAD pre-trim guard. Runs only for Whisper
        # engines with the guard enabled — see class docstring + the
        # module-level rationale block. On any failure the helper
        # returns ``skipped=True`` and we fall back to the original
        # unmodified path, so we never regress on pre-fix inputs.
        trim: _VADTrimResult | None = None
        if self._is_whisper and self._enable_vad_pretrim:
            trim = _maybe_vad_trim(audio_path)
            if not trim.skipped and not trim.has_speech:
                logger.debug(
                    "VAD pre-trim: no speech detected in %r; returning "
                    "empty TranscriptionResult without invoking Whisper.",
                    audio_path,
                )
                return TranscriptionResult(
                    text="",
                    language=None,
                    duration=0.0,
                    segments=[],
                )

        try:
            # Use the model's generate method directly
            kwargs = {"verbose": False}
            if language and not self._is_parakeet:
                kwargs["language"] = language
            if task and not self._is_parakeet:
                kwargs["task"] = task

            # Choose the input Whisper actually sees: either the
            # VAD-trimmed waveform (mono 16 kHz) or the original file
            # path (fallback / VAD skipped / non-Whisper).
            if trim is not None and not trim.skipped and trim.has_speech:
                audio_input: Any = trim.waveform
            else:
                audio_input = audio_path

            result = self.model.generate(audio_input, **kwargs)

            # Extract text and metadata from result
            text = getattr(result, "text", str(result)) if result else ""
            segments = getattr(result, "segments", None)
            detected_lang = getattr(result, "language", None)

            # Absolute-time contract: if we handed Whisper a trimmed
            # span starting at ``offset_seconds`` into the original
            # audio, every segment.start / segment.end (and per-word
            # timestamp, when present) must be shifted back by that
            # offset so downstream consumers see file-relative times.
            if (
                segments
                and trim is not None
                and not trim.skipped
                and trim.has_speech
                and trim.offset_seconds > 0.0
            ):
                for seg in segments:
                    _shift_segment_time(seg, trim.offset_seconds)

            # Calculate duration from segments if available
            duration = None
            if segments:
                last_seg = segments[-1] if segments else None
                if last_seg and hasattr(last_seg, "end"):
                    duration = last_seg.end

            return TranscriptionResult(
                text=text.strip() if isinstance(text, str) else str(text),
                language=detected_lang,
                duration=duration,
                segments=segments,
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self._loaded = False
        logger.info("STT model unloaded")


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
) -> TranscriptionResult:
    """
    Convenience function to transcribe audio without managing engine.

    Args:
        audio_path: Path to audio file
        model_name: Model to use
        language: Language code (optional)

    Returns:
        TranscriptionResult
    """
    engine = STTEngine(model_name)
    engine.load()
    return engine.transcribe(audio_path, language=language)
