# SPDX-License-Identifier: Apache-2.0
"""
Speech-to-Text (STT) engine using mlx-audio.

Supports:
- Whisper (multilingual, 99+ languages)
- Parakeet (English-focused, fast)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default models
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


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
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._is_parakeet = "parakeet" in model_name.lower()

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
            if not self._is_parakeet:
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
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language code (e.g., "en", "es"). Auto-detected if None.
            task: "transcribe" or "translate" (translate to English)

        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._loaded:
            self.load()

        audio_path = str(audio_path)

        try:
            # Use the model's generate method directly
            kwargs = {"verbose": False}
            if language and not self._is_parakeet:
                kwargs["language"] = language
            if task and not self._is_parakeet:
                kwargs["task"] = task

            result = self.model.generate(audio_path, **kwargs)

            # Extract text and metadata from result
            text = getattr(result, "text", str(result)) if result else ""
            segments = getattr(result, "segments", None)
            detected_lang = getattr(result, "language", None)

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
