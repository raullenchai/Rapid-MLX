# SPDX-License-Identifier: Apache-2.0
"""
Text-to-Speech (TTS) engine using mlx-audio.

Supports:
- Kokoro (fast, lightweight)
- Chatterbox (multilingual, expressive)
- VibeVoice (realtime, low latency)
- VoxCPM (Chinese/English, high quality)
"""

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default models
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"

# Available voices per model family
KOKORO_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]

CHATTERBOX_VOICES = ["default"]  # Uses reference audio for voice


def _list_snapshot_voices(model_name: str) -> list[str]:
    """Return the safetensors voice files cached for ``model_name``.

    R11-B-F1 (Bo 0.8.12 dogfood): pre-fix the route hard-coded a
    static voice list per family — ``["default"]`` for everything
    except kokoro / chatterbox. VibeVoice's HF repo ships per-language
    voice caches (``en-Grace_woman.safetensors``, ``en-Mike_man.safe
    tensors``, the eight non-English ``Spk0/Spk1`` pairs, ...) and NO
    ``default.safetensors`` — so EVERY ``/v1/audio/speech`` call 500'd:
    ``voice="default"`` -> ``FileNotFoundError`` deep in
    ``mlx_audio.tts.models.vibevoice.Model.load_voice``; a real name
    like ``en-Grace_woman`` -> the route's 400 ``invalid_voice``
    because the static list only contained ``"default"``.

    The fix is structural — enumerate the snapshot's ``voices/`` dir at
    validation time and use THAT list, instead of pinning a per-family
    constant. The helper:

    * Returns ``[]`` when the snapshot isn't cached locally — the
      caller falls back to the static list so the FIRST request (which
      triggers the download via ``load_model``) still proceeds. Once
      the snapshot is on disk every subsequent request validates
      against the true voice set.
    * Accepts both short aliases (``"vibevoice"``) and HF ids
      (``"mlx-community/VibeVoice-Realtime-0.5B-4bit"``) by going
      through ``huggingface_hub.try_to_load_from_cache`` on
      ``config.json`` to resolve the snapshot path. We never trigger a
      download here — this is a synchronous request-path call and
      blocking on a 500 MB pull would convert "validate voice" into
      "wait two minutes".
    * Strips the ``.safetensors`` suffix and sorts the result so the
      400 envelope's ``Available:`` preview is deterministic.

    Applies UNIFORMLY to every TTS family (kokoro, chatterbox,
    voxcpm, dia, vibevoice, ...). Chatterbox/voxcpm/dia ship a single
    ``default.safetensors`` so the enumeration returns ``["default"]``
    — same behaviour as the pre-fix static list. Kokoro's snapshot
    actually ships 50+ voice files (the pre-fix static list named only
    11), so the enumeration is a strict superset there.

    Local-only by design (``local_files_only=True``): we MUST NOT
    issue an HTTP roundtrip from inside ``_allowed_voices_for`` — the
    pre-flight 400 lookup runs before any weight load and a network
    stall would convert "client sent invalid voice" into "server
    appears to hang then times out".
    """
    # Local imports keep ``vllm_mlx.audio.tts`` cheap to import on
    # API-only runners that don't install huggingface_hub (the audio
    # extras pin it, but the boot-guard path probes this module
    # without the extras).
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover — covered by ``[audio]``
        return []

    # Strip a leading registry alias to its HF id so both shapes
    # resolve to the same snapshot. The lazy import avoids a hard
    # circular: registry doesn't import this module.
    try:
        from .registry import resolve_audio_alias
    except ImportError:
        resolve_audio_alias = None  # type: ignore[assignment]

    hf_id = model_name
    if resolve_audio_alias is not None:
        entry = resolve_audio_alias(model_name)
        if entry is not None:
            hf_id = entry.hf_id

    # ``/`` is the HF-id separator — a bare alias that didn't resolve
    # in the registry (e.g. a typo) won't be in the cache either, so
    # short-circuit to ``[]`` and let the caller fall back to its
    # static list. The same guard catches HF passthrough ids that
    # haven't been downloaded yet.
    if "/" not in hf_id:
        return []

    # Pick a small file every snapshot has so ``try_to_load_from
    # _cache`` returns the snapshot's resolved path without triggering
    # a download. ``config.json`` is universal across mlx-audio TTS
    # repos (kokoro / chatterbox / vibevoice / voxcpm / dia all ship
    # one at the snapshot root).
    cached = try_to_load_from_cache(repo_id=hf_id, filename="config.json")
    if not cached or not isinstance(cached, str):
        # Snapshot not on disk yet — caller falls back to static.
        return []

    voices_dir = Path(cached).parent / "voices"
    if not voices_dir.is_dir():
        return []

    # ``.stem`` strips the ``.safetensors`` suffix; the registry
    # advertises voices by their bare name. Sorting makes the
    # ``Available:`` 400 preview deterministic so doc snapshots and
    # operator scripts don't churn.
    return sorted(p.stem for p in voices_dir.glob("*.safetensors"))


class UnsupportedAudioFormatError(Exception):
    """The requested TTS ``response_format`` cannot be encoded here.

    R8-H5 (Bo 0.8.9 dogfood): the legacy ``to_bytes`` ignored
    ``format`` and returned RIFF/WAV bytes for every value, so the
    route then set ``Content-Type: audio/{format}`` on bytes that
    started with ``RIFF…WAVE`` — a structural mislabel that broke
    every non-wav client. The encoder now raises this typed
    exception when the requested format isn't producible (no codec
    in libsndfile, no entry in the encoder table, etc.) so the route
    can translate it to a 400 ``invalid_request_error`` envelope
    listing the formats this build DOES support. The caller then
    retries with a known-good format instead of receiving a 500 or a
    mislabeled body.
    """

    def __init__(
        self,
        requested: str,
        supported: list[str],
        hint: str | None = None,
    ):
        self.requested = requested
        self.supported = supported
        self.hint = hint
        msg = (
            f"response_format={requested!r} is not supported by this "
            f"build. Supported formats: {', '.join(supported)}."
        )
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)


@dataclass
class AudioOutput:
    """Output from TTS generation."""

    audio: np.ndarray
    sample_rate: int
    duration: float


class TTSEngine:
    """
    Text-to-Speech engine supporting multiple model families.

    Usage:
        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine.load()
        audio = engine.generate("Hello world!", voice="af_heart")
        engine.save(audio, "output.wav")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TTS_MODEL,
    ):
        """
        Initialize TTS engine.

        Args:
            model_name: HuggingFace model name. Supported families:
                - Kokoro: mlx-community/Kokoro-82M-bf16, Kokoro-82M-4bit
                - Chatterbox: mlx-community/chatterbox-turbo-fp16
                - VibeVoice: mlx-community/VibeVoice-Realtime-0.5B-4bit
                - VoxCPM: mlx-community/VoxCPM1.5
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._model_family = self._detect_family(model_name)

    def _detect_family(self, model_name: str) -> str:
        """Detect model family from name."""
        name_lower = model_name.lower()
        if "kokoro" in name_lower:
            return "kokoro"
        elif "chatterbox" in name_lower:
            return "chatterbox"
        elif "vibevoice" in name_lower:
            return "vibevoice"
        elif "voxcpm" in name_lower:
            return "voxcpm"
        elif "csm" in name_lower:
            return "csm"
        elif "cosyvoice" in name_lower:
            return "cosyvoice"
        else:
            return "kokoro"  # Default

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        try:
            from mlx_audio.tts.generate import load_model

            self.model = load_model(self.model_name)
            self._loaded = True
            logger.info(
                f"TTS model loaded: {self.model_name} (family: {self._model_family})"
            )
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required for TTS. Install with: pip install mlx-audio"
            ) from e

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> AudioOutput:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (model-specific)
            speed: Speech speed (0.5 to 2.0)
            lang_code: Language code (a=English, e=Spanish, f=French, etc.)

        Returns:
            AudioOutput with audio data and metadata
        """
        if not self._loaded:
            self.load()

        try:
            import mlx.core as mx

            audio_chunks = []
            sample_rate = 24000  # Default for most models

            for result in self.model.generate(
                text=text,
                voice=voice,
                speed=speed,
                lang_code=lang_code,
            ):
                audio_data = result.audio
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate

                # Convert mlx array to numpy
                if isinstance(audio_data, mx.array) or hasattr(audio_data, "tolist"):
                    audio_np = np.array(audio_data.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio_data, dtype=np.float32)

                audio_chunks.append(audio_np)

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            # Concatenate all chunks
            full_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )
            duration = len(full_audio) / sample_rate

            return AudioOutput(
                audio=full_audio,
                sample_rate=sample_rate,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def stream_generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> Iterator[AudioOutput]:
        """
        Stream speech generation chunk by chunk.

        Args:
            text: Text to synthesize
            voice: Voice ID
            speed: Speech speed

        Yields:
            AudioOutput chunks
        """
        if not self._loaded:
            self.load()

        sample_rate = 24000

        for result in self.model.generate(
            text=text,
            voice=voice,
            speed=speed,
        ):
            audio_data = result.audio
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

            if hasattr(audio_data, "tolist"):
                audio_np = np.array(audio_data.tolist(), dtype=np.float32)
            else:
                audio_np = np.array(audio_data, dtype=np.float32)

            yield AudioOutput(
                audio=audio_np,
                sample_rate=sample_rate,
                duration=len(audio_np) / sample_rate,
            )

    def save(
        self,
        audio: AudioOutput,
        path: str | Path,
        format: str = "wav",
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: AudioOutput to save
            path: Output file path
            format: Output format (wav, mp3)
        """
        try:
            from mlx_audio.tts import save_audio

            save_audio(audio.audio, str(path), sample_rate=audio.sample_rate)
            logger.info(f"Audio saved to {path}")
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile as wav

            # Ensure audio is in correct format
            audio_int16 = (audio.audio * 32767).astype(np.int16)
            wav.write(str(path), audio.sample_rate, audio_int16)
            logger.info(f"Audio saved to {path} (scipy fallback)")

    def to_bytes(
        self,
        audio: AudioOutput,
        format: str = "wav",
    ) -> bytes:
        """
        Convert audio to bytes in the requested container format.

        R8-H5 (Bo 0.8.9 dogfood): pre-fix every call returned RIFF/WAV
        bytes regardless of ``format`` — the route then set
        ``Content-Type: audio/{format}`` so a client asking for
        ``response_format="mp3"`` got an ``audio/mp3``-labelled body
        whose magic was ``RIFF…WAVE``. Browsers and ffmpeg both reject
        the mismatch; OpenAI parity was structurally broken on every
        non-wav format. The handler now branches on ``format`` and
        encodes via the appropriate codec:

        * ``wav`` → scipy (always available with the audio extra).
        * ``flac`` / ``ogg`` / ``opus`` → ``soundfile`` (libsndfile
          ≥1.0). Always shipped via ``rapid-mlx[audio]``.
        * ``mp3`` → ``soundfile`` when libsndfile ≥1.1 (the version
          that bundled the LAME-backed MP3 writer). Older builds raise
          ``LibsndfileError`` which the caller surfaces as a 400 with
          an actionable hint (see ``UnsupportedAudioFormatError``).
        * ``aac`` → not supported by libsndfile in any current release;
          raises :class:`UnsupportedAudioFormatError` so the route can
          translate to a 400 listing the formats this build supports.
          We do NOT silently relabel WAV bytes as ``audio/aac``.
        * ``pcm`` → raw little-endian int16 PCM (no container). Mirrors
          OpenAI's ``response_format="pcm"`` contract.

        Raises:
            UnsupportedAudioFormatError: when the requested format
                cannot be produced by the available encoder stack.
                The route catches this and emits a 400 envelope so
                the caller can fall back to a supported format rather
                than receive a mislabeled WAV.
        """
        fmt = (format or "wav").lower()
        audio_int16 = (np.clip(audio.audio, -1.0, 1.0) * 32767).astype(np.int16)

        if fmt == "wav":
            import scipy.io.wavfile as wav

            buffer = io.BytesIO()
            wav.write(buffer, audio.sample_rate, audio_int16)
            return buffer.getvalue()

        if fmt == "pcm":
            # OpenAI ``response_format="pcm"`` is raw 16-bit signed LE
            # PCM at the source sample rate — no header, no container.
            # Pre-fix we wrapped the same bytes in RIFF/WAVE and
            # labeled them ``audio/pcm`` which any decoder following
            # the OpenAI contract would mis-parse as PCM headers.
            return audio_int16.tobytes()

        # soundfile-backed formats. ``flac``/``ogg``/``opus`` are
        # always supported; ``mp3`` depends on the libsndfile version
        # the wheel was built against. Surface a clean error if the
        # encoder isn't available so the route can emit a 400 listing
        # the supported set rather than a 500 stack trace.
        try:
            import soundfile as sf
        except ImportError as e:  # pragma: no cover — covered by extras
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=["wav", "pcm"],
                hint="Install with: pip install 'rapid-mlx[audio]'",
            ) from e

        # Map our OpenAI-style ``response_format`` values onto the
        # ``(format, subtype)`` pair ``soundfile`` expects. ``opus`` is
        # the OGG container with the Opus codec — same wire shape that
        # OpenAI returns. ``mp3`` only encodes when the underlying
        # libsndfile shipped the LAME writer; older wheels (<1.1) raise
        # ``LibsndfileError`` which we translate to the 400 hint.
        soundfile_targets: dict[str, tuple[str, str | None]] = {
            "flac": ("FLAC", None),
            "ogg": ("OGG", "VORBIS"),
            "opus": ("OGG", "OPUS"),
            "mp3": ("MP3", None),
        }
        target = soundfile_targets.get(fmt)
        if target is None:
            # Anything not in the table (``aac``, future formats, typos)
            # gets a structured rejection. The route maps this to a 400
            # envelope listing the formats we DID support so the caller
            # can retry with a known-good value.
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=sorted(["wav", "pcm", *soundfile_targets.keys()]),
            )

        container, subtype = target
        buffer = io.BytesIO()
        try:
            sf.write(
                buffer,
                audio_int16,
                audio.sample_rate,
                format=container,
                subtype=subtype,
            )
        except Exception as e:
            # libsndfile raises a typed ``LibsndfileError`` when the
            # codec isn't compiled in (most often ``mp3`` on macOS
            # wheels built against an older libsndfile). Re-raise as
            # the structured envelope error so the route emits 400 with
            # the supported-set hint instead of a 500 stack trace.
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=sorted(["wav", "pcm", *soundfile_targets.keys()]),
                hint=(
                    f"Encoder for {fmt!r} is not available in this "
                    f"libsndfile build ({e}). Upgrade libsndfile to "
                    "the latest release, or request a supported format."
                ),
            ) from e
        return buffer.getvalue()

    def get_voices(self) -> list:
        """Get available voices for current model."""
        if self._model_family == "kokoro":
            return KOKORO_VOICES
        elif self._model_family == "chatterbox":
            return CHATTERBOX_VOICES
        else:
            return ["default"]

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self._loaded = False
        logger.info("TTS model unloaded")


def generate_speech(
    text: str,
    model_name: str = DEFAULT_TTS_MODEL,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> AudioOutput:
    """
    Convenience function to generate speech without managing engine.

    Args:
        text: Text to synthesize
        model_name: Model to use
        voice: Voice ID
        speed: Speech speed

    Returns:
        AudioOutput
    """
    engine = TTSEngine(model_name)
    engine.load()
    return engine.generate(text, voice=voice, speed=speed)
