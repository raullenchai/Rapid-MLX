# SPDX-License-Identifier: Apache-2.0
"""
Audio processing using mlx-audio.

Supports:
- SAM-Audio: Text-guided source separation (isolate voice from background)
- MossFormer2: Speech enhancement (noise removal)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default models
DEFAULT_SAM_MODEL = "mlx-community/sam-audio-large-fp16"


@dataclass
class SeparationResult:
    """Result from audio separation."""

    target: np.ndarray  # Isolated audio (e.g., voice)
    residual: np.ndarray  # Background audio
    sample_rate: int
    peak_memory: float  # GB


class AudioProcessor:
    """
    Audio processor for voice separation and enhancement.

    Uses SAM-Audio for text-guided source separation:
    - Isolate speech from music/noise
    - Extract specific sounds by description

    Usage:
        processor = AudioProcessor()
        processor.load()
        result = processor.separate("meeting.mp3", description="speech")
        processor.save(result.target, "voice_only.wav")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SAM_MODEL,
    ):
        """
        Initialize audio processor.

        Args:
            model_name: HuggingFace model name. Supported:
                - mlx-community/sam-audio-large-fp16 (best quality)
                - mlx-community/sam-audio-large
                - mlx-community/sam-audio-small-fp16 (faster)
                - mlx-community/sam-audio-small
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self.sample_rate = 44100  # SAM-Audio default

    def load(self) -> None:
        """Load the SAM-Audio model."""
        if self._loaded:
            return

        try:
            from mlx_audio.sts import SAMAudio, SAMAudioProcessor

            self.model = SAMAudio.from_pretrained(self.model_name)
            self.processor = SAMAudioProcessor.from_pretrained(self.model_name)

            if hasattr(self.model, "sample_rate"):
                self.sample_rate = self.model.sample_rate

            self._loaded = True
            logger.info(f"Audio processor loaded: {self.model_name}")
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required. Install with: pip install mlx-audio"
            ) from e

    def separate(
        self,
        audio_path: str | Path,
        description: str = "speech",
        chunk_seconds: float | None = None,
    ) -> SeparationResult:
        """
        Separate audio based on text description.

        Args:
            audio_path: Path to audio file
            description: What to isolate (e.g., "speech", "music", "a person speaking")
            chunk_seconds: Process in chunks for long audio (memory efficient)

        Returns:
            SeparationResult with target (isolated) and residual (background) audio
        """
        if not self._loaded:
            self.load()

        audio_path = str(audio_path)

        try:
            # Process input
            batch = self.processor(
                descriptions=[description],
                audios=[audio_path],
            )

            # Separate
            if chunk_seconds:
                # Memory-efficient for long audio
                result = self.model.separate_long(
                    audios=batch.audios,
                    descriptions=batch.descriptions,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=chunk_seconds / 3,
                    anchor_ids=getattr(batch, "anchor_ids", None),
                    anchor_alignment=getattr(batch, "anchor_alignment", None),
                )
            else:
                result = self.model.separate(
                    audios=batch.audios,
                    descriptions=batch.descriptions,
                    sizes=getattr(batch, "sizes", None),
                    anchor_ids=getattr(batch, "anchor_ids", None),
                    anchor_alignment=getattr(batch, "anchor_alignment", None),
                )

            # Convert to numpy
            target = self._to_numpy(result.target[0])
            residual = self._to_numpy(result.residual[0])

            return SeparationResult(
                target=target,
                residual=residual,
                sample_rate=self.sample_rate,
                peak_memory=getattr(result, "peak_memory", 0.0),
            )
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            raise

    def _to_numpy(self, audio) -> np.ndarray:
        """Convert audio to numpy array."""
        if hasattr(audio, "tolist"):
            return np.array(audio.tolist(), dtype=np.float32)
        return np.array(audio, dtype=np.float32)

    def save(
        self,
        audio: np.ndarray,
        path: str | Path,
        sample_rate: int | None = None,
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: Audio data as numpy array
            path: Output file path
            sample_rate: Sample rate (uses model default if None)
        """
        sr = sample_rate or self.sample_rate

        try:
            from mlx_audio.sts import save_audio

            save_audio(audio, str(path), sample_rate=sr)
        except ImportError:
            import scipy.io.wavfile as wav

            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(str(path), sr, audio_int16)

        logger.info(f"Audio saved to {path}")

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self.processor = None
        self._loaded = False
        logger.info("Audio processor unloaded")


def separate_voice(
    audio_path: str | Path,
    model_name: str = DEFAULT_SAM_MODEL,
    description: str = "speech",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to separate voice from audio.

    Args:
        audio_path: Path to audio file
        model_name: Model to use
        description: What to isolate

    Returns:
        Tuple of (voice_audio, background_audio) as numpy arrays
    """
    processor = AudioProcessor(model_name)
    processor.load()
    result = processor.separate(audio_path, description=description)
    return result.target, result.residual
