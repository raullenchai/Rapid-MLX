"""Metadata-based capacity accounting for auxiliary audio roles (#2305).

Charges come from the model-size manifest or a complete cached snapshot. An
unknown footprint remains unknown so an enforced ceiling can fail closed
before loading weights; model names are never used as capacity estimates.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

_MIB = 1024**2

CapacitySource = Literal["manifest", "local_cache", "unknown"]

# Engine activations only; request buffers are charged per request below.
AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES: int = 512 * _MIB

# Compressed upload size does not bound decoded duration.
MAX_TRANSCRIPTION_SECONDS: float = 2 * 60 * 60.0

MAX_TTS_REFERENCE_SECONDS: float = 30.0

STT_SAMPLE_RATE: int = 16_000

MAX_SPEECH_INPUT_CHARACTERS: int = 20_000

# Conservative duration estimate; real narration is typically much faster.
_SPEECH_CHARACTERS_PER_SECOND: float = 2.0

# Highest native output rate in the TTS registry.
_TTS_NATIVE_SAMPLE_RATE: int = 44_100

# Chunks, concatenation, int16 conversion, and encoded output overlap.
_TTS_PEAK_MULTIPLIER: float = 3.5

# Shared by the reservation and generation ceiling so they cannot drift.
_TTS_GENERATION_HEADROOM: float = 1.25


@dataclass(frozen=True)
class AudioRoleCapacity:
    """Admission charge for one audio role and its provenance."""

    reserved_bytes: int
    weight_bytes: int | None
    capacity_source: CapacitySource
    hf_id: str | None

    @property
    def is_known(self) -> bool:
        return self.capacity_source != "unknown"


def _canonical_hf_id(model_id: str) -> str | None:
    """Map an audio alias or raw ``org/repo`` id to its load target."""

    from ..audio.registry import resolve_audio_alias

    entry = resolve_audio_alias(model_id)
    if entry is not None:
        return entry.hf_id
    return model_id if "/" in model_id else None


def _snapshot_bytes(hf_id: str) -> int | None:
    """Sum unique files in the complete snapshot the loader would resolve."""

    try:
        if not _cache_is_complete(hf_id):
            return None

        snapshot_dir = _cached_snapshot_path(hf_id)
        if snapshot_dir is None:
            return None

        total = 0
        seen: set[tuple[int, int]] = set()
        for root, _dirs, files in os.walk(snapshot_dir):
            for name in files:
                try:
                    stat = os.stat(os.path.join(root, name))
                except OSError:
                    continue
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += stat.st_size
        return total or None
    except Exception:
        logger.debug("Failed to measure cached snapshot for %r", hf_id, exc_info=True)
        return None


def _cache_is_complete(hf_id: str) -> bool:
    """Check completeness using the existing cache layout probes."""

    from .._download_gate import (
        _snapshot_is_complete_split_model,
        _snapshot_is_complete_whisper_model,
        is_repo_cached,
    )

    return (
        is_repo_cached(hf_id)
        or _snapshot_is_complete_whisper_model(hf_id)
        or _snapshot_is_complete_split_model(hf_id)
    )


def _resolved_revision(repo_root: str) -> str | None:
    """Return the revision selected by ``refs/main``."""

    main_ref = os.path.join(repo_root, "refs", "main")
    try:
        if not os.path.isfile(main_ref):
            return None
        with open(main_ref) as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _cached_snapshot_path(hf_id: str) -> str | None:
    """Return the resolved cached snapshot directory, if present."""

    from huggingface_hub.constants import HF_HUB_CACHE

    repo_root = os.path.join(HF_HUB_CACHE, f"models--{hf_id.replace('/', '--')}")
    revision = _resolved_revision(repo_root)
    if revision is None:
        return None
    path = os.path.join(repo_root, "snapshots", revision)
    return path if os.path.isdir(path) else None


def resolve_audio_role_capacity(model_id: str) -> AudioRoleCapacity:
    """Return the admission charge for loading ``model_id`` into an audio role."""

    hf_id = _canonical_hf_id(model_id)
    if hf_id is None:
        return AudioRoleCapacity(
            reserved_bytes=0,
            weight_bytes=None,
            capacity_source="unknown",
            hf_id=None,
        )

    from ..model_sizes import size_bytes

    weight_bytes = size_bytes(hf_id)
    source: CapacitySource = "manifest"
    if weight_bytes is None:
        weight_bytes = _snapshot_bytes(hf_id)
        source = "local_cache"
    if weight_bytes is None:
        return AudioRoleCapacity(
            reserved_bytes=0,
            weight_bytes=None,
            capacity_source="unknown",
            hf_id=hf_id,
        )
    return AudioRoleCapacity(
        reserved_bytes=weight_bytes + AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES,
        weight_bytes=weight_bytes,
        capacity_source=source,
        hf_id=hf_id,
    )


def transcription_buffer_bytes(
    duration_seconds: float,
    *,
    source_rate: int,
    source_channels: int,
) -> int:
    """Peak decode bytes, sized from the measured source layout."""

    duration = max(0.0, duration_seconds)
    rate = max(1, source_rate)
    channels = max(1, source_channels)

    # Decoded source: float64 is libsndfile's default read dtype.
    source = duration * rate * channels * 8
    # Resampled mono float32 handed to the model, plus mel/window copies.
    resampled = duration * STT_SAMPLE_RATE * 4 * 3
    return int(source * 2 + resampled)


def _speech_seconds(characters: int, speed: float) -> float:
    """Shared duration estimate for reservation and generation limits."""

    seconds = max(0, characters) / _SPEECH_CHARACTERS_PER_SECOND
    seconds /= max(0.25, float(speed))
    return seconds * _TTS_GENERATION_HEADROOM


def speech_buffer_bytes(
    characters: int,
    *,
    speed: float = 1.0,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> int:
    """Peak synthesis bytes, including optional output conversion."""

    seconds = _speech_seconds(characters, speed)

    native = seconds * _TTS_NATIVE_SAMPLE_RATE * 4 * _TTS_PEAK_MULTIPLIER
    if sample_rate is None and channels is None:
        return int(native)

    out_rate = sample_rate if sample_rate else _TTS_NATIVE_SAMPLE_RATE
    out_channels = channels if channels else 1
    # float32 converted array + its int16 copy + the encoded byte buffer.
    converted = seconds * out_rate * out_channels * 4 * 2
    return int(native + converted)


def tts_reference_buffer_bytes(
    duration_seconds: float,
    *,
    source_rate: int,
    source_channels: int,
    encoded_bytes: int = 0,
    compressed_bytes: int = 0,
) -> int:
    """Peak bytes held while decoding and resampling a clone reference."""

    duration = max(0.0, duration_seconds)
    rate = max(1, source_rate)
    channels = max(1, source_channels)
    source = duration * rate * channels * 8
    resampled = duration * _TTS_NATIVE_SAMPLE_RATE * 4 * 3
    return int(
        source * 2
        + resampled
        + max(0, int(encoded_bytes))
        + max(0, int(compressed_bytes))
    )


def max_output_seconds_for(text: str, *, speed: float = 1.0) -> float:
    """Hard duration ceiling paired with :func:`speech_buffer_bytes`."""

    return _speech_seconds(len(text), speed)
