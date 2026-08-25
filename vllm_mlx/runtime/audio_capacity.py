"""Metadata-driven capacity accounting for auxiliary audio roles (#2305).

The resident-model control plane charges chat/vision/image engines against a
process-wide ceiling before their weights load. Speech input and speech output
historically bypassed that budget entirely: ``vllm_mlx.routes.audio`` cached its
engines in module globals, so a dictation request could add multiple GiB to a
process that was already at its ceiling with no admission decision at all.

Issue #2305 requires those roles to participate in the same policy, with one
hard constraint on *how* the charge is derived:

    "Use catalog/cache metadata and allocator/process measurements; do not
    infer capacity from model names or hashes."

That rules out :func:`vllm_mlx.runtime.resident_models.estimate_model_bytes`
for this path — it parses a parameter count out of the model *name*
(``_PARAM_RE``) and guesses bytes-per-param from a ``-4bit``/``-8bit`` token.
That heuristic is acceptable as a last-ditch fallback for arbitrary text
checkpoints, but it is exactly the inference this issue forbids, and it is
actively wrong for audio: ``whisper-large-v3`` carries no ``b`` suffix at all
and would size to the 4 GiB catch-all default, while ``kokoro`` (82M params,
~340 MB on disk) would do the same.

Resolution order, all three tiers metadata or measurement:

1. **Catalog manifest.** :mod:`vllm_mlx.audio.registry` maps the alias to its
   canonical ``hf_id``; :mod:`vllm_mlx.model_sizes` maps that repo to the
   download footprint recorded by ``scripts/gen_model_sizes.py``. Every audio
   repo currently in ``aliases.json`` has an entry, so this tier answers for
   every alias the product ships.
2. **Local cache measurement.** For a raw HuggingFace id a user typed (or a
   repo added to the registry before the size manifest was regenerated), sum
   the actual bytes of the snapshot that ``snapshot_download`` would resolve.
   This is a filesystem measurement, not an estimate.
3. **Unknown.** No trustworthy number exists. The role is REJECTED with a typed
   conflict rather than admitted with a zero charge — see below.

Tier 3 fails closed. An earlier revision charged nothing and relied on the
post-load footprint measurement to correct the books, but that inverts #2305's
central requirement: admission has to happen *before* weight loading, and a
zero charge makes ``_admit_role_locked`` skip the ceiling check entirely, so an
arbitrary ``org/repo`` typed into the ``model`` field could load several GiB
into a process already at its limit. The post-load measurement only fixes the
accounting for the *next* decision; the unsafe load has already happened.

Rejecting is also the better product behaviour here, because tier 3 is almost
always "these weights are not on disk yet". Loading them would trigger a
multi-GB download inside a request anyway, so the conflict tells the caller to
pull the model first — after which tier 2 answers and the load is budgeted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

_MIB = 1024**2

CapacitySource = Literal["manifest", "local_cache", "unknown"]

#: Working-set allowance added on top of the weight footprint.
#:
#: This covers ENGINE ACTIVATIONS only — encoder/decoder intermediates for the
#: largest checkpoints in the registry. Request payloads are charged separately
#: and per request; see :func:`transcription_buffer_bytes` and
#: :func:`speech_buffer_bytes`.
#:
#: An earlier revision claimed 512 MiB was an arithmetic bound on the decoded
#: audio buffer, reasoning from ``routes.audio.MAX_AUDIO_UPLOAD_SIZE`` (25 MB).
#: That reasoning was wrong twice over: the upload cap bounds COMPRESSED bytes
#: (25 MB of 6 kbps Opus is ~9.7 hours, ~2.1 GiB of float32 at 16 kHz), and a
#: fixed per-role constant cannot cover a per-request allocation at all. The
#: role is admitted once at load time; buffers are allocated per request, vary
#: by orders of magnitude, and several can be live at once.
AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES: int = 512 * _MIB

#: Longest audio a single transcription/alignment request may decode.
#:
#: The upload cap cannot express this: it limits compressed bytes, so a
#: low-bitrate stream slips hours of audio under it. This is an absolute
#: ceiling on request size; whether a given request actually fits is a separate
#: question answered against the live budget by :func:`transcription_buffer_bytes`.
MAX_TRANSCRIPTION_SECONDS: float = 2 * 60 * 60.0

#: Sample rate the STT lane resamples input to before inference.
STT_SAMPLE_RATE: int = 16_000

#: Longest text a single speech-synthesis request may vocalize.
#:
#: A character count alone does NOT bound memory — see
#: :func:`speech_buffer_bytes` for what actually does. This limit exists so the
#: request is rejected on a cheap check before any sizing work, and so the
#: error names the field the caller controls.
MAX_SPEECH_INPUT_CHARACTERS: int = 20_000

#: Speech produced per input character, at ``speed=1.0``.
#:
#: Deliberately conservative (slow speech = more samples = more memory). Real
#: narration runs 12-16 characters/second; 2.0 chars/second is well below that,
#: so the derived reservation over-estimates rather than under-estimates.
_SPEECH_CHARACTERS_PER_SECOND: float = 2.0

#: Highest NATIVE output sample rate across the TTS registry.
#:
#: Kokoro and Qwen3 emit 24 kHz, but Dia and VoxCPM emit 44.1 kHz. Assuming the
#: maximum keeps the estimate an upper bound for every registered engine rather
#: than a per-model guess that is wrong for half of them.
_TTS_NATIVE_SAMPLE_RATE: int = 44_100

#: Source layout assumed when a request's container metadata is unreadable.
#: 48 kHz stereo is the highest layout consumer recorders produce in practice,
#: so sizing against it bounds the decode rather than under-charging it.
_ASSUMED_SOURCE_RATE: int = 48_000

#: Peak-to-result multiplier for the synthesis pipeline.
#:
#: The generated waveform is not the peak. ``TTSEngine.generate`` retains every
#: chunk in ``audio_chunks`` and then allocates the concatenated result
#: (2x live), and ``to_bytes`` builds an int16 copy plus the encoded byte
#: buffer (another 1x of float32-equivalent between them). 3.5x covers the
#: overlap with margin.
_TTS_PEAK_MULTIPLIER: float = 3.5


@dataclass(frozen=True)
class AudioRoleCapacity:
    """Admission charge for one audio role, with its provenance."""

    #: Total bytes to reserve: weights plus
    #: :data:`AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES`. Zero when the source is
    #: ``"unknown"``, in which case the role must be REJECTED rather than
    #: admitted — a zero charge would bypass the ceiling check entirely.
    reserved_bytes: int
    #: Weight footprint alone, before the runtime allowance. ``None`` when
    #: unknown. Reported in telemetry so an operator can tell a small model
    #: apart from an unmeasured one.
    weight_bytes: int | None
    capacity_source: CapacitySource
    #: The HuggingFace repo the charge was resolved against, for diagnostics.
    hf_id: str | None

    @property
    def is_known(self) -> bool:
        return self.capacity_source != "unknown"


def _canonical_hf_id(model_id: str) -> str | None:
    """Map an audio alias or raw repo id to the repo that will be loaded.

    Registry-first so short aliases (``kokoro``, ``whisper-large-v3``) reach
    their canonical ``hf_id``. A bare ``org/name`` that the registry does not
    know is returned as-is — it is still a usable cache key even though no
    manifest entry exists. Anything else (a bare word that is not a registered
    alias) has no repo to measure.
    """

    from ..audio.registry import resolve_audio_alias

    entry = resolve_audio_alias(model_id)
    if entry is not None:
        return entry.hf_id
    return model_id if "/" in model_id else None


def _snapshot_bytes(hf_id: str) -> int | None:
    """Sum the on-disk bytes of the COMPLETE cached snapshot for ``hf_id``.

    Only the revision the loader will actually open counts, and only when it is
    verified complete. Both halves matter for admission:

    * **Revision.** ``snapshot_download`` resolves through ``refs/main``, so an
      unrelated stale snapshot would describe weights that are not the ones
      about to be read.
    * **Completeness.** A partial cache is the dangerous case. An interrupted
      pull can leave ``config.json`` and one shard of five — a few hundred KiB
      that this function would happily report as the model's footprint, after
      which the loader downloads the missing multi-GiB weights against a
      reservation sized for almost nothing. Delegating to the download gate's
      audio-aware completeness probes keeps one definition of "cached" in the
      repo rather than inventing a second, weaker one here.

    Returns ``None`` when nothing complete is cached, which the caller turns
    into a rejection rather than an unbudgeted load.

    HF snapshots are trees of symlinks into a sibling ``blobs/`` store, and one
    blob can back several snapshot entries. ``os.stat`` follows the symlink so
    the blob's true size is counted, and ``(st_dev, st_ino)`` de-duplicates
    shared blobs.
    """

    try:
        if not _cache_is_complete(hf_id):
            return None

        from huggingface_hub.constants import HF_HUB_CACHE

        repo_root = os.path.join(HF_HUB_CACHE, f"models--{hf_id.replace('/', '--')}")
        sha = _resolved_revision(repo_root)
        if sha is None:
            return None
        snapshot_dir = os.path.join(repo_root, "snapshots", sha)
        if not os.path.isdir(snapshot_dir):
            return None

        total = 0
        seen: set[tuple[int, int]] = set()
        for root, _dirs, files in os.walk(snapshot_dir):
            for name in files:
                try:
                    stat = os.stat(os.path.join(root, name))
                except OSError:
                    # A dangling symlink means the blob was reaped; it
                    # contributes no resident bytes.
                    continue
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += stat.st_size
        return total or None
    except Exception:
        # Capacity resolution is advisory input to an admission decision; a
        # broken cache directory must degrade to "unknown", never take the
        # audio lane down.
        logger.debug("Failed to measure cached snapshot for %r", hf_id, exc_info=True)
        return None


def _cache_is_complete(hf_id: str) -> bool:
    """True when ``hf_id`` has a fully-downloaded snapshot the loader can open.

    Dispatches on the LAYOUT ON DISK, not on registry membership. mlx-audio
    checkpoints ship ``config.json`` + ``weights.npz`` and carry no
    ``model*.safetensors``, so the generic text probe rejects them — and the
    routes accept any ``<org>/<repo>``, so gating the NPZ probe on "resolves to
    a registered Whisper alias" made a complete custom NPZ repo resolve to
    ``unknown`` and 507 under a ceiling.

    Applying the NPZ probe by layout is safe here in a way it is not in the
    download gate: this module is only ever asked about models that are about
    to be loaded into an AUDIO role, so an NPZ checkpoint is exactly what we
    expect. The gate keeps its family restriction because it also answers for
    text repositories, where a stray NPZ must not look runnable.
    """

    from .._download_gate import (
        _snapshot_is_complete_split_model,
        _snapshot_is_complete_whisper_model,
        is_repo_cached,
    )

    # ``_snapshot_is_complete_whisper_model`` is a pure layout check
    # (config.json + weights.npz, resolved revision, symlink-escape guarded)
    # despite its name; it never consults the registry.
    return (
        is_repo_cached(hf_id)
        or _snapshot_is_complete_whisper_model(hf_id)
        or _snapshot_is_complete_split_model(hf_id)
    )


def _resolved_revision(repo_root: str) -> str | None:
    """Return the sha ``snapshot_download`` would resolve for this repo.

    Mirrors ``_download_gate._resolved_snapshot_sha``: only ``refs/main``, so a
    legacy ``refs/master`` or a bare snapshot directory cannot shadow what the
    loader will really open.
    """

    main_ref = os.path.join(repo_root, "refs", "main")
    try:
        if not os.path.isfile(main_ref):
            return None
        with open(main_ref) as handle:
            return handle.read().strip() or None
    except OSError:
        return None


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
    source_rate: int = 0,
    source_channels: int = 0,
) -> int:
    """Peak bytes one transcription/alignment request needs.

    Sized from the SOURCE layout, not the resampled result. mlx-audio decodes
    the file at its own rate and channel count — and libsndfile's default dtype
    is float64 — before resampling to :data:`STT_SAMPLE_RATE` mono. A two-hour
    48 kHz stereo file is ~5.2 GiB as source float64 on its own, four times
    what the final 16 kHz mono float32 waveform would suggest.

    Peak holds the source array, the resampled copy, and the decoder's windowed
    views at once, so the source term carries a multiplier too.

    ``source_rate``/``source_channels`` default to the worst layout the request
    could plausibly carry when the caller could not read them
    (:data:`_ASSUMED_SOURCE_RATE` / stereo), keeping this an upper bound rather
    than an optimistic guess.
    """

    duration = max(0.0, duration_seconds)
    rate = source_rate if source_rate > 0 else _ASSUMED_SOURCE_RATE
    channels = source_channels if source_channels > 0 else 2

    # Decoded source: float64 is libsndfile's default read dtype.
    source = duration * rate * channels * 8
    # Resampled mono float32 handed to the model, plus mel/window copies.
    resampled = duration * STT_SAMPLE_RATE * 4 * 3
    return int(source * 2 + resampled)


def speech_buffer_bytes(
    characters: int,
    *,
    speed: float = 1.0,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> int:
    """Peak bytes one synthesis request needs for its output buffers.

    Character count alone is NOT a memory bound, which is why this exists.
    Four factors multiply it:

    * **Speed.** ``speed`` scales duration inversely, and the API permits
      0.25 — four times the samples of the same text at 1.0.
    * **Native output rate.** Engines differ; the registry spans 24 kHz
      (Kokoro/Qwen3) to 44.1 kHz (Dia/VoxCPM). The maximum is assumed so the
      figure bounds every engine.
    * **Requested conversion.** ``sample_rate``/``channels`` resample AFTER
      synthesis, up to 96 kHz stereo. Conversion is not in place: the source
      waveform and the converted array are both live, so the peak is their sum.
    * **Pipeline copies.** The generator retains per-chunk arrays and then
      allocates the concatenation; the encoder adds an int16 copy and the
      output byte buffer. See :data:`_TTS_PEAK_MULTIPLIER`.
    """

    seconds = max(0, characters) / _SPEECH_CHARACTERS_PER_SECOND
    seconds /= max(0.25, float(speed))

    native = seconds * _TTS_NATIVE_SAMPLE_RATE * 4 * _TTS_PEAK_MULTIPLIER
    if sample_rate is None and channels is None:
        return int(native)

    out_rate = sample_rate if sample_rate else _TTS_NATIVE_SAMPLE_RATE
    out_channels = channels if channels else 1
    # float32 converted array + its int16 copy + the encoded byte buffer.
    converted = seconds * out_rate * out_channels * 4 * 2
    return int(native + converted)


#: Longest audio an unreadable container is assumed to hold.
#:
#: Deriving this from a bitrate floor does NOT work. An earlier revision divided
#: the file size by "6 kbps, the lowest any practical codec produces", but that
#: is an assumption about encoders, not a bound: a lower-bitrate or highly
#: compressible stream decodes to more audio than the division predicts, and the
#: charge comes out too small — precisely the case the guard exists for.
#:
#: Since no sound upper bound is derivable from opaque bytes, an unreadable
#: container is charged the maximum a request may hold at all
#: (:data:`MAX_TRANSCRIPTION_SECONDS`). That is by construction an upper bound
#: for anything the route would accept, and it keeps the guard closed on memory
#: without rejecting formats the backend can decode but the probe cannot parse.
def worst_case_duration_seconds(compressed_bytes: int) -> float:
    """Longest audio ``compressed_bytes`` may be assumed to decode to.

    ``compressed_bytes`` is accepted so callers need not special-case an
    unreadable file, but it is deliberately NOT used to scale the result: see
    the note above on why a bitrate floor is not a bound.
    """

    del compressed_bytes
    return MAX_TRANSCRIPTION_SECONDS
