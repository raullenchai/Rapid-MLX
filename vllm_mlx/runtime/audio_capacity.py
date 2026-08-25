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
#: Two components, kept as one constant because they are charged together:
#:
#: * **Decoded audio buffer.** This one is an arithmetic bound, not a guess.
#:   ``routes.audio.MAX_AUDIO_UPLOAD_SIZE`` caps an upload at 25 MB, which is
#:   ~25 minutes of 16 kHz mono speech; decoded to float32 at the 16 kHz the
#:   STT lane resamples to, that is 25*60*16000*4 B ≈ 92 MiB. Whisper's mel
#:   spectrogram and the windowed copies the decoder holds roughly double it.
#: * **Engine activations.** Encoder/decoder activation memory for the
#:   largest audio checkpoints in the registry.
#:
#: 512 MiB covers both with margin. It is intentionally a fixed allowance
#: rather than a per-model number: the request-side buffer is bounded by the
#: upload cap for *every* model, and the activation side is corrected by the
#: post-load ``phys_footprint`` measurement in the residency manager, which
#: supersedes this reservation once it is larger.
AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES: int = 512 * _MIB


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
    """Sum the real on-disk bytes of the cached snapshot for ``hf_id``.

    Prefers the snapshot pinned by ``refs/main`` — that is the one the loader
    will actually open, and sizing an unrelated stale snapshot would charge the
    wrong number for the weights about to be read.

    Unlike ``_download_gate.is_repo_cached`` this falls back to the LARGEST
    snapshot when no ``refs/main`` exists. The gate is deciding "may we skip a
    download", where guessing wrong silently re-downloads; here the question is
    "how much memory should we reserve", where refusing to answer now rejects
    the request outright. Taking the largest candidate keeps the reservation
    conservative, and any snapshot on disk is a far better estimate than none.

    HF snapshots are trees of symlinks into a sibling ``blobs/`` store, and one
    blob can back several snapshot entries. ``os.stat`` follows the symlink so
    the blob's true size is counted, and ``(st_dev, st_ino)`` de-duplicates
    shared blobs. Returns ``None`` when nothing is cached.
    """

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_root = os.path.join(HF_HUB_CACHE, f"models--{hf_id.replace('/', '--')}")
        snapshots_root = os.path.join(repo_root, "snapshots")
        if not os.path.isdir(snapshots_root):
            return None

        candidates: list[str] = []
        main_ref = os.path.join(repo_root, "refs", "main")
        if os.path.isfile(main_ref):
            try:
                with open(main_ref) as handle:
                    sha = handle.read().strip()
            except OSError:
                sha = ""
            if sha and os.path.isdir(os.path.join(snapshots_root, sha)):
                candidates.append(os.path.join(snapshots_root, sha))
        if not candidates:
            candidates = [
                os.path.join(snapshots_root, name)
                for name in os.listdir(snapshots_root)
                if os.path.isdir(os.path.join(snapshots_root, name))
            ]
        if not candidates:
            return None

        best = 0
        for snapshot_dir in candidates:
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
            best = max(best, total)
        return best or None
    except Exception:
        # Capacity resolution is advisory input to an admission decision; a
        # broken cache directory must degrade to "unknown", never take the
        # audio lane down.
        logger.debug("Failed to measure cached snapshot for %r", hf_id, exc_info=True)
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
