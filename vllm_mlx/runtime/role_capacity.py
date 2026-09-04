"""Metadata-backed capacity charges for shared residency roles."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleCapacity:
    """A role's footprint charge resolved from durable metadata, not guesses."""

    requested_bytes: int | None
    source: str


# ``_LOCAL_CACHE_TTL_SECONDS`` bounds how long a local-footprint lookup (hit
# OR miss) is reused. The HF cache is mutable — a checkpoint can be downloaded,
# deleted or grown between requests — so a result must never be remembered
# forever. The bounded TTL serves two goals at once:
#
#   * a burst of repeated alignment requests (successful or rejected) does not
#     re-walk the whole HF cache each time, and
#   * a previously-missing or partially-cached checkpoint becomes discoverable
#     (re-scanned) once the TTL elapses.
#
# Thus a *complete* cached download is charged promptly, an absent one is
# retried soon, and a still-in-progress download does not get trusted until it
# finishes.
_LOCAL_CACHE_TTL_SECONDS = 30.0
_local_cache_lookups: dict[str, tuple[float, int | None]] = {}


def _local_cache_bytes(hf_id: str) -> int | None:
    """Return the verified on-disk footprint of ``hf_id`` in the local HF cache.

    Satisfies the "catalog OR verified local-cache metadata" contract for a
    checkpoint absent (or stale) in the checked-in manifest. Uses
    ``scan_cache_dir()``'s ``size_on_disk`` (huggingface_hub's own deduped byte
    count — exactly what ``rapid-mlx rm`` reports). ``None`` when the repo is
    not cached, is only partially downloaded, or the scan fails, so the caller
    fails closed rather than admitting on a partial/uncertain footprint.

    Both hits and misses are cached briefly (TTL), so a burst of arbitrary
    valid-looking model ids does not saturate the worker thread pool with full
    cache walks, while a later download still becomes discoverable once the TTL
    expires (cache is mutable and must be re-observed fresh).
    """
    lc = hf_id.lower()
    now = time.monotonic()
    cached = _local_cache_lookups.get(lc)
    if cached is not None and now - cached[0] < _LOCAL_CACHE_TTL_SECONDS:
        return cached[1]
    size = _scan_local_cache_bytes(hf_id)
    _local_cache_lookups[lc] = (now, size)
    return size


def _scan_local_cache_bytes(hf_id: str) -> int | None:
    """Return a COMPLETE cached footprint for ``hf_id``, else ``None``.

    We deliberately require a completed (ref-bound) download before trusting
    ``size_on_disk``: huggingface_hub writes the ``refs/<branch>`` pointer only
    after a ``snapshot_download`` finishes, so a repo whose ``size_on_disk`` we
    read is either a full, usable checkpoint or an in-progress/partial one. A
    partial cache reserves only its small on-disk bytes (e.g. tokenizer/config)
    and would under-charge the residency ledger — the later ``STTEngine.load``
    would download the remaining weights and blow past the ceiling. So a
    partial download returns ``None`` (fail closed). ``None`` on any smash too,
    so a cache-scan hiccup never admits blind.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:  # pragma: no cover - huggingface_hub is a core dep
        return None
    try:
        cache = scan_cache_dir()
        lc = hf_id.lower()
        for repo in cache.repos:
            if repo.repo_id.lower() != lc:
                continue
            # A ref-bound revision is a COMPLETED download. Size_on_disk on a
            # ref-less (in-progress) snapshot only reflects partial bytes.
            if not any(rev.refs for rev in repo.revisions):
                logger.debug(
                    "local cache for %r is partial (no completed snapshot); "
                    "failing closed instead of under-reserving",
                    repo.repo_id,
                )
                return None
            size = int(repo.size_on_disk or 0)
            return size if size > 0 else None
    except Exception as exc:  # noqa: BLE001 - a cache-scan hiccup must not admit blind
        logger.debug("scan_cache_dir for %r failed: %s", hf_id, exc)
        return None
    return None


def alignment_capacity(model_id: str) -> RoleCapacity:
    """Resolve an alignment-role charge from catalog or verified cache metadata.

    The forced-aligner aliases (``qwen3-aligner`` / ``qwen3-forced-aligner``)
    and the underlying HF id (``mlx-community/Qwen3-ForcedAligner-0.6B-8bit``)
    all resolve to the same ``hf_id`` through the audio-alias registry, from
    which the checked-in download-size manifest (:mod:`vllm_mlx.model_sizes`)
    derives a real byte footprint BEFORE any weight loading — so admission can
    decide with knowledge of the size rather than blind.

    When the manifest has no entry (repo not newly mirrored / not in the size
    table), we fall back to the checkpoint's verified local-cache footprint if
    it is already on disk AND fully downloaded. Only when BOTH the catalog and
    the (complete) local cache are empty does ``requested_bytes=None`` with
    ``source="unknown"``, so a configured residency ceiling fails closed
    (``role_capacity_unknown``) instead of admitting without a typed decision.
    """
    from ..audio.registry import resolve_audio_alias
    from ..model_sizes import size_bytes

    entry = resolve_audio_alias(model_id)
    hf_id = entry.hf_id if entry is not None else model_id
    requested_bytes = size_bytes(hf_id)
    if requested_bytes is not None:
        return RoleCapacity(requested_bytes=requested_bytes, source="catalog")
    cached_bytes = _local_cache_bytes(hf_id)
    if cached_bytes is not None:
        return RoleCapacity(requested_bytes=cached_bytes, source="local-cache")
    return RoleCapacity(requested_bytes=None, source="unknown")
