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


# ``_LOCAL_CACHE_TTL_SECONDS`` bounds how long a *successful* local-footprint
# lookup is reused. The HF cache is mutable: a checkpoint can be deleted or
# grown between requests, so the result must never be remembered forever. The
# TTL lets a burst of repeated rejected alignments skip re-walking the whole
# cache, while a later download (or rm) becomes discoverable once it expires.
# Only positive results are cached — a miss is never memoized, so a previously
# uncached checkpoint is always retried (round-4 NIT: bounded-TTL to avoid the
# repeated full scan; round-3: never permanently memorize a mutable footprint).
_LOCAL_CACHE_TTL_SECONDS = 30.0
_local_cache_hits: dict[str, tuple[float, int]] = {}


def _local_cache_bytes(hf_id: str) -> int | None:
    """Return the verified on-disk footprint of ``hf_id`` in the local HF cache.

    This satisfies the "catalog OR verified local-cache metadata" contract for
    a checkpoint that is already downloaded but absent (or stale) in the
    checked-in manifest. We deliberately use ``scan_cache_dir()``'s
    ``size_on_disk`` — huggingface_hub's own deduped byte count — rather than
    a directory walk, because ``size_on_disk`` is exactly what ``rapid-mlx rm``
    reports and is the same number a user would expect freeing. ``None`` when
    the repo is not cached or the lookup fails so the caller fails closed.
    """
    lc = hf_id.lower()
    now = time.monotonic()
    cached = _local_cache_hits.get(lc)
    if cached is not None and now - cached[0] < _LOCAL_CACHE_TTL_SECONDS:
        return cached[1]
    size = _scan_local_cache_bytes(hf_id)
    if size is not None:
        _local_cache_hits[lc] = (now, size)
    else:
        # Never memoize a miss: the checkpoint may be downloaded moments later
        # and must become discoverable on the next admission.
        _local_cache_hits.pop(lc, None)
    return size


def _scan_local_cache_bytes(hf_id: str) -> int | None:
    """Walk huggingface_hub's deduped cache to find ``hf_id``'s footprint."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:  # pragma: no cover - huggingface_hub is a core dep
        return None
    try:
        cache = scan_cache_dir()
        lc = hf_id.lower()
        for repo in cache.repos:
            if repo.repo_id.lower() == lc:
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
    it is already on disk. Only when BOTH the catalog and the local cache are
    empty does ``requested_bytes=None`` with ``source="unknown"``, so a
    configured residency ceiling fails closed (``role_capacity_unknown``)
    instead of admitting without a typed capacity decision.
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
