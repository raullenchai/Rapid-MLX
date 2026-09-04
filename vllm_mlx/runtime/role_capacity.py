"""Metadata-backed capacity charges for shared residency roles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleCapacity:
    """A role's footprint charge resolved from durable metadata, not guesses."""

    requested_bytes: int | None
    source: str


@lru_cache(maxsize=64)
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
