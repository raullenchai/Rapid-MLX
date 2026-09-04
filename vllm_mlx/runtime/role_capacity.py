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


# ``_LOCAL_CACHE_TTL_SECONDS`` bounds how long ONE indexed snapshot of the local
# HF cache is reused. The cache is mutable — a checkpoint can be downloaded,
# deleted or grown between requests — so an index must never be remembered
# forever. Two goals at once:
#
#   * every local-footprint lookup (for any number of model ids) reads from the
#     SAME cached ``scan_cache_dir()`` index, so a burst of arbitrary
#     (attacker-controlled) model ids performs ONE walk per TTL window, not one
#     per id — this also bounds memory to a single index, not one entry per id;
#   * a previously-missing or newly-downloaded checkpoint becomes discoverable
#     once the TTL elapses and the index is re-scanned.
#
# A *complete* cached download is charged promptly; an absent or still
# in-progress one is not trusted until a later re-scan observes it complete.
_LOCAL_CACHE_TTL_SECONDS = 30.0
# ``(monotonic_timestamp, {repo_id_lower: footprint_bytes})`` or ``None`` when
# no scan has completed yet.
_local_cache_snapshot: tuple[float, dict[str, int]] | None = None


def _local_cache_bytes(hf_id: str) -> int | None:
    """Return the verified on-disk footprint of ``hf_id`` in the local HF cache.

    Satisfies the "catalog OR verified local-cache metadata" contract for a
    checkpoint absent (or stale) in the checked-in manifest. Reads from a
    single TTL-bounded snapshot of ``scan_cache_dir()`` (built on demand), so
    repeated lookups coalesce into one cache walk per TTL window regardless of
    how many distinct model ids are probed — an attacker cannot drive one
    ``scan_cache_dir()`` traversal (or unbounded memory) per public alignment
    request. ``None`` when the repo is not cached, only partially downloaded,
    or the scan fails, so the caller fails closed rather than admitting on a
    partial/uncertain footprint.
    """
    global _local_cache_snapshot
    now = time.monotonic()
    if (
        _local_cache_snapshot is not None
        and now - _local_cache_snapshot[0] < _LOCAL_CACHE_TTL_SECONDS
    ):
        return _local_cache_snapshot[1].get(hf_id.lower())

    index = _scan_local_cache_index()
    _local_cache_snapshot = (now, index)
    return index.get(hf_id.lower())


def _scan_local_cache_index() -> dict[str, int]:
    """Return ``{repo_id_lower: completed-footprint}`` from the local HF cache.

    Only repos with a COMPLETE (ref-bound) default snapshot are indexed. The
    footprint for a repo is the byte total of its completed snapshot — the
    revision a ``load`` would use — NOT the repo's aggregate ``size_on_disk``,
    which would include a partially-downloaded second revision and understate
    what the load will materialize. A repo with no completed snapshot (still
    downloading, or only config/tokenizer cached) contributes nothing, so the
    caller fails closed (unknown -> 507 under a ceiling) instead of reserving
    only partial bytes and blowing past the ceiling on the real load.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:  # pragma: no cover - huggingface_hub is a core dep
        return {}
    try:
        cache = scan_cache_dir()
        index: dict[str, int] = {}
        for repo in cache.repos:
            # The loader resolves to the ref-bound (default) revision; only a
            # snapshot that is complete (has a ref pointing at it) is trusted.
            completed = [rev for rev in repo.revisions if rev.refs]
            if not completed:
                if any(rev for rev in repo.revisions):
                    logger.debug(
                        "local cache for %r has no completed snapshot; "
                        "failing closed instead of under-reserving",
                        repo.repo_id,
                    )
                continue
            # Sum the file bytes of the completed snapshot(s) actually present.
            size = 0
            for rev in completed:
                size += sum(int(f.size_on_disk or 0) for f in rev.files)
            if size > 0:
                index[repo.repo_id.lower()] = size
        return index
    except Exception as exc:  # noqa: BLE001 - a cache-scan hiccup must not admit blind
        logger.debug("scan_cache_dir failed: %s", exc)
        return {}


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
    it is already on disk AND fully downloaded (a complete, ref-bound
    snapshot). Only when BOTH the catalog and the (complete) local cache are
    empty does ``requested_bytes=None`` with ``source="unknown"``, so a
    configured residency ceiling fails closed (``role_capacity_unknown``)
    instead of admitting without a typed decision.
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
