"""Metadata-backed capacity charges for shared residency roles."""

from __future__ import annotations

import logging
import re
import threading
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
# no scan has completed yet. Guarded by ``_local_cache_lock`` because lookups
# run on worker threads (``asyncio.to_thread``) and may race each other.
_local_cache_snapshot: tuple[float, dict[str, int]] | None = None
_local_cache_lock = threading.Lock()

# A completed checkpoint that only carries config/tokenizer would vastly
# under-reserve. A weight file is the irreducible signal that the downloaded
# snapshot can actually run a load; a ref-bound snapshot WITHOUT one is a
# selective/partial download and must fail closed.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".npz", ".npy")


def _local_cache_bytes(hf_id: str) -> int | None:
    """Return the verified on-disk footprint of ``hf_id`` in the local HF cache.

    Satisfies the "catalog OR verified local-cache metadata" contract for a
    checkpoint absent (or stale) in the checked-in manifest. Reads from a
    single TTL-bounded snapshot of ``scan_cache_dir()`` (built on demand, under
    a thread lock so a burst of concurrent lookups coalesces into ONE cache
    walk per TTL window regardless of how many model ids are probed — an attacker
    cannot drive a traversal or unbounded memory per public alignment request,
    nor can racing threads each trigger the walk). ``None`` when the repo is not
    cached, only partially downloaded, or the scan fails, so the caller fails
    closed rather than admitting on a partial/uncertain footprint.
    """
    global _local_cache_snapshot
    lc = hf_id.lower()
    now = time.monotonic()
    with _local_cache_lock:
        if (
            _local_cache_snapshot is not None
            and now - _local_cache_snapshot[0] < _LOCAL_CACHE_TTL_SECONDS
        ):
            return _local_cache_snapshot[1].get(lc)
        # Under the lock we re-check freshness: the first thread to observe an
        # expired snapshot rebuilds it; the rest wait and then read the fresh
        # one instead of each re-walking the cache.
        index = _scan_local_cache_index()
        _local_cache_snapshot = (now, index)
        return index.get(lc)


def _scan_local_cache_index() -> dict[str, int]:
    """Return ``{repo_id_lower: completed-footprint}`` from the local HF cache.

    Only repos with a COMPLETE default snapshot are indexed. "Complete" is
    verified with two INDEPENDENT signals, so a partial/selective download
    cannot under-reserve and later blow past the ceiling:

      * the snapshot is ref-bound (a ``snapshot_download`` finished writing its
        ``refs/<branch>`` pointer), AND
      * the ref-bound revision actually contains a model WEIGHT file. A
        selective download that is ref-bound yet carries only config/tokenizer
        is NOT a usable checkpoint and fails closed (round-11).

    The footprint charged is the byte total of the revision the default ref
    resolves to (``main``), NOT the repo's aggregate ``size_on_disk`` (which
    would include a partially-downloaded sibling revision) and NOT the sum of
    every historical snapshot. A repo with no trustworthy snapshot contributes
    nothing, so the caller fails closed (unknown -> 507 under a ceiling).
    """
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:  # pragma: no cover - huggingface_hub is a core dep
        return {}
    try:
        cache = scan_cache_dir()
        index: dict[str, int] = {}
        for repo in cache.repos:
            # Resolve the revision the loader would use: the one the default
            # ``main`` ref points at.
            rev = _default_completed_revision(repo.revisions)
            if rev is None:
                continue
            if not _revision_is_complete(rev):
                logger.debug(
                    "local cache for %r is not verifiably complete; "
                    "failing closed instead of under-reserving",
                    repo.repo_id,
                )
                continue
            size = sum(int(f.size_on_disk or 0) for f in rev.files)
            if size > 0:
                index[repo.repo_id.lower()] = size
        return index
    except Exception as exc:  # noqa: BLE001 - a cache-scan hiccup must not admit blind
        logger.debug("scan_cache_dir failed: %s", exc)
        return {}


def _revision_is_complete(rev) -> bool:
    """Independently verify a snapshot actually holds ALL of a checkpoint.

    ``rev`` is a huggingface_hub ``CachedRevisionInfo``. A ref-bound snapshot
    alone does not prove the weights are present — a selective download can be
    ref-bound yet hold only config/tokenizer, and an interrupted multi-shard
    download can hold ONE weight shard. Verification:

      * if a shard index exists (``*.safetensors.index.json`` or
        ``pytorch_model.bin.index.json``), parse it and REQUIRE every shard it
        names to be present in the snapshot;
      * a shard-PATTERNED weight file (``*-NNNNN-of-MMMMM.safetensors`` /
        ``*.bin``) with NO readable index is an incomplete multi-shard download
        → fail closed, never charge a fraction;
      * otherwise a fully resident single-file layout is verifiably complete
        when it carries a canonical single-file weight: ``*.gguf`` (GGUF is
        always one self-contained file) or a canonical unsharded name
        (``model.safetensors``, ``pytorch_model.bin``, ``model.bin``,
        ``model.npz``, ``model.npy``) — a sharded download never uses those
        plain names, so a lone canonical file is the WHOLE checkpoint; any
        other bare weight name could be one piece of a selective download and
        still fails closed.
    """
    files = {f.file_name.lower() for f in rev.files}

    # Any index format present drives shard verification.
    index_name = next(
        (
            n
            for n in files
            if n.endswith((".safetensors.index.json", "pytorch_model.bin.index.json"))
        ),
        None,
    )
    if index_name is not None:
        try:
            idx_file = next(f for f in rev.files if f.file_name.lower() == index_name)
            import json

            data = json.loads(idx_file.file_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt index must not admit blind
            logger.debug("failed to read shard index: %s", exc)
            return False
        shards = set(data.get("weight_map", {}).values())
        if not shards:
            return False
        missing = [s for s in shards if s.lower() not in files]
        if missing:
            logger.debug(
                "shard index lists shards missing from the snapshot: %s",
                sorted(missing)[:5],
            )
            return False
        return True

    # A shard-PATTERN weight (safetensors, bin, OR split GGUF) without a readable
    # index is an incomplete multi-shard download — never charge a single shard
    # as the whole checkpoint.
    shard_pat = re.compile(r".*-\d{5}-of-\d{5}\.(safetensors|bin|gguf)$")
    if any(shard_pat.match(name) for name in files):
        logger.debug(
            "snapshot has sharded weight file(s) but no readable shard index; "
            "failing closed instead of under-reserving"
        )
        return False

    # With NO shard index, a ref-bound snapshot is complete when it holds ONE
    # self-contained weight via a CANONICAL single-file name. GGUF is always a
    # single self-contained file. A canonical unsharded name (`model.safetensors`,
    # `pytorch_model.bin`, `model.bin`, `model.npz`, `model.npy`) means the
    # checkpoint was published unsharded — a sharded download names its shards
    # `model-00001-of-000XX.*`, never a plain canonical name (those were already
    # rejected above), so a lone canonical file is the WHOLE checkpoint, not a
    # piece of a selective download. Any other single weight name (e.g.
    # `encoder-1.bin`) could be one of several in a selective download, so it
    # still fails closed unless an index proves the full set (pr_validate
    # round-21: a fully-cached canonical single-file checkpoint must not be
    # falsely rejected as unknown).
    canonical_single_file = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "model.npz",
        "model.npy",
    }
    return any(name.endswith(".gguf") for name in files) or any(
        name in canonical_single_file for name in files
    )


def _default_completed_revision(revisions):
    """Pick the completed revision the loader would use (the ``main`` ref).

    The loader resolves a repo's DEFAULT branch — conventionally ``main`` (the
    ref ``snapshot_download`` writes). A lone non-``main`` ref names some other
    branch we may not load, so charging it could understate the weights the real
    load fetches. Returns ``None`` unless a revision is bound to ``main``, so
    the caller fails closed rather than guess a revision to charge.
    """
    default_bound = [rev for rev in revisions if "main" in rev.refs]
    if default_bound:
        return default_bound[0]
    logger.debug(
        "local cache revisions %r are ref-bound but none bound to the default "
        "'main' branch; failing closed instead of charging an unrelated revision",
        [tuple(rev.refs) for rev in revisions],
    )
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
