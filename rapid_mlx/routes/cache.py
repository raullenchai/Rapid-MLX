# SPDX-License-Identifier: Apache-2.0
"""KV cache export/import HTTP API (issue #476).

This module defines the wire surface (request/response shapes, auth,
path-whitelist, manifest validation) AND the engine-level save/load body:

* ``POST /v1/cache/export`` — validates the request, resolves the
  destination under the sandbox, then calls ``EngineCore.save_cache_to_disk``
  to snapshot the in-memory prefix cache and writes a ``manifest.json``
  alongside it. Returns 200 with a byte/entry summary.
* ``POST /v1/cache/import`` — validates the request, resolves the source
  under the sandbox, reads + checks the manifest against caller
  expectations (protocol_version / model_id mismatch → 409), then calls
  ``EngineCore.load_cache_from_disk`` to hydrate the prefix cache. With
  ``merge_strategy="replace"`` the in-memory cache is cleared first.
* ``GET /v1/cache/info`` — reads the manifest at a whitelisted path and
  returns it. Lets a peer instance (or oai-mlx) GC / inspect an export
  root without round-tripping a full import. H-12: response carries
  ``protocol_version`` + ``manifest`` only — the resolved sandbox root
  stays in the server log, never on the wire.

Both engine-touching handlers 503 when no model is loaded (``cfg.engine
is None``), matching ``rapid_mlx.routes.health``'s ``clear_cache`` idiom.

Quiesce (out of MVP): the issue's ``wait_for_quiesce_seconds`` — draining
in-flight decode before snapshotting — is deliberately NOT implemented
here. ``EngineCore.save_cache_to_disk`` already serializes on the mlx-step
worker thread (that's why the KV arrays are materializable at all), so a
snapshot taken while a request is mid-decode captures a consistent
per-entry view; a torn cross-entry snapshot is possible only under
concurrent store()/evict, which is acceptable for the MVP (the importer
validates each entry independently and drops any that fail). A real
quiesce controller is tracked in #476's follow-up list.

Auth follows ``rapid_mlx.routes.health``'s ``router``: the bearer key is
enforced when ``--api-key`` is set, no new header is invented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Literal

import anyio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..cache.protocol import (
    MANIFEST_FILENAME,
    PROTOCOL_VERSION,
    CommittedIndexUnreadableError,
    EngineNotReadyError,
    InvalidExportPathError,  # noqa: F401 — re-exported for _resolve_or_400 callers
    MalformedManifestError,  # noqa: F401 — used via _read_manifest_or_http
    ManifestMismatchError,
    ManifestNotFoundError,  # noqa: F401 — used via _read_manifest_or_http
    build_manifest_from_engine_state,
    default_export_root,
    read_manifest,
    resolve_cache_dir,
    resolve_engine_cache_geometry,
    resolve_engine_model_id,
    write_manifest,
)
from ..config import get_config
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

# H-02/H-12: the resolved sandbox path never rides the wire. On the
# engine-touching paths we log the destination/source server-side and
# return only caller-oriented counters. ``_ENGINE_NOT_LOADED_DETAIL``
# keeps the same sanitized-envelope discipline the stub established for
# its 403 body — no path, no manifest excerpt.
_ENGINE_NOT_LOADED_DETAIL = "engine not loaded"

# H-02: sandbox-escape 403 envelope. The underlying
# ``InvalidExportPathError`` carries the caller-supplied path AND the
# fully resolved sandbox root (``/Users/<username>/.cache/rapid-mlx/
# cache_exports``). Echoing either to an unauthenticated caller leaks
# the operator's home dir + username. Same treatment as the #756 501
# envelope: generic wire message, full detail goes to the server log.
_SANDBOX_ESCAPE_MSG = "destination must resolve under the cache-export sandbox"
_SANDBOX_ESCAPE_DETAIL = {
    "error": {
        "message": _SANDBOX_ESCAPE_MSG,
        "type": "invalid_request_error",
        "code": "sandbox_escape",
    }
}


router = APIRouter(
    prefix="/v1/cache",
    tags=["cache"],
    dependencies=[Depends(verify_api_key)],
)


# #1100 codex round 3 (findings #2/#5): serialize the whole export/import
# transaction (save → manifest → committed-size gate → publish/discard, or
# manifest-read → load) PER RESOLVED DESTINATION. Two concurrent operations on
# the same path could otherwise interleave so one writes a manifest for
# another's snapshot, or an import loads a blob a concurrent export swapped
# after the manifest validated. A per-destination asyncio.Lock makes the path
# a single-writer resource; distinct paths still run concurrently. (The engine
# already serializes KV materialization on its mlx-step thread; this guards the
# route-side filesystem transaction around it.)
#
# #1100 codex round 4 (#5): the lock registry is REFERENCE-COUNTED so it can't
# grow without bound. The round-3 version cached an ``asyncio.Lock`` per
# caller-selected path FOREVER — every unique path (including a missing-source
# import that 404s) left a permanent dict entry, an unbounded-memory footgun
# under adversarial or high-cardinality path use. Now each entry tracks how
# many holders/waiters reference it and is EVICTED when the last one releases,
# so the registry only ever holds locks for paths with a transaction in flight.
_export_dest_locks: dict[str, _RefCountedLock] = {}
_export_locks_guard = asyncio.Lock()


class _RefCountedLock:
    """An ``asyncio.Lock`` plus a reference count for registry eviction."""

    __slots__ = ("lock", "refs")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.refs = 0


class _dest_lock:
    """Async context manager: acquire the per-destination lock for ``key``,
    reference-counted so the registry entry is evicted when idle (#1100 codex
    round 4 #5).

    Usage::

        async with _dest_lock(path), _InterProcessLock(path):
            ...

    The ref count is bumped under ``_export_locks_guard`` on entry (so a
    concurrent waiter keeps the entry alive) and decremented on exit; the entry
    is removed only when it drops to 0 AND the lock is not held.
    """

    __slots__ = ("_key", "_entry")

    def __init__(self, key: Path | str) -> None:
        self._key = str(key)
        self._entry: _RefCountedLock | None = None

    async def __aenter__(self) -> asyncio.Lock:
        async with _export_locks_guard:
            entry = _export_dest_locks.get(self._key)
            if entry is None:
                entry = _RefCountedLock()
                _export_dest_locks[self._key] = entry
            entry.refs += 1
            self._entry = entry
        try:
            await entry.lock.acquire()
        except BaseException:
            # #1100 codex round 5 (#3): a cancellation (or any error) while
            # awaiting the lock happens AFTER refs was bumped but BEFORE
            # ``__aexit__`` can run — the ``async with`` never entered its body,
            # so ``__aexit__`` won't fire and the ref would leak permanently
            # (a canceled waiter on every request would grow the registry
            # unboundedly). Undo the ref here and evict if we were the last.
            async with _export_locks_guard:
                entry.refs -= 1
                if entry.refs <= 0 and not entry.lock.locked():
                    _export_dest_locks.pop(self._key, None)
            self._entry = None
            raise
        return entry.lock

    async def __aexit__(self, *exc: object) -> None:
        entry = self._entry
        self._entry = None
        if entry is None:  # pragma: no cover — defensive
            return
        entry.lock.release()
        async with _export_locks_guard:
            entry.refs -= 1
            # Evict only when no other holder/waiter references this key AND
            # the lock is free — avoids removing an entry another coroutine is
            # about to acquire (it holds its own ref, so refs>0 protects it).
            if entry.refs <= 0 and not entry.lock.locked():
                _export_dest_locks.pop(self._key, None)


# #1100 codex round 4 (#5): the per-destination ``asyncio.Lock`` above is
# process-local — it serializes concurrent requests WITHIN one rapid-mlx
# instance but does nothing across SEPARATE instances that export/import to a
# SHARED destination (an NFS / SMB mount two servers both point at). Those
# processes collide on ``save_to_disk``'s fixed ``<dest>.new`` / ``<dest>.old``
# staging paths: instance A's ``.new`` gets clobbered by instance B, or A's
# atomic rename publishes B's half-written blob. An advisory whole-file
# ``flock`` on a ``<dest>.txlock`` sibling makes the WHOLE transaction (stage →
# publish → manifest) mutually exclusive across processes on the same host /
# lock-aware filesystem. The lock is a plain O_CREAT file we never delete
# (deleting it races the next acquirer); its bytes are meaningless.
#
# ``flock`` is POSIX-only and advisory (both parties must opt in — they do,
# it's the same code path). On a filesystem without working ``flock`` (some
# network mounts) the acquire raises.
#
# #1100 codex round 5 (#2): a flock failure used to merely log and PROCEED,
# which on an unsupported SHARED filesystem let two instances concurrently
# corrupt the fixed ``.new``/``.old`` staging. Since the process-local
# ``_dest_lock`` already fully serializes a SINGLE instance, the flock matters
# ONLY for the multi-instance shared-FS case — exactly where silently
# proceeding is unsafe. So the caller now FAILS the transaction (503) when the
# flock can't be acquired, UNLESS the operator explicitly opts out via
# ``RAPID_MLX_CACHE_ALLOW_UNSAFE_SHARED_FS=1`` (a documented escape hatch for a
# deployment that KNOWS it runs a single instance on a flock-less mount). The
# lock exposes ``degraded`` so the handler can decide.
_ALLOW_UNSAFE_SHARED_FS_ENV = "RAPID_MLX_CACHE_ALLOW_UNSAFE_SHARED_FS"


class _InterProcessLock:
    """Advisory cross-process exclusive lock on ``<path>.txlock`` via flock.

    Async-friendly: the (potentially blocking) ``flock`` acquire runs in a
    worker thread so the event loop isn't stalled while contending with
    another process. If ``fcntl``/``flock`` is unavailable or the filesystem
    rejects it, ``__aenter__`` sets ``self.degraded = True`` (and acquires
    nothing) so the caller can fail the transaction rather than proceed
    without cross-process exclusion (#1100 codex round 5 #2).
    """

    _degraded_logged = False

    def __init__(self, target: Path) -> None:
        # Sibling of the destination, NOT a child — a child under ``<dest>``
        # would be swept into the export blob / counted against max_bytes.
        self._lock_path = str(target).rstrip(os.sep) + ".txlock"
        self._fd: int | None = None
        self.degraded: bool = False

    async def __aenter__(self) -> _InterProcessLock:
        try:
            import fcntl

            def _acquire() -> int:
                # #1100 codex round 4 (#4): create the lockfile's parent dir
                # FIRST. For a NESTED destination (e.g. ``sub/snap``) whose
                # parent doesn't exist yet on a first export, ``os.open`` of
                # ``sub/snap.txlock`` would raise ENOENT — the old code caught
                # that as "flock unavailable" and SILENTLY disabled cross-
                # process exclusion. Ensuring the parent exists makes the lock
                # reliably acquirable for real nested paths; a genuine
                # flock-unsupported filesystem still surfaces below.
                parent = os.path.dirname(self._lock_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except OSError:
                    os.close(fd)
                    raise
                return fd

            self._fd = await anyio.to_thread.run_sync(_acquire)
        except (ImportError, OSError) as exc:
            self._fd = None
            self.degraded = True
            if not _InterProcessLock._degraded_logged:
                _InterProcessLock._degraded_logged = True
                logger.warning(
                    "cache: cross-process export lock unavailable (%s: %s); "
                    "cross-process exclusion is NOT active — the transaction "
                    "will be rejected unless %s=1 is set (single-instance "
                    "escape hatch)",
                    type(exc).__name__,
                    exc,
                    _ALLOW_UNSAFE_SHARED_FS_ENV,
                )
        return self

    async def __aexit__(self, *exc: object) -> None:
        fd = self._fd
        self._fd = None
        if fd is None:
            return

        def _release() -> None:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        await anyio.to_thread.run_sync(_release)


def _reject_if_ipc_lock_degraded(iplock: _InterProcessLock) -> None:
    """Raise 503 if the inter-process lock degraded and the operator has not
    opted into unsafe shared-FS operation (#1100 codex round 5 #2)."""
    if not iplock.degraded:
        return
    if os.environ.get(_ALLOW_UNSAFE_SHARED_FS_ENV) == "1":
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "cross-process cache lock unavailable on this filesystem; refusing "
            "to proceed without exclusion (set "
            f"{_ALLOW_UNSAFE_SHARED_FS_ENV}=1 to override on a single-instance "
            "deployment)"
        ),
    )


class ExportRequest(BaseModel):
    """Request body for ``POST /v1/cache/export``."""

    destination: str | None = Field(
        default=None,
        description=(
            "Path under RAPID_MLX_CACHE_EXPORT_DIR (default "
            "~/.cache/rapid-mlx/cache_exports/). May be relative (resolved "
            "against the sandbox root) or absolute (must resolve inside "
            "the sandbox). Omit to use the sandbox root itself."
        ),
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on the exported blob's COMMITTED ON-DISK size "
            "(the sum of real file sizes an importing peer will read: "
            "entry_*.safetensors + entry_*_tokens.bin + index.json + "
            "manifest.json, including safetensors headers / alignment / "
            "serialization overhead — NOT just the logical KV byte count). "
            "Enforced with 413 in TWO stages: a cheap pre-write check "
            "against the live in-memory footprint (rejects before touching "
            "disk), AND a precise post-write check against the actual "
            "committed on-disk directory size (catches cache growth that "
            "raced the snapshot AND the on-disk overhead the logical count "
            "excludes — the over-cap blob is then discarded from disk). MVP "
            "is all-or-nothing — no partial-entry eviction to fit under the "
            "cap."
        ),
    )


class ImportRequest(BaseModel):
    """Request body for ``POST /v1/cache/import``."""

    source: str = Field(
        ...,
        description=(
            "Path to an export root containing manifest.json + index.json. "
            "Resolved under the export sandbox (see ExportRequest.destination)."
        ),
    )
    expected_protocol_version: str = Field(
        default=PROTOCOL_VERSION,
        description=(
            "Manifest protocol version the caller expects. Mismatch → 409. "
            f"Current: {PROTOCOL_VERSION!r}."
        ),
    )
    expected_model_id: str | None = Field(
        default=None,
        description=(
            "Optional ADDITIONAL caller-side assertion on manifest.model_id "
            "(exact match; mismatch → 409). NOTE: the server ALWAYS rejects a "
            "manifest whose model_id differs from the model it loaded — "
            "omitting this does NOT disable identity checking, it only skips "
            "the extra caller-side assertion. Set it to pin a specific model "
            "id independently of the server's loaded model."
        ),
    )
    merge_strategy: Literal["replace", "merge"] = Field(
        default="merge",
        description=(
            "'merge' keeps existing entries and adds new ones (token-tuple "
            "key collisions resolved by the engine's ``store``). 'replace' "
            "clears the in-memory cache ATOMICALLY inside the step-thread "
            "load — only after the source index is validated — so the "
            "imported blob is the only thing in the cache, and a corrupt/"
            "missing source leaves the existing cache intact."
        ),
    )


class ExportResponse(BaseModel):
    """200 body for ``POST /v1/cache/export`` — caller-oriented summary.

    H-02/H-12: NO resolved sandbox path is echoed. ``manifest_path`` is the
    caller-relative filename (``manifest.json``), not the absolute on-disk
    location — a peer that wrote to ``destination="session-a"`` already
    knows where it asked to write; it does not need (and must not learn)
    the operator's ``$HOME``-rooted expansion.
    """

    protocol_version: str
    entries_exported: int
    bytes_written: int
    model_id: str
    quantization: str
    paged_cache: bool
    turboquant_kv: bool
    manifest_path: str


class ImportResponse(BaseModel):
    """200 body for ``POST /v1/cache/import`` — caller-oriented summary."""

    protocol_version: str
    entries_loaded: int
    entries_skipped: int
    bytes_loaded: int


def _resolve_or_400(caller_path: str | None) -> Path:
    """Wrap ``resolve_cache_dir`` so path violations surface as 403.

    H-02: ``InvalidExportPathError`` carries the caller-supplied path AND
    the resolved sandbox root (which expands to ``/Users/<USERNAME>/.cache
    /rapid-mlx/cache_exports`` on macOS). Both stay in the server log via
    ``logger.warning`` — only the sanitized envelope reaches the wire.
    """
    try:
        return resolve_cache_dir(caller_path)
    except InvalidExportPathError as exc:
        # 403 (not 400) because the request is well-formed JSON — what's
        # rejected is the *authorization* to write/read outside the sandbox.
        logger.warning(
            "cache: sandbox-escape rejected (caller_path=%r): %s",
            caller_path,
            exc,
        )
        raise HTTPException(status_code=403, detail=_SANDBOX_ESCAPE_DETAIL) from exc


def _read_manifest_or_http(root: Path):
    """Wrap ``read_manifest`` so missing/malformed surface as 404/400.

    Without this, a peer-written corrupt ``manifest.json`` would escape
    as a JSONDecodeError → FastAPI 500, hiding a caller-controlled bug
    inside an opaque server error. Mapping the three failure modes
    distinctly is what makes the contract usable from a client.

    Response details are caller-oriented — the fully resolved local
    filesystem path stays in the server log only, not in the HTTP body
    where a bearer-token holder could harvest the export-root layout.
    """
    try:
        return read_manifest(root)
    except ManifestNotFoundError as exc:
        logger.info("cache: manifest not found at %s", root)
        raise HTTPException(
            status_code=404,
            detail="no manifest.json at the requested cache path",
        ) from exc
    except MalformedManifestError as exc:
        # ``str(exc)`` is already path-free (see protocol.read_manifest).
        # It carries the structural reason — "not valid JSON: ...",
        # "must decode to a JSON object, got list", "manifest field
        # 'entries': expected int, got str" — which the client needs to
        # fix its own payload. The resolved path only lands in server logs.
        logger.warning("cache: malformed manifest at %s: %s", root, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _engine_or_503():
    """Return the loaded engine or raise 503 (matches ``health.clear_cache``).

    The engine-touching routes are meaningless before a model is loaded —
    there is no prefix cache to snapshot / hydrate. Same 503 idiom the rest
    of the cache-management surface uses so operators get one consistent
    "engine not loaded" signal across ``/v1/cache/clear``, ``/v1/cache/stats``
    and this pair.
    """
    engine = get_config().engine
    if engine is None:
        raise HTTPException(status_code=503, detail=_ENGINE_NOT_LOADED_DETAIL)
    return engine


def _prefix_cache(engine):
    """The engine's in-memory ``MemoryAwarePrefixCache`` or ``None``.

    Delegates to ``runtime.cache._resolve_memory_aware_cache`` so BOTH engine
    shapes resolve: a bare ``EngineCore`` (``engine.scheduler``) AND the
    production ``BatchedEngine`` (``engine._engine.engine.scheduler``). A
    naive ``getattr(engine, "scheduler")`` here returned None under
    BatchedEngine, which silently disarmed the ``max_bytes`` 413 gate (read
    ``_current_memory``=0) and the ``merge_strategy="replace"`` clear (cache
    None) — the real-hardware smoke bug this fix closes.

    ``None`` still means "no prefix cache" (disabled via
    ``--disable-prefix-cache`` or a genuinely foreign engine); callers treat
    it as an empty snapshot (0 entries / 0 bytes), not an error.
    """
    from ..runtime.cache import _resolve_memory_aware_cache

    return _resolve_memory_aware_cache(engine)


def _committed_dir_size(destination: Path) -> int:
    """Sum the actual on-disk byte size of the committed export blob.

    #1100 BLOCKING-4: ``max_bytes`` is documented as capping the committed
    ON-DISK blob size, but the old post-write gate summed only logical
    ``memory_bytes`` (``manifest.total_bytes``) — excluding the tokens.bin
    sidecars, index.json, manifest.json, and every safetensors header /
    alignment / serialization overhead. An accepted export could therefore
    exceed the cap on disk. This walks the export directory and sums real
    file sizes (``os.stat``) so the cap is enforced against what a peer
    actually reads.

    When ``destination`` IS the sandbox root (the ``destination=None``
    export shape), only the known ``save_to_disk`` + manifest artifacts are
    counted — unrelated files that happen to share the root are not this
    export's footprint and must not be charged against its cap.
    """
    root = default_export_root()
    total = 0
    try:
        if destination != root:
            # Dedicated export sub-dir: every file under it is this export.
            for dirpath, _dirnames, filenames in os.walk(destination):
                for name in filenames:
                    try:
                        total += os.stat(os.path.join(dirpath, name)).st_size
                    except OSError:  # pragma: no cover — racing unlink
                        pass
            return total
        # destination is the sandbox root — count only this export's blobs.
        for name in os.listdir(destination):
            is_blob = name in ("index.json", MANIFEST_FILENAME) or (
                name.startswith("entry_")
                and (name.endswith(".safetensors") or name.endswith("_tokens.bin"))
            )
            if is_blob:
                try:
                    total += os.stat(destination / name).st_size
                except OSError:  # pragma: no cover — racing unlink
                    pass
    except OSError:  # pragma: no cover — destination vanished
        pass
    return total


# The two files whose PRESENCE makes an export importable: ``manifest.json``
# (the import route reads it first — no manifest → 404) and ``index.json``
# (``load_from_disk`` needs it — no index → nothing loads). If cleanup can't
# delete a blob, quarantining THESE two (rename to ``.rejected``) is enough
# to guarantee the import path refuses it. The entry_* payload files alone
# are inert without an index pointing at them.
_IMPORT_CRITICAL_NAMES = (MANIFEST_FILENAME, "index.json")


def _has_committed_index(d: Path) -> bool:
    """True if ``d`` holds a readable ``index.json`` with >=1 declared entry.

    Cheap validity check — mirrors the ``_has_valid_index`` recovery gate
    inside ``MemoryAwarePrefixCache.load_from_disk`` (a zero-entry index is
    degenerate and treated as no snapshot).
    """
    p = d / "index.json"
    if not p.is_file():
        return False
    try:
        obj = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(obj, dict) and bool(obj.get("entries"))


def _is_importable_snapshot(d: Path) -> bool:
    """True if ``d`` is a snapshot a peer could actually IMPORT — a valid
    ``index.json`` (>=1 entry) AND a ``manifest.json``.

    #1100 codex round 5 (#5): ``save_to_disk``'s ``.new`` staging dir has a
    valid index but NO manifest (the route writes ``manifest.json`` only AFTER
    the atomic rename publishes the blob). So a ``.new`` that survived a failed
    save is index-valid yet NON-importable. Recovering it over a complete
    ``.old`` (a previously-published snapshot that DOES carry a manifest) would
    replace an importable snapshot with a broken one and then delete ``.old``.
    Recovery must therefore only promote a FULLY importable candidate.
    """
    return _has_committed_index(d) and (d / MANIFEST_FILENAME).is_file()


def _sweep_staging_dirs(destination: Path) -> None:
    """Clean up staging dirs a FAILED save left, WITHOUT destroying a
    recoverable snapshot.

    #1100 codex round 3 (#2): ``save_to_disk`` writes into ``<dest>.new`` and
    commits via a 3-step swap (``dest → .old``, ``.new → dest``, ``rm .old``).
    A non-committing save usually leaves ``<dest>`` untouched and only orphans
    the staging siblings — so we sweep those.

    #1100 codex round 4 (#1): BUT if the save failed mid-swap, ``<dest>`` can
    be MISSING while the last valid snapshot sits in ``.old`` (dest was already
    renamed away). Blindly deleting both would destroy the only recoverable
    copy — the exact data loss ``load_from_disk``'s crash recovery exists to
    prevent. So if ``<dest>`` isn't itself importable, RESTORE an importable
    snapshot from a staging sibling before removing leftovers.

    #1100 codex round 5 (#5): the candidate must be a FULLY IMPORTABLE snapshot
    (valid index AND a ``manifest.json``), and we prefer ``.old`` over ``.new``.
    ``.new`` is ``save_to_disk``'s raw staging — it has a valid index but the
    route writes ``manifest.json`` only AFTER the atomic publish, so a failed
    save's ``.new`` is manifest-LESS and non-importable. ``.old`` is the prior
    PUBLISHED snapshot (index + manifest). Promoting a manifest-less ``.new``
    over a complete ``.old`` (then deleting ``.old``) would replace a good
    snapshot with a broken one. Preferring ``.old`` and gating on
    ``_is_importable_snapshot`` avoids that. Best-effort: failures are logs.
    """
    base = str(destination).rstrip(os.sep)
    new_dir = Path(base + ".new")
    old_dir = Path(base + ".old")

    # Recover an importable snapshot into ``destination`` if it isn't itself
    # importable. Prefer ``.old`` (published: index + manifest) over ``.new``
    # (raw staging: index only). Never overwrite an importable published dest.
    if not _is_importable_snapshot(destination):
        for cand in (old_dir, new_dir):
            if _is_importable_snapshot(cand):
                try:
                    if destination.exists():
                        shutil.rmtree(destination, ignore_errors=True)
                    os.rename(cand, destination)
                    logger.warning(
                        "cache/export: recovered an importable snapshot from "
                        "%s → %s after a failed save (staging would otherwise "
                        "be swept)",
                        cand,
                        destination,
                    )
                    break
                except OSError as exc:  # pragma: no cover — defensive
                    logger.error(
                        "cache/export: could not recover snapshot from %s: %s",
                        cand,
                        exc,
                    )

    for staging in (new_dir, old_dir):
        if staging.exists():
            try:
                shutil.rmtree(staging, ignore_errors=True)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "cache/export: could not sweep staging dir %s: %s",
                    staging,
                    exc,
                )


class _ExportDiscardError(RuntimeError):
    """Raised when a rejected export blob could NOT be made non-importable.

    #1100 BLOCKING-5: a 413/500 reject MUST guarantee the blob is not
    importable. If deletion fails AND quarantine (rename the import-critical
    files to ``.rejected``) also fails, we cannot honor that invariant —
    surface a 500 rather than falsely claim the blob was discarded.
    """


def _quarantine_import_critical(destination: Path) -> bool:
    """Rename import-critical files to ``.rejected`` so import refuses them.

    Returns True if, after the attempt, NEITHER ``manifest.json`` nor
    ``index.json`` remains at ``destination`` (the blob can no longer be
    imported). Returns False if either still exists (quarantine failed).
    """
    for name in _IMPORT_CRITICAL_NAMES:
        src = destination / name
        if not src.exists():
            continue
        try:
            os.replace(src, destination / f"{name}.rejected")
        except OSError as exc:  # pragma: no cover — exercised via monkeypatch
            logger.error("cache/export: quarantine rename failed for %s: %s", src, exc)
    # Verify the invariant: no import-critical file remains.
    return not any((destination / n).exists() for n in _IMPORT_CRITICAL_NAMES)


def _blob_artifacts(destination: Path) -> list[str]:
    """Names under ``destination`` that belong to a save_to_disk export."""
    names = []
    for name in os.listdir(destination):
        is_blob = name in ("index.json", MANIFEST_FILENAME) or (
            name.startswith("entry_")
            and (name.endswith(".safetensors") or name.endswith("_tokens.bin"))
        )
        if is_blob:
            names.append(name)
    return names


def _discard_export(destination: Path) -> None:
    """Delete a just-written export blob after a post-write reject, and VERIFY
    it is actually gone (#1100 BLOCKING-2 max_bytes race + BLOCKING-5
    cleanup-verification).

    ``save_to_disk`` commits the blob (index.json + entry_* files) into
    ``destination`` via an atomic rename before the route can enforce the
    exact committed size / detect a failed save. When we reject we must not
    leave the blob on disk for a peer to import.

    The round-1 helper used ``shutil.rmtree(..., ignore_errors=True)`` and
    returned immediately, so a permission / filesystem failure silently left
    the rejected blob on disk while the API claimed it was discarded — a peer
    could then import it. This version:

    1. Attempts wholesale removal (dedicated sub-dir) or per-artifact unlink
       (sandbox-root export shape).
    2. VERIFIES the import-critical files (manifest.json + index.json) are
       gone. If any survives, QUARANTINES it (rename to ``.rejected`` so the
       import path — which reads ``manifest.json`` / ``index.json`` by exact
       name — refuses it).
    3. If even quarantine can't remove them, raises ``_ExportDiscardError``
       so the caller returns a 500 instead of falsely claiming the blob was
       discarded. The invariant a 413/500 reject upholds: the blob is NOT
       importable.

    When ``destination`` IS the sandbox root itself (the ``destination=None``
    export shape), only the known ``save_to_disk`` artifacts are touched —
    the root directory and anything else in it are preserved.
    """
    root = default_export_root()
    try:
        if destination != root:
            shutil.rmtree(destination, ignore_errors=True)
        else:
            # destination is the sandbox root — unlink only the blob artifacts.
            for name in _blob_artifacts(destination):
                try:
                    (destination / name).unlink()
                except OSError:  # pragma: no cover — racing unlink
                    pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "cache/export: rmtree/unlink during discard raised (destination=%s): %s",
            destination,
            exc,
        )

    # VERIFY the import-critical files are gone. On a dedicated sub-dir a
    # successful rmtree removed the whole tree (nothing remains); on the
    # sandbox-root shape the unlinks above should have cleared them.
    survivors = [n for n in _IMPORT_CRITICAL_NAMES if (destination / n).exists()]
    if not survivors:
        return  # clean — blob is not importable.

    # Deletion left import-critical files behind (EACCES, immutable flag, a
    # racing reopen). Quarantine them so import refuses the blob.
    logger.warning(
        "cache/export: discard could not delete %s at %s — quarantining",
        survivors,
        destination,
    )
    if _quarantine_import_critical(destination):
        logger.warning(
            "cache/export: quarantined rejected blob at %s (renamed %s → .rejected)",
            destination,
            survivors,
        )
        return

    # Could neither delete nor quarantine — the blob may still be importable.
    # Do NOT claim success.
    raise _ExportDiscardError(
        f"rejected export blob at {destination} could not be made "
        f"non-importable (survivors: {survivors})"
    )


@router.post("/export", response_model=ExportResponse)
async def export_cache(req: ExportRequest):
    """Export the engine's KV prefix cache to disk under the sandbox root.

    Flow: resolve the destination against the path whitelist (403 on
    escape) → require a loaded engine (503) → pre-write ``max_bytes`` gate
    on the cheap in-memory footprint (413) → snapshot via
    ``EngineCore.save_cache_to_disk`` → build the manifest from the
    committed on-disk index → post-write ``max_bytes`` gate on the exact
    committed size, discarding the blob if the cache raced over the cap
    (413) → write ``manifest.json`` alongside.

    An empty prefix cache (disabled or 0 entries) is a valid export: it
    returns 200 with ``entries_exported=0`` and still writes a manifest so
    a peer can inspect the (empty) blob's provenance.

    H-02/H-12: the resolved ``destination`` is logged server-side only; the
    200 body carries counters + the caller-relative manifest filename, never
    the ``$HOME``-rooted absolute path.
    """
    destination = _resolve_or_400(req.destination)
    engine = _engine_or_503()
    cache = _prefix_cache(engine)

    # Cheap pre-write size gate. ``_current_memory`` is the live ledger the
    # cache maintains on every store/evict — reading it costs nothing and
    # lets us reject an over-cap export before touching the disk (the issue's
    # H-04 "never start a write you can't afford" concern).
    #
    # #1100 codex round 4 (#1): we no longer sample ``len(cache)`` here to
    # tell an empty export from a failed save — that pre-write snapshot
    # raced a concurrent store/evict on the step thread. Instead the
    # serialized ``save_to_disk`` op records an AUTHORITATIVE tri-state
    # outcome on the cache (``_last_save_outcome`` ∈ empty/committed/failed)
    # that we read AFTER the future resolves (happens-after the write). The
    # only pre-write value we still need is the live footprint for the CHEAP
    # 413 pre-gate ("never start a write you can't afford", the issue's
    # H-04 concern) — an over-cap approximation is fine because the precise
    # post-write gate against the real committed on-disk size is the
    # authority (#1100 BLOCKING-2/4).
    current_bytes = 0
    if cache is not None:
        current_bytes = int(getattr(cache, "_current_memory", 0))
    if req.max_bytes is not None and current_bytes > req.max_bytes:
        logger.info(
            "cache/export: rejected — cache footprint %d B exceeds max_bytes %d B "
            "(destination=%s)",
            current_bytes,
            req.max_bytes,
            destination,
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"cache footprint {current_bytes} bytes exceeds max_bytes "
                f"{req.max_bytes}"
            ),
        )

    # #1100 codex round 3 (#2): serialize the ENTIRE save→manifest→size-
    # gate→publish transaction per destination. Without this, two concurrent
    # exports to the same path interleave: one writes a manifest describing
    # the other's snapshot, or the over-cap cleanup of one deletes the newer
    # snapshot the other just committed. The lock makes ``destination`` a
    # single-writer resource; distinct destinations still run concurrently.
    #
    # #1100 codex round 4 (#5): layer an advisory cross-PROCESS ``flock`` under
    # the in-process lock so two SEPARATE rapid-mlx instances exporting to a
    # SHARED destination can't collide on the fixed ``.new``/``.old`` staging
    # paths. Order matters: take the process-local lock FIRST (cheap, always
    # works), then the filesystem lock — so a single instance's queued requests
    # never each block a worker thread on ``flock`` acquisition.
    async with _dest_lock(destination), _InterProcessLock(destination) as _iplock:
        # #1100 codex round 5 (#2): if the cross-process lock couldn't be
        # acquired (flock-less shared FS), refuse rather than risk two
        # instances corrupting the shared staging dirs — unless the operator
        # opted into single-instance-unsafe mode.
        _reject_if_ipc_lock_degraded(_iplock)
        # Snapshot to disk. Routed through the mlx-step worker thread inside
        # the engine (that's where the KV arrays are materializable). Returns
        # True if at least one entry committed; False for an empty / no-op
        # cache. Run in a threadpool so the asyncio loop isn't blocked by the
        # (potentially multi-GB) write.
        # #1100 codex round 4 (#1/#2): get the AUTHORITATIVE save outcome as a
        # RETURN VALUE computed in the SAME step-thread task as the save — not
        # by reading a cache-global ``_last_save_outcome`` field on the asyncio
        # thread, which a concurrent export to ANOTHER destination (distinct
        # lock) could overwrite in the gap between op and read. Outcome ∈:
        #   "committed" → >=1 entry landed + atomically published
        #   "empty"     → the cache genuinely held 0 entries (legit no-op)
        #   "failed"    → had entries but nothing committed (staging vanished,
        #                 post-write verify dropped all, rename never landed)
        # ``save_cache_with_outcome`` is declared on ``BaseEngine`` (route-layer
        # contract) so every real engine has it — no ``hasattr`` guard (the
        # #500 silent-skip shape the route-contract test forbids).
        # #1100 codex round 8 (#4): a BatchedEngine whose INNER engine hasn't
        # started raises ``EngineNotReadyError`` here rather than masking as an
        # empty snapshot — map it to the same 503 an absent outer engine gets so
        # "engine not loaded" is one consistent signal.
        try:
            outcome_obj = await anyio.to_thread.run_sync(
                engine.save_cache_with_outcome, str(destination)
            )
        except EngineNotReadyError as exc:
            logger.warning("cache/export: engine not ready — %s", exc)
            raise HTTPException(
                status_code=503, detail=_ENGINE_NOT_LOADED_DETAIL
            ) from exc
        save_outcome = outcome_obj.outcome
        saved = save_outcome == "committed"

        # A non-empty cache whose save committed NOTHING is a FAILURE, not an
        # empty export (#1100 BLOCKING-3). #1100 codex round 3 (#2): the
        # failing save uses ``save_to_disk``'s own ``.new``/``.old`` staging
        # and does NOT touch the published destination unless its atomic
        # rename succeeds — so on a non-committing save we DELIBERATELY do NOT
        # ``_discard_export`` the destination (that would destroy a
        # previously-valid snapshot this failed save never modified). We only
        # sweep the orphaned ``.new``/``.old`` staging dirs it may have left.
        if save_outcome == "failed":
            _sweep_staging_dirs(destination)
            logger.error(
                "cache/export: save FAILED — cache had entries but nothing "
                "committed to disk (destination=%s); staging dirs swept, "
                "any prior published snapshot left intact",
                destination,
            )
            raise HTTPException(
                status_code=500,
                detail="cache export failed to commit any entry to disk",
            )

        # #1100 codex round 3 (#1): an EMPTY export (no entries committed) to a
        # destination that already holds a STALE prior snapshot must not
        # silently re-export the old entries. ``build_manifest_from_engine_
        # state`` reads counts from the committed ``index.json`` — a leftover
        # one would make an "empty" export report the old blob. Clear the
        # destination's blob artifacts first so the manifest honestly reports
        # 0/0 and no importable stale entry files remain. (Only reached when
        # the save committed nothing AND the cache was legitimately empty.)
        if save_outcome != "committed":
            try:
                _discard_export(destination)
            except _ExportDiscardError as exc:
                logger.error("cache/export: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "empty cache export could not clear a stale prior "
                        "snapshot at the destination"
                    ),
                ) from exc

        # Everything from here to publish is wrapped so that if manifest write
        # or the size gate raises (e.g. ENOSPC after a multi-GB commit — codex
        # round 3 #3), the just-committed blob is discarded before the error
        # propagates, rather than orphaned to worsen disk exhaustion on retry.
        try:
            # Build the manifest from the COMMITTED on-disk index (#1100
            # BLOCKING-3), not the live ledger — ``cache_dir=destination``
            # makes the counters match what a peer will actually load.
            #
            # #1100 codex round 6 (#2): when the save reported ``"committed"``
            # (>=1 entry atomically published), a readable committed
            # ``index.json`` MUST exist — pass ``require_committed_index=True``
            # so a torn/unreadable index raises ``CommittedIndexUnreadableError``
            # (caught below → discard + 500) instead of silently falling back to
            # the live ledger and publishing a manifest for a non-importable
            # snapshot. An EMPTY export legitimately has no index (0 entries),
            # so we require it only on the committed path.
            manifest = build_manifest_from_engine_state(
                engine,
                cache_dir=destination,
                require_committed_index=(save_outcome == "committed"),
            )

            # Write the manifest BEFORE the size gate so the committed-blob
            # measure below includes manifest.json — it's part of what a peer
            # imports, so it must count against the cap (#1100 BLOCKING-4).
            write_manifest(destination, manifest)

            # #1100 BLOCKING-2 + BLOCKING-4: precise post-write size
            # enforcement against the ACTUAL committed on-disk footprint. The
            # pre-write gate reads the live logical ledger, which (a) can
            # undercount if the cache grows between that check and the snapshot
            # the step thread takes, and (b) counts only logical
            # ``memory_bytes`` — excluding tokens.bin, index.json,
            # manifest.json, and safetensors serialization overhead.
            # ``_committed_dir_size`` sums the real file sizes now on disk.
            committed_bytes = _committed_dir_size(destination)
        except HTTPException:
            raise
        except CommittedIndexUnreadableError as exc:
            # #1100 codex round 6 (#2): the save said "committed" but its
            # ``index.json`` is missing/torn — the snapshot is NOT importable.
            # Discard the blob and 500 rather than publish a manifest that lies
            # about a non-loadable export.
            logger.error(
                "cache/export: committed save left an unreadable index "
                "(destination=%s): %s; discarding the blob",
                destination,
                exc,
            )
            try:
                _discard_export(destination)
            except _ExportDiscardError as discard_exc:
                logger.error(
                    "cache/export: could not discard blob after unreadable "
                    "committed index: %s",
                    discard_exc,
                )
            raise HTTPException(
                status_code=500,
                detail="cache export committed but its on-disk index was unreadable",
            ) from exc
        except Exception as exc:
            # Post-save processing failed (manifest write ENOSPC, etc.). Do NOT
            # leave the just-committed blob orphaned on disk — discard it, then
            # surface a 500. Discard is best-effort here (already in an error
            # path); a quarantine failure is logged, not re-raised over the
            # original fault.
            logger.error(
                "cache/export: post-save processing failed (destination=%s): "
                "%s; discarding the just-committed blob",
                destination,
                exc,
            )
            try:
                _discard_export(destination)
            except _ExportDiscardError as discard_exc:
                logger.error(
                    "cache/export: could not discard blob after post-save failure: %s",
                    discard_exc,
                )
            raise HTTPException(
                status_code=500,
                detail="cache export failed during manifest/size finalization",
            ) from exc

        if req.max_bytes is not None and committed_bytes > req.max_bytes:
            try:
                _discard_export(destination)
            except _ExportDiscardError as exc:
                # #1100 BLOCKING-5: a 413 MUST guarantee the blob is not
                # importable. If we couldn't delete OR quarantine it, we can't
                # honor that invariant — 500 instead of a 413 that lies.
                logger.error("cache/export: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "cache export exceeded max_bytes but the oversized "
                        "blob could not be discarded"
                    ),
                ) from exc
            logger.info(
                "cache/export: post-write reject — committed on-disk %d B "
                "exceeds max_bytes %d B; blob discarded (destination=%s)",
                committed_bytes,
                req.max_bytes,
                destination,
            )
            raise HTTPException(
                status_code=413,
                detail=(
                    f"cache footprint {committed_bytes} bytes exceeds max_bytes "
                    f"{req.max_bytes}"
                ),
            )

        logger.info(
            "cache/export: wrote %d entries (%d logical B, %d on-disk B, "
            "saved=%s) to destination=%s",
            manifest.entries,
            manifest.total_bytes,
            committed_bytes,
            saved,
            destination,
        )
    return ExportResponse(
        protocol_version=PROTOCOL_VERSION,
        entries_exported=manifest.entries,
        bytes_written=manifest.total_bytes,
        model_id=manifest.model_id,
        quantization=manifest.quantization,
        paged_cache=manifest.paged_cache,
        turboquant_kv=manifest.turboquant_kv,
        manifest_path="manifest.json",
    )


@router.post("/import", response_model=ImportResponse)
async def import_cache(req: ImportRequest):
    """Import a peer instance's export into the local engine.

    Flow: resolve the source under the sandbox (403 on escape) → read the
    manifest (404 missing / 400 malformed) → reject a protocol-version
    mismatch, a manifest model_id that differs from the loaded model
    (unconditional), or a caller ``expected_model_id`` mismatch (409) →
    require a loaded engine (503) → hydrate via
    ``EngineCore.load_cache_from_disk`` (``merge_strategy="replace"`` clears
    the cache atomically inside that step-thread load).

    ``entries_skipped`` is ``manifest.entries - entries_loaded`` floored at
    0: the loader drops entries that fail per-entry validation (truncated
    safetensors, cache-type incompatible under the current quant config —
    see ``MemoryAwarePrefixCache.load_from_disk``), so a caller can tell a
    partial import from a clean one without re-reading the blob.
    """
    source = _resolve_or_400(req.source)

    # #1100 codex round 5 (#1/#4): run the checks that need NEITHER the lock nor
    # the engine FIRST — before acquiring ``_InterProcessLock`` (which creates
    # ``<source>.txlock`` + parent dirs). This means:
    #   * a MISSING source 404s WITHOUT creating any lockfile / parent dir
    #     (round-4 #4's parent-creation would otherwise let a stream of unique
    #     nonexistent import sources consume unbounded dirs/inodes), and
    #   * a caller-side ``expected_model_id`` / protocol mismatch returns 409
    #     even when no engine is loaded — the documented contract. The round-4
    #     restructure moved ``_engine_or_503()`` ahead of the 409 gate, so a
    #     mismatch wrongly 503'd on an engine-less server.
    manifest = _read_manifest_or_http(source)

    if manifest.protocol_version != req.expected_protocol_version:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "protocol_version",
                    req.expected_protocol_version,
                    manifest.protocol_version,
                )
            ),
        )

    # Caller-side identity assertion — independent of the loaded engine, so it
    # must gate BEFORE ``_engine_or_503`` (a mismatch is 409, not 503).
    if req.expected_model_id is not None and manifest.model_id != req.expected_model_id:
        raise HTTPException(
            status_code=409,
            detail=str(
                ManifestMismatchError(
                    "model_id", req.expected_model_id, manifest.model_id
                )
            ),
        )

    # Now hold the SAME per-destination transaction lock export uses, keyed on
    # the resolved source path, from a RE-READ of the manifest through
    # completion of the engine load (#1100 codex round 4 #4). Without it a
    # concurrent export to this same path could swap ``index.json`` in the
    # window between validation and load — so we'd validate model_id/protocol
    # against manifest A but hydrate the KV blob of manifest B. The manifest is
    # re-read inside the lock so the identity/protocol facts we gate the LOAD on
    # reflect the blob actually on disk under the lock, not the pre-lock read.
    async with _dest_lock(source), _InterProcessLock(source) as _iplock:
        # #1100 codex round 5 (#2): refuse to import without cross-process
        # exclusion on a shared FS (a concurrent export could swap the blob
        # mid-load) unless the operator opted into single-instance-unsafe mode.
        _reject_if_ipc_lock_degraded(_iplock)
        manifest = _read_manifest_or_http(source)

        # Re-check protocol under the lock (a concurrent export could have
        # swapped in a differently-versioned blob after the pre-lock read).
        if manifest.protocol_version != req.expected_protocol_version:
            raise HTTPException(
                status_code=409,
                detail=str(
                    ManifestMismatchError(
                        "protocol_version",
                        req.expected_protocol_version,
                        manifest.protocol_version,
                    )
                ),
            )

        # Resolve the loaded engine BEFORE the SERVER-identity gate so we can
        # derive its model id the SAME way the manifest builder does (#1100
        # BLOCKING-2). The gate previously read ``get_config().model_name``,
        # which is empty for an embedded / unit-test engine that never
        # populated ServerConfig — but ``build_manifest_from_engine_state``
        # falls back to ``engine.config.model_name`` (populated), so an
        # embedded engine's manifest carried a real id while the gate compared
        # against "" and skipped the check, letting it import ANOTHER model's
        # KV state. ``resolve_engine_model_id`` is the shared source of truth
        # both call sites use so they can't drift again.
        engine = _engine_or_503()

        # #1100 BLOCKING-1: unconditionally reject a manifest whose model_id
        # does not match the model THIS server loaded. KV cache is model-
        # specific (layer/head/dim geometry, quant layout) — loading another
        # model's blob corrupts inference or crashes the fetch. The caller's
        # ``expected_model_id`` (already checked above) is an ADDITIONAL,
        # caller-side assertion; omitting it must NOT disable server-side
        # identity checking. The 409 detail names NEITHER the manifest's nor the
        # server's model id — a bearer-token holder shouldn't be able to probe
        # what this server runs by diffing mismatch messages; the caller can
        # read its own manifest (or GET /v1/cache/info) to see the id it shipped.
        server_model_id = resolve_engine_model_id(engine)
        if server_model_id:
            if manifest.model_id != server_model_id:
                raise HTTPException(
                    status_code=409,
                    detail=("manifest model_id does not match the loaded engine model"),
                )
        else:
            # #1100 codex round 3 (#4) → round 7 (#1): FAIL CLOSED — HARD —
            # when the loaded engine's model id cannot be resolved. The KV blob
            # is model-specific (layer/head/dim geometry, quant layout); loading
            # a foreign one corrupts inference or crashes the fetch, so the
            # server MUST positively confirm the blob matches the model it
            # actually loaded before importing.
            #
            # The round-3 fix let an id-less engine import if the caller pinned
            # ``expected_model_id == manifest.model_id`` — but BOTH of those are
            # CALLER-CONTROLLED and UNTRUSTED (the caller ships the manifest AND
            # picks expected_model_id), so that check compares an attacker's
            # value against the attacker's own manifest and proves NOTHING about
            # the geometry of the model this server loaded. It reduced to "trust
            # the caller", the exact hole the gate exists to close. There is no
            # trusted server-side signal to compare against here, so we reject
            # UNCONDITIONALLY. An operator who hits this must give the engine a
            # resolvable model id (serve with a named model / populate
            # ServerConfig.model_name) so the server can make the identity
            # assertion itself. 422 = "server cannot verify identity" (distinct
            # from the 409 "identities mismatch" above).
            logger.warning(
                "cache/import: rejected — loaded engine model id is "
                "unresolvable; refusing to import a model-specific KV blob the "
                "server cannot verify against its loaded model (source=%s)",
                source,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "cannot verify model identity: the loaded engine has no "
                    "resolvable model id, so the server cannot confirm this KV "
                    "cache matches the loaded model; serve with a named model "
                    "so the identity can be verified"
                ),
            )

        # Re-check the caller-side ``expected_model_id`` against the RE-READ
        # manifest (a concurrent export could have swapped the blob's model_id
        # after the pre-lock check). Cheap and keeps the load honest.
        if (
            req.expected_model_id is not None
            and manifest.model_id != req.expected_model_id
        ):
            raise HTTPException(
                status_code=409,
                detail=str(
                    ManifestMismatchError(
                        "model_id", req.expected_model_id, manifest.model_id
                    )
                ),
            )

        # #1100 codex round 9 (#1): KV-cache GEOMETRY gate. Matching model_id is
        # necessary but NOT sufficient — a blob exported under a different KV
        # dtype is byte-incompatible with the loaded engine even for the SAME
        # model, and hydrating it corrupts inference or crashes the fetch. The
        # manifest records the cache knobs; gate the import against the loaded
        # engine's actual geometry (read via the SAME helper the manifest
        # builder uses, so builder and gate can't drift).
        #
        # ``quantization`` (``kv_cache_dtype``) is the highest-signal axis — a
        # different dtype means a different tensor layout, guaranteed
        # corruption. It is the ONLY axis we hard-gate, and ONLY when BOTH sides
        # carry a KNOWN (non-empty) value: an empty string is "unknown" (a
        # legacy manifest predating this field, or an unresolvable scheduler),
        # and rejecting unknown-vs-known would break importing older-but-valid
        # exports — the manifest contract is additive/back-compat. The
        # ``paged_cache`` / ``turboquant_kv`` booleans default to False on an old
        # manifest, so a bare bool compare would false-positive against a
        # paged/turbo server; they are recorded for provenance but NOT hard-
        # gated here (a fuller fingerprint is a tracked follow-up). The 409 body
        # names the axis but not the values (no config-probe oracle).
        server_quant, _server_paged, _server_turbo = resolve_engine_cache_geometry(
            engine
        )
        if (
            manifest.quantization
            and server_quant
            and manifest.quantization != server_quant
        ):
            logger.warning(
                "cache/import: rejected — KV cache quantization mismatch (source=%s)",
                source,
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    "manifest KV-cache quantization does not match the loaded "
                    "engine (kv_cache_dtype); the exported cache is incompatible "
                    "with this server's cache configuration"
                ),
            )

        # merge_strategy="replace": the in-memory cache is cleared ATOMICALLY
        # inside the step-thread load — ``engine.load_cache_from_disk(...,
        # replace=True)`` forwards to ``MemoryAwarePrefixCache.load_from_disk``,
        # which clears only AFTER index.json is read + validated, on the same
        # mlx-step thread that runs the entry-load loop. #1100 BLOCKING-4: the
        # old code cleared here on the ASYNCIO thread before the step-thread
        # load, so (a) a corrupt/missing source destroyed the existing cache
        # before we knew loading would fail, and (b) a concurrent request could
        # ``store`` into the cache in the gap between clear and load. Pushing
        # the clear into the load call closes both. We still deliberately do
        # NOT use ``scheduler.deep_reset()`` (it would abort in-flight
        # requests). clear() carries the monotonic Prometheus counters over, so
        # replace preserves load_skipped / save_drift_drops.
        replace = req.merge_strategy == "replace"

        # #1100 BLOCKING-5: report the bytes THIS import actually loaded, not
        # the manifest's full ``total_bytes`` (which overstates when entries
        # are skipped).
        cache = _prefix_cache(engine)

        # Hydrate from disk. #1100 codex round 4 (#2/#3): the entries count AND
        # the loaded-byte total are computed in the SAME step-thread task and
        # returned as ONE ``LoadResult`` — the byte total is summed under the
        # cache lock over the entries THIS load installed (0 for an empty load
        # or an aborted replace). Returning it as a value (rather than reading
        # a cache-global ``_last_load_bytes`` field on the asyncio thread)
        # closes the cross-path race where a concurrent load to another
        # destination could clobber the field between op and read. ``replace``
        # is positional to fit ``anyio.to_thread.run_sync``'s *args.
        # ``load_cache_with_result`` is a ``BaseEngine`` route-layer-contract
        # method — present on every real engine, so no ``hasattr`` guard.
        # #1100 codex round 8 (#4): absent inner engine → ``EngineNotReadyError``
        # → 503 (not a 200 reporting a zero-entry "success").
        try:
            load_result = await anyio.to_thread.run_sync(
                engine.load_cache_with_result, str(source), replace
            )
        except EngineNotReadyError as exc:
            logger.warning("cache/import: engine not ready — %s", exc)
            raise HTTPException(
                status_code=503, detail=_ENGINE_NOT_LOADED_DETAIL
            ) from exc
        entries_loaded = load_result.entries
        bytes_loaded = load_result.bytes_loaded

    entries_skipped = max(0, manifest.entries - entries_loaded)
    logger.info(
        "cache/import: loaded %d/%d entries (skipped=%s, %d B, merge=%s) "
        "from source=%s",
        entries_loaded,
        manifest.entries,
        entries_skipped,
        bytes_loaded,
        req.merge_strategy,
        source,
    )
    return ImportResponse(
        protocol_version=PROTOCOL_VERSION,
        entries_loaded=entries_loaded,
        entries_skipped=entries_skipped,
        bytes_loaded=bytes_loaded,
    )


@router.get("/info")
async def cache_info(path: str | None = None):
    """Read the manifest at a whitelisted export root.

    Returns the manifest dict so callers (peer instances, oai-mlx, ops
    tooling) can GC / route / version-gate without paying a full import.
    Path resolution follows the same sandbox rules as export/import.

    H-12: pre-fix this handler echoed the resolved sandbox root back to
    the caller in a top-level ``"path"`` field. ``str(root)`` expands to
    ``/Users/<USERNAME>/.cache/rapid-mlx/cache_exports/<sub>`` on macOS
    — same operator home-dir / username disclosure that H-02 fixed on
    the 403 envelope. Same treatment here: keep the resolved root in
    the server log only, omit it from the wire envelope. Callers that
    need to dedupe by location already have the request-side ``path``
    they supplied.
    """
    root = _resolve_or_400(path)
    manifest = _read_manifest_or_http(root)

    # Codex r1 follow-up: log at DEBUG (not INFO) so the resolved root
    # only lands in operator logs when the operator explicitly opts in
    # (RAPID_MLX_LOG_LEVEL=DEBUG or equivalent). Routine 200 traffic
    # carries no path on the wire AND no path in the default log stream
    # — but the breadcrumb is still there for ops who need to debug a
    # peer-sync issue. Sibling concern: H-02's logger.warning on the
    # 403 path is fine because that's an anomaly worth recording at
    # default verbosity, whereas every successful info read shouldn't
    # rewrite the sandbox path into the rolling log.
    logger.debug(
        "cache/info: resolved root=%s model_id=%s entries=%s",
        root,
        manifest.model_id,
        manifest.entries,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "manifest": manifest.to_dict(),
    }
