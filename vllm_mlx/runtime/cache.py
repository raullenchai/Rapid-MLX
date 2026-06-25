# SPDX-License-Identifier: Apache-2.0
"""Prefix cache persistence — load/save KV cache to disk."""

from __future__ import annotations

import hashlib
import logging
import os
import time

from ..config import get_config

logger = logging.getLogger(__name__)

# SIGTERM-grace budget for the shutdown flush. Downstream supervisors
# (rapid-desktop / launchd / systemd / Docker) typically send SIGTERM
# then SIGKILL ~5-10s later if the process hasn't exited. The previous
# synchronous flush could run for tens of seconds on multi-GB caches
# and was consistently truncated mid-write, leaving ``<cache_dir>.new/``
# orphaned and losing the KV-cache hit on the next launch. The default
# of 3.5s is the largest value that still leaves enough room under a
# 5s SIGTERM grace for ``engine.stop()`` + telemetry session_end +
# uvicorn's own teardown to finish before SIGKILL. Override with the
# ``RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET`` env var (seconds, float;
# ``0`` disables the deadline and restores the old "flush everything"
# behavior — useful for offline CLI saves where no signal is coming).
_DEFAULT_SHUTDOWN_BUDGET_SEC = 3.5

# Headroom reserved after the per-entry loop exits for the atomic
# rename + ``index.json`` write + stale ``.old`` cleanup. Without it a
# perfectly-budgeted save could finish a write at T = deadline and then
# get SIGKILL'd during the commit — leaving ``cache_dir.new/`` orphaned
# anyway (the exact failure mode this whole gate exists to prevent).
# 400 ms is comfortably above the observed commit cost across all
# entry-count fixtures + leaves ~600 ms of slack under a 5 s SIGTERM
# grace for ``engine.stop()`` and uvicorn teardown.
_COMMIT_HEADROOM_SEC = 0.4


def _shutdown_budget_sec() -> float:
    raw = os.environ.get("RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET")
    if raw is None:
        return _DEFAULT_SHUTDOWN_BUDGET_SEC
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            f"[lifespan] invalid RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET={raw!r}, "
            f"falling back to default {_DEFAULT_SHUTDOWN_BUDGET_SEC}s"
        )
        return _DEFAULT_SHUTDOWN_BUDGET_SEC


def load_prefix_cache_from_disk() -> None:
    """Load prefix cache from disk during startup.

    R15-P1 (task #303): when the engine exposes a ``memory_aware_cache``
    with a radix index attached, we also try to load
    ``<cache_dir>/radix.index``. A missing or corrupt radix.index is
    NOT fatal — the index is silently rebuilt from the cache's loaded
    entries (the entries are the source of truth; the radix is a lookup
    accelerator).
    """
    cfg = get_config()
    if cfg.engine is None:
        return
    try:
        d = get_cache_dir()
        logger.info(f"[lifespan] Loading prefix cache from {d}")
        loaded = cfg.engine.load_cache_from_disk(d)
        if loaded > 0:
            logger.info(f"[lifespan] Loaded {loaded} prefix cache entries")
        else:
            logger.info("[lifespan] No prefix cache entries found on disk")
        _load_radix_index_after_cache(cfg.engine, d)
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


def _load_radix_index_after_cache(engine, cache_dir: str) -> None:
    """Best-effort radix-index restore + rebuild fallback.

    Order of operations:

    1. If the engine's scheduler doesn't have a memory-aware cache with
       a radix attached, no-op (we're on the hash path).
    2. If ``<cache_dir>/radix.index`` exists and parses cleanly, the
       radix populates from it. Cheap — just a JSON read + insert loop.
    3. If load fails (missing file on first boot after upgrade, version
       mismatch, JSON corruption), rebuild the radix from the keys
       already loaded into ``_entries``. This costs O(sum(len(tokens)))
       which is tiny relative to the model load that just ran.
    """
    cache = _resolve_memory_aware_cache(engine)
    if cache is None:
        return
    radix = getattr(cache, "_radix_index", None)
    if radix is None:
        return
    radix_path = os.path.join(cache_dir, "radix.index")
    if radix.load(radix_path):
        return
    # Fallback: reconstruct from currently loaded entries. Reads
    # ``_entries.keys()`` under the cache's lock to stay coherent with
    # any concurrent store/evict — though boot is single-threaded so
    # this is belt-and-suspenders.
    try:
        with cache._lock:  # noqa: SLF001 — coordinated rebuild
            keys = list(cache._entries.keys())  # noqa: SLF001
        if keys:
            radix.rebuild_from_keys(keys)
            logger.info(f"[radix] rebuilt index from {len(keys)} loaded cache entries")
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(f"[radix] rebuild_from_keys failed: {e}", exc_info=True)


def _resolve_memory_aware_cache(engine):
    """Return the engine's ``MemoryAwarePrefixCache`` if present.

    Walks the engine.scheduler.memory_aware_cache chain defensively —
    external engines (third-party, ``BatchedEngine`` wrappers, etc.)
    may not expose either attribute. Returning ``None`` means "no radix
    surface available", which is the same as the hash-index path.
    """
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None:
        return None
    return getattr(scheduler, "memory_aware_cache", None)


def save_prefix_cache_to_disk(budget_sec: float | None = None) -> None:
    """Save prefix cache to disk during shutdown.

    Runs against a wall-clock budget (default
    :data:`_DEFAULT_SHUTDOWN_BUDGET_SEC`, overridable via the
    ``RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET`` env var). When the
    deadline is reached the per-entry loop inside
    ``MemoryAwarePrefixCache.save_to_disk`` stops and the partial
    snapshot is committed via the same atomic rename as a full flush —
    so we never leave the staging ``<cache_dir>.new/`` directory orphaned
    when SIGKILL eventually lands. A budget of ``0`` (or a negative
    value) disables the deadline entirely.
    """
    cfg = get_config()
    if cfg.engine is None:
        return
    if budget_sec is None:
        budget_sec = _shutdown_budget_sec()
    should_abort = _make_should_abort(budget_sec) if budget_sec > 0 else None
    try:
        d = get_cache_dir()
        if should_abort is not None:
            logger.info(
                f"[lifespan] Saving prefix cache to {d} "
                f"(shutdown budget {budget_sec:.1f}s, "
                f"commit headroom {_COMMIT_HEADROOM_SEC:.1f}s)"
            )
        else:
            logger.info(f"[lifespan] Saving prefix cache to {d} (no shutdown budget)")
        saved = _call_save_cache_to_disk(cfg.engine, d, should_abort)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
        # R15-P1 (task #303): radix-index persistence runs AFTER the
        # entry-cache commit so a torn shutdown can never leave a
        # ``radix.index`` referencing entries that didn't make it to
        # disk. The radix is a best-effort accelerator — if this fails,
        # the next boot just rebuilds from ``_entries``.
        _save_radix_index_after_cache(cfg.engine, d)
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


def _save_radix_index_after_cache(engine, cache_dir: str) -> None:
    """Best-effort radix-index persistence."""
    cache = _resolve_memory_aware_cache(engine)
    if cache is None:
        return
    radix = getattr(cache, "_radix_index", None)
    if radix is None:
        return
    try:
        radix.save(os.path.join(cache_dir, "radix.index"))
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(f"[radix] save failed: {e}", exc_info=True)


def _make_should_abort(budget_sec: float):
    """Build a forward-looking deadline predicate.

    Returns a callable ``predicate(predicted_sec=0.0)`` that returns
    ``True`` when starting an operation of ``predicted_sec`` duration
    would push wall-clock past ``deadline - _COMMIT_HEADROOM_SEC``.

    The forward-looking shape is what codex flagged on PR #667 round 1:
    the previous predicate ``time.monotonic() >= deadline`` only fired
    BEFORE an entry's write started, so a single ``save_prompt_cache``
    call running past the budget would still get SIGKILL'd mid-write
    and leave ``cache_dir.new/`` orphaned — the exact failure this PR
    claims to fix. Callers (currently ``MemoryAwarePrefixCache.save_to
    _disk``) pass an estimated duration for the NEXT operation and the
    predicate decides whether to start it or commit-what-we-have.
    """
    deadline = time.monotonic() + budget_sec
    safe_deadline = deadline - _COMMIT_HEADROOM_SEC

    def predicate(predicted_sec: float = 0.0) -> bool:
        return time.monotonic() + predicted_sec >= safe_deadline

    return predicate


def _call_save_cache_to_disk(engine, cache_dir: str, should_abort):
    """Invoke ``engine.save_cache_to_disk`` with backwards-compat fallback.

    Internal engines (``BatchedEngine``, ``EngineCore``, ``Scheduler``)
    all accept the ``should_abort`` kwarg as of this PR, but external
    or third-party engine implementations may still expose the legacy
    one-argument signature. Without the fallback the kwarg would raise
    ``TypeError`` and the entire save would be lost — strictly worse
    than no-deadline persistence.

    Detection is signature-based (``inspect.signature``) rather than
    catch-and-retry-on-TypeError: codex PR #667 round 2 flagged that a
    compatible engine raising ``TypeError`` mid-execution with the
    ``should_abort`` substring would cause an unintended SECOND call
    via the legacy path, doubling any side effects (writes / index
    increments / metric counters). Inspecting the signature up front
    has zero chance of misclassifying an internal exception as a
    signature mismatch.
    """
    import inspect

    try:
        sig = inspect.signature(engine.save_cache_to_disk)
    except (TypeError, ValueError):
        # Builtin / C-extension methods may not expose a Python
        # signature. Conservatively call the deadline-aware path —
        # the engine almost certainly accepts the kwarg if it's been
        # updated. We don't fall back here because a fallback retry
        # is exactly the double-call hazard codex flagged.
        return engine.save_cache_to_disk(cache_dir, should_abort=should_abort)

    accepts_should_abort = "should_abort" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_should_abort:
        return engine.save_cache_to_disk(cache_dir, should_abort=should_abort)

    logger.warning(
        "[lifespan] engine.save_cache_to_disk does not accept "
        "should_abort kwarg — calling legacy signature "
        "(no deadline awareness for this engine)"
    )
    return engine.save_cache_to_disk(cache_dir)


def get_cache_dir() -> str:
    """Get cache persistence directory based on actual model path.

    The model name comes from CLI / config and is interpolated into a
    filesystem path, so it must not contain path-traversal sequences.
    HF repo names don't permit ``..`` today, but ``--model`` and
    ``--served-model-name`` are arbitrary user input — sanitize
    defensively (issue #194).

    Sanitization can collapse different model names to the same leaf
    (e.g. ``a/b`` and ``a--b`` both become ``a--b``; ``..`` and
    ``.default`` both fall back to ``default``). To keep prefix-cache
    entries from cross-contaminating, append a short stable hash of
    the *original* model identifier so distinct names always map to
    distinct directories. Benign HF names that didn't need
    sanitization gain the hash suffix too — invalidates pre-#194
    on-disk caches one time, but the loader's persistence path is
    best-effort and will silently rebuild them.
    """
    cfg = get_config()
    model_name = cfg.model_path or cfg.model_name or "default"
    raw = str(model_name)
    safe_name = (
        raw.replace("/", "--").replace("\\", "--").replace("..", "--").lstrip(".")
    ) or "default"
    # 8 hex chars of SHA-256 — 32 bits, collision-resistant for the
    # tens-of-models-per-user scale we'd ever see in practice.
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    leaf = f"{safe_name}--{digest}"
    # ~/.cache/rapid-mlx/ (was ~/.cache/vllm-mlx/ pre-rename). The cache is
    # best-effort and silently rebuilds, so the moved location just costs a
    # one-time recompute; any stale ~/.cache/vllm-mlx/ dir is inert and safe
    # to delete.
    return os.path.join(
        os.path.expanduser("~"), ".cache", "rapid-mlx", "prefix_cache", leaf
    )
