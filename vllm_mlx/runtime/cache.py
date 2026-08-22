# SPDX-License-Identifier: Apache-2.0
"""Prefix cache persistence — load/save KV cache to disk."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

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

# Bump whenever persisted KV semantics change in a way the safetensors schema
# alone cannot detect. Version 2 closes a release-blocking corruption found by
# the v0.12.19 dogfood: a cache written for an older checkpoint / KV dtype was
# structurally loadable but produced token-id-0-style garbage after restart.
_PREFIX_CACHE_NAMESPACE_VERSION = 2


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
        # #1111 codex r3: the STARTUP auto-load must NOT protect reloaded
        # entries. ``save_to_disk`` persists ALL live entries — including
        # opportunistic (unprotected) non-trimmable ones — so protecting them on
        # every boot would grow the protected set ~N per restart and defeat the
        # ``hybrid_reuse_max_entries`` cap. ``protected_import=False`` makes
        # reloaded non-trimmable entries obey the retention bound at commit;
        # only the EXPLICIT ``POST /v1/cache/import`` (#476) pins its entries.
        loaded = cfg.engine.load_cache_from_disk(d, protected_import=False)
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
    # The KV entries are authoritative. A prior shutdown may have failed
    # to serialize every cache entry while leaving an older ``radix.index``
    # behind, so accepting the radix solely because its JSON parses can
    # create terminal keys with no corresponding KV state.
    try:
        with cache._lock:  # noqa: SLF001 — coordinated rebuild
            keys = list(cache._entries.keys())  # noqa: SLF001
        radix_path = os.path.join(cache_dir, "radix.index")
        loaded_radix = radix.load(radix_path)
        radix_matches_entries = (
            loaded_radix
            and len(radix) == len(keys)
            and all(key in radix for key in keys)
        )
        if not radix_matches_entries:
            radix.rebuild_from_keys(keys)
            logger.info(f"[radix] rebuilt index from {len(keys)} loaded cache entries")
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(f"[radix] rebuild_from_keys failed: {e}", exc_info=True)


def _resolve_scheduler(engine):
    """Return the engine's ``Scheduler`` across the two engine shapes.

    Two live engine layouts expose the scheduler at DIFFERENT depths:

    * A bare ``EngineCore`` (unit tests, embedded use) has
      ``engine.scheduler`` directly.
    * The production ``BatchedEngine`` does NOT — its ``._engine`` is an
      ``AsyncEngineCore`` wrapper whose inner ``EngineCore`` holds the
      real scheduler, i.e. ``engine._engine.engine.scheduler``. The old
      ``getattr(engine, "scheduler", None)`` lookup silently returned
      ``None`` under ``BatchedEngine``, so both radix-index persistence
      AND the #476 cache-export path were no-ops in production. The same
      unwrap already lives in ``engine/batched.py:894`` for the LLM
      admission gate — this mirrors it as the single source of truth.

    Every access is ``getattr(..., None)``-guarded so a genuinely foreign
    engine (third-party, partially-built) yields ``None`` rather than
    raising — the None-graceful contract callers already rely on.
    """
    # Direct-EngineCore shape.
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is not None:
        return scheduler
    # BatchedEngine shape: engine._engine (AsyncEngineCore) → .engine
    # (EngineCore) → .scheduler. Mirrors engine/batched.py:894.
    wrapper = getattr(engine, "_engine", None)
    inner = getattr(wrapper, "engine", None) if wrapper is not None else None
    return getattr(inner, "scheduler", None)


def _resolve_memory_aware_cache(engine):
    """Return the engine's ``MemoryAwarePrefixCache`` if present.

    Resolves the scheduler via :func:`_resolve_scheduler` (which handles
    both the bare-``EngineCore`` and wrapped-``BatchedEngine`` shapes)
    then reads ``memory_aware_cache`` off it. Returning ``None`` means
    "no radix / prefix-cache surface available", which the callers treat
    as the hash-index path.
    """
    scheduler = _resolve_scheduler(engine)
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
        if saved:
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
    # 16 hex chars of SHA-256 (64 bits) make accidental semantic namespace
    # collisions negligible even across large fleets and long-lived caches.
    # Persisted KV tensors are reusable only for the exact model revision and
    # effective KV dtype that produced them. Tensor shape/type validation cannot
    # prove that semantic identity, so keep those axes in the directory key.
    identity = _semantic_cache_identity(cfg, raw)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    leaf = f"{safe_name}--{digest}"
    # ~/.cache/rapid-mlx/ (was ~/.cache/vllm-mlx/ pre-rename). The cache is
    # best-effort and silently rebuilds, so the moved location just costs a
    # one-time recompute; any stale ~/.cache/vllm-mlx/ dir is inert and safe
    # to delete.
    return os.path.join(
        os.path.expanduser("~"), ".cache", "rapid-mlx", "prefix_cache", leaf
    )


def _cached_model_revision(model_name: str) -> str:
    """Return a stable, network-free identity for the selected weights."""
    source = _resolved_model_source(model_name)
    candidate = Path(source).expanduser()
    if candidate.exists():
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.is_dir():
            digest = hashlib.sha256(str(resolved).encode("utf-8"))
            # Custom model code can read arbitrary checkpoint-local assets, so
            # an extension allowlist cannot define semantic identity safely.
            # Build a complete regular-file manifest in one traversal. Every
            # file gets replacement-sensitive metadata; reasonably small files
            # also get a byte hash without forcing startup to stream huge
            # weight shards from disk.
            tracked = [path for path in resolved.rglob("*") if path.is_file()]
            for path in sorted(tracked):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                try:
                    relative = path.relative_to(resolved)
                except ValueError:
                    continue
                digest.update(str(relative).encode("utf-8"))
                digest.update(_file_identity(stat).encode("ascii"))
                if stat.st_size <= 8 * 1024 * 1024:
                    try:
                        digest.update(path.read_bytes())
                    except OSError:
                        pass
            return f"local-{digest.hexdigest()[:16]}"
        try:
            stat = resolved.stat()
            return f"local-file-{_file_identity(stat)}"
        except OSError:
            return str(resolved)
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(source, "config.json")
        if isinstance(cached, str):
            path = Path(cached)
            if path.parent.parent.name == "snapshots":
                return path.parent.name
    except Exception:
        # Prefix persistence is best-effort and must not make startup depend on
        # optional Hugging Face cache metadata.
        pass
    return source


def _semantic_cache_identity(cfg, raw_model_name: str) -> str:
    """Capture immutable cache identity for one loaded engine lifetime."""
    engine = getattr(cfg, "engine", None)
    attr = "_rapid_mlx_prefix_cache_identity"
    if engine is not None:
        captured = getattr(engine, attr, None)
        if isinstance(captured, str) and captured:
            return captured

    kv_dtype = _effective_kv_cache_dtype(cfg)
    revision = _cached_model_revision(raw_model_name)
    identity = (
        f"{raw_model_name}\0prefix-cache-v{_PREFIX_CACHE_NAMESPACE_VERSION}"
        f"\0kv={kv_dtype}\0revision={revision}"
    )
    if engine is not None:
        try:
            setattr(engine, attr, identity)
        except (AttributeError, TypeError):
            # Foreign immutable engine wrappers still get a safe identity for
            # this call; production BatchedEngine supports the lifetime pin.
            pass
    return identity


def pin_prefix_cache_identity(
    engine, *, raw_model_name: str, checkpoint_source: str, kv_dtype: str
) -> str:
    """Pin cache identity before a loaded engine becomes concurrently visible."""
    revision = _cached_model_revision(checkpoint_source)
    identity = (
        f"{raw_model_name}\0prefix-cache-v{_PREFIX_CACHE_NAMESPACE_VERSION}"
        f"\0kv={kv_dtype}\0revision={revision}"
    )
    engine._rapid_mlx_prefix_cache_identity = identity
    return identity


def _file_identity(stat: os.stat_result) -> str:
    """Metadata identity that changes on content replacement.

    ``ctime_ns`` cannot be restored with ``utime`` after an in-place write, and
    ``st_ino`` changes for atomic replace deployments. Together they cover the
    same-size/preserved-mtime case without reading tens of GB of weights during
    every server boot.
    """
    return f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_ino}"


def _effective_kv_cache_dtype(cfg) -> str:
    """Read the canonical live scheduler dtype, falling back pre-load."""
    engine = getattr(cfg, "engine", None)
    scheduler = _resolve_scheduler(engine) if engine is not None else None
    scheduler_cfg = getattr(scheduler, "config", None)
    live = getattr(scheduler_cfg, "kv_cache_dtype", None)
    return str(live or getattr(cfg, "kv_cache_dtype", None) or "bf16")


def _resolved_model_source(model_name: str) -> str:
    """Resolve a built-in/user alias to the checkpoint source it names."""
    try:
        from ..model_aliases import resolve_model

        return resolve_model(model_name) or model_name
    except Exception:
        return model_name
