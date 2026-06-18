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
    """Load prefix cache from disk during startup."""
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
    except Exception as e:
        logger.warning(f"[lifespan] Failed to load cache from disk: {e}", exc_info=True)


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
    deadline = time.monotonic() + budget_sec if budget_sec > 0 else None
    should_abort = (
        (lambda dl=deadline: time.monotonic() >= dl) if deadline is not None else None
    )
    try:
        d = get_cache_dir()
        if deadline is not None:
            logger.info(
                f"[lifespan] Saving prefix cache to {d} "
                f"(shutdown budget {budget_sec:.1f}s)"
            )
        else:
            logger.info(f"[lifespan] Saving prefix cache to {d} (no shutdown budget)")
        saved = cfg.engine.save_cache_to_disk(d, should_abort=should_abort)
        if saved:
            logger.info(f"[lifespan] Saved prefix cache to {d}")
        else:
            logger.info("[lifespan] No cache to save")
    except Exception as e:
        logger.warning(f"[lifespan] Failed to save cache to disk: {e}", exc_info=True)


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
