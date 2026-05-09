# SPDX-License-Identifier: Apache-2.0
"""Prefix cache persistence — load/save KV cache to disk."""

from __future__ import annotations

import hashlib
import logging
import os

from ..config import get_config

logger = logging.getLogger(__name__)


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


def save_prefix_cache_to_disk() -> None:
    """Save prefix cache to disk during shutdown."""
    cfg = get_config()
    if cfg.engine is None:
        return
    try:
        d = get_cache_dir()
        logger.info(f"[lifespan] Saving prefix cache to {d}")
        saved = cfg.engine.save_cache_to_disk(d)
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
    return os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache", leaf
    )
