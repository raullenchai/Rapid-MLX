# SPDX-License-Identifier: Apache-2.0
"""
Model Registry for rapid-mlx.

Tracks which models are in use by which engines to prevent
BatchKVCache conflicts when multiple engines share a model.

The problem: mlx-lm's BatchGenerator maintains internal KV cache state
tied to the model. When multiple EngineCore instances use the same model,
the cache objects become incompatible, causing NoneType errors.

Solution: Track model ownership and ensure only one engine's BatchGenerator
is active for each model at a time.
"""

import collections
import logging
import threading
import weakref
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelOwnershipError(Exception):
    """Raised when a model is already owned by another engine."""

    pass


class ModelRegistry:
    """
    Global registry tracking model ownership.

    Uses weak references to automatically clean up when engines
    are garbage collected.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        """Initialize registry state."""
        # model_id -> (weak_ref_to_engine, engine_id)
        self._owners: dict[int, tuple[weakref.ref, str]] = {}
        self._registry_lock = threading.Lock()
        self._pending_releases: collections.deque[tuple[int, str, int | None]] = (
            collections.deque()
        )

    def acquire(
        self, model: Any, engine: Any, engine_id: str, force: bool = False
    ) -> bool:
        """
        Attempt to acquire ownership of a model.

        Args:
            model: The MLX model
            engine: The EngineCore instance
            engine_id: Unique identifier for the engine
            force: If True, forcibly take ownership (resets previous owner)

        Returns:
            True if ownership acquired

        Raises:
            ModelOwnershipError: If model is owned and force=False
        """
        model_id = id(model)

        with self._registry_lock:
            self._drain_pending_releases_locked()
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                owner = weak_ref()

                if owner is not None and owner_id != engine_id:
                    if force:
                        # Reset the previous owner's scheduler
                        logger.warning(
                            f"Model ownership transfer: {owner_id} -> {engine_id}"
                        )
                        self._reset_owner(owner)
                    else:
                        raise ModelOwnershipError(
                            f"Model is already owned by engine {owner_id}. "
                            f"Use force=True or call release() on the "
                            f"previous engine first."
                        )

            # Register new owner
            self._owners[model_id] = (weakref.ref(engine), engine_id)
            logger.debug(f"Engine {engine_id} acquired model {model_id}")
            self._drain_pending_releases_locked()
            return True

    def release_from_finalizer(
        self, model: Any, engine_id: str, engine: Any = None
    ) -> None:
        """Queue a release without blocking: finalizers can run at any allocation
        point, including inside acquire's registry critical section.
        """
        self._pending_releases.append(
            (id(model), engine_id, id(engine) if engine is not None else None)
        )
        if self._registry_lock.acquire(blocking=False):
            try:
                self._drain_pending_releases_locked()
            finally:
                self._registry_lock.release()

    def _drain_pending_releases_locked(self) -> None:
        """Process finalizer releases while holding the registry lock."""
        while True:
            try:
                model_id, engine_id, engine_obj_id = self._pending_releases.popleft()
            except IndexError:
                break
            entry = self._owners.get(model_id)
            if entry is not None:
                weak_ref, owner_id = entry
                target = weak_ref()
                # Weakrefs survive __del__ on refcount deallocation, but are cleared
                # during cyclic GC; equal live object IDs identify the same object.
                if owner_id == engine_id and (
                    target is None
                    or (engine_obj_id is not None and id(target) == engine_obj_id)
                ):
                    del self._owners[model_id]
                    logger.debug(f"Engine {engine_id} released model {model_id}")

    def release(self, model: Any, engine_id: str) -> bool:
        """
        Release ownership of a model.

        Args:
            model: The MLX model
            engine_id: The engine releasing ownership

        Returns:
            True if ownership was released
        """
        model_id = id(model)

        with self._registry_lock:
            self._drain_pending_releases_locked()
            if model_id in self._owners:
                _, owner_id = self._owners[model_id]
                if owner_id == engine_id:
                    del self._owners[model_id]
                    logger.debug(f"Engine {engine_id} released model {model_id}")
                    return True
        return False

    def is_owned(self, model: Any) -> tuple[bool, str | None]:
        """
        Check if a model is owned.

        Returns:
            Tuple of (is_owned, owner_engine_id)
        """
        model_id = id(model)

        with self._registry_lock:
            self._drain_pending_releases_locked()
            if model_id in self._owners:
                weak_ref, owner_id = self._owners[model_id]
                if weak_ref() is not None:
                    return (True, owner_id)
                else:
                    # Owner was garbage collected, clean up
                    del self._owners[model_id]
        return (False, None)

    def _reset_owner(self, owner: Any) -> None:
        """Reset the scheduler of a previous owner."""
        try:
            if hasattr(owner, "scheduler"):
                owner.scheduler.deep_reset()
        except Exception as e:
            logger.warning(f"Failed to reset previous owner: {e}")

    def cleanup(self) -> int:
        """
        Clean up stale entries (owners that were garbage collected).

        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        with self._registry_lock:
            self._drain_pending_releases_locked()
            stale = [
                model_id
                for model_id, (weak_ref, _) in self._owners.items()
                if weak_ref() is None
            ]
            for model_id in stale:
                del self._owners[model_id]
                cleaned += 1
        if cleaned:
            logger.debug(f"Cleaned up {cleaned} stale registry entries")
        return cleaned

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        with self._registry_lock:
            self._drain_pending_releases_locked()
            active = sum(1 for _, (ref, _) in self._owners.items() if ref() is not None)
            return {
                "total_entries": len(self._owners),
                "active_owners": active,
            }


# Global singleton
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry
