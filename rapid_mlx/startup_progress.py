# SPDX-License-Identifier: Apache-2.0
"""Best-effort startup observations; never a source of readiness decisions.

Bytes describe materialized model parameters, not downloaded checkpoint bytes.
Only the first eligible loader reports a denominator: auxiliary/draft models,
retries and specialized loaders remain indeterminate rather than resetting it.
"""

from __future__ import annotations

import inspect

import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

_Result = TypeVar("_Result")


class StartupProgress:
    _phases = (
        "starting",
        "resolving",
        "loading_weights",
        "initializing",
        "warming_up",
        "ready",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._phase = "starting"
        self._model = ""
        self._claimed = False
        self._total = 0
        self._done = 0

    def begin(self, model: str) -> None:
        """Begin a new server startup, before any worker is dispatched."""
        with self._lock:
            self._active = True
            self._phase = "resolving"
            # Never expose a local checkpoint path on an unauthenticated probe.
            self._model = str(model).rstrip("/").rsplit("/", 1)[-1]
            self._claimed = False
            self._total = self._done = 0

    def phase(self, phase: str) -> None:
        rank = self._phases.index(phase)
        with self._lock:
            if self._active and rank >= self._phases.index(self._phase):
                self._phase = phase
                if phase == "ready":
                    self._active = False

    def claim_weights(self) -> bool:
        with self._lock:
            if (
                not self._active
                or self._claimed
                or self._phases.index(self._phase)
                > self._phases.index("loading_weights")
            ):
                return False
            self._claimed = True
            self._phase = "loading_weights"
            return True

    def weights_total(self, total: int) -> None:
        with self._lock:
            if self._active and self._phase == "loading_weights" and not self._total:
                self._total = max(0, total)

    def weights_done(self, done: int) -> None:
        with self._lock:
            if self._active and self._phase == "loading_weights":
                self._done = min(self._total, max(self._done, done))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {"phase": self._phase}
            if self._model:
                result["model"] = self._model
            if self._phase == "loading_weights" and self._total:
                result["progress"] = {
                    "done": self._done,
                    "total": self._total,
                    "unit": "bytes",
                }
            return result


startup_progress = StartupProgress()
_EVAL_BATCH_BYTES = 1024**3


def _accepts_lazy(loader: Callable[..., Any]) -> bool:
    # Only loaders with an explicit ``lazy`` keyword can defer evaluation.
    try:
        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return False
    return "lazy" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def load_with_progress(
    loader: Callable[..., _Result], *args: Any, **kwargs: Any
) -> _Result:
    """Replace an eager loader's final eval with bounded, observed evals.

    Called only at loaders with an explicit lazy contract. No global MLX hooks,
    extra reads, or work on another Metal thread. Disk-stream/lazy callers and
    loads outside server startup keep their original call and behavior.
    """
    if (
        kwargs.get("lazy")
        or not _accepts_lazy(loader)
        or not startup_progress.claim_weights()
    ):
        return loader(*args, **kwargs)

    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
    except ImportError:
        startup_progress.phase("initializing")
        return loader(*args, **kwargs)

    try:
        result = loader(*args, **{**kwargs, "lazy": True})
        model = cast(Any, result)[0]
        # Tied parameters can appear more than once; count/evaluate each once.
        arrays = {id(value): value for _, value in tree_flatten(model.parameters())}
        startup_progress.weights_total(sum(value.nbytes for value in arrays.values()))
        done = pending_bytes = 0
        pending = []
        for value in arrays.values():
            pending.append(value)
            pending_bytes += value.nbytes
            if pending_bytes >= _EVAL_BATCH_BYTES:
                mx.eval(pending)
                done += pending_bytes
                startup_progress.weights_done(done)
                pending = []
                pending_bytes = 0
        if pending:
            mx.eval(pending)
            startup_progress.weights_done(done + pending_bytes)
        return result
    finally:
        # Tokenizers, adapters, MTP/draft setup and fallback loaders have no
        # shared byte denominator. Never leave a stale 100% bar over that work.
        startup_progress.phase("initializing")
