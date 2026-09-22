# SPDX-License-Identifier: Apache-2.0
"""Best-effort telemetry-v2 inference and capability emitters."""

from __future__ import annotations

import atexit
import os
import queue
import threading
from collections.abc import AsyncIterable, AsyncIterator, Callable
from functools import partial
from typing import Any
from urllib.parse import urlsplit

from rapid_mlx.telemetry import model_id, redact, registry, store
from rapid_mlx.telemetry import track as track_module

_MODEL_TYPES = frozenset(
    {
        "llm",
        "vlm",
        "embedding",
        "image-gen",
        "video-gen",
        "text-diffusion",
        "audio",
        "other",
    }
)

_MAX_PENDING = 64
_QUEUE: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=_MAX_PENDING)
_WORKER: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()
_SHUTTING_DOWN = False
_AT_FORK_INSTALLED = globals().get("_AT_FORK_INSTALLED", False)
_ATEXIT_INSTALLED = globals().get("_ATEXIT_INSTALLED", False)


def _worker_main(work_queue: queue.Queue[Callable[[], None]]) -> None:
    """Run queued writes serially on a daemon that never owns process exit."""
    while True:
        callback = work_queue.get()
        try:
            callback()
        except Exception:
            pass
        finally:
            work_queue.task_done()


def _ensure_worker() -> bool:
    """Lazily start the process-local daemon worker."""
    global _WORKER
    with _WORKER_LOCK:
        if _SHUTTING_DOWN:
            return False
        if _WORKER is not None and _WORKER.is_alive():
            return True
        try:
            worker = threading.Thread(
                target=_worker_main,
                args=(_QUEUE,),
                name="rapid-mlx-inference-telemetry",
                daemon=True,
            )
            worker.start()
        except Exception:
            return False
        _WORKER = worker
        return True


def _submit(callback: Callable[[], None]) -> bool:
    """Submit without blocking; drop when the bounded lane is saturated."""
    if not _ensure_worker():
        return False
    try:
        _QUEUE.put_nowait(callback)
    except queue.Full:
        return False
    return True


def _drop_pending_at_exit() -> None:
    """Discard queued writes without waiting for a blocked daemon worker."""
    global _SHUTTING_DOWN
    _SHUTTING_DOWN = True
    while True:
        try:
            _QUEUE.get_nowait()
        except queue.Empty:
            return
        else:
            _QUEUE.task_done()


def _after_fork_child() -> None:
    """Forget inherited thread state; only the forking thread survives."""
    global _QUEUE, _WORKER, _WORKER_LOCK, _SHUTTING_DOWN
    _QUEUE = queue.Queue(maxsize=_MAX_PENDING)
    _WORKER = None
    _WORKER_LOCK = threading.Lock()
    _SHUTTING_DOWN = False


def _install_lifecycle_hooks() -> None:
    global _ATEXIT_INSTALLED, _AT_FORK_INSTALLED
    if not _ATEXIT_INSTALLED:
        atexit.register(_drop_pending_at_exit)
        _ATEXIT_INSTALLED = True
    if not _AT_FORK_INSTALLED and hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork_child)
        _AT_FORK_INSTALLED = True


_install_lifecycle_hooks()


def model_type_token(source: object | None) -> str:
    """Classify a resolved engine/profile into the registry vocabulary."""
    try:
        if source is None:
            return "other"
        if isinstance(source, str):
            return source if source in _MODEL_TYPES else "other"

        modality = getattr(source, "modality", None)
        if modality == "text":
            return "vlm" if getattr(source, "supports_image_input", False) else "llm"
        if isinstance(modality, str) and modality in _MODEL_TYPES - {
            "llm",
            "vlm",
            "other",
        }:
            return modality
        if getattr(source, "is_image_gen", False):
            return "image-gen"
        if getattr(source, "is_video_gen", False):
            return "video-gen"
        if getattr(source, "is_embedding", False):
            return "embedding"
        if getattr(source, "is_audio", False):
            return "audio"
        return "vlm" if getattr(source, "is_mllm", False) else "llm"
    except Exception:
        return "other"


def _record_completed_request(
    *,
    model: str,
    endpoint: str,
    caller_agent: str | None,
    caller_client: str | None,
    result: str,
) -> None:
    """Worker-thread half of :func:`emit_completed_request`.

    Callers pass the resolved telemetry model id, never the request's model
    field. Every operation is best effort; no exception can escape the worker.
    """
    try:
        safe_model = model_id.telemetry_model_id(model)
        try:
            endpoint_path = urlsplit(endpoint).path
        except (TypeError, ValueError):
            endpoint_path = ""
        allowed_endpoints = registry.load_registry()["enums"]["endpoint"]["values"]
        safe_endpoint = endpoint_path if endpoint_path in allowed_endpoints else "other"
        caller = redact.normalize_caller_agent(caller_agent, caller_client)
        allowed_callers = registry.load_registry()["enums"]["caller"]["values"]
        if caller not in allowed_callers:
            caller = "other"
        outcome = result if result in ("ok", "failed") else "failed"
        crossing = store.record(f"inf|{safe_model}|{safe_endpoint}|{caller}|{outcome}")
        if crossing is not None:
            track_module.track(
                "inference_bucket_reached",
                {
                    "model": safe_model,
                    "endpoint": safe_endpoint,
                    "caller": caller,
                    "result": outcome,
                    "count_bucket": crossing.bucket,
                    "bucket_source": crossing.bucket_source,
                },
            )
        if outcome == "ok":
            track_module.emit_active_day()
    except Exception:
        return


def emit_completed_request(
    *,
    model: str,
    endpoint: str,
    caller_agent: str | None,
    caller_client: str | None,
    result: str,
) -> None:
    """Gate, then enqueue one completed-request update without blocking.

    The official-build and live-consent checks intentionally happen before
    submitting work. Ineligible processes therefore create no telemetry state.
    SQLite and capture work always run outside the request coroutine.
    """
    try:
        if not track_module._upload_allowed():
            return
        _submit(
            partial(
                _record_completed_request,
                model=model,
                endpoint=endpoint,
                caller_agent=caller_agent,
                caller_client=caller_client,
                result=result,
            )
        )
    except Exception:
        return


async def emit_failed_on_stream_error(
    source: AsyncIterable[Any],
    *,
    model: str,
    endpoint: str,
    caller_agent: str | None,
    caller_client: str | None,
    failure_latch: list[bool] | None = None,
) -> AsyncIterator[Any]:
    """Forward a generation stream and count only non-cancellation failures."""
    try:
        async for item in source:
            yield item
    except Exception:
        if failure_latch is not None:
            failure_latch[:] = [True]
        emit_completed_request(
            model=model,
            endpoint=endpoint,
            caller_agent=caller_agent,
            caller_client=caller_client,
            result="failed",
        )
        raise


def emit_capability_rejected(capability: str, *, model_type: str = "other") -> None:
    """Enqueue one closed-vocabulary capability rejection without blocking."""
    try:
        if not track_module._upload_allowed():
            return
        _submit(
            partial(
                _record_capability_rejected,
                capability=capability,
                model_type=model_type,
            )
        )
    except Exception:
        return


def _record_capability_rejected(*, capability: str, model_type: str) -> None:
    """Worker-thread half of :func:`emit_capability_rejected`."""
    try:
        allowed = registry.load_registry()["enums"]["capability"]["values"]
        if capability not in allowed:
            return
        track_module.track(
            "capability_rejected",
            {
                "capability": capability,
                "model_type": model_type_token(model_type),
            },
        )
    except Exception:
        return


def request_caller_headers(request: Any | None) -> tuple[str | None, str | None]:
    """Return the raw caller headers shared by every inference route."""
    try:
        if request is None:
            return None, None
        return (
            request.headers.get("user-agent"),
            request.headers.get("x-rapid-client"),
        )
    except Exception:
        return None, None
