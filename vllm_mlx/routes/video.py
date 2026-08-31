# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible video jobs backed by MLX-native video pipelines."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ..middleware.auth import _verify_api_key_values, verify_api_key
from ..model_aliases import resolve_profile

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_REFERENCE_BYTES = 20 * 1024 * 1024
_VIDEO_REQUEST_BYTES = _MAX_REFERENCE_BYTES + 1024 * 1024
_MAX_JOBS = 100
_MAX_PIXEL_FRAMES = 768 * 512 * 97
_MAX_REFERENCE_PIXELS = 16_777_216
_VIDEO_JOB_SCHEMA_VERSION = 1
_VIDEO_JOB_METADATA = "job.json"
_MAX_VIDEO_JOB_METADATA_BYTES = 64 * 1024
_VIDEO_ID_RE = re.compile(r"video_[0-9a-f]{32}\Z")
_jobs_lock = threading.Lock()
_ephemeral_jobs_roots = {Path(tempfile.mkdtemp(prefix="rapid-mlx-videos-"))}
_jobs_root = next(iter(_ephemeral_jobs_roots))
_jobs_are_persistent = False


@dataclass
class _VideoJob:
    id: str
    model: str
    prompt: str
    seconds: str
    size: str
    status: str = "queued"
    progress: int = 0
    created_at: int = 0
    completed_at: int | None = None
    error: dict[str, str] | None = None
    output_path: str | None = None
    generation_finished: bool = False

    def public(self) -> dict:
        value = asdict(self)
        value.pop("output_path")
        value.pop("generation_finished")
        value["object"] = "video"
        return value


_jobs: dict[str, _VideoJob] = {}
_tasks: dict[str, asyncio.Task] = {}
_cleanup_tasks: set[asyncio.Task] = set()
_generation_threads: set[threading.Thread] = set()
_persistence_threads: set[threading.Thread] = set()
_accepting_jobs = True
_generation_gates: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Lock
] = weakref.WeakKeyDictionary()
_generation_gates_lock = threading.Lock()


class _VideoBodyTooLargeError(Exception):
    """Internal ASGI sentinel for a streamed multipart body over the cap."""


class VideoBodyLimitMiddleware:
    """Authenticate and cap video multipart bodies before FastAPI parses them."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/videos"
        ):
            return await self.app(scope, receive, send)

        headers = {name.lower(): value for name, value in scope.get("headers", ())}
        authorization = headers.get(b"authorization", b"").decode(
            "latin-1", errors="ignore"
        )
        scheme, _, token = authorization.partition(" ")
        bearer = token if scheme.lower() == "bearer" and token else None
        try:
            _verify_api_key_values(bearer)
        except HTTPException as exc:
            response = JSONResponse(
                status_code=exc.status_code, content={"detail": exc.detail}
            )
            return await response(scope, receive, send)

        advertised = headers.get(b"content-length")
        if advertised is not None:
            try:
                advertised_bytes = int(advertised.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                advertised_bytes = None
            if advertised_bytes is not None and advertised_bytes > _VIDEO_REQUEST_BYTES:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "video request body exceeds 21 MB"},
                )
                return await response(scope, receive, send)

        total = 0
        limit_tripped = False
        replacement_sent = False
        downstream_started = False
        downstream_completed = False

        async def bounded_receive():
            nonlocal total, limit_tripped
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                total += len(chunk)
                if total > _VIDEO_REQUEST_BYTES:
                    limit_tripped = True
                    raise _VideoBodyTooLargeError
            return message

        async def guarded_send(message):
            nonlocal replacement_sent, downstream_started, downstream_completed
            if limit_tripped:
                # Starlette's ServerErrorMiddleware catches the receive
                # sentinel and attempts to emit a 500 before re-raising it.
                # Replace that response at the same ASGI boundary, and swallow
                # its remaining frames, so the client sees exactly one 413.
                if not downstream_started and not replacement_sent:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "video request body exceeds 21 MB"},
                    )
                    await response(scope, receive, send)
                    replacement_sent = True
                return

            message_type = message.get("type")
            if message_type == "http.response.start":
                downstream_started = True
            elif message_type == "http.response.body" and not message.get(
                "more_body", False
            ):
                downstream_completed = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except _VideoBodyTooLargeError:
            if replacement_sent or downstream_completed:
                return
            if downstream_started:
                # The video route parses its multipart body before entering the
                # handler, so this should be unreachable. If a future handler
                # starts a response first, it is too late to change the status;
                # terminate the stream instead of emitting a second response.
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
                return
            response = JSONResponse(
                status_code=413,
                content={"detail": "video request body exceeds 21 MB"},
            )
            await response(scope, receive, send)


def install_video_body_limit_middleware(app) -> None:
    app.add_middleware(VideoBodyLimitMiddleware)


def _generation_gate_for_current_loop() -> asyncio.Lock:
    """Return the serial generation gate owned by the active event loop."""
    loop = asyncio.get_running_loop()
    with _generation_gates_lock:
        gate = _generation_gates.get(loop)
        if gate is None:
            gate = asyncio.Lock()
            _generation_gates[loop] = gate
        return gate


def _cleanup_jobs() -> None:
    for root in tuple(_ephemeral_jobs_roots):
        shutil.rmtree(root, ignore_errors=True)


atexit.register(_cleanup_jobs)


def configure_video_jobs(output_dir: str | Path | None) -> Path:
    """Select the job store before the server lifecycle starts.

    ``None`` preserves the historical process-temporary behavior. An explicit
    directory opts into completed-job persistence; the directory is resolved
    once so a later working-directory change cannot redirect cleanup or writes.
    """
    global _jobs_are_persistent, _jobs_root

    persistent = output_dir is not None
    if persistent:
        raw_output_dir = str(output_dir).strip()
        if not raw_output_dir:
            raise ValueError("video output directory must not be empty")
        root = Path(raw_output_dir).expanduser().resolve(strict=False)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Fail before a large model download when the selected store cannot be
        # written. NamedTemporaryFile uses 0600 and removes the probe on close.
        with tempfile.NamedTemporaryFile(prefix=".rapid-mlx-write-", dir=root):
            pass
    else:
        root = Path(tempfile.mkdtemp(prefix="rapid-mlx-videos-"))
        _ephemeral_jobs_roots.add(root)

    with _jobs_lock:
        active_tasks = [task for task in _tasks.values() if not task.done()]
        active_cleanup = [task for task in _cleanup_tasks if not task.done()]
        if (
            active_tasks
            or active_cleanup
            or _generation_threads
            or _persistence_threads
        ):
            raise RuntimeError("cannot reconfigure the video job store while jobs run")
        _jobs.clear()
        _tasks.clear()
        _jobs_root = root
        _jobs_are_persistent = persistent
    return root


def _completed_job_record(job: _VideoJob) -> dict:
    return {
        "schema_version": _VIDEO_JOB_SCHEMA_VERSION,
        **job.public(),
    }


def _video_directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:  # pragma: no cover - POSIX guard
        raise OSError("secure video directory access is unavailable")
    return int(os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0))


def _open_video_job_directory(video_id: str) -> tuple[int, int]:
    """Open the configured root and one owned job directory without links."""
    root_fd = os.open(_jobs_root, _video_directory_open_flags())
    try:
        job_fd = os.open(
            video_id,
            _video_directory_open_flags(),
            dir_fd=root_fd,
        )
    except BaseException:
        os.close(root_fd)
        raise
    return root_fd, job_fd


def _open_video_output(job_fd: int) -> int:
    """Open and validate a non-empty regular MP4 relative to its job dir."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:  # pragma: no cover - POSIX guard
        raise OSError("secure video output access is unavailable")
    output_fd = os.open(
        "output.mp4",
        os.O_RDONLY
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=job_fd,
    )
    try:
        output_stat = os.fstat(output_fd)
        if not stat.S_ISREG(output_stat.st_mode) or output_stat.st_size <= 0:
            raise OSError("video output is not a non-empty regular file")
    except BaseException:
        os.close(output_fd)
        raise
    return output_fd


def _persist_completed_job(job: _VideoJob) -> None:
    """Durably commit one completed MP4 and its atomic manifest."""
    root_fd, job_fd = _open_video_job_directory(job.id)
    temporary_name = f".job-{uuid.uuid4().hex}.json.tmp"
    try:
        output_fd = _open_video_output(job_fd)
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)

        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=job_fd,
        )
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as temporary:
            json.dump(
                _completed_job_record(job),
                temporary,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(
            temporary_name,
            _VIDEO_JOB_METADATA,
            src_dir_fd=job_fd,
            dst_dir_fd=job_fd,
        )
        # Persist the output/manifest entries, then the job directory's entry
        # in the configured root. Only after both barriers may the API expose
        # the job as completed.
        os.fsync(job_fd)
        os.fsync(root_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=job_fd)
        os.close(job_fd)
        os.close(root_fd)


def _load_completed_job(job_dir: Path) -> _VideoJob | None:
    if not _VIDEO_ID_RE.fullmatch(job_dir.name):
        return None
    output_path = job_dir / "output.mp4"
    root_fd = job_fd = metadata_fd = output_fd = None
    try:
        root_fd, job_fd = _open_video_job_directory(job_dir.name)
        metadata_fd = os.open(
            _VIDEO_JOB_METADATA,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=job_fd,
        )
        metadata_stat = os.fstat(metadata_fd)
        if (
            not stat.S_ISREG(metadata_stat.st_mode)
            or metadata_stat.st_size > _MAX_VIDEO_JOB_METADATA_BYTES
        ):
            return None
        with os.fdopen(metadata_fd, "rb") as metadata_source:
            metadata_fd = None
            raw_metadata = metadata_source.read(_MAX_VIDEO_JOB_METADATA_BYTES + 1)
        if len(raw_metadata) > _MAX_VIDEO_JOB_METADATA_BYTES:
            return None
        value = json.loads(raw_metadata.decode("utf-8"))
        output_fd = _open_video_output(job_fd)
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable video job record: %s", job_dir.name)
        return None
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if metadata_fd is not None:
            os.close(metadata_fd)
        if job_fd is not None:
            os.close(job_fd)
        if root_fd is not None:
            os.close(root_fd)

    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != _VIDEO_JOB_SCHEMA_VERSION:
        return None
    if value.get("object") != "video" or value.get("id") != job_dir.name:
        return None
    if value.get("status") != "completed" or value.get("progress") != 100:
        return None
    if value.get("error") is not None:
        return None

    model = value.get("model")
    prompt = value.get("prompt")
    seconds = value.get("seconds")
    size = value.get("size")
    created_at = value.get("created_at")
    completed_at = value.get("completed_at")
    if (
        not isinstance(model, str)
        or not model
        or len(model) > 2048
        or not isinstance(prompt, str)
        or not prompt
        or len(prompt) > 4096
        or not isinstance(seconds, str)
        or not seconds
        or len(seconds) > 32
        or not isinstance(size, str)
        or not size
        or len(size) > 64
        or type(created_at) is not int
        or type(completed_at) is not int
        or created_at < 0
        or completed_at < created_at
    ):
        return None

    return _VideoJob(
        id=job_dir.name,
        model=model,
        prompt=prompt,
        seconds=seconds,
        size=size,
        status="completed",
        progress=100,
        created_at=created_at,
        completed_at=completed_at,
        output_path=str(output_path),
        generation_finished=True,
    )


def _restore_completed_jobs() -> list[_VideoJob]:
    try:
        candidates = list(_jobs_root.iterdir())
    except OSError:
        logger.exception("Unable to scan the video job store")
        return []

    restored = [job for path in candidates if (job := _load_completed_job(path))]
    restored.sort(key=lambda item: (item.created_at, item.id), reverse=True)
    for expired in restored[_MAX_JOBS:]:
        shutil.rmtree(_jobs_root / expired.id, ignore_errors=True)
    return restored[:_MAX_JOBS]


def start_video_jobs() -> None:
    """Reset the route lifecycle when an application instance starts."""
    global _accepting_jobs
    restored = _restore_completed_jobs() if _jobs_are_persistent else []
    with _jobs_lock:
        _accepting_jobs = True
        if _jobs_are_persistent:
            # A lifespan restart must reflect the validated disk snapshot, not
            # merge it with stale failed/partial entries or artifacts removed
            # while the server was stopped.
            _jobs.clear()
            _jobs.update((job.id, job) for job in restored)


async def shutdown_video_jobs(timeout: float = 30.0) -> None:
    """Stop admission and drain video jobs within a bounded shutdown budget.

    MLX generation cannot be interrupted safely once a Metal graph is running.
    Queued jobs are cancelled immediately; an active job gets ``timeout``
    seconds to finish. Generation itself runs on a daemon thread, so exceeding
    the budget cannot keep the Python process alive indefinitely.
    """
    global _accepting_jobs
    loop = asyncio.get_running_loop()
    with _jobs_lock:
        _accepting_jobs = False
        cancelled_job_ids: list[str] = []
        current_tasks = [
            task
            for task in _tasks.values()
            if not task.done() and task.get_loop() is loop
        ]
        for job_id, task in list(_tasks.items()):
            job = _jobs.get(job_id)
            if task in current_tasks and job is not None and job.status == "queued":
                job.status = "failed"
                job.error = {
                    "code": "video_server_shutdown",
                    "message": "Video generation was cancelled during server shutdown.",
                }
                job.generation_finished = True
                cancelled_job_ids.append(job_id)
                task.cancel()

    if current_tasks:
        _, pending = await asyncio.wait(current_tasks, timeout=max(0.0, timeout))
        if pending:
            logger.warning(
                "Video shutdown budget expired with %d active generation job(s); "
                "detaching daemon MLX worker(s)",
                len(pending),
            )
            with _jobs_lock:
                pending_set = set(pending)
                for job_id, task in _tasks.items():
                    if task in pending_set:
                        job = _jobs.get(job_id)
                        if job is not None and not (
                            job.generation_finished and job.output_path is not None
                        ):
                            job.status = "failed"
                            job.error = {
                                "code": "video_server_shutdown",
                                "message": (
                                    "Video generation was cancelled during "
                                    "server shutdown."
                                ),
                            }
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    for job_id in cancelled_job_ids:
        await asyncio.to_thread(shutil.rmtree, _jobs_root / job_id, ignore_errors=True)


async def _persist_completed_job_in_daemon_thread(job: _VideoJob) -> None:
    """Persist a manifest without allowing slow storage to block shutdown.

    Unlike generation, manifest persistence is safe to detach after the MP4 is
    complete: the atomic temporary-file protocol leaves either the previous
    complete record or no record. A daemon thread lets the server honor its
    shutdown deadline even when an external volume stalls in ``fsync`` or
    ``replace``.
    """
    loop = asyncio.get_running_loop()
    completed = loop.create_future()
    store_root = _jobs_root

    def finish(result: BaseException | None) -> None:
        if completed.done():  # pragma: no cover - defensive duplicate callback
            return
        if result is None:
            completed.set_result(None)
        else:
            completed.set_exception(result)

    def target() -> None:
        result: BaseException | None
        try:
            _persist_completed_job(job)
        except Exception as exc:  # noqa: BLE001
            result = exc
        except BaseException:  # pragma: no cover - defensive thread boundary
            result = RuntimeError("video job metadata worker terminated unexpectedly")
        else:
            result = None
        finally:
            with _jobs_lock:
                _persistence_threads.discard(thread)
        expired_id: str | None = None
        if result is None:
            with _jobs_lock:
                if (
                    _jobs_are_persistent
                    and _jobs_root == store_root
                    and _accepting_jobs
                ):
                    # A new lifespan may have scanned while this detached write
                    # was still blocked. Register the now-durable result into
                    # that current registry instead of requiring another
                    # process restart to discover it.
                    _jobs[job.id] = replace(job)
                    if len(_jobs) > _MAX_JOBS:
                        oldest = min(
                            _jobs.values(), key=lambda item: (item.created_at, item.id)
                        )
                        _jobs.pop(oldest.id, None)
                        expired_id = oldest.id
        if expired_id is not None:
            shutil.rmtree(store_root / expired_id, ignore_errors=True)
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(finish, result)

    thread = threading.Thread(
        target=target, name="rapid-mlx-video-persistence", daemon=True
    )
    with _jobs_lock:
        _persistence_threads.add(thread)
    try:
        thread.start()
    except BaseException:
        with _jobs_lock:
            _persistence_threads.discard(thread)
        raise
    try:
        await asyncio.shield(completed)
    except asyncio.CancelledError:
        # If the server abandons this best-effort write at its shutdown
        # deadline, consume a late worker exception without holding shutdown.
        def consume_outcome(done: asyncio.Future) -> None:
            if not done.cancelled():
                done.exception()

        completed.add_done_callback(consume_outcome)
        raise


async def _run_in_generation_thread(function, /, **kwargs) -> None:
    """Run uncancellable MLX work without owning a non-daemon executor thread."""
    loop = asyncio.get_running_loop()
    completed = loop.create_future()

    def finish(result: BaseException | None) -> None:
        if completed.done():
            return
        if result is None:
            completed.set_result(None)
        else:
            completed.set_exception(result)

    def target() -> None:
        try:
            function(**kwargs)
        except Exception as exc:  # noqa: BLE001
            result = exc
        except BaseException:  # pragma: no cover - defensive thread boundary
            result = RuntimeError("video generation worker terminated unexpectedly")
        else:
            result = None
        finally:
            with _jobs_lock:
                _generation_threads.discard(thread)
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(finish, result)

    thread = threading.Thread(
        target=target, name="rapid-mlx-video-generation", daemon=True
    )
    with _jobs_lock:
        _generation_threads.add(thread)
    thread.start()
    await completed


def _video_engine():
    from ..config import get_config

    engine = get_config().engine
    if engine is None or not getattr(engine, "is_video_gen", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is not running a video model. Start it with "
                        "`rapid-mlx serve ltx-2.3-mlx-q4`."
                    ),
                    "type": "invalid_request_error",
                    "code": "video_model_not_loaded",
                    "param": "model",
                }
            },
        )
    return engine


def _parse_size(value: str, *, multiple: int = 64) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="size must be WIDTHxHEIGHT"
        ) from exc
    openai_sizes = {(1280, 720), (720, 1280)}
    is_model_aligned = width % multiple == 0 and height % multiple == 0
    if not (
        256 <= width <= 1920
        and 256 <= height <= 1920
        and (is_model_aligned or (width, height) in openai_sizes)
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"video width/height must be 256..1920 and divisible by {multiple}, "
                "or use 1280x720 / 720x1280"
            ),
        )
    return width, height


def _frame_count(seconds: int, fps: int = 24) -> int:
    requested = seconds * fps
    return max(9, round((requested - 1) / 8) * 8 + 1)


def _video_capabilities(engine) -> dict:
    """Describe the live video model using the route's validation contract."""
    family = getattr(engine, "video_family", "ltx-2.3")
    native_fps = 5 if family == "cogvideox-fun" else getattr(engine, "native_fps", 24)
    model_type = None
    max_area = None
    if family == "wan":
        wan_engine = getattr(engine, "_wan_engine", None)
        model_type = getattr(wan_engine, "model_type", None)
        max_area = getattr(wan_engine, "max_area", None)

    if family == "cogvideox-fun":
        modes = ["text-to-video"]
        size = {"type": "fixed", "values": ["672x384"]}
        seconds = {"minimum": 1, "maximum": 1, "default": 1}
        frames = {"minimum": 5, "maximum": 1201, "step": 4, "offset": 1}
    elif family == "wan":
        modes = {
            "t2v": ["text-to-video"],
            "i2v": ["image-to-video"],
            "ti2v": ["text-to-video", "image-to-video"],
        }.get(model_type, ["text-to-video", "image-to-video"])
        openai_sizes = [(1280, 720), (720, 1280)]
        supported_openai_sizes = [
            f"{width}x{height}"
            for width, height in openai_sizes
            if max_area is None
            or ((width + 63) // 64) * 64 * (((height + 63) // 64) * 64) <= max_area
        ]
        size = {
            "type": "range",
            "width": {"minimum": 256, "maximum": 1920, "multiple_of": 64},
            "height": {"minimum": 256, "maximum": 1920, "multiple_of": 64},
            "maximum_area": max_area,
            "also_supported": supported_openai_sizes,
        }
        seconds = {"minimum": 1, "maximum": 20, "default": 4}
        frames = {"minimum": 5, "maximum": 1201, "step": 4, "offset": 1}
    elif family == "ltx-2.5":
        modes = ["text-to-video", "image-to-video"]
        size = {
            "type": "range",
            "width": {"minimum": 256, "maximum": 1920, "multiple_of": 32},
            "height": {"minimum": 256, "maximum": 1920, "multiple_of": 32},
            "also_supported": ["1280x720", "720x1280"],
        }
        seconds = {"minimum": 1, "maximum": 20, "default": 4}
        frames = {"minimum": 9, "maximum": 1201, "step": 8, "offset": 1}
    else:
        modes = ["text-to-video", "image-to-video"]
        size = {
            "type": "range",
            "width": {"minimum": 256, "maximum": 1920, "multiple_of": 64},
            "height": {"minimum": 256, "maximum": 1920, "multiple_of": 64},
            "also_supported": ["1280x720", "720x1280"],
        }
        seconds = {"minimum": 1, "maximum": 20, "default": 4}
        frames = {"minimum": 9, "maximum": 1201, "step": 8, "offset": 1}

    return {
        "object": "video.capabilities",
        "model": engine.model_name,
        "modality": "video-gen",
        "family": family,
        "modes": modes,
        "limits": {
            "size": size,
            "seconds": seconds,
            "fps": {
                "minimum": native_fps if family == "wan" else 1,
                "maximum": native_fps if family == "wan" else 60,
                "default": native_fps,
                "fixed": family == "wan",
            },
            "frames": frames,
            "workload": {
                "metric": "pixel_frames",
                "maximum": _MAX_PIXEL_FRAMES,
                "dimension_rounding": (
                    "none"
                    if family == "cogvideox-fun"
                    else ("ceil_to_32" if family == "ltx-2.5" else "ceil_to_64")
                ),
            },
            "input_reference": {
                "maximum_bytes": _MAX_REFERENCE_BYTES,
                "maximum_pixels": _MAX_REFERENCE_PIXELS,
                "formats": ["jpeg", "png", "webp"],
            },
        },
        "controls": {
            "guidance_scale": (
                None if family == "ltx-2.5" else {"minimum": 1.0, "maximum": 30.0}
            ),
            "conditioning_strength": (
                {"minimum": 0.0, "maximum": 1.0}
                if family in {"ltx-2.3", "ltx-2.5"}
                else None
            ),
            "negative_prompt": family != "ltx-2.5",
        },
    }


def _validate_reference_image(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="image-to-video requires `pip install 'rapid-mlx[video]'`",
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
                if width <= 0 or height <= 0 or width * height > _MAX_REFERENCE_PIXELS:
                    raise ValueError("reference image exceeds 16 megapixels")
                if image_format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("reference image must be JPEG, PNG, or WebP")
                image.verify()
    except (OSError, ValueError, Image.DecompressionBombError, Warning) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid input_reference: {exc}"
        ) from exc


async def _run_job(
    job: _VideoJob,
    *,
    engine,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    image_path: Path | None,
    negative_prompt: str | None,
    guidance_scale: float | None,
    conditioning_strength: float | None,
) -> None:
    started = False
    generation_completed = False
    completed_job: _VideoJob | None = None
    output = _jobs_root / job.id / "output.mp4"
    generation_gate = _generation_gate_for_current_loop()
    is_cogvideox = getattr(engine, "video_family", "") == "cogvideox-fun"
    alignment = 32 if getattr(engine, "video_family", "") == "ltx-2.5" else 64
    generation_width = (
        width if is_cogvideox else ((width + alignment - 1) // alignment) * alignment
    )
    generation_height = (
        height if is_cogvideox else ((height + alignment - 1) // alignment) * alignment
    )

    async def generate_under_gate() -> bool:
        nonlocal started
        # This inner task owns the gate, so cancellation of the request-facing
        # job never releases it while an uncancellable MLX thread is running.
        async with generation_gate:
            with _jobs_lock:
                if job.status == "failed":
                    return False
                started = True
                job.status = "in_progress"
                job.progress = 1
            await _run_in_generation_thread(
                engine.generate,
                prompt=job.prompt,
                output_path=output,
                width=generation_width,
                height=generation_height,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                image=image_path,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                conditioning_strength=conditioning_strength,
                output_width=width,
                output_height=height,
            )
            return True

    runner = asyncio.create_task(generate_under_gate())
    try:
        generated = await asyncio.shield(runner)
        if not generated:
            return
        completed_job = replace(
            job,
            status="completed",
            progress=100,
            completed_at=int(time.time()),
            output_path=str(output),
            generation_finished=True,
        )
        with _jobs_lock:
            # Keep the job in progress until its atomic manifest commit
            # finishes. That prevents normal retention from evicting its
            # directory while the persistence thread still owns it.
            job.output_path = completed_job.output_path
            job.generation_finished = completed_job.generation_finished
        generation_completed = True
        if _jobs_are_persistent:
            try:
                await _persist_completed_job_in_daemon_thread(completed_job)
            except (OSError, RuntimeError):
                # The MP4 is already complete and remains downloadable for this
                # process. Keep that user result available while making the
                # durability loss visible to operators.
                logger.exception(
                    "Unable to persist completed video job metadata for %s", job.id
                )
        with _jobs_lock:
            job.status = completed_job.status
            job.progress = completed_job.progress
            job.completed_at = completed_job.completed_at
    except asyncio.CancelledError:
        if generation_completed:
            # The completed MP4 remains valid. Metadata persistence is atomic
            # and may finish in its detached daemon thread after shutdown.
            assert completed_job is not None
            with _jobs_lock:
                job.status = completed_job.status
                job.progress = completed_job.progress
                job.completed_at = completed_job.completed_at
            raise
        with _jobs_lock:
            if job.status != "failed":
                job.status = "failed"
                job.error = {
                    "code": "video_generation_cancelled",
                    "message": "Video generation was cancelled.",
                }
        if not started:
            runner.cancel()

        async def reap_cancelled_job() -> None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await runner
            with _jobs_lock:
                job.generation_finished = True
            await asyncio.to_thread(
                shutil.rmtree, _jobs_root / job.id, ignore_errors=True
            )

        cleanup = asyncio.create_task(reap_cancelled_job())
        _cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(_cleanup_tasks.discard)
        raise
    except Exception as exc:  # noqa: BLE001
        from ..runtime.video_lane import VideoRuntimeError

        logger.exception("Video generation job %s failed", job.id)
        message = (
            str(exc)
            if isinstance(exc, VideoRuntimeError)
            else "Video generation failed; check the server logs for details."
        )
        with _jobs_lock:
            job.status = "failed"
            job.error = {"code": "video_generation_failed", "message": message}
            job.generation_finished = True
        await asyncio.to_thread(shutil.rmtree, _jobs_root / job.id, ignore_errors=True)


@router.post("/v1/videos", dependencies=[Depends(verify_api_key)])
async def create_video(
    prompt: str = Form(..., min_length=1, max_length=4096),
    model: str = Form("ltx-2.3-mlx-q4"),
    seconds: str = Form("4"),
    size: str = Form("768x512"),
    seed: int = Form(42),
    fps: Annotated[int | None, Form()] = None,
    frames: Annotated[int | None, Form()] = None,
    guidance_scale: Annotated[float | None, Form()] = None,
    conditioning_strength: Annotated[float | None, Form()] = None,
    negative_prompt: Annotated[str | None, Form(max_length=4096)] = None,
    input_reference: UploadFile | None = File(None),
):
    engine = _video_engine()
    is_cogvideox = getattr(engine, "video_family", "") == "cogvideox-fun"
    is_wan = getattr(engine, "video_family", "") == "wan"
    is_ltx25 = getattr(engine, "video_family", "") == "ltx-2.5"
    with _jobs_lock:
        if not _accepting_jobs:
            raise HTTPException(status_code=503, detail="video server is shutting down")
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be blank")
    allowed_models = {engine.model_name}
    profile = resolve_profile(model)
    if profile is not None and profile.hf_path == engine.model_name:
        allowed_models.add(model)
    if not (is_cogvideox or is_wan or is_ltx25):
        allowed_models.add("ltx-2.3-mlx-q4")
    if model not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"model must match the served video model ({engine.model_name})",
        )
    if frames is None:
        try:
            seconds_int = int(seconds)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="seconds must be an integer"
            ) from exc
        if is_cogvideox and seconds_int != 1:
            raise HTTPException(
                status_code=400,
                detail="CogVideoX-Fun MVP currently supports seconds=1 only",
            )
        if not 1 <= seconds_int <= 20:
            raise HTTPException(
                status_code=400, detail="seconds must be between 1 and 20"
            )
    else:
        # An explicit frame count is the temporal source of truth. Derive the
        # public duration after request FPS validation below.
        seconds_int = 1
    if is_cogvideox:
        if size != "672x384":
            raise HTTPException(
                status_code=400,
                detail="CogVideoX-Fun MVP currently supports size=672x384 only",
            )
        width, height = 672, 384
    else:
        width, height = _parse_size(size, multiple=32 if is_ltx25 else 64)
    native_fps = 5 if is_cogvideox else getattr(engine, "native_fps", 24)
    request_fps = native_fps if fps is None else fps
    if not 1 <= request_fps <= 60:
        raise HTTPException(status_code=400, detail="fps must be between 1 and 60")
    if is_wan and request_fps != native_fps:
        raise HTTPException(
            status_code=400,
            detail=f"Wan output fps is fixed by this checkpoint at {native_fps}",
        )
    if frames is not None and not 1 <= frames <= 1201:
        raise HTTPException(status_code=400, detail="frames must be between 1 and 1201")
    if frames is not None:
        seconds_int = max(1, (frames + request_fps - 1) // request_fps)
    if guidance_scale is not None and not 1.0 <= guidance_scale <= 30.0:
        raise HTTPException(
            status_code=400,
            detail="guidance_scale must be between 1 and 30",
        )
    if conditioning_strength is not None:
        if not 0.0 <= conditioning_strength <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="conditioning_strength must be between 0 and 1",
            )
        if is_cogvideox or is_wan:
            raise HTTPException(
                status_code=400,
                detail="conditioning_strength is currently supported by LTX only",
            )
        if input_reference is None:
            raise HTTPException(
                status_code=400,
                detail="conditioning_strength requires input_reference",
            )
    if negative_prompt is not None:
        negative_prompt = negative_prompt.strip() or None
    if is_ltx25 and (negative_prompt or guidance_scale is not None):
        raise HTTPException(
            status_code=400,
            detail=(
                "LTX-2.5 distilled generation does not support negative_prompt "
                "or guidance_scale"
            ),
        )
    alignment = 32 if is_ltx25 else 64
    generation_width = ((width + alignment - 1) // alignment) * alignment
    generation_height = ((height + alignment - 1) // alignment) * alignment
    if frames is not None:
        request_frames = frames
    elif is_cogvideox:
        request_frames = 5
    else:
        # Normalize seconds × fps to the temporal shape the diffusion
        # backends require. Only an explicitly supplied frame count should
        # ever reach the validation below unnormalized.
        request_frames = _frame_count(
            seconds_int, native_fps if is_wan else request_fps
        )
    if is_cogvideox and request_frames < 5:
        raise HTTPException(
            status_code=400, detail="CogVideoX-Fun frames must be at least 5"
        )
    if is_cogvideox and request_frames % 4 != 1:
        raise HTTPException(
            status_code=400,
            detail="CogVideoX-Fun frames must be 4n+1 (for example 5, 9, 13)",
        )
    if is_wan and (request_frames < 5 or request_frames % 4 != 1):
        raise HTTPException(
            status_code=400,
            detail="Wan frames must be 4n+1 and at least 5 (for example 5, 9, 13)",
        )
    if not (is_cogvideox or is_wan):
        if request_frames < 9 or request_frames % 8 != 1:
            raise HTTPException(
                status_code=400,
                detail="LTX frames must be 8n+1 and at least 9 (for example 9, 17, 25)",
            )
    workload_width = width if is_cogvideox else generation_width
    workload_height = height if is_cogvideox else generation_height
    if workload_width * workload_height * request_frames > _MAX_PIXEL_FRAMES:
        family = (
            "CogVideoX-Fun"
            if is_cogvideox
            else ("Wan" if is_wan else ("LTX-2.5 Q8" if is_ltx25 else "LTX-2.3 Q4"))
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"requested video exceeds the safe {family} workload limit; "
                "reduce size or duration"
            ),
        )

    job_id = f"video_{uuid.uuid4().hex}"
    job_dir = _jobs_root / job_id
    job_dir.mkdir(mode=0o700)
    image_path = None
    enqueued = False
    evicted_id: str | None = None
    task: asyncio.Task | None = None
    try:
        if is_cogvideox and input_reference is not None:
            raise HTTPException(
                status_code=400,
                detail="CogVideoX-Fun MVP currently supports text-to-video only",
            )
        if input_reference is not None:
            image_path = job_dir / "reference.img"
            total = 0
            target = await asyncio.to_thread(image_path.open, "xb")
            try:
                while chunk := await input_reference.read(1024 * 1024):
                    total += len(chunk)
                    if total > _MAX_REFERENCE_BYTES:
                        raise HTTPException(
                            status_code=413, detail="input_reference exceeds 20 MB"
                        )
                    await asyncio.to_thread(target.write, chunk)
            finally:
                await asyncio.to_thread(target.close)
            await asyncio.to_thread(_validate_reference_image, image_path)

        num_frames = request_frames
        if is_wan:
            from ..runtime.video_lane import validate_video_request

            try:
                validate_video_request(
                    engine,
                    width=generation_width,
                    height=generation_height,
                    num_frames=num_frames,
                    image=image_path,
                )
            except ValueError as exc:
                from ..video.wan import WanRequestError

                if not isinstance(exc, WanRequestError):
                    raise
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        job = _VideoJob(
            id=job_id,
            model=model if (is_cogvideox or is_wan or is_ltx25) else "ltx-2.3-mlx-q4",
            prompt=prompt,
            seconds=str(seconds_int),
            size=f"{width}x{height}",
            created_at=int(time.time()),
        )
        with _jobs_lock:
            if not _accepting_jobs:
                raise HTTPException(
                    status_code=503, detail="video server is shutting down"
                )
            if len(_jobs) >= _MAX_JOBS:
                finished = [
                    item
                    for item in _jobs.values()
                    if item.status in {"completed", "failed"}
                    and item.generation_finished
                ]
                if not finished:
                    raise HTTPException(
                        status_code=429, detail="video job queue is full"
                    )
                oldest = min(finished, key=lambda item: item.created_at)
                _jobs.pop(oldest.id, None)
                evicted_id = oldest.id
            _jobs[job.id] = job
            task = asyncio.create_task(
                _run_job(
                    job,
                    engine=engine,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=request_fps,
                    seed=seed,
                    image_path=image_path,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    conditioning_strength=conditioning_strength,
                )
            )
            _tasks[job.id] = task
            enqueued = True
    finally:
        if not enqueued:
            await asyncio.to_thread(shutil.rmtree, job_dir, ignore_errors=True)
    assert task is not None

    def discard_task(done: asyncio.Task) -> None:
        if _tasks.get(job.id) is done:
            _tasks.pop(job.id, None)

    task.add_done_callback(discard_task)
    if evicted_id is not None:
        await asyncio.to_thread(
            shutil.rmtree, _jobs_root / evicted_id, ignore_errors=True
        )
    return job.public()


@router.get("/v1/videos/capabilities", dependencies=[Depends(verify_api_key)])
async def video_capabilities():
    """Return machine-readable limits for the currently served video model."""
    return _video_capabilities(_video_engine())


def _get_job(video_id: str) -> _VideoJob:
    with _jobs_lock:
        job = _jobs.get(video_id)
        snapshot = replace(job) if job is not None else None
    if snapshot is None:
        raise HTTPException(status_code=404, detail="video job not found")
    return snapshot


@router.get("/v1/videos/{video_id}", dependencies=[Depends(verify_api_key)])
async def retrieve_video(video_id: str):
    return _get_job(video_id).public()


@router.get("/v1/videos/{video_id}/content", dependencies=[Depends(verify_api_key)])
async def retrieve_video_content(video_id: str):
    job = _get_job(video_id)
    if job.status != "completed" or job.output_path is None:
        raise HTTPException(status_code=409, detail="video is not completed")
    # Open relative to no-follow directory descriptors. This prevents a local
    # process with access to the operator-owned store from swapping either the
    # restored job directory or output file for a symlink between startup
    # validation and download. O_NONBLOCK also prevents a substituted FIFO
    # from stalling the event loop before fstat can reject it.
    root_fd = job_fd = output_fd = None
    try:
        root_fd, job_fd = _open_video_job_directory(job.id)
        output_fd = _open_video_output(job_fd)
        source = os.fdopen(output_fd, "rb")
        output_fd = None
    except OSError as exc:
        raise HTTPException(
            status_code=410, detail="video content has expired"
        ) from exc
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if job_fd is not None:
            os.close(job_fd)
        if root_fd is not None:
            os.close(root_fd)

    def chunks():
        try:
            while data := source.read(1024 * 1024):
                yield data
        finally:
            source.close()

    return StreamingResponse(
        chunks(),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{job.id}.mp4"'},
    )


@router.get("/v1/videos", dependencies=[Depends(verify_api_key)])
async def list_videos(limit: int = Query(20, ge=1, le=100)):
    with _jobs_lock:
        data = [
            replace(job)
            for job in sorted(
                _jobs.values(), key=lambda item: item.created_at, reverse=True
            )[:limit]
        ]
    return {"object": "list", "data": [job.public() for job in data]}


@router.delete("/v1/videos/{video_id}", dependencies=[Depends(verify_api_key)])
async def delete_video(video_id: str):
    with _jobs_lock:
        job = _jobs.get(video_id)
        if job is not None and job.status == "in_progress":
            raise HTTPException(
                status_code=409, detail="video generation is in progress"
            )
        task = _tasks.get(video_id) if job is not None else None
        if job is not None and job.status == "queued":
            job.status = "failed"
            job.error = {
                "code": "video_generation_cancelled",
                "message": "Video generation was cancelled.",
            }
        if job is not None and job.status != "queued":
            _jobs.pop(video_id, None)
    if job is None:
        raise HTTPException(status_code=404, detail="video job not found")
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with _jobs_lock:
            _jobs.pop(video_id, None)
    await asyncio.to_thread(shutil.rmtree, _jobs_root / video_id, ignore_errors=True)
    response = job.public()
    response["deleted"] = True
    return response
