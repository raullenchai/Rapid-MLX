# SPDX-License-Identifier: Apache-2.0
"""Renders and edits run as jobs, not as the request that asked for them.

The engine answers an image request only once the whole render is finished,
so relaying it inline held one connection open with no bytes flowing for
minutes. Cloudflare cuts that at 100 s and returns 524 — measured through a
trycloudflare tunnel, a ~20 s generation survived and an edit did not.

So the POST starts a job and answers immediately, and the page polls for the
result. Polled rather than streamed for the same reason the download feed is:
a sparse SSE body is buffered indefinitely by the same tunnel.

Only the last job is kept, forever. The engine renders one image at a time
and this server has a single user, so a job table would only add expiry.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

import httpx

from .lifecycle import ActivityLease, EngineLifecycle

# What the job's work returns: the engine's status code and decoded body.
ImageWork = Callable[[], Awaitable[tuple[int, dict]]]


class ImageJobState(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ImageJobError(RuntimeError):
    """A job could not be started."""


@dataclass
class ImageJob:
    """One in-flight or finished render.

    A cancelled render is ``DONE``, not a state of its own: the engine
    stops at its next denoise step and returns whatever finished, so an
    empty result with ``cancelled`` set is a success.
    """

    id: str
    # Which resident model is rendering, so progress is read from it rather
    # than from the engine's primary — which is the chat model whenever the
    # image model was hot-loaded beside it.
    model: str | None
    state: ImageJobState = ImageJobState.RUNNING
    b64_json: str | None = None
    cancelled: bool = False
    error: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.value,
            "b64_json": self.b64_json,
            "cancelled": self.cancelled,
            "error": self.error,
        }


class ImageJobManager:
    """Runs at most one render at a time, detached from any request."""

    def __init__(self, lifecycle: EngineLifecycle) -> None:
        self._lifecycle = lifecycle
        self._job: ImageJob | None = None
        self._task: asyncio.Task | None = None

    def is_running(self) -> bool:
        return self._job is not None and self._job.state is ImageJobState.RUNNING

    def get(self, job_id: str) -> ImageJob | None:
        job = self._job
        return job if job is not None and job.id == job_id else None

    def start(self, work: ImageWork, *, model: str | None) -> ImageJob:
        if self.is_running():
            raise ImageJobError(
                "a render is already running; wait for it to finish or stop it"
            )
        lease = self._lifecycle.acquire_activity()
        if lease is None:
            raise ImageJobError("the engine is changing models; wait for it to finish")
        job = ImageJob(id=uuid4().hex, model=model)
        self._job = job
        self._task = asyncio.create_task(self._run(job, work, lease))
        return job

    async def _run(self, job: ImageJob, work: ImageWork, lease: ActivityLease) -> None:
        # Counted for the JOB's life, not a request's: switching models
        # restarts the engine, which would destroy the render.
        with lease:
            try:
                status, body = await work()
            except httpx.HTTPError as exc:
                _fail(
                    job,
                    502,
                    "engine_transport",
                    f"connection to the engine failed: {exc}",
                )
                return
            except Exception as exc:  # noqa: BLE001
                # Nothing awaits this task, so an escaping exception would
                # leave the job RUNNING and the page polling forever.
                _fail(job, 500, "image_job_failed", str(exc))
                return

            error = body.get("error") if isinstance(body, dict) else None
            if status >= 400 or isinstance(error, dict):
                message = (
                    error.get("message")
                    if isinstance(error, dict) and error.get("message")
                    else f"the engine returned {status}"
                )
                code = (
                    error.get("type")
                    if isinstance(error, dict) and error.get("type")
                    else "engine_error"
                )
                _fail(job, status, code, message)
                return

            data = body.get("data") or []
            first = data[0] if data else None
            job.b64_json = first.get("b64_json") if isinstance(first, dict) else None
            job.cancelled = bool(body.get("cancelled"))
            job.state = ImageJobState.DONE

    async def shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None


def _fail(job: ImageJob, status: int, code: str, message: str) -> None:
    job.error = {"message": message, "type": code, "status": status}
    job.state = ImageJobState.FAILED
