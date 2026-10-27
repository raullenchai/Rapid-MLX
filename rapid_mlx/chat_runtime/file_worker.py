# SPDX-License-Identifier: Apache-2.0
"""Cancellable subprocess isolation for bounded, read-only document work."""

import asyncio
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


async def run_worker(module, payload, *, arguments=(), timeout=10):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",  # Never import project code from cwd, PYTHONPATH, or user site.
        "-m",
        module,
        *arguments,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(payload), timeout)
        if process.returncode == 0:
            return json.loads(stdout)
        return {
            "error": {
                "code": "worker_failed",
                "message": "File operation failed safely.",
            }
        }
    except asyncio.TimeoutError:
        return {
            "error": {
                "code": "timeout",
                "message": "File operation timed out; narrow the path, pattern, or document.",
            }
        }
    finally:
        # Own the worker process group; Stop leaves no work running.
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def serialized_read(
    function: Callable[Concatenate[Any, P], Awaitable[R]],
) -> Callable[Concatenate[Any, P], Awaitable[R]]:
    """Share the turn's output budget without changing a tool's schema."""

    @wraps(function)
    async def locked(ctx: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        async with ctx.deps.read_lock:
            return await function(ctx, *args, **kwargs)

    return locked
