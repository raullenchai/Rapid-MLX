# SPDX-License-Identifier: Apache-2.0
"""Shared worker lane for retrieval-model MLX operations."""

import asyncio
import concurrent.futures
import functools

from ..config import get_config

_retrieval_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mlx-retrieval"
)


async def run_on_mlx_worker(func, /, *args, **kwargs):
    """Run retrieval work on the chat worker or shared fallback lane."""
    primary = get_config().engine
    execute = getattr(primary, "execute_on_model_worker", None)
    if callable(execute):
        started = False

        def call():
            nonlocal started
            started = True
            return func(*args, **kwargs)

        try:
            return await execute(call)
        except RuntimeError:
            # Only fall back when dispatch failed before ``func`` started.
            if started:
                raise
    return await asyncio.get_running_loop().run_in_executor(
        _retrieval_executor, functools.partial(func, *args, **kwargs)
    )
