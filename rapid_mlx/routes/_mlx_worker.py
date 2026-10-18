# SPDX-License-Identifier: Apache-2.0
"""Shared worker lane for retrieval-model MLX operations."""

import asyncio
import concurrent.futures
import functools

_retrieval_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mlx-retrieval"
)


async def run_on_mlx_worker(func, /, *args, **kwargs):
    """Run retrieval MLX work on its dedicated owning thread."""
    return await asyncio.get_running_loop().run_in_executor(
        _retrieval_executor, functools.partial(func, *args, **kwargs)
    )


def run_on_mlx_worker_sync(func, /, *args, **kwargs):
    """Blocking entry point for loading retrieval models on their owner."""
    return _retrieval_executor.submit(func, *args, **kwargs).result()
