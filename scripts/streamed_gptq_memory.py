"""Shared RSS watchdog and native allocator limits for streamed builds."""

import os
import sys
import threading

import mlx.core as mx


class MemoryGuard:
    """Fail-closed RSS watchdog plus MLX allocator/cache limits.

    The watchdog exits during long native operations too, leaving the last
    committed layer boundary untouched. RSS and MLX memory overlap: cap each,
    do not add the counters and double-count unified memory.
    """

    def __init__(self, c):
        import psutil

        self.process = psutil.Process()
        self.limit = int(c["memory_limit_gb"] * 1e9)
        self.peak = self.mlx_peak = 0
        self.event = threading.Event()
        self.thread = None
        self.previous_memory_limit = mx.set_memory_limit(self.limit)
        try:
            self.previous_cache_limit = mx.set_cache_limit(
                int(c["cache_limit_gb"] * 1e9)
            )
        except BaseException:
            mx.set_memory_limit(self.previous_memory_limit)
            raise

    def restore_limits(self):
        mx.set_cache_limit(self.previous_cache_limit)
        mx.set_memory_limit(self.previous_memory_limit)

    def check(self):
        rss = self.process.memory_info().rss
        mlx_bytes = mx.get_active_memory() + mx.get_cache_memory()
        self.peak, self.mlx_peak = max(self.peak, rss), max(self.mlx_peak, mlx_bytes)
        if max(rss, mlx_bytes) > self.limit:
            raise MemoryError(
                f"GPTQ memory cap exceeded: RSS={rss}, MLX={mlx_bytes}, cap={self.limit}"
            )

    def __enter__(self):
        try:
            self.check()
        except BaseException:
            self.restore_limits()
            raise

        def watch():
            while not self.event.wait(0.02):
                try:
                    self.check()
                except MemoryError as exc:
                    print(str(exc), file=sys.stderr, flush=True)
                    os._exit(86)

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.event.set()
        self.thread.join()
        self.restore_limits()

    def report(self):
        self.check()
        return dict(
            peak_rss_bytes=self.peak,
            peak_mlx_active_cache_bytes=self.mlx_peak,
            memory_cap_bytes=self.limit,
            rss_poll_seconds=0.02,
        )
