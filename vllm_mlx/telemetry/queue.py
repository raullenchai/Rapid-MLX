# SPDX-License-Identifier: Apache-2.0
"""Bounded, lossy, daemon-flushed in-process event queue.

Phase 2 sites construct payload dicts and call ``enqueue()`` on the
process-singleton queue. A background daemon thread drains the queue
on a 60-second idle interval, eagerly when 10 events are pending, and
once more at interpreter shutdown via ``atexit``.

Three invariants the design pins down:

1. **Telemetry must not slow the foreground.** ``enqueue()`` does only
   a deque append inside one lock — no JSON dump, no HTTP, no sleep.
   The flush daemon owns every blocking call.

2. **The queue must not grow unbounded.** ``deque(maxlen=N)`` drops
   the oldest event when at capacity. A stuck collector or an offline
   user must not leave the process consuming RAM on telemetry waste.

3. **Shutdown must not hang.** The flush thread is ``daemon=True`` so
   a missed join still lets the interpreter exit. ``atexit`` calls
   ``shutdown()`` with a 2-second join budget — long enough for one
   final POST round-trip, short enough not to annoy a user on Ctrl-C.

Important non-feature: there is **no asyncio integration here**. The
queue must serve both ``rapid-mlx serve`` (async FastAPI) and
``rapid-mlx chat`` (sync REPL) without forcing every caller into an
event loop. A plain daemon thread is the lowest common denominator.
"""

from __future__ import annotations

import atexit
import threading
from collections import deque
from typing import Any

from vllm_mlx.telemetry.transport import post_batch

MAX_QUEUE_LEN = 100
FLUSH_INTERVAL_S = 60.0
FLUSH_THRESHOLD = 10
SHUTDOWN_BUDGET_S = 2.0


class TelemetryQueue:
    """Process-singleton telemetry event queue + flush daemon.

    Construct once per process. ``start()`` is idempotent; calling it
    twice from the same process is a no-op so cli.py + lifespan can
    both call without ordering assumptions.
    """

    def __init__(
        self,
        *,
        max_len: int = MAX_QUEUE_LEN,
        flush_interval_s: float = FLUSH_INTERVAL_S,
        flush_threshold: int = FLUSH_THRESHOLD,
        flusher: Any = post_batch,
    ) -> None:
        # ``flusher`` is injectable so tests can hand in a fake without
        # patching the transport module globally.
        self._events: deque[dict[str, Any]] = deque(maxlen=max_len)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._flush_interval_s = flush_interval_s
        self._flush_threshold = flush_threshold
        self._flusher = flusher
        self._atexit_registered = False
        # Bookkeeping the ``rapid-mlx telemetry status`` subcommand can
        # surface. ``last_flush_ts`` is the monotonic clock at the last
        # _completed_ flush attempt (success or failure).
        self.events_enqueued = 0
        self.events_dropped = 0
        self.flushes_ok = 0
        self.flushes_failed = 0

    # ----------------------------------------------------------------- API

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Append a payload. Drop the oldest if at capacity.

        Cheap and lock-only. Wakes the flusher when the queue crosses
        ``flush_threshold``; otherwise relies on the idle timer.
        """
        with self._lock:
            full = len(self._events) == self._events.maxlen
            self._events.append(payload)
            if full:
                self.events_dropped += 1
            self.events_enqueued += 1
            should_wake = len(self._events) >= self._flush_threshold
        if should_wake:
            self._wake.set()

    def start(self) -> None:
        """Start the flush daemon if not already running.

        Idempotent — safe to call from multiple init paths (cli.py
        main, FastAPI lifespan, test fixture).
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="rapid-mlx-telemetry",
            daemon=True,
        )
        self._thread.start()
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

    def shutdown(self, *, timeout: float = SHUTDOWN_BUDGET_S) -> None:
        """Stop the flush daemon and drain whatever is still in the queue.

        Safe to call multiple times. Always returns within ``timeout``
        seconds — a slow Worker must not leave the interpreter hanging.

        ``self._thread`` is cleared only after the join confirms the
        daemon actually exited. Round 1 codex review caught that
        unconditional clearing let a later ``start()`` spawn a second
        daemon while the original was still draining a slow flusher,
        producing parallel flushers writing to the same queue.
        """
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if not thread.is_alive():
                self._thread = None

    def snapshot(self) -> dict[str, int]:
        """Cheap counters for ``rapid-mlx telemetry status``."""
        with self._lock:
            pending = len(self._events)
        return {
            "pending": pending,
            "enqueued_total": self.events_enqueued,
            "dropped_total": self.events_dropped,
            "flushes_ok": self.flushes_ok,
            "flushes_failed": self.flushes_failed,
        }

    # --------------------------------------------------------------- internals

    def _loop(self) -> None:
        while not self._stop.is_set():
            # ``wait`` returns True if woken via ``_wake.set()`` (queue
            # crossed threshold or shutdown), False on timeout (idle
            # flush). Either way we drain.
            self._wake.wait(timeout=self._flush_interval_s)
            self._wake.clear()
            self._drain_once()
        # Final drain after stop signal so events queued between the
        # last wake and shutdown still get a try.
        self._drain_once()

    def _drain_once(self) -> None:
        # Move all currently-queued events to a local batch, then release
        # the lock before doing network I/O. enqueue() during the POST
        # must not block.
        with self._lock:
            if not self._events:
                return
            batch = list(self._events)
            self._events.clear()
        try:
            ok = self._flusher(batch)
        except Exception:
            # Flusher must never raise on its own contract, but if a
            # bug slips through (or a tests stub raises), we treat it
            # as a failure rather than crashing the daemon. The daemon
            # crashing would silently disable telemetry for the rest
            # of the process — worse than a logged failure.
            ok = False
        if ok:
            self.flushes_ok += 1
        else:
            self.flushes_failed += 1
