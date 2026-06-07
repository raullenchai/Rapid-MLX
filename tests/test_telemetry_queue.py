# SPDX-License-Identifier: Apache-2.0
"""Contract pins for ``vllm_mlx.telemetry.queue.TelemetryQueue``.

Properties to lock down:
- ``enqueue`` is non-blocking (no network on hot path).
- Queue is bounded; oldest drops first when at capacity.
- Daemon flushes on threshold and on shutdown.
- ``shutdown`` returns within budget even when flusher hangs.
- ``snapshot`` exposes the counters the ``telemetry status`` UX needs.
"""

from __future__ import annotations

import threading
import time

from vllm_mlx.telemetry.queue import (
    FLUSH_INTERVAL_S,
    FLUSH_THRESHOLD,
    MAX_QUEUE_LEN,
    TelemetryQueue,
)


def test_enqueue_buffers_until_threshold():
    flushed: list[list[dict]] = []

    def flusher(batch):
        flushed.append(batch)
        return True

    q = TelemetryQueue(flusher=flusher, flush_interval_s=60.0, flush_threshold=5)
    q.start()
    try:
        for i in range(4):
            q.enqueue({"i": i})
        # No flush should have run yet (under threshold + no time elapsed).
        time.sleep(0.05)
        assert flushed == []

        q.enqueue({"i": 4})  # crosses threshold
        # Wait briefly for the daemon to wake + drain.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not flushed:
            time.sleep(0.01)
        assert flushed, "daemon never flushed on threshold cross"
        # The first flush should have grabbed all 5 events.
        assert len(flushed[0]) == 5
    finally:
        q.shutdown(timeout=0.5)


def test_drops_oldest_when_over_capacity():
    captured: list[dict] = []
    started = threading.Event()

    def flusher(batch):
        started.set()
        captured.extend(batch)
        return True

    q = TelemetryQueue(
        flusher=flusher,
        max_len=3,
        flush_interval_s=60.0,
        flush_threshold=999,  # disable threshold flushing
    )
    q.start()
    try:
        # Enqueue 6 events into a queue with capacity 3 — first 3 should
        # be dropped, last 3 retained.
        for i in range(6):
            q.enqueue({"i": i})
        snap = q.snapshot()
        assert snap["pending"] == 3
        assert snap["enqueued_total"] == 6
        assert snap["dropped_total"] == 3
    finally:
        q.shutdown(timeout=0.5)
    # After shutdown the daemon drains. The retained events must be the
    # newest three.
    assert started.is_set()
    assert [e["i"] for e in captured] == [3, 4, 5]


def test_shutdown_drains_remaining_events():
    captured: list[dict] = []

    def flusher(batch):
        captured.extend(batch)
        return True

    q = TelemetryQueue(flusher=flusher, flush_interval_s=60.0, flush_threshold=999)
    q.start()
    q.enqueue({"a": 1})
    q.enqueue({"a": 2})
    q.shutdown(timeout=1.0)
    assert [e["a"] for e in captured] == [1, 2]


def test_shutdown_does_not_orphan_thread_for_restart_when_join_times_out():
    """Codex round 1 caught this: ``shutdown()`` cleared ``_thread``
    unconditionally, so a subsequent ``start()`` could spawn a SECOND
    daemon while the original was still draining a slow flusher. With
    two flushers attached to the same ``_events`` deque, events would
    flush in arbitrary interleaving and the queue lock would not
    suffice (each batch is read-then-clear, not atomic with the wake)."""
    release = threading.Event()

    def slow_flusher(batch):
        release.wait(timeout=5.0)
        return True

    q = TelemetryQueue(flusher=slow_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    q.enqueue({"x": 1})
    time.sleep(0.1)  # let the daemon enter slow_flusher

    q.shutdown(timeout=0.1)  # times out — flusher still blocked on release
    # The daemon must NOT be considered "gone" yet: a restart would
    # double the flusher.
    original_thread = q._thread
    assert original_thread is not None
    assert original_thread.is_alive()

    # Round 2 codex review caught that an ``active_count`` assert is
    # too weak — pin the precise property we care about: ``start()``
    # MUST NOT spawn a sibling. Compare by object identity of the
    # daemon thread, and also count threads named
    # ``rapid-mlx-telemetry`` directly so a name-only regression
    # cannot slip past.
    before_named = [t for t in threading.enumerate() if t.name == "rapid-mlx-telemetry"]
    q.start()
    after_named = [t for t in threading.enumerate() if t.name == "rapid-mlx-telemetry"]
    assert q._thread is original_thread, "start() replaced live daemon"
    assert len(after_named) == len(before_named), (
        f"start() spawned a second telemetry daemon "
        f"(before={len(before_named)}, after={len(after_named)})"
    )

    # Release the old daemon so the test process can exit.
    release.set()
    q.shutdown(timeout=1.0)


def test_shutdown_returns_within_budget_even_if_flusher_hangs():
    # Flusher that blocks longer than the shutdown budget.
    release = threading.Event()

    def slow_flusher(batch):
        release.wait(timeout=5.0)
        return True

    q = TelemetryQueue(flusher=slow_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    q.enqueue({"x": 1})
    time.sleep(0.1)  # let daemon start the flush

    t0 = time.monotonic()
    q.shutdown(timeout=0.2)
    elapsed = time.monotonic() - t0
    release.set()
    # We accept some daemon-cleanup overhead, but it must not block on
    # the slow flusher.
    assert elapsed < 0.5, f"shutdown blocked {elapsed:.2f}s on slow flusher"


def test_flusher_exception_increments_failed_not_crash():
    def bad_flusher(batch):
        raise RuntimeError("synthetic")

    q = TelemetryQueue(flusher=bad_flusher, flush_interval_s=60.0, flush_threshold=1)
    q.start()
    q.enqueue({"x": 1})
    time.sleep(0.2)
    snap = q.snapshot()
    assert snap["flushes_failed"] >= 1
    assert snap["flushes_ok"] == 0
    q.shutdown(timeout=0.5)
    # Daemon must still be capable of accepting more work after the bug.
    q.start()
    q.enqueue({"x": 2})
    q.shutdown(timeout=0.5)


def test_start_is_idempotent():
    q = TelemetryQueue(flusher=lambda b: True)
    q.start()
    q.start()  # second call must be a no-op, not crash
    q.shutdown(timeout=0.1)


def test_concurrent_start_does_not_spawn_duplicate_daemons():
    """Round 4 codex review: the previous ``start()`` check was
    unlocked, so a multi-init-path race (cli main + FastAPI lifespan
    both calling ``start()`` from different threads) could pass the
    ``is_alive()`` check on both and spawn two daemons against the
    same queue. The fix is a dedicated lifecycle lock around the
    check + creation."""
    q = TelemetryQueue(flusher=lambda b: True)
    started = threading.Event()

    def racer():
        started.wait(timeout=2.0)
        q.start()

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    # Release all at once so the race is real.
    started.set()
    for t in threads:
        t.join(timeout=2.0)

    named = [t for t in threading.enumerate() if t.name == "rapid-mlx-telemetry"]
    assert len(named) == 1, (
        f"concurrent start() spawned {len(named)} daemons (want exactly 1)"
    )
    q.shutdown(timeout=0.5)


def test_snapshot_shape():
    q = TelemetryQueue(flusher=lambda b: True)
    snap = q.snapshot()
    assert set(snap) == {
        "pending",
        "enqueued_total",
        "dropped_total",
        "flushes_ok",
        "flushes_failed",
    }


def test_module_defaults_are_sane():
    # Pin the public defaults the design doc references so a stray bump
    # doesn't silently change shipped behaviour.
    assert MAX_QUEUE_LEN == 100
    assert FLUSH_INTERVAL_S == 60.0
    assert FLUSH_THRESHOLD == 10
