# SPDX-License-Identifier: Apache-2.0
"""Contract pins for ``rapid_mlx.telemetry.posthog_sender``.

The sender is the last stop before PostHog Cloud, so each test here
guards one of its rules and is written so that removing the rule turns
it red — verified by fault injection before merge, not assumed:

- the official-build gate and the LIVE ``allowed()`` check drop items;
- the burst caps (30/min per event name, 64-name registry, 1000 per
  session, 5000-item queue) each drop, and log ONCE per cap;
- the daemon flushes at 20 items or 10 s, chunks at 100, and its wire
  body is exactly ``envelope.build_batch`` compactly JSON-encoded with
  the stamp's key;
- failure handling: 4xx dropped, 5xx/transport retried exactly once,
  hostile injectables and garbage items never raise;
- ``flush`` is bounded by its timeout even with a hanging POST, and
  late captures after ``close()`` never enqueue.

Every network-facing test uses the injectable ``post``/``clock``/
``gate``/``allowed`` seams or a real ``http.server`` on 127.0.0.1 — no
real network, no real sleeping beyond short event waits.
"""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

from rapid_mlx.telemetry import posthog_sender as ph
from rapid_mlx.telemetry.build_gate import ReleaseStamp
from rapid_mlx.telemetry.posthog_sender import (
    MAX_EVENT_BUCKETS,
    MAX_QUEUE_ITEMS,
    POSTHOG_BATCH_URL,
    SESSION_ITEM_CEILING,
    PostHogSender,
    _reset_for_tests,
    get_sender,
    install_atexit,
)

STAMP = ReleaseStamp(channel="stable", posthog_key="phc_" + "a" * 32)
SENDER_LOGGER = "rapid_mlx.telemetry.posthog_sender"

REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_DEFAULT_POST = ph.default_post


# ----------------------------------------------------------------- helpers


class FakeClock:
    """Injectable monotonic clock the tests drive by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingPost:
    """Fake transport: records ``(url, body, timeout)``, returns a status."""

    def __init__(self, status: int = 200) -> None:
        self.calls: list[tuple[str, bytes, float]] = []
        self.status = status
        self.lock = threading.Lock()

    def __call__(self, url: str, body: bytes, timeout: float) -> int:
        with self.lock:
            self.calls.append((url, body, timeout))
        return self.status

    def __len__(self) -> int:
        with self.lock:
            return len(self.calls)

    def batches(self) -> list[list[dict[str, object]]]:
        with self.lock:
            return [
                json.loads(body)["batch"]  # type: ignore[no-any-return]
                for _, body, _ in self.calls
            ]


class HangOnFirstPost:
    """Hangs inside the FIRST post until released; records the rest."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[str, bytes, float]] = []

    def __call__(self, url: str, body: bytes, timeout: float) -> int:
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(timeout=10.0)
            return 200
        self.calls.append((url, body, timeout))
        return 200


class _ExplodingMapping(dict):
    """A mapping whose ``get`` raises — the capture fuzz must survive it."""

    def get(self, key: object, default: object = None) -> object:
        raise RuntimeError("exploded get")


def item(name: str = "app_opened", n: int = 0) -> dict[str, object]:
    """A well-formed batch item, the shape ``build_batch_item`` produces."""
    return {
        "uuid": str(uuid.uuid4()),
        "event": name,
        "distinct_id": "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f",
        "timestamp": "2026-01-01T00:00:00Z",
        "properties": {"n": n},
    }


def wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` (real seconds)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def sender_env(monkeypatch, tmp_path):
    """Hermetic env: no URL override, no kill switches, isolated HOME."""
    monkeypatch.delenv(ph.POSTHOG_URL_ENV, raising=False)
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    for var in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "TRAVIS",
        "BUILDKITE",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from rapid_mlx.telemetry.state import set_cli_kill_switch

    set_cli_kill_switch(False)
    return monkeypatch


def make_sender(
    post: Callable[[str, bytes, float], int] | None = None,
    clock: FakeClock | None = None,
) -> PostHogSender:
    """A sender gated to the test stamp with permission granted."""
    return PostHogSender(
        post=post if post is not None else RecordingPost(),
        clock=clock,
        gate=lambda: STAMP,
        allowed=lambda: True,
    )


# ------------------------------------------------------------ drop reasons


def test_not_official_build_drops(sender_env):
    post = RecordingPost()
    # ``allowed`` is pinned True so the official-build gate is the ONLY
    # thing standing between the item and the queue.
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: None,
        allowed=lambda: True,
    )
    assert s.capture(item()) is False
    s.flush(0.5)
    assert len(post) == 0
    s.close(0.5)


def test_allowed_flip_is_honoured_live(sender_env):
    """``allowed()`` is checked per capture, not once at construction."""
    post = RecordingPost()
    flag = {"on": True}
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: flag["on"],
    )
    assert s.capture(item(n=1)) is True
    flag["on"] = False
    assert s.capture(item(n=2)) is False
    assert s.capture(item(n=3)) is False
    flag["on"] = True
    assert s.capture(item(n=4)) is True
    s.flush(1.0)
    batches = post.batches()
    assert len(batches) == 1
    assert [props["properties"]["n"] for props in batches[0]] == [1, 4]
    s.close(0.5)


def test_item_without_a_usable_event_name_drops(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    assert s.capture({"no_event": 1}) is False
    assert s.capture({"event": None}) is False
    assert s.capture({"event": ""}) is False
    s.flush(0.5)
    assert len(post) == 0
    s.close(0.5)


# ------------------------------------------------------------- burst caps


def test_31st_event_within_a_minute_dropped(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    for i in range(30):
        assert s.capture(item(n=i)) is True
    assert s.capture(item(n=30)) is False
    assert s.capture(item(n=31)) is False
    s.flush(1.0)
    batches = post.batches()
    assert sum(len(batch) for batch in batches) == 30
    s.close(0.5)


def test_100_capability_rejections_capture_at_most_30(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    accepted = [s.capture(item("capability_rejected", n=i)) for i in range(100)]
    assert sum(accepted) == 30
    s.flush(1.0)
    assert sum(len(batch) for batch in post.batches()) == 30
    s.close(0.5)


def test_bucket_refills_with_the_clock(sender_env):
    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    for i in range(30):
        assert s.capture(item(n=i)) is True
    assert s.capture(item(n=30)) is False
    clock.advance(2.0)  # 0.5 tokens/s → exactly one token
    assert s.capture(item(n=31)) is True
    assert s.capture(item(n=32)) is False  # spent it
    clock.advance(0.5)  # only a quarter token accrued
    assert s.capture(item(n=33)) is False
    clock.advance(1.5)
    assert s.capture(item(n=34)) is True
    s.close(0.5)


def test_other_event_names_unaffected(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    for i in range(30):
        assert s.capture(item("a", n=i)) is True
    assert s.capture(item("a", n=30)) is False
    assert s.capture(item("b", n=0)) is True
    s.close(0.5)


def test_bucket_capacity_bounded_after_long_idle(sender_env):
    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    for i in range(30):
        assert s.capture(item(n=i)) is True
    clock.advance(3600.0)  # an idle hour must not bank more than one burst
    for i in range(30):
        assert s.capture(item(n=100 + i)) is True
    assert s.capture(item(n=200)) is False
    s.close(0.5)


def test_bucket_registry_capped_at_64(sender_env):
    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    for i in range(MAX_EVENT_BUCKETS):
        assert s.capture(item(f"ev{i}", n=i)) is True
        clock.advance(2.0)
    # A 65th distinct name drops even though its bucket would be full.
    assert s.capture(item("ev64")) is False
    assert s.capture(item("ev65")) is False
    # Existing buckets keep working.
    clock.advance(2.0)
    assert s.capture(item("ev0", n=999)) is True
    s.close(0.5)


def test_session_ceiling_1000(sender_env):
    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    names = [f"ev{i}" for i in range(40)]
    for i in range(SESSION_ITEM_CEILING):
        assert s.capture(item(names[i % 40], n=i)) is True
        clock.advance(2.0)  # keep every bucket topped up
    assert s.capture(item(names[0], n=1001)) is False
    assert s.capture(item(names[1], n=1002)) is False
    s.close(1.0)


def test_session_ceiling_drops_do_not_spend_burst_tokens(sender_env):
    clock = FakeClock()
    s = make_sender(clock=clock)
    with s._lock:
        s._accepted = SESSION_ITEM_CEILING
    assert s.capture(item("ceiling")) is False
    assert s.capture(item("ceiling")) is False
    with s._lock:
        bucket = s._buckets.get("ceiling")
        assert bucket is None or bucket.tokens == ph.PER_EVENT_BURST
    s.close(0.5)


def test_queue_full_drops_the_new_item(sender_env, caplog):
    """The 5000-item queue cap drops NEW items and never blocks.

    The 1000-item session ceiling bounds total accepts below the queue
    capacity, so natural overflow is impossible by design — the queue
    cap is defense in depth. Stuff the queue directly to exercise it.
    """
    post = RecordingPost()
    s = make_sender(post)
    with s._lock:
        s._queue.extend(item("stuffed", n=i) for i in range(MAX_QUEUE_ITEMS))
    with caplog.at_level(logging.DEBUG, logger=SENDER_LOGGER):
        assert s.capture(item("new")) is False
        assert s.capture(item("new2")) is False
    assert len(post) == 0
    full_logs = [r for r in caplog.records if "queue-full" in r.getMessage()]
    assert len(full_logs) == 1
    s.close(0.5)


# ---------------------------------------------------------------- log once


def test_rate_cap_logs_once(sender_env, caplog):
    s = make_sender()
    for i in range(30):
        assert s.capture(item(n=i)) is True
    with caplog.at_level(logging.DEBUG, logger=SENDER_LOGGER):
        assert s.capture(item(n=30)) is False
        assert s.capture(item(n=31)) is False
        assert s.capture(item(n=32)) is False
    logs = [r for r in caplog.records if "per-event-burst" in r.getMessage()]
    assert len(logs) == 1
    s.close(0.5)


def test_bucket_registry_cap_logs_once(sender_env, caplog):
    clock = FakeClock()
    s = make_sender(clock=clock)
    for i in range(MAX_EVENT_BUCKETS):
        assert s.capture(item(f"ev{i}")) is True
        clock.advance(2.0)
    with caplog.at_level(logging.DEBUG, logger=SENDER_LOGGER):
        assert s.capture(item("ev64")) is False
        assert s.capture(item("ev65")) is False
    logs = [r for r in caplog.records if "event-bucket-registry" in r.getMessage()]
    assert len(logs) == 1
    s.close(0.5)


def test_session_ceiling_logs_once(sender_env, caplog):
    clock = FakeClock()
    s = make_sender(clock=clock)
    names = [f"ev{i}" for i in range(40)]
    for i in range(SESSION_ITEM_CEILING):
        assert s.capture(item(names[i % 40], n=i)) is True
        clock.advance(2.0)
    with caplog.at_level(logging.DEBUG, logger=SENDER_LOGGER):
        assert s.capture(item(names[0], n=1001)) is False
        assert s.capture(item(names[1], n=1002)) is False
    logs = [r for r in caplog.records if "session-ceiling" in r.getMessage()]
    assert len(logs) == 1
    s.close(1.0)


# ---------------------------------------------------------------- batching


def test_batches_at_20_items(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    for i in range(19):
        assert s.capture(item(n=i)) is True
    # Below threshold with a frozen clock nothing flushes.
    assert wait_for(lambda: len(post) > 0, timeout=0.3) is False
    assert len(post) == 0
    assert s.capture(item(n=19)) is True  # the 20th crosses the threshold
    assert wait_for(lambda: len(post) == 1, timeout=2.0)
    assert len(post.batches()[0]) == 20
    s.close(0.5)


def test_batches_at_10s(sender_env):
    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    for i in range(3):
        assert s.capture(item(n=i)) is True
    assert wait_for(lambda: len(post) > 0, timeout=0.3) is False
    clock.advance(ph.FLUSH_INTERVAL_S)
    assert wait_for(lambda: len(post) == 1, timeout=2.0)
    assert len(post.batches()[0]) == 3
    s.close(0.5)


def test_chunks_at_100_items(sender_env):
    post = HangOnFirstPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    assert s.capture(item("solo")) is True
    clock.advance(ph.FLUSH_INTERVAL_S)  # the solo item becomes due
    assert post.entered.wait(timeout=2.0)  # flusher is now hung inside post
    names = [f"ev{i}" for i in range(40)]
    for i in range(150):
        assert s.capture(item(names[i % 40], n=i)) is True
        clock.advance(2.0)
    post.release.set()
    assert wait_for(lambda: len(post.calls) >= 2, timeout=2.0)
    sizes = [len(json.loads(body)["batch"]) for _, body, _ in post.calls]
    assert sizes == [100, 50]
    s.close(1.0)


def test_body_is_build_batch_output_with_the_stamp_key(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    plain = item(n=7)
    # An item whose ``properties`` is not a mapping still flows through:
    # the envelope is structural, and the sender must not editorialize.
    weird: dict[str, object] = {
        "uuid": str(uuid.uuid4()),
        "event": "weird",
        "properties": ["not", "a", "map"],
    }
    assert s.capture(plain) is True
    assert s.capture(weird) is True
    s.flush(1.0)
    assert len(post) == 1
    url, body, timeout = post.calls[0]
    assert url == POSTHOG_BATCH_URL
    assert timeout == ph.DEFAULT_POST_TIMEOUT_S
    decoded = json.loads(body.decode("utf-8"))
    assert decoded == {"api_key": STAMP.posthog_key, "batch": [plain, weird]}
    # Compact encoding: no separator spaces anywhere on the wire.
    assert b", " not in body
    assert b": " not in body
    s.close(0.5)


# ------------------------------------------------------- failure handling


def test_4xx_dropped_without_retry(sender_env):
    post = RecordingPost(status=499)
    s = make_sender(post)
    for i in range(ph.FLUSH_THRESHOLD):
        assert s.capture(item(n=i)) is True
    assert wait_for(lambda: len(post) == 1, timeout=2.0)
    time.sleep(0.15)  # a buggy retry would land in this window
    assert len(post) == 1
    s.close(0.5)


def test_3xx_dropped_without_retry(sender_env):
    post = RecordingPost(status=302)
    s = make_sender(post)
    assert s.capture(item()) is True
    s.flush(1.0)
    assert len(post) == 1
    s.close(0.5)


def test_5xx_retried_once_then_dropped(sender_env):
    post = RecordingPost(status=500)
    s = make_sender(post)
    for i in range(ph.FLUSH_THRESHOLD):
        assert s.capture(item(n=i)) is True
    assert wait_for(lambda: len(post) == 2, timeout=2.0)
    time.sleep(0.15)  # a retry storm would grow the count here
    assert len(post) == 2
    assert post.calls[0][1] == post.calls[1][1]
    first_uuid = post.batches()[0][0]["uuid"]
    assert post.batches()[1][0]["uuid"] == first_uuid
    s.close(0.5)


def test_exception_in_post_is_swallowed(sender_env):
    calls = {"n": 0}
    lock = threading.Lock()

    def exploding_post(url: str, body: bytes, timeout: float) -> int:
        with lock:
            calls["n"] += 1
        raise RuntimeError("socket exploded")

    s = PostHogSender(
        post=exploding_post,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: True,
    )
    for i in range(ph.FLUSH_THRESHOLD):
        assert s.capture(item(n=i)) is True
    assert wait_for(lambda: calls["n"] >= 2, timeout=2.0)
    time.sleep(0.15)
    assert calls["n"] == 2  # one retry, then the chunk is dropped for good
    # The sender stays healthy: the next capture still queues and sends.
    assert s.capture(item("later", n=99)) is True
    s.flush(1.0)
    assert calls["n"] == 4
    s.close(0.5)


def test_flush_bounded_with_a_hanging_post(sender_env):
    post = HangOnFirstPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    assert s.capture(item()) is True
    clock.advance(ph.FLUSH_INTERVAL_S)
    assert post.entered.wait(timeout=2.0)  # flusher hung inside post
    # Empty queue + in-flight drain is distinct from fully idle; a zero
    # budget must return without trying to force nonexistent queued work.
    s.flush(0.0)
    # Queue more while the flusher is stuck, then force a drain: it must
    # give up after its budget, not wait for the hung POST.
    for i in range(25):
        assert s.capture(item(n=i)) is True
        clock.advance(2.0)
    start = time.monotonic()
    s.flush(0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 1.5
    post.release.set()
    s.close(0.5)


def test_drain_drops_everything_when_the_gate_turns_off(sender_env, monkeypatch):
    """Fail closed at send time too: no stamp, no POST."""
    post = RecordingPost()
    holder = {"stamp": STAMP}
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: holder["stamp"],
        allowed=lambda: True,
    )
    assert s.capture(item()) is True
    holder["stamp"] = None
    build_calls: list[object] = []

    def build_spy(*args: object, **kwargs: object) -> None:
        build_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(ph.envelope, "build_batch", build_spy)
    s._drain()  # direct call makes a missing explicit guard observable
    assert build_calls == []
    assert len(post) == 0
    s.close(0.5)


def test_drain_drops_everything_when_permission_is_withdrawn(sender_env):
    post = RecordingPost()
    permission = {"allowed": True}
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: permission["allowed"],
    )
    assert s.capture(item()) is True
    permission["allowed"] = False
    s.flush(1.0)
    assert len(post) == 0
    assert s._draining is False
    s.close(0.5)


@pytest.mark.parametrize("revocation", ["gate", "permission"])
def test_drain_rechecks_both_gates_before_every_chunk(sender_env, revocation):
    authorization = {"stamp": STAMP, "allowed": True}
    entered = threading.Event()
    release = threading.Event()
    calls: list[bytes] = []

    def slow_post(url: str, body: bytes, timeout: float) -> int:
        calls.append(body)
        if revocation == "gate":
            authorization["stamp"] = None
        else:
            authorization["allowed"] = False
        entered.set()
        release.wait(timeout=0.2)
        return 200

    s = PostHogSender(
        post=slow_post,
        clock=FakeClock(),
        gate=lambda: authorization["stamp"],
        allowed=lambda: authorization["allowed"],
    )
    with s._lock:
        s._queue.extend(item(f"event-{i}", i) for i in range(250))
        s._first_queued_at = s._clock()
    worker = threading.Thread(target=s._drain, daemon=True)
    worker.start()
    assert entered.wait(timeout=1.0)
    release.set()
    worker.join(timeout=1.0)
    assert worker.is_alive() is False
    assert len(calls) == 1
    s.close(0.5)


@pytest.mark.parametrize("revocation", ["gate", "permission"])
def test_drain_rechecks_both_gates_before_retry(sender_env, revocation):
    authorization = {"stamp": STAMP, "allowed": True}
    calls = 0

    def refusing_post(url: str, body: bytes, timeout: float) -> int:
        nonlocal calls
        calls += 1
        if revocation == "gate":
            authorization["stamp"] = None
        else:
            authorization["allowed"] = False
        return 500

    s = PostHogSender(
        post=refusing_post,
        clock=FakeClock(),
        gate=lambda: authorization["stamp"],
        allowed=lambda: authorization["allowed"],
    )
    assert s.capture(item()) is True
    s.flush(1.0)
    assert calls == 1
    s.close(0.5)


def test_drain_rechecks_permission_after_build_before_post(sender_env, monkeypatch):
    permission = {"allowed": True}
    post = RecordingPost()
    real_build_batch = ph.envelope.build_batch

    def build_then_withdraw(items, api_key):
        payload = real_build_batch(items, api_key)
        permission["allowed"] = False
        return payload

    monkeypatch.setattr(ph.envelope, "build_batch", build_then_withdraw)
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: permission["allowed"],
    )
    assert s.capture(item()) is True
    s.flush(1.0)
    assert len(post) == 0
    s.close(0.5)


def test_drain_treats_permission_exception_as_denied(sender_env):
    post = RecordingPost()
    permission = {"raises": False}

    def allowed() -> bool:
        if permission["raises"]:
            raise RuntimeError("consent lookup exploded")
        return True

    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=allowed,
    )
    assert s.capture(item()) is True
    permission["raises"] = True
    s.flush(1.0)
    assert len(post) == 0
    assert s._draining is False
    s.close(0.5)


def test_drain_drops_chunks_with_an_invalid_key(sender_env):
    """build_batch rejecting the envelope means nothing malformed is sent."""
    post = RecordingPost()
    tampered = ReleaseStamp(channel="stable", posthog_key="")
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=lambda: tampered,
        allowed=lambda: True,
    )
    assert s.capture(item()) is True  # capture only needs a stamp, any stamp
    s.flush(1.0)
    assert len(post) == 0
    s.close(0.5)


def test_flush_after_close_is_a_quiet_noop(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    assert s.capture(item()) is True
    s.close(1.0)
    assert s._thread.is_alive() is False
    assert s.capture(item()) is False  # shutdown gate
    calls_before = len(post)
    s.flush(0.5)  # must not raise and must not send anything new
    assert len(post) == calls_before
    s.close(0.5)  # idempotent


def test_capture_crossing_close_is_dropped(sender_env):
    """A capture in flight when ``close()`` lands must not enqueue."""
    gate_blocked = threading.Event()
    release_gate = threading.Event()

    def blocking_gate() -> ReleaseStamp:
        gate_blocked.set()
        release_gate.wait(timeout=5.0)
        return STAMP

    post = RecordingPost()
    s = PostHogSender(
        post=post,
        clock=FakeClock(),
        gate=blocking_gate,
        allowed=lambda: True,
    )
    result: dict[str, bool] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("ok", s.capture(item())), daemon=True
    )
    worker.start()
    assert gate_blocked.wait(timeout=2.0)  # capture is past the outer check
    s.close(1.0)  # flips closed; nothing was ever accepted, so no daemon
    release_gate.set()
    worker.join(timeout=2.0)
    assert result.get("ok") is False  # the in-lock re-check caught it
    assert len(post) == 0


# ------------------------------------------------------------ URL override


def test_url_override_loopback_only(sender_env, monkeypatch):
    post = RecordingPost()
    s = make_sender(post)
    # Unset → production.
    assert s.capture(item("a")) is True
    s.flush(1.0)
    assert post.calls[0][0] == POSTHOG_BATCH_URL
    # Loopback override is honoured (tests must never hit production).
    monkeypatch.setenv(ph.POSTHOG_URL_ENV, "http://127.0.0.1:8787/batch/")
    assert s.capture(item("b")) is True
    s.flush(1.0)
    assert post.calls[1][0] == "http://127.0.0.1:8787/batch/"
    # Any other host is ignored: a user must not be redirectable.
    monkeypatch.setenv(ph.POSTHOG_URL_ENV, "https://evil.example/batch/")
    assert s.capture(item("c")) is True
    s.flush(1.0)
    assert post.calls[2][0] == POSTHOG_BATCH_URL
    # A malformed override is ignored too (fails closed on the port).
    monkeypatch.setenv(ph.POSTHOG_URL_ENV, "http://localhost:bad/")
    assert s.capture(item("d")) is True
    s.flush(1.0)
    assert post.calls[3][0] == POSTHOG_BATCH_URL
    # A non-http(s) scheme is ignored even on a loopback hostname.
    monkeypatch.setenv(ph.POSTHOG_URL_ENV, "ftp://localhost/batch/")
    assert s.capture(item("e")) is True
    s.flush(1.0)
    assert post.calls[4][0] == POSTHOG_BATCH_URL
    s.close(0.5)


@pytest.mark.parametrize(
    "hostile",
    [
        "http://evil.localhost/",
        "http://127.0.0.1.evil.com/",
        "http://localhost.evil.com/",
    ],
)
def test_url_override_rejects_loopback_suffixes(sender_env, monkeypatch, hostile):
    monkeypatch.setenv(ph.POSTHOG_URL_ENV, hostile)
    assert ph._resolve_posthog_url() == POSTHOG_BATCH_URL


# ------------------------------------------------------- thread behaviour


def test_empty_flush_never_starts_a_thread_even_after_close(sender_env):
    idle = make_sender()
    assert idle._thread is None
    idle.flush(0.5)
    assert idle._thread is None

    closed = make_sender()
    closed.close(0.5)
    assert closed._thread is None
    closed.flush(0.5)
    assert closed._thread is None


def test_flush_thread_is_daemon_and_starts_lazily(sender_env):
    s = make_sender()
    assert s._thread is None  # construction starts nothing
    assert s.capture(item()) is True
    assert s._thread is not None
    assert s._thread.daemon is True
    s.close(1.0)
    assert s._thread.is_alive() is False


def test_hostile_clock_kills_only_the_thread(sender_env):
    """A raising injectable must not take the process down."""
    clock = FakeClock()
    post = RecordingPost()
    s = make_sender(post=post, clock=clock)
    assert s.capture(item()) is True
    s._clock = lambda: (_ for _ in ()).throw(RuntimeError("clock exploded"))
    assert wait_for(lambda: s._thread is not None and not s._thread.is_alive())
    s._clock = clock
    assert s.capture(item("after-restart")) is True
    s.flush(1.0)
    assert [event["event"] for batch in post.batches() for event in batch] == [
        "app_opened",
        "after-restart",
    ]
    s.close(0.5)


def test_flush_restarts_a_dead_thread_with_queued_work(sender_env):
    clock = FakeClock()
    post = RecordingPost()
    s = make_sender(post=post, clock=clock)
    assert s.capture(item()) is True
    s._clock = lambda: (_ for _ in ()).throw(RuntimeError("clock exploded"))
    assert wait_for(lambda: s._thread is not None and not s._thread.is_alive())
    s._clock = clock
    s.flush(0.2)
    assert len(post) == 1
    assert s._queue == []
    s.close(0.5)


def test_second_small_batch_waits_its_own_full_interval(sender_env):
    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post=post, clock=clock)
    assert s.capture(item(n=1)) is True
    clock.advance(ph.FLUSH_INTERVAL_S)
    assert wait_for(lambda: len(post) == 1)

    assert s.capture(item(n=2)) is True
    clock.advance(ph.FLUSH_INTERVAL_S - 0.1)
    assert wait_for(lambda: len(post) > 1, timeout=0.15) is False
    clock.advance(0.1)
    assert wait_for(lambda: len(post) == 2)
    s.close(0.5)


def test_capture_snapshots_item_and_properties(sender_env):
    post = RecordingPost()
    s = make_sender(post=post, clock=FakeClock())
    original = item("original", n=7)
    captured_uuid = original["uuid"]
    assert s.capture(original) is True
    original["uuid"] = str(uuid.uuid4())
    original["event"] = "mutated"
    properties = original["properties"]
    assert isinstance(properties, dict)
    properties["n"] = 999
    properties["secret"] = "late mutation"
    s.flush(1.0)
    [sent] = post.batches()[0]
    assert sent["uuid"] == captured_uuid
    assert sent["event"] == "original"
    assert sent["properties"] == {"n": 7}
    s.close(0.5)


def test_fork_child_resets_sender_state_and_locks(sender_env):
    _reset_for_tests()
    singleton = get_sender()
    post = RecordingPost()
    s = make_sender(post=post, clock=FakeClock())
    assert s.capture(item("parent")) is True

    lock_held = threading.Event()
    release_lock = threading.Event()
    sender_lock_held = threading.Event()
    release_sender_lock = threading.Event()

    def hold_parent_lock() -> None:
        with s._lock:
            lock_held.set()
            release_lock.wait(timeout=2.0)

    holder = threading.Thread(target=hold_parent_lock, daemon=True)
    holder.start()
    assert lock_held.wait(timeout=1.0)

    def hold_sender_lock() -> None:
        with ph._sender_lock:
            sender_lock_held.set()
            release_sender_lock.wait(timeout=2.0)

    sender_lock_holder = threading.Thread(target=hold_sender_lock, daemon=True)
    sender_lock_holder.start()
    assert sender_lock_held.wait(timeout=1.0)
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
            signal.alarm(2)
            assert s._queue == []
            assert s._thread is None
            assert s._draining is False
            assert s._force is False
            assert s._lock.acquire(blocking=False)
            s._lock.release()
            assert get_sender() is singleton
            assert s.capture(item("child")) is True
            signal.alarm(0)
            os.write(write_fd, b"ok")
            os._exit(0)
        except BaseException as exc:
            os.write(write_fd, repr(exc).encode("utf-8", errors="replace"))
            os._exit(1)

    os.close(write_fd)
    release_lock.set()
    release_sender_lock.set()
    holder.join(timeout=1.0)
    sender_lock_holder.join(timeout=1.0)
    _, status = os.waitpid(child_pid, 0)
    report = os.read(read_fd, 4096)
    os.close(read_fd)
    assert os.waitstatus_to_exitcode(status) == 0, report.decode()
    assert report == b"ok"
    # The parent's state is untouched by the child's reset.
    assert [queued["event"] for queued in s._queue] == ["parent"]
    s.close(0.5)
    _reset_for_tests()
    assert singleton._thread is None
    # Exercise both singleton and non-singleton registration paths in-process;
    # the real fork above proves the callback is actually registered.
    new_singleton = get_sender()
    another = make_sender()
    ph._after_fork_child()
    assert new_singleton._thread is None
    assert another._thread is None
    _reset_for_tests()
    another.close(0.5)
    # Cover the no-singleton form too.
    ph._after_fork_child()


# -------------------------------------------------------------- never raise


@pytest.mark.parametrize(
    "garbage",
    [
        None,
        42,
        "app_opened",
        {"no_event": 1},
        {"uuid": "6f1b1d3e-4a2b-4c9d-8e7f-0a1b2c3d4e5f"},
        {"event": None},
        {"event": ""},
        _ExplodingMapping({"event": "boom"}),
    ],
)
def test_capture_never_raises_on_garbage(sender_env, garbage):
    post = RecordingPost()
    s = make_sender(post)
    assert s.capture(garbage) is False  # type: ignore[arg-type]
    s.flush(0.5)
    assert len(post) == 0
    s.close(0.5)


def test_capture_rejects_missing_uuid_individually(sender_env):
    s = make_sender(RecordingPost())
    uuidless = item()
    del uuidless["uuid"]
    assert [s.capture(uuidless) for _ in range(25)] == [False] * 25
    assert s._queue == []
    s.close(0.5)


def test_non_json_serializable_item_drops_at_drain(sender_env):
    """A value json cannot encode drops the chunk instead of raising."""

    post = RecordingPost()
    clock = FakeClock()
    s = make_sender(post, clock)
    assert (
        s.capture(
            {
                "uuid": str(uuid.uuid4()),
                "event": "bad",
                "properties": {"obj": object()},
            }
        )
        is True
    )
    s.flush(1.0)  # drain builds the envelope fine, encoding explodes
    assert len(post) == 0
    # The sender is still healthy afterwards.
    assert s.capture(item("good")) is True
    s.flush(1.0)
    assert len(post) == 1
    s.close(0.5)


def test_bad_first_chunk_does_not_drop_later_chunks(sender_env):
    post = RecordingPost()
    s = make_sender(post=post, clock=FakeClock())
    queued = [item(f"event-{i}", n=i) for i in range(150)]
    properties = queued[0]["properties"]
    assert isinstance(properties, dict)
    properties["poison"] = object()
    with s._lock:
        s._queue.extend(queued)
        s._first_queued_at = s._clock()
    s._drain()
    batches = post.batches()
    assert len(batches) == 1
    assert [event["properties"]["n"] for event in batches[0]] == list(range(100, 150))
    s.close(0.5)


def test_unbuildable_first_chunk_does_not_drop_two_later_chunks(sender_env, caplog):
    post = RecordingPost()
    s = make_sender(post=post, clock=FakeClock())
    queued = [item(f"event-{i}", n=i) for i in range(250)]
    del queued[0]["uuid"]
    with s._lock:
        s._queue.extend(queued)
        s._first_queued_at = s._clock()
    with caplog.at_level(logging.DEBUG, logger=SENDER_LOGGER):
        s._drain()
    assert [len(batch) for batch in post.batches()] == [100, 50]
    assert [
        record.message
        for record in caplog.records
        if "build_batch rejected a chunk" in record.message
    ] == ["posthog build_batch rejected a chunk; malformed events dropped"]
    with s._lock:
        s._queue.append({"event": "still-bad"})
        s._first_queued_at = s._clock()
    s._drain()
    assert [
        record.message
        for record in caplog.records
        if "build_batch rejected a chunk" in record.message
    ] == ["posthog build_batch rejected a chunk; malformed events dropped"]
    s.close(0.5)


# ------------------------------------------------------------ default seams


def test_default_allowed_uses_live_v2_permission(sender_env, monkeypatch):
    """The default permission reads ``upload_allowed`` on every capture."""
    post = RecordingPost()
    allowed = {"value": True}
    monkeypatch.setattr(ph.consent_runtime, "upload_allowed", lambda: allowed["value"])

    # Keep v1 enabled so reverting the sender default to emit.is_enabled
    # makes the second capture incorrectly pass this mutation pin.
    from rapid_mlx.telemetry.state import record_consent

    record_consent(True, rapid_mlx_version="0.14.3")
    s = PostHogSender(post=post, clock=FakeClock(), gate=lambda: STAMP)
    assert s.capture(item()) is True
    allowed["value"] = False
    assert s.capture(item(n=2)) is False
    assert s._accepted == 1
    s.close(0.5)


def test_exit_flush_refuses_late_captures(sender_env, monkeypatch):
    post = RecordingPost()
    s = make_sender(post)
    monkeypatch.setattr(ph, "_sender", s)
    assert s.capture(item()) is True
    ph._flush_at_exit()
    assert s.capture(item(n=2)) is False
    assert len(post) == 1
    s.close(0.5)


def test_nonclosing_flush_allows_later_capture(sender_env):
    post = RecordingPost()
    s = make_sender(post)
    assert s.capture(item()) is True
    s.flush(1.0)
    assert s.capture(item(n=2)) is True
    s.flush(1.0)
    assert len(post) == 2
    s.close(0.5)


def test_get_sender_singleton_and_reset(sender_env):
    first = get_sender()
    assert get_sender() is first
    _reset_for_tests()
    second = get_sender()
    assert second is not first
    _reset_for_tests()  # resetting with no singleton is a no-op


def test_install_atexit_registers_once(sender_env, monkeypatch):
    registered: list[Callable[[], None]] = []
    monkeypatch.setattr(ph.atexit, "register", lambda fn: registered.append(fn))
    install_atexit()
    install_atexit()
    assert len(registered) == 1
    registered[0]()  # the exit handler runs clean on an empty sender


def test_reset_unregisters_exit_hook_and_clears_latch(sender_env, monkeypatch):
    registered: list[Callable[[], None]] = []
    unregistered: list[Callable[[], None]] = []
    monkeypatch.setattr(ph.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(ph.atexit, "unregister", lambda fn: unregistered.append(fn))
    install_atexit()
    _reset_for_tests()
    assert unregistered == registered == [ph._flush_at_exit]
    install_atexit()
    assert registered == [ph._flush_at_exit, ph._flush_at_exit]


def test_at_fork_registration_is_idempotent_across_reload(sender_env):
    code = (
        "import importlib, os\n"
        "calls = []\n"
        "os.register_at_fork = lambda **kwargs: calls.append(kwargs)\n"
        "import rapid_mlx.telemetry.posthog_sender as m\n"
        "importlib.reload(m)\n"
        "ours = [c for c in calls if "
        "getattr(c.get('after_in_child'), '__module__', '') == m.__name__]\n"
        "assert len(ours) == 1, calls\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        cwd=str(REPO_ROOT),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    ph._register_at_fork_once()


def test_import_starts_no_thread_no_socket_no_mlx(sender_env):
    """Importing the module itself must have zero runtime side effects."""
    code = (
        "import sys, threading\n"
        "import rapid_mlx.telemetry.posthog_sender as m\n"
        "assert all('posthog' not in t.name for t in threading.enumerate())\n"
        "assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)\n"
        "print(m.__file__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        cwd=str(REPO_ROOT),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    # The module under test must be THIS worktree, not another install.
    assert proc.stdout.strip().startswith(str(REPO_ROOT))


# -------------------------------------------------- real-socket transport


class _StubPostHogHandler(BaseHTTPRequestHandler):
    """Minimal /batch/ stand-in that records requests and answers a status."""

    def do_POST(self) -> None:
        server: HTTPServer = self.server  # type: ignore[assignment]
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        server.requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "content_type": self.headers.get("Content-Type", ""),
                "user_agent": self.headers.get("User-Agent", ""),
                "body": body,
            }
        )
        status: int = server.respond_status  # type: ignore[attr-defined]
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


class _RedirectSourceHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        server: HTTPServer = self.server  # type: ignore[assignment]
        server.requests.append(self.path)  # type: ignore[attr-defined]
        self.send_response(302)
        self.send_header("Location", server.redirect_url)  # type: ignore[attr-defined]
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


class _RedirectSinkHandler(BaseHTTPRequestHandler):
    def _record(self) -> None:
        server: HTTPServer = self.server  # type: ignore[assignment]
        server.requests.append(self.path)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def stub_posthog():
    server = HTTPServer(("127.0.0.1", 0), _StubPostHogHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.respond_status = 200  # type: ignore[attr-defined]
    thread = threading.Thread(
        target=server.serve_forever, name="stub-posthog", daemon=True
    )
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=1.0)


def test_default_post_happy_path_over_a_real_socket(
    sender_env, stub_posthog, monkeypatch
):
    monkeypatch.setenv(
        ph.POSTHOG_URL_ENV,
        f"http://127.0.0.1:{stub_posthog.server_port}/batch/",
    )
    s = PostHogSender(
        post=_REAL_DEFAULT_POST,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: True,
    )
    for i in range(ph.FLUSH_THRESHOLD):
        assert s.capture(item(n=i)) is True
    assert wait_for(lambda: len(stub_posthog.requests) == 1, timeout=3.0)
    req = stub_posthog.requests[0]
    assert req["path"] == "/batch/"
    assert req["content_type"] == "application/json"
    assert req["user_agent"] == "rapid-mlx-telemetry"
    decoded = json.loads(req["body"].decode("utf-8"))
    assert decoded["api_key"] == STAMP.posthog_key
    assert len(decoded["batch"]) == ph.FLUSH_THRESHOLD
    assert decoded["batch"][0]["event"] == "app_opened"
    s.close(1.0)
    assert s._thread.is_alive() is False


def test_default_post_refuses_redirects(sender_env):
    sink = HTTPServer(("127.0.0.1", 0), _RedirectSinkHandler)
    sink.requests = []  # type: ignore[attr-defined]
    source = HTTPServer(("127.0.0.1", 0), _RedirectSourceHandler)
    source.requests = []  # type: ignore[attr-defined]
    source.redirect_url = f"http://127.0.0.1:{sink.server_port}/stolen"  # type: ignore[attr-defined]
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (source, sink)
    ]
    for thread in threads:
        thread.start()
    try:
        status = _REAL_DEFAULT_POST(
            f"http://127.0.0.1:{source.server_port}/batch/",
            b'{"secret":"payload"}',
            1.0,
        )
        assert status == 302
        assert source.requests == ["/batch/"]  # type: ignore[attr-defined]
        assert sink.requests == []  # type: ignore[attr-defined]
    finally:
        for server in (source, sink):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=1.0)


def test_default_post_closes_http_error_response(sender_env, monkeypatch):
    response_body = io.BytesIO(b"error")
    response = HTTPError(
        POSTHOG_BATCH_URL,
        500,
        "server error",
        hdrs=None,
        fp=response_body,
    )

    class RaisingOpener:
        def open(self, req, timeout):
            raise response

    monkeypatch.setattr(ph, "build_opener", lambda *handlers: RaisingOpener())
    assert _REAL_DEFAULT_POST(POSTHOG_BATCH_URL, b"{}", 1.0) == 500
    assert response_body.closed


def test_default_post_retries_a_500_once(sender_env, stub_posthog, monkeypatch):
    stub_posthog.respond_status = 500
    monkeypatch.setenv(
        ph.POSTHOG_URL_ENV,
        f"http://127.0.0.1:{stub_posthog.server_port}/batch/",
    )
    s = PostHogSender(
        post=_REAL_DEFAULT_POST,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: True,
    )
    for i in range(ph.FLUSH_THRESHOLD):
        assert s.capture(item(n=i)) is True
    # urllib surfaces the 500 as an HTTPError, unwrapped to the status 500,
    # so the sender retries exactly once and then drops the chunk.
    assert wait_for(lambda: len(stub_posthog.requests) == 2, timeout=3.0)
    time.sleep(0.15)
    assert len(stub_posthog.requests) == 2
    s.close(1.0)


def test_default_post_swallows_connection_refused(sender_env, monkeypatch):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens there any more
    monkeypatch.setenv(ph.POSTHOG_URL_ENV, f"http://127.0.0.1:{port}/batch/")
    s = PostHogSender(
        post=_REAL_DEFAULT_POST,
        clock=FakeClock(),
        gate=lambda: STAMP,
        allowed=lambda: True,
    )
    for i in range(ph.FLUSH_THRESHOLD):
        assert s.capture(item(n=i)) is True
    s.flush(2.0)  # both attempts refuse instantly; the chunk is dropped
    s.close(1.0)
    assert s._thread.is_alive() is False
