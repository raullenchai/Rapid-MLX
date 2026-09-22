# SPDX-License-Identifier: Apache-2.0
"""PostHog Cloud sender — transport, bounded queue, and burst caps.

This is the telemetry v2 sender component: it accepts already-built
batch items (what :func:`rapid_mlx.telemetry.envelope.build_batch_item`
produced), enforces Orca-style volume limits, and POSTs batches to
PostHog Cloud. Behaviour is modelled on Orca's
``src/main/telemetry/client.ts`` + ``burst-cap.ts``.

Wire contract
=============

Events reach PostHog Cloud by POSTing JSON to the ``/batch/`` endpoint
(:data:`POSTHOG_BATCH_URL`)::

    {"api_key": <project api key>, "batch": [<item>, ...]}

Each item is exactly ``{"uuid", "event", "distinct_id", "timestamp",
"properties"}``. At most ``envelope.MAX_BATCH_ITEMS`` (100) items go in one
request — the sender chunks longer queues itself. Bodies are JSON-encoded
compactly and sent with a fixed ``User-Agent: rapid-mlx-telemetry``.

The seven drop reasons
======================

``capture()`` is bounded, lossy, and never raises. An item is dropped
(returns ``False``, by design silently) when ANY of these holds:

1. **Not an official build** — the official-build gate returns ``None``
   (developer checkout, editable install, CI machine, fork rebuild).
2. **Permission withdrawn** — ``allowed()`` reads ``False`` at capture
   time (environment kill switch active, or consent does not currently
   permit upload). Checked LIVE on every capture, like Orca, so a user
   who withdraws mid-session goes dark on the very next event.
3. **Malformed envelope item** — the top-level ``uuid`` is missing or is
   not parseable as a UUID, or the event name is missing/empty.
4. **Per-session ceiling** — this process has already accepted 1000
   items.
5. **Queue full** — the in-memory queue holds 5000 items; the NEW item
   is dropped (never the oldest, and never by blocking the caller).
6. **Per-event burst cap** — the event name's token bucket (30 events
   per rolling minute, lazy refill, capacity bounded) is empty.
   Unknown event names must not create unbounded buckets: the bucket
   registry itself is capped at 64 names, beyond which items drop.
7. **Shutdown has started** — the exit flush has latched the sender closed
   to new captures before draining its existing queue.

Each cap overflow logs ONCE per cap per process at debug level, then
stays silent for the rest of the process.

Bounded IN-MEMORY buffering — not durable
=========================================

The queue holds accepted items in process memory only. If the process
exits, crashes, or a flush POST fails, those events are gone: this is
best-effort product analytics, not a durable pipeline. Nothing here
writes to disk, and a failed batch is dropped rather than retried more
than once (at most ONE immediate retry for 5xx / transport failures;
3xx/4xx is never retried).

The stdlib transport deliberately honours standard proxy environment
variables; corporate networks commonly require them. A forked child discards
queued work and synchronization/thread state, but deliberately inherits burst
buckets and the accepted session count. That conservative accounting prevents
a fork from resetting volume limits.

The v1 transport (:mod:`rapid_mlx.telemetry.transport`) remains separate.
Importing this module starts no thread and opens no socket — the flush daemon
starts lazily on the first accepted capture, and startup wiring calls
:func:`install_atexit` for the shutdown flush.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rapid_mlx.telemetry import build_gate, consent_runtime, envelope
from rapid_mlx.telemetry.build_gate import ReleaseStamp

logger = logging.getLogger(__name__)

#: PostHog Cloud batch ingest (US region).
POSTHOG_BATCH_URL = "https://us.i.posthog.com/batch/"

#: Test-only URL override. Honoured ONLY when it points at loopback —
#: a user must not be able to redirect telemetry at a third party by
#: accident, and tests must never hit production.
POSTHOG_URL_ENV = "RAPID_MLX_POSTHOG_URL"

DEFAULT_POST_TIMEOUT_S = 5.0
USER_AGENT = "rapid-mlx-telemetry"

#: Burst caps (Orca's burst-cap.ts): 30 events per rolling minute per
#: event name, 64 distinct event names, 1000 accepted items per process,
#: 5000 queued items. Closed-form session/queue checks run before the
#: per-event bucket so rejected items do not consume burst credit.
PER_EVENT_BURST = 30
PER_EVENT_REFILL_PERIOD_S = 60.0
MAX_EVENT_BUCKETS = 64
SESSION_ITEM_CEILING = 1000
MAX_QUEUE_ITEMS = 5000

#: The flush daemon drains when this many items are queued, or when
#: this much time has passed since the first queued item.
FLUSH_THRESHOLD = 20
FLUSH_INTERVAL_S = 10.0

#: Wake-up slice while items are pending, so an injectable clock's
#: advance is noticed promptly. An idle sender (empty queue) parks on
#: the wake event instead and costs nothing.
_POLL_S = 0.1


def _is_loopback_url(url: str) -> bool:
    """True iff ``url`` targets loopback — the only allowed override.

    Same rule as the v1 transport: the match is on the parsed netloc,
    never a substring, and a malformed port fails closed here rather
    than surfacing later from inside the POST.
    """
    try:
        parts = urlparse(url)
        _ = parts.port  # force port validation; raises ValueError if malformed
    except (TypeError, ValueError):
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = (parts.hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1")


def _resolve_posthog_url() -> str:
    """The POST target: production, plus the loopback-only test override.

    Resolved per drain so env changes take effect live. A set-but-
    non-loopback override is IGNORED (production stays in place) —
    unlike the v1 transport's fail-closed ``None``, because this sender
    must keep behaving normally for real users when a stale test env
    var leaks into their environment.
    """
    raw = os.environ.get(POSTHOG_URL_ENV)
    if raw is not None and _is_loopback_url(raw):
        return raw
    return POSTHOG_BATCH_URL


def default_post(url: str, body: bytes, timeout: float) -> int:
    """Stdlib transport: ``(url, body, timeout) -> HTTP status``.

    HTTP-level outcomes come back as the status code — the opener
    raises ``HTTPError`` for >=400, and the code is unwrapped here so
    the sender's retry policy sees a status like the contract promises.
    Redirects are refused so a 3xx cannot forward telemetry to a new
    origin; its status is returned and the sender drops it without retry.
    Standard urllib proxy environment variables are honoured by design.
    Transport failures (DNS, refused connection, timeout) RAISE; the
    sender treats them like 5xx and retries once.
    """
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-type": "application/json", "User-agent": USER_AGENT},
    )
    try:
        with build_opener(_NoRedirect()).open(req, timeout=timeout) as resp:
            return int(resp.status)
    except HTTPError as e:
        # HTTPError holds a file-like response body; close it explicitly
        # so the socket does not linger across retries.
        e.close()
        return int(e.code)


class _TokenBucket:
    """Lazy-refill token bucket: 30 events per rolling minute, bounded."""

    __slots__ = ("tokens", "updated_at")

    def __init__(self, *, now: float) -> None:
        self.tokens = float(PER_EVENT_BURST)
        self.updated_at = now

    def try_take(self, now: float) -> bool:
        elapsed = now - self.updated_at
        if elapsed > 0.0:
            refill = elapsed * (PER_EVENT_BURST / PER_EVENT_REFILL_PERIOD_S)
            # Capacity-bounded: idling never stores more than one burst.
            self.tokens = min(float(PER_EVENT_BURST), self.tokens + refill)
            self.updated_at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class _NoRedirect(HTTPRedirectHandler):
    """Turn redirects into their original 3xx ``HTTPError`` response."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


_instances: weakref.WeakSet[PostHogSender] = weakref.WeakSet()


class PostHogSender:
    """Accepts built envelope items, enforces the caps, POSTs batches.

    Every public method is best-effort: it never raises, and it never
    blocks the caller beyond an explicit ``timeout`` budget. The flush
    thread starts lazily on the first accepted capture and is a daemon,
    so it never keeps the interpreter alive.
    """

    def __init__(
        self,
        *,
        post: Callable[[str, bytes, float], int] | None = None,
        clock: Callable[[], float] | None = None,
        gate: Callable[[], ReleaseStamp | None] | None = None,
        allowed: Callable[[], bool] | None = None,
    ) -> None:
        self._post = post if post is not None else default_post
        self._clock = clock if clock is not None else time.monotonic
        self._gate = gate if gate is not None else build_gate.official_build
        self._allowed = (
            allowed if allowed is not None else consent_runtime.upload_allowed
        )
        self._post_timeout = DEFAULT_POST_TIMEOUT_S
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: list[dict[str, object]] = []
        self._first_queued_at: float | None = None
        self._buckets: dict[str, _TokenBucket] = {}
        self._accepted = 0
        self._force = False
        self._draining = False
        self._closing = False
        self._closed = False
        self._cap_logged: set[str] = set()
        self._invalid_batch_logged = False
        _instances.add(self)

    # ----------------------------------------------------------------- API

    def capture(self, item: Mapping[str, object]) -> bool:
        """Accept one built item; ``True`` iff it was queued for sending.

        Never raises, never blocks, O(1). Drops (returns ``False``) for
        any of the six reasons in the module docstring.
        """
        try:
            return self._capture(item)
        except Exception:
            # Hostile items (a mapping whose accessors explode) drop like
            # any other rejection. KeyboardInterrupt / SystemExit are not
            # Exception subclasses and still propagate untouched.
            return False

    def flush(self, timeout: float = 2.0) -> None:
        """Bounded synchronous drain for shutdown.

        Asks the flush thread to drain everything currently queued and
        waits up to ``timeout`` seconds (real wall-clock time, not the
        injectable clock — the shutdown budget must hold regardless of
        test clocks) for the queue to empty AND any in-flight POST to
        finish, so a returned ``flush`` means everything accepted before
        it was sent or permanently dropped. Returns with items still
        pending if that could not happen in time (e.g. a hung POST).
        Never raises; safe to call after :meth:`close`.
        """
        with self._lock:
            restart_thread = bool(self._queue) and not self._closed
        if restart_thread:
            self._ensure_thread()
        with self._lock:
            if not self._queue and not self._draining:
                return
            self._force = True
        self._wake.set()
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                if not self._queue and not self._draining:
                    return
                if self._queue:
                    # Items that arrived mid-flush must drain too.
                    self._force = True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._wake.wait(timeout=min(_POLL_S, remaining))

    def close(self, timeout: float = 2.0) -> None:
        """Stop accepting captures and drain what is queued. Idempotent.

        Registers nothing global — process-exit wiring lives in
        :func:`install_atexit`. After ``close()`` every ``capture``
        returns ``False``.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._wake.set()
        with self._lifecycle_lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))

    # ------------------------------------------------------------ internals

    def _capture(self, item: Mapping[str, object]) -> bool:
        if self._closed or self._closing:
            return False
        # Gate and permission are LIVE per capture: a build that stops
        # being official, or a user who withdraws mid-session, goes dark
        # on the very next event.
        if self._gate() is None:
            return False
        if not self._allowed():
            return False
        snapshot = envelope._snapshot_item(item)
        if snapshot is None:
            return False
        event = snapshot.get("event")
        if not isinstance(event, str) or not event:
            # The per-event caps key off the event name; an item without
            # one cannot be accounted for, so it never queues.
            return False
        now = self._clock()
        with self._lock:
            if self._closed or self._closing:
                # ``close()`` landed while this capture sat between the
                # outer check and here: a late arrival against a
                # flushing client is dropped.
                return False
            drop_reason = self._admit_locked(event, now)
            log_cap: str | None = None
            if drop_reason is None:
                self._queue.append(snapshot)
                self._accepted += 1
                if self._first_queued_at is None:
                    self._first_queued_at = now
            elif drop_reason not in self._cap_logged:
                log_cap = drop_reason
                self._cap_logged.add(drop_reason)
        if drop_reason is not None:
            if log_cap is not None:
                logger.debug(
                    "posthog cap %s reached; further drops of this cap "
                    "stay silent for this process",
                    log_cap,
                )
            return False
        self._ensure_thread()
        self._wake.set()
        return True

    def _admit_locked(self, event: str, now: float) -> str | None:
        """The cap checks, under the lock. ``None`` means admitted.

        Closed-form checks run first: session ceiling, queue, then burst
        bucket. A dropped item never displaces a queued one or consumes
        burst credit unless the burst cap itself is the reason.
        """
        if self._accepted >= SESSION_ITEM_CEILING:
            return "session-ceiling"
        if len(self._queue) >= MAX_QUEUE_ITEMS:
            return "queue-full"
        bucket = self._buckets.get(event)
        if bucket is None:
            # Bounded registry: unknown event names beyond the cap drop
            # instead of growing the mapping without limit.
            if len(self._buckets) >= MAX_EVENT_BUCKETS:
                return "event-bucket-registry"
            bucket = _TokenBucket(now=now)
            self._buckets[event] = bucket
        if not bucket.try_take(now):
            return "per-event-burst"
        return None

    def _ensure_thread(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(
                target=self._run, name="rapid-mlx-posthog", daemon=True
            )
            thread.start()
            self._thread = thread

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._closed:
                        break
                    queued = len(self._queue)
                    forced = self._force
                    first = self._first_queued_at
                if queued == 0:
                    # Nothing pending: park until the next capture, a
                    # forced flush, or close().
                    self._wake.wait()
                    self._wake.clear()
                    continue
                due = (
                    forced
                    or queued >= FLUSH_THRESHOLD
                    or (first is not None and self._clock() - first >= FLUSH_INTERVAL_S)
                )
                if due:
                    self._drain()
                else:
                    self._wake.wait(timeout=_POLL_S)
                    self._wake.clear()
            # Closed: one final drain so shutdown loses nothing already
            # queued.
            self._drain()
        except Exception:
            # The flush thread must never take the process down. The
            # expected source is a hostile injectable (clock/gate);
            # anything already queued stays for a later flush or close.
            pass

    def _drain(self) -> None:
        """Pop everything queued and POST it in chunks. Never raises."""
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
            self._first_queued_at = None
            self._force = False
            # ``_draining`` covers the whole POST phase so ``flush`` can
            # wait for it, and is set in the SAME critical section as the
            # pop so no observer can see "empty queue, not draining".
            if batch:
                self._draining = True
        if not batch:
            return
        try:
            url = _resolve_posthog_url()
            for start in range(0, len(batch), envelope.MAX_BATCH_ITEMS):
                stamp = self._permission_stamp()
                if stamp is None:
                    # Drop this chunk and everything after it when the build
                    # gate or permission changes during a multi-POST drain.
                    break
                chunk = batch[start : start + envelope.MAX_BATCH_ITEMS]
                try:
                    payload = envelope.build_batch(chunk, stamp.posthog_key)
                    if payload is None:
                        # build_batch rejected the envelope (e.g. a tampered
                        # key): drop the chunk rather than send malformed.
                        self._log_invalid_batch_once()
                        continue
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                except Exception:
                    # A malformed chunk must not discard later chunks that
                    # were already popped from the queue.
                    # envelope.build_batch_item items are registry-validated and always JSON-serialisable; only a hand-built item can poison a chunk, and dropping it is accepted.
                    continue
                if not self._send_chunk(url, body):
                    break
        finally:
            with self._lock:
                self._draining = False

    def _permission_stamp(self) -> ReleaseStamp | None:
        """Read both live send gates; any exception is a denial."""
        try:
            stamp = self._gate()
            if stamp is None or not self._allowed():
                return None
            return stamp
        except Exception:
            return None

    def _log_invalid_batch_once(self) -> None:
        with self._lock:
            if self._invalid_batch_logged:
                return
            self._invalid_batch_logged = True
        logger.debug("posthog build_batch rejected a chunk; malformed events dropped")

    def _after_fork_child(self) -> None:
        """Discard inherited work and replace synchronization in a child."""
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._wake = threading.Event()
        self._queue = []
        self._first_queued_at = None
        self._thread = None
        self._draining = False
        self._force = False
        self._closing = False

    def _send_chunk(self, url: str, body: bytes) -> bool:
        """POST one chunk with the one-retry discipline.

        The permission check is deliberately duplicated in the drain loop and
        before each attempt; this per-attempt check is the load-bearing one.
        Returns ``False`` only when a live gate refuses immediately before an
        attempt, telling the drain to drop every remaining chunk. 2xx: done.
        3xx/4xx: retrying will not change the answer — drop this chunk.
        5xx or a transport failure: ONE immediate retry, then drop for
        good. No retry storm, nothing raised.
        """
        if self._permission_stamp() is None:
            return False
        try:
            status = self._post(url, body, self._post_timeout)
        except Exception:
            status = None
        if isinstance(status, int) and 200 <= status < 300:
            return True
        if isinstance(status, int) and 300 <= status < 500:
            return True
        if self._permission_stamp() is None:
            return False
        try:
            self._post(url, body, self._post_timeout)
        except Exception:
            pass
        return True


# -------------------------------------------------------------- singleton

_sender: PostHogSender | None = None
_sender_lock = threading.Lock()
_atexit_installed = False
_at_fork_installed = globals().get("_at_fork_installed", False)


def _after_fork_child() -> None:
    """Reset every live sender without touching possibly locked state."""
    global _sender_lock
    _sender_lock = threading.Lock()
    singleton = _sender
    if singleton is not None:
        singleton._after_fork_child()
    for instance in list(_instances):
        if instance is not singleton:
            instance._after_fork_child()


def _register_at_fork_once() -> None:
    """Register the child reset once, including across module reloads."""
    global _at_fork_installed
    if _at_fork_installed:
        return
    os.register_at_fork(after_in_child=_after_fork_child)
    _at_fork_installed = True


_register_at_fork_once()


def get_sender() -> PostHogSender:
    """Process-singleton sender, constructed on first call.

    Constructing (and importing) starts no thread and opens no socket;
    the flush daemon starts on the first ACCEPTED capture.
    """
    global _sender
    with _sender_lock:
        if _sender is None:
            _sender = PostHogSender()
        return _sender


def _reset_for_tests() -> None:
    """Drop singleton and process-exit wiring. Tests-only seam."""
    global _atexit_installed, _sender
    with _sender_lock:
        sender, _sender = _sender, None
        unregister_exit = _atexit_installed
        _atexit_installed = False
    if unregister_exit:
        atexit.unregister(_flush_at_exit)
    if sender is not None:
        sender.close(timeout=0.5)


def _flush_at_exit() -> None:
    """Drain for at most two seconds during interpreter shutdown.

    A black-holed endpoint can therefore add up to about two seconds to a
    short CLI command at exit. This bounded delay is intentional Orca parity.
    """
    sender = get_sender()
    with sender._lock:
        sender._closing = True
    sender.flush(2.0)


def install_atexit() -> None:
    """Register the process-exit flush. Idempotent; called by the wiring PR.

    Import registers only the idempotent fork-safety hook. Neither import nor
    ``close()`` registers an exit callback; only this explicit call hooks the
    sender into interpreter shutdown. A black-holed endpoint can add up to
    about two seconds to a short CLI command at exit; the bounded delay is
    intentional Orca parity.
    """
    global _atexit_installed
    with _sender_lock:
        if _atexit_installed:
            return
        _atexit_installed = True
    atexit.register(_flush_at_exit)
