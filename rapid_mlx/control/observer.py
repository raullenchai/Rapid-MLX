# SPDX-License-Identifier: Apache-2.0
"""Supervised runner observation: no receive interception, bodies, or request I/O."""

import asyncio
import atexit
import contextvars
import http.client
import os
import queue
import threading
import time
import uuid

from . import RUNNER_ENV, SECRET_ENV, SOCKET_ENV
from .protocol import request

_current = contextvars.ContextVar("rapid_call_metadata", default=None)
_delivery = None


def metadata(
    *,
    model=None,
    prompt_tokens=None,
    completion_tokens=None,
    outcome=None,
    ttft_ms=None,
):
    """Adapters share scalar counters with the outer observer, including child tasks."""
    current = _current.get()
    if current is not None:
        for key, value in locals().copy().items():
            if key != "current" and value is not None:
                current[key] = value


def usage(output):
    timing = getattr(output, "timing_metrics", None) or {}
    metadata(
        prompt_tokens=getattr(output, "prompt_tokens", None),
        completion_tokens=getattr(output, "completion_tokens", None),
        ttft_ms=timing.get(
            "time_to_first_token_ms", getattr(output, "time_to_first_token_ms", None)
        ),
    )


class Delivery:
    def __init__(self, *, capacity=1024):
        self.queue = queue.Queue(maxsize=capacity)
        self.queue_dropped = 0
        self.transport_dropped = 0
        self.skipped = 0
        self.closed = threading.Event()
        self.pending = threading.Event()
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="rapid-call-log"
        )

    @property
    def dropped(self):
        return self.queue_dropped + self.transport_dropped

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.queue_dropped += 1
        self.pending.set()

    def flush(self, batch):
        events = [item for item in batch if "event" in item]
        records = [item for item in batch if "event" not in item]
        try:
            request(
                os.environ[SOCKET_ENV],
                os.environ[SECRET_ENV],
                "POST",
                {
                    "pid": os.getpid(),
                    "records": records,
                    "events": events,
                    "dropped": self.dropped,
                    "skipped": self.skipped,
                },
                target="/records",
            )
        except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException):
            self.transport_dropped += len(batch)

    def close(self):
        self.closed.set()
        self.pending.set()
        self.thread.join(timeout=0.2)
        if self.thread.is_alive():
            return  # A timed socket operation already owns delivery.
        deadline = time.monotonic() + 0.5
        while not self.queue.empty() and time.monotonic() < deadline:
            batch = []
            while len(batch) < 8 and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            self.flush(batch)
        self.queue_dropped += self.queue.qsize()

    def run(self):
        while not self.closed.is_set():
            # Records and shutdown wake immediately; idle drop reports must not
            # turn into ten empty control-socket requests per second.
            self.pending.wait(5.0)
            self.pending.clear()
            if self.closed.is_set():
                break
            batch = []
            while len(batch) < 8:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            if not self.queue.empty():
                self.pending.set()
            # Empty reports keep drops visible even when no further request arrives.
            self.flush(batch)


def delivery():
    global _delivery
    if _delivery is None:
        _delivery = Delivery()
        _delivery.thread.start()
        atexit.register(_delivery.close)
    return _delivery


def residency_event(event, *, model):
    """Bounded, nonblocking delivery from residency's lock-held mutations."""
    if os.environ.get(RUNNER_ENV) == "1":
        delivery().enqueue({"event": event, "model": model})


class CallObserver:
    def __init__(self, app, sink=None):
        self.app = app
        self.sink = sink if sink is not None else delivery()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("_rapid_call_observed"):
            await self.app(scope, receive, send)
            return
        scope["_rapid_call_observed"] = True
        data = scope.setdefault("state", {}).setdefault("rapid_call", {})
        token = _current.set(data)
        started = time.perf_counter()
        status = None
        first = None
        complete = False
        outcome = "ok"
        emitted = False

        async def observe_send(message):
            nonlocal status, first, complete
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                if message.get("body") and first is None:
                    first = (time.perf_counter() - started) * 1000
                if not message.get("more_body", False):
                    complete = True
            await send(message)

        try:
            await self.app(scope, receive, observe_send)
        except (asyncio.CancelledError, OSError):
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            route = getattr(scope.get("route"), "path", None)
            eligible = (
                route and route.startswith("/v1/") and route.rstrip("/") != "/v1/models"
            )
            if not eligible:
                self.sink.skipped += 1
            if not emitted and eligible:
                emitted = True
                peer = scope.get("_rapid_mlx_transport_peer", scope.get("client"))
                headers = dict(scope.get("headers", ()))
                client = headers.get(
                    b"x-rapid-client", headers.get(b"user-agent", b"")
                )[:128].decode("utf-8", "replace")
                route = getattr(scope.get("route"), "path", None)
                self.sink.enqueue(
                    {
                        "ts": time.time(),
                        "request_id": uuid.uuid4().hex,
                        "client": f"{peer[0] if peer else 'unknown'} {client}"[:256],
                        "app": None,
                        "model": str(data["model"])[:256]
                        if data.get("model") is not None
                        else None,
                        "route": route or "<unmatched>",
                        "http_status": status if status is not None else 500,
                        "outcome": data.get("outcome")
                        or (
                            outcome
                            if outcome != "ok"
                            else "cancelled"
                            if not complete
                            else "cancelled"
                            if status == 499
                            else "error"
                            if status and status >= 400
                            else "ok"
                        ),
                        "prompt_tokens": data.get("prompt_tokens"),
                        "completion_tokens": data.get("completion_tokens"),
                        "ttft_ms": data.get("ttft_ms", first),
                        "duration_ms": (time.perf_counter() - started) * 1000,
                    }
                )
            _current.reset(token)


def install(app):
    return CallObserver(app) if os.environ.get(RUNNER_ENV) == "1" else app
