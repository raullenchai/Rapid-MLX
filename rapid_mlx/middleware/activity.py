"""Track HTTP model-work activity through the complete ASGI response."""

import threading
import time


class ActivityTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = 0
        self._last_activity = time.monotonic()

    def start(self):
        with self._lock:
            self._in_flight += 1
            self._last_activity = time.monotonic()

    def finish(self):
        with self._lock:
            self._in_flight -= 1
            self._last_activity = time.monotonic()

    def snapshot(self):
        with self._lock:
            return {
                "in_flight": self._in_flight,
                "idle_seconds": 0.0
                if self._in_flight
                else max(0.0, time.monotonic() - self._last_activity),
            }


activity_tracker = ActivityTracker()


class ActivityMiddleware:
    def __init__(self, app, tracker=activity_tracker):
        self.app = app
        self.tracker = tracker

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] == "GET"
            or not scope["path"].startswith("/v1/")
        ):
            await self.app(scope, receive, send)
            return
        self.tracker.start()
        try:
            await self.app(scope, receive, send)
        finally:
            self.tracker.finish()
