"""Local mock sites for computer-use evaluation (no real accounts or money).

Three sites model the hand-the-Mac-to-the-agent scenarios:

* ValueMart  (/mart/)     — order history and a support chat in an iframe
                            whose agent replies with delays and pushes back;
* BulkFresh  (/fresh/)    — grocery search over a long catalog, cookie
                            banner, out-of-stock substitution, delivery
                            slots, checkout;
* CityPower  (/paycity/)  — utility bill pay behind a login and a one-time
                            code (the human's step), review, submit.

Pages report what happens through POST /api/log; GET /api/state returns the
event log that ``mockctl.py check`` scores; GET /api/run names the run so
pages drop browser state left over from an earlier one, and page events from
an earlier run are rejected. The human's side of a run — approving a
money step, typing a password or code — is recorded with POST /api/approve
and POST /api/human so the scorer can tell who did what and in which order.

    python evals/cua_mock/server.py [--port 8810]
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITES = Path(__file__).resolve().parent / "sites"
_LOCK = threading.Lock()
_EVENTS: list[dict] = []
_RUN = {"id": uuid.uuid4().hex[:8]}


_SEQ = itertools.count(1)


def _append(kind: str, payload: dict) -> None:
    # Caller holds _LOCK. ``n`` orders events strictly (the scorer compares
    # it); ``t`` is for people.
    _EVENTS.append(
        {**payload, "n": next(_SEQ), "t": round(time.time(), 3), "kind": kind}
    )


def _record(kind: str, payload: dict) -> None:
    with _LOCK:
        _append(kind, payload)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITES), **kwargs)

    def _json(self, status: int, body: object) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/state":
            with _LOCK:
                return self._json(200, {"events": list(_EVENTS)})
        if self.path == "/api/run":
            return self._json(200, {"run": _RUN["id"]})
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {"raw": raw.decode(errors="replace")}
        if self.path == "/api/log":
            # Check and append under one lock, so a reset between them cannot
            # let an event from the old run into the new one.
            with _LOCK:
                fresh = isinstance(body, dict) and body.get("run") == _RUN["id"]
                if fresh:
                    _append("page", body)
            if not fresh:
                # A tab (or a timer in one) from before the last reset.
                return self._json(409, {"error": "stale run; reload the page"})
        elif self.path == "/api/approve":
            _record("approve", body)
        elif self.path == "/api/human":
            _record("human", body)
        elif self.path == "/api/reset":
            with _LOCK:
                _EVENTS.clear()
                _RUN["id"] = uuid.uuid4().hex[:8]
                _append("reset", body)
        else:
            return self._json(404, {"error": "unknown endpoint"})
        return self._json(200, {"ok": True})

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8810)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
