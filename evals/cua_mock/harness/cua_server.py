"""Eval harness: one PerceptionSession behind a local socket.

Model channel   POST /       {"op": observe|windows|apps|front|handback|wait|handoff|open_url|
                               read|click|fill|type|key|scroll|drag|action, ...} -> {"text": ...}
User channel    POST /human  {"op": fill|type|click|key|done, ...}

The user channel is the person at the Mac typing during a handoff (a
password, a code). The model is never given it. Whether to ask the user
before a commit is the brain's call, made in its reply; the session has no
approvals. With MOCK_ORACLE set, the user's steps are also reported to the
mock sites' oracle log for scoring.
"""

import json
import os
import sys
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# perception lands with the perception-session PR; import it through the
# package so the import order holds before and after that PR.
from rapid_mlx.computer_use import backend, perception
from rapid_mlx.computer_use.errors import ComputerUseError

SESSION = perception.PerceptionSession()
LOG = open(os.environ.get("CUA_SERVER_LOG", "cua_server.log"), "a", buffering=1)
ORACLE = os.environ.get("MOCK_ORACLE")


def _report(kind: str, payload: dict) -> None:
    LOG.write(f"EVENT {kind} {json.dumps(payload, ensure_ascii=False)}\n")
    if not ORACLE:
        return
    path = "/api/human" if kind == "human_input" else None
    if kind == "handoff":
        # The user's own typing during a handoff is a human step; the oracle
        # learns what it was for from the reason.
        path, payload = "/api/human", {"op": "handoff", "label": payload["reason"]}
    if path:
        req = urllib.request.Request(
            ORACLE + path,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5).read()


SESSION.on_event = _report


def _receipt(out: dict, full: bool) -> str:
    lines = ["receipt " + json.dumps(out["receipt"], ensure_ascii=False)]
    lines.append(out["observation"].render(full=full))
    return "\n".join(lines)


def handle(req: dict) -> str:
    op = req.pop("op")
    full = bool(req.pop("full", True))
    if op == "observe":
        obs = SESSION.observe(req["app"], req.get("window_id"))
        if req.get("find"):
            return obs.find(str(req["find"]))
        everything = str(req.get("all", "")).lower() in ("1", "true", "yes")
        return obs.render(full=not req.get("changes_only"), everything=everything)
    if op == "windows":
        rows = backend.list_windows(req["app"])
        return "\n".join(
            f'window {w["window_id"]} "{w.get("title", "")}" {w.get("width")}x{w.get("height")}'
            for w in rows
            if w.get("title")
            or (w.get("width", 0) >= 200 and w.get("height", 0) >= 150)
        )
    if op == "front":
        return json.dumps(SESSION.take_front(req["app"], req["window_id"]))
    if op == "handback":
        return json.dumps(SESSION.hand_back())
    if op == "apps":
        from AppKit import NSWorkspace

        return "\n".join(
            f"{a.localizedName()} ({a.bundleIdentifier()})"
            for a in NSWorkspace.sharedWorkspace().runningApplications()
            if a.activationPolicy() == 0
        )
    if op in ("wait", "handoff"):
        wid = req.pop("window_id")
        if op == "wait":
            out = SESSION.wait(wid, **req)
        else:
            out = SESSION.handoff(wid, req.pop("reason"), **req)
        partly = "" if out.get("complete", True) else " (page only partly read)"
        return f"{op} met={out['met']}{partly}\n" + out["observation"].render(full=full)
    if op == "read":
        out = SESSION.read(
            req["ref"],
            start=int(req.get("start", 0)),
            max_chars=int(req.get("max_chars", backend.MAX_READ_CHARS)),
        )
        end = out["start"] + len(out["text"])
        return f"chars {out['start']}-{end} of {out['total_chars']}\n{out['text']}"
    if op == "open_url":
        return _receipt(SESSION.open_url(req["window_id"], req["url"]), full)
    return _receipt(SESSION.act(op, req.pop("ref", None), **req), full)


def handle_human(req: dict) -> str:
    op = req.pop("op")
    if op == "done":
        SESSION.human_done(req["window_id"])
        return "ok"
    out = SESSION.human_act(op, req.pop("ref", None), **req)
    return "receipt " + json.dumps(out["receipt"], ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        human = self.path.rstrip("/") == "/human"
        # What the user types (a password, a code) never reaches the log.
        logged = {**body, "text": "<redacted>"} if human and "text" in body else body
        LOG.write(
            ("HUMAN " if human else "") + json.dumps(logged, ensure_ascii=False) + "\n"
        )
        try:
            text = (handle_human if human else handle)(dict(body))
            status = 200
        except ComputerUseError as exc:
            text, status = f"error {exc.code}: {exc.message}", 200
        except Exception as exc:  # noqa: BLE001
            text, status = (
                "internal " + "".join(traceback.format_exception(exc))[-1500:],
                500,
            )
        LOG.write(text[:4000] + "\n---\n")
        data = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
