"""Local Research Loop web server. Python standard library only."""

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
INBOX_LOCK = threading.Lock()


def records(path, offset=0):
    """Read complete JSONL records only; retain partial writes for the next poll."""
    if not path.exists():
        return [], offset
    rows = []
    with path.open("rb") as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            offset = stream.tell()
            try:
                rows.append((offset, json.loads(line)))
            except (ValueError, UnicodeDecodeError):
                continue
    return rows, offset


def event_stream(path, speed=0, after=0, alive=lambda: True, sleep=time.sleep):
    """SSE with byte-offset IDs, heartbeats, and original-timing replay."""
    previous = None
    offset = after
    heartbeat = time.monotonic()
    while True:
        rows, offset = records(path, offset)
        for event_id, event in rows:
            if speed and previous is not None:
                delay = max(0, (event.get("ts", previous) - previous) / speed)
                # Keep long gaps connected and compress dense training curves.
                if event.get("event") == "curve":
                    delay = min(delay, 0.15)
                while delay > 0:
                    interval = min(delay, 5)
                    sleep(interval)
                    delay -= interval
                    yield b": replay\n\n"
            previous = event.get("ts", previous)
            yield f"id: {event_id}\ndata: {json.dumps(event)}\n\n".encode()
            if event.get("event") == "done":
                return
        if not alive():
            if records(path, offset)[0]:
                continue
            yield b"event: closed\ndata: {}\n\n"
            return
        if time.monotonic() - heartbeat >= 5:
            yield b": waiting\n\n"
            heartbeat = time.monotonic()
        sleep(0.2)


def append_idea(run, data):
    hypothesis = data.get("hypothesis")
    change = data.get("change", "")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("Write a hypothesis first.")
    if not isinstance(change, str) or len(hypothesis) + len(change) > 16000:
        raise ValueError("Idea must be text under 16,000 characters.")
    with INBOX_LOCK, (run / "inbox.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"hypothesis": hypothesis.strip(), "change": change}) + "\n"
        )


class App(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, base_url, model, rounds=10, runs=None):
        super().__init__(address, Handler)
        self.base_url = base_url.rstrip("/")
        self.model, self.rounds = model, rounds
        self.runs = runs or ROOT / "runs"
        self.runs.mkdir(exist_ok=True)
        self.processes = {}

    def run_path(self, name):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Invalid run ID.")
        path = self.runs / name
        if not path.is_dir() or path.is_symlink():
            raise ValueError("Run not found.")
        return path

    def info(self, run):
        meta = run / "run.json"
        return (
            json.loads(meta.read_text())
            if meta.exists()
            else {"id": run.name, "task": "grokking", "model": self.model}
        )

    def active(self, run):
        process = self.processes.get(run.name)
        return process is not None and process.poll() is None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        try:
            if parsed.path == "/api/runs":
                runs = [
                    dict(
                        self.server.info(p),
                        active=self.server.active(p),
                        paused=(p / "PAUSE").exists(),
                    )
                    for p in sorted(self.server.runs.iterdir(), reverse=True)
                    if p.is_dir() and not p.is_symlink()
                ]
                return self.json(
                    {
                        "runs": runs,
                        "model": self.server.model,
                        "tasks": [
                            p.name
                            for p in (ROOT / "tasks").iterdir()
                            if (p / "task.json").exists()
                        ],
                    }
                )
            if (
                len(parts) == 4
                and parts[:2] == ["api", "runs"]
                and parts[3] == "events"
            ):
                run = self.server.run_path(parts[2])
                query = parse_qs(parsed.query)
                speed = (
                    float(query.get("speed", ["8"])[0])
                    if query.get("replay") == ["1"]
                    else 0
                )
                if not math.isfinite(speed) or speed < 0 or speed > 1000:
                    raise ValueError("Invalid replay speed.")
                after = max(0, int(self.headers.get("Last-Event-ID", "0")))
                self.stream_headers()
                for frame in event_stream(
                    run / "events.jsonl",
                    speed,
                    after,
                    lambda: (
                        self.server.active(run)
                        or (not speed and run.name not in self.server.processes)
                    ),
                ):
                    self.wfile.write(frame)
                    self.wfile.flush()
                return
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if parsed.path not in assets:
                return self.json({"error": "Not found"}, 404)
            file, mime = assets[parsed.path]
            body = (ROOT / "web" / file).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, OSError) as exc:
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.json({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            # Mutations belong to this local page, not cross-origin websites.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.json({"error": "Cross-origin request rejected"}, 403)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100000:
                raise ValueError("Expected a JSON body under 100 KB.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            parts = urlparse(self.path).path.strip("/").split("/")
            if parts == ["api", "runs"]:
                task, model = (
                    data.get("task", "grokking"),
                    data.get("model", self.server.model),
                )
                if (
                    not isinstance(task, str)
                    or not re.fullmatch(r"[\w-]+", task)
                    or not (ROOT / "tasks" / task / "task.json").is_file()
                ):
                    raise ValueError("Unknown task.")
                if not isinstance(model, str) or not model.strip():
                    raise ValueError("Model is required.")
                rounds = int(data.get("rounds", self.server.rounds))
                if not 1 <= rounds <= 100:
                    raise ValueError("Rounds must be between 1 and 100.")
                name = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
                run = self.server.runs / name
                run.mkdir()
                meta = {"id": name, "task": task, "model": model, "rounds": rounds}
                (run / "run.json").write_text(json.dumps(meta))
                with (run / "loop.log").open("w") as log:
                    self.server.processes[name] = subprocess.Popen(
                        [
                            sys.executable,
                            str(ROOT / "loop.py"),
                            str(ROOT / "tasks" / task),
                            str(run),
                            "--base-url",
                            self.server.base_url,
                            "--model",
                            model,
                            "--rounds",
                            str(rounds),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                return self.json(meta, 201)
            if parts == ["api", "chat"]:
                return self.chat(data)
            if len(parts) == 4 and parts[:2] == ["api", "runs"]:
                run = self.server.run_path(parts[2])
                action = parts[3]
                if action == "idea":
                    if not self.server.active(run):
                        raise ValueError(
                            "Run has ended. Start a new run to test an idea."
                        )
                    append_idea(run, data)
                elif action == "pause":
                    (run / "PAUSE").touch()
                elif action in ("resume", "stop"):
                    if action == "stop":
                        (run / "STOP").touch()
                    (run / "PAUSE").unlink(missing_ok=True)
                else:
                    return self.json({"error": "Not found"}, 404)
                return self.json({"ok": True, "action": action})
            self.json({"error": "Not found"}, 404)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.json({"error": str(exc)}, 400)

    def chat(self, data):
        run = self.server.run_path(data.get("run_id", ""))
        meta = self.server.info(run)
        messages = data.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or len(messages) > 40
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("user", "assistant")
                or not isinstance(m.get("content"), str)
                for m in messages
            )
        ):
            raise ValueError("Expected 1–40 user/assistant messages.")
        events, _ = records(run / "events.jsonl")
        history = [
            {
                k: e[k]
                for k in (
                    "event",
                    "round",
                    "hypothesis",
                    "change",
                    "verdict",
                    "value",
                    "result",
                )
                if k in e
            }
            for _, e in events
            if e["event"] in ("idea", "result", "baseline")
        ]
        program = (ROOT / "tasks" / meta["task"] / "program.md").read_text()
        system = (
            "Discuss concrete testable ideas briefly. Experiments run locally. Do not claim to write papers or invent results.\n"
            + program
            + "\nObserved run history:\n"
            + json.dumps(history)
        )
        body = json.dumps(
            {
                "model": meta["model"],
                "messages": [{"role": "system", "content": system}] + messages,
                "stream": True,
                "max_tokens": 1200,
            }
        ).encode()
        request = Request(
            self.server.base_url + "/chat/completions",
            body,
            {"Content-Type": "application/json"},
        )
        try:
            upstream = urlopen(request, timeout=1800)
        except (HTTPError, URLError) as exc:
            return self.json({"error": f"Local model unavailable: {exc}"}, 502)
        with upstream:
            self.stream_headers()
            try:
                for line in upstream:
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4900)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    app = App((args.host, args.port), args.base_url, args.model, args.rounds)
    print(f"Research Loop: http://{args.host}:{args.port}", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()
