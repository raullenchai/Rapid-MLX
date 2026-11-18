"""Local Research Loop web server. Python standard library only."""

import argparse
import difflib
import json
import math
import re
import shutil
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
INBOX_LOCK = threading.RLock()
SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"


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
    with INBOX_LOCK, safe_path(run, "inbox.jsonl", exists=False).open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "hypothesis": hypothesis.strip(),
                    "change": change,
                    "source": "you",
                    "ts": time.time(),
                }
            )
            + "\n"
        )


def safe_path(root, *parts, exists=True):
    """Reject traversal and every symlink component, even within the root."""
    path = Path(root)
    if path.is_symlink():
        raise ValueError("Symlinks are not allowed.")
    for part in parts:
        if (
            not isinstance(part, str)
            or not re.fullmatch(r"[a-zA-Z0-9_.-]+", part)
            or part in (".", "..")
        ):
            raise ValueError("Invalid path.")
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed.")
    if exists and not path.exists():
        raise ValueError("Not found.")
    return path


def read_json(root, name):
    return json.loads(safe_path(root, name).read_text())


def jsonl(root, name):
    return [row for _, row in records(safe_path(root, name, exists=False))[0]]


def append_message(project, role, content, **extra):
    with safe_path(project, "discussion.jsonl", exists=False).open("a") as stream:
        stream.write(
            json.dumps(dict(role=role, content=content, ts=time.time(), **extra)) + "\n"
        )


def parse_models(value, base_url, model):
    models = {}
    if value:
        for entry in value.split(","):
            key, separator, url = entry.partition("=")
            if (
                not separator
                or not re.fullmatch(r"[a-zA-Z0-9_.-]+", key)
                or urlparse(url).scheme not in ("http", "https")
                or not urlparse(url).netloc
            ):
                raise ValueError(
                    "Use --models name=http://host:port/v1,name2=http://host:port/v1"
                )
            if key in models:
                raise ValueError("Duplicate model name.")
            models[key] = url.rstrip("/")
    return models or {model: base_url.rstrip("/")}


class App(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        base_url="http://127.0.0.1:8000/v1",
        model="default",
        rounds=10,
        projects=None,
        templates=None,
        models=None,
    ):
        super().__init__(address, Handler)
        self.models = models or {model: base_url.rstrip("/")}
        self.model = next(iter(self.models))
        self.base_url, self.rounds = self.models[self.model], rounds
        self.projects = Path(projects or ROOT / "projects").absolute()
        self.templates = Path(templates or ROOT / "templates").absolute()
        safe_path(self.projects, exists=False)
        safe_path(self.templates)
        self.projects.mkdir(parents=True, exist_ok=True)
        self.processes, self.chat_locks = {}, {}

    def folder(self, root, slug):
        if not isinstance(slug, str) or not re.fullmatch(SLUG, slug):
            raise ValueError("Invalid project or template slug.")
        path = safe_path(root, slug)
        if not path.is_dir():
            raise ValueError("Folder not found.")
        return path

    def project_path(self, slug):
        return self.folder(self.projects, slug)

    def task(self, folder):
        task = read_json(folder, "task.json")
        name = task.get("editable")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+\.py", name):
            raise ValueError("Editable must be a Python filename.")
        safe_path(folder, name)
        safe_path(folder, "program.md")
        if task.get("direction") not in ("max", "min"):
            raise ValueError("Metric direction must be max or min.")
        return task

    def run_path(self, project, name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Invalid run ID.")
        path = safe_path(project, "runs", name)
        if not path.is_dir():
            raise ValueError("Run not found.")
        return path

    def active(self, run):
        process = self.processes.get(str(run))
        return process is not None and process.poll() is None

    def info(self, run):
        path = safe_path(run, "run.json", exists=False)
        meta = (
            read_json(run, "run.json")
            if path.exists()
            else {"id": run.name, "model": self.model}
        )
        events = jsonl(run, "events.jsonl")
        values = [e for e in events if e.get("event") in ("baseline", "result", "done")]
        best = None
        for event in values:
            if event["event"] == "baseline":
                best = event.get("result", {}).get(event.get("metric"))
            elif event.get("verdict") == "supported":
                best = event.get("value", best)
            elif event["event"] == "done":
                best = event.get("best", best)
        return dict(
            meta,
            id=run.name,
            created=meta.get("created")
            or (events[0].get("ts") if events else None)
            or getattr(run.stat(), "st_birthtime", run.stat().st_mtime),
            active=self.active(run),
            paused=safe_path(run, "PAUSE", exists=False).exists(),
            experiments=sum(e.get("event") == "result" for e in events),
            best=best,
            updated=events[-1].get("ts")
            if events
            else path.stat().st_mtime
            if path.exists()
            else run.stat().st_mtime,
        )

    def runs(self, project):
        folder = safe_path(project, "runs", exists=False)
        return (
            [
                self.info(p)
                for p in sorted(folder.iterdir(), reverse=True)
                if p.is_dir() and not p.is_symlink()
            ]
            if folder.exists()
            else []
        )

    def summary(self, project):
        meta = read_json(project, "project.json")
        task, runs = self.task(project), self.runs(project)
        activity = [
            safe_path(project, name).stat().st_mtime
            for name in (
                "project.json",
                "program.md",
                "discussion.jsonl",
                "ideas.jsonl",
            )
            if safe_path(project, name, exists=False).exists()
        ]
        values = [r["best"] for r in runs if isinstance(r.get("best"), (int, float))]
        best = (
            (max(values) if task["direction"] == "max" else min(values))
            if values
            else None
        )
        return dict(
            meta,
            slug=project.name,
            task=task,
            experiments=sum(r["experiments"] for r in runs),
            best=best,
            updated=max(activity + [r["updated"] for r in runs if r.get("updated")]),
        )

    def detail(self, project):
        runs = self.runs(project)
        ideas = jsonl(project, "ideas.jsonl")
        for run in runs:
            if run["active"]:
                folder = self.run_path(project, run["id"])
                used = sum(
                    e.get("event") == "idea" and e.get("source") in ("you", "human")
                    for e in jsonl(folder, "events.jsonl")
                )
                ideas += jsonl(folder, "inbox.jsonl")[used:]
        return {
            "project": self.summary(project),
            "program": safe_path(project, "program.md").read_text(),
            "task": self.task(project),
            "runs": runs,
            "discussion": jsonl(project, "discussion.jsonl"),
            "ideas": ideas,
            "models": list(self.models),
            "default_model": self.model,
            "rounds": self.rounds,
        }

    def artifact(self, project, run, name):
        task = self.task(project)
        editable = re.escape(task["editable"])
        if name not in (
            "events.jsonl",
            "loop.log",
            "history.json",
            "best_" + task["editable"],
        ) and not re.fullmatch(r"round\d+_" + editable, name):
            raise ValueError("File is not an allowed artifact.")
        path = safe_path(run, name, exists=False)
        if name == "best_" + task["editable"] and not path.exists():
            # Older runs may only have numbered files. Find the kept best from events.
            best = 0
            for event in jsonl(run, "events.jsonl"):
                if (
                    event.get("event") == "result"
                    and event.get("verdict") == "supported"
                ):
                    best = event["round"]
            path = safe_path(run, f"round{best}_" + task["editable"])
        if not path.is_file():
            raise ValueError("File not found.")
        return path


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def json(self, data, status=200):
        self.bytes(json.dumps(data).encode(), "application/json", status)

    def bytes(self, body, mime, status=200, download=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        if download:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{download}"'
            )
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
        query = parse_qs(parsed.query)
        try:
            if parts == ["api", "templates"]:
                return self.json(
                    {
                        "templates": [
                            dict(slug=p.name, **self.server.task(p))
                            for p in sorted(self.server.templates.iterdir())
                            if p.is_dir()
                            and not p.is_symlink()
                            and re.fullmatch(SLUG, p.name)
                        ]
                    }
                )
            if parts == ["api", "projects"]:
                return self.json(
                    {
                        "projects": [
                            self.server.summary(p)
                            for p in sorted(self.server.projects.iterdir())
                            if p.is_dir()
                            and not p.is_symlink()
                            and re.fullmatch(SLUG, p.name)
                        ],
                        "models": list(self.server.models),
                    }
                )
            if len(parts) >= 3 and parts[:2] == ["api", "projects"]:
                project = self.server.project_path(parts[2])
                if len(parts) == 3:
                    return self.json(self.server.detail(project))
                if len(parts) >= 5 and parts[3] == "runs":
                    run = self.server.run_path(project, parts[4])
                    if len(parts) == 5:
                        return self.json(
                            {
                                "run": self.server.info(run),
                                "events": jsonl(run, "events.jsonl"),
                                "files": [
                                    p.name
                                    for p in sorted(run.iterdir())
                                    if p.is_file()
                                    and not p.is_symlink()
                                    and (
                                        p.name
                                        in (
                                            "events.jsonl",
                                            "loop.log",
                                            "history.json",
                                            "best_"
                                            + self.server.task(project)["editable"],
                                        )
                                        or re.fullmatch(
                                            r"round\d+_"
                                            + re.escape(
                                                self.server.task(project)["editable"]
                                            ),
                                            p.name,
                                        )
                                    )
                                ],
                            }
                        )
                    if len(parts) == 6 and parts[5] == "events":
                        speed = (
                            float(query.get("speed", ["8"])[0])
                            if query.get("replay") == ["1"]
                            else 0
                        )
                        if not math.isfinite(speed) or speed < 0 or speed > 1000:
                            raise ValueError("Invalid replay speed.")
                        after = max(0, int(self.headers.get("Last-Event-ID", "0")))
                        path = safe_path(run, "events.jsonl", exists=False)
                        self.stream_headers()
                        for frame in event_stream(
                            path,
                            speed,
                            after,
                            lambda: (
                                self.server.active(run)
                                or (not speed and str(run) not in self.server.processes)
                            ),
                        ):
                            self.wfile.write(frame)
                            self.wfile.flush()
                        return
                    if len(parts) == 7 and parts[5] == "files":
                        path = self.server.artifact(project, run, parts[6])
                        return self.bytes(
                            path.read_bytes(),
                            "text/plain; charset=utf-8",
                            download=parts[6]
                            if query.get("download") == ["1"]
                            else None,
                        )
                    if len(parts) == 7 and parts[5] == "diff" and parts[6].isdigit():
                        number = int(parts[6])
                        events = jsonl(run, "events.jsonl")
                        idea = next(
                            (
                                e
                                for e in events
                                if e.get("event") == "idea" and e.get("round") == number
                            ),
                            {},
                        )
                        parent = idea.get("parent", 0)
                        editable = self.server.task(project)["editable"]
                        current = self.server.artifact(
                            project, run, f"round{number}_{editable}"
                        )
                        original = self.server.artifact(
                            project, run, f"round{parent}_{editable}"
                        )
                        diff = "".join(
                            difflib.unified_diff(
                                original.read_text().splitlines(True),
                                current.read_text().splitlines(True),
                                f"round{parent}/{editable}",
                                f"round{number}/{editable}",
                            )
                        )
                        return self.json(
                            {"parent": parent, "round": number, "diff": diff}
                        )
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
                "/fonts/PublicSans.ttf": ("fonts/PublicSans.ttf", "font/ttf"),
            }
            if parsed.path in assets:
                file, mime = assets[parsed.path]
                return self.bytes((ROOT / "web" / file).read_bytes(), mime)
            self.json({"error": "Not found"}, 404)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.json({"error": str(exc)}, 400)

    def body(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            raise PermissionError("Cross-origin request rejected")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 100000:
            raise ValueError("Expected a JSON body under 100 KB.")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def do_PUT(self):
        self.mutate("PUT")

    def do_POST(self):
        self.mutate("POST")

    def mutate(self, method):
        try:
            data = self.body()
            parts = urlparse(self.path).path.strip("/").split("/")
            if method == "POST" and parts == ["api", "projects"]:
                return self.create_project(data)
            if len(parts) >= 4 and parts[:2] == ["api", "projects"]:
                project = self.server.project_path(parts[2])
                if len(parts) == 4:
                    action = parts[3]
                    if method == "PUT" and action == "program":
                        text = data.get("program")
                        if (
                            not isinstance(text, str)
                            or not text.strip()
                            or len(text) > 60000
                        ):
                            raise ValueError(
                                "Research goal must be text under 60,000 characters."
                            )
                        with INBOX_LOCK:
                            if any(r["active"] for r in self.server.runs(project)):
                                raise ValueError(
                                    "Pause goal editing until this run finishes; its budget and goal stay fixed."
                                )
                            safe_path(project, "program.md").write_text(text)
                        return self.json({"program": text})
                    if method == "POST" and action == "chat":
                        return self.chat(project, data)
                    if method == "POST" and action == "ideas":
                        with INBOX_LOCK:
                            active = [
                                r for r in self.server.runs(project) if r["active"]
                            ]
                            if active:
                                append_idea(
                                    self.server.run_path(project, active[0]["id"]), data
                                )
                            else:
                                # Reuse inbox validation while writing the project queue.
                                hypothesis, change = (
                                    data.get("hypothesis"),
                                    data.get("change", ""),
                                )
                                if (
                                    not isinstance(hypothesis, str)
                                    or not hypothesis.strip()
                                    or not isinstance(change, str)
                                    or len(hypothesis) + len(change) > 16000
                                ):
                                    raise ValueError(
                                        "Idea must contain a hypothesis and text under 16,000 characters."
                                    )
                                with safe_path(
                                    project, "ideas.jsonl", exists=False
                                ).open("a") as stream:
                                    stream.write(
                                        json.dumps(
                                            dict(
                                                hypothesis=hypothesis.strip(),
                                                change=change,
                                                source="you",
                                                ts=time.time(),
                                            )
                                        )
                                        + "\n"
                                    )
                        return self.json(
                            {"ideas": self.server.detail(project)["ideas"]}, 201
                        )
                    if method == "POST" and action == "runs":
                        return self.start_run(project, data)
                if method == "POST" and len(parts) == 6 and parts[3] == "runs":
                    run = self.server.run_path(project, parts[4])
                    action = parts[5]
                    if not self.server.active(run):
                        raise ValueError(
                            "Run is no longer active. Start a new run to test more ideas."
                        )
                    with INBOX_LOCK:
                        if action == "idea":
                            safe_path(run, "inbox.jsonl", exists=False)
                            append_idea(run, data)
                        elif action == "pause":
                            safe_path(run, "PAUSE", exists=False).touch()
                        elif action in ("resume", "stop"):
                            if action == "stop":
                                safe_path(run, "STOP", exists=False).touch()
                            safe_path(run, "PAUSE", exists=False).unlink(
                                missing_ok=True
                            )
                        else:
                            return self.json({"error": "Not found"}, 404)
                    return self.json({"ok": True, "action": action})
            self.json({"error": "Not found"}, 404)
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.json({"error": str(exc)}, 400)

    def create_project(self, data):
        title, goal = data.get("title"), data.get("goal", "")
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError("Title must contain 1–200 characters.")
        if not isinstance(goal, str) or len(goal) > 60000:
            raise ValueError("Goal must be text under 60,000 characters.")
        template = self.server.folder(self.server.templates, data.get("template"))
        task = self.server.task(template)
        slug = (
            re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:64].rstrip("-")
            or "project"
        )
        with INBOX_LOCK:
            if (self.server.projects / slug).exists() or (
                self.server.projects / slug
            ).is_symlink():
                slug += "-" + uuid.uuid4().hex[:6]
            project = safe_path(self.server.projects, slug, exists=False)
            project.mkdir()
            for name in ("task.json", "program.md", task["editable"]):
                shutil.copyfile(safe_path(template, name), project / name)
            if goal.strip():
                (project / "program.md").write_text(
                    "# Research goal\n\n"
                    + goal.strip()
                    + "\n\n"
                    + (project / "program.md").read_text()
                )
            meta = dict(
                title=title.strip(),
                template=template.name,
                created=time.time(),
                model=self.server.model,
            )
            (project / "project.json").write_text(json.dumps(meta, indent=2) + "\n")
            (project / "runs").mkdir()
            (project / "discussion.jsonl").touch()
            (project / "ideas.jsonl").touch()
        return self.json(self.server.detail(project), 201)

    def start_run(self, project, data):
        model = data.get("model", self.server.model)
        if model not in self.server.models:
            raise ValueError("Choose a configured model.")
        rounds = int(data.get("rounds", self.server.rounds))
        if not 1 <= rounds <= 100:
            raise ValueError("Rounds must be between 1 and 100.")
        with INBOX_LOCK:
            if any(r["active"] for r in self.server.runs(project)):
                raise ValueError("This project already has an active run.")
            task = self.server.task(project)
            for name in ("ideas.jsonl", "discussion.jsonl"):
                safe_path(project, name, exists=False)
            name = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            runs = safe_path(project, "runs", exists=False)
            runs.mkdir(exist_ok=True)
            run = runs / name
            run.mkdir()
            meta = dict(
                id=name,
                model=model,
                rounds=rounds,
                task=task,
                project=project.name,
                created=time.time(),
            )
            (run / "run.json").write_text(json.dumps(meta, indent=2))
            with (run / "loop.log").open("w") as log:
                self.server.processes[str(run)] = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "loop.py"),
                        str(project),
                        str(run),
                        "--base-url",
                        self.server.models[model],
                        "--model",
                        model,
                        "--rounds",
                        str(rounds),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            project_meta = read_json(project, "project.json")
            project_meta["model"] = model
            safe_path(project, "project.json").write_text(
                json.dumps(project_meta, indent=2)
            )
        return self.json(dict(meta, active=True, paused=False), 201)

    def chat(self, project, data):
        message, model = data.get("message"), data.get("model", self.server.model)
        if not isinstance(message, str) or not message.strip() or len(message) > 16000:
            raise ValueError("Message must contain text under 16,000 characters.")
        if model not in self.server.models:
            raise ValueError("Choose a configured model.")
        with INBOX_LOCK:
            lock = self.server.chat_locks.setdefault(project.name, threading.Lock())
        if not lock.acquire(blocking=False):
            return self.json(
                {"error": "A reply is already in progress for this project."}, 409
            )
        try:
            task = self.server.task(project)
            history = []
            for run in reversed(self.server.runs(project)[:5]):
                ideas = {}
                for e in jsonl(
                    self.server.run_path(project, run["id"]), "events.jsonl"
                ):
                    if e.get("event") == "idea":
                        ideas[e["round"]] = e.get("change") or e.get("hypothesis")
                    if e.get("event") in ("baseline", "result"):
                        history.append(
                            {
                                "run": run["id"],
                                "round": e.get("round", 0),
                                "change": ideas.get(e.get("round"), "Baseline"),
                                "verdict": e.get("verdict", "baseline"),
                                "value": e.get(
                                    "value", e.get("result", {}).get(task["metric"])
                                ),
                            }
                        )
            system = (
                "Discuss concrete, testable ideas briefly. Experiments run locally. Do not invent results.\n"
                + safe_path(project, "program.md").read_text()
                + "\nMetric and fixed budget:\n"
                + json.dumps(task)
                + "\nObserved verdict ledger (latest 60):\n"
                + json.dumps(history[-60:])
            )
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in jsonl(project, "discussion.jsonl")[-30:]
            ]
            with INBOX_LOCK:
                append_message(project, "user", message.strip(), model=model)
                project_meta = read_json(project, "project.json")
                project_meta["model"] = model
                safe_path(project, "project.json").write_text(
                    json.dumps(project_meta, indent=2)
                )
            body = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "system", "content": system}]
                    + messages
                    + [{"role": "user", "content": message.strip()}],
                    "stream": True,
                    "max_tokens": 1600,
                }
            ).encode()
            request = Request(
                self.server.models[model] + "/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )
            try:
                upstream = urlopen(request, timeout=1800)
            except (HTTPError, URLError) as exc:
                return self.json(
                    {
                        "error": f"Local model unavailable: {exc}. Your message was saved; retry when the model is ready."
                    },
                    502,
                )
            content, connected = "", True
            with upstream:
                try:
                    self.stream_headers()
                except (BrokenPipeError, ConnectionResetError):
                    connected = False
                try:
                    for line in upstream:
                        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                            payload = json.loads(line[5:])
                            if payload.get("error"):
                                raise ValueError(str(payload["error"]))
                            content += (payload.get("choices") or [{}])[0].get(
                                "delta", {}
                            ).get("content") or ""
                        if connected:
                            try:
                                self.wfile.write(line)
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                connected = False  # Finish and persist even if the browser navigates away.
                except (ValueError, OSError) as exc:
                    if connected:
                        self.wfile.write(
                            f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n".encode()
                        )
                finally:
                    if content:
                        append_message(project, "assistant", content, model=model)
        finally:
            lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument(
        "--models", help="Ordered name=URL pairs, separated by commas; first is default"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4900)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    try:
        models = parse_models(args.models, args.base_url, args.model)
    except ValueError as exc:
        parser.error(str(exc))
    app = App(
        (args.host, args.port), args.base_url, args.model, args.rounds, models=models
    )
    print(f"Research Loop: http://{args.host}:{args.port}", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()
