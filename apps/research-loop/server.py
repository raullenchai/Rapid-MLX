"""Local Research Loop web server with CPU-only literature extraction helpers."""

import argparse
import base64
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

import hypotheses
import literature
import protocol

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


def event_stream(
    path,
    speed=0,
    after=0,
    alive=lambda: True,
    sleep=time.sleep,
    project=None,
    task=None,
):
    """SSE cursor is the journal sequence (ordinal for read-only legacy runs)."""
    previous = None
    cursor = after
    heartbeat = time.monotonic()
    while True:
        snapshot = protocol.events(path.parent)
        rows = [e for e in snapshot if e["seq"] > cursor]
        if not rows and snapshot and snapshot[-1].get("event") == "done":
            return
        for event in rows:
            if speed and previous is not None:
                delay = max(0, (event.get("ts", previous) - previous) / speed)
                if event.get("event") == "curve":
                    delay = min(delay, 0.15)
                while delay > 0:
                    interval = min(delay, 5)
                    sleep(interval)
                    delay -= interval
                    yield b": replay\n\n"
            previous = event.get("ts", previous)
            cursor = event["seq"]
            if project is not None and event.get("event") in (
                "baseline",
                "idea",
                "result_record",
                "result",
                "research_ended",
                "done",
            ):
                event = dict(
                    event,
                    hypotheses=hypotheses.collect(
                        project, task, through={path.parent.name: cursor}
                    ),
                )
            yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n".encode()
            if event.get("event") == "done":
                return
        if not alive():
            if any(e["seq"] > cursor for e in protocol.events(path.parent)):
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
        raise ValueError("Hypothesis must be text under 16,000 characters.")
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


VENDOR_TYPES = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2"}


def vendor_asset(path):
    """Self-hosted third-party files under web/vendor (KaTeX). Plain names only, no traversal or symlinks."""
    parts = path.split("/")
    if len(parts) not in (4, 5) or parts[:2] != ["", "vendor"]:
        return None
    if not all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", p) and not p.startswith(".") for p in parts[2:]
    ):
        return None
    file = ROOT / "web" / "/".join(parts[1:])
    mime = VENDOR_TYPES.get(file.suffix)
    if not mime or file.is_symlink() or not file.is_file():
        return None
    return file, mime


def run_task(info):
    """A run's frozen task config; legacy runs stored only the task name."""
    task = info.get("task") if isinstance(info, dict) else None
    return task if isinstance(task, dict) else None


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
        embeddings=None,
        embedding_model=literature.DEFAULT_MODEL,
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
        self.literature = literature.Literature(
            literature.HTTPEmbedder(embeddings, embedding_model) if embeddings else None
        )
        self.embeddings, self.embedding_model = embeddings, embedding_model

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
        hypotheses.limit(task)
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
        if process is not None:
            return process.poll() is None
        # A loop started by an earlier server process keeps running after a
        # restart; treat it as active while its event log is fresh and open.
        events = Path(run) / "events.jsonl"
        try:
            fresh = time.time() - events.stat().st_mtime < 600
        except OSError:
            return False
        if not fresh or any(
            e.get("event") == "done" for e in jsonl(Path(run), "events.jsonl")
        ):
            return False
        try:
            found = subprocess.run(
                ["pgrep", "-f", f"loop.py .* {run} --base-url"],
                capture_output=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return found.returncode == 0

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
            elif hypotheses.experiment_verdict(event.get("verdict")) == "improved":
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
                "hypotheses.jsonl",
            )
            if safe_path(project, name, exists=False).exists()
        ]
        values = [r["best"] for r in runs if isinstance(r.get("best"), (int, float))]
        best = (
            (max(values) if task["direction"] == "max" else min(values))
            if values
            else None
        )
        progress, running_best, experiments = [], None, 0
        for run in sorted(runs, key=lambda r: r["created"]):
            for event in jsonl(self.run_path(project, run["id"]), "events.jsonl"):
                if event.get("event") == "result":
                    experiments += 1
                if event.get("event") not in ("baseline", "result"):
                    continue
                candidate = (
                    event.get("result", {}).get(event.get("metric", task["metric"]))
                    if event["event"] == "baseline"
                    else event.get("value", event.get("result", {}).get(task["metric"]))
                    if hypotheses.experiment_verdict(event.get("verdict")) == "improved"
                    else None
                )
                if isinstance(candidate, (int, float)) and math.isfinite(candidate):
                    running_best = (
                        candidate
                        if running_best is None
                        else (min if task["direction"] == "min" else max)(
                            running_best, candidate
                        )
                    )
                if running_best is not None:
                    progress.append(running_best)
        lines = [
            line.strip()
            for line in safe_path(project, "program.md").read_text().splitlines()
            if line.strip()
        ]
        description = next(
            (line for line in lines if not re.match(r"^#{1,6}\s", line)),
            re.sub(r"^#{1,6}\s+", "", lines[0]) if lines else "",
        )
        return dict(
            meta,
            slug=project.name,
            task=task,
            hypotheses=len(hypotheses.collect(project, task)),
            experiments=experiments,
            best=running_best if running_best is not None else best,
            progress=progress,
            description=description,
            status="Running"
            if any(r["active"] for r in runs)
            else "Finished"
            if runs
            else "No runs",
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
            "literature": self.literature.status(project),
            "ideas": ideas,  # Legacy API alias; new clients use hypotheses.
            "hypotheses": hypotheses.collect(project, self.task(project)),
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
                    and hypotheses.experiment_verdict(event.get("verdict"))
                    == "improved"
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
        if mime.startswith("image/"):
            self.send_header(
                "Content-Security-Policy",
                "sandbox; default-src 'none'; style-src 'unsafe-inline'",
            )
        if download:
            download = re.sub(r"[^A-Za-z0-9_.-]", "_", str(download))
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
                if parts[3:] == ["literature"]:
                    return self.json(self.server.literature.status(project))
                if len(parts) == 3:
                    return self.json(self.server.detail(project))
                if len(parts) >= 5 and parts[3] == "runs":
                    run = self.server.run_path(project, parts[4])
                    view = None
                    if len(parts) == 5 or (
                        len(parts) >= 6
                        and parts[5]
                        in ("artifacts", "records", "ledger.csv", "bundle.zip")
                    ):
                        view = protocol.snapshot(
                            run,
                            run_task(self.server.info(run))
                            or self.server.task(project),
                        )
                    if len(parts) == 6 and parts[5] == "artifacts":
                        artifacts = view["artifacts"]
                        for key in ("experiment", "hypothesis_id", "kind", "role"):
                            if key in query:
                                artifacts = [
                                    a
                                    for a in artifacts
                                    if str(
                                        a.get(
                                            key,
                                            a["owner"].get(key)
                                            if isinstance(a.get("owner"), dict)
                                            else None,
                                        )
                                    )
                                    == query[key][0]
                                ]
                        return self.json(
                            dict(
                                protocol=1,
                                watermark=view["watermark"],
                                artifacts=artifacts,
                            )
                        )
                    if len(parts) == 6 and parts[5] == "records":
                        return self.json(
                            dict(
                                protocol=1,
                                watermark=view["watermark"],
                                records=view["records"],
                            )
                        )
                    if len(parts) == 6 and parts[5] == "ledger.csv":
                        return self.bytes(
                            protocol.ledger_csv(view["records"]),
                            "text/csv",
                            download="ledger.csv",
                        )
                    if len(parts) == 6 and parts[5] == "bundle.zip":
                        return self.bytes(
                            protocol.bundle(run, view),
                            "application/zip",
                            download="research-bundle.zip",
                        )
                    if len(parts) == 7 and parts[5] == "artifacts":
                        artifact = next(
                            (
                                a
                                for a in reversed(view["artifacts"])
                                if a.get("id") == parts[6]
                                and (
                                    "revision" not in query
                                    or str(a.get("revision")) == query["revision"][0]
                                )
                            ),
                            None,
                        )
                        if artifact is None:
                            raise ValueError("Artifact not found")
                        content = protocol.artifact_content(run, artifact)
                        if query.get("rows"):
                            rows = protocol.data_rows(content, artifact["mime"])
                            maximum = min(500, max(0, int(query["rows"][0])))
                            return self.json(
                                dict(
                                    rows=rows[:maximum],
                                    total=len(rows),
                                    columns=artifact.get("columns", {}),
                                    watermark=view["watermark"],
                                )
                            )
                        download = (
                            query.get("download") == ["1"]
                            or artifact["kind"] not in protocol.KINDS
                            or artifact["kind"] == "file"
                        )
                        return self.bytes(
                            content,
                            artifact["mime"]
                            if not download
                            else "application/octet-stream",
                            download=Path(artifact["path"]).name if download else None,
                        )
                    if len(parts) == 5:
                        return self.json(
                            {
                                "run": self.server.info(run),
                                "events": view["events"],
                                "watermark": view["watermark"],
                                "artifacts": view["artifacts"],
                                "records": view["records"],
                                "protocol": 1,
                                "legacy": view["legacy"],
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
                        after = max(
                            0,
                            int(
                                self.headers.get("Last-Event-ID")
                                or query.get("after", ["0"])[0]
                            ),
                        )
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
                            project=project,
                            task=self.server.task(project),
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
            vendor = vendor_asset(parsed.path)
            if vendor:
                return self.bytes(vendor[0].read_bytes(), vendor[1])
            self.json({"error": "Not found"}, 404)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.json({"error": str(exc)}, 400)

    def body(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            raise PermissionError("Cross-origin request rejected")
        length = int(self.headers.get("Content-Length", "0"))
        upload = urlparse(self.path).path.endswith("/literature/upload")
        if not 0 < length <= (28 * 1024 * 1024 if upload else 100000):
            raise ValueError(
                "Expected a JSON body under 28 MiB for uploads or 100 KB otherwise."
            )
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
                if len(parts) == 5 and parts[3] == "literature" and method == "POST":
                    return self.literature_action(project, parts[4], data)
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
                    if method == "POST" and action in ("hypotheses", "ideas"):
                        hypotheses.migrate_queue(project)
                        if "discussion_index" in data:
                            index = data["discussion_index"]
                            messages = jsonl(project, "discussion.jsonl")
                            if (
                                type(index) is not int
                                or not 0 <= index < len(messages)
                                or messages[index]["role"] != "assistant"
                            ):
                                raise ValueError("Choose a saved assistant message.")
                            data["sources"] = messages[index].get("citations", [])
                        row = hypotheses.add(project, data, data.get("source", "you"))
                        return self.json(
                            {
                                "hypothesis": row,
                                "hypotheses": hypotheses.collect(
                                    project, self.server.task(project)
                                ),
                            },
                            201,
                        )
                    if method == "POST" and action == "runs":
                        return self.start_run(project, data)
                if method == "POST" and len(parts) == 6 and parts[3] == "runs":
                    run = self.server.run_path(project, parts[4])
                    action = parts[5]
                    if not self.server.active(run):
                        raise ValueError(
                            "Run is no longer active. Start a new run to test more hypotheses."
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

    def literature_action(self, project, action, data):
        service = self.server.literature
        if service.embedder is None:
            raise ValueError(
                "Literature embeddings are not configured. Start with --embeddings URL."
            )
        with INBOX_LOCK, service.lock:
            if service.jobs.get(str(project), {}).get("state") == "indexing":
                raise ValueError(
                    "Indexing is in progress. Wait before changing sources."
                )
            folder = safe_path(project, "literature", exists=False)
            folder.mkdir(exist_ok=True)
            if action == "upload":
                name = data.get("name", "")
                path = literature.upload_path(project, name)
                if path.suffix.lower() not in literature.SUFFIXES:
                    raise ValueError("Supported files: PDF, Markdown, text, TeX, HTML.")
                content = base64.b64decode(data.get("content", ""), validate=True)
                if not content or len(content) > literature.MAX_BYTES:
                    raise ValueError("File must contain 1 byte to 20 MiB.")
                path.write_bytes(content)
            elif action in ("folder", "remove-folder"):
                meta = read_json(project, "project.json")
                dirs = meta.get("literature_dirs", [])
                if action == "folder":
                    path = str(literature.validate_folder(data.get("path", "")))
                    dirs = list(dict.fromkeys([*dirs, path]))
                else:
                    dirs = [p for p in dirs if p != data.get("path")]
                meta["literature_dirs"] = dirs
                safe_path(project, "project.json").write_text(
                    json.dumps(meta, indent=2)
                )
            elif action == "remove":
                literature.upload_path(project, data.get("name", "")).unlink()
            elif action != "index":
                raise ValueError("Unknown literature action.")
            service.start(project)
        return self.json(service.status(project), 202)

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
            (project / "hypotheses.jsonl").touch()
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
            for name in ("hypotheses.jsonl", "discussion.jsonl"):
                safe_path(project, name, exists=False)
            hypotheses.migrate_queue(project)
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
                        *(
                            [
                                "--embeddings",
                                self.server.embeddings,
                                "--embedding-model",
                                self.server.embedding_model,
                            ]
                            if self.server.embeddings
                            else []
                        ),
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
                                "verdict": hypotheses.experiment_verdict(
                                    e.get("verdict", "baseline")
                                ),
                                "value": e.get(
                                    "value", e.get("result", {}).get(task["metric"])
                                ),
                            }
                        )
            system = (
                "Discuss testable hypotheses: claims about the system, with why and prediction. "
                "A hypothesis is tested by one or more experiments (concrete changes). "
                "Experiment verdicts are improved, worse, no effect, crashed; hypothesis statuses "
                "are untested, testing, supported, refuted, inconclusive. Do not invent results.\n"
                + safe_path(project, "program.md").read_text()
                + "\nMetric and fixed budget:\n"
                + json.dumps(task)
                + "\nHypotheses with derived statuses and experiments:\n"
                + json.dumps(hypotheses.collect(project, task))
                + "\nObserved experiment ledger (latest 60):\n"
                + json.dumps(history[-60:])
            )
            sources, source_note = [], "Literature not configured."
            if self.server.literature.embedder:
                try:
                    sources = self.server.literature.retrieve(
                        project, message, safe_path(project, "program.md").read_text()
                    )
                    source_note = "" if sources else "No relevant passages found."
                except (ValueError, OSError, KeyError) as exc:
                    source_note = f"Literature retrieval unavailable: {exc}"
                system += literature.context(sources)
                if source_note:
                    system += "\n" + source_note
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
                        append_message(
                            project,
                            "assistant",
                            content,
                            model=model,
                            citations=literature.cited(content, sources),
                            sources_note=source_note
                            or (
                                "No provided sources cited."
                                if not literature.cited(content, sources)
                                else ""
                            ),
                        )
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
    parser.add_argument("--embeddings", help="OpenAI-compatible embedding endpoint URL")
    parser.add_argument("--embedding-model", default=literature.DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        models = parse_models(args.models, args.base_url, args.model)
    except ValueError as exc:
        parser.error(str(exc))
    app = App(
        (args.host, args.port),
        args.base_url,
        args.model,
        args.rounds,
        models=models,
        embeddings=args.embeddings,
        embedding_model=args.embedding_model,
    )
    print(f"Research Loop: http://{args.host}:{args.port}", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()
