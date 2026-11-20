# SPDX-License-Identifier: Apache-2.0
"""Parent-owned metadata storage. All entry points run on the control loop."""

import collections
import contextlib
import fcntl
import functools
import json
import os
import re
import stat
import threading
import time
from urllib.parse import parse_qs, urlsplit

from . import CONTROL_PREFIX
from .protocol import private_directory

FIELDS = frozenset(
    [
        "ts",
        "request_id",
        "client",
        "app",
        "model",
        "route",
        "http_status",
        "outcome",
        "prompt_tokens",
        "completion_tokens",
        "ttft_ms",
        "duration_ms",
    ]
)


def write_private(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
    temporary.replace(path)


def serialized(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.locked():
            return method(self, *args, **kwargs)

    return wrapped


def read_private(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise OSError("history must be an owned regular file")
        return stream.read()


class History:
    def __init__(
        self,
        root,
        *,
        disabled=False,
        cap=50 * 1024 * 1024,
        event_limit=1000,
        instance=None,
        port=None,
        readonly=False,
    ):
        self.readonly = readonly
        self.instance = instance
        self.port = port
        self.mutex = threading.RLock()
        self.depth = 0
        self.root = root if readonly else private_directory(root)
        if root.exists():
            info = root.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise OSError("history must be an owned 0700 directory")
        self.settings_path = (
            root / "settings" if readonly else private_directory(root / "settings")
        ) / "log.json"
        self.settings = {"enabled": True, "max_bytes": cap}
        if self.settings_path.exists():
            self.settings.update(json.loads(read_private(self.settings_path)))
        self.disabled = disabled
        self.logs = root / "calls.jsonl"
        self.old = root / "calls.1.jsonl"
        self.event_path = root / "events.json"
        self.events = collections.deque(maxlen=event_limit)
        if self.event_path.exists():
            self.events.extend(json.loads(read_private(self.event_path)))
        self.event_cursor = root / "events.sequence.json"
        self.sequence = (
            json.loads(read_private(self.event_cursor))
            if self.event_cursor.exists()
            else (self.events[-1]["id"] if self.events else 0)
        )
        self.cursor_path = root / "calls.sequence.json"
        self.call_sequence = (
            json.loads(read_private(self.cursor_path))
            if self.cursor_path.exists()
            else 0
        )
        self.seen = set()
        self.dropped = 0
        self.runner_dropped = {}
        self.runner_skipped = {}
        with self.locked():
            self.upgrade()

    def upgrade(self):
        # Upgrade retained records once; cursors belong to the parent, never the runner.
        for path in (self.old, self.logs):
            if path.exists():
                items = [json.loads(line) for line in read_private(path).splitlines()]
                upgraded = False
                for item in items:
                    if "seq" not in item:
                        self.call_sequence += 1
                        item["seq"] = self.call_sequence
                        upgraded = True
                    else:
                        self.call_sequence = max(self.call_sequence, item["seq"])
                if upgraded and not self.readonly:
                    write_private(self.cursor_path, json.dumps(self.call_sequence))
                    write_private(
                        path, "".join(json.dumps(item) + "\n" for item in items)
                    )

    @contextlib.contextmanager
    def locked(self):
        with self.mutex:
            if self.depth:
                yield
                return
            if self.readonly and not (self.root / "history.lock").exists():
                yield
                return
            fd = os.open(
                self.root / "history.lock",
                (os.O_RDONLY if self.readonly else os.O_CREAT | os.O_RDWR)
                | os.O_NOFOLLOW,
                0o600,
            )
            try:
                if (
                    not stat.S_ISREG(os.fstat(fd).st_mode)
                    or os.fstat(fd).st_uid != os.getuid()
                ):
                    raise OSError("history lock must be an owned regular file")
                if not self.readonly:
                    os.fchmod(fd, 0o600)
                fcntl.flock(fd, fcntl.LOCK_SH if self.readonly else fcntl.LOCK_EX)
                self.depth += 1
                if self.settings_path.exists():
                    self.settings.update(json.loads(read_private(self.settings_path)))
                if self.cursor_path.exists():
                    self.call_sequence = max(
                        self.call_sequence, json.loads(read_private(self.cursor_path))
                    )
                if self.event_cursor.exists():
                    self.sequence = max(
                        self.sequence, json.loads(read_private(self.event_cursor))
                    )
                self.seen = {r["request_id"] for r in self._records()}
                yield
            finally:
                self.depth = 0
                os.close(fd)

    @serialized
    def cleanup_legacy(self):
        """Ignore unrecognized entries; remove only private legacy stores."""
        names = {"calls.jsonl", "calls.1.jsonl", "calls.sequence.json", "events.json"}
        for path in self.root.glob("i-*"):
            info = path.lstat()
            if (
                not re.fullmatch(r"i-[a-z0-9_]{8}", path.name)
                or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                continue
            entries = list(path.iterdir())
            if any(
                p.name not in names
                or not stat.S_ISREG(p.lstat().st_mode)
                or p.lstat().st_uid != os.getuid()
                for p in entries
            ):
                continue
            for entry in entries:
                entry.unlink()
            path.rmdir()

    @property
    def enabled(self):
        return self.settings["enabled"] and not self.disabled

    def records(self):
        with self.locked():
            yield from self._records()

    def _records(self):
        for path in (self.old, self.logs):
            if path.exists():
                for line in read_private(path).splitlines():
                    yield json.loads(line)

    @serialized
    def status(self):
        return {
            "enabled": self.enabled,
            "records": sum(1 for _ in self.records()),
            "bytes": sum(p.stat().st_size for p in (self.logs, self.old) if p.exists()),
            "dropped": self.dropped + sum(self.runner_dropped.values()),
            "skipped": sum(self.runner_skipped.values()),
        }

    @serialized
    def append(self, records, pid, dropped=0, *, retain=False):
        self.runner_dropped[pid] = max(self.runner_dropped.get(pid, 0), int(dropped))
        if not self.enabled and not retain:
            return
        for record in records:
            # Rebuild from an allowlist: even a buggy runner cannot store content.
            item = {key: record.get(key) for key in FIELDS}
            item["instance"] = self.instance
            item["port"] = self.port
            item["app"] = None  # P5 will supply verified identities.
            rid = item["request_id"]
            if not isinstance(rid, str) or not rid or len(rid) > 128:
                self.dropped += 1
                continue
            if rid in self.seen:
                continue
            for key in ("client", "model", "route", "outcome"):
                value = item[key]
                item[key] = value[:256] if isinstance(value, str) else None
            for key in (
                "ts",
                "http_status",
                "prompt_tokens",
                "completion_tokens",
                "ttft_ms",
                "duration_ms",
            ):
                if type(item[key]) not in (int, float):
                    item[key] = None
            item["seq"] = self.call_sequence + 1
            line = json.dumps(item, allow_nan=False) + "\n"
            half = (self.settings["max_bytes"] - 512) * 3 // 8
            if len(line.encode()) > half:
                self.dropped += 1
                continue
            if (
                self.logs.exists()
                and self.logs.stat().st_size + len(line.encode()) > half
            ):
                self.logs.replace(self.old)
                self.seen = {r["request_id"] for r in self.records()}
            self.call_sequence = item["seq"]
            write_private(self.cursor_path, json.dumps(self.call_sequence))
            fd = os.open(
                self.logs, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "a") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(line)
            self.seen.add(rid)

    @serialized
    def event(self, kind, **fields):
        if self.event_path.exists():
            self.events.clear()
            self.events.extend(json.loads(read_private(self.event_path)))
        self.sequence = (
            max(self.sequence, self.events[-1]["id"] if self.events else 0) + 1
        )
        item = {
            "id": self.sequence,
            "ts": time.time(),
            "event": kind,
            "instance": self.instance,
            "port": self.port,
            **fields,
        }
        self.events.append(item)
        while (
            self.events
            and len(json.dumps(list(self.events)).encode())
            > (self.settings["max_bytes"] - 512) // 4
        ):
            self.events.popleft()
        try:
            write_private(self.event_cursor, json.dumps(self.sequence))
            write_private(self.event_path, json.dumps(list(self.events)))
        except OSError:
            # Disk failure must never interrupt runner shutdown/restart cleanup.
            self.dropped += 1
        return item

    @serialized
    def api(self, method, target, data):
        url = urlsplit(target)
        query = parse_qs(url.query)
        if url.path == "/rapid/v1/log/settings":
            if method == "PUT":
                if set(data) - {"enabled", "max_bytes"} or (
                    "enabled" in data and type(data["enabled"]) is not bool
                ):
                    raise ValueError("invalid settings")
                if "max_bytes" in data and (
                    type(data["max_bytes"]) is not int
                    or not 4096 <= data["max_bytes"] <= 50 * 1024 * 1024
                ):
                    raise ValueError("invalid size cap")
                self.settings.update(data)
                write_private(self.settings_path, json.dumps(self.settings))
                half = (self.settings["max_bytes"] - 512) * 3 // 8
                for path in (self.old, self.logs):
                    if path.exists() and path.stat().st_size > half:
                        retained = collections.deque()
                        size = 0
                        for line in read_private(path).splitlines(keepends=True):
                            retained.append(line)
                            size += len(line.encode())
                            while size > half:
                                size -= len(retained.popleft().encode())
                        write_private(path, "".join(retained))
                self.seen = {r["request_id"] for r in self.records()}
                self.event("log_settings_changed", **self.settings)
            elif method != "GET":
                return 404, {}
            return 200, {**self.settings, "enabled": self.enabled}
        if url.path == "/rapid/v1/log":
            if method == "DELETE":
                self.logs.unlink(missing_ok=True)
                self.old.unlink(missing_ok=True)
                self.seen.clear()
                return 200, {}
            if method == "GET":
                limit = max(1, min(1000, int(query.get("limit", ["100"])[0])))
                after = int(query["after"][0]) if "after" in query else None
                result = collections.deque(maxlen=limit)
                for record in self.records():
                    if after is not None and record["seq"] <= after:
                        continue
                    if "since" in query and record["ts"] <= float(query["since"][0]):
                        continue
                    if "port" in query and record.get("port") != int(query["port"][0]):
                        continue
                    if all(
                        record.get(k) == query[k][0]
                        for k in ("client", "model", "route", "outcome")
                        if k in query
                    ):
                        result.append(record)
                        if after is not None and len(result) == limit:
                            break
                return 200, list(result) if after is not None else list(
                    reversed(result)
                )
        if url.path == CONTROL_PREFIX + "/events" and method == "GET":
            since = int(query.get("since", ["0"])[0])
            events = (
                json.loads(read_private(self.event_path))
                if self.event_path.exists()
                else []
            )
            return 200, [e for e in events if e["id"] > since]
        return 404, {}
