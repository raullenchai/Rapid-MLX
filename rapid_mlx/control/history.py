# SPDX-License-Identifier: Apache-2.0
"""Parent-owned metadata storage. All entry points run on the control loop."""

import collections
import json
import os
import time
from urllib.parse import parse_qs, urlsplit

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


class History:
    def __init__(self, root, *, disabled=False, cap=50 * 1024 * 1024, event_limit=1000):
        self.root = private_directory(root)
        self.settings_path = private_directory(root.parent / "settings") / "log.json"
        self.settings = {"enabled": True, "max_bytes": cap}
        if self.settings_path.exists():
            self.settings.update(json.loads(self.settings_path.read_text()))
        self.disabled = disabled
        self.logs = root / "calls.jsonl"
        self.old = root / "calls.1.jsonl"
        self.event_path = root / "events.json"
        self.events = collections.deque(maxlen=event_limit)
        if self.event_path.exists():
            self.events.extend(json.loads(self.event_path.read_text()))
        self.sequence = self.events[-1]["id"] if self.events else 0
        self.seen = {r["request_id"] for r in self.records()}
        self.dropped = 0
        self.runner_dropped = {}

    @property
    def enabled(self):
        return self.settings["enabled"] and not self.disabled

    def records(self):
        for path in (self.old, self.logs):
            if path.exists():
                with path.open() as stream:
                    for line in stream:
                        yield json.loads(line)

    def status(self):
        return {
            "enabled": self.enabled,
            "records": sum(1 for _ in self.records()),
            "bytes": sum(p.stat().st_size for p in (self.logs, self.old) if p.exists()),
            "dropped": self.dropped + sum(self.runner_dropped.values()),
        }

    def append(self, records, pid, dropped=0, *, retain=False):
        self.runner_dropped[pid] = max(self.runner_dropped.get(pid, 0), int(dropped))
        if not self.enabled and not retain:
            return
        for record in records:
            # Rebuild from an allowlist: even a buggy runner cannot store content.
            item = {key: record.get(key) for key in FIELDS}
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
            line = json.dumps(item, allow_nan=False) + "\n"
            half = self.settings["max_bytes"] // 2
            if len(line.encode()) > half:
                self.dropped += 1
                continue
            if (
                self.logs.exists()
                and self.logs.stat().st_size + len(line.encode()) > half
            ):
                self.logs.replace(self.old)
                self.seen = {r["request_id"] for r in self.records()}
            fd = os.open(
                self.logs, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "a") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(line)
            self.seen.add(rid)

    def event(self, kind, **fields):
        self.sequence += 1
        item = {"id": self.sequence, "ts": time.time(), "event": kind, **fields}
        self.events.append(item)
        try:
            write_private(self.event_path, json.dumps(list(self.events)))
        except OSError:
            # Disk failure must never interrupt runner shutdown/restart cleanup.
            self.dropped += 1
        return item

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
                if self.status()["bytes"] > self.settings["max_bytes"]:
                    retained = list(self.records())
                    self.api("DELETE", "/rapid/v1/log", {})
                    self.append(retained, 0, retain=True)
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
                result = collections.deque(maxlen=limit)
                for record in self.records():
                    if "since" in query and record["ts"] <= float(query["since"][0]):
                        continue
                    if all(
                        record.get(k) == query[k][0]
                        for k in ("client", "model", "route", "outcome")
                        if k in query
                    ):
                        result.append(record)
                return 200, list(reversed(result))
        if url.path == "/rapid/v1/events" and method == "GET":
            since = int(query.get("since", ["0"])[0])
            return 200, [e for e in self.events if e["id"] > since]
        return 404, {}
