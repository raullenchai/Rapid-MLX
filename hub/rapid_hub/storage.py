"""Private, bounded metadata journal with explicit project payload retention."""

import json
import os
import re
import time
from datetime import datetime, timezone


def private_write(path, text, append=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC), 0o600
    )
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def atomic_write(path, text):
    temporary = path.with_suffix(".tmp")
    private_write(temporary, text)
    temporary.replace(path)


def redact(value):
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if re.search(r"api.?key|token|password|secret|authorization", k, re.I)
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def captured(value):
    """Redact before bounding; unparseable/truncated structures are never captured."""
    if isinstance(value, (bytes, str)):
        try:
            value = json.loads(value)
        except (ValueError, UnicodeError):
            # Parse each SSE data object, preserving no unstructured fragments.
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            events = []
            for line in value.splitlines():
                if line.startswith("data:"):
                    try:
                        events.append(redact(json.loads(line[5:])))
                    except ValueError:
                        pass
            value = events
    encoded = json.dumps(redact(value), ensure_ascii=False).encode()
    return encoded[:65536].decode(errors="ignore")


class Journal:
    def __init__(self, directory, max_mb=500):
        self.directory = directory
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.dropped_records = 0
        self.policy_path = directory.parent / "capture.json"
        self.policies = (
            json.loads(self.policy_path.read_text())
            if self.policy_path.exists()
            else {}
        )
        self.prune()

    def configure(self, plugin, project, enabled, ttl):
        self.policies[plugin + "/" + project] = dict(payload=enabled, ttl_hours=ttl)
        atomic_write(self.policy_path, json.dumps(self.policies))

    def policy(self, plugin, project):
        return self.policies.get(
            plugin + "/" + (project or ""),
            self.policies.get("*/" + (project or ""), {}),
        )

    def append(self, entry):
        try:
            entry = dict(entry, ts=datetime.now(timezone.utc).isoformat())
            policy = self.policy(entry["plugin"], entry.get("project"))
            request, response = entry.pop("request", None), entry.pop("response", None)
            if policy.get("payload"):
                entry.update(
                    request=captured(request),
                    response=captured(response),
                    payload_expires_at=time.time() + policy["ttl_hours"] * 3600,
                )
            path = self.directory / entry["plugin"] / (entry["ts"][:10] + ".jsonl")
            text = json.dumps(entry) + "\n"
            self.prune(required=len(text.encode()))
            if (
                sum(p.stat().st_size for p in self.directory.glob("*/*.jsonl"))
                + len(text.encode())
                > self.max_bytes
            ):
                self.dropped_records += 1
                return
            private_write(path, text, append=True)
        except (OSError, ValueError, TypeError):
            self.dropped_records += 1

    def prune(self, required=0, plugin=None, project=None, delete=False):
        now = time.time()
        try:
            for path in sorted(self.directory.glob("*/*.jsonl")):
                rows = []
                changed = False
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                        age = now - datetime.fromisoformat(row["ts"]).timestamp()
                    except (ValueError, KeyError):
                        changed = True
                        continue
                    if age > 7 * 86400 or (
                        delete
                        and (plugin is None or row.get("plugin") == plugin)
                        and (project is None or row.get("project") == project)
                    ):
                        changed = True
                        continue
                    if row.get("payload_expires_at", float("inf")) <= now or (
                        "request" in row and "payload_expires_at" not in row
                    ):
                        for key in ("request", "response", "payload_expires_at"):
                            row.pop(key, None)
                        changed = True
                    rows.append(json.dumps(row) + "\n")
                if not rows:
                    path.unlink()
                elif changed:
                    atomic_write(path, "".join(rows))
            paths = sorted(
                self.directory.glob("*/*.jsonl"), key=lambda p: (p.name, p.parent.name)
            )
            total = sum(p.stat().st_size for p in paths)
            for path in paths:
                if total + required <= self.max_bytes:
                    break
                total -= path.stat().st_size
                path.unlink()
        except OSError:
            self.dropped_records += 1

    def read(self, plugin=None, project=None, limit=50):
        self.prune()
        entries = {}
        for path in sorted(self.directory.glob("*/*.jsonl")):
            if plugin and path.parent.name != plugin:
                continue
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if project is None or row.get("project") == project:
                    entries[row.get("request_id", row["ts"])] = row
        return sorted(entries.values(), key=lambda r: r["ts"], reverse=True)[:limit]
