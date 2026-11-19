"""Private, bounded metadata journal with explicit project payload retention."""

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
        self.sizes = {}
        self.total_bytes = 0
        self.worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hub-journal"
        )
        self.prune()

    def append(self, entry):
        return self.worker.submit(self._append, entry).result()

    def read(self, plugin=None, project=None, limit=50):
        return self.worker.submit(self._read, plugin, project, limit).result()

    def prune(self, **kwargs):
        return self.worker.submit(self._prune, **kwargs).result()

    def configure(self, plugin, project, enabled, ttl):
        return self.worker.submit(
            self._configure, plugin, project, enabled, ttl
        ).result()

    async def call(self, operation, *args, **kwargs):
        """Serialize disk operations without blocking admission or SSE delivery."""
        future = self.worker.submit(getattr(self, "_" + operation), *args, **kwargs)
        return await asyncio.wrap_future(future)

    async def close(self):
        await asyncio.to_thread(self.worker.shutdown, wait=True)

    def _account(self):
        self.sizes = {p: p.stat().st_size for p in self.directory.glob("*/*.jsonl")}
        self.total_bytes = sum(self.sizes.values())

    def _evict(self, required):
        for path in sorted(self.sizes, key=lambda p: (p.name, p.parent.name)):
            if self.total_bytes + required <= self.max_bytes:
                break
            path.unlink(missing_ok=True)
            self.total_bytes -= self.sizes.pop(path)

    def _configure(self, plugin, project, enabled, ttl):
        self.policies[plugin + "/" + project] = dict(payload=enabled, ttl_hours=ttl)
        atomic_write(self.policy_path, json.dumps(self.policies))

    def policy(self, plugin, project):
        return self.policies.get(
            plugin + "/" + (project or ""),
            self.policies.get("*/" + (project or ""), {}),
        )

    def _append(self, entry):
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
            size = len(text.encode())
            if size > self.max_bytes:
                self.dropped_records += 1
                return
            self._evict(size)
            private_write(path, text, append=True)
            self.sizes[path] = self.sizes.get(path, 0) + size
            self.total_bytes += size
        except (OSError, ValueError, TypeError):
            self.dropped_records += 1

    def _prune(self, required=0, plugin=None, project=None, delete=False):
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
            self._account()
            self._evict(required)
        except OSError:
            self.dropped_records += 1

    def _read(self, plugin=None, project=None, limit=50):
        # Read tails backwards, bounding disk I/O even with sparse project filters.
        budget = 8 * 1024 * 1024
        entries = {}
        now = time.time()
        paths = sorted(self.sizes, key=lambda p: (p.name, p.parent.name), reverse=True)
        for path in paths:
            if plugin and path.parent.name != plugin:
                continue
            if len(entries) >= limit:
                cutoff = sorted((r["ts"] for r in entries.values()), reverse=True)[
                    limit - 1
                ]
                if path.stem < cutoff[:10]:
                    break
            found = 0
            with path.open("rb") as stream:
                position = stream.seek(0, os.SEEK_END)
                pending = b""
                while position and budget > 0 and found < limit:
                    size = min(position, 16384, budget)
                    position -= size
                    budget -= size
                    stream.seek(position)
                    parts = (stream.read(size) + pending).split(b"\n")
                    pending = parts.pop(0) if position else b""
                    for line in reversed(parts):
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            if (
                                now - datetime.fromisoformat(row["ts"]).timestamp()
                                > 7 * 86400
                            ):
                                continue
                        except (ValueError, KeyError):
                            continue
                        if project is not None and row.get("project") != project:
                            continue
                        key = row.get("request_id", row["ts"])
                        if key in entries:
                            continue
                        if row.get("payload_expires_at", float("inf")) <= now or (
                            "request" in row and "payload_expires_at" not in row
                        ):
                            for field in ("request", "response", "payload_expires_at"):
                                row.pop(field, None)
                        entries[key] = row
                        found += 1
                        if found >= limit:
                            break
        return sorted(entries.values(), key=lambda r: r["ts"], reverse=True)[:limit]
