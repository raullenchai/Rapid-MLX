"""Private local persistence."""

import json
import os
from datetime import datetime, timezone


def private_write(path, text, append=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC), 0o600
    )
    with os.fdopen(fd, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


class Journal:
    def __init__(self, directory):
        self.directory = directory

    def append(self, entry):
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        path = self.directory / entry["plugin"] / (entry["ts"][:10] + ".jsonl")
        private_write(path, json.dumps(entry) + "\n", append=True)

    def read(self, plugin=None, project=None, limit=50):
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
                    entries[row.get("id", row["ts"])] = row
        return sorted(entries.values(), key=lambda r: r["ts"], reverse=True)[:limit]
