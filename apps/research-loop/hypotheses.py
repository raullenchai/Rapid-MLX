"""Project hypotheses: locked append-only JSONL, derived evidence, read-only legacy views."""

import fcntl
import json
import os
import time
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Symlinks are not allowed.")
    if not path.exists():
        return []
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            continue
    return rows


def experiment_verdict(value):
    return {"supported": "improved", "refuted": "worse", "not run": "crashed"}.get(
        value, value
    )


def limit(task):
    value = task.get("experiments_per_hypothesis", 2)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("experiments_per_hypothesis must be a positive integer.")
    return value


def status(experiments, task):
    """Only terminal experiment verdicts count toward the conclusion threshold."""
    verdicts = [experiment_verdict(e.get("verdict")) for e in experiments]
    if "improved" in verdicts:
        return "supported"
    finished = [v for v in verdicts if v in ("worse", "no effect", "crashed")]
    if len(finished) >= limit(task):
        return "inconclusive" if all(v == "crashed" for v in finished) else "refuted"
    return "testing" if experiments else "untested"


def add(project, data, source="you"):
    statement = data.get("statement", data.get("hypothesis"))
    fields = [
        statement,
        data.get("why", ""),
        data.get("prediction", ""),
        data.get("change", ""),
    ]
    if (
        not isinstance(statement, str)
        or not statement.strip()
        or any(not isinstance(v, str) for v in fields)
        or sum(len(v) for v in fields) > 16000
    ):
        raise ValueError(
            "Hypothesis must contain a claim and text under 16,000 characters."
        )
    if source not in ("you", "discussion", "model"):
        raise ValueError("Invalid hypothesis source.")
    path = Path(project) / "hypotheses.jsonl"
    if path.is_symlink():
        raise ValueError("Symlinks are not allowed.")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        content = stream.read()
        # An interrupted append is not a committed event. Keep every complete row.
        if content and not content.endswith(b"\n"):
            content = content[: content.rfind(b"\n") + 1]
            stream.truncate(len(content))
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        # A legacy queue migration can be retried without duplicating a claim.
        key = data.get("migration_key")
        if key and (
            existing := next((r for r in rows if r.get("migration_key") == key), None)
        ):
            return existing
        number = (
            max(
                [int(r["id"][1:]) for r in rows if r.get("id", "")[1:].isdigit()] or [0]
            )
            + 1
        )
        row = dict(
            id=f"h{number}",
            statement=statement.strip(),
            why=fields[1],
            prediction=fields[2],
            source=source,
            created=time.time(),
        )
        if key:
            row["migration_key"] = key
        if fields[3]:
            # Optional legacy researcher experiment instruction.
            row["change"] = fields[3]
        stream.seek(0, 2)
        stream.write((json.dumps(row) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
        return row


def legacy_id(statement):
    # Same unsigned UTF-8 hash in the browser; this is an identity, not a secret.
    value = 14695981039346656037
    for byte in statement.encode():
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"legacy-{value:016x}"


def collect(project, task, extra_runs=()):
    """Join project claims to all runs. Never write or repair a run artifact."""
    hypotheses = {
        h["id"]: dict(h, experiments=[])
        for h in read_rows(Path(project) / "hypotheses.jsonl")
    }
    paths = set(Path(project).glob("runs/*/events.jsonl")) | {
        Path(p) for p in extra_runs
    }
    for path in sorted(paths):
        if path.parent.is_symlink() or path.parent.parent.is_symlink():
            raise ValueError("Symlinks are not allowed.")
        rounds = {}
        for event in read_rows(path):
            number = event.get("round", 0)
            if not number or event.get("event") not in ("idea", "result"):
                continue
            row = rounds.setdefault(
                number, {"round": number, "run": path.parent.name, "verdict": "testing"}
            )
            row.update(event)
        for row in rounds.values():
            statement = row.get("hypothesis") or f"Experiment {row['round']}"
            hid = row.get("hypothesis_id") or next(
                (h["id"] for h in hypotheses.values() if h["statement"] == statement),
                legacy_id(statement),
            )
            if hid not in hypotheses:
                hypotheses[hid] = dict(
                    id=hid,
                    statement=statement,
                    why=row.get("why", ""),
                    prediction=row.get("prediction", ""),
                    source="you"
                    if row.get("source") == "human"
                    else row.get("source", "model"),
                    created=row.get("ts", 0),
                    experiments=[],
                )
            row["hypothesis_id"] = hid
            row["verdict"] = experiment_verdict(row["verdict"])
            hypotheses[hid]["experiments"].append(row)
    # Show an old project queue immediately, even before its first v4 mutation.
    for row in read_rows(Path(project) / "ideas.jsonl"):
        statement = row.get("hypothesis", "")
        if statement and not any(
            h["statement"] == statement for h in hypotheses.values()
        ):
            hid = legacy_id(statement)
            hypotheses[hid] = dict(
                id=hid,
                statement=statement,
                why=row.get("why", ""),
                prediction=row.get("prediction", ""),
                source="you",
                created=row.get("ts", 0),
                experiments=[],
            )
    for h in hypotheses.values():
        h["status"] = status(h["experiments"], task)
    return list(hypotheses.values())


def migrate_queue(project):
    """Import the old project inbox idempotently. Leave old run inboxes untouched."""
    for i, row in enumerate(read_rows(Path(project) / "ideas.jsonl")):
        add(
            project,
            dict(
                row,
                migration_key=f"ideas:{i}:{row.get('ts', '')}:{row.get('hypothesis', '')}",
            ),
        )
