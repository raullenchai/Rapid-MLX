"""Research Loop protocol 1. Standard-library validation, durable journal and artifacts."""

import ast
import csv
import difflib
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import hypotheses

PROTOCOL = 1
RULE_VERSION = hypotheses.RULE_VERSION
ARTIFACT_LIMIT = 5 * 1024 * 1024
RUN_LIMIT = 200 * 1024 * 1024
KINDS = {"code", "report", "data", "chart", "image", "log", "file"}
ID = r"[A-Za-z0-9_-]{1,120}"


def obj(properties):
    return dict(
        type="object",
        properties=properties,
        required=list(properties),
        additionalProperties=False,
    )


TEXT = {"type": "string"}
HYPOTHESIS = obj(
    {
        "statement": dict(
            TEXT, description="A testable claim: one sentence, at most 30 words."
        ),
        "why": TEXT,
        "prediction": TEXT,
    }
)
SCHEMAS = {
    "hypothesis": HYPOTHESIS,
    "experiment_plan": obj(
        {
            "hypothesis_id": {"type": ["string", "null"]},
            "new_hypothesis": {"anyOf": [HYPOTHESIS, {"type": "null"}]},
            "change": TEXT,
            "prediction": TEXT,
        }
    ),
    "edit": obj(
        {
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": obj(
                    {
                        "search_lines": {"type": "array", "minItems": 1, "items": TEXT},
                        "replace_lines": {"type": "array", "items": TEXT},
                    }
                ),
            },
            "base_sha256": TEXT,
        }
    ),
    "conclusion": obj({"hypothesis_id": TEXT, "summary_md": TEXT}),
    "run_report": obj({"summary_md": TEXT, "next_steps_md": TEXT}),
}


def experiment_plan_schema(open_ids, priority=None):
    schema = SCHEMAS["experiment_plan"]
    return dict(
        schema,
        properties=dict(
            schema["properties"],
            hypothesis_id={
                "type": ["string", "null"],
                "enum": [priority] if priority is not None else [*open_ids, None],
            },
        ),
    )


def validate(value, schema, path="message"):
    """Validate the exact JSON-schema vocabulary used by our constrained messages."""
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: must be one of {schema['enum']!r}")
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                validate(value, choice, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: does not match any allowed schema")
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    allowed = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "null": value is None,
    }
    if types and not any(allowed.get(t, False) for t in types):
        raise ValueError(f"{path}: expected {types}")
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        if set(schema.get("required", [])) - value.keys():
            raise ValueError(
                f"{path}: missing {sorted(set(schema['required']) - value.keys())}"
            )
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ValueError(
                f"{path}: unknown fields {sorted(value.keys() - props.keys())}"
            )
        for key in value:
            validate(value[key], props[key], f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: needs at least {schema['minItems']} items")
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{index}]")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def contained(root, relative):
    root = Path(root).absolute()
    if not isinstance(relative, (str, Path)) or any(ord(c) < 32 for c in str(relative)):
        raise ValueError("Invalid artifact path")
    relative = Path(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in ("..", ".") for p in relative.parts)
    ):
        raise ValueError("Path must be relative and inside artifact directory")
    path = root / relative
    for component in [root, *root.parents, path, *path.parents]:
        if component.is_symlink():
            raise ValueError("Symlinks are not allowed")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes artifact directory")
    return path


def read_file(root, relative, limit=ARTIFACT_LIMIT):
    path = contained(root, relative)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        import stat

        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Artifact must be a regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Per-artifact quota exceeded (5 MB)")
    return data


def events(run):
    path = contained(run, "events.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(dict(row, seq=row.get("seq", len(rows) + 1)))
        except (ValueError, UnicodeDecodeError):
            continue
    return rows


class Journal:
    def __init__(self, run):
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=True)

    def append(self, event, **data):
        with contained(self.run, "events.jsonl").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            return self._append(stream, event, data)

    def _append(self, stream, event, data):
        stream.seek(0, 2)
        end = stream.tell()
        offset, tail = end, b""
        while offset > 0 and tail.count(b"\n") < 2:
            amount = min(offset, 65536)
            offset -= amount
            stream.seek(offset)
            tail = stream.read(amount) + tail
        committed_end = (
            end
            if tail.endswith(b"\n")
            else offset + tail.rfind(b"\n") + 1
            if b"\n" in tail
            else 0
        )
        committed = tail[: committed_end - offset]
        last = json.loads(committed.splitlines()[-1]) if committed.strip() else None
        if last and (last.get("protocol") != PROTOCOL or "seq" not in last):
            raise ValueError("Legacy journals are read-only")
        if end != committed_end:
            stream.truncate(committed_end)
        sequence = last["seq"] if last else 0
        row = dict(
            data,
            protocol=PROTOCOL,
            seq=sequence + 1,
            ts=time.time(),
            event=event,
        )
        stream.seek(0, 2)
        stream.write((json.dumps(row, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
        return row


def validate_chart(spec):
    def channel(value, color=False):
        if not isinstance(value, dict) or set(value) - (
            {"field"} if color else {"field", "title", "scale"}
        ):
            raise ValueError("Invalid chart encoding")
        if not isinstance(value.get("field"), str) or not value["field"]:
            raise ValueError("Chart encoding needs a field")
        if "title" in value and not isinstance(value["title"], str):
            raise ValueError("Chart title must be text")
        if "scale" in value and value["scale"] not in (
            {"type": "linear"},
            {"type": "log"},
        ):
            raise ValueError("Chart scale must be linear or log")

    if not isinstance(spec, dict) or set(spec) - {
        "$schema",
        "mark",
        "data",
        "encoding",
        "layer",
        "band",
        "reference",
    }:
        raise ValueError("Unknown chart fields")
    if spec.get("$schema") != "rl-chart/1" or spec.get("mark") not in (
        "line",
        "point",
        "bar",
    ):
        raise ValueError("Invalid chart schema or mark")
    data = spec.get("data")
    if (
        not isinstance(data, dict)
        or set(data) != {"artifact"}
        or not re.fullmatch(ID, str(data["artifact"]))
    ):
        raise ValueError("Chart data must reference an artifact")
    encoding = spec.get("encoding")
    if (
        not isinstance(encoding, dict)
        or not {"x", "y"} <= encoding.keys()
        or encoding.keys() - {"x", "y", "color"}
    ):
        raise ValueError("Chart requires x and y encodings")
    for key, value in encoding.items():
        channel(value, key == "color")
    if "band" in spec:
        band = spec["band"]
        if (
            not isinstance(band, dict)
            or set(band) != {"low", "high"}
            or not all(isinstance(v, str) for v in band.values())
        ):
            raise ValueError("Invalid chart band")
    if "reference" in spec:
        ref = spec["reference"]
        if (
            not isinstance(ref, dict)
            or set(ref) != {"value", "label"}
            or not isinstance(ref["value"], (int, float))
            or isinstance(ref["value"], bool)
            or not math.isfinite(ref["value"])
            or not isinstance(ref["label"], str)
        ):
            raise ValueError("Invalid chart reference")
    if "layer" in spec:
        if not isinstance(spec["layer"], list) or len(spec["layer"]) > 12:
            raise ValueError("Chart permits at most 12 layers")
        for layer in spec["layer"]:
            if not isinstance(layer, dict) or set(layer) - {"mark", "encoding"}:
                raise ValueError("Layers only override mark/encoding")
            validate_chart(
                {k: v for k, v in dict(spec, **layer).items() if k != "layer"}
            )
    return spec


def data_rows(content, mime):
    text = content.decode("utf-8")
    if mime == "text/csv":
        reader = csv.reader(io.StringIO(text), strict=True)
        try:
            header = next(reader, [])
            if len(set(header)) != len(header) or any(not name for name in header):
                raise ValueError("CSV needs unique nonempty column names")
            rows = []
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError("CSV row does not match column count")
                rows.append(dict(zip(header, row, strict=True)))
            return rows
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV: {exc}") from exc

    def invalid_constant(value):
        raise ValueError(f"Nonfinite value {value} is not JSON")

    rows = [
        json.loads(line, parse_constant=invalid_constant)
        for line in text.splitlines()
        if line.strip()
    ]
    if any(
        not isinstance(r, dict) or any(isinstance(v, (list, dict)) for v in r.values())
        for r in rows
    ):
        raise ValueError("Data JSONL must contain flat objects")
    return rows


def sniff(content, name, kind):
    ext = Path(name).suffix.lower()
    text = None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if kind == "image":
        if ext == ".png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if ext in (".jpg", ".jpeg") and content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if ext == ".svg" and text is not None:
            import xml.etree.ElementTree as ET

            try:
                if ET.fromstring(text).tag in (
                    "svg",
                    "{http://www.w3.org/2000/svg}svg",
                ):
                    return "image/svg+xml"
            except ET.ParseError:
                pass
        raise ValueError("Image MIME does not match extension/content")
    if kind == "file":
        return "application/octet-stream"
    if (
        text is None
        or "\x00" in text
        or (text.lstrip().startswith(("<svg", "<!DOCTYPE html", "<html")))
    ):
        raise ValueError("Expected text; MIME does not match kind")
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip"):
        raise ValueError("MIME does not match extension")
    if kind == "data":
        mime = {".csv": "text/csv", ".jsonl": "application/x-ndjson"}.get(ext)
        if not mime:
            raise ValueError("Data must be CSV or flat JSONL")
        data_rows(content, mime)
        return mime
    if kind == "chart":
        if ext != ".json":
            raise ValueError("Chart requires .json")
        validate_chart(json.loads(text))
        return "application/json"
    if kind == "report" and ext != ".md":
        raise ValueError("Report requires .md")
    return "text/markdown" if kind == "report" else "text/plain"


class Artifacts:
    def __init__(self, run):
        self.run, self.journal = Path(run), Journal(run)

    def register(
        self,
        content,
        *,
        id,
        kind,
        title,
        path,
        owner,
        role="attachment",
        refs=(),
        producer="harness",
        provenance=None,
        columns=None,
    ):
        try:
            return self._register(
                content,
                id=id,
                kind=kind,
                title=title,
                path=path,
                owner=owner,
                role=role,
                refs=refs,
                producer=producer,
                provenance=provenance,
                columns=columns,
            )
        except (ValueError, OSError, TypeError) as exc:
            self.journal.append(
                "artifact_rejected", artifact_id=str(id), reason=str(exc)
            )
            raise

    def _register(
        self,
        content,
        *,
        id,
        kind,
        title,
        path,
        owner,
        role,
        refs,
        producer,
        provenance,
        columns,
    ):
        if (
            not re.fullmatch(ID, id)
            or kind not in KINDS
            or producer not in ("harness", "script", "model")
        ):
            raise ValueError("Invalid artifact identity, kind or producer")
        if not isinstance(title, str) or not isinstance(role, str):
            raise ValueError("Title and role must be text")
        validate_owner(owner)
        contained(self.run, path)
        if len(content) > ARTIFACT_LIMIT:
            raise ValueError("Per-artifact quota exceeded (5 MB)")
        mime = sniff(content, path, kind)
        if not isinstance(refs, (list, tuple)) or not all(
            isinstance(r, str) for r in refs
        ):
            raise ValueError("refs must be artifact IDs")
        if columns is not None and (
            not isinstance(columns, dict)
            or any(
                not isinstance(v, dict)
                or set(v) - {"type", "unit"}
                or any(not isinstance(x, str) for x in v.values())
                for v in columns.values()
            )
        ):
            raise ValueError("Column metadata must contain type/unit text")
        with contained(self.run, "events.jsonl").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            rows = events(self.run)
            artifacts = [e["artifact"] for e in rows if e["event"] == "artifact"]
            by_id = {a["id"]: a for a in artifacts}
            sha = digest(content)
            old = by_id.get(id)
            identical = next(
                (a for a in artifacts if a["id"] == id and a["sha256"] == sha), None
            )
            if identical:
                return identical
            if owner["run"] != self.run.name:
                raise ValueError("Artifact owner run does not match journal")
            related = [
                e["record"]
                for e in rows
                if e["event"] == "experiment_record"
                and e["record"]["experiment"] == owner.get("experiment")
                and e["record"]["attempt"]
                == owner.get("attempt", e["record"]["attempt"])
            ]
            if related:
                record = related[-1]
                if owner.get("hypothesis_id") != record["hypothesis_id"] or (
                    owner["scope"] == "seed" and owner["seed"] not in record["seeds"]
                ):
                    raise ValueError("Owner contradicts frozen experiment record")
            if any(ref not in by_id for ref in refs):
                raise ValueError("Unknown artifact reference")
            if kind == "chart":
                spec = json.loads(content)
                target = spec["data"]["artifact"]
                if target not in refs or by_id.get(target, {}).get("kind") != "data":
                    raise ValueError(
                        "Chart must reference a registered data artifact in refs"
                    )
                fields = set().union(
                    *(
                        r.keys()
                        for r in data_rows(
                            artifact_content(self.run, by_id[target]),
                            by_id[target]["mime"],
                        )
                    )
                )
                for layer in [spec, *spec.get("layer", [])]:
                    if any(
                        c["field"] not in fields
                        for c in layer.get("encoding", {}).values()
                    ):
                        raise ValueError("Chart field absent from data")
                if any(f not in fields for f in spec.get("band", {}).values()):
                    raise ValueError("Band field absent from data")
            if sum(a["bytes"] for a in artifacts) + len(content) > RUN_LIMIT:
                raise ValueError("Per-run quota exceeded (200 MB)")
            revision = old["revision"] + 1 if old else 1
            relative = f"artifacts/{id}-r{revision}{Path(path).suffix}"
            dest = contained(self.run, relative)
            dest.parent.mkdir(exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=dest.parent)
            try:
                with os.fdopen(fd, "wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, dest)
                directory = os.open(dest.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(temporary).unlink(missing_ok=True)
            copied = read_file(self.run, relative)
            envelope = dict(
                protocol=PROTOCOL,
                id=id,
                revision=revision,
                kind=kind,
                role=role,
                title=title,
                path=relative,
                mime=mime,
                bytes=len(copied),
                sha256=digest(copied),
                producer=producer,
                owner=owner,
                refs=list(refs),
                provenance=provenance
                or {
                    "model_id": None,
                    "prompt_id": None,
                    "code_sha256": None,
                    "env": {},
                },
                created=time.time(),
            )
            if old:
                envelope["supersedes"] = {"id": id, "revision": old["revision"]}
            if columns:
                envelope["columns"] = columns
            self.journal._append(stream, "artifact", {"artifact": envelope})
            return envelope

    def manifest(self, directory, owner, prefix, provenance=None):
        directory = Path(directory)
        registered = []
        try:
            path = contained(directory, "manifest.jsonl")
            if not path.exists():
                return []
            lines = read_file(directory, "manifest.jsonl").splitlines()
        except (ValueError, OSError) as exc:
            self.journal.append("artifact_rejected", reason=str(exc))
            return []
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) - {
                    "kind",
                    "path",
                    "title",
                    "role",
                    "refs",
                    "columns",
                }:
                    raise ValueError("Invalid manifest entry")
                content = read_file(directory, item["path"])
                registered.append(
                    self.register(
                        content,
                        id=f"{prefix}-{index}",
                        owner=owner,
                        producer="script",
                        provenance=provenance,
                        **item,
                    )
                )
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.journal.append(
                    "artifact_rejected", manifest_line=index + 1, reason=str(exc)
                )
        return registered


def validate_owner(owner):
    if (
        not isinstance(owner, dict)
        or owner.get("scope") not in ("run", "hypothesis", "experiment", "seed")
        or not isinstance(owner.get("run"), str)
    ):
        raise ValueError("Owner requires exactly one scope and run ID")
    required = {
        "run": {"run"},
        "hypothesis": {"run", "hypothesis_id"},
        "experiment": {"run", "experiment"},
        "seed": {"run", "experiment", "seed"},
    }[owner["scope"]]
    for key in ("experiment", "attempt", "seed"):
        if key in owner and (
            not isinstance(owner[key], int)
            or isinstance(owner[key], bool)
            or owner[key] < 0
        ):
            raise ValueError(f"Owner {key} must be a nonnegative integer")
    if owner.get("hypothesis_id") is not None and not isinstance(
        owner["hypothesis_id"], str
    ):
        raise ValueError("hypothesis_id must be a string or null")
    if owner["scope"] == "hypothesis" and not owner["hypothesis_id"]:
        raise ValueError("Hypothesis owner needs an ID")
    if not required <= owner.keys() or owner.keys() - {
        "scope",
        "run",
        "hypothesis_id",
        "experiment",
        "seed",
        "attempt",
    }:
        raise ValueError("Invalid owner related IDs")
    if owner["scope"] == "run" and set(owner) != {"scope", "run"}:
        raise ValueError("Run owner cannot contain child IDs")
    if owner["scope"] == "hypothesis" and ({"experiment", "seed"} & owner.keys()):
        raise ValueError("Hypothesis owner cannot contain experiment/seed")
    if owner["scope"] == "experiment" and "seed" in owner:
        raise ValueError("Experiment owner cannot contain seed")


def public_envelope(value):
    """Unknown/invalid envelopes stay downloadable, without gaining an active renderer."""
    envelope = dict(value) if isinstance(value, dict) else {}
    try:
        required = {
            "protocol",
            "id",
            "revision",
            "kind",
            "role",
            "title",
            "path",
            "mime",
            "bytes",
            "sha256",
            "producer",
            "owner",
            "refs",
            "provenance",
            "created",
        }
        if (
            not required <= envelope.keys()
            or envelope["protocol"] != PROTOCOL
            or envelope["kind"] not in KINDS
        ):
            raise ValueError("Unknown artifact kind or invalid envelope")
        if (
            not re.fullmatch(ID, envelope["id"])
            or not isinstance(envelope["revision"], int)
            or envelope["revision"] < 1
            or not re.fullmatch(r"[a-f0-9]{64}", envelope["sha256"])
        ):
            raise ValueError("Invalid identity/hash")
        if (
            not isinstance(envelope["bytes"], int)
            or envelope["bytes"] < 0
            or not isinstance(envelope["title"], str)
            or not isinstance(envelope["role"], str)
            or not isinstance(envelope["path"], str)
            or not isinstance(envelope["mime"], str)
            or not isinstance(envelope["refs"], list)
            or envelope["producer"] not in ("harness", "script", "model")
        ):
            raise ValueError("Invalid metadata")
        validate_owner(envelope["owner"])
    except (ValueError, TypeError, KeyError):
        envelope.update(
            kind="file", warning="Unknown kind or invalid envelope; download only"
        )
    return envelope


def artifact_content(run, envelope):
    content = (
        envelope["inline"].encode()
        if "inline" in envelope
        else read_file(run, envelope["path"])
    )
    if digest(content) != envelope["sha256"] or len(content) != envelope["bytes"]:
        raise ValueError("Artifact hash/size mismatch")
    return content


def apply_edits(src, message, fixed=(), filename="experiment.py"):
    validate(message, SCHEMAS["edit"])
    if message["base_sha256"] != digest(src.encode()):
        raise ValueError("base_sha256 differs from current source")
    lines = src.splitlines(keepends=True)
    plain = src.splitlines()
    patches = []
    for block in message["edits"]:
        old, new = block["search_lines"], block["replace_lines"]
        if any("\n" in line or "\r" in line for line in old + new):
            raise ValueError("Line arrays cannot contain newline characters")
        starts = [
            i
            for i in range(len(plain) - len(old) + 1)
            if plain[i : i + len(old)] == old
        ]
        if len(starts) != 1:
            match = difflib.SequenceMatcher(
                None, old, plain, autojunk=False
            ).find_longest_match()
            start = max(0, match.b - match.a)
            if not match.size:
                start = max(
                    range(len(plain)),
                    key=lambda i: difflib.SequenceMatcher(
                        None, old[0], plain[i]
                    ).ratio(),
                    default=0,
                )
            start = max(0, start - 2)
            nearest = "\n".join(
                f"{i + 1}: {plain[i]}"
                for i in range(
                    start, min(len(plain), start + min(40, max(6, len(old))))
                )
            )
            raise ValueError(
                f"SEARCH text found {len(starts)} times (must be exactly once): {old!r}\n"
                f"Nearest matching lines in the current base (line numbers are not source):\n{nearest}"
            )
        start, end = starts[0], starts[0] + len(old)
        if any(start < b and a < end for a, b, _ in patches):
            raise ValueError("Edit blocks overlap in the original source")
        replacement = [line + "\n" for line in new]
        if replacement and end == len(lines) and lines and not lines[-1].endswith("\n"):
            replacement[-1] = replacement[-1][:-1]
        patches.append((start, end, replacement))
    for start, end, replacement in sorted(patches, reverse=True):
        lines[start:end] = replacement
    result = "".join(lines)
    if filename.endswith(".py"):
        try:
            ast.parse(result)
        except SyntaxError as exc:
            raise ValueError(
                f"Python syntax error at line {exc.lineno}: {exc.msg}"
            ) from exc
    bad = [pattern for pattern in fixed if not re.search(pattern, result, re.M)]
    if bad:
        raise ValueError(f"edits changed lines that are fixed by the task: {bad}")
    return result


def parse_message(content, kind, strict, prompt):
    if strict:
        return json.loads(content)
    if kind == "edit":
        blocks = re.findall(
            r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", content, re.S
        )
        base = re.search(r"base_sha256: ([a-f0-9]{64})", prompt)
        if blocks and base:
            return {
                "base_sha256": base[1],
                "edits": [
                    {
                        "search_lines": old.splitlines(),
                        "replace_lines": new.splitlines(),
                    }
                    for old, new in blocks
                ],
            }
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    text = blocks[-1] if blocks else content
    try:
        value = json.loads(text)
    except ValueError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
                break
            except ValueError:
                continue
        else:
            raise ValueError("No JSON object in model response") from None
    legacy_keys = {"hypothesis_id", "hypothesis", "why", "prediction", "change"}
    if (
        kind == "experiment_plan"
        and isinstance(value, dict)
        and "new_hypothesis" not in value
        and set(value) <= legacy_keys
    ):
        value = {
            "hypothesis_id": value.get("hypothesis_id"),
            "new_hypothesis": {
                "statement": value.get("hypothesis", ""),
                "why": value.get("why", ""),
                "prediction": value.get("prediction", ""),
            }
            if value.get("hypothesis_id") is None
            else None,
            "change": value.get("change"),
            "prediction": value.get("prediction"),
        }
    return value


class Model:
    """Constrain, check completion, validate, retry; explicit older-engine fallback."""

    def __init__(
        self, base_url, model_id, log, system="You are a careful research assistant."
    ):
        self.base_url, self.model_id, self.log, self.system = (
            base_url,
            model_id,
            log,
            system,
        )
        self.strict = True
        self.prompt_id = None

    def message(
        self, kind, prompt, max_tokens=12000, check=None, retries=1, schema=None
    ):
        schema = SCHEMAS[kind] if schema is None else schema
        self.prompt_id = digest((kind + prompt).encode())
        initial_id = self.prompt_id
        for attempt in range(retries + 1):
            body = dict(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system},
                    {
                        "role": "user",
                        "content": prompt
                        + "\nReply with JSON matching this schema: "
                        + json.dumps(schema),
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.6,
            )
            while True:
                if self.strict:
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": kind,
                            "schema": schema,
                            "strict": True,
                        },
                    }
                else:
                    body.pop("response_format", None)
                try:
                    request = urllib.request.Request(
                        f"{self.base_url}/chat/completions",
                        json.dumps(body).encode(),
                        {"Content-Type": "application/json"},
                    )
                    response = json.load(urllib.request.urlopen(request, timeout=1800))
                    break
                except urllib.error.HTTPError as exc:
                    diagnostic = exc.read().decode(errors="replace")
                    if (
                        self.strict
                        and exc.code in (400, 422, 500, 501)
                        and re.search(
                            r"response_format|json_schema|guided|constrain",
                            diagnostic,
                            re.I,
                        )
                    ):
                        self.strict = False
                        self.log(
                            "model_format_fallback",
                            kind=kind,
                            prompt_id=initial_id,
                            diagnostic=diagnostic,
                        )
                        continue
                    raise
            path = "strict" if self.strict else "prompt-and-parse"
            choice = response.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content") or ""
            self.log(
                "model_attempt",
                kind=kind,
                prompt_id=initial_id,
                attempt=attempt,
                path=path,
                finish_reason=choice.get("finish_reason"),
            )
            try:
                if choice.get("finish_reason") != "stop":
                    raise ValueError(
                        f"finish_reason must be stop, got {choice.get('finish_reason')!r}"
                    )
                value = parse_message(content, kind, self.strict, prompt)
                validate(value, schema)
                result = check(value) if check else value
                self.log(
                    "model_message",
                    kind=kind,
                    prompt_id=initial_id,
                    message=value,
                )
                return result
            except (ValueError, TypeError, KeyError) as exc:
                self.log(
                    "model_rejected",
                    kind=kind,
                    prompt_id=initial_id,
                    attempt=attempt,
                    path=path,
                    reason=str(exc),
                    content=content,
                )
                if attempt == retries:
                    raise ValueError(f"{kind} rejected: {exc}") from exc
                prompt += (
                    f"\nPrevious proposal rejected: {exc}. Return a corrected message."
                )
                if kind == "edit":
                    prompt += (
                        " No rejected edits were applied. Copy SEARCH lines exactly from the "
                        "CURRENT BASE SOURCE above, including blank lines and indentation; "
                        "keep its base_sha256."
                    )


def default_chart(curve, data_id):
    series = curve.get("series", [])
    if not series:
        return None
    encoding = {
        "x": {"field": curve["x"], "title": curve.get("x_label", curve["x"])},
        "y": {
            "field": series[0]["key"],
            "title": series[0].get("label", series[0]["key"]),
        },
    }
    spec = {
        "$schema": "rl-chart/1",
        "mark": "point" if series[0].get("style") == "points" else "line",
        "data": {"artifact": data_id},
        "encoding": encoding,
    }
    if len(series) > 1:
        spec["layer"] = [
            {
                "mark": "point" if s.get("style") == "points" else "line",
                "encoding": {
                    "x": encoding["x"],
                    "y": {"field": s["key"], "title": s.get("label", s["key"])},
                },
            }
            for s in series
        ]
    return spec


def result_record(record, values, crashed=(), parent_value=None):
    """Incomplete evidence never earns a comparative verdict or promotes a parent."""
    metric = record["metric"]
    valid = {
        str(seed): row[metric]
        for seed, row in values.items()
        if isinstance(row.get(metric), (int, float))
        and not isinstance(row[metric], bool)
        and math.isfinite(row[metric])
    }
    missing = [s for s in record["seeds"] if str(s) not in valid and s not in crashed]
    aggregate = sum(valid.values()) / len(valid) if valid else None
    delta = (
        aggregate - parent_value
        if aggregate is not None and parent_value is not None
        else None
    )
    if not valid and crashed and not missing:
        verdict = "crashed"
    elif missing or crashed:
        verdict = "incomplete"
    elif delta is None:
        verdict = "no effect"
    else:
        signed = delta * (1 if record["direction"] == "max" else -1)
        verdict = (
            "improved"
            if signed > record["noise"]
            else "worse"
            if signed < -record["noise"]
            else "no effect"
        )
    return dict(
        experiment=record["experiment"],
        attempt=record["attempt"],
        hypothesis_id=record["hypothesis_id"],
        per_seed=valid,
        missing_seeds=missing,
        crashed_seeds=list(crashed),
        aggregate=aggregate,
        delta=delta,
        verdict=verdict,
        decision_rule_version=RULE_VERSION,
    )


def runtime_env(task):
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "variables": task.get("env", {}),
    }


def frozen_record(
    experiment,
    attempt,
    task,
    source,
    *,
    hypothesis_id=None,
    parent_experiment=None,
    change="Baseline",
    model_id=None,
    prompt_id=None,
):
    return dict(
        experiment=experiment,
        attempt=attempt,
        hypothesis_id=hypothesis_id,
        parent_experiment=parent_experiment,
        change=change,
        metric=task["metric"],
        direction=task["direction"],
        budget={"fixed": task.get("fixed", []), "env": task.get("env", {})},
        env=runtime_env(task),
        seeds=list(task["seeds"]),
        aggregation="mean",
        noise=task["noise"],
        decision_rule_version=RULE_VERSION,
        code_sha256=digest(source.encode()),
        model_id=model_id,
        prompt_id=prompt_id,
    )


def snapshot(run, task):
    """One journal read defines the watermark. Old files are projected, never changed."""
    rows = events(run)
    watermark = rows[-1]["seq"] if rows else 0
    modern = any(e.get("protocol") == PROTOCOL for e in rows)
    if modern:
        return dict(
            protocol=PROTOCOL,
            watermark=watermark,
            events=rows,
            artifacts=[
                public_envelope(e["artifact"]) for e in rows if e["event"] == "artifact"
            ],
            records=[
                {"type": e["event"], **e["record"]}
                for e in rows
                if e["event"] in ("experiment_record", "result_record")
            ],
            legacy=False,
        )
    return legacy_snapshot(run, task, rows, watermark)


def legacy_snapshot(run, task, rows, watermark):
    run = Path(run)
    artifacts, research = [], []
    created = rows[0].get("ts", 0) if rows else 0

    def artifact(id, kind, role, title, path, owner, content=None, refs=()):
        try:
            inline = content is not None
            content = content if inline else read_file(run, path)
            mime = sniff(content, path, kind)
            row = dict(
                protocol=PROTOCOL,
                id=id,
                revision=1,
                kind=kind,
                role=role,
                title=title,
                path=path,
                mime=mime,
                bytes=len(content),
                sha256=digest(content),
                producer="harness",
                owner=owner,
                refs=list(refs),
                provenance={
                    "model_id": None,
                    "prompt_id": None,
                    "code_sha256": None,
                    "env": {},
                    "legacy": True,
                },
                created=created,
            )
            if inline:
                row["inline"] = content.decode()
            artifacts.append(row)
            return row
        except (ValueError, OSError):
            return None

    numbers = sorted(
        {
            e.get("round", 0)
            for e in rows
            if e.get("event") in ("idea", "baseline", "run_start", "result", "curve")
        }
    )
    parent_values = {}
    for number in numbers:
        related = [e for e in rows if e.get("round", 0) == number]
        idea = next((e for e in related if e["event"] == "idea"), {})
        hid = idea.get("hypothesis_id") or (
            hypotheses.legacy_id(idea.get("hypothesis", f"Experiment {number}"))
            if number
            else None
        )
        owner = dict(
            scope="experiment", run=run.name, experiment=number, hypothesis_id=hid
        )
        filename = f"round{number}_{task['editable']}"
        code = artifact(
            f"e{number}-code", "code", "primary_code", filename, filename, owner
        )
        record = frozen_record(
            number,
            0,
            task,
            "",
            hypothesis_id=hid,
            parent_experiment=idea.get("parent", 0) if number else None,
            change=idea.get("change", "Baseline"),
        )
        record.update(
            code_sha256=code["sha256"] if code else None,
            legacy=True,
            decision_rule_version="legacy",
            env={"python": None, "platform": None, "variables": task.get("env", {})},
        )
        research.append(dict(type="experiment_record", **record))
        result = next(
            (e for e in reversed(related) if e["event"] in ("result", "baseline")), None
        )
        if result:
            scalar = result.get("value", result.get("result", {}).get(task["metric"]))
            parent_values[number] = scalar
            seeds = result.get("result", {}).get("per_seed", [])
            per_seed = {
                str(s): v.get(task["metric"])
                for s, v in zip(task["seeds"], seeds, strict=False)
            }
            research.append(
                dict(
                    type="result_record",
                    experiment=number,
                    attempt=0,
                    hypothesis_id=hid,
                    per_seed=per_seed,
                    missing_seeds=[s for s in task["seeds"] if str(s) not in per_seed],
                    crashed_seeds=[],
                    aggregate=scalar,
                    delta=result.get("delta"),
                    verdict=hypotheses.experiment_verdict(
                        result.get("verdict", "no effect")
                    ),
                    legacy=True,
                    decision_rule_version="legacy",
                )
            )
        curve = task.get("curve")
        if curve:
            for seed in task["seeds"]:
                points = [
                    {
                        k: v
                        for k, v in e.items()
                        if k
                        not in ("event", "round", "seq", "seed", "ts", "hypothesis_id")
                    }
                    for e in related
                    if e["event"] == "curve" and e.get("seed") == seed
                ]
                if not points:
                    try:
                        points = json.loads(
                            read_file(run, f"round{number}_seed{seed}_curve.json")
                        )
                    except (ValueError, OSError):
                        pass
                if points:
                    seed_owner = dict(owner, scope="seed", seed=seed)
                    data_id = f"e{number}-s{seed}-curve"
                    data = artifact(
                        data_id,
                        "data",
                        "curve",
                        f"Measurements · seed {seed}",
                        f"{data_id}.jsonl",
                        seed_owner,
                        ("\n".join(json.dumps(p) for p in points) + "\n").encode(),
                    )
                    spec = default_chart(curve, data_id)
                    if data and spec:
                        artifact(
                            f"{data_id}-chart",
                            "chart",
                            "curve",
                            f"Curve · seed {seed}",
                            "curve.chart.json",
                            seed_owner,
                            json.dumps(spec).encode(),
                            [data_id],
                        )
        parent = idea.get("parent", 0)
        if number and code:
            try:
                import difflib

                original = read_file(run, f"round{parent}_{task['editable']}").decode()
                current = artifact_content(run, code).decode()
                diff = "".join(
                    difflib.unified_diff(
                        original.splitlines(True),
                        current.splitlines(True),
                        f"experiment-{parent}",
                        f"experiment-{number}",
                    )
                )
                artifact(
                    f"e{number}-diff",
                    "code",
                    "diff",
                    f"Diff vs experiment {parent}",
                    "change.diff",
                    owner,
                    diff.encode(),
                    [f"e{parent}-code"],
                )
            except (ValueError, OSError):
                pass
    if (run / "loop.log").exists():
        artifact(
            "run-log",
            "log",
            "stdout",
            "Loop log",
            "loop.log",
            {"scope": "run", "run": run.name},
        )
    return dict(
        protocol=PROTOCOL,
        watermark=watermark,
        events=rows,
        artifacts=artifacts,
        records=research,
        legacy=True,
    )


def ledger_csv(records):
    output = io.StringIO()
    columns = [
        "experiment",
        "attempt",
        "hypothesis_id",
        "aggregate",
        "delta",
        "verdict",
        "missing_seeds",
        "crashed_seeds",
        "decision_rule_version",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(r for r in records if r["type"] == "result_record")
    return output.getvalue().encode()


def bundle(run, view):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "PROTOCOL.md", Path(__file__).with_name("PROTOCOL.md").read_bytes()
        )
        archive.writestr("envelopes.json", json.dumps(view["artifacts"], indent=2))
        archive.writestr("records.json", json.dumps(view["records"], indent=2))
        archive.writestr(
            "events.jsonl", "".join(json.dumps(e) + "\n" for e in view["events"])
        )
        archive.writestr("ledger.csv", ledger_csv(view["records"]))
        for index, artifact in enumerate(view["artifacts"]):
            name = re.sub(
                r"[^A-Za-z0-9_-]", "_", str(artifact.get("id", f"invalid-{index}"))
            )
            revision = re.sub(r"[^0-9]", "", str(artifact.get("revision", 1))) or "1"
            suffix = Path(str(artifact.get("path", "file.bin"))).suffix
            if not re.fullmatch(r"\.[A-Za-z0-9]{1,16}", suffix):
                suffix = ".bin"
            try:
                archive.writestr(
                    f"artifacts/{name}-r{revision}{suffix}",
                    artifact_content(run, artifact),
                )
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
                archive.writestr(f"warnings/{index}-{name}.txt", str(exc))
    return output.getvalue()
