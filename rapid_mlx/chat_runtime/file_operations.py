# SPDX-License-Identifier: Apache-2.0
"""Read-only workspace operations, run in a disposable bounded worker.

No model-supplied shell commands are executed. Search reads only files opened
through the workspace policy and pinned file descriptors.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

from rapid_mlx.local_search.extraction import (
    open_root,
    read_safe,
)

from .file_extraction import MAX_INPUT_BYTES, DocumentError, extract_document
from .path_policy import FileError, resolve_path, sensitive

MAX_OUTPUT = 18000
MAX_FILES = 10000
MAX_SEARCH_BYTES = 1024 * 1024
SEARCH_SECONDS = 5
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__"}


def read_file(workspace, path, offset=1, limit=200, *, output_cap=MAX_OUTPUT):
    if offset < 1 or not 1 <= limit <= 2000:
        raise FileError(
            "invalid_arguments", "Use offset >= 1 and limit from 1 to 2000 lines."
        )
    root, target = resolve_path(workspace, path)
    data, _ = read_safe(target, root, MAX_INPUT_BYTES)
    text = extract_document(data, target.name)
    lines = text.splitlines()
    # A single long/minified line must remain pageable without silently losing
    # its tail. Display rows are stable across all pages of this extraction.
    rows = [
        line[i : i + 2000] for line in lines for i in range(0, max(1, len(line)), 2000)
    ]
    wrapped = len(rows) != len(lines)
    selected = []
    size = 0
    for index in range(offset - 1, min(len(rows), offset - 1 + limit)):
        line = f"{index + 1}|{rows[index]}"
        if (
            size + len(json.dumps(line, ensure_ascii=False).encode("utf-8")) + 1
            > output_cap - 800
        ):
            break
        selected.append(line)
        size += len(json.dumps(line, ensure_ascii=False).encode("utf-8")) + 1
    if not selected and offset <= len(rows):
        raise FileError(
            "context_limit",
            "This line cannot fit the remaining context budget; ask a focused question in a new turn.",
        )
    next_offset = offset + len(selected)
    truncated = next_offset <= len(rows)
    content = "\n".join(selected)
    if truncated:
        content += f"\n[Truncated: use read_file(path={path!r}, offset={next_offset}, limit={limit}) to continue.]"
    return {
        "path": str(target.relative_to(root)),
        "content": content,
        "total_lines": len(rows),
        "next_offset": next_offset if truncated else None,
        "truncated": truncated,
        "note": (
            "Lines longer than 2000 characters are split into numbered display rows."
            if wrapped
            else "Offsets are 1-based line numbers."
        ),
    }


def _segments_match(pattern: str, value: str) -> bool:
    """Slash-aware glob matching; ** spans zero or more path components."""
    pats, parts = pattern.split("/"), value.split("/")

    def match(i: int, j: int) -> bool:
        if i == len(pats):
            return j == len(parts)
        if pats[i] == "**":
            if i == len(pats) - 1:
                return j < len(parts)
            return match(i + 1, j) or (j < len(parts) and match(i, j + 1))
        return (
            j < len(parts)
            and fnmatch.fnmatchcase(parts[j], pats[i])
            and match(i + 1, j + 1)
        )

    return match(0, 0)


def matches(pattern: str, relative: str) -> bool:
    pattern = pattern.removeprefix("./")
    if "/" not in pattern:
        return fnmatch.fnmatchcase(relative.rsplit("/", 1)[-1], pattern)
    return _segments_match(pattern.lstrip("/"), relative)


def _ignore_rules(root, directory):
    try:
        data, _ = read_safe(directory / ".gitignore", root, 65536)
    except (OSError, ValueError):
        return []
    rules = []
    for raw in data.decode("utf-8", errors="replace").splitlines():
        # Git ignores unescaped trailing spaces, and permits literal # and !.
        line = re.sub(r"(?<!\\) +$", "", raw)
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        line = line[1:] if negate else line
        line = line.replace(r"\#", "#").replace(r"\!", "!").replace(r"\ ", " ")
        directory_only = line.endswith("/")
        anchored = line.startswith("/") or "/" in line.rstrip("/")
        rules.append((directory, line.strip("/"), negate, directory_only, anchored))
    return rules


def _ignored(path, directory, rules):
    ignored = False
    for base, pattern, negate, directory_only, anchored in rules:
        relative = path.relative_to(base).as_posix()
        parts = relative.split("/")
        candidates = (
            ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
            if anchored
            else parts
        )
        for index, candidate in enumerate(candidates):
            is_dir = index < len(parts) - 1 or directory
            if (not directory_only or is_dir) and _segments_match(pattern, candidate):
                ignored = not negate
                break
    return ignored


def walk_files(root: Path, target: Path, deadline: float):
    """Pinned traversal; ignored directories and symlinks are never descended."""
    # Include ancestors' ignore rules even when search starts in a subfolder.
    rules = []
    parent = target if target.is_dir() else target.parent
    current = root
    rules.extend(_ignore_rules(root, current))
    if (current / "pyvenv.cfg").exists():
        return
    for part in parent.relative_to(root).parts:
        current /= part
        if (current / "pyvenv.cfg").exists():
            return
        if (
            _ignored(current, True, rules)
            or sensitive(current)
            or current.name.lower() in SKIP_DIRS
        ):
            return
        rules.extend(_ignore_rules(root, current))
    if target.is_file():
        if not _ignored(target, False, rules):
            yield target
        return
    with open_root(target) as start:

        def walk(fd, directory, inherited, depth):
            if depth > 64 or time.monotonic() > deadline:
                raise FileError(
                    "search_limit",
                    "Search reached its time/depth limit; narrow the path or pattern.",
                )
            with os.scandir(fd) as entries:
                for entry in entries:
                    if time.monotonic() > deadline:
                        raise FileError(
                            "search_limit",
                            "Search timed out; narrow the path or pattern.",
                        )
                    path = directory / entry.name
                    info = entry.stat(follow_symlinks=False)
                    is_dir = stat.S_ISDIR(info.st_mode)
                    if sensitive(path) or _ignored(path, is_dir, inherited):
                        continue
                    if is_dir:
                        if entry.name.lower() in SKIP_DIRS:
                            continue
                        child = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd,
                        )
                        try:
                            try:
                                os.stat(
                                    "pyvenv.cfg", dir_fd=child, follow_symlinks=False
                                )
                            except FileNotFoundError:
                                yield from walk(
                                    child,
                                    path,
                                    inherited + _ignore_rules(root, path),
                                    depth + 1,
                                )
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode):
                        yield path

        yield from walk(start, target, rules, 0)


def search(
    workspace,
    pattern,
    path=".",
    glob=None,
    case=True,
    max_results=50,
    *,
    filenames=False,
    output_cap=MAX_OUTPUT,
):
    if not pattern or len(pattern) > 1000 or not 1 <= max_results <= 200:
        raise FileError(
            "invalid_arguments",
            "Use a nonempty pattern (up to 1000 characters) and max_results from 1 to 200.",
        )
    root, target = resolve_path(workspace, path)
    if target.name.lower() in SKIP_DIRS or (
        target.is_dir() and (target / "pyvenv.cfg").exists()
    ):
        return {"matches": [], "truncated": False}
    deadline = time.monotonic() + SEARCH_SECONDS
    expression = None
    if not filenames:
        try:
            expression = re.compile(pattern, 0 if case else re.IGNORECASE)
        except re.error:
            raise FileError("invalid_pattern", "Invalid regular expression.") from None
    results, scanned, size = [], 0, 0
    truncated = False
    skipped = 0
    candidates = []
    try:
        for candidate in walk_files(root, target, deadline):
            scanned += 1
            if scanned > MAX_FILES:
                truncated = True
                break
            relative = candidate.relative_to(
                target if target.is_dir() else target.parent
            ).as_posix()
            display = candidate.relative_to(root).as_posix()
            if filenames:
                if matches(pattern, relative):
                    candidates.append((candidate.stat().st_mtime_ns, display))
                continue
            if glob and not matches(glob, relative):
                continue
            try:
                data, _ = read_safe(candidate, root, MAX_SEARCH_BYTES)
                text = data.decode("utf-8")
                if any(ord(c) < 32 and c not in "\n\r\t\f" for c in text):
                    continue
            except (OSError, ValueError) as error:
                # Binary files are expected exclusions; inaccessible/oversized
                # text may hide matches and must not produce a complete result.
                if not isinstance(error, UnicodeDecodeError):
                    skipped += 1
                continue
            hits = (
                (number, line)
                for number, line in enumerate(text.splitlines(), 1)
                if expression.search(line)
            )
            for number, line in hits:
                item = {"path": display, "line": number, "text": line[:1000]}
                if len(line) > 1000:
                    item["text"] += " [Line truncated; use read_file to inspect.]"
                item_size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
                if len(results) >= max_results or size + item_size > output_cap - 800:
                    truncated = True
                    break
                results.append(item)
                size += item_size
            if truncated:
                break
    except FileError as error:
        if error.code != "search_limit":
            raise
        truncated = True
    if filenames:
        for _, display in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if (
                len(results) >= max_results
                or size + len(display.encode("utf-8")) > output_cap - 800
            ):
                truncated = True
                break
            results.append(display)
            size += len(display.encode("utf-8")) + 4
    return {
        "matches": results,
        "truncated": truncated or bool(skipped),
        "skipped_files": skipped,
        "note": (
            "Incomplete: scan/output limit reached or unreadable/oversized files skipped; narrow the path or use read_file for a specific file."
            if truncated or skipped
            else ""
        ),
        "backend": "glob" if filenames else "python",
    }


def dispatch(request):
    try:
        args = request["args"]
        cap = request.get("output_cap", MAX_OUTPUT)
        if cap < 1024:
            raise FileError(
                "context_limit",
                "Tool context budget exhausted; ask a focused question in a new turn.",
            )
        if request["operation"] == "read_file":
            return read_file(request.get("workspace"), **args, output_cap=cap)
        return search(
            request.get("workspace"),
            **args,
            filenames=request["operation"] == "glob",
            output_cap=cap,
        )
    except (FileError, DocumentError) as error:
        return {"error": {"code": error.code, "message": error.message}}
    except FileNotFoundError:
        return {"error": {"code": "not_found", "message": "File or folder not found."}}
    except (OSError, ValueError, RuntimeError):
        return {
            "error": {
                "code": "unreadable",
                "message": "Cannot read this file safely; it may be unavailable, too large, or changed during access.",
            }
        }


if __name__ == "__main__":
    print(json.dumps(dispatch(json.load(sys.stdin)), ensure_ascii=False))
