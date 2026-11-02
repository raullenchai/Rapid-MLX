#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail fast when an immutable release sidecar snapshot is not cached.

The release gate is intentionally offline.  This preflight mirrors the cache
coordinate used by ``huggingface_hub.snapshot_download(..., revision=<sha>,
local_files_only=True)`` without importing or loading model code.  It runs
before the release venv, agent smoke, sidecar build, or any model load.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

_REVISION = re.compile(r"[0-9a-f]{40}")
_KEY = re.compile(r"[a-z][a-z0-9_]*")
_REPOSITORY = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)
_READ_TIMEOUT_SECONDS = 5


class PreflightError(Exception):
    """The pin manifest or local cache cannot satisfy the offline gate."""


def default_cache_root() -> Path:
    if value := os.environ.get("HF_HUB_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("HF_HOME"):
        return Path(value).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def load_pins(path: Path) -> dict[str, tuple[str, str, tuple[str, ...]]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read pin manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise PreflightError("sidecar pin manifest must be an object with schema 1")
    models = payload.get("models")
    required_models = {"qwen", "gemma", "flux"}
    if not isinstance(models, dict) or set(models) != required_models:
        raise PreflightError(
            "sidecar pin manifest must define exactly qwen, gemma, and flux"
        )

    result: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for key, entry in models.items():
        if not _KEY.fullmatch(key) or not isinstance(entry, dict):
            raise PreflightError(f"invalid sidecar pin entry {key!r}")
        repository = entry.get("repository")
        revision = entry.get("revision")
        files = entry.get("files")
        if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
            raise PreflightError(f"{key}.repository must be an owner/name ID")
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise PreflightError(f"{key}.revision must be a full lowercase commit SHA")
        if (
            not isinstance(files, list)
            or not files
            or not all(isinstance(file, str) for file in files)
            or len(files) != len(set(files))
        ):
            raise PreflightError(f"{key}.files must be a non-empty unique string list")
        for file in files:
            file_path = PurePosixPath(file)
            if (
                file_path.is_absolute()
                or not file_path.parts
                or any(part in {"", ".", ".."} for part in file_path.parts)
                or "\\" in file
                or "\n" in file
            ):
                raise PreflightError(f"{key}.files contains an unsafe path {file!r}")
        result[key] = (repository, revision, tuple(files))
    return result


def snapshot_path(cache_root: Path, repository: str, revision: str) -> Path:
    return (
        cache_root / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    )


# Per-file probe outcomes.  ``BLOCKED`` means the path exists but opening it
# was denied or timed out, which on a macOS runner usually indicates a missing
# privacy (Full Disk Access) grant rather than an absent download.
READABLE = "readable"
MISSING = "missing"
BLOCKED = "blocked"
UNREADABLE = "unreadable"

_PERMISSION_EXIT = 13
_PROBE = (
    "import sys\n"
    "try:\n"
    "    open(sys.argv[1], 'rb').read(1)\n"
    "except PermissionError:\n"
    f"    sys.exit({_PERMISSION_EXIT})\n"
)


def probe_file(path: Path) -> str:
    """Probe one byte out-of-process so a macOS TCC denial cannot hang CI."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _PROBE, str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_READ_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return BLOCKED
    except (OSError, subprocess.SubprocessError):
        return UNREADABLE
    if result.returncode == 0:
        return READABLE
    if result.returncode == _PERMISSION_EXIT:
        return BLOCKED
    return UNREADABLE


def file_is_readable(path: Path) -> bool:
    return probe_file(path) == READABLE


def unavailable_pins(
    pins: dict[str, tuple[str, str, tuple[str, ...]]], cache_root: Path
) -> list[tuple[str, str, tuple[tuple[str, str], ...]]]:
    """Return ``(repository, revision, ((file, outcome), ...))`` per bad pin."""
    unavailable: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for repository, revision, files in pins.values():
        snapshot = snapshot_path(cache_root, repository, revision)
        bad_files: list[tuple[str, str]] = []
        for file in files:
            candidate = snapshot / file
            try:
                # Reading one byte catches macOS privacy/TCC denials on a
                # warm-tier symlink. ``is_file`` alone can succeed while the
                # release runner later blocks forever opening the same blob.
                # The probe is an isolated, bounded subprocess for that reason.
                if stat.S_ISDIR(snapshot.stat().st_mode) and stat.S_ISREG(
                    candidate.stat().st_mode
                ):
                    outcome = probe_file(candidate)
                else:
                    outcome = MISSING
            except (FileNotFoundError, NotADirectoryError):
                outcome = MISSING
            except PermissionError:
                outcome = BLOCKED
            except OSError:
                outcome = UNREADABLE
            if outcome == READABLE:
                continue
            bad_files.append((file, outcome))
            # A path that exists but cannot be opened normally indicates a
            # snapshot-wide host access boundary. Stop after the first
            # proof rather than multiplying the timeout by every shard.
            if outcome != MISSING:
                break
        if bad_files:
            unavailable.append((repository, revision, tuple(bad_files)))
    return unavailable


def missing_pins(
    pins: dict[str, tuple[str, str, tuple[str, ...]]], cache_root: Path
) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (repository, revision, tuple(file for file, _ in bad_files))
        for repository, revision, bad_files in unavailable_pins(pins, cache_root)
    ]


def write_outputs(
    path: Path, pins: dict[str, tuple[str, str, tuple[str, ...]]]
) -> None:
    lines: list[str] = []
    for key, (repository, revision, _) in pins.items():
        lines.extend((f"{key}_model={repository}", f"{key}_revision={revision}"))
    with path.open("a") as output:
        output.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--github-output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        pins = load_pins(args.manifest)
    except PreflightError as exc:
        print(f"sidecar cache preflight: FAIL: {exc}", file=sys.stderr)
        return 2

    cache_root = (args.cache_root or default_cache_root()).expanduser()
    unavailable = unavailable_pins(pins, cache_root)
    if unavailable:
        absent = [
            entry
            for entry in unavailable
            if any(outcome != BLOCKED for _, outcome in entry[2])
        ]
        blocked = [
            (repository, revision, file)
            for repository, revision, bad_files in unavailable
            for file, outcome in bad_files
            if outcome == BLOCKED
        ]
        if absent:
            print(
                f"sidecar cache preflight: FAIL: {len(absent)} immutable snapshot(s) "
                f"missing or unreadable in {cache_root}",
                file=sys.stderr,
            )
            for repository, revision, bad_files in absent:
                print(f"  - {repository}@{revision}", file=sys.stderr)
                for file, outcome in bad_files:
                    if outcome != BLOCKED:
                        print(f"      unavailable: {file}", file=sys.stderr)
                print(
                    '    repair cache access, or restore: python3 -c "from huggingface_hub import '
                    f"snapshot_download; snapshot_download('{repository}', "
                    f"revision='{revision}')\"",
                    file=sys.stderr,
                )
        if blocked:
            print(
                f"sidecar cache preflight: FAIL: {len(blocked)} file(s) exist but "
                f"could not be read (permission denied or read timed out) in "
                f"{cache_root}. On a macOS self-hosted runner this usually means "
                "the runner service lacks Full Disk Access to the volume holding "
                "the cache; grant it to the runner binaries and restart the "
                "service. Do not re-download.",
                file=sys.stderr,
            )
            for repository, revision, file in blocked:
                print(f"  - {repository}@{revision}: blocked: {file}", file=sys.stderr)
        print(
            "No download was attempted; release gate remains offline.", file=sys.stderr
        )
        return 1

    if args.github_output is not None:
        write_outputs(args.github_output, pins)
    print(
        f"sidecar cache preflight: PASS: {len(pins)} pinned snapshots in {cache_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
