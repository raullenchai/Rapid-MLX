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
import sys
from pathlib import Path, PurePosixPath

_REVISION = re.compile(r"[0-9a-f]{40}")
_KEY = re.compile(r"[a-z][a-z0-9_]*")
_REPOSITORY = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)


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


def missing_pins(
    pins: dict[str, tuple[str, str, tuple[str, ...]]], cache_root: Path
) -> list[tuple[str, str, tuple[str, ...]]]:
    missing: list[tuple[str, str, tuple[str, ...]]] = []
    for repository, revision, files in pins.values():
        snapshot = snapshot_path(cache_root, repository, revision)
        missing_files: list[str] = []
        for file in files:
            try:
                # is_file follows the cache's blob symlinks. It catches both
                # absent entries and evicted blobs without opening model data.
                present = snapshot.is_dir() and (snapshot / file).is_file()
            except OSError:
                present = False
            if not present:
                missing_files.append(file)
        if missing_files:
            missing.append((repository, revision, tuple(missing_files)))
    return missing


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
    missing = missing_pins(pins, cache_root)
    if missing:
        print(
            f"sidecar cache preflight: FAIL: {len(missing)} immutable snapshot(s) "
            f"missing from {cache_root}",
            file=sys.stderr,
        )
        for repository, revision, missing_files in missing:
            print(f"  - {repository}@{revision}", file=sys.stderr)
            for file in missing_files:
                print(f"      missing: {file}", file=sys.stderr)
            print(
                '    restore: python3 -c "from huggingface_hub import '
                f"snapshot_download; snapshot_download('{repository}', "
                f"revision='{revision}')\"",
                file=sys.stderr,
            )
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
