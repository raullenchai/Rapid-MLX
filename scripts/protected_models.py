#!/usr/bin/env python3
"""Validate and print the host-hygiene protected model inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("protected_models.json")


def cache_name(repository: str) -> str:
    return "models--" + repository.replace("/", "--")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, object]]:
    raw = json.loads(path.read_text())
    if raw.get("schema") != 1 or not isinstance(raw.get("models"), list):
        raise ValueError("protected-model manifest must use schema 1 with models")
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for model in raw["models"]:
        repository = model.get("repository")
        sources = model.get("sources")
        if not isinstance(repository, str) or repository.count("/") != 1:
            raise ValueError(f"invalid protected repository: {repository!r}")
        if repository in seen:
            raise ValueError(f"duplicate protected repository: {repository}")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(source, str) for source in sources)
        ):
            raise ValueError(f"{repository} needs non-empty sources")
        revision = model.get("revision")
        if revision is not None and (
            not isinstance(revision, str) or len(revision) != 40
        ):
            raise ValueError(f"{repository} has an invalid pinned revision")
        seen.add(repository)
        result.append(model)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-names", action="store_true")
    args = parser.parse_args(argv)
    for model in load_manifest(args.manifest):
        repository = str(model["repository"])
        if args.cache_names:
            print(cache_name(repository))
            continue
        revision = model.get("revision")
        pin = f"@{revision}" if revision else ""
        print(f"{repository}{pin}\t{','.join(model['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
