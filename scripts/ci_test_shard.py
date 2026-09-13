#!/usr/bin/env python3
"""Partition pytest files deterministically without collecting/importing them.

The hosted Linux matrix runs each shard on a separate clean runner.  We balance
whole files by physical line count so module-scoped fixtures and import state
never straddle workers, while keeping planning fast and dependency-free.  Both
repository pytest configurations restrict discovery to ``test_*.py``; a
contract test keeps that setting synchronized with this planner.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestFile:
    path: str
    weight: int


def discover(root: Path, suite: str) -> list[TestFile]:
    tests_root = root / "tests"
    if not tests_root.is_dir():
        raise ValueError(f"missing tests directory: {tests_root}")

    files: list[TestFile] = []
    for path in tests_root.rglob("test_*.py"):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("tests/integrations/"):
            continue
        is_headless = relative.startswith("tests/headless_mlx/")
        if (suite == "headless") != is_headless:
            continue
        # A one-line floor keeps empty/new placeholder modules represented.
        weight = max(1, sum(1 for _ in path.open(encoding="utf-8")))
        files.append(TestFile(relative, weight))

    if not files:
        raise ValueError(f"no {suite} test files discovered below {tests_root}")
    return sorted(files, key=lambda item: item.path)


def partition(files: Iterable[TestFile], shard_count: int) -> list[list[TestFile]]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")

    shards: list[list[TestFile]] = [[] for _ in range(shard_count)]
    weights = [0] * shard_count
    # Longest-processing-time scheduling is deterministic here: path breaks
    # equal-weight ties, then the lowest shard number breaks equal totals.
    for item in sorted(files, key=lambda value: (-value.weight, value.path)):
        target = min(range(shard_count), key=lambda index: (weights[index], index))
        shards[target].append(item)
        weights[target] += item.weight
    for shard in shards:
        shard.sort(key=lambda value: value.path)
    return shards


def ignored_paths(
    root: Path, suite: str, shard_index: int, shard_count: int
) -> list[str]:
    if not 1 <= shard_index <= shard_count:
        raise ValueError(
            f"shard index must be between 1 and {shard_count}, got {shard_index}"
        )
    files = discover(root, suite)
    shards = partition(files, shard_count)
    selected = {item.path for item in shards[shard_index - 1]}
    ignored = sorted(item.path for item in files if item.path not in selected)
    if not selected:
        raise ValueError(f"shard {shard_index}/{shard_count} selected no files")
    return ignored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--suite", choices=("ordinary", "headless"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    try:
        ignored = ignored_paths(
            args.root.resolve(), args.suite, args.shard_index, args.shard_count
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    for path in ignored:
        print(f"--ignore={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
