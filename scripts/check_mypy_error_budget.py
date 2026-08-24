#!/usr/bin/env python3
"""Enforce a shrink-only, per-file mypy error budget.

This deliberately does not infer semantic identity from diagnostic text or line
numbers. Existing dirty files are grandfathered at a reviewed count; new dirty
files and per-file count growth fail. When a count falls, the checked-in
baseline must be tightened so the improvement cannot regress.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

BASELINE_PATH = Path("config/mypy-error-baseline.txt")
_ERROR_RE = re.compile(r"^(?P<path>.+?):\d+(?::\d+)?: error:")
_ACCEPTED_MYPY_EXITS = {0, 1}


def parse_error_counts(output: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in output.splitlines():
        match = _ERROR_RE.match(raw_line.strip())
        if match is not None:
            counts[match.group("path").removeprefix("./")] += 1
    return counts


def parse_baseline(text: str) -> dict[str, int]:
    baseline: dict[str, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"baseline line {line_number} must be '<path> <count>'")
        path, raw_count = parts
        if path in baseline:
            raise ValueError(f"duplicate baseline path on line {line_number}: {path}")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(
                f"baseline line {line_number} has a non-integer count"
            ) from error
        if count <= 0:
            raise ValueError(f"baseline line {line_number} count must be positive")
        baseline[path] = count
    return baseline


def compare_budget(
    baseline: Mapping[str, int], current: Mapping[str, int]
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    growth: dict[str, tuple[int, int]] = {}
    reductions: dict[str, tuple[int, int]] = {}
    for path in sorted(set(baseline) | set(current)):
        allowed = baseline.get(path, 0)
        actual = current.get(path, 0)
        if actual > allowed:
            growth[path] = (allowed, actual)
        elif actual < allowed:
            reductions[path] = (allowed, actual)
    return growth, reductions


def render_baseline(counts: Mapping[str, int]) -> str:
    lines = [
        "# Grandfathered mypy error counts by file.",
        "# This is a shrink-only ratchet: new files/count growth fail CI.",
        "# Tighten after fixes: python scripts/check_mypy_error_budget.py --update",
        "# Environment: Python 3.11 + config/mypy-requirements.txt",
        "",
    ]
    lines.extend(
        f"{path} {counts[path]}" for path in sorted(counts) if counts[path] > 0
    )
    return "\n".join(lines) + "\n"


def run_mypy(command: Sequence[str]) -> Counter[str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode not in _ACCEPTED_MYPY_EXITS:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"mypy failed operationally with exit {result.returncode}")
    counts = parse_error_counts(result.stdout)
    if result.returncode == 1 and not counts:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("mypy reported failure but produced no parseable errors")
    return counts


def _mypy_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "mypy",
        "vllm_mlx/",
        "videox_fun_mlx/",
        "--ignore-missing-imports",
        "--no-error-summary",
        "--show-error-codes",
        "--no-pretty",
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--update", action="store_true", help="record reductions only")
    parser.add_argument(
        "--init", action="store_true", help="create the initial baseline"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.update and args.init:
        parser.error("--update and --init are mutually exclusive")

    try:
        current = run_mypy(_mypy_command())
    except RuntimeError as error:
        print(f"mypy budget could not produce a trustworthy result: {error}")
        return 2

    if args.init:
        if args.baseline.exists():
            print(f"refusing to replace existing baseline: {args.baseline}")
            return 2
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(render_baseline(current))
        print(f"initialized {args.baseline} with {sum(current.values())} errors")
        return 0

    try:
        baseline = parse_baseline(args.baseline.read_text())
    except (OSError, ValueError) as error:
        print(f"invalid mypy baseline: {error}")
        return 2

    growth, reductions = compare_budget(baseline, current)
    if growth:
        print("mypy error budget increased:")
        for path, (allowed, actual) in growth.items():
            label = "new dirty file" if allowed == 0 else "count growth"
            print(f"  - {path}: {allowed} -> {actual} ({label})")
        print("Fix the errors; --update never accepts growth or new dirty files.")
        return 1

    if reductions:
        if args.update:
            args.baseline.write_text(render_baseline(current))
            print(
                f"tightened {args.baseline}: "
                f"{sum(baseline.values())} -> {sum(current.values())} errors"
            )
            return 0
        print("mypy debt fell; tighten the baseline so it cannot return:")
        for path, (allowed, actual) in reductions.items():
            print(f"  - {path}: {allowed} -> {actual}")
        print("Run: python scripts/check_mypy_error_budget.py --update")
        return 1

    print(
        f"mypy error budget OK: {sum(current.values())} grandfathered errors "
        f"across {len(current)} files; no growth"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
