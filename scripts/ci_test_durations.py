#!/usr/bin/env python3
"""Regenerate ordinary-suite shard weights from timestamped verbose logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULT = re.compile(
    r"^(\d{4}-\d\d-\d\dT\S+Z)\s+(tests/\S+?\.py)::.*\s"
    r"(?:PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)(?:\s|$)"
)


def durations_from_logs(paths: list[Path]) -> dict[str, int]:
    totals: dict[str, float] = defaultdict(float)
    for path in paths:
        previous = None
        count = 0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "test session starts" in line:
                previous = None
            match = RESULT.match(line)
            if not match:
                continue
            stamp, filename = match.groups()
            current = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if previous is not None and current < previous:
                raise ValueError(f"nonmonotonic result timestamps in {path}")
            if not filename.startswith(("tests/headless_mlx/", "tests/integrations/")):
                totals[filename] += (
                    (current - previous).total_seconds() * 1000
                    if previous is not None
                    else 0
                )
                count += 1
            previous = current
        if not count:
            raise ValueError(f"no ordinary verbose results in {path}")
    return {name: max(1, round(value)) for name, value in sorted(totals.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-log", type=Path, nargs="+", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("config/ci-test-durations.json")
    )
    args = parser.parse_args()
    payload = {
        "source": args.source,
        "unit": "milliseconds",
        "description": (
            "Python 3.12 ordinary-suite weights from timestamp deltas between "
            "consecutive verbose test results, summed by file. Includes fixture "
            "and log overhead; not pytest call-only timings. The first result "
            "in each session has no preceding delta. Unknown files retain a "
            "one-millisecond-per-line estimate. Regenerate after substantial "
            "suite changes with scripts/ci_test_durations.py --from-log LOG... "
            "--source RUN_URL."
        ),
        "files": durations_from_logs(args.from_log),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
