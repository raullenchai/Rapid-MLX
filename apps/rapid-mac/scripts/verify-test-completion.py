#!/usr/bin/env python3
"""Require a completed Swift Testing schema-0 event stream, not console text."""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import TextIO


def check_completion(stream: TextIO) -> tuple[int, int]:
    """Check every run segment, including parameterized cases and merged runs.

    SwiftPM forwards this stable event schema on the supported Swift 6+
    toolchains. Test declarations are discovery metadata, not completion.
    Unknown record/event kinds are extensible under the schema contract.
    """
    running = False
    pending_discovery = False
    completed_runs = completed_tests = skipped_tests = 0
    active: Counter[tuple[str, str]] = Counter()
    for line in stream:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("malformed event JSON") from exc
        if (
            not isinstance(record, dict)
            or type(record.get("version")) is not int
            or record["version"] != 0
        ):
            raise ValueError("unsupported or missing event schema")
        if record.get("kind") == "test" and not running:
            # Merged products may discover their tests before runStarted. A
            # prior complete product cannot cover a later truncated discovery.
            pending_discovery = True
        if record.get("kind") != "event":
            continue
        event = record.get("payload")
        if not isinstance(event, dict):
            raise ValueError("invalid event payload")
        kind = event.get("kind")
        if kind == "runStarted":
            if running:
                raise ValueError("overlapping test runs")
            running = True
            pending_discovery = False
        elif kind == "runEnded":
            if not running or active:
                raise ValueError(
                    "test run ended with unfinished tests or without starting"
                )
            running = False
            completed_runs += 1
        elif kind in ("testStarted", "testCaseStarted", "testEnded", "testCaseEnded"):
            test_id = event.get("testID")
            if not running or not isinstance(test_id, str) or not test_id:
                raise ValueError("test event outside a run or missing its identity")
            category = "case" if kind.startswith("testCase") else "test"
            key = (category, test_id)
            if kind.endswith("Started"):
                active[key] += 1
            else:
                if not active[key]:
                    raise ValueError("test ended without a matching start")
                active[key] -= 1
                if not active[key]:
                    del active[key]
                if category == "test":
                    completed_tests += 1
        elif kind == "testSkipped":
            if not running:
                raise ValueError("test skipped outside a run")
            skipped_tests += 1
        elif kind in ("testCancelled", "testCaseCancelled"):
            raise ValueError("test execution was cancelled")
        elif kind == "issueRecorded":
            issue = event.get("issue")
            # Expected known issues and explicitly non-failing diagnostics keep
            # their framework semantics. An unexpected failure cannot be made
            # green merely by exiting the process with status zero.
            if not isinstance(issue, dict) or (
                issue.get("isKnown") is not True and issue.get("isFailure") is not False
            ):
                raise ValueError("test stream contains an unexpected failing issue")
    if (
        running
        or pending_discovery
        or active
        or not completed_runs
        or not (completed_tests or skipped_tests)
    ):
        raise ValueError("missing complete test-run evidence")
    return completed_runs, completed_tests


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: verify-test-completion.py EVENT_STREAM", file=sys.stderr)
        return 2
    try:
        with Path(argv[0]).open(encoding="utf-8") as stream:
            runs, tests = check_completion(stream)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: incomplete Desktop test run: {exc}", file=sys.stderr)
        return 1
    print(
        f"Desktop test completion verified: {runs} run(s), {tests} test lifecycle(s) ended"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
