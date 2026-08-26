#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live release-blocker evidence, queried from GitHub at run time.

An immutable release tag may only be claimed once the live set of open
``release-blocker`` issues is either closed or explicitly waived for THIS
version by a named owner. The evidence is captured live from the GitHub API —
not from a static, self-referential SHA/READY inventory — so it reflects the
state of the repo at the moment of validation and again immediately before the
tag is claimed.

Contract (all stdlib; no regex parsing; every error path is fail-closed):

  * A live query runs via ``gh issue list --state open --label release-blocker
    --limit 1000 --json number,title,url``. A non-zero exit, an unreadable or
    malformed response, or a REST record that is actually a pull request all
    fail closed.
  * Open blocker issue numbers are compared exactly against a structured,
    version-scoped waiver file
    ``docs/development/release-blockers/waivers-<version>.json``:
        { "version": "X.Y.Z-rcN",
          "waivers": [ {"issue": 2301, "reason": "...", "by": "<owner>"} ] }
    Every open blocker number must be present, with a non-empty reason and
    owner. An open blocker without a waiver fails.
  * When ``--expected-open-ids`` is given (re-query immediately before the
    tag), the freshly queried open set must EXACTLY equal the candidate-time
    set. Any difference (an issue opened/closed/waved in between) fails closed
    as a TOCTOU change.
  * Emitted evidence binds the runtime ``--source-sha``, ``--version``, the
    waiver file, and the live open issue ids.

The waiver file carries no SHA and no status field — it only lists issue ids
that are waived for a version, so it is never self-referential and never a
timeless boolean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


class BlockerCheckError(Exception):
    """Raised on any fail-closed blocker condition."""


def _assert_commit_sha(source_sha: str) -> None:
    """Fail closed on a malformed source-sha rather than bind evidence to it."""

    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or any(ch not in "0123456789abcdef" for ch in source_sha)
    ):
        raise BlockerCheckError(
            "source-sha must be a 40-character lowercase Git commit SHA; "
            f"got {str(source_sha)!r}"
        )


def _run_gh(gh: str, repo: str) -> list[dict]:
    """Run the live open-release-blocker query; return parsed issue records.

    ``gh issue list`` honours ``GH_TOKEN`` from the environment (never passed on
    the command line). REST ``/issues`` also returns pull requests; we drop any
    record that carries a ``pull_request`` key so a PR can never be mistaken for
    a blocker issue.
    """

    cmd = [
        gh,
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--label",
        "release-blocker",
        "--limit",
        "1000",
        "--json",
        "number,title,url",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BlockerCheckError(f"failed to run gh issue list: {exc}") from exc
    if proc.returncode != 0:
        raise BlockerCheckError(
            f"gh issue list failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    try:
        records = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BlockerCheckError(f"gh issue list returned invalid JSON: {exc}") from exc
    if not isinstance(records, list):
        raise BlockerCheckError("gh issue list did not return a JSON array")
    # REST /issues also returns pull requests; a PR record is never a blocker.
    # Everything else must be a well-formed issue with an integer number — a
    # malformed or duplicate record is a query-contract violation, not something
    # to silently ignore.
    issues: list[dict] = []
    seen: set[int] = set()
    for rec in records:
        if not isinstance(rec, dict):
            raise BlockerCheckError(
                f"gh issue list returned a malformed record (not an object): {rec!r}"
            )
        if "pull_request" in rec:
            continue  # REST lists PRs alongside issues; exclude them.
        num = rec.get("number")
        if isinstance(num, bool) or not isinstance(num, int) or num <= 0:
            raise BlockerCheckError(
                f"gh issue list record has a malformed issue number: {rec!r}"
            )
        if num in seen:
            raise BlockerCheckError(
                f"gh issue list returned a duplicate open issue record #{num}"
            )
        seen.add(num)
        issues.append(rec)
    return issues


def _waivers(waivers_dir: Path, version: str) -> dict[int, dict]:
    """Read the structured waiver file for ``version``; return {issue: waiver}."""

    path = waivers_dir / f"waivers-{version}.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockerCheckError(f"cannot read waiver file {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != version:
        raise BlockerCheckError(
            f"waiver file {path} does not declare the requested version {version!r}"
        )
    waivers = data.get("waivers")
    if not isinstance(waivers, list):
        raise BlockerCheckError(f"waiver file {path} has no 'waivers' list")
    out: dict[int, dict] = {}
    for w in waivers:
        if not isinstance(w, dict):
            raise BlockerCheckError(f"waiver file {path} has a non-object waiver")
        issue = w.get("issue")
        if isinstance(issue, bool) or not isinstance(issue, int) or issue <= 0:
            raise BlockerCheckError(
                f"waiver file {path} has a waiver without an integer issue id"
            )
        reason = w.get("reason")
        by = w.get("by")
        if not isinstance(reason, str) or not reason.strip():
            raise BlockerCheckError(
                f"waiver file {path} waiver #{issue} has no 'reason'"
            )
        if not isinstance(by, str) or not by.strip():
            raise BlockerCheckError(
                f"waiver file {path} waiver #{issue} has no 'by' owner"
            )
        if issue in out:
            raise BlockerCheckError(
                f"waiver file {path} duplicates waiver for issue #{issue}"
            )
        out[issue] = w
    return out


def check_live_blockers(
    *,
    version: str,
    source_sha: str,
    gh: str,
    repo: str,
    waivers_dir: Path,
    expected_open_ids: list[int] | None,
) -> tuple[list[str], list[int]]:
    """Return (evidence_lines, sorted_open_ids) or raise BlockerCheckError."""

    _assert_commit_sha(source_sha)
    issues = _run_gh(gh, repo)
    open_ids = sorted(
        issue["number"] for issue in issues if isinstance(issue.get("number"), int)
    )

    waivers = _waivers(waivers_dir, version)

    # Every open blocker must be waived for this exact version.
    uncovered = sorted(set(open_ids) - set(waivers))
    if uncovered:
        raise BlockerCheckError(
            f"open release-blocker issues without a waiver for {version}: "
            + ", ".join(f"#{i}" for i in uncovered)
        )

    # Every waiver must correspond to a currently open blocker. A waiver whose
    # issue is no longer open is stale (real world has moved past the waiver
    # file) and must be cleaned up before a tag may be claimed.
    stale = sorted(set(waivers) - set(open_ids))
    if stale:
        raise BlockerCheckError(
            f"stale waiver(s) for {version} whose issue is not open: "
            + ", ".join(f"#{i}" for i in stale)
        )

    # TOCTOU: the pre-tag re-query must match the candidate-time open set exactly.
    if expected_open_ids is not None and open_ids != sorted(expected_open_ids):
        raise BlockerCheckError(
            "release-blocker set changed after candidate validation: "
            f"candidate-time={sorted(expected_open_ids)} now={open_ids}"
        )

    evidence = [
        f"live release blockers for {version} (source {source_sha})",
        "  open release-blocker issues: "
        + (", ".join(f"#{i}" for i in open_ids) or "<none>"),
    ]
    for issue in issues:
        num = issue.get("number")
        title = issue.get("title", "")
        if isinstance(num, int) and num in waivers:
            evidence.append(
                f"  #{num} {title} -> WAIVED by @{waivers[num]['by']}: {waivers[num]['reason']}"
            )
    evidence.append(f"  waiver file: {waivers_dir}/waivers-{version}.json")
    return evidence, open_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--waivers-dir",
        type=Path,
        default=Path("docs/development/release-blockers"),
    )
    parser.add_argument(
        "--expected-open-ids",
        default=None,
        help=(
            "comma-separated candidate-time open ids the pre-tag re-query must match "
            "exactly; pass an empty string to expect an empty set. Omit entirely to "
            "skip the TOCTOU comparison (candidate-time query)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # None (flag absent) => no TOCTOU check. "" (flag present, empty) => expect an
    # empty open set, still enforced. This keeps "candidate-time zero" meaningful:
    # if zero were open at candidate time, a changed pre-tag set must still fail.
    expected = None
    if args.expected_open_ids is not None:
        expected = [int(x) for x in args.expected_open_ids.split(",") if x.strip()]
    try:
        evidence, open_ids = check_live_blockers(
            version=args.version,
            source_sha=args.source_sha,
            gh=args.gh,
            repo=args.repo,
            waivers_dir=args.waivers_dir,
            expected_open_ids=expected,
        )
    except BlockerCheckError as exc:
        print(f"release blockers: {exc}", file=sys.stderr)
        return 1
    print("\n".join(evidence))
    print(f"OPEN_IDS={','.join(f'{i}' for i in open_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
