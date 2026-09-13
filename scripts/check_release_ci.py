#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wait for exact-source required CI evidence before a production release.

The version-bump push starts ``ci.yml``, ``rapid-mac-ci.yml`` and
``auto-release.yml`` independently.  A successful release-specific model or
packaging gate therefore says nothing about the two ordinary required-check
facades on that final commit.  This helper joins those workflows by immutable
source SHA and refuses publication until their aggregate jobs succeed.

Cancelled runs are not evidence.  They are ignored when an earlier or later
non-cancelled run for the same workflow/SHA exists; if cancellation is the only
terminal result, the gate fails instead of leaving publication pending forever.
All API, identity and schema errors fail closed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


class ReleaseCIGateError(RuntimeError):
    """The exact release CI evidence is missing, malformed, or red."""


@dataclass(frozen=True)
class RequiredWorkflow:
    workflow: str
    aggregate_job: str


@dataclass(frozen=True)
class WorkflowRun:
    run_id: int
    status: str
    conclusion: str | None
    url: str


def _validate_sha(value: str) -> None:
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ReleaseCIGateError(
            f"source SHA must be 40 lowercase hexadecimal characters; got {value!r}"
        )


def _run_gh(gh: str, repo: str, *args: str) -> str:
    env = dict(os.environ)
    env["GH_REPO"] = repo
    try:
        result = subprocess.run(
            [gh, *args],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseCIGateError(f"cannot execute gh: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise ReleaseCIGateError(
            f"gh {' '.join(args)} failed ({result.returncode}): {detail}"
        )
    return result.stdout


def _json_array(raw: str, *, source: str) -> list[dict]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReleaseCIGateError(f"{source} returned malformed JSON: {exc}") from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ReleaseCIGateError(f"{source} did not return an array of objects")
    return payload


def _workflow_runs(
    gh: str, repo: str, requirement: RequiredWorkflow, source_sha: str
) -> list[WorkflowRun]:
    raw = _run_gh(
        gh,
        repo,
        "api",
        f"repos/{repo}/actions/workflows/{requirement.workflow}/runs",
        "-X",
        "GET",
        "-f",
        f"head_sha={source_sha}",
        "-f",
        "event=push",
        "-f",
        "per_page=30",
        "--jq",
        ".workflow_runs",
    )
    records = _json_array(raw, source=f"{requirement.workflow} runs API")
    runs: list[WorkflowRun] = []
    for record in records:
        run_id = record.get("id")
        head_sha = record.get("head_sha")
        event = record.get("event")
        status = record.get("status")
        conclusion = record.get("conclusion")
        url = record.get("html_url")
        if (
            type(run_id) is not int
            or not isinstance(head_sha, str)
            or not isinstance(event, str)
            or not isinstance(status, str)
            or (conclusion is not None and not isinstance(conclusion, str))
            or not isinstance(url, str)
        ):
            raise ReleaseCIGateError(
                f"{requirement.workflow} runs API returned a malformed record"
            )
        if status == "completed" and conclusion is None:
            raise ReleaseCIGateError(
                f"{requirement.workflow} runs API returned a completed run "
                "without a conclusion"
            )
        # Defend against an API/filter regression rather than trusting query
        # parameters for the release identity boundary.
        if head_sha == source_sha and event == "push":
            runs.append(WorkflowRun(run_id, status, conclusion, url))
    return sorted(runs, key=lambda run: run.run_id, reverse=True)


def _aggregate_conclusion(
    gh: str, repo: str, run: WorkflowRun, aggregate_job: str
) -> str:
    raw = _run_gh(
        gh,
        repo,
        "api",
        f"repos/{repo}/actions/runs/{run.run_id}/jobs",
        "--paginate",
        "--slurp",
        "-X",
        "GET",
        "-f",
        "filter=latest",
        "-f",
        "per_page=100",
    )
    pages = _json_array(raw, source=f"run {run.run_id} jobs API pages")
    jobs: list[dict] = []
    for page in pages:
        page_jobs = page.get("jobs")
        if not isinstance(page_jobs, list) or not all(
            isinstance(job, dict) for job in page_jobs
        ):
            raise ReleaseCIGateError(
                f"run {run.run_id} jobs API returned a malformed page"
            )
        jobs.extend(page_jobs)
    matches = [job for job in jobs if job.get("name") == aggregate_job]
    if len(matches) != 1:
        raise ReleaseCIGateError(
            f"run {run.run_id} exposes {len(matches)} jobs named {aggregate_job!r}; "
            "expected exactly one required-check facade"
        )
    conclusion = matches[0].get("conclusion")
    if not isinstance(conclusion, str):
        raise ReleaseCIGateError(
            f"run {run.run_id} aggregate {aggregate_job!r} has no terminal conclusion"
        )
    return conclusion


def _evaluate(
    gh: str, repo: str, requirement: RequiredWorkflow, source_sha: str
) -> tuple[str, str, int | None]:
    runs = _workflow_runs(gh, repo, requirement, source_sha)
    if not runs:
        return (
            "wait",
            f"{requirement.workflow}: exact-SHA push run not visible yet",
            None,
        )

    relevant = [run for run in runs if run.conclusion != "cancelled"]
    if not relevant:
        newest = runs[0]
        return (
            "wait",
            f"{requirement.workflow}: cancellation-only evidence; waiting for a "
            f"replacement after run {newest.run_id} ({newest.url})",
            newest.run_id,
        )

    # GitHub can leave an older duplicate running after a newer run has
    # completed.  The newest non-cancelled run is authoritative: wait when it
    # is active, otherwise evaluate its required-check facade.  This mirrors
    # the status GitHub presents for the latest retry without allowing a newer
    # cancelled duplicate to erase valid evidence.
    newest = relevant[0]
    if newest.status != "completed":
        return (
            "wait",
            f"{requirement.workflow}: run {newest.run_id} is {newest.status} ({newest.url})",
            newest.run_id,
        )
    conclusion = _aggregate_conclusion(gh, repo, newest, requirement.aggregate_job)
    if conclusion != "success":
        raise ReleaseCIGateError(
            f"{requirement.workflow}: required aggregate {requirement.aggregate_job!r} "
            f"is {conclusion} in exact-SHA run {newest.run_id} ({newest.url})"
        )
    return (
        "success",
        f"{requirement.workflow}: {requirement.aggregate_job} passed in "
        f"run {newest.run_id} ({newest.url})",
        newest.run_id,
    )


def verify(
    *,
    source_sha: str,
    repo: str,
    requirements: tuple[RequiredWorkflow, ...],
    gh: str = "gh",
    deadline_sec: float = 5400,
    sleep_sec: float = 20,
) -> list[str]:
    _validate_sha(source_sha)
    if not repo or "/" not in repo:
        raise ReleaseCIGateError(f"invalid repository name: {repo!r}")
    if not requirements:
        raise ReleaseCIGateError("at least one required workflow is required")

    deadline = time.monotonic() + deadline_sec
    last_messages: list[str] = []
    while True:
        waiting = False
        messages: list[str] = []
        successful_run_ids: list[int | None] = []
        for requirement in requirements:
            state, message, run_id = _evaluate(gh, repo, requirement, source_sha)
            messages.append(message)
            successful_run_ids.append(run_id)
            waiting = waiting or state == "wait"
        if not waiting:
            # A retry can appear after one workflow was read but before the
            # other workflow finishes. Re-read the complete set and only
            # accept two consecutive snapshots with the same authoritative
            # run IDs. A changed/new active retry goes around the poll loop.
            confirmed: list[str] = []
            confirmed_ids: list[int | None] = []
            for requirement in requirements:
                state, message, run_id = _evaluate(gh, repo, requirement, source_sha)
                confirmed.append(message)
                confirmed_ids.append(run_id)
                waiting = waiting or state == "wait"
            if not waiting and confirmed_ids == successful_run_ids:
                return confirmed
            messages = confirmed
        if time.monotonic() >= deadline:
            raise ReleaseCIGateError(
                "timed out waiting for exact-SHA release CI:\n- "
                + "\n- ".join(messages or last_messages)
            )
        if messages != last_messages:
            print("\n".join(messages), flush=True)
            last_messages = messages
        time.sleep(sleep_sec)


def _requirement(value: str) -> RequiredWorkflow:
    workflow, separator, job = value.partition(":")
    if not separator or not workflow.endswith((".yml", ".yaml")) or not job:
        raise argparse.ArgumentTypeError("expected WORKFLOW.yml:aggregate-job")
    return RequiredWorkflow(workflow=workflow, aggregate_job=job)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--require",
        action="append",
        type=_requirement,
        required=True,
        dest="requirements",
        help="Required workflow and aggregate job: WORKFLOW.yml:job-name",
    )
    parser.add_argument("--deadline-min", type=float, default=90)
    parser.add_argument("--sleep-sec", type=float, default=20)
    parser.add_argument("--gh", default="gh")
    args = parser.parse_args(argv)
    try:
        messages = verify(
            source_sha=args.source_sha,
            repo=args.repo,
            requirements=tuple(args.requirements),
            gh=args.gh,
            deadline_sec=args.deadline_min * 60,
            sleep_sec=args.sleep_sec,
        )
    except ReleaseCIGateError as exc:
        print(f"release CI gate: {exc}", file=sys.stderr)
        return 1
    print("Exact-SHA release CI passed:\n- " + "\n- ".join(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
