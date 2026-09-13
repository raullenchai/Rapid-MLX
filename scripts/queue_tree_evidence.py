#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create and verify trusted CI evidence for an identical Git tree.

Mergify validates a synthetic merge commit, then writes the same tree to
``main`` with a different commit SHA.  This helper lets the main-push workflow
reuse that expensive evidence without trusting branch names, mutable refs, or
candidate-authored artifacts.  The evidence producer is a privileged
``workflow_run`` workflow whose definition comes from the default branch and
which never executes candidate code.

Every mismatch is a cache miss, not a pass: callers fall back to ordinary CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMAS = {
    "mac": "rapid-mlx/queue-tree-evidence/mac/v1",
    "ci": "rapid-mlx/queue-tree-evidence/ci/v1",
}
CONTEXTS = {scope: f"queue-tree-evidence/{scope}" for scope in SCHEMAS}
MAC_CONTEXT = CONTEXTS["mac"]
ATTESTATION_WORKFLOW = ".github/workflows/queue-tree-attestation.yml"
MAC_WORKFLOW = "rapid-mac-ci.yml"
MAC_WORKFLOW_PATH = f".github/workflows/{MAC_WORKFLOW}"
CI_WORKFLOW = "ci.yml"
CI_WORKFLOW_PATH = f".github/workflows/{CI_WORKFLOW}"
CANDIDATE_RE = re.compile(r"^mergify/merge-queue/[0-9a-f]{10}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_URL_RE = re.compile(
    r"^https://github\.com/(?P<repo>[^/]+/[^/]+)/actions/runs/(?P<run_id>[1-9][0-9]*)$"
)
MAC_CONTROL_PATHS = (
    ATTESTATION_WORKFLOW,
    MAC_WORKFLOW_PATH,
    "scripts/queue_tree_evidence.py",
    "scripts/select_gui_flows.py",
    "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml",
)
CI_CONTROL_PATHS = (
    ATTESTATION_WORKFLOW,
    CI_WORKFLOW_PATH,
    "scripts/queue_tree_evidence.py",
    "scripts/classify_ci_changes.py",
)
WORKFLOWS = {"mac": MAC_WORKFLOW, "ci": CI_WORKFLOW}
WORKFLOW_PATHS = {"mac": MAC_WORKFLOW_PATH, "ci": CI_WORKFLOW_PATH}
CONTROL_PATHS = {"mac": MAC_CONTROL_PATHS, "ci": CI_CONTROL_PATHS}
REQUIRED_MAC_JOBS = (
    "changes",
    "accessibility-identifiers",
    "accessibility-identifier-tests",
    "gui-harness-contracts",
    "build",
    "gui-app-build",
    "desktop-tests",
)
REQUIRED_CI_JOBS = (
    "changes",
    "lint",
    "merge-lane-mac",
    "engine-contracts",
    "mlx-bound-guard",
    "type-check",
    "test-apple-silicon",
    "linux-coverage",
    "changed-lines-coverage",
    "tests",
)
REQUIRED_CI_MATRIX_PREFIXES = {
    "test-matrix (": 9,
    "l1-smoke (": 5,
}


class EvidenceError(RuntimeError):
    """Evidence is missing, malformed, stale, or not authoritative."""


class GitHubClient:
    """Small strict wrapper around ``gh api`` for local and Actions use."""

    def __init__(self, repo: str, gh: str = "gh") -> None:
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
            raise EvidenceError(f"invalid repository name: {repo!r}")
        self.repo = repo
        self.gh = gh

    def json(self, endpoint: str, *fields: str, paginate: bool = False) -> Any:
        command = [self.gh, "api", endpoint, "-X", "GET"]
        for field in fields:
            command.extend(("-f", field))
        if paginate:
            command.extend(("--paginate", "--slurp"))
        env = dict(os.environ)
        env["GH_REPO"] = self.repo
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=12,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvidenceError(f"cannot query GitHub: {exc}") from exc
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise EvidenceError(f"{' '.join(command)} failed: {detail}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"GitHub returned malformed JSON: {exc}") from exc

    def commit(self, sha: str) -> dict[str, Any]:
        _require_sha(sha)
        value = self.json(f"repos/{self.repo}/git/commits/{sha}")
        if not isinstance(value, dict):
            raise EvidenceError("commit API returned a non-object")
        return value

    def jobs(self, run_id: int) -> list[dict[str, Any]]:
        pages = self.json(
            f"repos/{self.repo}/actions/runs/{run_id}/jobs",
            "filter=all",
            "per_page=100",
            paginate=True,
        )
        if not isinstance(pages, list):
            raise EvidenceError("jobs API returned malformed pages")
        jobs: list[dict[str, Any]] = []
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("jobs"), list):
                raise EvidenceError("jobs API returned a malformed page")
            if not all(isinstance(item, dict) for item in page["jobs"]):
                raise EvidenceError("jobs API returned a malformed job")
            jobs.extend(page["jobs"])
        return jobs


def _require_sha(value: str) -> None:
    if not SHA_RE.fullmatch(value):
        raise EvidenceError(f"expected a lowercase 40-character SHA, got {value!r}")


def _field(record: dict[str, Any], name: str, kind: type) -> Any:
    value = record.get(name)
    if type(value) is not kind:
        raise EvidenceError(f"GitHub record has invalid {name!r}")
    return value


def _tree_sha(client: GitHubClient, sha: str) -> str:
    commit = client.commit(sha)
    tree = commit.get("tree")
    if not isinstance(tree, dict):
        raise EvidenceError("commit API omitted tree identity")
    value = tree.get("sha")
    if not isinstance(value, str):
        raise EvidenceError("commit API returned malformed tree identity")
    _require_sha(value)
    return value


def _path_blob(client: GitHubClient, ref: str, path: str) -> str:
    record = client.json(f"repos/{client.repo}/contents/{path}", f"ref={ref}")
    if not isinstance(record, dict) or record.get("type") != "file":
        raise EvidenceError(f"cannot resolve control file {path!r} at {ref}")
    sha = record.get("sha")
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        raise EvidenceError(f"control file {path!r} has no blob identity")
    return sha


def _workflow_runs(
    client: GitHubClient, workflow: str, sha: str
) -> list[dict[str, Any]]:
    pages = client.json(
        f"repos/{client.repo}/actions/workflows/{workflow}/runs",
        f"head_sha={sha}",
        "event=pull_request",
        "per_page=100",
        paginate=True,
    )
    if not isinstance(pages, list):
        raise EvidenceError("workflow runs API returned malformed pages")
    records: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict) or not isinstance(
            page.get("workflow_runs"), list
        ):
            raise EvidenceError("workflow runs API returned a malformed page")
        records.extend(page["workflow_runs"])
    return records


def _latest_authoritative_run(
    client: GitHubClient, workflow: str, sha: str, branch: str
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for run in _workflow_runs(client, workflow, sha):
        if not isinstance(run, dict):
            raise EvidenceError("workflow runs API returned a malformed run")
        if (
            run.get("head_sha") == sha
            and run.get("head_branch") == branch
            and run.get("event") == "pull_request"
            and run.get("conclusion") != "cancelled"
        ):
            candidates.append(run)
    if not candidates:
        raise EvidenceError(f"{workflow}: no non-cancelled candidate run")
    latest = max(
        candidates,
        key=lambda run: (_field(run, "id", int), _field(run, "run_attempt", int)),
    )
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise EvidenceError(
            f"{workflow}: latest authoritative run {latest['id']} concluded "
            f"{latest.get('conclusion')!r} with status {latest.get('status')!r}"
        )
    return latest


def _successful_jobs(client: GitHubClient, run: dict[str, Any]) -> list[dict[str, Any]]:
    attempt = _field(run, "run_attempt", int)
    selected = [
        job
        for job in client.jobs(_field(run, "id", int))
        if job.get("run_attempt") == attempt
    ]
    if not selected:
        raise EvidenceError(f"run {run['id']} has no jobs for attempt {attempt}")
    return selected


def _require_unique_success(jobs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [job for job in jobs if job.get("name") == name]
    if len(matches) != 1:
        raise EvidenceError(f"expected one job named {name!r}; found {len(matches)}")
    if (
        matches[0].get("status") != "completed"
        or matches[0].get("conclusion") != "success"
    ):
        raise EvidenceError(f"required job {name!r} did not succeed")
    return matches[0]


def _manifest_groups(path: Path) -> set[str]:
    try:
        source = path.read_text()
    except (OSError, UnicodeError) as exc:
        raise EvidenceError(f"cannot read GUI journey manifest: {exc}") from exc
    groups = set(re.findall(r"(?m)^    group: ([a-z][a-z-]*)\s*$", source))
    if not groups:
        raise EvidenceError("GUI journey manifest has no groups")
    return groups


def _validate_mac_jobs(
    client: GitHubClient, run: dict[str, Any], expected_groups: set[str]
) -> list[dict[str, Any]]:
    jobs = _successful_jobs(client, run)
    selected = [_require_unique_success(jobs, name) for name in REQUIRED_MAC_JOBS]
    matrix: dict[str, list[dict[str, Any]]] = {group: [] for group in expected_groups}
    for job in jobs:
        name = job.get("name")
        if not isinstance(name, str) or not name.startswith("gui-golden-flows ("):
            continue
        for group in expected_groups:
            if name.startswith(f"gui-golden-flows ({group},"):
                matrix[group].append(job)
    missing = sorted(group for group, matches in matrix.items() if len(matches) != 1)
    if missing:
        raise EvidenceError(
            "GUI candidate did not expose exactly one matrix job for groups: "
            + ", ".join(missing)
        )
    for group, (job,) in matrix.items():
        if job.get("status") != "completed" or job.get("conclusion") != "success":
            raise EvidenceError(f"GUI matrix group {group!r} did not succeed")
        selected.append(job)
    return selected


def _validate_ci_jobs(
    client: GitHubClient, run: dict[str, Any]
) -> list[dict[str, Any]]:
    jobs = _successful_jobs(client, run)
    selected = [_require_unique_success(jobs, name) for name in REQUIRED_CI_JOBS]
    for prefix, expected_count in REQUIRED_CI_MATRIX_PREFIXES.items():
        matches = [job for job in jobs if str(job.get("name", "")).startswith(prefix)]
        if len(matches) != expected_count:
            raise EvidenceError(
                f"expected {expected_count} successful jobs beginning {prefix!r}; "
                f"found {len(matches)}"
            )
        if any(
            job.get("status") != "completed" or job.get("conclusion") != "success"
            for job in matches
        ):
            raise EvidenceError(f"CI matrix {prefix!r} did not fully succeed")
        selected.extend(matches)
    return selected


def _run_summary(run: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": _field(run, "id", int),
        "attempt": _field(run, "run_attempt", int),
        "path": _field(run, "path", str),
        "url": _field(run, "html_url", str),
        "jobs": [
            {
                "id": _field(job, "id", int),
                "name": _field(job, "name", str),
                "conclusion": _field(job, "conclusion", str),
            }
            for job in sorted(jobs, key=lambda item: _field(item, "name", str))
        ],
    }


def create_evidence(
    client: GitHubClient,
    scope: str,
    source_run_id: int,
    attestation_run_id: int,
    trusted_ref: str,
    manifest: Path,
) -> dict[str, Any]:
    if scope not in SCHEMAS:
        raise EvidenceError(f"unknown evidence scope: {scope!r}")
    _require_sha(trusted_ref)
    source = client.json(f"repos/{client.repo}/actions/runs/{source_run_id}")
    if not isinstance(source, dict):
        raise EvidenceError("source workflow run is malformed")
    sha = _field(source, "head_sha", str)
    branch = _field(source, "head_branch", str)
    _require_sha(sha)
    if not CANDIDATE_RE.fullmatch(branch):
        raise EvidenceError("source is not an exact Mergify candidate branch")
    if source.get("event") != "pull_request" or source.get("conclusion") != "success":
        raise EvidenceError("source candidate run is not a successful pull_request run")
    if source.get("path") != WORKFLOW_PATHS[scope]:
        raise EvidenceError(f"source run is not the attested {scope} workflow")
    head_repo = source.get("head_repository")
    if not isinstance(head_repo, dict) or head_repo.get("full_name") != client.repo:
        raise EvidenceError("source candidate did not run from this repository")

    pulls = client.json(
        f"repos/{client.repo}/pulls",
        "state=all",
        f"head={client.repo.split('/', 1)[0]}:{branch}",
        "per_page=10",
    )
    if not isinstance(pulls, list):
        raise EvidenceError("pull request API returned malformed data")
    matches = [
        pull
        for pull in pulls
        if isinstance(pull, dict)
        and pull.get("user", {}).get("login") == "mergify[bot]"
        and pull.get("head", {}).get("sha") == sha
        and pull.get("head", {}).get("ref") == branch
        and pull.get("head", {}).get("repo", {}).get("full_name") == client.repo
        and pull.get("base", {}).get("ref") == "main"
        and pull.get("base", {}).get("repo", {}).get("full_name") == client.repo
    ]
    if len(matches) != 1:
        raise EvidenceError(
            "candidate identity did not resolve to one trusted queue PR"
        )

    control_paths = CONTROL_PATHS[scope]
    changed_controls = [
        path
        for path in control_paths
        if _path_blob(client, sha, path) != _path_blob(client, trusted_ref, path)
    ]
    if changed_controls:
        raise EvidenceError(
            f"candidate changes its own {scope} evidence controls: "
            + ", ".join(changed_controls)
        )

    source_run = _latest_authoritative_run(client, WORKFLOWS[scope], sha, branch)
    source_jobs = (
        _validate_mac_jobs(client, source_run, _manifest_groups(manifest))
        if scope == "mac"
        else _validate_ci_jobs(client, source_run)
    )
    return {
        "schema": SCHEMAS[scope],
        "scope": scope,
        "repository": client.repo,
        "candidate_sha": sha,
        "candidate_ref": branch,
        "candidate_tree": _tree_sha(client, sha),
        "candidate_pr": _field(matches[0], "number", int),
        "attestation_run_id": attestation_run_id,
        "controls": {path: _path_blob(client, sha, path) for path in control_paths},
        "source": _run_summary(source_run, source_jobs),
    }


@dataclass(frozen=True)
class Discovery:
    scope: str
    candidate_sha: str
    candidate_tree: str
    attestation_run_id: int
    artifact_name: str
    source_run_id: int


@dataclass(frozen=True)
class DiscoveryProbe:
    evidence: Discovery | None
    retryable: bool


def _recent_runs(client: GitHubClient, scope: str) -> list[dict[str, Any]]:
    if scope not in SCHEMAS:
        raise EvidenceError(f"unknown evidence scope: {scope!r}")
    page = client.json(
        f"repos/{client.repo}/actions/workflows/{WORKFLOWS[scope]}/runs",
        "event=pull_request",
        "per_page=100",
    )
    if not isinstance(page, dict) or not isinstance(page.get("workflow_runs"), list):
        raise EvidenceError(f"recent {scope} runs API returned a malformed page")
    return [run for run in page["workflow_runs"] if isinstance(run, dict)]


def discover(client: GitHubClient, scope: str, main_sha: str) -> DiscoveryProbe:
    main_tree = _tree_sha(client, main_sha)
    candidates = sorted(
        (
            run
            for run in _recent_runs(client, scope)
            if run.get("conclusion") != "cancelled"
            and run.get("head_sha") != main_sha
            and isinstance(run.get("head_sha"), str)
            and CANDIDATE_RE.fullmatch(str(run.get("head_branch", "")))
        ),
        key=lambda run: _field(run, "id", int),
        reverse=True,
    )[:8]
    for run in candidates:
        candidate_sha = _field(run, "head_sha", str)
        try:
            if _tree_sha(client, candidate_sha) != main_tree:
                continue
            # Runs are newest-first. A later non-cancelled failure supersedes
            # every older success for the same immutable candidate/tree.
            if run.get("conclusion") != "success":
                return DiscoveryProbe(None, False)
            status_pages = client.json(
                f"repos/{client.repo}/commits/{candidate_sha}/statuses",
                "per_page=100",
                paginate=True,
            )
            if not isinstance(status_pages, list) or not all(
                isinstance(page, list) for page in status_pages
            ):
                raise EvidenceError("commit statuses API returned malformed pages")
            statuses = sorted(
                (status for page in status_pages for status in page),
                key=lambda status: (
                    status.get("id", 0)
                    if isinstance(status, dict) and type(status.get("id")) is int
                    else 0
                ),
                reverse=True,
            )
            matching_statuses = [
                status
                for status in statuses
                if isinstance(status, dict) and status.get("context") == CONTEXTS[scope]
            ]
            if matching_statuses:
                # A commit-status context is a mutable decision stream. Only
                # its newest record is authoritative: accepting an older
                # success after a newer failure would resurrect revoked
                # evidence. A pending record may still settle during this
                # main-push job's bounded discovery window.
                status = matching_statuses[0]
                state = status.get("state")
                if state != "success":
                    return DiscoveryProbe(None, state == "pending")
                target = status.get("target_url")
                if not isinstance(target, str) or not (
                    match := RUN_URL_RE.fullmatch(target)
                ):
                    return DiscoveryProbe(None, False)
                if match.group("repo") != client.repo:
                    return DiscoveryProbe(None, False)
                run_id = int(match.group("run_id"))
                attestation = client.json(f"repos/{client.repo}/actions/runs/{run_id}")
                if not isinstance(attestation, dict):
                    return DiscoveryProbe(None, False)
                if (
                    attestation.get("path") != ATTESTATION_WORKFLOW
                    or attestation.get("event") != "workflow_run"
                    or attestation.get("status") != "completed"
                    or attestation.get("conclusion") != "success"
                    or attestation.get("repository", {}).get("full_name") != client.repo
                ):
                    return DiscoveryProbe(None, False)
                return DiscoveryProbe(
                    Discovery(
                        scope,
                        candidate_sha,
                        main_tree,
                        run_id,
                        f"queue-tree-evidence-{scope}-{candidate_sha}",
                        _field(run, "id", int),
                    ),
                    False,
                )
            # The newest successful run for this identical tree is the only
            # relevant candidate. Its attestation may still be starting; do
            # not burn API quota walking unrelated historical runs each poll.
            return DiscoveryProbe(None, True)
        except EvidenceError as exc:
            print(
                f"warning: ignoring candidate {candidate_sha}: {exc}", file=sys.stderr
            )
    return DiscoveryProbe(None, False)


def validate_evidence(
    client: GitHubClient,
    main_sha: str,
    discovery: Discovery,
    evidence: dict[str, Any],
    manifest: Path,
) -> None:
    if set(evidence) != {
        "schema",
        "scope",
        "repository",
        "candidate_sha",
        "candidate_ref",
        "candidate_tree",
        "candidate_pr",
        "attestation_run_id",
        "controls",
        "source",
    }:
        raise EvidenceError("evidence has unexpected or missing top-level fields")
    if (
        evidence.get("schema") != SCHEMAS[discovery.scope]
        or evidence.get("scope") != discovery.scope
        or evidence.get("repository") != client.repo
        or evidence.get("candidate_sha") != discovery.candidate_sha
        or evidence.get("candidate_tree") != discovery.candidate_tree
        or evidence.get("attestation_run_id") != discovery.attestation_run_id
        or _tree_sha(client, main_sha) != discovery.candidate_tree
    ):
        raise EvidenceError("evidence identity does not match this main tree")
    branch = evidence.get("candidate_ref")
    if not isinstance(branch, str) or not CANDIDATE_RE.fullmatch(branch):
        raise EvidenceError("evidence contains an invalid candidate branch")

    control_paths = CONTROL_PATHS[discovery.scope]
    controls = evidence.get("controls")
    if not isinstance(controls, dict) or set(controls) != set(control_paths):
        raise EvidenceError(
            f"evidence contains malformed {discovery.scope} control identities"
        )
    for path in control_paths:
        if controls[path] != _path_blob(client, main_sha, path):
            raise EvidenceError(
                f"{discovery.scope} control file changed after attestation: {path}"
            )

    summary = evidence.get("source")
    if not isinstance(summary, dict):
        raise EvidenceError("evidence source run is malformed")
    run_id = summary.get("id")
    if type(run_id) is not int:
        raise EvidenceError("evidence source has invalid run id")
    if run_id != discovery.source_run_id:
        raise EvidenceError(
            f"attested {discovery.scope} run is not the latest candidate run"
        )
    run = client.json(f"repos/{client.repo}/actions/runs/{run_id}")
    if not isinstance(run, dict):
        raise EvidenceError(f"source run {run_id} is malformed")
    if (
        run.get("path") != WORKFLOW_PATHS[discovery.scope]
        or run.get("head_sha") != discovery.candidate_sha
        or run.get("head_branch") != branch
        or run.get("event") != "pull_request"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("run_attempt") != summary.get("attempt")
        or run.get("html_url") != summary.get("url")
        or run.get("head_repository", {}).get("full_name") != client.repo
    ):
        raise EvidenceError(f"source run {run_id} no longer matches evidence")
    current_jobs = (
        _validate_mac_jobs(client, run, _manifest_groups(manifest))
        if discovery.scope == "mac"
        else _validate_ci_jobs(client, run)
    )
    recorded_jobs = summary.get("jobs")
    if not isinstance(recorded_jobs, list):
        raise EvidenceError(f"source run {run_id} has malformed recorded jobs")
    for job in recorded_jobs:
        if not isinstance(job, dict) or set(job) != {"id", "name", "conclusion"}:
            raise EvidenceError(f"source run {run_id} has malformed recorded job")
        _field(job, "id", int)
        _field(job, "name", str)
        _field(job, "conclusion", str)
    current = sorted(
        (job.get("id"), job.get("name"), job.get("conclusion")) for job in current_jobs
    )
    recorded = sorted(
        (job.get("id"), job.get("name"), job.get("conclusion"))
        for job in recorded_jobs
        if isinstance(job, dict)
    )
    if current != recorded:
        raise EvidenceError(f"source run {run_id} jobs changed after attestation")


def _write_output(**values: str | int | bool) -> None:
    destination = os.environ.get("GITHUB_OUTPUT")
    lines = [
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in values.items()
    ]
    if destination:
        with open(destination, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--scope", choices=sorted(SCHEMAS), required=True)
    create.add_argument("--source-run-id", type=int, required=True)
    create.add_argument("--attestation-run-id", type=int, required=True)
    create.add_argument("--trusted-ref", required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)

    find = subparsers.add_parser("discover")
    find.add_argument("--scope", choices=sorted(SCHEMAS), required=True)
    find.add_argument("--main-sha", required=True)
    find.add_argument("--wait-seconds", type=int, default=0)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--scope", choices=sorted(SCHEMAS), required=True)
    validate.add_argument("--main-sha", required=True)
    validate.add_argument("--candidate-sha", required=True)
    validate.add_argument("--candidate-tree", required=True)
    validate.add_argument("--attestation-run-id", type=int, required=True)
    validate.add_argument("--source-run-id", type=int, required=True)
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.repo:
        raise SystemExit("--repo or GITHUB_REPOSITORY is required")
    client = GitHubClient(args.repo)
    try:
        if args.command == "create":
            evidence = create_evidence(
                client,
                args.scope,
                args.source_run_id,
                args.attestation_run_id,
                args.trusted_ref,
                args.manifest,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n"
            )
            _write_output(
                created=True,
                scope=args.scope,
                candidate_sha=evidence["candidate_sha"],
                artifact_name=f"queue-tree-evidence-{args.scope}-{evidence['candidate_sha']}",
            )
            return 0
        if args.command == "discover":
            _require_sha(args.main_sha)
            deadline = time.monotonic() + args.wait_seconds
            while True:
                probe = discover(client, args.scope, args.main_sha)
                found = probe.evidence
                if found is not None:
                    _write_output(
                        found=True,
                        candidate_sha=found.candidate_sha,
                        candidate_tree=found.candidate_tree,
                        attestation_run_id=found.attestation_run_id,
                        artifact_name=found.artifact_name,
                        source_run_id=found.source_run_id,
                    )
                    return 0
                if not probe.retryable or time.monotonic() >= deadline:
                    _write_output(found=False)
                    return 0
                time.sleep(min(15, max(0, deadline - time.monotonic())))
        if args.command == "validate":
            _require_sha(args.main_sha)
            _require_sha(args.candidate_sha)
            _require_sha(args.candidate_tree)
            try:
                payload = json.loads(args.evidence.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise EvidenceError(f"cannot read evidence artifact: {exc}") from exc
            if not isinstance(payload, dict):
                raise EvidenceError("evidence artifact is not an object")
            discovery = Discovery(
                args.scope,
                args.candidate_sha,
                args.candidate_tree,
                args.attestation_run_id,
                f"queue-tree-evidence-{args.scope}-{args.candidate_sha}",
                args.source_run_id,
            )
            validate_evidence(client, args.main_sha, discovery, payload, args.manifest)
            _write_output(reuse=True)
            return 0
    except EvidenceError as exc:
        print(f"queue-tree evidence unavailable: {exc}", file=sys.stderr)
        if args.command == "create":
            _write_output(created=False)
            return 0
        _write_output(reuse=False)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
