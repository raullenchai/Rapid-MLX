"""Audit candidate path/base eligibility without reducing or qualifying CI.

Run only from trusted default-branch code after full candidate attestation.
The output is advisory: neither main reuse nor merge protection consumes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import queue_tree_evidence as evidence
from scripts.classify_ci_changes import source_canary_tests

SCHEMA = "rapid-mlx/candidate-route-shadow/v1"
MAIN_JOBS = (
    "changes",
    "lint",
    "engine-contracts",
    "type-check",
    "test-apple-silicon",
    "linux-coverage",
    "tests",
)


def _tip(client: evidence.GitHubClient) -> str:
    value = client.json(f"repos/{client.repo}/git/ref/heads/main")
    sha = value["object"]["sha"]
    evidence._require_sha(sha)
    return sha


def _latest_main(client: evidence.GitHubClient, base: str) -> dict[str, Any]:
    page = client.json(
        f"repos/{client.repo}/actions/workflows/ci.yml/runs",
        f"head_sha={base}",
        "event=push",
        "branch=main",
        "per_page=100",
    )
    runs = page["workflow_runs"]
    if not isinstance(runs, list) or not runs:
        raise evidence.EvidenceError("no exact-base main qualification")
    # Include cancelled, pending and failed runs: an older success cannot
    # resurrect a newer non-green attempt. Re-runs keep the same run ID.
    latest = max(
        runs,
        key=lambda r: (
            evidence._field(r, "id", int),
            evidence._field(r, "run_attempt", int),
        ),
    )
    if (
        latest.get("head_sha") != base
        or latest.get("head_branch") != "main"
        or latest.get("event") != "push"
        or latest.get("path") != evidence.CI_WORKFLOW_PATH
        or latest.get("head_repository", {}).get("full_name") != client.repo
        or latest.get("status") != "completed"
        or latest.get("conclusion") != "success"
    ):
        raise evidence.EvidenceError(
            "latest exact-base main run is not completed green"
        )
    return latest


def inspect_candidate(
    client: evidence.GitHubClient, full: dict[str, Any], attestation_run_id: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": "advisory-candidate-path-and-base",
        "repository": client.repo,
        "route": "full",
        "reason": "unqualified",
        "authorizes_reduced_ci": False,
        "execution_proof_checked": False,
    }
    try:
        if (
            full.get("schema") != evidence.SCHEMAS["ci"]
            or full.get("scope") != "ci"
            or full.get("repository") != client.repo
            or full.get("attestation_run_id") != attestation_run_id
        ):
            raise evidence.EvidenceError(
                "not locally produced complete candidate CI evidence"
            )
        sha = evidence._field(full, "candidate_sha", str)
        evidence._require_sha(sha)
        result["candidate_sha"] = sha
        pull = client.json(
            f"repos/{client.repo}/pulls/{evidence._field(full, 'candidate_pr', int)}"
        )
        if (
            pull.get("user", {}).get("login") != "mergify[bot]"
            or pull.get("head", {}).get("sha") != sha
            or pull.get("head", {}).get("ref") != full.get("candidate_ref")
            or not evidence.CANDIDATE_RE.fullmatch(str(full.get("candidate_ref", "")))
            or pull.get("head", {}).get("repo", {}).get("full_name") != client.repo
            or pull.get("base", {}).get("ref") != "main"
            or pull.get("base", {}).get("repo", {}).get("full_name") != client.repo
        ):
            raise evidence.EvidenceError("candidate does not match trusted queue PR")
        base = pull["base"]["sha"]
        evidence._require_sha(base)
        result["base_sha"] = base
        commit = client.commit(sha)
        # Real queue candidates are merge commits with two parents. Their
        # first parent, not the number of parents, must bind the main base.
        if (
            not commit.get("parents")
            or commit["parents"][0].get("sha") != base
            or commit.get("tree", {}).get("sha") != full.get("candidate_tree")
        ):
            raise evidence.EvidenceError("candidate base/tree identity mismatch")
        if _tip(client) != base:
            raise evidence.EvidenceError("candidate base is not current main")
        compare = client.json(f"repos/{client.repo}/compare/{base}...{sha}")
        files = compare["files"]
        if (
            compare.get("merge_base_commit", {}).get("sha") != base
            or not isinstance(files, list)
            or not files
            or len(files) >= 300
        ):
            raise evidence.EvidenceError(
                "combined diff is empty, truncated, or has a different merge base"
            )
        paths: set[str] = set()
        for file in files:
            paths.add(evidence._field(file, "filename", str))
            if "previous_filename" in file:
                paths.add(evidence._field(file, "previous_filename", str))
        tests = source_canary_tests(paths)
        result["paths"] = sorted(paths)
        if not tests:
            raise evidence.EvidenceError(
                "combined diff includes unmapped/critical/control paths"
            )
        result["mapped_tests"] = list(tests)
        main = _latest_main(client, base)
        jobs = evidence._successful_jobs(client, main)
        for name in MAIN_JOBS:
            evidence._require_unique_success(jobs, name)
        for prefix, names in evidence.REQUIRED_CI_MATRIX_JOBS.items():
            selected = [
                job for job in jobs if str(job.get("name", "")).startswith(prefix)
            ]
            if len(selected) != len(names) or {
                job.get("name") for job in selected
            } != set(names):
                raise evidence.EvidenceError(
                    "base main lacks exact full CPU9/model5 identities"
                )
            for name in names:
                evidence._require_unique_success(selected, name)
        # This first shadow accepts actually executed full main jobs only.
        # Reused full proof needs its own authenticated revalidation before
        # real reduced routing; a skipped matrix is never counted as passed.
        newest = _latest_main(client, base)
        if (newest["id"], newest["run_attempt"]) != (
            main["id"],
            main["run_attempt"],
        ) or _tip(client) != base:
            raise evidence.EvidenceError("main qualification changed during inspection")
        result.update(
            route="mapped-eligible-shadow",
            reason="exact-base-executed-full-main",
            base_run=main["id"],
            base_attempt=main["run_attempt"],
        )
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--full-evidence", type=Path, required=True)
    parser.add_argument("--attestation-run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_candidate(
        evidence.GitHubClient(args.repo),
        json.loads(args.full_evidence.read_text()),
        args.attestation_run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Advisory only: {result['route']}: {result['reason']}")


if __name__ == "__main__":
    main()
