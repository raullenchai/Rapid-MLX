"""Trusted candidate qualification contract; never emit complete-evidence schema.

The mapped producer remains default-off until its execution workflow and queue
consumer are independently reviewed and deployed. Full repair candidates do
not require a green main prerequisite.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import zipfile
from typing import Any

from scripts import queue_tree_evidence as evidence
from scripts.ci_main_qualification import qualify_main
from scripts.ci_mapped_execution import validate_manifests
from scripts.classify_ci_changes import source_canary_tests

SCHEMA = "rapid-mlx/candidate-qualification/ci/v1"
INPUT_SCHEMA = "rapid-mlx/candidate-mapped-input/v1"
CONTROL_PATHS = (
    *evidence.CI_CONTROL_PATHS,
    "scripts/ci_main_qualification.py",
    "scripts/ci_mapped_execution.py",
    "scripts/ci_candidate_qualification.py",
    ".github/workflows/candidate-qualification.yml",
)
MAPPED_JOBS = (
    "changes",
    "lint",
    "merge-lane-mac",
    "engine-contracts",
    "mlx-bound-guard",
    "type-check",
    "candidate-canary-unit",
    "tests",
)
QUEUE_ARTIFACT_MAX_BYTES = 100_000
MERGE_SUBJECT = re.compile(r"^Merge of #(\d+)$")


def _queue_metadata(client: evidence.GitHubClient, run: dict[str, Any]) -> dict:
    run_id, attempt, sha = run["id"], run["run_attempt"], run["head_sha"]
    page = client.json(
        f"repos/{client.repo}/actions/runs/{run_id}/artifacts", "per_page=100"
    )
    artifacts = page.get("artifacts")
    count = page.get("total_count")
    if (
        type(count) is not int
        or not 0 < count < 100
        or not isinstance(artifacts, list)
        or len(artifacts) != count
    ):
        raise evidence.EvidenceError("queue metadata artifact listing is malformed")
    name = f"candidate-queue-identity-{sha}-{run_id}-{attempt}"
    matches = [item for item in artifacts if item.get("name") == name]
    if len(matches) != 1:
        raise evidence.EvidenceError(
            "missing or duplicate exact-attempt queue metadata"
        )
    artifact = matches[0]
    if (
        type(artifact.get("id")) is not int
        or artifact["id"] < 1
        or artifact.get("expired") is not False
        or artifact.get("workflow_run", {}).get("id") != run_id
        or artifact.get("workflow_run", {}).get("head_sha") != sha
        or type(artifact.get("size_in_bytes")) is not int
        or not 0 < artifact["size_in_bytes"] <= QUEUE_ARTIFACT_MAX_BYTES
    ):
        raise evidence.EvidenceError("queue metadata artifact identity is invalid")
    raw = subprocess.run(
        [
            client.gh,
            "api",
            f"repos/{client.repo}/actions/artifacts/{artifact['id']}/zip",
        ],
        capture_output=True,
        timeout=20,
        check=True,
    ).stdout
    if len(raw) > QUEUE_ARTIFACT_MAX_BYTES:
        raise evidence.EvidenceError("queue metadata artifact exceeds bound")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (
            len(members) != 1
            or members[0].filename != "candidate-queue-identity.json"
            or members[0].file_size > QUEUE_ARTIFACT_MAX_BYTES
        ):
            raise evidence.EvidenceError("unexpected queue metadata archive contents")
        value = json.loads(archive.read(members[0]))
    if not isinstance(value, dict):
        raise evidence.EvidenceError("queue metadata is not an object")
    return value


def _source_identity(
    client: evidence.GitHubClient, run: dict[str, Any], base: str
) -> list[dict[str, Any]]:
    metadata = _queue_metadata(client, run)
    if metadata.get("checking_base_sha") != base:
        raise evidence.EvidenceError("queue metadata does not bind candidate base")
    sources = metadata.get("pull_requests")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 2:
        raise evidence.EvidenceError("candidate must contain one or two source PRs")
    numbers: list[int] = []
    for source in sources:
        number = source.get("number") if isinstance(source, dict) else None
        if type(number) is not int or number < 1:
            raise evidence.EvidenceError("queue metadata source PR is invalid")
        numbers.append(number)
    if len(set(numbers)) != len(numbers):
        raise evidence.EvidenceError("queue metadata source PRs are duplicated")

    current = run["head_sha"]
    newest: list[tuple[int, str]] = []
    while current != base and len(newest) <= 2:
        commit = client.commit(current)
        parents = commit.get("parents")
        lines = commit.get("message", "").splitlines()
        message = lines[0] if lines else ""
        match = MERGE_SUBJECT.fullmatch(message)
        if not isinstance(parents, list) or len(parents) != 2 or not match:
            raise evidence.EvidenceError("candidate has malformed integration lineage")
        parent, source_sha = parents[0].get("sha"), parents[1].get("sha")
        evidence._require_sha(parent)
        evidence._require_sha(source_sha)
        newest.append((int(match.group(1)), source_sha))
        current = parent
    ordered = list(reversed(newest))
    if current != base or [number for number, _ in ordered] != numbers:
        raise evidence.EvidenceError("queue metadata does not match candidate lineage")
    identities = []
    for number, source_sha in ordered:
        source = client.json(f"repos/{client.repo}/pulls/{number}")
        if (
            source.get("head", {}).get("repo", {}).get("full_name") != client.repo
            or source.get("head", {}).get("sha") != source_sha
        ):
            raise evidence.EvidenceError(
                "source PR current head does not match candidate"
            )
        identities.append({"number": number, "head_sha": source_sha})
    return identities


def _latest(client: evidence.GitHubClient, sha: str, branch: str) -> dict[str, Any]:
    runs = evidence._workflow_runs(client, evidence.CI_WORKFLOW, sha)
    selected = [
        r
        for r in runs
        if r.get("head_sha") == sha
        and r.get("head_branch") == branch
        and r.get("event") == "pull_request"
    ]
    if not selected:
        raise evidence.EvidenceError("no current candidate run")
    latest = max(
        selected,
        key=lambda r: (
            evidence._field(r, "id", int),
            evidence._field(r, "run_attempt", int),
        ),
    )
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise evidence.EvidenceError("latest candidate run is not completed green")
    return latest


def _candidate(
    client: evidence.GitHubClient, run: dict[str, Any]
) -> tuple[dict, str, str, list[dict[str, Any]]]:
    sha, branch = run["head_sha"], run["head_branch"]
    evidence._require_sha(sha)
    if (
        run.get("path") != evidence.CI_WORKFLOW_PATH
        or run.get("event") != "pull_request"
        or run.get("head_repository", {}).get("full_name") != client.repo
        or not evidence.CANDIDATE_RE.fullmatch(branch)
    ):
        raise evidence.EvidenceError("not an own-repository queue CI run")
    pulls = client.json(
        f"repos/{client.repo}/pulls",
        "state=open",
        f"head={client.repo.split('/')[0]}:{branch}",
        "per_page=100",
    )
    if not isinstance(pulls, list) or len(pulls) != 1:
        raise evidence.EvidenceError("candidate must identify one open queue PR")
    pull = pulls[0]
    if (
        pull.get("user", {}).get("login") != "mergify[bot]"
        or pull.get("head", {}).get("sha") != sha
        or pull.get("head", {}).get("ref") != branch
        or pull.get("head", {}).get("repo", {}).get("full_name") != client.repo
        or pull.get("base", {}).get("ref") != "main"
        or pull.get("base", {}).get("repo", {}).get("full_name") != client.repo
    ):
        raise evidence.EvidenceError("untrusted queue PR identity")
    base = pull["base"]["sha"]
    evidence._require_sha(base)
    commit = client.commit(sha)
    tree = commit["tree"]["sha"]
    evidence._require_sha(tree)
    sources = _source_identity(client, run, base)
    return pull, base, tree, sources


def _paths(client: evidence.GitHubClient, base: str, sha: str) -> set[str]:
    compare = client.json(f"repos/{client.repo}/compare/{base}...{sha}")
    files = compare["files"]
    if (
        compare.get("merge_base_commit", {}).get("sha") != base
        or not isinstance(files, list)
        or not 0 < len(files) < 300
    ):
        raise evidence.EvidenceError(
            "missing/truncated combined diff or wrong merge base"
        )
    paths = set()
    for file in files:
        paths.add(evidence._field(file, "filename", str))
        if "previous_filename" in file:
            paths.add(evidence._field(file, "previous_filename", str))
    return paths


def qualify_candidate(
    client: evidence.GitHubClient,
    source_run_id: int,
    trusted_ref: str,
    *,
    mapped_enabled: bool = False,
    mapped_input: dict | None = None,
) -> dict:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": client.repo,
        "qualified": False,
        "authorizes_reduced_ci": False,
    }
    try:
        evidence._require_sha(trusted_ref)
        run = client.json(f"repos/{client.repo}/actions/runs/{source_run_id}")
        pull, base, tree, sources = _candidate(client, run)
        sha, branch = run["head_sha"], run["head_branch"]
        latest = _latest(client, sha, branch)
        if (latest["id"], latest["run_attempt"]) != (source_run_id, run["run_attempt"]):
            raise evidence.EvidenceError("trigger is not the latest candidate attempt")
        result.update(
            candidate_sha=sha,
            candidate_tree=tree,
            candidate_pr=pull["number"],
            base_sha=base,
            checking_base_sha=base,
            source_pull_requests=sources,
            source_run_id=source_run_id,
            source_attempt=run["run_attempt"],
        )
        try:
            evidence._validate_ci_jobs(client, run)
            kind = "full"
        except evidence.EvidenceError:
            if mapped_enabled is not True:
                raise evidence.EvidenceError(
                    "incomplete full candidate; mapped producer is disabled"
                )
            tests = source_canary_tests(_paths(client, base, sha))
            if not tests:
                raise evidence.EvidenceError(
                    "combined diff is critical, mixed or unmapped"
                )
            for path in CONTROL_PATHS:
                if evidence._path_blob(client, sha, path) != evidence._path_blob(
                    client, trusted_ref, path
                ):
                    raise evidence.EvidenceError(
                        "mapped candidate changes a trusted controller"
                    )
            jobs = evidence._successful_jobs(client, run)
            for name in MAPPED_JOBS:
                evidence._require_unique_success(jobs, name)
            for name in (
                "test-matrix",
                "l1-smoke",
                "test-apple-silicon",
                "linux-coverage",
                "changed-lines-coverage",
                "source-canary-unit",
            ):
                matches = [
                    j
                    for j in jobs
                    if j.get("name") == name
                    or str(j.get("name", "")).startswith(name + " (")
                ]
                if (
                    len(matches) != 1
                    or matches[0].get("status") != "completed"
                    or matches[0].get("conclusion") != "skipped"
                ):
                    raise evidence.EvidenceError(
                        "unexpected full/source job on mapped candidate"
                    )
            job = evidence._require_unique_success(jobs, "candidate-canary-unit")
            # Pinning the workflow makes these successful steps authoritative;
            # candidate-authored JSON alone can never attest coverage execution.
            for name in (
                "Run complete mapped regressions",
                "Enforce mandatory changed-line coverage",
            ):
                steps = [s for s in job.get("steps", []) if s.get("name") == name]
                if (
                    len(steps) != 1
                    or steps[0].get("conclusion") != "success"
                    or steps[0].get("status") != "completed"
                ):
                    raise evidence.EvidenceError(
                        "mapped execution/coverage step did not run"
                    )
            proof = mapped_input or {}
            if (
                proof.get("schema") != INPUT_SCHEMA
                or proof.get("scope") != "candidate-mapped-only"
                or proof.get("head") != sha
                or proof.get("base") != base
                or proof.get("source_run_id") != source_run_id
                or proof.get("source_attempt") != run["run_attempt"]
                or proof.get("tests") != list(tests)
            ):
                raise evidence.EvidenceError("mapped artifact identity/scope mismatch")
            tested_sha = proof["tested_sha"]
            evidence._require_sha(tested_sha)
            if evidence._tree_sha(client, tested_sha) != tree:
                raise evidence.EvidenceError("tested tree differs from queue candidate")
            validate_manifests(proof["baseline"], proof["executed"], list(tests))
            anchor = qualify_main(client, base)
            if not anchor["qualified"]:
                raise evidence.EvidenceError(
                    "mapped candidate lacks current full-main qualification"
                )
            result.update(mapped_tests=list(tests), base_qualification=anchor)
            kind = "mapped"
        current = _latest(client, sha, branch)
        if (current["id"], current["run_attempt"]) != (
            latest["id"],
            latest["run_attempt"],
        ):
            raise evidence.EvidenceError(
                "candidate attempt changed during qualification"
            )
        if _candidate(client, current)[1:] != (base, tree, sources):
            raise evidence.EvidenceError("queue candidate base/tree changed")
        if kind == "mapped" and not qualify_main(client, base)["qualified"]:
            raise evidence.EvidenceError("main changed before mapped qualification")
        result.update(
            qualified=True, kind=kind, authorizes_reduced_ci=(kind == "mapped")
        )
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result


def main() -> None:
    """Qualify current source; mapped requires live enrolled rollout policy."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source-run-id", required=True, type=int)
    parser.add_argument("--trusted-ref", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    from scripts.ci_candidate_rollout import ImmutableContentsClient, qualify_source

    result = qualify_source(
        ImmutableContentsClient(evidence.GitHubClient(args.repo)),
        args.source_run_id,
        args.trusted_ref,
    )
    if args.expected and result != json.loads(args.expected.read_text()):
        raise evidence.EvidenceError("qualification changed before publication")
    if args.expected and not result.get("qualified"):
        raise evidence.EvidenceError("qualification revoked before publication")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"Candidate qualification: {result.get('kind', 'rejected')}: {result.get('reason', '')}"
    )


if __name__ == "__main__":
    main()
