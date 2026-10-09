"""Revalidate exact-tip full main CI; this library does not authorize routing."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from scripts import queue_tree_evidence as evidence

MAIN_EXECUTED_JOBS = (
    "lint",
    "engine-contracts",
    "type-check",
    "test-apple-silicon",
    "linux-coverage",
)


def _tip(client: evidence.GitHubClient) -> str:
    sha = client.json(f"repos/{client.repo}/git/ref/heads/main")["object"]["sha"]
    evidence._require_sha(sha)
    return sha


def _latest(client: evidence.GitHubClient, base: str) -> dict[str, Any]:
    pages = client.json(
        f"repos/{client.repo}/actions/workflows/ci.yml/runs",
        f"head_sha={base}",
        "event=push",
        "branch=main",
        "per_page=100",
        paginate=True,
    )
    runs = [run for page in pages for run in page["workflow_runs"]]
    if not runs:
        raise evidence.EvidenceError("no exact-base main run")
    for run in runs:
        for field in ("id", "run_attempt"):
            if type(run.get(field)) is not int or run[field] < 1:
                raise evidence.EvidenceError("invalid main run identity")
    latest = max(runs, key=lambda r: (r["id"], r["run_attempt"]))
    if (
        latest.get("head_sha") != base
        or latest.get("head_branch") != "main"
        or latest.get("event") != "push"
        or latest.get("path") != evidence.CI_WORKFLOW_PATH
        or latest.get("head_repository", {}).get("full_name") != client.repo
        or latest.get("status") != "completed"
        or latest.get("conclusion") != "success"
    ):
        raise evidence.EvidenceError("latest exact-base main is not completed green")
    return latest


def _matrix(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for prefix, names in evidence.REQUIRED_CI_MATRIX_JOBS.items():
        group = [
            j
            for j in jobs
            if j.get("name") == prefix.rstrip(" (")
            or str(j.get("name", "")).startswith(prefix)
        ]
        if len(group) != len(names) or {j.get("name") for j in group} != set(names):
            raise evidence.EvidenceError("main lacks exact CPU9/model5 job identities")
        selected.extend(group)
    return selected


def _artifact(client: evidence.GitHubClient, discovery: evidence.Discovery) -> dict:
    page = client.json(
        f"repos/{client.repo}/actions/runs/{discovery.attestation_run_id}/artifacts",
        "per_page=100",
    )
    if type(page.get("total_count")) is not int or page["total_count"] >= 100:
        raise evidence.EvidenceError("artifact listing is truncated or malformed")
    matches = [a for a in page["artifacts"] if a.get("name") == discovery.artifact_name]
    if len(matches) != 1 or matches[0].get("expired") is not False:
        raise evidence.EvidenceError(
            "missing, duplicate or expired trusted full artifact"
        )
    artifact = matches[0]
    if type(artifact.get("id")) is not int or artifact["id"] < 1:
        raise evidence.EvidenceError("invalid artifact identity")
    if (
        type(artifact.get("size_in_bytes")) is not int
        or not 0 < artifact["size_in_bytes"] <= 1_000_000
    ):
        raise evidence.EvidenceError("full artifact exceeds bounded size")
    try:
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
        if len(raw) > 1_000_000:
            raise evidence.EvidenceError("downloaded artifact exceeds bounded size")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if (
                len(members) != 1
                or members[0].filename != "evidence.json"
                or members[0].file_size > 1_000_000
            ):
                raise evidence.EvidenceError(
                    "unexpected full artifact archive contents"
                )
            return json.loads(archive.read(members[0]))
    except (
        OSError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise evidence.EvidenceError("cannot read trusted full artifact") from exc


def _latest_tree_candidate(client: evidence.GitHubClient, base: str) -> dict[str, Any]:
    # The legacy discovery helper ignores cancelled candidate runs. Here a
    # newer cancellation of the same tree must not resurrect an older proof.
    tree = evidence._tree_sha(client, base)
    candidates = sorted(
        (
            r
            for r in evidence._recent_runs(client, "ci")
            if evidence.CANDIDATE_RE.fullmatch(str(r.get("head_branch", "")))
        ),
        key=lambda r: evidence._field(r, "id", int),
        reverse=True,
    )[:8]
    newest = next(
        (r for r in candidates if evidence._tree_sha(client, r["head_sha"]) == tree),
        None,
    )
    if (
        newest is None
        or newest.get("status") != "completed"
        or newest.get("conclusion") != "success"
    ):
        raise evidence.EvidenceError(
            "latest identical-tree candidate is not completed green"
        )
    return newest


def _reused(client: evidence.GitHubClient, base: str) -> dict[str, Any]:
    newest = _latest_tree_candidate(client, base)
    discovery = evidence.discover(client, "ci", base).evidence
    if discovery is None or discovery.source_run_id != newest["id"]:
        raise evidence.EvidenceError("no authoritative latest full candidate proof")
    full = _artifact(client, discovery)
    evidence.validate_evidence(
        client, base, discovery, full, Path("unused-ci-manifest")
    )
    # A status/run can be revoked while the artifact and source jobs are read.
    current = _latest_tree_candidate(client, base)
    if (current["id"], current["run_attempt"]) != (
        newest["id"],
        newest["run_attempt"],
    ) or evidence.discover(client, "ci", base).evidence != discovery:
        raise evidence.EvidenceError("full attestation changed during validation")
    return {
        "candidate_sha": discovery.candidate_sha,
        "attestation_run_id": discovery.attestation_run_id,
        "source_run_id": discovery.source_run_id,
    }


def qualify_main(client: evidence.GitHubClient, base: str) -> dict[str, Any]:
    """Fail closed; positive result is a full-base prerequisite, not merge proof."""
    result: dict[str, Any] = {
        "qualified": False,
        "base_sha": base,
        "authorizes_reduced_ci": False,
    }
    try:
        evidence._require_sha(base)
        if _tip(client) != base:
            raise evidence.EvidenceError("base is not current main")
        run = _latest(client, base)
        jobs = evidence._successful_jobs(client, run)
        evidence._require_unique_success(jobs, "changes")
        evidence._require_unique_success(jobs, "tests")
        skipped_matrix = []
        for prefix, names in evidence.REQUIRED_CI_MATRIX_JOBS.items():
            generic = prefix.rstrip(" (")
            selected = [
                j
                for j in jobs
                if j.get("name") == generic or str(j.get("name", "")).startswith(prefix)
            ]
            # Actions creates one generic job when a whole matrix is skipped.
            identities = {j.get("name") for j in selected}
            if (len(selected) == 1 and identities == {generic}) or (
                len(selected) == len(names) and identities == set(names)
            ):
                skipped_matrix.extend(selected)
        reuse = (
            bool(skipped_matrix)
            and all(
                j.get("status") == "completed" and j.get("conclusion") == "skipped"
                for j in skipped_matrix
            )
            and len({j["name"].split(" (")[0] for j in skipped_matrix}) == 2
        )
        if not reuse:
            matrix = _matrix(jobs)
            for job in matrix:
                evidence._require_unique_success(matrix, job["name"])
            for name in MAIN_EXECUTED_JOBS:
                evidence._require_unique_success(jobs, name)
            kind = "executed-full"
        else:
            evidence._require_unique_success(jobs, "queue-tree-evidence")
            for name in MAIN_EXECUTED_JOBS:
                matches = [j for j in jobs if j.get("name") == name]
                if (
                    len(matches) != 1
                    or matches[0].get("status") != "completed"
                    or matches[0].get("conclusion") != "skipped"
                ):
                    raise evidence.EvidenceError(
                        "main reuse has inconsistent skipped jobs"
                    )
            result["full_evidence"] = _reused(client, base)
            kind = "authenticated-full-reuse"
        latest = _latest(client, base)
        if (latest["id"], latest["run_attempt"]) != (
            run["id"],
            run["run_attempt"],
        ) or _tip(client) != base:
            raise evidence.EvidenceError("main qualification changed during validation")
        result.update(
            qualified=True, kind=kind, run_id=run["id"], run_attempt=run["run_attempt"]
        )
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result
