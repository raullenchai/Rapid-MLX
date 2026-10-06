"""Read-only full qualification transport; never authorize merging or mapped CI."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from datetime import datetime
from typing import Any

from scripts import ci_candidate_qualification as producer
from scripts import queue_tree_evidence as evidence

WORKFLOW = ".github/workflows/candidate-qualification.yml"
CONTEXT = "candidate-qualification/ci"
MAX_BYTES = 1_000_000


def _status(client: evidence.GitHubClient, sha: str) -> dict:
    pages = client.json(
        f"repos/{client.repo}/commits/{sha}/statuses",
        "per_page=100",
        paginate=True,
    )
    statuses = [s for page in pages for s in page if s.get("context") == CONTEXT]
    if not statuses:
        raise evidence.EvidenceError("qualification status is absent")
    status = max(statuses, key=lambda s: evidence._field(s, "id", int))
    if (
        status.get("state") != "success"
        or status.get("creator", {}).get("login") != "github-actions[bot]"
    ):
        raise evidence.EvidenceError("qualification status is untrusted or revoked")
    return status


def _producer_run(client: evidence.GitHubClient, run_id: int) -> dict:
    run = client.json(f"repos/{client.repo}/actions/runs/{run_id}")
    workflow = client.json(
        f"repos/{client.repo}/actions/workflows/candidate-qualification.yml"
    )
    if (
        run.get("id") != run_id
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] < 1
        or run.get("workflow_id") != workflow.get("id")
        or workflow.get("path") != WORKFLOW
        or run.get("path") != WORKFLOW
        or run.get("event") != "workflow_run"
        or run.get("repository", {}).get("full_name") != client.repo
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        raise evidence.EvidenceError("producer identity/attempt is not successful")
    job = evidence._require_unique_success(
        evidence._successful_jobs(client, run), "qualify"
    )
    for name in ("Upload separate qualification", "Index qualified candidate"):
        steps = [step for step in job.get("steps", []) if step.get("name") == name]
        if (
            len(steps) != 1
            or steps[0].get("status") != "completed"
            or steps[0].get("conclusion") != "success"
        ):
            raise evidence.EvidenceError(
                "current producer attempt did not publish qualification"
            )
    return run


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise evidence.EvidenceError("missing publication timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise evidence.EvidenceError("publication timestamp lacks timezone")
    return parsed


def _artifact(
    client: evidence.GitHubClient, run_id: int, sha: str, started: datetime
) -> dict:
    page = client.json(
        f"repos/{client.repo}/actions/runs/{run_id}/artifacts", "per_page=100"
    )
    if type(page.get("total_count")) is not int or not 0 < page["total_count"] < 100:
        raise evidence.EvidenceError("artifact listing is empty or truncated")
    artifacts = page["artifacts"]
    if len(artifacts) != page["total_count"]:
        raise evidence.EvidenceError("incomplete artifact listing")
    matching = [
        a for a in artifacts if a.get("name") == f"candidate-qualification-ci-{sha}"
    ]
    if len(matching) != 1:
        raise evidence.EvidenceError("missing or duplicate qualification artifact")
    artifact = matching[0]
    if (
        type(artifact.get("id")) is not int
        or artifact["id"] < 1
        or artifact.get("workflow_run", {}).get("id") != run_id
        or _time(artifact.get("created_at")) < started
        or artifact.get("expired") is not False
        or type(artifact.get("size_in_bytes")) is not int
        or not 0 < artifact["size_in_bytes"] <= MAX_BYTES
    ):
        raise evidence.EvidenceError("artifact is expired, malformed or too large")
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
    if len(raw) > MAX_BYTES:
        raise evidence.EvidenceError("artifact download exceeds bound")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (
            len(members) != 1
            or members[0].filename != "candidate-qualification.json"
            or members[0].file_size > MAX_BYTES
        ):
            raise evidence.EvidenceError("unexpected archive content")
        # Read only; no archive path is ever extracted or executed.
        return json.loads(archive.read(members[0]))


def consume_full(client: evidence.GitHubClient, candidate_sha: str) -> dict[str, Any]:
    """Recompute full proof, then recheck mutable state; unused by queue rules."""
    result: dict[str, Any] = {
        "verified": False,
        "authorizes_merge": False,
        "authorizes_reduced_ci": False,
    }
    try:
        evidence._require_sha(candidate_sha)
        status = _status(client, candidate_sha)
        target = evidence.RUN_URL_RE.fullmatch(str(status.get("target_url", "")))
        if not target or target.group("repo") != client.repo:
            raise evidence.EvidenceError("status does not bind own-repository producer")
        run_id = int(target.group("run_id"))
        run = _producer_run(client, run_id)
        started = _time(run.get("run_started_at"))
        if _time(status.get("created_at")) < started:
            raise evidence.EvidenceError("status predates current producer attempt")
        record = _artifact(client, run_id, candidate_sha, started)
        if (
            record.get("schema") != producer.SCHEMA
            or record.get("kind") != "full"
            or record.get("candidate_sha") != candidate_sha
            or record.get("qualified") is not True
            or record.get("authorizes_reduced_ci") is not False
            or type(record.get("source_run_id")) is not int
            or record["source_run_id"] < 1
            or type(record.get("source_attempt")) is not int
            or record["source_attempt"] < 1
            or type(record.get("candidate_pr")) is not int
            or record["candidate_pr"] < 1
        ):
            raise evidence.EvidenceError("not a full qualification record")
        tip = client.json(f"repos/{client.repo}/git/ref/heads/main")["object"]["sha"]
        evidence._require_sha(tip)
        if record.get("base_sha") != tip:
            raise evidence.EvidenceError("qualification base is stale")
        # The transport is not authority: repeat actual full jobs, latest source
        # attempt, creator/ref/base/tree/first-parent and open-candidate checks.
        current = producer.qualify_candidate(client, record["source_run_id"], tip)
        if current != record or not current.get("qualified"):
            raise evidence.EvidenceError(
                "artifact differs from live full qualification"
            )
        if _producer_run(client, run_id) != run:
            raise evidence.EvidenceError("producer changed during verification")
        if _status(client, candidate_sha) != status:
            raise evidence.EvidenceError("status changed during verification")
        if (
            client.json(f"repos/{client.repo}/git/ref/heads/main")["object"]["sha"]
            != tip
        ):
            raise evidence.EvidenceError("main changed during verification")
        if producer.qualify_candidate(client, record["source_run_id"], tip) != current:
            raise evidence.EvidenceError("candidate changed during verification")
        result.update(verified=True, kind="full", qualification=current)
    except (
        evidence.EvidenceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
        UnicodeError,
    ) as exc:
        result["reason"] = str(exc)[:500]
    return result
