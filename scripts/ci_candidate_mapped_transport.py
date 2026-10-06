"""Dormant mapped input transport: no routing, publishing or merge permission."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from typing import Any

from scripts import ci_candidate_qualification as producer
from scripts import queue_tree_evidence as evidence

CONTROL_PATH = "scripts/ci_candidate_mapped_transport.py"
MAX_BYTES = 1_000_000


def _input(client: evidence.GitHubClient, run: dict) -> dict:
    sha = run["head_sha"]
    run_id, attempt = run["id"], run["run_attempt"]
    job = evidence._require_unique_success(
        evidence._successful_jobs(client, run), "candidate-canary-unit"
    )
    steps = [
        s
        for s in job.get("steps", [])
        if s.get("name") == "Upload mapped execution proof"
    ]
    if (
        len(steps) != 1
        or steps[0].get("status") != "completed"
        or steps[0].get("conclusion") != "success"
    ):
        raise evidence.EvidenceError("mapped input was not uploaded by current job")
    page = client.json(
        f"repos/{client.repo}/actions/runs/{run_id}/artifacts", "per_page=100"
    )
    count = page.get("total_count")
    if type(count) is not int or not 0 < count < 100 or len(page["artifacts"]) != count:
        raise evidence.EvidenceError(
            "mapped artifact listing is malformed or truncated"
        )
    name = f"candidate-mapped-input-{sha}-{run_id}-{attempt}"
    matching = [a for a in page["artifacts"] if a.get("name") == name]
    if len(matching) != 1:
        raise evidence.EvidenceError("missing or duplicate exact-attempt mapped input")
    artifact = matching[0]
    if (
        type(artifact.get("id")) is not int
        or artifact["id"] < 1
        or artifact.get("expired") is not False
        or artifact.get("workflow_run", {}).get("id") != run_id
        or artifact.get("workflow_run", {}).get("head_sha") != sha
        or type(artifact.get("size_in_bytes")) is not int
        or not 0 < artifact["size_in_bytes"] <= MAX_BYTES
    ):
        raise evidence.EvidenceError("mapped input identity/size/expiry mismatch")
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
        raise evidence.EvidenceError("mapped input download exceeds bound")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (
            len(members) != 1
            or members[0].filename != "candidate-mapped-input.json"
            or members[0].file_size > MAX_BYTES
        ):
            raise evidence.EvidenceError("unexpected mapped archive contents")
        return json.loads(archive.read(members[0]))


def inspect_mapped_transport(
    client: evidence.GitHubClient, source_run_id: int, trusted_ref: str
) -> dict[str, Any]:
    """Inspect a future artifact; caller cannot activate any CI route here."""
    result: dict[str, Any] = {
        "verified": False,
        "authorizes_merge": False,
        "authorizes_reduced_ci": False,
    }
    try:
        if type(source_run_id) is not int or source_run_id < 1:
            raise evidence.EvidenceError("invalid source run identity")
        evidence._require_sha(trusted_ref)
        run = client.json(f"repos/{client.repo}/actions/runs/{source_run_id}")
        if (
            run.get("id") != source_run_id
            or type(run.get("run_attempt")) is not int
            or run["run_attempt"] < 1
        ):
            raise evidence.EvidenceError("invalid mapped run/attempt identity")
        producer._candidate(client, run)
        latest = producer._latest(client, run["head_sha"], run["head_branch"])
        if (latest["id"], latest["run_attempt"]) != (source_run_id, run["run_attempt"]):
            raise evidence.EvidenceError(
                "mapped input is not from latest source attempt"
            )
        sha = run["head_sha"]
        if evidence._path_blob(client, sha, CONTROL_PATH) != evidence._path_blob(
            client, trusted_ref, CONTROL_PATH
        ):
            raise evidence.EvidenceError(
                "mapped candidate changes transport controller"
            )
        proof = _input(client, run)
        for field, expected in (
            ("source_run_id", source_run_id),
            ("source_attempt", run["run_attempt"]),
        ):
            if type(proof.get(field)) is not int or proof[field] != expected:
                raise evidence.EvidenceError("mapped input source identity is invalid")
        qualification = producer.qualify_candidate(
            client, source_run_id, trusted_ref, mapped_enabled=True, mapped_input=proof
        )
        if not qualification.get("qualified") or qualification.get("kind") != "mapped":
            raise evidence.EvidenceError(
                "transport lacks current mapped execution qualification"
            )
        if client.json(f"repos/{client.repo}/actions/runs/{source_run_id}") != run:
            raise evidence.EvidenceError("mapped source changed during transport")
        # A second live qualification rejects observed job/PR/tree/main changes.
        if (
            producer.qualify_candidate(
                client,
                source_run_id,
                trusted_ref,
                mapped_enabled=True,
                mapped_input=proof,
            )
            != qualification
        ):
            raise evidence.EvidenceError(
                "mapped qualification changed during transport"
            )
        result.update(
            verified=True,
            qualification=dict(qualification, authorizes_reduced_ci=False),
        )
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
