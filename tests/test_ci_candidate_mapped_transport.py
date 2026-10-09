"""Mapped input is separate from full evidence and bound to live execution."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from scripts import ci_candidate_mapped_transport as transport
from tests.test_ci_candidate_qualification import fixture
from tests.test_queue_tree_evidence import CANDIDATE, REPO, TRUSTED


def setup(monkeypatch):
    client, proof = fixture(monkeypatch, True)
    client.gh = "gh"
    for sha in (CANDIDATE, TRUSTED):
        client.blobs[sha, transport.CONTROL_PATH] = "f" * 40
    job = next(
        j for j in client.job_records[20] if j["name"] == "candidate-canary-unit"
    )
    job["steps"].append(
        {
            "name": "Upload mapped execution proof",
            "status": "completed",
            "conclusion": "success",
        }
    )
    artifact = {
        "id": 101,
        "name": f"candidate-mapped-input-{CANDIDATE}-20-1",
        "workflow_run": {"id": 20, "head_sha": CANDIDATE},
        "expired": False,
        "size_in_bytes": 100,
    }
    client.responses[f"repos/{REPO}/actions/runs/20/artifacts"] = {
        "total_count": 1,
        "artifacts": [artifact],
    }

    def download(command, **kwargs):
        assert command == ["gh", "api", f"repos/{REPO}/actions/artifacts/101/zip"]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("candidate-mapped-input.json", json.dumps(proof))
        return SimpleNamespace(stdout=buf.getvalue())

    monkeypatch.setattr(transport.subprocess, "run", download)
    return client, proof, job, artifact


def test_zip_and_actual_qualification_logic(monkeypatch):
    client, *_ = setup(monkeypatch)
    result = transport.inspect_mapped_transport(client, 20, TRUSTED)
    assert result["verified"] and result["qualification"]["kind"] == "mapped"
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]
    assert not result["qualification"]["authorizes_reduced_ci"]


@pytest.mark.parametrize(
    "change",
    [
        "old-attempt-name",
        "source-artifact",
        "expired",
        "artifact-run",
        "artifact-head",
        "size",
        "truncated",
        "duplicate",
        "no-upload",
        "coverage",
        "skipped-test",
        "tree",
        "schema",
        "controller",
        "cancelled",
        "boolean-run",
    ],
)
def test_transport_rejects_invalid_proof(monkeypatch, change):
    client, proof, job, artifact = setup(monkeypatch)
    run_id = 20
    if change == "old-attempt-name":
        artifact["name"] = artifact["name"].removesuffix("1") + "0"
    elif change == "source-artifact":
        artifact["name"] = f"source-canary-unit-{CANDIDATE}"
    elif change == "expired":
        artifact["expired"] = True
    elif change == "artifact-run":
        artifact["workflow_run"]["id"] = 19
    elif change == "artifact-head":
        artifact["workflow_run"]["head_sha"] = TRUSTED
    elif change == "size":
        artifact["size_in_bytes"] = transport.MAX_BYTES + 1
    elif change == "truncated":
        client.responses[f"repos/{REPO}/actions/runs/20/artifacts"]["total_count"] = 100
    elif change == "duplicate":
        client.responses[f"repos/{REPO}/actions/runs/20/artifacts"] = {
            "total_count": 2,
            "artifacts": [artifact, artifact],
        }
    elif change == "no-upload":
        job["steps"].pop()
    elif change == "coverage":
        job["steps"][1]["conclusion"] = "failure"
    elif change == "skipped-test":
        proof["executed"]["reports"][0]["outcome"] = "skipped"
    elif change == "tree":
        proof["tested_sha"] = "bad"
    elif change == "schema":
        proof["schema"] = "rapid-mlx/queue-tree-evidence/ci/v1"
    elif change == "controller":
        client.blobs[CANDIDATE, transport.CONTROL_PATH] = "e" * 40
    elif change == "cancelled":
        client.responses[f"repos/{REPO}/actions/runs/20"]["conclusion"] = "cancelled"
    elif change == "boolean-run":
        run_id = True
    assert not transport.inspect_mapped_transport(client, run_id, TRUSTED)["verified"]


def test_current_full_ci_cannot_fake_mapped_transport(monkeypatch):
    client, _ = fixture(monkeypatch)
    for sha in (CANDIDATE, TRUSTED):
        client.blobs[sha, transport.CONTROL_PATH] = "f" * 40
    result = transport.inspect_mapped_transport(client, 20, TRUSTED)
    assert not result["verified"] and "candidate-canary-unit" in result["reason"]


@pytest.mark.parametrize(
    "change",
    [
        "red-main",
        "boolean-proof",
        "archive-path",
        "large-download",
        "incomplete-list",
        "late-source",
        "late-main",
    ],
)
def test_main_attempt_and_archive_boundaries(monkeypatch, change):
    client, proof, _, _ = setup(monkeypatch)
    if change == "red-main":
        monkeypatch.setattr(
            transport.producer, "qualify_main", lambda *a: {"qualified": False}
        )
    elif change == "boolean-proof":
        proof["source_attempt"] = True
    elif change == "archive-path":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("../candidate-mapped-input.json", json.dumps(proof))
        monkeypatch.setattr(
            transport.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout=buf.getvalue()),
        )
    elif change == "large-download":
        monkeypatch.setattr(
            transport.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout=b"x" * (transport.MAX_BYTES + 1)),
        )
    elif change == "incomplete-list":
        client.responses[f"repos/{REPO}/actions/runs/20/artifacts"]["total_count"] = 2
    elif change == "late-main":
        calls = []

        def anchor(*a):
            calls.append(True)
            return {"qualified": len(calls) < 3}

        monkeypatch.setattr(transport.producer, "qualify_main", anchor)
    elif change == "late-source":
        original = client.json
        count = []

        def changed(endpoint, *a, **k):
            value = original(endpoint, *a, **k)
            if endpoint == f"repos/{REPO}/actions/runs/20":
                count.append(True)
                if len(count) == 3:
                    return dict(value, updated_at="later")
            return value

        client.json = changed
    assert not transport.inspect_mapped_transport(client, 20, TRUSTED)["verified"]


@pytest.mark.parametrize("change", ["invalid-run-attempt", "newer-source-run"])
def test_source_identity_cannot_use_old_success(monkeypatch, change):
    client, *_ = setup(monkeypatch)
    run = client.responses[f"repos/{REPO}/actions/runs/20"]
    if change == "invalid-run-attempt":
        run["run_attempt"] = True
    else:
        client.responses[f"repos/{REPO}/actions/workflows/ci.yml/runs"][
            "workflow_runs"
        ].append(dict(run, id=21))
    assert not transport.inspect_mapped_transport(client, 20, TRUSTED)["verified"]
