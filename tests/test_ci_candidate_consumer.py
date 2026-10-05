"""Authenticated transport cannot resurrect stale or scoped candidate proof."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import ci_candidate_consumer as consumer
from scripts import ci_candidate_qualification as producer
from tests.test_ci_candidate_qualification import fixture
from tests.test_queue_tree_evidence import CANDIDATE, MAIN, REPO


def setup(monkeypatch):
    client, _ = fixture(monkeypatch)
    client.gh = "gh"
    record = producer.qualify_candidate(client, 20, MAIN)
    assert record["qualified"]
    client.responses[f"repos/{REPO}/git/ref/heads/main"] = {"object": {"sha": MAIN}}
    status = {
        "created_at": "2026-10-05T23:00:02Z",
        "id": 501,
        "context": consumer.CONTEXT,
        "state": "success",
        "creator": {"login": "github-actions[bot]"},
        "target_url": f"https://github.com/{REPO}/actions/runs/100",
    }
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [status]
    run = {
        "run_started_at": "2026-10-05T23:00:00Z",
        "id": 100,
        "workflow_id": 5,
        "run_attempt": 1,
        "path": consumer.WORKFLOW,
        "event": "workflow_run",
        "repository": {"full_name": REPO},
        "status": "completed",
        "conclusion": "success",
    }
    client.responses[f"repos/{REPO}/actions/runs/100"] = run
    client.responses[f"repos/{REPO}/actions/workflows/candidate-qualification.yml"] = {
        "id": 5,
        "path": consumer.WORKFLOW,
    }
    client.job_records[100] = [
        {
            "id": 1001,
            "run_attempt": 1,
            "name": "qualify",
            "status": "completed",
            "conclusion": "success",
            "steps": [
                {"name": name, "status": "completed", "conclusion": "success"}
                for name in (
                    "Upload separate qualification",
                    "Index qualified candidate",
                )
            ],
        }
    ]
    artifact = {
        "workflow_run": {"id": 100},
        "created_at": "2026-10-05T23:00:01Z",
        "id": 101,
        "name": f"candidate-qualification-ci-{CANDIDATE}",
        "expired": False,
        "size_in_bytes": 100,
    }
    client.responses[f"repos/{REPO}/actions/runs/100/artifacts"] = {
        "total_count": 1,
        "artifacts": [artifact],
    }

    def download(command, **kwargs):
        assert command == ["gh", "api", f"repos/{REPO}/actions/artifacts/101/zip"]
        assert kwargs["check"] and kwargs["timeout"] == 20
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("candidate-qualification.json", json.dumps(record))
        return SimpleNamespace(stdout=buf.getvalue())

    monkeypatch.setattr(consumer.subprocess, "run", download)
    return client, record, status, run, artifact


def test_real_archive_and_live_full_jobs(monkeypatch):
    client, _, _, _, _ = setup(monkeypatch)
    result = consumer.consume_full(client, CANDIDATE)
    assert result["verified"] and result["kind"] == "full"
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]


@pytest.mark.parametrize(
    "change",
    [
        "status-missing",
        "status-revoked",
        "status-author",
        "status-url",
        "producer-event",
        "producer-workflow",
        "producer-repo",
        "producer-red",
        "producer-pending",
        "producer-attempt",
        "workflow-id",
        "artifact-expired",
        "artifact-large",
        "artifact-duplicate",
        "artifact-truncated",
        "artifact-id",
        "record-source",
        "record-schema",
        "record-mapped",
        "record-base",
        "record-tree",
        "record-attempt",
        "record-repo",
        "record-authority",
        "source-new-attempt",
        "source-cancelled",
        "source-full-missing",
        "candidate-closed",
        "candidate-author",
    ],
)
def test_fail_closed_transport_and_live_controls(monkeypatch, change):
    client, record, status, run, artifact = setup(monkeypatch)
    prefix = f"repos/{REPO}"
    if change == "status-missing":
        client.responses[f"{prefix}/commits/{CANDIDATE}/statuses"] = []
    elif change == "status-revoked":
        client.responses[f"{prefix}/commits/{CANDIDATE}/statuses"].append(
            dict(status, id=502, state="failure")
        )
    elif change == "status-author":
        status["creator"]["login"] = "human"
    elif change == "status-url":
        status["target_url"] = "https://github.com/other/repo/actions/runs/100"
    elif change.startswith("producer-"):
        fields = {
            "event": ("event", "pull_request"),
            "workflow": ("path", ".github/workflows/ci.yml"),
            "repo": ("repository", {"full_name": "other/repo"}),
            "red": ("conclusion", "failure"),
            "pending": ("status", "in_progress"),
            "attempt": ("run_attempt", True),
        }
        key, value = fields[change.removeprefix("producer-")]
        run[key] = value
    elif change == "workflow-id":
        run["workflow_id"] = 6
    elif change == "artifact-expired":
        artifact["expired"] = True
    elif change == "artifact-large":
        artifact["size_in_bytes"] = consumer.MAX_BYTES + 1
    elif change == "artifact-id":
        artifact["id"] = True
    elif change == "artifact-duplicate":
        client.responses[f"{prefix}/actions/runs/100/artifacts"] = {
            "total_count": 2,
            "artifacts": [artifact, deepcopy(artifact)],
        }
    elif change == "artifact-truncated":
        client.responses[f"{prefix}/actions/runs/100/artifacts"]["total_count"] = 100
    elif change.startswith("record-"):
        fields = {
            "source": ("source_run_id", True),
            "schema": ("schema", "rapid-mlx/source-canary-unit/v1"),
            "mapped": ("kind", "mapped"),
            "base": ("base_sha", "e" * 40),
            "tree": ("candidate_tree", "e" * 40),
            "attempt": ("source_attempt", 2),
            "repo": ("repository", "other/repo"),
            "authority": ("authorizes_reduced_ci", True),
        }
        key, value = fields[change.removeprefix("record-")]
        record[key] = value
    elif change == "source-new-attempt":
        client.responses[f"{prefix}/actions/runs/20"]["run_attempt"] = 2
    elif change == "source-cancelled":
        client.responses[f"{prefix}/actions/runs/20"]["conclusion"] = "cancelled"
    elif change == "source-full-missing":
        client.job_records[20].pop()
    elif change == "candidate-closed":
        client.responses[f"{prefix}/pulls"] = []
    elif change == "candidate-author":
        client.responses[f"{prefix}/pulls"][0]["user"]["login"] = "human"
    result = consumer.consume_full(client, CANDIDATE)
    assert not result["verified"] and not result["authorizes_merge"]
    assert not result["authorizes_reduced_ci"]


@pytest.mark.parametrize("entry", ["../candidate-qualification.json", "evidence.json"])
def test_zip_path_is_rejected_without_extraction(monkeypatch, entry):
    client, record, *_ = setup(monkeypatch)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(entry, json.dumps(record))
    monkeypatch.setattr(
        consumer.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=buf.getvalue()),
    )
    assert not consumer.consume_full(client, CANDIDATE)["verified"]


@pytest.mark.parametrize("boundary", ["main", "producer", "status", "candidate"])
def test_mutation_during_final_reread_is_rejected(monkeypatch, boundary):
    client, _, status, run, _ = setup(monkeypatch)
    original = client.json
    counts = {}
    endpoints = {
        "main": f"repos/{REPO}/git/ref/heads/main",
        "producer": f"repos/{REPO}/actions/runs/100",
        "status": f"repos/{REPO}/commits/{CANDIDATE}/statuses",
        "candidate": f"repos/{REPO}/pulls",
    }

    def changing(endpoint, *fields, **kwargs):
        counts[endpoint] = counts.get(endpoint, 0) + 1
        if endpoint == endpoints[boundary] and counts[endpoint] == 2:
            if boundary == "main":
                return {"object": {"sha": "e" * 40}}
            if boundary == "producer":
                return dict(run, run_attempt=2)
            if boundary == "status":
                return [[dict(status, id=502, state="failure")]]
            if boundary == "candidate":
                return []
        return original(endpoint, *fields, **kwargs)

    client.json = changing
    assert not consumer.consume_full(client, CANDIDATE)["verified"]


def test_download_failure_is_cache_miss(monkeypatch):
    client, *_ = setup(monkeypatch)

    def failed(*a, **k):
        raise subprocess.TimeoutExpired("gh", 20)

    monkeypatch.setattr(consumer.subprocess, "run", failed)
    assert not consumer.consume_full(client, CANDIDATE)["verified"]


@pytest.mark.parametrize(
    "change",
    [
        "old-status",
        "old-artifact",
        "wrong-artifact-run",
        "skipped-upload",
        "old-producer-jobs",
        "missing-time",
    ],
)
def test_publication_must_bind_current_producer_attempt(monkeypatch, change):
    client, _, status, _, artifact = setup(monkeypatch)
    if change == "old-status":
        status["created_at"] = "2026-10-04T23:00:02Z"
    elif change == "old-artifact":
        artifact["created_at"] = "2026-10-04T23:00:01Z"
    elif change == "wrong-artifact-run":
        artifact["workflow_run"]["id"] = 99
    elif change == "skipped-upload":
        client.job_records[100][0]["steps"][0]["conclusion"] = "skipped"
    elif change == "old-producer-jobs":
        client.job_records[100][0]["run_attempt"] = 0
    elif change == "missing-time":
        status.pop("created_at")
    assert not consumer.consume_full(client, CANDIDATE)["verified"]


@pytest.mark.parametrize(
    "change",
    [
        "timezone",
        "incomplete-list",
        "download-bound",
        "late-producer",
        "late-status",
        "late-candidate",
    ],
)
def test_bounds_and_late_identity_changes(monkeypatch, change):
    client, _, status, run, _ = setup(monkeypatch)
    if change == "timezone":
        status["created_at"] = "2026-10-05T23:00:02"
    elif change == "incomplete-list":
        client.responses[f"repos/{REPO}/actions/runs/100/artifacts"]["total_count"] = 2
    elif change == "download-bound":
        monkeypatch.setattr(
            consumer.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout=b"x" * (consumer.MAX_BYTES + 1)),
        )
    else:
        original = client.json
        calls = {}
        endpoint = {
            "late-producer": f"repos/{REPO}/actions/runs/100",
            "late-status": f"repos/{REPO}/commits/{CANDIDATE}/statuses",
            "late-candidate": f"repos/{REPO}/pulls",
        }[change]

        def altered(path, *a, **k):
            calls[path] = calls.get(path, 0) + 1
            if path == endpoint and calls[path] == (
                3 if change == "late-candidate" else 2
            ):
                if change == "late-producer":
                    return dict(run, updated_at="2026-10-05T23:01:00Z")
                if change == "late-status":
                    return [[dict(status, id=502)]]
                return []
            return original(path, *a, **k)

        client.json = altered
    assert not consumer.consume_full(client, CANDIDATE)["verified"]


@pytest.mark.parametrize("field", ["source_attempt", "candidate_pr"])
def test_boolean_identity_is_not_integer(monkeypatch, field):
    client, record, *_ = setup(monkeypatch)
    record[field] = True
    assert not consumer.consume_full(client, CANDIDATE)["verified"]
