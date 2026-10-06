"""Full-main prerequisites never authorize reduced candidates by themselves."""

from __future__ import annotations

import copy
import io
import zipfile
from types import SimpleNamespace

import pytest

from scripts import ci_main_qualification as qualify
from scripts import queue_tree_evidence as evidence

BASE = "a" * 40
REPO = "owner/repo"


class Client:
    repo = REPO
    gh = "gh"

    def __init__(self):
        self.tip = BASE
        self.runs = [
            {
                "id": 100,
                "run_attempt": 1,
                "head_sha": BASE,
                "head_branch": "main",
                "head_repository": {"full_name": REPO},
                "path": evidence.CI_WORKFLOW_PATH,
                "event": "push",
                "status": "completed",
                "conclusion": "success",
            }
        ]
        names = [
            "changes",
            "tests",
            *qualify.MAIN_EXECUTED_JOBS,
            *(n for group in evidence.REQUIRED_CI_MATRIX_JOBS.values() for n in group),
        ]
        self.records = [
            {
                "id": i,
                "name": n,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
            for i, n in enumerate(names)
        ]
        self.calls = 0
        self.race = None

    def json(self, endpoint, *fields, **kwargs):
        if endpoint.endswith("/git/ref/heads/main"):
            return {"object": {"sha": self.tip}}
        assert endpoint.endswith("/actions/workflows/ci.yml/runs")
        assert kwargs["paginate"] is True
        assert f"head_sha={BASE}" in fields
        self.calls += 1
        if self.calls > 1 and self.race:
            self.race(self)
        return [{"workflow_runs": copy.deepcopy(self.runs)}]

    def jobs(self, run_id):
        assert run_id == 100
        return copy.deepcopy(self.records)


def test_complete_current_full_main_is_prerequisite_only():
    result = qualify.qualify_main(Client(), BASE)
    assert result == {
        "qualified": True,
        "base_sha": BASE,
        "authorizes_reduced_ci": False,
        "kind": "executed-full",
        "run_id": 100,
        "run_attempt": 1,
    }


@pytest.mark.parametrize(
    "status,conclusion",
    [
        ("queued", None),
        ("in_progress", None),
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("completed", "skipped"),
    ],
)
def test_latest_non_green_supersedes_older_success(status, conclusion):
    client = Client()
    newer = dict(client.runs[0], id=101, status=status, conclusion=conclusion)
    client.runs.append(newer)
    assert qualify.qualify_main(client, BASE)["qualified"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("head_branch", "feature"),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
        ("head_repository", {"full_name": "fork/repo"}),
        ("run_attempt", True),
        ("id", 0),
    ],
)
def test_run_identity_drift_rejected(field, value):
    client = Client()
    client.runs[0][field] = value
    assert qualify.qualify_main(client, BASE)["qualified"] is False


@pytest.mark.parametrize(
    "name",
    [
        "changes",
        "tests",
        *qualify.MAIN_EXECUTED_JOBS,
        *(n for group in evidence.REQUIRED_CI_MATRIX_JOBS.values() for n in group),
    ],
)
@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "cancelled", "old-attempt"]
)
def test_every_full_identity_is_required(name, mutation):
    client = Client()
    job = next(j for j in client.records if j["name"] == name)
    if mutation == "missing":
        client.records.remove(job)
    elif mutation == "duplicate":
        client.records.append(dict(job, id=999))
    elif mutation == "cancelled":
        job["conclusion"] = "cancelled"
    else:
        job["run_attempt"] = 0
    assert qualify.qualify_main(client, BASE)["qualified"] is False


@pytest.mark.parametrize("race", ["tip", "new-run", "new-attempt", "cancelled"])
def test_revalidation_races_fail_closed(race):
    client = Client()

    def change(c):
        if race == "tip":
            c.tip = "b" * 40
        elif race == "new-run":
            c.runs.append(dict(c.runs[0], id=101))
        elif race == "new-attempt":
            c.runs[0]["run_attempt"] = 2
        else:
            c.runs[0]["conclusion"] = "cancelled"

    client.race = change
    assert qualify.qualify_main(client, BASE)["qualified"] is False


def reused_client():
    client = Client()
    for job in client.records:
        if job["name"] not in ("changes", "tests"):
            job["conclusion"] = "skipped"
    client.records.append(
        {
            "name": "queue-tree-evidence",
            "id": 999,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "success",
        }
    )
    return client


def test_skipped_matrix_requires_authenticated_full_proof(monkeypatch):
    client = reused_client()

    def revalidate(c, base):
        assert c is client and base == BASE
        return {"source_run_id": 20}

    monkeypatch.setattr(qualify, "_reused", revalidate)
    result = qualify.qualify_main(client, BASE)
    assert result["qualified"] is True
    assert result["kind"] == "authenticated-full-reuse"
    assert result["full_evidence"] == {"source_run_id": 20}
    assert result["authorizes_reduced_ci"] is False


def test_green_reuse_aggregate_does_not_bypass_invalid_artifact(monkeypatch):
    def reject(*args):
        raise evidence.EvidenceError("untrusted artifact")

    monkeypatch.setattr(qualify, "_reused", reject)
    assert qualify.qualify_main(reused_client(), BASE)["qualified"] is False


def test_mixed_skipped_executed_matrix_cannot_claim_full_or_reuse():
    client = reused_client()
    client.records[-2]["conclusion"] = "success"
    assert qualify.qualify_main(client, BASE)["qualified"] is False


@pytest.mark.parametrize(
    "page", [None, {}, [{"workflow_runs": None}], [{"workflow_runs": []}]]
)
def test_malformed_api_is_not_qualification(page):
    client = Client()
    original = client.json
    client.json = lambda endpoint, *args, **kwargs: (
        original(endpoint, *args, **kwargs)
        if endpoint.endswith("/git/ref/heads/main")
        else page
    )
    assert qualify.qualify_main(client, BASE)["qualified"] is False


def archive(name="evidence.json", content=b'{"scope":"ci"}'):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(name, content)
    return stream.getvalue()


@pytest.mark.parametrize(
    "name,content",
    [
        ("../evidence.json", b"{}"),
        ("evidence.json", b"not-json"),
        ("evidence.json", b"x" * 1_000_001),
    ],
)
def test_artifact_archive_negative_controls(monkeypatch, name, content):
    client = Client()
    client.json = lambda *a, **k: {
        "total_count": 1,
        "artifacts": [
            {"name": "full", "expired": False, "id": 10, "size_in_bytes": 100}
        ],
    }
    monkeypatch.setattr(
        qualify.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=archive(name, content)),
    )
    discovery = evidence.Discovery("ci", "b" * 40, "c" * 40, 30, "full", 20)
    with pytest.raises(evidence.EvidenceError):
        qualify._artifact(client, discovery)


def test_trusted_artifact_download_is_pinned_and_never_extracted(monkeypatch):
    client = Client()
    client.json = lambda *a, **k: {
        "total_count": 1,
        "artifacts": [
            {"name": "full", "expired": False, "id": 10, "size_in_bytes": 100}
        ],
    }

    def download(command, **kwargs):
        assert command == ["gh", "api", f"repos/{REPO}/actions/artifacts/10/zip"]
        assert kwargs["timeout"] == 20 and kwargs["check"]
        return SimpleNamespace(stdout=archive())

    monkeypatch.setattr(qualify.subprocess, "run", download)
    assert qualify._artifact(
        client, evidence.Discovery("ci", "b" * 40, "c" * 40, 30, "full", 20)
    ) == {"scope": "ci"}


class ReuseClient(Client):
    candidate = "b" * 40
    tree = "c" * 40

    def __init__(self):
        main = reused_client()
        self.__dict__.update(main.__dict__)
        self.candidate_run = dict(
            self.runs[0],
            id=20,
            head_sha=self.candidate,
            head_branch="mergify/merge-queue/0123456789",
            event="pull_request",
            html_url=f"https://github.com/{REPO}/actions/runs/20",
        )
        names = [
            *evidence.REQUIRED_CI_JOBS,
            *(n for group in evidence.REQUIRED_CI_MATRIX_JOBS.values() for n in group),
        ]
        self.candidate_jobs = [
            {
                "id": i + 1000,
                "run_attempt": 1,
                "name": n,
                "status": "completed",
                "conclusion": "success",
            }
            for i, n in enumerate(names)
        ]
        self.status = "success"
        self.newer_cancelled = False
        self.full = {
            "schema": evidence.SCHEMAS["ci"],
            "scope": "ci",
            "repository": REPO,
            "candidate_sha": self.candidate,
            "candidate_ref": self.candidate_run["head_branch"],
            "candidate_tree": self.tree,
            "candidate_pr": 123,
            "attestation_run_id": 30,
            "controls": {p: "d" * 40 for p in evidence.CI_CONTROL_PATHS},
            "source": {
                "id": 20,
                "attempt": 1,
                "url": self.candidate_run["html_url"],
                "jobs": [
                    {"id": j["id"], "name": j["name"], "conclusion": "success"}
                    for j in self.candidate_jobs
                ],
            },
        }

    def commit(self, sha):
        assert sha in (BASE, self.candidate)
        return {"tree": {"sha": self.tree}}

    def jobs(self, run_id):
        return (
            copy.deepcopy(self.candidate_jobs) if run_id == 20 else super().jobs(run_id)
        )

    def json(self, endpoint, *fields, **kwargs):
        if (
            endpoint.endswith("/actions/workflows/ci.yml/runs")
            and "event=pull_request" in fields
        ):
            runs = [copy.deepcopy(self.candidate_run)]
            if self.newer_cancelled:
                runs.append(dict(self.candidate_run, id=21, conclusion="cancelled"))
            page = {"workflow_runs": runs}
            return [page] if kwargs.get("paginate") else page
        if "/statuses" in endpoint:
            return [
                [
                    {
                        "id": 1,
                        "context": evidence.CONTEXTS["ci"],
                        "state": self.status,
                        "target_url": f"https://github.com/{REPO}/actions/runs/30",
                    }
                ]
            ]
        if endpoint.endswith("/actions/runs/30"):
            return {
                "path": evidence.ATTESTATION_WORKFLOW,
                "event": "workflow_run",
                "status": "completed",
                "conclusion": "success",
                "repository": {"full_name": REPO},
            }
        if endpoint.endswith("/actions/runs/20"):
            return copy.deepcopy(self.candidate_run)
        if "/contents/" in endpoint:
            return {"type": "file", "sha": "d" * 40}
        if endpoint.endswith("/actions/runs/30/artifacts"):
            return {
                "total_count": 1,
                "artifacts": [
                    {
                        "name": f"queue-tree-evidence-ci-{self.candidate}",
                        "id": 10,
                        "expired": False,
                        "size_in_bytes": 1000,
                    }
                ],
            }
        return super().json(endpoint, *fields, **kwargs)


def test_actual_full_namespace_consumer_revalidates_reused_main(monkeypatch):
    import json

    client = ReuseClient()
    monkeypatch.setattr(
        qualify.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=archive(content=json.dumps(client.full).encode())
        ),
    )
    result = qualify.qualify_main(client, BASE)
    assert result["qualified"] and result["kind"] == "authenticated-full-reuse"
    assert result["full_evidence"] == {
        "candidate_sha": client.candidate,
        "attestation_run_id": 30,
        "source_run_id": 20,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "source-schema",
        "reduced-schema",
        "wrong-tree",
        "control-drift",
        "recorded-jobs",
        "live-jobs",
        "new-attempt",
        "revoked",
        "newer-cancelled",
    ],
)
def test_actual_full_consumer_negative_controls(monkeypatch, mutation):
    import json

    client = ReuseClient()
    if mutation == "source-schema":
        client.full["schema"] = "rapid-mlx/source-canary-unit/v1"
    elif mutation == "reduced-schema":
        client.full["schema"] = "rapid-mlx/candidate-qualification/v1"
    elif mutation == "wrong-tree":
        client.full["candidate_tree"] = "e" * 40
    elif mutation == "control-drift":
        client.full["controls"][evidence.CI_WORKFLOW_PATH] = "e" * 40
    elif mutation == "recorded-jobs":
        client.full["source"]["jobs"].pop()
    elif mutation == "live-jobs":
        client.candidate_jobs[-1]["conclusion"] = "cancelled"
    elif mutation == "new-attempt":
        client.candidate_run["run_attempt"] = 2
    elif mutation == "revoked":
        client.status = "failure"
    else:
        client.newer_cancelled = True
    monkeypatch.setattr(
        qualify.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=archive(content=json.dumps(client.full).encode())
        ),
    )
    assert qualify.qualify_main(client, BASE)["qualified"] is False


def test_real_actions_generic_skipped_matrices_require_trusted_full_proof(monkeypatch):
    import json

    client = ReuseClient()
    client.records = [
        j
        for j in client.records
        if not any(
            str(j["name"]).startswith(p) for p in evidence.REQUIRED_CI_MATRIX_JOBS
        )
    ]
    for i, name in enumerate(("test-matrix", "l1-smoke")):
        client.records.append(
            {
                "id": 500 + i,
                "name": name,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "skipped",
            }
        )
    monkeypatch.setattr(
        qualify.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=archive(content=json.dumps(client.full).encode())
        ),
    )
    assert qualify.qualify_main(client, BASE)["qualified"] is True


@pytest.mark.parametrize(
    "mutation", ["truncated", "missing", "duplicate", "expired", "id", "oversized"]
)
def test_artifact_metadata_never_bypasses_authentication(mutation):
    client = Client()
    artifact = {"name": "full", "expired": False, "id": 10, "size_in_bytes": 100}
    page = {"total_count": 1, "artifacts": [artifact]}
    if mutation == "truncated":
        page["total_count"] = 100
    elif mutation == "missing":
        page["artifacts"] = []
    elif mutation == "duplicate":
        page["artifacts"].append(dict(artifact))
    elif mutation == "expired":
        artifact["expired"] = True
    elif mutation == "id":
        artifact["id"] = True
    else:
        artifact["size_in_bytes"] = 1_000_001
    client.json = lambda *a, **k: page
    with pytest.raises(evidence.EvidenceError):
        qualify._artifact(
            client, evidence.Discovery("ci", "b" * 40, "c" * 40, 30, "full", 20)
        )


def test_stale_base_does_not_query_old_green_main():
    client = Client()
    client.tip = "b" * 40
    assert qualify.qualify_main(client, BASE)["qualified"] is False
    assert client.calls == 0


def test_reuse_with_partially_executed_static_jobs_is_rejected():
    client = reused_client()
    next(j for j in client.records if j["name"] == "lint")["conclusion"] = "success"
    assert qualify.qualify_main(client, BASE)["qualified"] is False


def test_attestation_revoked_during_download_cannot_qualify(monkeypatch):
    import json

    client = ReuseClient()

    def download(*a, **k):
        client.status = "failure"
        return SimpleNamespace(stdout=archive(content=json.dumps(client.full).encode()))

    monkeypatch.setattr(qualify.subprocess, "run", download)
    assert qualify.qualify_main(client, BASE)["qualified"] is False


def test_new_identical_tree_cancellation_during_download_cannot_qualify(monkeypatch):
    import json

    client = ReuseClient()

    def download(*a, **k):
        client.newer_cancelled = True
        return SimpleNamespace(stdout=archive(content=json.dumps(client.full).encode()))

    monkeypatch.setattr(qualify.subprocess, "run", download)
    assert qualify.qualify_main(client, BASE)["qualified"] is False


def test_api_error_never_becomes_full_main_qualification():
    client = Client()

    def unavailable(*a, **k):
        raise evidence.EvidenceError("API unavailable")

    client.json = unavailable
    assert qualify.qualify_main(client, BASE)["qualified"] is False
