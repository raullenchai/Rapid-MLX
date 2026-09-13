# SPDX-License-Identifier: Apache-2.0
"""Security and routing contracts for identical-tree CI evidence reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import queue_tree_evidence as evidence

REPO = "owner/repo"
CANDIDATE = "a" * 40
MAIN = "b" * 40
TREE = "c" * 40
TRUSTED = "d" * 40
BRANCH = "mergify/merge-queue/0123456789"


class FakeClient:
    repo = REPO

    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.job_records: dict[int, list[dict[str, Any]]] = {}
        self.trees = {CANDIDATE: TREE, MAIN: TREE}
        control_paths = tuple(
            dict.fromkeys((*evidence.MAC_CONTROL_PATHS, *evidence.CI_CONTROL_PATHS))
        )
        self.blobs = {
            (ref, path): str(index) * 40
            for index, path in enumerate(control_paths, start=1)
            for ref in (CANDIDATE, TRUSTED, MAIN)
        }

    def json(self, endpoint: str, *fields: str, paginate: bool = False) -> Any:
        key = endpoint
        if "/contents/" in endpoint:
            path = endpoint.split("/contents/", 1)[1]
            (ref,) = [
                field.removeprefix("ref=")
                for field in fields
                if field.startswith("ref=")
            ]
            return {"type": "file", "sha": self.blobs[(ref, path)]}
        if endpoint.endswith("/git/commits/" + CANDIDATE):
            return {"tree": {"sha": self.trees[CANDIDATE]}}
        if endpoint.endswith("/git/commits/" + MAIN):
            return {"tree": {"sha": self.trees[MAIN]}}
        value = self.responses[key]
        return [value] if paginate else value

    def commit(self, sha: str) -> dict[str, Any]:
        return {"tree": {"sha": self.trees[sha]}}

    def jobs(self, run_id: int) -> list[dict[str, Any]]:
        return self.job_records[run_id]


def _run(run_id: int, path: str) -> dict[str, Any]:
    return {
        "id": run_id,
        "run_attempt": 1,
        "head_sha": CANDIDATE,
        "head_branch": BRANCH,
        "head_repository": {"full_name": REPO},
        "repository": {"full_name": REPO},
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "path": path,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


def _job(
    run_id: int, job_id: int, name: str, conclusion: str = "success"
) -> dict[str, Any]:
    return {
        "id": job_id,
        "run_attempt": 1,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}/job/{job_id}",
    }


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "journeys.yaml"
    path.write_text(
        "version: 1\njourneys:\n"
        "  - name: first\n    group: chat\n"
        "  - name: second\n    group: images\n"
    )
    return path


def _mac_jobs() -> list[dict[str, Any]]:
    names = list(evidence.REQUIRED_MAC_JOBS) + [
        'gui-golden-flows (chat, ["first"], 1)',
        'gui-golden-flows (images, ["second"], 1)',
    ]
    return [_job(10, 100 + index, name) for index, name in enumerate(names)]


def _ci_jobs() -> list[dict[str, Any]]:
    names = (
        list(evidence.REQUIRED_CI_JOBS)
        + [
            f"test-matrix ({version}, {shard})"
            for version in ("3.10", "3.11", "3.12")
            for shard in (1, 2, 3)
        ]
        + [
            "l1-smoke (first)",
            "l1-smoke (second)",
            "l1-smoke (third)",
            "l1-smoke (fourth)",
            "l1-smoke (fifth)",
        ]
    )
    return [_job(20, 200 + index, name) for index, name in enumerate(names)]


def _configured_client() -> FakeClient:
    client = FakeClient()
    mac_run = _run(10, evidence.MAC_WORKFLOW_PATH)
    ci_run = _run(20, evidence.CI_WORKFLOW_PATH)
    client.responses |= {
        f"repos/{REPO}/actions/runs/10": mac_run,
        f"repos/{REPO}/actions/runs/20": ci_run,
        f"repos/{REPO}/pulls": [
            {
                "number": 99,
                "user": {"login": "mergify[bot]"},
                "head": {
                    "sha": CANDIDATE,
                    "ref": BRANCH,
                    "repo": {"full_name": REPO},
                },
                "base": {"ref": "main", "repo": {"full_name": REPO}},
            }
        ],
        f"repos/{REPO}/actions/workflows/{evidence.MAC_WORKFLOW}/runs": {
            "workflow_runs": [mac_run]
        },
        f"repos/{REPO}/actions/workflows/{evidence.CI_WORKFLOW}/runs": {
            "workflow_runs": [ci_run]
        },
    }
    client.job_records[10] = _mac_jobs()
    client.job_records[20] = _ci_jobs()
    return client


def test_create_requires_complete_mac_matrix(tmp_path: Path):
    client = _configured_client()
    payload = evidence.create_evidence(
        client, "mac", 10, 30, TRUSTED, _manifest(tmp_path)
    )

    assert payload["candidate_tree"] == TREE
    assert payload["attestation_run_id"] == 30
    assert payload["scope"] == "mac"
    assert payload["source"]["id"] == 10


def test_create_accepts_engine_workflow_as_second_completion(tmp_path: Path):
    client = _configured_client()

    payload = evidence.create_evidence(
        client, "ci", 20, 30, TRUSTED, _manifest(tmp_path)
    )

    assert payload["candidate_sha"] == CANDIDATE
    assert payload["scope"] == "ci"
    assert payload["source"]["id"] == 20


def test_create_rejects_candidate_modified_trust_controls(tmp_path: Path):
    client = _configured_client()
    client.blobs[(CANDIDATE, evidence.MAC_WORKFLOW_PATH)] = "f" * 40

    with pytest.raises(evidence.EvidenceError, match="changes its own mac evidence"):
        evidence.create_evidence(client, "mac", 10, 30, TRUSTED, _manifest(tmp_path))


def test_create_rejects_partial_gui_matrix(tmp_path: Path):
    client = _configured_client()
    client.job_records[10] = [
        job for job in client.job_records[10] if "(images," not in job["name"]
    ]

    with pytest.raises(evidence.EvidenceError, match="groups: images"):
        evidence.create_evidence(client, "mac", 10, 30, TRUSTED, _manifest(tmp_path))


def test_create_rejects_partial_engine_matrix(tmp_path: Path):
    client = _configured_client()
    client.job_records[20] = [
        job for job in client.job_records[20] if job["name"] != "test-matrix (3.12, 3)"
    ]

    with pytest.raises(evidence.EvidenceError, match="expected 9 successful jobs"):
        evidence.create_evidence(client, "ci", 20, 30, TRUSTED, _manifest(tmp_path))


def test_create_rejects_older_success_while_newer_run_is_in_progress(
    tmp_path: Path,
):
    client = _configured_client()
    pending = _run(11, evidence.MAC_WORKFLOW_PATH) | {
        "status": "in_progress",
        "conclusion": None,
    }
    client.responses[f"repos/{REPO}/actions/workflows/{evidence.MAC_WORKFLOW}/runs"] = {
        "workflow_runs": [pending, client.responses[f"repos/{REPO}/actions/runs/10"]]
    }

    with pytest.raises(evidence.EvidenceError, match="status 'in_progress'"):
        evidence.create_evidence(client, "mac", 10, 30, TRUSTED, _manifest(tmp_path))


def test_manifest_read_failure_is_a_cache_miss(tmp_path: Path):
    with pytest.raises(evidence.EvidenceError, match="cannot read GUI journey"):
        evidence._manifest_groups(tmp_path / "missing.yaml")


def test_discover_requires_trusted_attestation_target():
    client = _configured_client()
    candidate_run = _run(10, evidence.MAC_WORKFLOW_PATH)
    client.responses[f"repos/{REPO}/actions/workflows/{evidence.MAC_WORKFLOW}/runs"] = {
        "workflow_runs": [candidate_run]
    }
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/30",
        }
    ]
    client.responses[f"repos/{REPO}/actions/runs/30"] = {
        "path": evidence.ATTESTATION_WORKFLOW,
        "event": "workflow_run",
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": REPO},
    }

    probe = evidence.discover(client, "mac", MAIN)
    assert probe == evidence.DiscoveryProbe(
        evidence.Discovery(
            "mac", CANDIDATE, TREE, 30, f"queue-tree-evidence-mac-{CANDIDATE}", 10
        ),
        False,
    )

    client.responses[f"repos/{REPO}/actions/runs/30"]["path"] = (
        ".github/workflows/evil.yml"
    )
    assert evidence.discover(client, "mac", MAIN) == evidence.DiscoveryProbe(
        None, False
    )


def test_discover_uses_independent_ci_attestation():
    client = _configured_client()
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "context": evidence.CONTEXTS["ci"],
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/31",
        }
    ]
    client.responses[f"repos/{REPO}/actions/runs/31"] = {
        "path": evidence.ATTESTATION_WORKFLOW,
        "event": "workflow_run",
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": REPO},
    }

    probe = evidence.discover(client, "ci", MAIN)

    assert probe == evidence.DiscoveryProbe(
        evidence.Discovery(
            "ci", CANDIDATE, TREE, 31, f"queue-tree-evidence-ci-{CANDIDATE}", 20
        ),
        False,
    )


def test_discover_prefers_newest_valid_attestation_status():
    client = _configured_client()
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "id": 1,
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/30",
        },
        {
            "id": 2,
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/32",
        },
    ]
    for run_id in (30, 32):
        client.responses[f"repos/{REPO}/actions/runs/{run_id}"] = {
            "path": evidence.ATTESTATION_WORKFLOW,
            "event": "workflow_run",
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": REPO},
        }

    probe = evidence.discover(client, "mac", MAIN)

    assert probe.evidence is not None
    assert probe.evidence.attestation_run_id == 32


def test_discover_rejects_older_status_success_after_newer_failure():
    client = _configured_client()
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "id": 1,
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/30",
        },
        {
            "id": 2,
            "context": evidence.MAC_CONTEXT,
            "state": "failure",
            "target_url": f"https://github.com/{REPO}/actions/runs/32",
        },
    ]

    assert evidence.discover(client, "mac", MAIN) == evidence.DiscoveryProbe(
        None, False
    )


def test_discover_waits_for_newest_pending_status_without_using_older_success():
    client = _configured_client()
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "id": 1,
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/30",
        },
        {
            "id": 2,
            "context": evidence.MAC_CONTEXT,
            "state": "pending",
            "target_url": f"https://github.com/{REPO}/actions/runs/32",
        },
    ]

    assert evidence.discover(client, "mac", MAIN) == evidence.DiscoveryProbe(None, True)


def test_discover_does_not_fall_back_when_newest_success_target_is_untrusted():
    client = _configured_client()
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "id": 1,
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/30",
        },
        {
            "id": 2,
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": "https://github.com/attacker/repo/actions/runs/32",
        },
    ]

    assert evidence.discover(client, "mac", MAIN) == evidence.DiscoveryProbe(
        None, False
    )


def test_discover_rejects_older_success_after_newer_failure():
    client = _configured_client()
    success = _run(10, evidence.MAC_WORKFLOW_PATH)
    failure = _run(11, evidence.MAC_WORKFLOW_PATH) | {"conclusion": "failure"}
    client.responses[f"repos/{REPO}/actions/workflows/{evidence.MAC_WORKFLOW}/runs"] = {
        "workflow_runs": [failure, success]
    }
    client.responses[f"repos/{REPO}/commits/{CANDIDATE}/statuses"] = [
        {
            "context": evidence.MAC_CONTEXT,
            "state": "success",
            "target_url": f"https://github.com/{REPO}/actions/runs/30",
        }
    ]

    assert evidence.discover(client, "mac", MAIN) == evidence.DiscoveryProbe(
        None, False
    )


def test_discover_rejects_older_success_while_newer_run_is_in_progress():
    client = _configured_client()
    success = _run(10, evidence.MAC_WORKFLOW_PATH)
    pending = _run(11, evidence.MAC_WORKFLOW_PATH) | {
        "status": "in_progress",
        "conclusion": None,
    }
    client.responses[f"repos/{REPO}/actions/workflows/{evidence.MAC_WORKFLOW}/runs"] = {
        "workflow_runs": [pending, success]
    }

    assert evidence.discover(client, "mac", MAIN) == evidence.DiscoveryProbe(
        None, False
    )


def test_validate_rejects_source_rerun_after_attestation(tmp_path: Path):
    client = _configured_client()
    payload = evidence.create_evidence(
        client, "mac", 10, 30, TRUSTED, _manifest(tmp_path)
    )
    client.responses[f"repos/{REPO}/actions/runs/10"]["run_attempt"] = 2

    with pytest.raises(evidence.EvidenceError, match="no longer matches evidence"):
        evidence.validate_evidence(
            client,
            MAIN,
            evidence.Discovery("mac", CANDIDATE, TREE, 30, "unused", 10),
            payload,
            _manifest(tmp_path),
        )


def test_validate_rejects_attestation_for_older_successful_run(tmp_path: Path):
    client = _configured_client()
    payload = evidence.create_evidence(
        client, "mac", 10, 30, TRUSTED, _manifest(tmp_path)
    )

    with pytest.raises(evidence.EvidenceError, match="not the latest"):
        evidence.validate_evidence(
            client,
            MAIN,
            evidence.Discovery("mac", CANDIDATE, TREE, 30, "unused", 11),
            payload,
            _manifest(tmp_path),
        )


def test_validate_rejects_malformed_recorded_job_without_type_error(tmp_path: Path):
    client = _configured_client()
    payload = evidence.create_evidence(
        client, "mac", 10, 30, TRUSTED, _manifest(tmp_path)
    )
    payload["source"]["jobs"][0]["id"] = "not-an-integer"

    with pytest.raises(evidence.EvidenceError, match="invalid 'id'"):
        evidence.validate_evidence(
            client,
            MAIN,
            evidence.Discovery("mac", CANDIDATE, TREE, 30, "unused", 10),
            payload,
            _manifest(tmp_path),
        )


def test_workflows_fail_closed_and_never_execute_candidate_code():
    root = Path(__file__).resolve().parent.parent
    attestation = yaml.load(
        (root / evidence.ATTESTATION_WORKFLOW).read_text(), Loader=yaml.BaseLoader
    )
    desktop = yaml.safe_load((root / ".github/workflows/rapid-mac-ci.yml").read_text())
    engine = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())

    assert attestation["on"]["workflow_run"]["workflows"] == [
        "rapid-mac CI",
        "CI",
    ]
    checkout = attestation["jobs"]["attest"]["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert checkout["with"]["persist-credentials"] == "false"
    create_step = attestation["jobs"]["attest"]["steps"][1]
    assert '--trusted-ref "$GITHUB_SHA"' in create_step["run"]
    assert '--scope "$scope"' in create_step["run"]
    assert '"rapid-mac CI") scope=mac' in create_step["run"]
    assert '"CI") scope=ci' in create_step["run"]
    assert attestation["permissions"] == {
        "actions": "read",
        "contents": "read",
        "statuses": "write",
    }
    assert evidence.ATTESTATION_WORKFLOW in evidence.MAC_CONTROL_PATHS
    assert "scripts/queue_tree_evidence.py" in evidence.MAC_CONTROL_PATHS

    jobs = desktop["jobs"]
    assert "github.event_name == 'push'" in str(jobs["queue-tree-evidence"]["if"])
    assert "refs/heads/main" in str(jobs["queue-tree-evidence"]["if"])
    assert jobs["queue-tree-evidence"]["permissions"] == {
        "actions": "read",
        "contents": "read",
        "statuses": "read",
    }
    for name in (
        "accessibility-identifiers",
        "accessibility-identifier-tests",
        "gui-harness-contracts",
        "build",
        "gui-app-build",
        "gui-golden-flows",
        "desktop-tests",
    ):
        assert "always()" in str(jobs[name]["if"])
    assert str(jobs["changes"]["if"]) == "always()"
    for name in ("build", "gui-app-build", "gui-golden-flows"):
        assert "reuse_mac != 'true'" in str(jobs[name]["if"])
        assert "always()" in str(jobs[name]["if"])
        assert "needs.changes.result == 'success'" in str(jobs[name]["if"])
        assert "queue-tree-evidence" in jobs[name]["needs"]
    aggregate = jobs["desktop-tests"]
    assert "queue-tree-evidence" in aggregate["needs"]
    script = next(
        step["run"]
        for step in aggregate["steps"]
        if step["name"] == "Check desktop results"
    )
    assert 'if [ "$REUSE_MAC" = true ]' in script
    assert '"$BUILD" != skipped' in script
    download = next(
        step
        for step in jobs["queue-tree-evidence"]["steps"]
        if step.get("name") == "Download trusted evidence artifact"
    )
    validate = next(
        step
        for step in jobs["queue-tree-evidence"]["steps"]
        if step.get("name") == "Revalidate source runs and jobs"
    )
    assert download["continue-on-error"] is True
    assert str(validate["if"]).startswith("always()")
    assert "--scope mac" in validate["run"]
    assert (
        '--source-run-id "${{ steps.discover.outputs.source_run_id }}"'
        in validate["run"]
    )

    engine_jobs = engine["jobs"]
    assert engine_jobs["queue-tree-evidence"]["permissions"] == {
        "actions": "read",
        "contents": "read",
        "statuses": "read",
    }
    engine_evidence = engine_jobs["queue-tree-evidence"]
    assert "github.event_name == 'push'" in str(engine_evidence["if"])
    assert "refs/heads/main" in str(engine_evidence["if"])
    assert engine_jobs["changes"]["needs"] == "queue-tree-evidence"
    assert str(engine_jobs["changes"]["if"]) == "always()"
    for name in (
        "lint",
        "engine-contracts",
        "type-check",
        "test-matrix",
        "test-apple-silicon",
        "l1-smoke",
    ):
        assert "always()" in str(engine_jobs[name]["if"])
        assert "reuse_ci != 'true'" in str(engine_jobs[name]["if"])
    for name in ("merge-lane-no-mac", "merge-lane-mac", "mlx-bound-guard"):
        assert "always()" in str(engine_jobs[name]["if"])
        assert "needs.changes.result == 'success'" in str(engine_jobs[name]["if"])
    aggregate_script = next(
        step["run"]
        for step in engine_jobs["tests"]["steps"]
        if step["name"] == "Check test results"
    )
    assert 'if [ "${{ needs.changes.outputs.reuse_ci }}" = true ]' in aggregate_script
    engine_download = next(
        step
        for step in engine_evidence["steps"]
        if step.get("name") == "Download trusted evidence artifact"
    )
    engine_validate = next(
        step
        for step in engine_evidence["steps"]
        if step.get("name") == "Revalidate source runs and jobs"
    )
    assert engine_download["continue-on-error"] is True
    assert str(engine_validate["if"]).startswith("always()")
    assert "--scope ci" in engine_validate["run"]
    assert (
        '--source-run-id "${{ steps.discover.outputs.source_run_id }}"'
        in engine_validate["run"]
    )


def test_mergify_candidate_selects_complete_gui_inventory():
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parent.parent
            / ".github/workflows/rapid-mac-ci.yml"
        ).read_text()
    )
    script = next(
        step["run"]
        for step in workflow["jobs"]["changes"]["steps"]
        if step.get("name") == "Classify desktop lane"
    )
    assert 'if [ "$is_mergify" = true ]' in script
    assert "python3 scripts/select_gui_flows.py --github-output" in script
    assert "--paths-file /tmp/changed-paths" in script
