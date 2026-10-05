"""Candidate shadow observations cannot qualify reduced or complete CI."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts import ci_candidate_shadow as shadow
from scripts import queue_tree_evidence as evidence

REPO = "owner/repo"
BASE, HEAD, TREE, SOURCE = (char * 40 for char in "abcd")
BRANCH = "mergify/merge-queue/0123456789"


class Client:
    repo = REPO

    def __init__(self):
        self.tip = BASE
        self.tip_calls = 0
        self.advance_tip = False
        self.rerun = False
        self.run_calls = 0
        self.pull = {
            "user": {"login": "mergify[bot]"},
            "head": {"sha": HEAD, "ref": BRANCH, "repo": {"full_name": REPO}},
            "base": {"sha": BASE, "ref": "main", "repo": {"full_name": REPO}},
        }
        self.candidate = {
            "parents": [{"sha": BASE}, {"sha": SOURCE}],
            "tree": {"sha": TREE},
        }
        self.diff = {
            "merge_base_commit": {"sha": BASE},
            "files": [{"filename": "tests/test_cli_cheetah_banner.py"}],
        }
        self.runs = [
            {
                "id": 100,
                "run_attempt": 1,
                "head_sha": BASE,
                "head_branch": "main",
                "head_repository": {"full_name": REPO},
                "event": "push",
                "path": evidence.CI_WORKFLOW_PATH,
                "status": "completed",
                "conclusion": "success",
            }
        ]
        names = [
            *shadow.MAIN_JOBS,
            *(
                name
                for group in evidence.REQUIRED_CI_MATRIX_JOBS.values()
                for name in group
            ),
        ]
        self.records = [
            {
                "id": i,
                "name": name,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
            for i, name in enumerate(names)
        ]

    def json(self, endpoint, *fields, **kwargs):
        if endpoint.endswith("/git/ref/heads/main"):
            self.tip_calls += 1
            return {
                "object": {
                    "sha": HEAD if self.advance_tip and self.tip_calls > 1 else self.tip
                }
            }
        if endpoint.endswith("/pulls/123"):
            return copy.deepcopy(self.pull)
        if "/compare/" in endpoint:
            return copy.deepcopy(self.diff)
        if endpoint.endswith("/actions/workflows/ci.yml/runs"):
            assert (
                f"head_sha={BASE}" in fields
                and "event=push" in fields
                and "branch=main" in fields
            )
            self.run_calls += 1
            runs = copy.deepcopy(self.runs)
            if self.rerun and self.run_calls > 1:
                runs[0]["run_attempt"] = 2
            return {"workflow_runs": runs}
        raise AssertionError(endpoint)

    def commit(self, sha):
        assert sha == HEAD
        return copy.deepcopy(self.candidate)

    def jobs(self, run_id):
        assert run_id == 100
        return copy.deepcopy(self.records)


def full():
    return {
        "schema": evidence.SCHEMAS["ci"],
        "scope": "ci",
        "repository": REPO,
        "candidate_sha": HEAD,
        "candidate_ref": BRANCH,
        "candidate_tree": TREE,
        "candidate_pr": 123,
        "attestation_run_id": 900,
    }


def test_real_two_parent_shape_with_full_green_main_is_advisory_only():
    result = shadow.inspect_candidate(Client(), full(), 900)
    assert result["route"] == "mapped-eligible-shadow"
    assert result["base_run"] == 100 and result["base_attempt"] == 1
    assert result["mapped_tests"]
    assert result["authorizes_reduced_ci"] is False
    assert result["execution_proof_checked"] is False
    assert result["schema"] not in evidence.SCHEMAS.values()
    assert result["scope"] not in evidence.SCHEMAS


def test_post_merge_snapshot_retains_paths_without_qualifying_old_base():
    client = Client()
    client.tip = SOURCE
    result = shadow.inspect_candidate(client, full(), 900)
    assert result["route"] == "full"
    assert result["paths"] == ["tests/test_cli_cheetah_banner.py"]
    assert result["mapped_tests"]
    assert result["reason"] == "candidate base is not current main"
    assert result["authorizes_reduced_ci"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "rapid-mlx/source-canary-unit/v1"),
        ("scope", "mac"),
        ("repository", "fork/repo"),
        ("candidate_sha", "bad"),
    ],
)
def test_wrong_full_producer_identity_falls_back(field, value):
    record = full()
    record[field] = value
    assert shadow.inspect_candidate(Client(), record, 900)["route"] == "full"


@pytest.mark.parametrize(
    "change",
    [
        "author",
        "head",
        "branch",
        "fork",
        "base_branch",
        "base_repo",
        "first_parent",
        "tree",
        "tip",
    ],
)
def test_candidate_base_and_repository_identity_fail_closed(change):
    client = Client()
    if change == "author":
        client.pull["user"]["login"] = "contributor"
    elif change == "head":
        client.pull["head"]["sha"] = SOURCE
    elif change == "branch":
        client.pull["head"]["ref"] = "source"
    elif change == "fork":
        client.pull["head"]["repo"]["full_name"] = "fork/repo"
    elif change == "base_branch":
        client.pull["base"]["ref"] = "dev"
    elif change == "base_repo":
        client.pull["base"]["repo"]["full_name"] = "fork/repo"
    elif change == "first_parent":
        client.candidate["parents"][0]["sha"] = SOURCE
    elif change == "tree":
        client.candidate["tree"]["sha"] = SOURCE
    else:
        client.tip = SOURCE
    assert shadow.inspect_candidate(client, full(), 900)["route"] == "full"


@pytest.mark.parametrize(
    "files",
    [
        [],
        [{"filename": "rapid_mlx/server.py"}],
        [
            {"filename": "tests/test_cli_cheetah_banner.py"},
            {"filename": "tests/conftest.py"},
        ],
        [
            {
                "filename": "tests/test_cli_cheetah_banner.py",
                "previous_filename": "tests/test_security.py",
            }
        ],
        [{"filename": "tests/test_cli_cheetah_banner.py"}] * 300,
        [{}],
    ],
)
def test_combined_unknown_mixed_control_rename_and_truncated_diff_stay_full(files):
    client = Client()
    client.diff["files"] = files
    assert shadow.inspect_candidate(client, full(), 900)["route"] == "full"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", "cancelled"),
        ("head_sha", SOURCE),
        ("head_branch", "dev"),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
        ("head_repository", {"full_name": "fork/repo"}),
    ],
)
def test_latest_non_green_or_wrong_main_run_cannot_use_older_success(field, value):
    client = Client()
    newer = copy.deepcopy(client.runs[0])
    newer.update(id=101)
    newer[field] = value
    client.runs.append(newer)
    assert shadow.inspect_candidate(client, full(), 900)["route"] == "full"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "wrong_cpu",
        "wrong_model",
        "skipped",
        "failed",
        "prior_attempt",
        "reused",
        "empty_runs",
        "different_mergebase",
        "malformed_api",
        "api_error",
    ],
)
def test_base_full_execution_and_api_errors_fail_closed(change):
    client = Client()
    if change == "missing":
        client.records.pop()
    elif change == "duplicate":
        client.records.append(copy.deepcopy(client.records[-1]))
    elif change == "wrong_cpu":
        next(j for j in client.records if j["name"].startswith("test-matrix"))[
            "name"
        ] = "test-matrix (3.13, 1)"
    elif change == "wrong_model":
        client.records[-1]["name"] = "l1-smoke (unknown, 0)"
    elif change in {"skipped", "failed"}:
        client.records[-1]["conclusion"] = (
            "skipped" if change == "skipped" else "failure"
        )
    elif change == "prior_attempt":
        client.records[-1]["run_attempt"] = 0
    elif change == "reused":
        for record in client.records:
            record["conclusion"] = "skipped"
    elif change == "empty_runs":
        client.runs = []
    elif change == "different_mergebase":
        client.diff["merge_base_commit"]["sha"] = SOURCE
    elif change == "malformed_api":
        client.diff = {}
    else:

        def fail(*args, **kwargs):
            raise evidence.EvidenceError("API unavailable")

        client.json = fail
    assert shadow.inspect_candidate(client, full(), 900)["route"] == "full"


@pytest.mark.parametrize("change", ["advance_tip", "rerun"])
def test_live_recheck_rejects_main_or_attempt_changes(change):
    client = Client()
    setattr(client, change, True)
    assert shadow.inspect_candidate(client, full(), 900)["route"] == "full"


def test_wrong_attestation_run_cannot_use_full_artifact():
    assert shadow.inspect_candidate(Client(), full(), 901)["route"] == "full"


def test_privileged_observer_does_not_delay_or_mutate_existing_full_proof():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/candidate-routing-shadow.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["workflow_run"] == {
        "workflows": ["Queue tree attestation"],
        "types": ["completed"],
    }
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    steps = workflow["jobs"]["observe"]["steps"]
    checkout = steps[0]["with"]
    assert (
        checkout["ref"] == "${{ github.sha }}"
        and checkout["persist-credentials"] is False
    )
    download = next(
        s for s in steps if s.get("name") == "Download only the trusted full artifact"
    )
    assert download["with"]["run-id"] == "${{ github.event.workflow_run.id }}"
    assert download["with"]["repository"] == "${{ github.repository }}"
    observation = next(
        s
        for s in steps
        if s.get("name") == "Observe candidate path and base qualification"
    )
    assert '--attestation-run-id "$ATTESTATION_RUN"' in observation["run"]
    assert (
        observation["env"]["ATTESTATION_RUN"] == "${{ github.event.workflow_run.id }}"
    )
    assert all("statuses/" not in s.get("run", "") for s in steps)
    artifact = next(s for s in steps if s.get("name") == "Upload advisory snapshot")
    assert artifact["with"]["name"].startswith("candidate-route-shadow-")
