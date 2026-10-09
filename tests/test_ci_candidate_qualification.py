"""Candidate qualification is scoped, current and disabled for mapped by default."""

from __future__ import annotations

import pytest

from scripts import ci_candidate_qualification as qualify
from scripts import queue_tree_evidence as evidence
from tests.test_queue_tree_evidence import (
    CANDIDATE,
    MAIN,
    REPO,
    TREE,
    TRUSTED,
    _configured_client,
)


def fixture(monkeypatch, mapped=False):
    client = _configured_client()
    client.responses[f"repos/{REPO}/pulls"][0]["base"]["sha"] = MAIN
    client.commit = lambda sha: {
        "sha": sha,
        "message": "Merge of #77",
        "tree": {"sha": client.trees.get(sha, TREE)},
        "parents": [{"sha": MAIN}, {"sha": TRUSTED}],
    }
    client.responses[f"repos/{REPO}/pulls/77"] = {
        "head": {"sha": TRUSTED, "repo": {"full_name": REPO}}
    }
    monkeypatch.setattr(
        qualify,
        "_queue_metadata",
        lambda *_: {"checking_base_sha": MAIN, "pull_requests": [{"number": 77}]},
    )
    for path in qualify.CONTROL_PATHS:
        client.blobs[CANDIDATE, path] = client.blobs[TRUSTED, path] = "f" * 40
    client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"] = {
        "merge_base_commit": {"sha": MAIN},
        "files": [{"filename": "tests/test_cli_cheetah_banner.py"}],
    }
    proof = None
    if mapped:
        names = [
            *qualify.MAPPED_JOBS,
            "test-matrix",
            "l1-smoke",
            "test-apple-silicon",
            "linux-coverage",
            "changed-lines-coverage",
            "source-canary-unit",
        ]
        client.job_records[20] = [
            {
                "id": i,
                "run_attempt": 1,
                "name": n,
                "status": "completed",
                "conclusion": "success" if n in qualify.MAPPED_JOBS else "skipped",
            }
            for i, n in enumerate(names)
        ]
        job = next(
            j for j in client.job_records[20] if j["name"] == "candidate-canary-unit"
        )
        job["steps"] = [
            {"name": n, "status": "completed", "conclusion": "success"}
            for n in (
                "Run complete mapped regressions",
                "Enforce mandatory changed-line coverage",
            )
        ]
        tests = list(qualify.source_canary_tests({"tests/test_cli_cheetah_banner.py"}))
        nodes = [test + "::test_one" for test in tests]
        baseline = {
            "schema": "rapid-mlx/mapped-execution/v1",
            "collect_only": True,
            "exitstatus": 0,
            "deselected": 0,
            "collection_errors": 0,
            "nodes": nodes,
            "reports": [],
        }
        executed = dict(
            baseline,
            collect_only=False,
            reports=[
                {"node": n, "stage": s, "outcome": "passed", "xfail": False}
                for n in nodes
                for s in ("setup", "call", "teardown")
            ],
        )
        proof = {
            "schema": qualify.INPUT_SCHEMA,
            "scope": "candidate-mapped-only",
            "head": CANDIDATE,
            "base": MAIN,
            "source_run_id": 20,
            "source_attempt": 1,
            "tested_sha": CANDIDATE,
            "tests": tests,
            "baseline": baseline,
            "executed": executed,
        }
    monkeypatch.setattr(
        qualify,
        "qualify_main",
        lambda *a: {"qualified": True, "authorizes_reduced_ci": False},
    )
    return client, proof


def test_ordered_two_source_identity(monkeypatch):
    client, _ = fixture(monkeypatch)
    middle = "e" * 40
    source_two = "1" * 40
    client.responses[f"repos/{REPO}/pulls/88"] = {
        "head": {"sha": source_two, "repo": {"full_name": REPO}}
    }
    commits = {
        CANDIDATE: {
            "sha": CANDIDATE,
            "message": "Merge of #88",
            "tree": {"sha": TREE},
            "parents": [{"sha": middle}, {"sha": source_two}],
        },
        middle: {
            "sha": middle,
            "message": "Merge of #77",
            "tree": {"sha": "2" * 40},
            "parents": [{"sha": MAIN}, {"sha": TRUSTED}],
        },
    }
    client.commit = commits.__getitem__
    monkeypatch.setattr(
        qualify,
        "_queue_metadata",
        lambda *_: {
            "checking_base_sha": MAIN,
            "pull_requests": [{"number": 77}, {"number": 88}],
        },
    )
    result = qualify.qualify_candidate(client, 20, TRUSTED)
    assert result["qualified"]
    assert result["source_pull_requests"] == [
        {"number": 77, "head_sha": TRUSTED},
        {"number": 88, "head_sha": source_two},
    ]


def test_pull_commit_message_shape_cannot_substitute_for_git_database_message(
    monkeypatch,
):
    client, _ = fixture(monkeypatch)
    client.commit = lambda sha: {
        "sha": sha,
        "tree": {"sha": client.trees.get(sha, TREE)},
        "commit": {"message": "Merge of #77"},
        "parents": [{"sha": MAIN}, {"sha": TRUSTED}],
    }

    result = qualify.qualify_candidate(client, 20, TRUSTED)

    assert not result["qualified"]
    assert result["reason"] == "candidate has malformed integration lineage"


def test_actual_queue_metadata_archive_is_exact_attempt(monkeypatch):
    import io
    import json
    import zipfile
    from types import SimpleNamespace

    client, _ = fixture(monkeypatch)
    monkeypatch.undo()
    client.gh = "gh"
    run = client.responses[f"repos/{REPO}/actions/runs/20"]
    name = f"candidate-queue-identity-{CANDIDATE}-20-1"
    client.responses[f"repos/{REPO}/actions/runs/20/artifacts"] = {
        "total_count": 1,
        "artifacts": [
            {
                "id": 501,
                "name": name,
                "expired": False,
                "size_in_bytes": 100,
                "workflow_run": {"id": 20, "head_sha": CANDIDATE},
            }
        ],
    }
    payload = {"checking_base_sha": MAIN, "pull_requests": [{"number": 77}]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("candidate-queue-identity.json", json.dumps(payload))
    monkeypatch.setattr(
        qualify.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=buffer.getvalue()),
    )
    assert qualify._queue_metadata(client, run) == payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("listing", "listing is malformed"),
        ("name", "missing or duplicate"),
        ("identity", "artifact identity"),
        ("download", "exceeds bound"),
        ("member", "unexpected queue metadata"),
        ("payload", "not an object"),
    ],
)
def test_queue_metadata_archive_fails_closed(monkeypatch, change, message):
    import io
    import json
    import zipfile
    from types import SimpleNamespace

    client, _ = fixture(monkeypatch)
    monkeypatch.undo()
    client.gh = "gh"
    run = client.responses[f"repos/{REPO}/actions/runs/20"]
    artifact = {
        "id": 501,
        "name": f"candidate-queue-identity-{CANDIDATE}-20-1",
        "expired": False,
        "size_in_bytes": 100,
        "workflow_run": {"id": 20, "head_sha": CANDIDATE},
    }
    page = {"total_count": 1, "artifacts": [artifact]}
    client.responses[f"repos/{REPO}/actions/runs/20/artifacts"] = page
    payload = (
        []
        if change == "payload"
        else {
            "checking_base_sha": MAIN,
            "pull_requests": [{"number": 77}],
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "other.json" if change == "member" else "candidate-queue-identity.json",
            json.dumps(payload),
        )
    raw = buffer.getvalue()
    if change == "listing":
        page["total_count"] = 2
    elif change == "name":
        artifact["name"] = "other"
    elif change == "identity":
        artifact["workflow_run"]["head_sha"] = MAIN
    elif change == "download":
        raw = b"x" * (qualify.QUEUE_ARTIFACT_MAX_BYTES + 1)
    monkeypatch.setattr(
        qualify.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=raw)
    )
    with pytest.raises(evidence.EvidenceError, match=message):
        qualify._queue_metadata(client, run)


def test_workflow_publishes_exact_attempt_queue_identity():
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["changes"]["steps"]
    capture = next(step for step in steps if step.get("id") == "candidate-queue-info")
    upload = next(
        step
        for step in steps
        if step.get("name") == "Upload trusted candidate identity"
    )
    assert capture["run"] == "mergify ci queue-info"
    assert (
        upload["with"]["name"]
        == "candidate-queue-identity-${{ github.event.pull_request.head.sha }}-${{ github.run_id }}-${{ github.run_attempt }}"
    )


def test_mandatory_linux_coverage_enrolls_candidate_controllers():
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["test-matrix"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Run unit tests (no MLX required)"
    )
    required = {
        "--cov=scripts.ci_candidate_qualification",
        "--cov=scripts.ci_candidate_consumer",
        "--cov=scripts.ci_candidate_admission",
        "--cov=scripts.ci_candidate_execution",
        "--cov=scripts.ci_candidate_rollout",
        "--cov=scripts.sidecar_macho_inventory",
        "--cov=scripts.pr_validate.steps.stress_e2e_bench",
    }
    assert required <= set(run.split())
    assert "--cov=rapid_mlx" in run
    assert (
        Path(".github/workflows/ci.yml")
        .read_text()
        .count("--cov=scripts.pr_validate.steps.stress_e2e_bench")
        == 1
    )
    assert (
        Path(".github/workflows/ci.yml")
        .read_text()
        .count("--cov=scripts.sidecar_macho_inventory")
        == 1
    )
    apple_steps = workflow["jobs"]["test-apple-silicon"]["steps"]
    assert all(
        "--cov=scripts.pr_validate.steps.stress_e2e_bench" not in step.get("run", "")
        for step in apple_steps
    )
    assert all(
        "--cov=scripts.sidecar_macho_inventory" not in step.get("run", "")
        for step in apple_steps
    )


@pytest.mark.parametrize(
    "change", ["order", "duplicate", "head", "base", "three", "invalid"]
)
def test_batch_identity_fails_closed(monkeypatch, change):
    client, _ = fixture(monkeypatch)
    metadata = {"checking_base_sha": MAIN, "pull_requests": [{"number": 77}]}
    if change == "order":
        metadata["pull_requests"] = [{"number": 88}]
    elif change == "duplicate":
        metadata["pull_requests"] *= 2
    elif change == "head":
        client.responses[f"repos/{REPO}/pulls/77"]["head"]["sha"] = "3" * 40
    elif change == "base":
        metadata["checking_base_sha"] = "4" * 40
    elif change == "invalid":
        metadata["pull_requests"] = [{"number": False}]
    else:
        metadata["pull_requests"] *= 3
    monkeypatch.setattr(qualify, "_queue_metadata", lambda *_: metadata)
    assert not qualify.qualify_candidate(client, 20, TRUSTED)["qualified"]


def test_full_repair_does_not_require_green_main(monkeypatch):
    client, _ = fixture(monkeypatch)
    monkeypatch.setattr(
        qualify, "qualify_main", lambda *a: pytest.fail("full repair consulted main")
    )
    result = qualify.qualify_candidate(client, 20, TRUSTED)
    assert result["qualified"] and result["kind"] == "full"
    assert not result["authorizes_reduced_ci"]
    assert result["schema"] not in evidence.SCHEMAS.values()


def test_mapped_producer_is_default_off(monkeypatch):
    client, proof = fixture(monkeypatch, True)
    assert not qualify.qualify_candidate(client, 20, TRUSTED, mapped_input=proof)[
        "qualified"
    ]


def test_mapped_requires_execution_and_distinct_namespace(monkeypatch):
    client, proof = fixture(monkeypatch, True)
    result = qualify.qualify_candidate(
        client, 20, TRUSTED, mapped_enabled=True, mapped_input=proof
    )
    assert (
        result["qualified"]
        and result["kind"] == "mapped"
        and result["authorizes_reduced_ci"]
    )
    assert result["schema"] not in evidence.SCHEMAS.values()


@pytest.mark.parametrize(
    "change",
    [
        "fork",
        "author",
        "base",
        "source-schema",
        "attempt",
        "tree",
        "skipped-test",
        "coverage-step",
        "critical",
        "rename",
        "controls",
        "red-main",
        "cancelled",
        "static-job",
    ],
)
def test_mapped_candidate_negative_controls(monkeypatch, change):
    client, proof = fixture(monkeypatch, True)
    pull = client.responses[f"repos/{REPO}/pulls"][0]
    run = client.responses[f"repos/{REPO}/actions/runs/20"]
    jobs = client.job_records[20]
    diff = client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]
    if change == "fork":
        run["head_repository"] = {"full_name": "fork/repo"}
    elif change == "author":
        pull["user"]["login"] = "human"
    elif change == "base":
        pull["base"]["ref"] = "other"
    elif change == "source-schema":
        proof["schema"] = "rapid-mlx/source-canary-unit/v1"
    elif change == "attempt":
        proof["source_attempt"] = 0
    elif change == "tree":
        proof["tested_sha"] = "bad"
    elif change == "skipped-test":
        proof["executed"]["reports"][0]["outcome"] = "skipped"
    elif change == "coverage-step":
        next(j for j in jobs if j["name"] == "candidate-canary-unit")["steps"].pop()
    elif change == "critical":
        diff["files"].append({"filename": "rapid_mlx/server.py"})
    elif change == "rename":
        diff["files"][0]["previous_filename"] = "rapid_mlx/server.py"
    elif change == "controls":
        client.blobs[CANDIDATE, evidence.CI_WORKFLOW_PATH] = "e" * 40
    elif change == "red-main":
        monkeypatch.setattr(qualify, "qualify_main", lambda *a: {"qualified": False})
    elif change == "cancelled":
        run["conclusion"] = "cancelled"
    else:
        next(j for j in jobs if j["name"] == "lint")["conclusion"] = "failure"
    assert not qualify.qualify_candidate(
        client, 20, TRUSTED, mapped_enabled=True, mapped_input=proof
    )["qualified"]


def test_trusted_producer_workflow_has_separate_unconsumed_namespace():
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/candidate-qualification.yml").read_text()
    )
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "statuses": "write",
    }
    steps = workflow["jobs"]["qualify"]["steps"]
    assert steps[0]["with"]["ref"] == "${{ github.sha }}"
    assert steps[0]["with"]["persist-credentials"] is False
    upload = next(
        i
        for i, s in enumerate(steps)
        if s.get("name") == "Upload separate qualification"
    )
    publish = next(
        i for i, s in enumerate(steps) if s.get("name") == "Index qualified candidate"
    )
    assert upload < publish
    assert "candidate-qualification/ci" in steps[publish]["run"]
    assert "queue-tree-evidence/" not in steps[publish]["run"]
    for step in steps:
        if "uses" in step:
            assert len(step["uses"].split("@")[1]) == 40
    queue = Path(".mergify.yml").read_text()
    assert "candidate-qualification/ci" not in queue
    code = (
        Path("scripts/ci_candidate_qualification.py")
        .read_text()
        .split("def main() -> None:")[1]
    )
    assert "--mapped" not in code and "mapped_enabled=True" not in code


@pytest.mark.parametrize(
    "change",
    [
        "missing-run",
        "missing-pr",
        "parent",
        "truncated",
        "new-trigger",
        "extra-matrix",
        "tested-tree",
    ],
)
def test_additional_identity_and_scope_rejections(monkeypatch, change):
    client, proof = fixture(monkeypatch, True)
    if change == "missing-run":
        client.responses[f"repos/{REPO}/actions/workflows/ci.yml/runs"][
            "workflow_runs"
        ] = []
    elif change == "missing-pr":
        client.responses[f"repos/{REPO}/pulls"] = []
    elif change == "parent":
        client.commit = lambda sha: {
            "tree": {"sha": TREE},
            "parents": [{"sha": TRUSTED}],
        }
    elif change == "truncated":
        client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]["files"] *= 300
    elif change == "new-trigger":
        newer = dict(client.responses[f"repos/{REPO}/actions/runs/20"], id=21)
        client.responses[f"repos/{REPO}/actions/workflows/ci.yml/runs"][
            "workflow_runs"
        ].append(newer)
    elif change == "extra-matrix":
        client.job_records[20].append(
            {
                "id": 999,
                "name": "test-matrix (3.11, 1)",
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "skipped",
            }
        )
    else:
        proof["tested_sha"] = "e" * 40
        client.trees[proof["tested_sha"]] = "f" * 40
    assert not qualify.qualify_candidate(
        client, 20, TRUSTED, mapped_enabled=True, mapped_input=proof
    )["qualified"]


@pytest.mark.parametrize("race", ["attempt", "tree", "main"])
def test_prepublication_races_reject_qualification(monkeypatch, race):
    client, proof = fixture(monkeypatch, race == "main")
    if race == "attempt":
        original = client.json
        count = 0

        def changed(endpoint, *args, **kwargs):
            nonlocal count
            value = original(endpoint, *args, **kwargs)
            if endpoint.endswith("/actions/workflows/ci.yml/runs"):
                count += 1
                if count > 1:
                    import copy

                    value = copy.deepcopy(value)
                    value[0]["workflow_runs"][0]["run_attempt"] = 2
            return value

        client.json = changed
    elif race == "tree":
        count = 0
        original = client.commit

        def changed_tree(sha):
            nonlocal count
            count += 1
            result = original(sha)
            if count > 1:
                result["tree"]["sha"] = "e" * 40
            return result

        client.commit = changed_tree
    else:
        calls = iter([{"qualified": True}, {"qualified": False}])
        monkeypatch.setattr(qualify, "qualify_main", lambda *a: next(calls))
    assert not qualify.qualify_candidate(
        client, 20, TRUSTED, mapped_enabled=True, mapped_input=proof
    )["qualified"]


def test_cli_full_only_actual_entrypoint(monkeypatch, tmp_path):
    import json
    import runpy
    import sys

    client, _ = fixture(monkeypatch)
    monkeypatch.setattr(evidence, "GitHubClient", lambda repo: client)
    output = tmp_path / "qualification.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--repo",
            REPO,
            "--source-run-id",
            "20",
            "--trusted-ref",
            TRUSTED,
            "--output",
            str(output),
        ],
    )
    qualify.main()
    runpy.run_path("scripts/ci_candidate_qualification.py", run_name="__main__")
    result = json.loads(output.read_text())
    assert result["qualified"] and result["kind"] == "full"
    assert result["authorizes_reduced_ci"] is False
