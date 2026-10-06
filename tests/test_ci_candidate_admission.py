"""A producer notification cannot bypass current authenticated full evidence."""

import copy
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from scripts import ci_candidate_admission as admission
from tests.test_ci_candidate_consumer import setup
from tests.test_queue_tree_evidence import CANDIDATE, REPO


def test_real_archive_and_full_validation_admission(monkeypatch):
    client, _, _, _, _ = setup(monkeypatch)
    result = admission.verify_admission(client, 100)
    assert result["verified"] and result["kind"] == "full"
    assert result["candidate_sha"] == CANDIDATE
    assert result["source_run_id"] == 20 and result["producer_attempt"] == 1
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]


@pytest.mark.parametrize(
    "change",
    [
        "invalid-run",
        "failed-producer",
        "truncated",
        "duplicate",
        "missing",
        "expired",
        "wrong-run",
        "superseded",
        "mapped",
        "cancelled-source",
        "closed-candidate",
        "main-moved",
    ],
)
def test_bad_notification_or_evidence_never_publishes_success(monkeypatch, change):
    client, record, status, run, artifact = setup(monkeypatch)
    prefix = f"repos/{REPO}"
    page = client.responses[f"{prefix}/actions/runs/100/artifacts"]
    run_id = 100
    if change == "invalid-run":
        run_id = True
    elif change == "failed-producer":
        run["conclusion"] = "failure"
    elif change == "truncated":
        page["total_count"] = 100
    elif change == "duplicate":
        page["artifacts"].append(dict(artifact, id=102))
        page["total_count"] = 2
    elif change == "missing":
        artifact["name"] = "unrelated"
    elif change == "expired":
        artifact["expired"] = True
    elif change == "wrong-run":
        artifact["workflow_run"]["id"] = 101
    elif change == "superseded":
        status["target_url"] = f"https://github.com/{REPO}/actions/runs/101"
    elif change == "mapped":
        record["kind"] = "mapped"
    elif change == "cancelled-source":
        client.responses[f"{prefix}/actions/runs/20"]["conclusion"] = "cancelled"
    elif change == "closed-candidate":
        client.responses[f"{prefix}/pulls"] = []
    else:
        client.responses[f"{prefix}/git/ref/heads/main"]["object"]["sha"] = "e" * 40
    result = admission.verify_admission(client, run_id)
    assert not result["verified"] and "candidate_sha" not in result
    assert not result["authorizes_merge"] and not result["authorizes_reduced_ci"]


@pytest.mark.parametrize("change", ["producer", "index", "candidate"])
def test_last_rereads_reject_mutations(monkeypatch, change):
    client, _, _, _, _ = setup(monkeypatch)
    if change == "candidate":
        original = admission.consumer.consume_full
        calls = 0

        def consume(*args):
            nonlocal calls
            calls += 1
            return original(*args) if calls == 1 else {"verified": False}

        monkeypatch.setattr(admission.consumer, "consume_full", consume)
    else:
        name = "_producer_run" if change == "producer" else "_status"
        original = getattr(admission.consumer, name)
        calls = 0

        # Initial selector, consume_full initial/final, then final selector.
        def read(*args):
            nonlocal calls
            calls += 1
            result = copy.deepcopy(original(*args))
            if calls == 4:
                result["mutated"] = True
            return result

        monkeypatch.setattr(admission.consumer, name, read)
    assert not admission.verify_admission(client, 100)["verified"]


@pytest.mark.parametrize("good", [True, False])
def test_cli_only_exposes_validated_sha(monkeypatch, tmp_path, good):
    client, _, _, run, _ = setup(monkeypatch)
    if not good:
        run["conclusion"] = "failure"
    monkeypatch.setattr(admission.evidence, "GitHubClient", lambda repo: client)
    out, record = tmp_path / "out", tmp_path / "record"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admission",
            "--repo",
            REPO,
            "--producer-run-id",
            "100",
            "--github-output",
            str(out),
            "--output",
            str(record),
        ],
    )
    admission.main()
    values = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert values["verified"] == str(good).lower()
    assert ("candidate_sha" in values) is good
    assert json.loads(record.read_text())["verified"] is good


def test_module_entrypoint_rejects_invalid_run_without_api(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admission",
            "--repo",
            REPO,
            "--producer-run-id",
            "0",
            "--github-output",
            str(tmp_path / "out"),
            "--output",
            str(tmp_path / "record"),
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("scripts.ci_candidate_admission", run_name="__main__")
    assert (tmp_path / "out").read_text() == "verified=false\n"


def test_workflow_uses_trusted_checkout_and_indexes_after_upload():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/candidate-admission.yml").read_text()
    )
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
        "statuses": "write",
    }
    job = workflow["jobs"]["admit"]
    assert job["if"] == "github.event.workflow_run.conclusion == 'success'"
    steps = job["steps"]
    assert steps[0]["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": False,
    }
    assert all(
        len(step["uses"].split("@")[1]) == 40 for step in steps if "uses" in step
    )
    upload, index = steps[-2:]
    assert upload["if"] == index["if"] == "steps.result.outputs.verified == 'true'"
    assert "candidate-admission/ci" in index["run"]
    assert '--producer-run-id "$PRODUCER_RUN"' in steps[1]["run"]
    assert "candidate-admission/ci" not in (root / ".mergify.yml").read_text()
