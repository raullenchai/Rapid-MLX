"""Additive candidate execution must retain full validation and exact proof."""

import copy
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import ci_candidate_execution as execution
from tests.test_ci_candidate_qualification import fixture
from tests.test_queue_tree_evidence import CANDIDATE, MAIN, REPO

ROOT = Path(__file__).resolve().parents[1]


def configured(monkeypatch):
    client, proof = fixture(monkeypatch, True)
    for path in (*execution.producer.CONTROL_PATHS, execution.CONTROL_PATH):
        client.blobs[CANDIDATE, path] = client.blobs[MAIN, path] = "f" * 40
    monkeypatch.setattr(execution, "qualify_main", lambda *a: {"qualified": True})
    return client, proof


def test_disabled_shadow_never_calls_github():
    class Unavailable:
        def json(self, *a):
            pytest.fail("disabled shadow consulted GitHub")

    assert execution.select_shadow(Unavailable(), 20, CANDIDATE, MAIN, False) == {
        "selected": False,
        "tests": [],
        "authorizes_reduced_ci": False,
    }


def test_shadow_selects_complete_combined_mapping_without_authority(monkeypatch):
    client, proof = configured(monkeypatch)
    result = execution.select_shadow(client, 20, CANDIDATE, MAIN, True)
    assert result["selected"] and result["tests"] == proof["tests"]
    assert result["head"] == CANDIDATE and result["source_attempt"] == 1
    assert not result["authorizes_reduced_ci"]


@pytest.mark.parametrize(
    "change",
    [
        "critical",
        "mixed",
        "control",
        "red-main",
        "closed",
        "fork",
        "cancelled",
        "bad-head",
        "bad-run",
        "superseded",
    ],
)
def test_shadow_negative_controls_fall_back_to_full(monkeypatch, change):
    client, _ = configured(monkeypatch)
    diff = client.responses[f"repos/{REPO}/compare/{MAIN}...{CANDIDATE}"]
    run = client.responses[f"repos/{REPO}/actions/runs/20"]
    head, run_id = CANDIDATE, 20
    if change in ("critical", "mixed"):
        diff["files"].append(
            {"filename": "rapid_mlx/server.py" if change == "critical" else "README.md"}
        )
    elif change == "control":
        client.blobs[CANDIDATE, execution.CONTROL_PATH] = "e" * 40
    elif change == "red-main":
        monkeypatch.setattr(execution, "qualify_main", lambda *a: {"qualified": False})
    elif change == "closed":
        client.responses[f"repos/{REPO}/pulls"] = []
    elif change == "fork":
        run["head_repository"]["full_name"] = "fork/repo"
    elif change == "cancelled":
        run["conclusion"] = "cancelled"
    elif change == "superseded":
        old = client.responses[f"repos/{REPO}/actions/runs/20"]
        client.responses[
            f"repos/{REPO}/actions/workflows/{execution.evidence.CI_WORKFLOW}/runs"
        ]["workflow_runs"].append(dict(old, id=21, conclusion="cancelled"))
    elif change == "bad-head":
        head = "d" * 40
    else:
        run_id = True
    result = execution.select_shadow(client, run_id, head, MAIN, True)
    assert not result["selected"] and not result["authorizes_reduced_ci"]


def packed(monkeypatch, mutation=None):
    _, proof = configured(monkeypatch)
    junit = (
        "<testsuite>" + "<testcase/>" * len(proof["executed"]["nodes"]) + "</testsuite>"
    ).encode()
    coverage = b'<coverage lines-valid="1"><line number="1" hits="1"/></coverage>'
    args = dict(
        base=MAIN,
        head=CANDIDATE,
        run_id=20,
        attempt=1,
        tests=proof["tests"],
        baseline=proof["baseline"],
        executed=copy.deepcopy(proof["executed"]),
        junit=junit,
        coverage=coverage,
        tested_sha=CANDIDATE,
    )
    if mutation:
        mutation(args)
    return execution.pack_input(**args)


def test_pack_preserves_complete_execution_and_exact_identity(monkeypatch):
    record = packed(monkeypatch)
    assert record["scope"] == "candidate-mapped-only"
    assert record["head"] == record["tested_sha"] == CANDIDATE
    assert len(record["diagnostics"]["junit_sha256"]) == 64
    assert not record["diagnostics"]["production_changed_line_pilot_proven"]


@pytest.mark.parametrize(
    "change",
    [
        "skipped",
        "missing-phase",
        "removed-node",
        "junit-failed",
        "junit-short",
        "empty-coverage",
        "wrong-head",
        "bad-attempt",
    ],
)
def test_pack_rejects_incomplete_execution(monkeypatch, change):
    def mutate(args):
        if change == "skipped":
            args["executed"]["reports"][0]["outcome"] = "skipped"
        elif change == "missing-phase":
            args["executed"]["reports"].pop()
        elif change == "removed-node":
            args["executed"]["nodes"].pop()
        elif change == "junit-failed":
            args["junit"] = args["junit"].replace(
                b"<testcase/>", b"<testcase><failure/></testcase>", 1
            )
        elif change == "junit-short":
            args["junit"] = b"<testsuite/>"
        elif change == "empty-coverage":
            args["coverage"] = b'<coverage lines-valid="0"/>'
        elif change == "wrong-head":
            args["tested_sha"] = MAIN
        else:
            args["attempt"] = True

    with pytest.raises(ValueError):
        packed(monkeypatch, mutate)


@pytest.mark.parametrize(
    ("shadow", "outcome", "full_outcome", "passes"),
    [
        ("true", "success", "success", True),
        ("true", "failure", "success", False),
        ("true", "skipped", "success", False),
        ("true", "cancelled", "success", False),
        ("true", "", "success", False),
        ("true", "success", "failure", False),
        ("false", "skipped", "success", True),
        ("false", "success", "success", False),
    ],
)
def test_actual_aggregate_requires_both_shadow_and_full(
    shadow, outcome, full_outcome, passes
):
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    values = {f"needs.{job}.result": "success" for job in jobs["tests"]["needs"]}
    values.update(
        {
            "needs.changes.outputs.reuse_ci": "false",
            "needs.changes.outputs.engine": "true",
            "needs.changes.outputs.full_gate": "true",
            "needs.changes.outputs.source_canary": "false",
            "needs.changes.outputs.candidate_shadow": shadow,
            "needs.candidate-canary-unit.result": outcome,
            "needs.source-canary-unit.result": "skipped",
            "needs.test-matrix.result": full_outcome,
            "github.event_name": "pull_request",
        }
    )
    rendered = re.sub(
        r"\$\{\{\s*(.*?)\s*\}\}",
        lambda m: values.get(m[1], ""),
        jobs["tests"]["steps"][0]["run"],
    )
    result = subprocess.run(["bash", "-c", rendered], capture_output=True, text=True)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@pytest.mark.parametrize("change", ["missing-run", "cancel-race", "red-main-race"])
def test_selection_rechecks_live_prerequisites(monkeypatch, change):
    client, _ = configured(monkeypatch)
    if change == "missing-run":
        client.responses[
            f"repos/{REPO}/actions/workflows/{execution.evidence.CI_WORKFLOW}/runs"
        ]["workflow_runs"] = []
    elif change == "cancel-race":
        original = client.json
        reads = 0

        def json(endpoint, *args, **kwargs):
            nonlocal reads
            result = copy.deepcopy(original(endpoint, *args, **kwargs))
            if endpoint.endswith("/actions/runs/20"):
                reads += 1
                if reads == 2:
                    result["conclusion"] = "cancelled"
            return result

        monkeypatch.setattr(client, "json", json)
    else:
        calls = 0

        def qualify(*args):
            nonlocal calls
            calls += 1
            return {"qualified": calls == 1}

        monkeypatch.setattr(execution, "qualify_main", qualify)
    assert not execution.select_shadow(client, 20, CANDIDATE, MAIN, True)["selected"]


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_select_cli_writes_only_selected_outputs(monkeypatch, tmp_path, enabled):
    import json
    import sys

    client, proof = configured(monkeypatch)
    output = tmp_path / "outputs"
    monkeypatch.setattr(execution.evidence, "GitHubClient", lambda repo: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execution",
            "select",
            "--repo",
            REPO,
            "--enabled",
            str(enabled).lower(),
            "--github-output",
            str(output),
            "--run-id",
            "20",
            "--head",
            CANDIDATE,
            "--base",
            MAIN,
        ],
    )
    execution.main()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["candidate_shadow"] == str(enabled).lower()
    assert values["candidate_shadow_tests"].split() == (
        proof["tests"] if enabled else []
    )
    assert "authorizes_reduced_ci" not in json.dumps(values)


def test_actual_pack_cli_reads_checkout_and_writes_bound_record(monkeypatch, tmp_path):
    import json
    import sys

    _, proof = configured(monkeypatch)
    files = {
        "baseline": json.dumps(proof["baseline"]),
        "executed": json.dumps(proof["executed"]),
        "junit": "<testsuite>"
        + "<testcase/>" * len(proof["executed"]["nodes"])
        + "</testsuite>",
        "coverage": '<coverage lines-valid="1"><line number="1" hits="1"/></coverage>',
    }
    argv = [
        "execution",
        "pack",
        "--head",
        CANDIDATE,
        "--base",
        MAIN,
        "--run-id",
        "20",
        "--attempt",
        "1",
        "--tests",
        *proof["tests"],
    ]
    for name, contents in files.items():
        path = tmp_path / name
        path.write_text(contents)
        argv.extend(["--" + name, str(path)])
    output = tmp_path / "record.json"
    argv.extend(["--output", str(output)])
    calls = []
    monkeypatch.setattr(
        execution.subprocess, "check_output", lambda command, **kwargs: CANDIDATE + "\n"
    )
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    monkeypatch.setattr(sys, "argv", argv)
    execution.main()
    assert calls == [(["git", "diff", "--quiet", "HEAD", "--"], {"check": True})]
    assert json.loads(output.read_text())["tested_sha"] == CANDIDATE


def test_module_entrypoint_default_off(monkeypatch, tmp_path):
    import runpy
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execution",
            "select",
            "--repo",
            REPO,
            "--github-output",
            str(tmp_path / "out"),
            "--run-id",
            "20",
            "--head",
            CANDIDATE,
            "--base",
            MAIN,
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("scripts.ci_candidate_execution", run_name="__main__")
    assert "candidate_shadow=false" in (tmp_path / "out").read_text()
