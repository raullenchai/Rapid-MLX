"""Source-only mapped proof must never stand in for full candidate evidence."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.classify_ci_changes import classify_policy, source_canary_tests

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        "rapid_mlx/cli_help.py",
        "rapid_mlx/_banner.py",
        "tests/test_cli_help_groups.py",
        "tests/test_cli_cheetah_banner.py",
        "rapid_mlx/telemetry/chip.py",
        "tests/test_telemetry_chip.py",
        "rapid_mlx/telemetry/events.json",
        "rapid_mlx/telemetry/registry.py",
        "tests/test_telemetry_registry.py",
        "tests/test_telemetry_registry_drift.py",
    ],
)
def test_exact_leaf_paths_select_existing_regressions(path):
    policy = classify_policy([path], source_canary=True)
    assert policy.source_canary_tests
    assert all((ROOT / test).is_file() for test in policy.source_canary_tests)
    assert policy.as_outputs()["source_canary"] == "true"
    assert not classify_policy(
        [path], force_full=True, source_canary=True
    ).source_canary_tests
    assert not classify_policy([path]).source_canary_tests


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["rapid_mlx/cli.py"],
        ["rapid_mlx/cli_parser.py"],
        ["rapid_mlx/chip_tier.py"],
        ["rapid_mlx/scheduler.py"],
        ["rapid_mlx/telemetry/model_id.py"],
        ["tests/conftest.py"],
        [".github/workflows/ci.yml"],
        ["scripts/classify_ci_changes.py"],
        ["pyproject.toml"],
        ["new-area/foo.py"],
        ["../rapid_mlx/cli_help.py"],
        ["rapid_mlx/cli_help.py", "README.md"],
        ["rapid_mlx/cli_help.py", "rapid_mlx/cli.py"],
        ["rapid_mlx/_banner.py", "apps/rapid-mac/Sources/Rapid/App.swift"],
    ],
)
def test_unknown_critical_control_and_mixed_paths_force_full(paths):
    policy = classify_policy(paths, source_canary=True)
    assert not policy.source_canary_tests
    assert policy.linux_matrix_mode == "full"


def test_cli_and_telemetry_mapping_unions_all_required_tests():
    assert set(
        source_canary_tests({"rapid_mlx/_banner.py", "rapid_mlx/telemetry/registry.py"})
    ) == {
        "tests/test_cli_cheetah_banner.py",
        "tests/test_cli_help_groups.py",
        "tests/test_cli_parser_snapshot.py",
        "tests/test_telemetry_registry.py",
        "tests/test_telemetry_registry_drift.py",
    }


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]


@pytest.mark.parametrize(
    ("updates", "passes"),
    [
        ({}, True),
        ({"needs.source-canary-unit.result": "failure"}, False),
        ({"needs.source-canary-unit.result": "skipped"}, False),
        ({"needs.source-canary-unit.result": "cancelled"}, False),
        ({"needs.source-canary-unit.result": ""}, False),
        ({"needs.changes.outputs.full_gate": "true"}, False),
        ({"github.event_name": "push"}, False),
        ({"needs.lint.result": "failure"}, False),
        ({"needs.test-matrix.result": "success"}, False),
        ({"needs.test-apple-silicon.result": "failure"}, False),
        ({"needs.changes.outputs.source_canary": "false"}, False),
    ],
)
def test_real_required_aggregate_rejects_failed_missing_or_wrong_scope(updates, passes):
    script = workflow()["tests"]["steps"][0]["run"]
    values = {
        "needs.changes.result": "success",
        "needs.changes.outputs.reuse_ci": "false",
        "needs.changes.outputs.engine": "true",
        "needs.changes.outputs.source_canary": "true",
        "needs.changes.outputs.full_gate": "false",
        "github.event_name": "pull_request",
        "needs.source-canary-unit.result": "success",
    }
    for job in ["lint", "engine-contracts", "type-check", "mlx-bound-guard"]:
        values[f"needs.{job}.result"] = "success"
    for job in [
        "test-matrix",
        "test-apple-silicon",
        "linux-coverage",
        "changed-lines-coverage",
        "l1-smoke",
    ]:
        values[f"needs.{job}.result"] = "skipped"
    values.update(updates)
    rendered = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values.get(m[1], ""), script)
    result = subprocess.run(["bash", "-c", rendered], capture_output=True, text=True)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


def test_canary_has_distinct_artifact_mandatory_coverage_and_full_backstop():
    jobs = workflow()
    steps = jobs["source-canary-unit"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name")
        == "Run mapped regressions and mandatory changed-line coverage"
    )
    install = next(
        step["run"]
        for step in steps
        if step.get("name") == "Install mapped CPU test dependencies"
    )
    assert "pip install -e . --no-deps" in install
    assert "config/requirements-ci-linux.txt" in install
    assert "--fail-under 100" in run
    assert "mapped-cpu-only" in run
    artifact = next(
        step["with"]["name"]
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert artifact == "source-canary-unit-${{ github.sha }}"
    for job in [
        "test-matrix",
        "test-apple-silicon",
        "linux-coverage",
        "changed-lines-coverage",
    ]:
        assert "needs.changes.outputs.source_canary != 'true'" in jobs[job]["if"]
    classifier = next(
        step["run"]
        for step in jobs["changes"]["steps"]
        if step.get("name") == "Classify validation lanes"
    )
    assert '"$CANARY_ENABLED" = true' in classifier
    assert "--force-reason source-canary-disabled" in classifier
    assert "--force-reason promoted-head" in classifier
    assert "echo 'source_canary=false'" in classifier


@pytest.mark.parametrize(
    "xml,passes",
    [
        ('<testsuite><testcase name="ok"/></testsuite>', True),
        ("<testsuite/>", False),
        ("<testsuite><testcase><skipped/></testcase></testsuite>", False),
        ("<testsuite><testcase><failure/></testcase></testsuite>", False),
        ("<testsuite><testcase><error/></testcase></testsuite>", False),
    ],
)
def test_actual_source_execution_guard_rejects_empty_or_skipped_proof(
    tmp_path, xml, passes
):
    steps = workflow()["source-canary-unit"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if step.get("name")
        == "Run mapped regressions and mandatory changed-line coverage"
    )
    command = next(
        line.strip()
        for line in script.splitlines()
        if line.strip().startswith("python -c")
    )
    (tmp_path / "source-canary-junit.xml").write_text(xml)
    command = command.replace("python -c", f"'{sys.executable}' -c", 1)
    result = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True
    )
    assert (result.returncode == 0) == passes


def test_red_pending_missing_or_wrong_base_anchor_is_not_eligible(tmp_path):
    # Exercise the same exact-head/complete-success predicate used by the
    # workflow against completed, red, pending, absent and mismatched records.
    script = next(
        step["run"]
        for step in workflow()["changes"]["steps"]
        if step.get("name") == "Classify validation lanes"
    )
    predicate = '.[0] | .headSha == $base and .status == "completed" and .conclusion == "success"'
    assert predicate in script
    assert (
        "--event push" in script and '--branch main --commit "$PR_BASE_SHA"' in script
    )
    assert "CANARY_ENABLED=false" in script
    import json

    for record, eligible in [
        ([dict(headSha="a" * 40, status="completed", conclusion="success")], True),
        ([dict(headSha="a" * 40, status="completed", conclusion="failure")], False),
        ([dict(headSha="a" * 40, status="in_progress", conclusion="")], False),
        ([dict(headSha="b" * 40, status="completed", conclusion="success")], False),
        ([], False),
    ]:
        result = subprocess.run(
            ["jq", "-e", "--arg", "base", "a" * 40, predicate],
            input=json.dumps(record),
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == eligible


@pytest.mark.parametrize("force_full", [False, True])
def test_cli_flags_and_output_preserve_source_vs_promoted_scope(
    tmp_path, monkeypatch, force_full
):
    import scripts.classify_ci_changes as classifier

    paths = tmp_path / "paths"
    output = tmp_path / "outputs"
    paths.write_text("rapid_mlx/cli_help.py\n")
    arguments = [
        "classifier",
        "--source-canary",
        "--paths-file",
        str(paths),
        "--github-output",
        str(output),
    ]
    if force_full:
        arguments += ["--force-full", "--force-reason", "promoted-head"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert classifier.main() == 0
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["source_canary"] == str(not force_full).lower()
    if force_full:
        assert outputs["linux_matrix_mode"] == "full"
        assert outputs["linux_matrix_reason"] == "promoted-head"
        assert outputs["source_canary_tests"] == ""
    else:
        assert "tests/test_cli_help_groups.py" in outputs["source_canary_tests"].split()
