"""Mapped CPU proof must describe the whole selected suite actually executed."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci_mapped_execution import SCHEMA, validate_manifests

ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_leaf.py", "tests/test_other.py"]
NODES = [f"{path}::test_ok" for path in TESTS]


def test_workflow_binds_base_collection_and_actual_execution():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    steps = jobs["source-canary-unit"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name")
        == "Run mapped regressions and mandatory changed-line coverage"
    )
    assert 'git worktree add --detach "$baseline" "$PR_BASE_SHA"' in run
    assert 'PYTHONPATH="$baseline" python -m pytest "${selected[@]}" -o addopts=' in run
    assert "--collect-only -p scripts.ci_mapped_execution" in run
    assert (
        'python -m pytest "${selected[@]}" -o addopts= -p scripts.ci_mapped_execution'
        in run
    )
    assert "--baseline source-canary-base-collection.json" in run
    assert '--executed source-canary-execution.json --tests "${selected[@]}"' in run
    assert "--fail-under 100" in run
    proof = next(
        step["with"]
        for step in steps
        if step.get("name") == "Upload separate source-only proof"
    )
    assert proof["name"] == "source-canary-unit-${{ github.sha }}"
    assert "source-canary-base-collection.json" in proof["path"]
    assert "source-canary-execution.json" in proof["path"]
    assert "full_gate" in jobs["source-canary-unit"]["if"]


def manifest(collect_only=False):
    return {
        "schema": SCHEMA,
        "collect_only": collect_only,
        "exitstatus": 0,
        "nodes": list(NODES),
        "deselected": 0,
        "collection_errors": 0,
        "reports": []
        if collect_only
        else [
            {"node": node, "stage": stage, "outcome": "passed", "xfail": False}
            for node in NODES
            for stage in ("setup", "call", "teardown")
        ],
    }


def test_complete_proof_accepts_additional_executed_tests():
    base, executed = manifest(True), manifest()
    node = "tests/test_leaf.py::test_new"
    executed["nodes"].append(node)
    executed["reports"] += [
        {"node": node, "stage": stage, "outcome": "passed", "xfail": False}
        for stage in ("setup", "call", "teardown")
    ]
    validate_manifests(base, executed, TESTS)


@pytest.mark.parametrize("side", ["base", "executed"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "full-proof"),
        ("exitstatus", 1),
        ("deselected", 1),
        ("collection_errors", 1),
        ("nodes", []),
        ("nodes", ["bad"]),
        ("nodes", NODES * 2),
        ("nodes", NODES[:1]),
        ("nodes", [*NODES, "tests/test_unselected.py::test_no"]),
    ],
)
def test_incomplete_or_wrong_collection_rejected(side, field, value):
    base, executed = manifest(True), manifest()
    (base if side == "base" else executed)[field] = value
    with pytest.raises(ValueError):
        validate_manifests(base, executed, TESTS)


def test_collect_only_cannot_masquerade_as_execution():
    with pytest.raises(ValueError):
        validate_manifests(manifest(True), manifest(True), TESTS)
    with pytest.raises(ValueError):
        validate_manifests(manifest(), manifest(), TESTS)


def test_renamed_test_does_not_shrink_baseline():
    executed = manifest()
    executed["nodes"][0] += "_renamed"
    with pytest.raises(ValueError, match="removed or renamed"):
        validate_manifests(manifest(True), executed, TESTS)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "unknown",
        "skipped",
        "failed",
        "xfail",
        "stage",
        "malformed",
    ],
)
def test_every_collected_node_must_execute_all_stages_once(change):
    executed = manifest()
    if change == "missing":
        executed["reports"].pop()
    elif change == "duplicate":
        executed["reports"].append(copy.deepcopy(executed["reports"][0]))
    elif change == "malformed":
        executed["reports"] = None
    else:
        key, value = {
            "unknown": ("node", "tests/test_other.py::test_unknown"),
            "skipped": ("outcome", "skipped"),
            "failed": ("outcome", "failed"),
            "xfail": ("xfail", True),
            "stage": ("stage", "unknown"),
        }[change]
        executed["reports"][0][key] = value
    with pytest.raises(ValueError):
        validate_manifests(manifest(True), executed, TESTS)


@pytest.mark.parametrize("selection", [[], TESTS * 2])
def test_empty_or_duplicate_selection_rejected(selection):
    with pytest.raises(ValueError):
        validate_manifests(manifest(True), manifest(), selection)


@pytest.mark.parametrize(
    "mode", ["pass", "skip", "xfail", "deselect", "collection_skip"]
)
def test_real_pytest_recording_and_cli(tmp_path, mode):
    tests = tmp_path / "tests"
    tests.mkdir()
    code = "def test_ok():\n    assert True\n"
    decorators = {
        "skip": "@pytest.mark.skip(reason='negative control')\n",
        "xfail": "@pytest.mark.xfail(reason='negative control')\n",
    }
    if mode in decorators:
        code = "import pytest\n" + decorators[mode] + code
    elif mode == "collection_skip":
        code = (
            "import pytest\npytest.skip('negative control', allow_module_level=True)\n"
            + code
        )
    (tests / "test_leaf.py").write_text(code)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    common = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_leaf.py",
        "-o",
        "addopts=",
        "-p",
        "scripts.ci_mapped_execution",
    ]
    baseline = tmp_path / "base.json"
    subprocess.run(
        [*common, "--collect-only", "--mapped-execution-manifest", str(baseline)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    executed = tmp_path / "run.json"
    extra = ["-k", "not test_ok"] if mode == "deselect" else []
    subprocess.run(
        [*common, *extra, "--mapped-execution-manifest", str(executed)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci_mapped_execution.py"),
            "--baseline",
            str(baseline),
            "--executed",
            str(executed),
            "--tests",
            "tests/test_leaf.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (mode == "pass"), result.stderr
    assert json.loads(executed.read_text())["schema"] == SCHEMA
