"""Cancellation stops compute and leaves a failing required verdict."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]


def test_workloads_reuse_existing_desktop_cancellation_pattern():
    jobs = workflow()
    assert str(jobs["changes"]["if"]) == "${{ !cancelled() }}"
    for name in (
        "merge-lane-mac",
        "merge-lane-no-mac",
        "lint",
        "engine-contracts",
        "type-check",
        "mlx-bound-guard",
        "source-canary-unit",
        "candidate-canary-unit",
        "test-matrix",
        "linux-coverage",
        "test-apple-silicon",
        "changed-lines-coverage",
        "l1-smoke",
    ):
        condition = str(jobs[name]["if"])
        assert "!cancelled()" in condition, name
        assert "always()" not in condition, name
        assert "needs.changes.result == 'success'" in condition, name
    assert str(jobs["tests"]["if"]) == "always()"


@pytest.mark.parametrize("scope", ["full", "reuse", "docs", "mapped"])
@pytest.mark.parametrize("cancelled", [False, True, "after-verdict"])
def test_real_aggregate_cannot_green_a_cancelled_scope(scope, cancelled):
    jobs = workflow()
    values = {
        "cancelled()": str(cancelled).lower(),
        "github.event_name": "pull_request",
        "needs.changes.result": "success",
        "needs.changes.outputs.engine": "false" if scope == "docs" else "true",
        "needs.changes.outputs.full_gate": "false" if scope == "mapped" else "true",
        "needs.changes.outputs.reuse_ci": "true" if scope == "reuse" else "false",
        "needs.changes.outputs.source_canary": "true" if scope == "mapped" else "false",
        "needs.changes.outputs.candidate_shadow": "false",
        "needs.candidate-canary-unit.result": "skipped",
        "needs.source-canary-unit.result": "success"
        if scope == "mapped"
        else "skipped",
    }
    for name in ("lint", "engine-contracts", "type-check", "mlx-bound-guard"):
        values[f"needs.{name}.result"] = "success"
    for name in (
        "test-matrix",
        "linux-coverage",
        "test-apple-silicon",
        "changed-lines-coverage",
        "l1-smoke",
    ):
        values[f"needs.{name}.result"] = "skipped" if scope == "mapped" else "success"
    outputs = []
    failed = False
    # Run actual checked-in step commands, including the final rejection.
    conditions = {
        "${{ !cancelled() }}": cancelled is not True,
        "${{ cancelled() }}": cancelled,
    }
    for step in jobs["tests"]["steps"]:
        if "if" in step:
            assert step["if"] in conditions
            if not conditions[step["if"]]:
                continue
        rendered = re.sub(
            r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values.get(m[1], ""), step["run"]
        )
        result = subprocess.run(
            ["bash", "-c", rendered], capture_output=True, text=True
        )
        outputs.append(result.stdout + result.stderr)
        failed |= result.returncode != 0
    assert failed == bool(cancelled), "\n".join(outputs)
    if cancelled:
        assert "workflow was cancelled" in "\n".join(outputs)
