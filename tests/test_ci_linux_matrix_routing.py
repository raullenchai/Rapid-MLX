# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for risk-routed Linux interpreter breadth."""

from pathlib import Path

import yaml

from scripts.classify_ci_changes import linux_test_matrix
from scripts.queue_tree_evidence import REQUIRED_CI_MATRIX_PREFIXES

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _classification_run() -> str:
    steps = _workflow()["jobs"]["changes"]["steps"]
    return next(
        step["run"] for step in steps if step.get("name") == "Classify validation lanes"
    )


def test_test_matrix_comes_from_tested_classifier_output() -> None:
    jobs = _workflow()["jobs"]
    outputs = jobs["changes"]["outputs"]

    assert (
        outputs["linux_matrix_mode"] == "${{ steps.policy.outputs.linux_matrix_mode }}"
    )
    assert outputs["linux_matrix_reason"] == (
        "${{ steps.policy.outputs.linux_matrix_reason }}"
    )
    assert outputs["test_matrix"] == "${{ steps.policy.outputs.test_matrix }}"
    assert jobs["test-matrix"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.changes.outputs.test_matrix) }}"
    )

    explain = next(
        step
        for step in jobs["changes"]["steps"]
        if step.get("name") == "Explain Linux matrix route"
    )
    assert explain["env"] == {
        "MATRIX_MODE": "${{ steps.policy.outputs.linux_matrix_mode }}",
        "MATRIX_REASON": "${{ steps.policy.outputs.linux_matrix_reason }}",
    }
    assert 'echo "Linux matrix: ${MATRIX_MODE} (${MATRIX_REASON})"' in explain["run"]
    assert '>> "$GITHUB_STEP_SUMMARY"' in explain["run"]


def test_promoted_and_non_pr_heads_keep_full_supported_python_matrix() -> None:
    run = _classification_run()
    full_json = (
        '{"include":[{"python-version":"3.10","shard":1},'
        '{"python-version":"3.10","shard":2},'
        '{"python-version":"3.10","shard":3},'
        '{"python-version":"3.11","shard":1},'
        '{"python-version":"3.11","shard":2},'
        '{"python-version":"3.11","shard":3},'
        '{"python-version":"3.12","shard":1},'
        '{"python-version":"3.12","shard":2},'
        '{"python-version":"3.12","shard":3}]}'
    )

    assert "--force-full" in run
    assert "--force-reason promoted-head" in run
    assert "linux_matrix_mode=full" in run
    assert "linux_matrix_reason=non-pr-full" in run
    assert f"test_matrix={full_json}" in run
    assert linux_test_matrix("full") == {
        "include": [
            {"python-version": version, "shard": shard}
            for version in ("3.10", "3.11", "3.12")
            for shard in (1, 2, 3)
        ]
    }


def test_reduced_route_keeps_all_three_coverage_shards() -> None:
    workflow = _workflow()
    matrix = linux_test_matrix("py311")["include"]

    assert matrix == [
        {"python-version": "3.11", "shard": 1},
        {"python-version": "3.11", "shard": 2},
        {"python-version": "3.11", "shard": 3},
    ]
    steps = workflow["jobs"]["test-matrix"]["steps"]
    upload = next(
        step for step in steps if step.get("name") == "Upload Linux shard coverage data"
    )
    assert upload["if"] == "matrix.python-version == '3.11'"
    assert upload["with"]["if-no-files-found"] == "error"

    linux_coverage = workflow["jobs"]["linux-coverage"]
    assert set(linux_coverage["needs"]) == {"changes", "test-matrix"}
    assert "needs.test-matrix.result == 'success'" in linux_coverage["if"]


def test_candidate_tree_evidence_still_requires_nine_matrix_jobs() -> None:
    assert REQUIRED_CI_MATRIX_PREFIXES["test-matrix ("] == 9
