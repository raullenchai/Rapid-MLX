"""Ordinary CPU prefilter never substitutes for combined-candidate evidence."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci_test_shard import discover, partition
from scripts.classify_ci_changes import classify_policy

ROOT = Path(__file__).resolve().parents[1]


def jobs():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]


@pytest.mark.parametrize(
    "path",
    [
        "rapid_mlx/server.py",
        "rapid_mlx/scheduler.py",
        "rapid_mlx/cli.py",
        "rapid_mlx/telemetry/track.py",
        "tests/test_telemetry_loop_dedupe.py",
    ],
)
def test_prefilter_runs_all_three_unit_shards_without_claiming_full(path):
    policy = classify_policy([path], source_preflight=True)
    assert policy.source_preflight
    assert policy.linux_matrix_reason == "source-preflight"
    assert json.loads(policy.as_outputs()["test_matrix"]) == {
        "include": [{"python-version": "3.11", "shard": i} for i in (1, 2, 3)]
    }
    assert policy.as_outputs()["source_canary"] == "false"
    assert not classify_policy([path]).source_preflight
    forced = classify_policy([path], source_preflight=True, force_full=True)
    assert not forced.source_preflight
    assert len(json.loads(forced.as_outputs()["test_matrix"])["include"]) == 9


@pytest.mark.parametrize(
    "paths",
    [
        [
            "rapid_mlx/byom/support_request.py",
            "rapid_mlx/cli_parser.py",
            "tests/fixtures/cli_parser_snapshot.json",
            "tests/test_byom_support_request.py",
        ],
        [
            "rapid_mlx/_download_gate.py",
            "rapid_mlx/cli.py",
            "rapid_mlx/cli_parser.py",
            "tests/fixtures/cli_parser_snapshot.json",
            "tests/test_cli_start.py",
        ],
        [
            "rapid_mlx/_banner.py",
            "rapid_mlx/cli.py",
            "rapid_mlx/cli_help.py",
            "rapid_mlx/cli_parser.py",
            "rapid_mlx/first_run.py",
            "rapid_mlx/front_door.py",
            "tests/fixtures/cli_parser_snapshot.json",
            "tests/test_cli_cheetah_banner.py",
            "tests/test_cli_help_groups.py",
            "tests/test_first_run_guide.py",
            "tests/test_front_door.py",
        ],
        [
            "rapid_mlx/agents/adapter.py",
            "rapid_mlx/agents/base.py",
            "rapid_mlx/agents/setup.py",
            "rapid_mlx/cli.py",
            "rapid_mlx/cli_parser.py",
            "tests/fixtures/cli_parser_snapshot.json",
            "tests/test_agent_first_class_setup.py",
            "tests/test_cli_start.py",
        ],
    ],
)
def test_cli_snapshot_historical_unions_use_source_preflight_only(paths):
    policy = classify_policy(paths, source_preflight=True)
    assert policy.source_preflight
    assert policy.linux_matrix_mode == "py311"
    assert len(json.loads(policy.as_outputs()["test_matrix"])["include"]) == 3

    disabled = classify_policy(paths)
    assert not disabled.source_preflight
    assert disabled.linux_matrix_mode == "full"

    promoted = classify_policy(paths, source_preflight=True, force_full=True)
    assert not promoted.source_preflight
    assert promoted.linux_matrix_mode == "full"
    assert len(json.loads(promoted.as_outputs()["test_matrix"])["include"]) == 9


def test_cli_snapshot_fixture_has_two_cpu_consumers_each_in_one_shard():
    fixture_read = "SNAPSHOT.read_text()"
    consumers = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if path.name != Path(__file__).name and fixture_read in path.read_text()
    ]
    assert sorted(consumers) == [
        "tests/test_cli_parser_snapshot.py",
        "tests/test_serve_command_characterization.py",
    ]

    shards = partition(discover(ROOT, "ordinary"), 3)
    for consumer in consumers:
        selected = [
            index
            for index, shard in enumerate(shards, 1)
            if any(item.path == consumer for item in shard)
        ]
        assert len(selected) == 1


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["../rapid_mlx/server.py"],
        ["/rapid_mlx/server.py"],
        ["tests/conftest.py"],
        ["tests/fixtures/config.py"],
        ["rapid_mlx/cli.py", "tests/fixtures/other.json"],
        ["rapid_mlx/cli.py", "tests/headless_mlx/test_engine_lifecycle.py"],
        ["tests/test_ci_main_qualification.py"],
        ["tests/test_queue_tree_evidence.py"],
        ["scripts/classify_ci_changes.py"],
        [".github/workflows/ci.yml"],
        ["pyproject.toml"],
        ["uv.lock"],
        ["new-root/thing.py"],
        ["rapid_mlx/server.py", "apps/rapid-mac/App.swift"],
        ["rapid_mlx/server.py", ".coveragerc"],
    ],
)
def test_controls_unknown_collection_and_mixed_diffs_remain_full(paths):
    policy = classify_policy(paths, source_preflight=True)
    assert not policy.source_preflight
    assert policy.linux_matrix_mode == "full"


def test_existing_mapped_source_route_takes_precedence():
    policy = classify_policy(
        ["rapid_mlx/_banner.py"], source_canary=True, source_preflight=True
    )
    assert policy.source_canary_tests
    assert not policy.source_preflight


@pytest.mark.parametrize(
    ("updates", "passes"),
    [
        ({}, True),
        ({"needs.changes.outputs.source_canary": "true"}, False),
        ({"needs.test-matrix.result": "failure"}, False),
        ({"needs.test-matrix.result": "cancelled"}, False),
        ({"needs.test-matrix.result": "skipped"}, False),
        ({"needs.linux-coverage.result": ""}, False),
        ({"needs.linux-coverage.result": "failure"}, False),
        ({"needs.changes.outputs.full_gate": "true"}, False),
        ({"needs.changes.outputs.linux_matrix_mode": "full"}, False),
        ({"github.event_name": "push"}, False),
        ({"github.event.pull_request.head.repo.full_name": "fork/repo"}, False),
        ({"needs.lint.result": "failure"}, False),
        ({"needs.engine-contracts.result": "failure"}, False),
        ({"needs.type-check.result": "failure"}, False),
        ({"needs.mlx-bound-guard.result": "failure"}, False),
        ({"needs.test-apple-silicon.result": "cancelled"}, False),
        ({"needs.changed-lines-coverage.result": "failure"}, False),
        ({"needs.source-canary-unit.result": "success"}, False),
        ({"needs.l1-smoke.result": "success"}, False),
    ],
)
def test_rendered_aggregate_has_no_failure_or_wrong_context_success(updates, passes):
    values = {
        "needs.changes.result": "success",
        "needs.changes.outputs.engine": "true",
        "needs.changes.outputs.source_preflight": "true",
        "needs.changes.outputs.source_canary": "false",
        "needs.changes.outputs.full_gate": "false",
        "needs.changes.outputs.linux_matrix_mode": "py311",
        "needs.changes.outputs.reuse_ci": "false",
        "needs.changes.outputs.candidate_shadow": "false",
        "github.event_name": "pull_request",
        "github.repository": "owner/repo",
        "github.event.pull_request.head.repo.full_name": "owner/repo",
    }
    for name in (
        "lint",
        "engine-contracts",
        "mlx-bound-guard",
        "type-check",
        "test-matrix",
        "linux-coverage",
    ):
        values[f"needs.{name}.result"] = "success"
    for name in (
        "candidate-canary-unit",
        "source-canary-unit",
        "test-apple-silicon",
        "changed-lines-coverage",
        "l1-smoke",
    ):
        values[f"needs.{name}.result"] = "skipped"
    values.update(updates)
    script = jobs()["tests"]["steps"][0]["run"]
    rendered = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda m: values.get(m[1], ""), script)
    result = subprocess.run(["bash", "-c", rendered], capture_output=True, text=True)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    if passes:
        assert "full combined candidate" in result.stdout


@pytest.mark.parametrize(
    ("event", "head_repo", "head_ref", "enabled", "expected"),
    [
        ("pull_request", "owner/repo", "feature/server", "true", True),
        ("pull_request", "fork/repo", "feature/server", "true", False),
        ("pull_request", "owner/repo", "train/feature", "true", False),
        ("pull_request", "owner/repo", "mergify/merge-queue/0123456789", "true", False),
        ("pull_request", "owner/repo", "feature/server", "false", False),
        ("push", "owner/repo", "", "true", False),
        ("merge_group", "owner/repo", "", "true", False),
    ],
)
def test_actual_workflow_classifier_keeps_candidates_main_and_forks_full(
    tmp_path,
    event,
    head_repo,
    head_ref,
    enabled,
    expected,
):
    script = next(
        s["run"] for s in jobs()["changes"]["steps"] if s.get("id") == "policy"
    )
    # Run the workflow itself with only the diff transport stubbed. Classifier
    # is the real checked-in CLI, not a second implementation of its routing.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text('#!/bin/sh\nprintf "rapid_mlx/server.py\\n"\n')
    git.chmod(0o755)
    python = bindir / "python"
    python.symlink_to(sys.executable)
    output = tmp_path / "output"
    script = script.replace("/tmp/changed-paths", str(tmp_path / "paths"))
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        GITHUB_OUTPUT=str(output),
        EVENT_NAME=event,
        HEAD_REPO=head_repo,
        REPO="owner/repo",
        HEAD_REF=head_ref,
        PR_BASE_SHA="a" * 40,
        GITHUB_SHA="b" * 40,
        CANARY_ENABLED="false",
        SOURCE_PREFLIGHT_ENABLED=enabled,
        CANDIDATE_SHADOW_ENABLED="false",
        CANDIDATE_CANARY_ENABLED="false",
    )
    result = subprocess.run(
        ["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["source_preflight"] == str(expected).lower()
    assert len(json.loads(values["test_matrix"])["include"]) == (3 if expected else 9)
    if head_ref.startswith(("train/", "mergify/")) or event != "pull_request":
        assert values["full_gate"] == "true"
        assert len(json.loads(values["l1_matrix"])["include"]) == 5


def test_only_source_apple_and_changedlines_deferred_coverage_still_required():
    workflow = jobs()
    for name in ("test-apple-silicon", "changed-lines-coverage"):
        assert (
            "needs.changes.outputs.source_preflight != 'true'" in workflow[name]["if"]
        )
    for name in ("test-matrix", "linux-coverage"):
        assert "source_preflight" not in workflow[name]["if"]
    assert "linux-coverage" in workflow["tests"]["needs"]
    assert "test-apple-silicon" in workflow["tests"]["needs"]
    assert "changed-lines-coverage" in workflow["tests"]["needs"]


def test_real_full_evidence_validator_rejects_source_prefilter_jobs():
    from scripts import queue_tree_evidence as evidence

    names = ["tests", "linux-coverage", "lint", "engine-contracts", "type-check"]
    names += [f"test-matrix (3.11, {i})" for i in (1, 2, 3)]
    records = [
        dict(id=i, name=name, status="completed", conclusion="success")
        for i, name in enumerate(names, 1)
    ]

    class Client:
        def jobs(self, run_id):
            assert run_id == 123
            return records

    with pytest.raises(evidence.EvidenceError):
        evidence._validate_ci_jobs(Client(), {"id": 123, "run_attempt": 1})


@pytest.mark.parametrize(
    "control",
    [
        "tests/test_check_gha_pinning.py",
        "tests/test_mergify_mlx_attestation.py",
        "tests/test_check_release_ci.py",
        "tests/test_integration_collection_policy.py",
        "tests/test_pr_validate_runner.py",
        "tests/test_mirror_drift_workflow.py",
        "tests/test_dev_test_script.py",
        "tests/test_train_gates_matches_ci.py",
        "tests/test_no_mlx_marker_contract.py",
        "tests/test_mlx_bound_guard.py",
        "tests/test_desktop_promotion.py",
        "tests/test_community_benchmark_release_provenance.py",
    ],
)
@pytest.mark.parametrize("with_engine", [False, True])
def test_controller_and_collection_regressions_keep_full_source_in_actual_cli(
    tmp_path, control, with_engine
):
    paths = [control] + (["rapid_mlx/server.py"] if with_engine else [])
    policy = classify_policy(paths, source_preflight=True)
    assert not policy.source_preflight
    assert policy.linux_matrix_mode == "full"
    script = next(
        s["run"] for s in jobs()["changes"]["steps"] if s.get("id") == "policy"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text("#!/bin/sh\ncat <<'PATHS'\n" + "\n".join(paths) + "\nPATHS\n")
    git.chmod(0o755)
    (bindir / "python").symlink_to(sys.executable)
    output = tmp_path / "output"
    script = script.replace("/tmp/changed-paths", str(tmp_path / "paths"))
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        GITHUB_OUTPUT=str(output),
        EVENT_NAME="pull_request",
        HEAD_REPO="owner/repo",
        REPO="owner/repo",
        HEAD_REF="feature/test",
        PR_BASE_SHA="a" * 40,
        GITHUB_SHA="b" * 40,
        CANARY_ENABLED="false",
        SOURCE_PREFLIGHT_ENABLED="true",
        CANDIDATE_SHADOW_ENABLED="false",
    )
    result = subprocess.run(
        ["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["source_preflight"] == "false"
    assert len(json.loads(outputs["test_matrix"])["include"]) == 9


@pytest.mark.parametrize(
    "doc",
    [
        "README.md",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "LICENSE",
        "docs/usage/community-model.md",
    ],
)
def test_documentation_does_not_duplicate_feature_source_full_checks(tmp_path, doc):
    paths = ["rapid_mlx/server.py", doc]
    assert classify_policy(paths, source_preflight=True).source_preflight
    doc_only = classify_policy([doc], source_preflight=True)
    assert not doc_only.source_preflight
    assert not doc_only.lanes.engine
    promoted = classify_policy(paths, source_preflight=True, force_full=True)
    assert not promoted.source_preflight
    assert len(json.loads(promoted.as_outputs()["test_matrix"])["include"]) == 9
    script = next(
        s["run"] for s in jobs()["changes"]["steps"] if s.get("id") == "policy"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text("#!/bin/sh\ncat <<'PATHS'\n" + "\n".join(paths) + "\nPATHS\n")
    git.chmod(0o755)
    (bindir / "python").symlink_to(sys.executable)
    output = tmp_path / "output"
    script = script.replace("/tmp/changed-paths", str(tmp_path / "paths"))
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        GITHUB_OUTPUT=str(output),
        EVENT_NAME="pull_request",
        HEAD_REPO="owner/repo",
        REPO="owner/repo",
        HEAD_REF="feature/server",
        PR_BASE_SHA="a" * 40,
        GITHUB_SHA="b" * 40,
        CANARY_ENABLED="false",
        SOURCE_PREFLIGHT_ENABLED="true",
        CANDIDATE_SHADOW_ENABLED="false",
    )
    result = subprocess.run(
        ["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["source_preflight"] == "true"
    assert outputs["source_canary"] == "false"
    assert len(json.loads(outputs["test_matrix"])["include"]) == 3


@pytest.mark.parametrize(
    "other",
    [
        "../docs/setup.md",
        "docs/../scripts/classify_ci_changes.py",
        "/docs/setup.md",
        "new-docs/setup.md",
        ".github/workflows/ci.yml",
        "tests/test_mergify_mlx_attestation.py",
        "tests/conftest.py",
        "apps/rapid-mac/App.swift",
        "pyproject.toml",
    ],
)
def test_documentation_cannot_neutralize_invalid_control_or_cross_product(other):
    paths = ["rapid_mlx/server.py", "README.md", other]
    policy = classify_policy(paths, source_preflight=True)
    assert not policy.source_preflight
    assert policy.linux_matrix_mode == "full"


def test_documentation_does_not_expand_the_mapped_source_namespace():
    policy = classify_policy(["rapid_mlx/_banner.py", "README.md"], source_canary=True)
    assert not policy.source_canary_tests
    assert policy.linux_matrix_mode == "full"
