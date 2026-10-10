"""Source CPU routing preserves integration authority and type/collection guards."""

import json
import os
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
    "control",
    [
        "tests/headless_mlx/conftest.py",
        "tests/headless_mlx/__init__.py",
        "tests/headless_mlx/fixtures/model.py",
        "tests/headless_mlx/nested/test_engine.py",
        "tests/headless_mlx/test_ci_evidence.py",
        "tests/headless_mlx/test_release_policy.py",
        "tests/headless_mlx/test_engine_workflow.py",
        "tests/conftest.py",
        "pytest.ini",
        "config/mypy-requirements.txt",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
        "scripts/ci_test_shard.py",
        "tests/../tests/headless_mlx/test_engine.py",
    ],
)
def test_headless_and_type_baseline_do_not_neutralize_control_changes(control):
    paths = [
        "rapid_mlx/server.py",
        "tests/headless_mlx/test_engine_lifecycle.py",
        "config/mypy-error-baseline.txt",
        control,
    ]
    policy = classify_policy(paths, source_preflight=True)
    assert not policy.source_preflight
    assert policy.linux_matrix_mode == "full"


def test_headless_source_tests_remain_enrolled_once_and_in_separate_process():
    files = discover(ROOT, "headless")
    shards = partition(files, 3)
    assert sorted(item.path for shard in shards for item in shard) == sorted(
        item.path for item in files
    )
    assert len({item.path for shard in shards for item in shard}) == len(files)
    for item in files:
        policy = classify_policy([item.path], source_preflight=True)
        if item.path.split("/")[-1].startswith(("test_ci_", "test_release_")):
            continue
        assert policy.source_preflight, item.path
    command = next(
        s["run"]
        for s in jobs()["test-matrix"]["steps"]
        if "--suite headless" in s.get("run", "")
    )
    assert command.count("pytest \\") == 2
    assert "tests/headless_mlx" in command
    assert "--cov-append" in command
    assert "--shard-count 3" in command


def test_type_baseline_source_fast_route_preserves_existing_type_ratchet():
    workflow = jobs()
    assert "source_preflight" not in workflow["type-check"]["if"]
    assert any(
        "check_mypy_error_budget.py" in s.get("run", "")
        for s in workflow["type-check"]["steps"]
    )
    assert "type-check" in workflow["tests"]["needs"]


@pytest.mark.parametrize(
    "paths",
    [
        [
            "rapid_mlx/community_bench/runner.py",
            "tests/headless_mlx/test_community_bench_text_lane.py",
        ],
        ["rapid_mlx/cli.py", "config/mypy-error-baseline.txt"],
    ],
)
@pytest.mark.parametrize(
    "event,head_repo,head_ref,enabled,expected",
    [
        ("pull_request", "owner/repo", "feature/engine", "true", True),
        ("pull_request", "fork/repo", "feature/engine", "true", False),
        ("pull_request", "owner/repo", "train/feature", "true", False),
        ("pull_request", "owner/repo", "mergify/merge-queue/0123456789", "true", False),
        ("pull_request", "owner/repo", "feature/engine", "false", False),
        ("push", "owner/repo", "", "true", False),
        ("merge_group", "owner/repo", "", "true", False),
    ],
)
def test_actual_engine_workflow_source_context(
    tmp_path, paths, event, head_repo, head_ref, enabled, expected
):
    script = next(
        s["run"] for s in jobs()["changes"]["steps"] if s.get("id") == "policy"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text('#!/bin/sh\nprintf "%s\\n" ' + " ".join(paths) + "\n")
    git.chmod(0o755)
    (bindir / "python").symlink_to(sys.executable)
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
    if event != "pull_request" or head_ref.startswith(("train/", "mergify/")):
        assert values["full_gate"] == "true"
        assert len(json.loads(values["l1_matrix"])["include"]) == 5
