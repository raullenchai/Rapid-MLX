# SPDX-License-Identifier: Apache-2.0
"""Execute the real Desktop classifier step against complete main-push diffs."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in (
        "scripts/classify_ci_changes.py",
        "scripts/select_gui_flows.py",
        "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "CI test")
    _git(repo, "config", "user.email", "ci-test@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "baseline")
    return repo


def _commit(repo, paths):
    for relative in paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            path.read_text() + "\nchange\n" if path.exists() else "change\n"
        )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "change", "--allow-empty")
    return _git(repo, "rev-parse", "HEAD")


def _execute(repo, before, **overrides):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["changes"]["steps"]
    step = next(s for s in steps if s.get("name") == "Classify desktop lane")
    output = repo.parent / "outputs"
    output.write_text("")
    env = os.environ | {
        "EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": _git(repo, "rev-parse", "HEAD"),
        "PUSH_BEFORE_SHA": before,
        "PUSH_FORCED": "false",
        "RUNNER_TEMP": str(repo.parent),
        "GITHUB_OUTPUT": str(output),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    env.update(overrides)
    subprocess.run(["bash", "-c", step["run"]], cwd=repo, env=env, check=True)
    result = dict(line.split("=", 1) for line in output.read_text().splitlines())
    # Every required Desktop route retains the entire manifest and full gate.
    all_flows = json.loads(
        subprocess.run(
            [sys.executable, "scripts/select_gui_flows.py"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert result["full_gate"] == "true"
    assert json.loads(result["gui_flows"]) == all_flows
    assert int(result["gui_flow_count"]) == len(all_flows)
    return result


@pytest.mark.parametrize(
    ("paths", "desktop"),
    [
        (["rapid_mlx/server.py"], "false"),
        (["rapid_mlx/server.py", "docs/example.md"], "false"),
        (["docs/example.md"], "false"),
        (["apps/rapid-mac/Sources/Rapid/Chat/Test.swift"], "true"),
        ([".github/workflows/ci.yml"], "true"),
        (["new-product/unknown.py"], "true"),
        (
            ["rapid_mlx/server.py", "apps/rapid-mac/Sources/Rapid/Chat/Test.swift"],
            "true",
        ),
    ],
)
def test_normal_main_push_classifies_complete_diff(repository, paths, desktop):
    before = _git(repository, "rev-parse", "HEAD")
    _commit(repository, paths)
    result = _execute(repository, before)
    assert result["desktop"] == desktop


@pytest.mark.parametrize(
    "overrides",
    [
        {"PUSH_BEFORE_SHA": ""},
        {"PUSH_BEFORE_SHA": "0" * 40},
        {"PUSH_BEFORE_SHA": "f" * 40},
        {"PUSH_BEFORE_SHA": "HEAD~1"},
        {"PUSH_BEFORE_SHA": "--all"},
        {"PUSH_FORCED": "true"},
        {"PUSH_FORCED": ""},
        {"GITHUB_SHA": "f" * 40},
        {"GITHUB_SHA": "HEAD"},
        {"GITHUB_REF": "refs/heads/other"},
        {"EVENT_NAME": "merge_group"},
    ],
)
def test_unqualified_push_or_merge_group_stays_full(repository, overrides):
    before = _git(repository, "rev-parse", "HEAD")
    _commit(repository, ["rapid_mlx/server.py"])
    result = _execute(repository, before, **overrides)
    assert result["desktop"] == result["engine"] == "true"


def test_empty_diff_stays_full(repository):
    head = _git(repository, "rev-parse", "HEAD")
    assert _execute(repository, head)["desktop"] == "true"


def test_nonancestor_before_stays_full(repository):
    before = _commit(repository, ["rapid_mlx/branch_only.py"])
    _git(repository, "checkout", "--detach", "HEAD~1")
    _commit(repository, ["rapid_mlx/server.py"])
    assert _execute(repository, before)["desktop"] == "true"


def test_valid_after_commit_must_match_checkout(repository):
    before = _git(repository, "rev-parse", "HEAD")
    after = _commit(repository, ["rapid_mlx/server.py"])
    _commit(repository, ["docs/later.md"])
    assert _execute(repository, before, GITHUB_SHA=after)["desktop"] == "true"


def test_multicommit_push_keeps_earlier_desktop_change(repository):
    before = _git(repository, "rev-parse", "HEAD")
    _commit(repository, ["apps/rapid-mac/Sources/Rapid/Chat/Test.swift"])
    _commit(repository, ["rapid_mlx/server.py"])
    assert _execute(repository, before)["desktop"] == "true"


def test_rename_preserves_removed_desktop_path(repository):
    before = _commit(repository, ["apps/rapid-mac/Sources/Rapid/Chat/Test.swift"])
    destination = repository / "rapid_mlx/moved.swift"
    destination.parent.mkdir(parents=True)
    _git(
        repository,
        "mv",
        "apps/rapid-mac/Sources/Rapid/Chat/Test.swift",
        str(destination),
    )
    _git(repository, "commit", "-qm", "move out of Desktop")
    assert _execute(repository, before)["desktop"] == "true"
