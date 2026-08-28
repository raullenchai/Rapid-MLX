# SPDX-License-Identifier: Apache-2.0
"""Contracts for the repository-scoped interactive macOS GUI runner."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
PLIST = ROOT / "scripts/actions-runner-mini-gui.plist.template"
DOC = ROOT / "docs/development/self-hosted-mac-runner.md"


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text())


def _selector_run() -> str:
    return next(
        step
        for step in _workflow()["jobs"]["gui-runner"]["steps"]
        if step["name"] == "Select GUI runner"
    )["run"]


def _execute_selector(
    tmp_path: Path,
    *,
    trusted_head: bool,
    token: str,
    runners: dict[str, object] | None = None,
    api_fails: bool = False,
) -> tuple[dict[str, str], bool]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    called = tmp_path / "called"
    payload = tmp_path / "runners.json"
    payload.write_text(json.dumps(runners or {"total_count": 0, "runners": []}))
    mock_gh = bin_dir / "gh"
    mock_gh.write_text(
        """#!/bin/bash
set -euo pipefail
touch "$MOCK_GH_CALLED"
if [ "$MOCK_GH_FAILS" = true ]; then
  exit 1
fi
cat "$MOCK_RUNNERS_JSON"
"""
    )
    mock_gh.chmod(0o755)
    output = tmp_path / "github-output"
    env = os.environ | {
        "GITHUB_OUTPUT": str(output),
        "IS_TRUSTED_HEAD": str(trusted_head).lower(),
        "MINI_GUI_RUNNER_READ_TOKEN": token,
        "MOCK_GH_CALLED": str(called),
        "MOCK_GH_FAILS": str(api_fails).lower(),
        "MOCK_RUNNERS_JSON": str(payload),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO": "test/repo",
    }
    subprocess.run(["bash", "-c", _selector_run()], check=True, env=env)
    values = dict(
        line.split("=", maxsplit=1) for line in output.read_text().splitlines()
    )
    return values, called.exists()


def test_runner_selector_fails_safe_to_hosted_for_external_heads_and_api_failures():
    job = _workflow()["jobs"]["gui-runner"]
    step = next(step for step in job["steps"] if step["name"] == "Select GUI runner")
    run = step["run"]

    assert "head.repo.full_name == github.repository" in step["env"][
        "IS_TRUSTED_HEAD"
    ]
    assert "selected='[\"macos-15\"]'" in run
    assert 'if [ "$IS_TRUSTED_HEAD" != true ]' in run
    assert 'if [ -z "$MINI_GUI_RUNNER_READ_TOKEN" ]' in run
    assert 'elif ! runners=$(GH_TOKEN="$MINI_GUI_RUNNER_READ_TOKEN"' in run
    assert 'gh api "repos/${REPO}/actions/runners?per_page=100"' in run
    assert "falling back to hosted macOS" in run


def test_runner_selector_executes_external_missing_token_and_api_fallbacks(tmp_path):
    external, called = _execute_selector(
        tmp_path / "external", trusted_head=False, token="secret"
    )
    assert external == {"runs_on": '["macos-15"]', "runner_kind": "hosted"}
    assert not called

    missing, called = _execute_selector(
        tmp_path / "missing", trusted_head=True, token=""
    )
    assert missing == {"runs_on": '["macos-15"]', "runner_kind": "hosted"}
    assert not called

    failed, called = _execute_selector(
        tmp_path / "failed", trusted_head=True, token="secret", api_fails=True
    )
    assert failed == {"runs_on": '["macos-15"]', "runner_kind": "hosted"}
    assert called


def test_runner_selector_requires_one_online_exact_label_match():
    step = next(
        step
        for step in _workflow()["jobs"]["gui-runner"]["steps"]
        if step["name"] == "Select GUI runner"
    )
    run = step["run"]
    assert '.status == "online"' in run
    for label in ("self-hosted", "macOS", "mini-gui"):
        assert f'.name == "{label}"' in run
    assert 'selected=\'["self-hosted","macOS","mini-gui"]\'' in run


def test_runner_selector_executes_online_and_offline_routing(tmp_path):
    labels = [
        {"name": "self-hosted"},
        {"name": "macOS"},
        {"name": "ARM64"},
        {"name": "mini-gui"},
    ]
    online, called = _execute_selector(
        tmp_path / "online",
        trusted_head=True,
        token="secret",
        runners={
            "total_count": 1,
            "runners": [{"status": "online", "busy": False, "labels": labels}],
        },
    )
    assert online == {
        "runs_on": '["self-hosted","macOS","mini-gui"]',
        "runner_kind": "mini-gui",
    }
    assert called

    offline, called = _execute_selector(
        tmp_path / "offline",
        trusted_head=True,
        token="secret",
        runners={
            "total_count": 1,
            "runners": [{"status": "offline", "busy": False, "labels": labels}],
        },
    )
    assert offline == {"runs_on": '["macos-15"]', "runner_kind": "hosted"}
    assert called


def test_gui_jobs_use_selector_output_and_generic_build_stays_hosted():
    jobs = _workflow()["jobs"]
    for name in ("gui-app-build", "gui-golden-flows"):
        job = jobs[name]
        assert "gui-runner" in job["needs"]
        assert job["runs-on"] == "${{ fromJSON(needs.gui-runner.outputs.runs_on) }}"

    assert jobs["build"]["runs-on"] == "macos-15"


def test_launchagent_template_is_interactive_and_holds_awake_assertions():
    text = PLIST.read_text()
    assert "{{RunnerRoot}}/runsvc.sh" in text
    assert "{{User}}" in text and "{{UserHome}}" in text
    assert "<string>/usr/bin/caffeinate</string>" in text
    assert "<string>-dimsu</string>" in text
    assert "<key>ProcessType</key>\n    <string>Interactive</string>" in text
    assert "<key>SessionCreate</key>\n    <true/>" in text


def test_operator_doc_pins_security_pause_and_recovery_contracts():
    text = DOC.read_text()
    assert "Never run code from a fork" in text
    assert "MINI_GUI_RUNNER_READ_TOKEN" in text
    assert "Administration: read" in text
    assert "./svc.sh stop" in text
    assert "./svc.sh start" in text
    assert "~/Library/LaunchAgents" in text
    assert "/Volumes/Extreme SSD" not in text
