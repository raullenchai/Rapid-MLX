"""Contracts for Desktop PR evidence and bounded release CI waits."""

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import check_release_ci
from scripts.ci_test_durations import durations_from_logs

ROOT = Path(__file__).resolve().parents[1]


def jobs(workflow):
    return yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]


@pytest.mark.parametrize(
    "event", ["pull_request", "push", "schedule", "workflow_dispatch"]
)
@pytest.mark.parametrize("head_repo", ["owner/repo", "contributor/fork"])
@pytest.mark.parametrize("head_ref", ["feature/images", "train/images"])
def test_desktop_promotion_preserves_trust_and_pr_flow_subset(
    tmp_path, event, head_repo, head_ref
):
    step = next(
        s
        for s in jobs("rapid-mac-ci.yml")["changes"]["steps"]
        if s.get("id") == "policy"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text(
        "#!/bin/sh\necho apps/rapid-mac/Sources/Rapid/Images/ImageClient.swift\n"
    )
    git.chmod(0o755)
    (bindir / "python").symlink_to(sys.executable)
    output = tmp_path / "outputs"
    script = step["run"].replace("/tmp/changed-paths", str(tmp_path / "paths"))
    subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=os.environ
        | {
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "EVENT_NAME": event,
            "HEAD_REPO": head_repo,
            "REPO": "owner/repo",
            "HEAD_REF": head_ref,
            "PR_BASE_SHA": "a" * 40,
            "GITHUB_SHA": "b" * 40,
        },
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert (
        values["full_gate"]
        == str(event != "pull_request" or head_repo == "owner/repo").lower()
    )
    assert values["desktop"] == "true"
    if event == "pull_request":
        assert json.loads(values["gui_flows"]) == ["image-generation"]
        assert values["gui_flow_count"] == "1"
    else:
        assert int(values["gui_flow_count"]) > 1


@pytest.mark.parametrize("job_name", ["gui-app-build", "gui-golden-flows"])
@pytest.mark.parametrize("count", ["0", "1"])
@pytest.mark.parametrize("full", ["true", "false"])
def test_gui_jobs_skip_empty_matrices_and_untrusted_heads(job_name, count, full):
    condition = jobs("rapid-mac-ci.yml")[job_name]["if"]
    values = {
        "!cancelled()": "True",
        "needs.changes.result": "'success'",
        "needs.gui-app-build.result": "'success'",
        "needs.changes.outputs.desktop": "'true'",
        "needs.changes.outputs.full_gate": repr(full),
        "needs.changes.outputs.gui_flow_count": repr(count),
    }
    for key, value in values.items():
        condition = condition.replace(key, value)
    assert eval(condition.replace("&&", "and"), {"__builtins__": {}}) == (
        full == "true" and count != "0"
    )


def test_release_toolchain_build_matches_release_lane_and_is_required():
    ci = jobs("rapid-mac-ci.yml")
    job = ci["release-toolchain-build"]
    assert (
        job["runs-on"]
        == jobs("rapid-mac-release.yml")["build"]["runs-on"]
        == "macos-15"
    )
    assert job["permissions"] == {"contents": "read"}
    assert job["needs"] == "changes"
    assert job["timeout-minutes"] == 30
    assert (
        job["if"]
        == "!cancelled() && needs.changes.result == 'success' && needs.changes.outputs.desktop == 'true'"
    )
    assert job["defaults"]["run"]["working-directory"] == "apps/rapid-mac"
    assert job["steps"][0]["uses"] == ci["build"]["steps"][0]["uses"]
    assert re.fullmatch(r"actions/checkout@[0-9a-f]{40}", job["steps"][0]["uses"])
    assert [s["run"] for s in job["steps"][1:]] == [
        "xcodebuild -version\nswift --version\n",
        "swift build -c release",
    ]
    assert "secrets." not in json.dumps(job)
    aggregate = ci["desktop-tests"]
    assert "release-toolchain-build" in aggregate["needs"]
    env = aggregate["steps"][0]["env"]
    assert env["RELEASE_BUILD"] == "${{ needs.release-toolchain-build.result }}"
    assert env["GUI_FLOW_COUNT"] == "${{ needs.changes.outputs.gui_flow_count }}"


def test_release_ci_waits_have_three_hour_deadlines_and_job_margin(monkeypatch):
    workflows = jobs("auto-release.yml")
    for name in ("release-final-ci", "release"):
        job = workflows[name]
        waits = [
            s["run"]
            for s in job["steps"]
            if "scripts/check_release_ci.py" in s.get("run", "")
        ]
        assert len(waits) == 1
        assert "--deadline-min 180" in waits[0]
        assert job.get("timeout-minutes", 360) >= 195
    assert (
        inspect.signature(check_release_ci.verify).parameters["deadline_sec"].default
        == 10800
    )
    captured = {}

    def verify(**kwargs):
        captured.update(kwargs)
        return ["passed"]

    monkeypatch.setattr(check_release_ci, "verify", verify)
    assert (
        check_release_ci.main(
            [
                "--source-sha",
                "a" * 40,
                "--repo",
                "owner/repo",
                "--require",
                "ci.yml:tests",
            ]
        )
        == 0
    )
    assert captured["deadline_sec"] == 10800


def test_timestamp_weights_sum_results_and_reset_between_sessions_and_logs(tmp_path):
    first = tmp_path / "first.log"
    first.write_text(
        "2026-01-01T00:00:00.0000000Z tests/test_a.py::test_one PASSED [ 25%]\n"
        "2026-01-01T00:00:01.2500000Z unrelated output\n"
        "2026-01-01T00:00:02.0000000Z tests/test_a.py::test_two SKIPPED [ 50%]\n"
        "2026-01-01T00:00:03.5000000Z tests/test_b.py::test_one XFAIL [100%]\n"
        "2026-01-01T00:01:00.0000000Z ==== test session starts ====\n"
        "2026-01-01T00:01:10.0000000Z tests/headless_mlx/test_x.py::test_one PASSED [100%]\n"
    )
    second = tmp_path / "second.log"
    second.write_text(
        "2026-01-01T00:02:00.0000000Z tests/test_b.py::test_two PASSED [ 50%]\n"
        "2026-01-01T00:02:00.2500000Z tests/test_b.py::test_three PASSED [100%]\n"
    )
    assert durations_from_logs([first, second]) == {
        "tests/test_a.py": 2000,
        "tests/test_b.py": 1750,
    }
    output = tmp_path / "weights.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/ci_test_durations.py",
            "--from-log",
            str(first),
            str(second),
            "--source",
            "run-url",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(output.read_text())
    assert payload["files"] == durations_from_logs([first, second])
    assert payload["source"] == "run-url"
    assert payload["unit"] == "milliseconds"
