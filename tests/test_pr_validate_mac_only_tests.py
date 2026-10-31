# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.pr_validate.steps import mac_only_tests

ROOT = Path(__file__).resolve().parents[1]


def list_files(*args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.pr_validate.steps.mac_only_tests", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_selection_and_shards():
    result = list_files("--list")
    assert result.returncode == 0, result.stderr
    files = result.stdout.splitlines()
    assert files == mac_only_tests.select_mac_only_files(ROOT)
    shards = []
    for i in range(1, 4):
        result = list_files("--list", "--shard", f"{i}/3")
        assert result.returncode == 0, result.stderr
        shard = result.stdout.splitlines()
        assert shard == files[i - 1 :: 3]
        shards.append(set(shard))
    assert set.union(*shards) == set(files)
    assert sum(map(len, shards)) == len(files)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_cli_deselect_args():
    result = list_files("--deselect-args")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"--deselect={node}" for node in mac_only_tests.HOSTED_DESELECT
    ]
    for node in mac_only_tests.HOSTED_DESELECT:
        path = ROOT / node.split("::", 1)[0]
        assert path.is_relative_to(ROOT / "tests") and path.is_file()


@pytest.mark.parametrize(
    "spec", ["", "1", "1/", "/3", "0/3", "4/3", "1/0", "-1/3", "1/3/4", " 1/3", "1.0/3"]
)
def test_cli_rejects_bad_shards(spec):
    assert list_files("--list", "--shard", spec).returncode == 2


def test_hosted_mac_only_contract():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    job = jobs["test-apple-silicon-mac-only"]
    assert job["runs-on"] == "macos-15"
    assert job["strategy"]["matrix"]["shard"] == [1, 2, 3]
    run = next(
        step["run"] for step in job["steps"] if step.get("name") == "Run mac-only tests"
    )
    assert (
        "scripts.pr_validate.steps.mac_only_tests --list --shard ${{ matrix.shard }}/3"
        in run
    )
    assert "test-apple-silicon-mac-only" in jobs["tests"]["needs"]
    check = next(
        step["run"]
        for step in jobs["tests"]["steps"]
        if step.get("name") == "Check test results"
    )
    assert '${{ needs.test-apple-silicon-mac-only.result }}" != "success"' in check


def test_selection(tmp_path):
    files = {
        ".github/workflows/ci.yml": (
            "  test-apple-silicon:\n    tests/test_roster.py\n"
            "    tests/test_missing.py\n"
            "  other-job:\n    tests/test_other.py\n"
        ),
        "tests/test_other.py": "requires_mlx",
        "tests/test_roster.py": "requires_mlx",
        "tests/test_yes.py": "importorskip('mlx')",
        "tests/sub/test_yes.py": "sys.platform",
        "tests/test_no.py": "pass",
        "tests/integrations/test_skip.py": "requires_mlx",
        "tests/headless_mlx/test_skip.py": "darwin",
        "tests/test_event_loop.py": "darwin",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    assert mac_only_tests.select_roster_files(tmp_path) == ["tests/test_roster.py"]
    assert mac_only_tests.select_mac_only_files(tmp_path) == [
        "tests/sub/test_yes.py",
        "tests/test_other.py",
        "tests/test_yes.py",
    ]


@pytest.mark.parametrize(
    ("codes", "status"),
    [
        ((0, 5), "pass"),
        ((5, 0), "pass"),
        ((1, 0), "fail"),
        ((0, 1), "fail"),
        ((5, 5), "skip"),
    ],
)
def test_run(monkeypatch, tmp_path, codes, status):
    ctx = SimpleNamespace(
        repo_root=tmp_path,
        artifact_path=lambda name: tmp_path / name,
        run_log=lambda msg: None,
    )
    monkeypatch.setattr(
        mac_only_tests, "select_mac_only_files", lambda root: ["tests/test_a.py"]
    )
    monkeypatch.setattr(
        mac_only_tests, "select_roster_files", lambda root: ["tests/test_roster.py"]
    )
    monkeypatch.setattr(mac_only_tests, "pytest_parallel_args", lambda: ([], "serial"))
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=codes[len(calls) - 1], stdout="", stderr="")

    monkeypatch.setattr(mac_only_tests, "_run_group_bounded", run)
    result = mac_only_tests.MacOnlyTestsStep().run(ctx)
    assert result.status == status
    assert len(calls) == 2
    assert "tests/test_a.py" in calls[0] and "1 files" in result.summary
    assert "tests/test_roster.py" in calls[1]
    assert calls[1][-2:] == ["-m", "real_hf_cache or requires_network"]
    assert "roster gated:" in result.summary
