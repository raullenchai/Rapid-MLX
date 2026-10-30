# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from scripts.pr_validate.steps import mac_only_tests


def test_selection(tmp_path):
    files = {
        ".github/workflows/ci.yml": (
            "  test-apple-silicon:\n    tests/test_roster.py\n"
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
    assert mac_only_tests.select_mac_only_files(tmp_path) == [
        "tests/sub/test_yes.py",
        "tests/test_other.py",
        "tests/test_yes.py",
    ]


@pytest.mark.parametrize(("code", "status"), [(0, "pass"), (1, "fail"), (5, "skip")])
def test_run(monkeypatch, tmp_path, code, status):
    ctx = SimpleNamespace(
        repo_root=tmp_path,
        artifact_path=lambda name: tmp_path / name,
        run_log=lambda msg: None,
    )
    monkeypatch.setattr(
        mac_only_tests, "select_mac_only_files", lambda root: ["tests/test_a.py"]
    )
    monkeypatch.setattr(mac_only_tests, "pytest_parallel_args", lambda: ([], "serial"))
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=code, stdout="", stderr="")

    monkeypatch.setattr(mac_only_tests, "_run_group_bounded", run)
    result = mac_only_tests.MacOnlyTestsStep().run(ctx)
    assert result.status == status
    assert "tests/test_a.py" in calls[0] and "1 files" in result.summary
