"""A headless agent must not launch a GUI with the harness's temporary HOME."""

import os
import shlex
import sys

import pytest

from rapid_mlx.agents.testing import (
    TestStatus,
    _agent_query,
    _install_gui_launch_guard,
    _test_e2e_chat,
)


@pytest.mark.parametrize("launcher", ["open", "xdg-open", "browser"])
def test_gui_launch_fails_even_if_agent_prints_expected_answer(
    tmp_path, monkeypatch, launcher
):
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("CFFIXED_USER_HOME", str(real_home))
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    browser = _install_gui_launch_guard(fake_home)
    env = {
        "HOME": str(fake_home),
        "PATH": f"{browser.parent}{os.pathsep}{os.environ['PATH']}",
        "BROWSER": str(browser),
    }
    command = "os.environ['BROWSER']" if launcher == "browser" else repr(launcher)
    script = (
        "import os, subprocess; "
        f"subprocess.run([{command}, 'https://example.invalid'], check=False); "
        "print('777777')"
    )
    query_cmd = f"python -c {shlex.quote(script)}"
    result = _test_e2e_chat(
        sys.executable,
        query_cmd,
        timeout=5,
        cwd=str(tmp_path),
        env_overrides=env,
    )
    assert result.status is TestStatus.FAIL
    assert "agent tried to open a browser" in result.message
    assert (
        f"sign in to {os.path.basename(sys.executable)} normally first"
        in result.message
    )
    assert "then re-run --test" in result.message
    assert os.environ["HOME"] == str(real_home)
    assert os.environ["CFFIXED_USER_HOME"] == str(real_home)
    assert list(real_home.iterdir()) == []


def test_child_never_receives_cffixed_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CFFIXED_USER_HOME", str(tmp_path / "real-home"))
    script = "import os; print(os.environ.get('CFFIXED_USER_HOME', 'absent'))"
    output, error = _agent_query(
        sys.executable,
        f"python -c {shlex.quote(script)}",
        "ignored",
        env_overrides={"CFFIXED_USER_HOME": "also-forbidden"},
    )
    assert error is None
    assert output.strip() == "absent"


def test_unrelated_browser_override_does_not_touch_its_directory(tmp_path):
    marker = tmp_path / ".gui-launch-attempted"
    marker.write_text("keep", encoding="utf-8")
    output, error = _agent_query(
        sys.executable,
        f"python -c {shlex.quote('print(123)')}",
        "ignored",
        env_overrides={"BROWSER": str(tmp_path / "other-browser")},
    )
    assert error is None
    assert output.strip() == "123"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_launch_attempt_remains_failure_if_agent_times_out(tmp_path):
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    browser = _install_gui_launch_guard(fake_home)
    script = (
        "import subprocess, time; "
        "subprocess.run(['open', 'https://example.invalid']); "
        "time.sleep(5)"
    )
    output, error = _agent_query(
        sys.executable,
        f"python -c {shlex.quote(script)}",
        "ignored",
        timeout=2,
        env_overrides={
            "HOME": str(fake_home),
            "PATH": f"{browser.parent}{os.pathsep}{os.environ['PATH']}",
            "BROWSER": str(browser),
        },
    )
    assert output is None
    assert "agent tried to open a browser" in error
