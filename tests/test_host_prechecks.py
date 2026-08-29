from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRECHECK = ROOT / "apps/rapid-mac/scripts/dogfood-host-precheck.sh"
LOCK = ROOT / "scripts/large-model-run.py"
TART = ROOT / "scripts/tart-guest-ready.sh"


def test_host_precheck_names_lock_and_screensaver_faults() -> None:
    env = dict(os.environ, RAPID_HOST_SAFETY_TESTING="1", RAPID_HOST_TEST_IDLE_TIME="0")
    locked = subprocess.run(
        [str(PRECHECK)],
        env=dict(env, RAPID_HOST_TEST_LOCKED="true"),
        text=True,
        capture_output=True,
    )
    assert locked.returncode == 1
    assert "console is locked" in locked.stderr
    idle = subprocess.run(
        [str(PRECHECK)],
        env=dict(env, RAPID_HOST_TEST_LOCKED="false", RAPID_HOST_TEST_IDLE_TIME="300"),
        text=True,
        capture_output=True,
    )
    assert idle.returncode == 1
    assert "idleTime must be 0" in idle.stderr


def test_ci_precheck_executes_command_without_host_assumptions() -> None:
    result = subprocess.run(
        [str(PRECHECK), "--", "/usr/bin/printf", "ok"],
        env=dict(os.environ, CI="true"),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert result.stdout == "ok"


@pytest.mark.skipif(shutil.which("lockf") is None, reason="BSD lockf required")
def test_two_large_loads_serialize_on_one_host_lock(tmp_path: Path) -> None:
    lock = tmp_path / "large.lock"
    ledger = tmp_path / "ledger"
    child = (
        "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); name=sys.argv[2]; "
        "p.open('a').write(name+'-start\\n'); time.sleep(.35); "
        "p.open('a').write(name+'-end\\n')"
    )
    env = dict(
        os.environ,
        RAPID_HOST_SAFETY_TESTING="1",
        RAPID_LARGE_MODEL_TEST_AVAILABLE_GB="100",
    )
    base = [
        sys.executable,
        str(LOCK),
        "--working-set-gb",
        "21",
        "--lock-file",
        str(lock),
        "--",
        sys.executable,
        "-c",
        child,
        str(ledger),
    ]
    first = subprocess.Popen([*base, "a"], env=env)
    deadline = time.monotonic() + 2
    while (
        not ledger.exists() or "a-start" not in ledger.read_text()
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ledger.exists() and "a-start" in ledger.read_text()
    second = subprocess.Popen([*base, "b"], env=env)
    assert first.wait(timeout=10) == 0
    assert second.wait(timeout=10) == 0
    assert ledger.read_text().splitlines() == ["a-start", "a-end", "b-start", "b-end"]


def test_memory_is_rechecked_after_lock_is_acquired(tmp_path: Path) -> None:
    env = dict(
        os.environ,
        RAPID_HOST_SAFETY_TESTING="1",
        RAPID_LARGE_MODEL_TEST_AVAILABLE_GB="22",
        RAPID_LARGE_MODEL_LOCK_HELD="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(LOCK),
            "--working-set-gb",
            "21",
            "--lock-file",
            str(tmp_path / "lock"),
            "--",
            "/usr/bin/true",
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "insufficient memory under lock" in result.stderr


@pytest.mark.skipif(shutil.which("lockf") is None, reason="BSD lockf required")
def test_termination_reaches_locked_server_descendants(tmp_path: Path) -> None:
    pid_file = tmp_path / "descendant.pid"
    child = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    env = dict(
        os.environ,
        RAPID_HOST_SAFETY_TESTING="1",
        RAPID_LARGE_MODEL_TEST_AVAILABLE_GB="100",
    )
    wrapper = subprocess.Popen(
        [
            sys.executable,
            str(LOCK),
            "--working-set-gb",
            "21",
            "--lock-file",
            str(tmp_path / "lock"),
            "--",
            sys.executable,
            "-c",
            child,
            str(pid_file),
        ],
        env=env,
    )
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_file.exists()
    descendant = int(pid_file.read_text())
    wrapper.terminate()
    wrapper.wait(timeout=5)
    deadline = time.monotonic() + 5
    while (
        subprocess.run(
            ["ps", "-p", str(descendant)], capture_output=True, check=False
        ).returncode
        == 0
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    assert (
        subprocess.run(
            ["ps", "-p", str(descendant)], capture_output=True, check=False
        ).returncode
        != 0
    )


def test_tart_ready_retries_then_succeeds(tmp_path: Path) -> None:
    fake = tmp_path / "tart"
    count = tmp_path / "count"
    fake.write_text(
        '#!/bin/sh\nn=0; [ -f "$COUNT" ] && n=$(cat "$COUNT"); '
        'n=$((n+1)); echo $n > "$COUNT"; [ $n -ge 2 ]\n'
    )
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", COUNT=str(count))
    result = subprocess.run(
        [str(TART), "--timeout", "3", "--interval", "0", "guest"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "guest is ready" in result.stdout


def test_real_dogfood_entrypoints_share_the_host_guards() -> None:
    isolate = (ROOT / "apps/rapid-mac/scripts/dogfood-isolate.sh").read_text()
    mvp = (ROOT / "scripts/run_dogfood_mvp.sh").read_text()
    ax = (ROOT / "apps/rapid-mac/scripts/gui-ax-smoke.sh").read_text()
    golden = (ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh").read_text()

    assert 'working_set_gb="\\${DOGFOOD_WORKING_SET_GB:-21}"' in isolate
    assert '"$SAFETY_DIR/dogfood-host-precheck.sh"' in isolate
    assert '"$SAFETY_DIR/large-model-run.py"' in isolate
    assert '"$ROOT/scripts/large-model-run.py"' in mvp
    assert 'size_args=(--model "$MODEL")' in mvp
    assert 'exec "$SCRIPT_DIR/dogfood-host-precheck.sh"' in ax
    assert 'exec "$ROOT/scripts/dogfood-host-precheck.sh"' in golden
    # The launch helper centralizes the host guard for localized, default,
    # and restart launches.
    assert golden.count("DOGFOOD_WORKING_SET_GB=0.1") == 2
    assert golden.count("launch_persona_app ") == 3
    assert "launch_persona_app append" in golden
