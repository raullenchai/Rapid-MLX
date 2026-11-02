# SPDX-License-Identifier: Apache-2.0
"""Process-level cancellation must include detached descendants."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("worker_lane", [False, True])
def test_signal_kills_bounded_group(tmp_path, signum, worker_lane):
    pidfile = tmp_path / "pids"
    child = (
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
        'Path(sys.argv[1]).write_text(f"{os.getpid()} {p.pid}")\n'
        "time.sleep(60)\n"
    )
    driver = (
        "from scripts.pr_validate import pr_validate as cli\n"
        "from scripts.pr_validate.steps.diff_coverage import _run_group_bounded\n"
        f"cli.run_pipeline = lambda *a, **kw: _run_group_bounded("
        f"{[sys.executable, '-c', child, str(pidfile)]!r}, cwd={str(tmp_path)!r}, timeout=30)\n"
        'cli.main(["1"])\n'
    )
    if worker_lane:
        driver = driver.replace(
            'cli.main(["1"])',
            "import threading\n"
            "run = cli.run_pipeline\n"
            "def pipeline(*a, **kw):\n"
            "    worker = threading.Thread(target=run)\n"
            "    worker.start()\n"
            "    worker.join()\n"
            "cli.run_pipeline = pipeline\n"
            'cli.main(["1"])',
        )
    proc = subprocess.Popen(
        [sys.executable, "-c", driver], cwd=Path(__file__).resolve().parents[1]
    )
    pids = []

    def alive(pid):
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    try:
        deadline = time.monotonic() + 8
        while not pidfile.exists() and time.monotonic() < deadline:
            assert proc.poll() is None, "driver exited before child started"
            time.sleep(0.02)
        assert pidfile.exists()
        pids = [int(pid) for pid in pidfile.read_text().split()]
        assert len(pids) == 2
        proc.send_signal(signum)
        proc.wait(timeout=8)
        deadline = time.monotonic() + 4
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(alive(pid) for pid in pids), (
            "bounded descendants survived cancellation"
        )
        assert proc.returncode == 128 + signum
    finally:
        if pidfile.exists():
            pids = [int(pid) for pid in pidfile.read_text().split()]
        if pids:
            try:
                os.killpg(pids[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
