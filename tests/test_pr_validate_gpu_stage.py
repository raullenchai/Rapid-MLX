# SPDX-License-Identifier: Apache-2.0
"""GPU lock and cache contracts for the local validation gate."""

import fcntl
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from scripts.pr_validate import gpu_cache
from scripts.pr_validate.base import Step, StepResult
from scripts.pr_validate.context import Context
from scripts.pr_validate.gpu_lock import hold_gpu_lock
from scripts.pr_validate.runner import run_pipeline
from scripts.pr_validate.steps.stress_e2e_bench import StressE2EBenchStep

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PR_VALIDATE_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("PR_VALIDATE_GPU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PR_VALIDATE_NO_GPU_CACHE", raising=False)
    monkeypatch.setattr(gpu_cache, "_repo_state", lambda _: (SHA, True))
    ctx = Context(pr_number=1, head_sha=SHA, base_sha="b" * 40, work_dir=tmp_path)
    return ctx, tmp_path / "gpu.lock"


def lock_fd(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_free_lock_released(tmp_path):
    path = tmp_path / "dir" / "lock"
    with hold_gpu_lock(path, log=lambda _: None) as info:
        assert info.mode == "acquired"
    fd = lock_fd(path)
    os.close(fd)


def test_lock_waits_for_other_holder(tmp_path):
    path = tmp_path / "lock"
    fd = lock_fd(path)
    entered = threading.Event()
    result = []

    def worker():
        with hold_gpu_lock(
            path,
            log=lambda _: None,
            poll_seconds=0.02,
            holder_pids=lambda _: [],
            ancestor_pids=lambda: set(),
        ) as info:
            result.append(info)
            entered.set()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert not entered.wait(0.1)
    finally:
        os.close(fd)
    thread.join(2)
    assert not thread.is_alive()
    assert result[0].mode == "acquired" and result[0].waited_seconds > 0


def test_legacy_holder(tmp_path):
    path = tmp_path / "lock"
    fd = lock_fd(path)
    try:
        with hold_gpu_lock(
            path,
            log=lambda _: None,
            holder_pids=lambda _: [4242],
            ancestor_pids=lambda: {4242},
        ) as info:
            assert info.mode == "held-by-caller"
    finally:
        os.close(fd)


def test_non_gpu_pipeline_runs_concurrently_under_lock(setup):
    _, path = setup
    fd = lock_fd(path)
    intervals = []
    barrier = threading.Barrier(2)

    class FakeStep(Step):
        name = "fetch"

        def run(self, ctx):
            barrier.wait(timeout=2)
            start = time.monotonic()
            time.sleep(0.3)
            intervals.append((start, time.monotonic()))
            return StepResult(name=self.name, status="pass", summary="ok")

    results = []

    def worker():
        results.append(run_pipeline(1, steps=[FakeStep(), StressE2EBenchStep()]))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert all(not thread.is_alive() for thread in threads)
        assert results == [0, 0]
        assert max(x[0] for x in intervals) < min(x[1] for x in intervals)
    finally:
        os.close(fd)


def test_matrix_holds_lock(setup, monkeypatch):
    ctx, path = setup
    monkeypatch.setenv("PR_VALIDATE_NO_GPU_CACHE", "1")
    step = StressE2EBenchStep()

    def matrix(_):
        with pytest.raises(BlockingIOError):
            fd = lock_fd(path)
            os.close(fd)
        return StepResult(name=step.name, status="pass", summary="ok")

    monkeypatch.setattr(step, "_run_matrix", matrix)
    assert "shared lock" in step.run(ctx).details


def test_matrix_with_minimal_context(tmp_path, monkeypatch):
    monkeypatch.setenv("PR_VALIDATE_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("PR_VALIDATE_GPU_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PR_VALIDATE_NO_GPU_CACHE", raising=False)
    ctx = types.SimpleNamespace(repo_root=tmp_path, base_sha="0" * 40)
    step = StressE2EBenchStep()
    result = StepResult(name=step.name, status="pass", summary="ok")
    monkeypatch.setattr(step, "_run_matrix", lambda _: result)

    assert step.run(ctx) is result
    assert not (tmp_path / "cache").exists()


def test_cache_pass_and_miss_conditions(setup, monkeypatch):
    ctx, path = setup
    step = StressE2EBenchStep()
    calls = []
    status = ["pass"]

    def matrix(_):
        calls.append(1)
        return StepResult(name=step.name, status=status[0], summary="ok")

    monkeypatch.setattr(step, "_run_matrix", matrix)
    assert step.run(ctx).status == "pass"
    fd = lock_fd(path)
    try:
        assert step.run(ctx).summary.startswith("cached pass")
    finally:
        os.close(fd)
    assert len(calls) == 1
    ctx.head_sha = "c" * 40
    status[0] = "fail"
    assert step.run(ctx).status == "fail"
    assert step.run(ctx).status == "fail"
    assert len(calls) == 3
    monkeypatch.setattr(gpu_cache, "_repo_state", lambda _: (SHA, False))
    assert gpu_cache.cache_key(ctx) is None
    monkeypatch.setattr(gpu_cache, "_repo_state", lambda _: (SHA, True))
    monkeypatch.setenv("PR_VALIDATE_NO_GPU_CACHE", "1")
    assert gpu_cache.cache_key(ctx) is None


@pytest.mark.skipif(
    sys.platform != "darwin"
    or not Path("/usr/bin/lockf").exists()
    or not shutil.which("lsof"),
    reason="requires macOS lockf and lsof",
)
def test_legacy_lockf_subprocess(tmp_path):
    path = tmp_path / "lock"
    code = (
        "from pathlib import Path\n"
        "from scripts.pr_validate.gpu_lock import hold_gpu_lock\n"
        f"with hold_gpu_lock(Path({str(path)!r}), log=print) as info:\n"
        " print(info.mode)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        ["/usr/bin/lockf", str(path), sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "held-by-caller" in result.stdout
