# SPDX-License-Identifier: Apache-2.0
"""Tests for the download manager and its gates.

``rapid-mlx pull`` is replaced by a shell stub that emits the real
``[bytes] done/total`` heartbeat format. That format is a documented
contract in ``vllm_mlx/_mirror.py`` (chosen by ``isatty()`` alone, so a
captured pipe always gets it), which is why progress here is parsed
rather than scraped from tqdm.
"""

from __future__ import annotations

import asyncio

import pytest

from rmlx_web import downloads as downloads_module
from rmlx_web.downloads import (
    DISK_HEADROOM_BYTES,
    DownloadError,
    DownloadManager,
    DownloadState,
    check_disk_budget,
    parse_progress,
)


class TestParseProgress:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("  [bytes] 5750583/649378984", (5750583, 649378984)),
            ("[bytes] 0/100", (0, 100)),
            ("   [bytes]   12/34   ", (12, 34)),
        ],
    )
    def test_heartbeat_lines_are_parsed(self, line, expected):
        assert parse_progress(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "  [1/11] config.json R2 (0 MB)",
            "  Cached at: /Users/x/.cache",
            "  [bytes] abc/def",
            # A percentage bar, not the machine contract — the TTY form.
            "  Downloading 83%",
        ],
    )
    def test_other_output_is_ignored(self, line):
        assert parse_progress(line) is None


class TestDiskBudget:
    def test_unknown_size_is_refused(self):
        # model_sizes.json has no entry for every repo (size_bytes is
        # None for e.g. google/embeddinggemma-300m-6bit). "Unknown" must
        # never be read as "small" — guessing here is how a remotely
        # reachable endpoint fills the host's disk.
        for unknown in (None, 0, -1):
            reason = check_disk_budget(unknown)
            assert reason is not None
            assert "unknown" in reason

    def test_a_model_that_fits_is_allowed(self, monkeypatch):
        monkeypatch.setattr(
            downloads_module, "free_disk_bytes", lambda *a: 500 * 1024**3
        )
        assert check_disk_budget(5 * 1024**3) is None

    def test_a_model_that_does_not_fit_is_refused(self, monkeypatch):
        monkeypatch.setattr(downloads_module, "free_disk_bytes", lambda *a: 3 * 1024**3)
        reason = check_disk_budget(5 * 1024**3)
        assert reason is not None
        assert "not enough free space" in reason

    def test_headroom_is_reserved_beyond_the_model_size(self, monkeypatch):
        # Free space that would fit the model exactly but leave nothing
        # over must still be refused: a disk filled to the last byte
        # takes the whole Mac down, not just this feature.
        size = 5 * 1024**3
        monkeypatch.setattr(
            downloads_module, "free_disk_bytes", lambda *a: size + 1024**3
        )
        reason = check_disk_budget(size)
        assert reason is not None

        monkeypatch.setattr(
            downloads_module,
            "free_disk_bytes",
            lambda *a: int(size * 1.15) + DISK_HEADROOM_BYTES + 1,
        )
        assert check_disk_budget(size) is None


@pytest.fixture
def pull_stub(tmp_path):
    """A stub `rapid-mlx` whose `pull` emits real heartbeat lines."""

    def make(*, lines=None, exit_code=0, hang=False):
        body = ["#!/bin/sh"]
        if lines is None:
            lines = [
                "  Alias: qwen3-0.6b-8bit -> mlx-community/Qwen3-0.6B-8bit",
                "  [bytes] 100/1000",
                "  [1/2] config.json R2 (0 MB)",
                "  [bytes] 600/1000",
                "  [bytes] 1000/1000",
            ]
        for line in lines:
            body.append(f"echo '{line}'")
        if hang:
            body.append("sleep 60")
        body.append(f"exit {exit_code}")

        script = tmp_path / "rapid-mlx"
        script.write_text("\n".join(body) + "\n")
        script.chmod(0o755)
        return str(script)

    return make


async def _drain(manager: DownloadManager) -> None:
    """Wait for the supervising task to finish."""
    if manager._task is not None:
        await asyncio.wait_for(manager._task, timeout=15)


class TestDownloadLifecycle:
    @pytest.mark.asyncio
    async def test_progress_is_tracked_and_the_job_completes(self, pull_stub):
        manager = DownloadManager(pull_stub())
        job = await manager.start("qwen3-0.6b-8bit", total_bytes=1000)
        assert job.state is DownloadState.RUNNING

        await _drain(manager)

        assert job.state is DownloadState.DONE
        assert job.done_bytes == 1000

    @pytest.mark.asyncio
    async def test_completion_snaps_the_bar_to_the_total(self, pull_stub):
        # The last heartbeat can land short of the total, leaving a bar
        # stuck at 99% on a download that actually finished.
        manager = DownloadManager(pull_stub(lines=["  [bytes] 900/1000"]))
        job = await manager.start("m", total_bytes=1000)
        await _drain(manager)

        assert job.state is DownloadState.DONE
        assert job.done_bytes == 1000

    @pytest.mark.asyncio
    async def test_progress_never_goes_backwards(self, pull_stub):
        # Transfer workers heartbeat concurrently, so a stale line can
        # arrive after a fresher one.
        manager = DownloadManager(
            pull_stub(lines=["  [bytes] 800/1000", "  [bytes] 300/1000"])
        )
        job = await manager.start("m", total_bytes=1000)
        await _drain(manager)

        assert job.done_bytes >= 800

    @pytest.mark.asyncio
    async def test_the_live_total_overrides_the_manifest_estimate(self, pull_stub):
        # The manifest is a snapshot and can disagree with the actual
        # transfer; the live number keeps the bar honest.
        manager = DownloadManager(pull_stub(lines=["  [bytes] 10/4242"]))
        job = await manager.start("m", total_bytes=1000)
        await _drain(manager)

        assert job.total_bytes == 4242

    @pytest.mark.asyncio
    async def test_a_failing_pull_is_recorded_with_its_output(self, pull_stub):
        manager = DownloadManager(
            pull_stub(lines=["  Error: model not found"], exit_code=1)
        )
        job = await manager.start("m", total_bytes=1000)
        await _drain(manager)

        # The exit code is the authoritative signal, not any output line:
        # a partial transfer that failed still prints status lines.
        assert job.state is DownloadState.FAILED
        assert "not found" in job.detail

    @pytest.mark.asyncio
    async def test_a_missing_binary_raises(self):
        manager = DownloadManager("/nonexistent/rapid-mlx")
        with pytest.raises(DownloadError):
            await manager.start("m", total_bytes=1000)

    @pytest.mark.asyncio
    async def test_only_one_download_runs_at_a_time(self, pull_stub):
        manager = DownloadManager(pull_stub(hang=True))
        await manager.start("first", total_bytes=1000)

        # Concurrent multi-GB pulls contend for the same bandwidth and
        # disk, finishing no sooner while doubling the peak footprint —
        # which is what the budget check is defending.
        with pytest.raises(DownloadError) as excinfo:
            await manager.start("second", total_bytes=1000)
        assert "already running" in str(excinfo.value)

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_a_finished_download_frees_the_slot(self, pull_stub):
        manager = DownloadManager(pull_stub())
        await manager.start("first", total_bytes=1000)
        await _drain(manager)

        job = await manager.start("second", total_bytes=1000)
        assert job.alias == "second"
        await _drain(manager)


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_stops_a_running_download(self, pull_stub):
        manager = DownloadManager(pull_stub(hang=True))
        job = await manager.start("m", total_bytes=1000)

        assert await manager.cancel() is True
        await _drain(manager)

        # A cancelled pull exits non-zero; that must not be relabelled
        # as a failure the user did not cause.
        assert job.state is DownloadState.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_kills_the_whole_process_group(self, pull_stub):
        # `sleep` is a CHILD of the stub shell. Signalling only the
        # leader leaves it running and holding the stdout pipe, so the
        # supervisor never sees EOF.
        manager = DownloadManager(pull_stub(hang=True))
        await manager.start("m", total_bytes=1000)

        await manager.cancel()
        # Bounded generously: this asserts the drain terminates at all,
        # not how fast. Without the process-group kill it never does.
        await asyncio.wait_for(_drain(manager), timeout=15)

    @pytest.mark.asyncio
    async def test_cancel_with_nothing_running_returns_false(self):
        manager = DownloadManager("/bin/true")
        assert await manager.cancel() is False

    @pytest.mark.asyncio
    async def test_shutdown_stops_an_in_flight_download(self, pull_stub):
        manager = DownloadManager(pull_stub(hang=True))
        job = await manager.start("m", total_bytes=1000)

        await asyncio.wait_for(manager.shutdown(), timeout=15)

        # A pull left running past the supervisor keeps writing to the
        # cache with nothing watching it, and the user has no way to stop
        # it short of hunting down the PID.
        assert job.state is DownloadState.CANCELLED
