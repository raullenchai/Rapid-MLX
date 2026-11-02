# SPDX-License-Identifier: Apache-2.0
"""Lock-file identity must survive an unlinking holder's handoff."""

import fcntl
import os

import pytest

from scripts.pr_validate import gpu_lock


@pytest.mark.parametrize("recreate", [False, True], ids=["missing", "replaced"])
def test_waiter_reopens_unlinked_lock(tmp_path, monkeypatch, recreate):
    path = tmp_path / "gpu.lock"
    holder = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    stale_probe = os.open(path, os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    third = None
    polls = 0

    def handoff(_):
        nonlocal holder, third, polls
        polls += 1
        if polls == 1:
            # The waiter has already opened the original inode and hit contention.
            path.unlink()
            if recreate:
                third = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
                fcntl.flock(third, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.close(holder)
            holder = None
        elif polls == 2 and recreate:
            # Reaching the new holder must first release the stale inode.
            fcntl.flock(stale_probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stale_probe, fcntl.LOCK_UN)
            os.close(third)
            third = None
        else:
            pytest.fail("unexpected lock retry")

    monkeypatch.setattr(gpu_lock.time, "sleep", handoff)
    try:
        with gpu_lock.hold_gpu_lock(
            path,
            log=lambda _: None,
            holder_pids=lambda _: [],
            ancestor_pids=lambda: set(),
            poll_seconds=0.01,
        ) as info:
            assert polls == (2 if recreate else 1)
            assert path.exists(), "waiter acquired the deleted inode"
            assert info.mode == "acquired"
            probe = os.open(path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
        probe = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)
    finally:
        for fd in (holder, stale_probe, third):
            if fd is not None:
                os.close(fd)
