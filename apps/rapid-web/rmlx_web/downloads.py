# SPDX-License-Identifier: Apache-2.0
"""Model downloads, driven through ``rapid-mlx pull``.

Progress is not scraped from tqdm. ``rapid-mlx pull`` emits a
machine-readable heartbeat on stdout whenever stdout is not a TTY::

      [bytes] 5750583/649378984

``vllm_mlx/_mirror.py`` picks the mode from ``isatty()`` alone, so a
captured pipe always gets the machine form. Interleaved human status
lines are ignored, and the authoritative completion signal is the exit
code — a failed partial transfer prints status lines too.

Downloads are gated because this endpoint is remotely reachable once a
tunnel is attached and consumes an unbounded amount of somebody else's
disk. Three gates, all enforced by the caller in ``app.py``: off unless
enabled, refused unless the size is known and fits, and restricted to
catalog aliases.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass, field
from enum import Enum

# Must remain free after the download. A disk filled to the last byte
# takes the whole Mac down: the OS needs swap and the engine writes a
# Metal shader cache on first load.
DISK_HEADROOM_BYTES = 10 * 1024**3

# HuggingFace stages a blob then moves it, so peak exceeds final size.
# Rough — DISK_HEADROOM_BYTES dominates.
_TRANSFER_OVERHEAD = 1.15

# ``  [bytes] 5750583/649378984``
_BYTES_RE = re.compile(r"^\s*\[bytes\]\s+(\d+)\s*/\s*(\d+)\s*$")

# SIGTERM→SIGKILL grace on cancel. huggingface_hub unwinds a partial blob
# on the way out; killing instantly strands an ``.incomplete`` file that
# nothing collects until the next pull of the same repo.
_TERM_GRACE_S = 10.0

_OUTPUT_TAIL_LINES = 40


class DownloadState(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadError(RuntimeError):
    """A download could not be started."""


@dataclass
class DownloadJob:
    """One in-flight or finished pull."""

    alias: str
    total_bytes: int | None
    state: DownloadState = DownloadState.RUNNING
    done_bytes: int = 0
    detail: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        # The denominator comes from the pull's own heartbeat once it
        # starts reporting, and from the size manifest before that. The
        # two can disagree slightly (the manifest is a snapshot), so the
        # live value wins to keep the bar monotonic.
        return {
            "alias": self.alias,
            "state": self.state.value,
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "detail": self.detail,
        }


def free_disk_bytes(path: str | None = None) -> int:
    """Bytes available on the filesystem holding the HF cache.

    Measured where the download lands, not on ``/``: an ``HF_HOME`` on an
    external volume is common, and the wrong filesystem gives an answer
    that is confidently wrong in either direction.
    """
    target = path or _hf_cache_root()
    # Nearest existing ancestor — the cache dir may not exist yet.
    while target and not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    return shutil.disk_usage(target or "/").free


def _hf_cache_root() -> str:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(var)
        if value:
            return value
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def check_disk_budget(size_bytes: int | None) -> str | None:
    """Reject a download that does not fit. Returns a reason, or None.

    **Fails closed on an unknown size.** ``model_sizes.json`` has no entry
    for every repo (``None`` for e.g. ``google/embeddinggemma-300m-6bit``),
    and guessing is how a publicly reachable endpoint fills the host's disk.
    """
    if not size_bytes or size_bytes <= 0:
        return (
            "the download size for this model is unknown, so it cannot be "
            "checked against free space. Pull it from the Mac instead."
        )

    required = int(size_bytes * _TRANSFER_OVERHEAD) + DISK_HEADROOM_BYTES
    free = free_disk_bytes()
    if free < required:
        return (
            f"not enough free space: this needs about {_gib(required)} "
            f"(including {_gib(DISK_HEADROOM_BYTES)} headroom) "
            f"but only {_gib(free)} is available."
        )
    return None


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def parse_progress(line: str) -> tuple[int, int] | None:
    """Extract ``(done, total)`` from a heartbeat line, if it is one."""
    match = _BYTES_RE.match(line)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


class DownloadManager:
    """Runs at most one ``rapid-mlx pull`` at a time.

    A policy choice: concurrent multi-GB pulls contend for the same
    bandwidth and disk, finishing no sooner than in sequence while
    doubling the peak footprint that the budget check defends.
    """

    def __init__(self, binary: str) -> None:
        self._binary = binary
        self._job: DownloadJob | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._output_tail: list[str] = []
        self._lock = asyncio.Lock()

    @property
    def job(self) -> DownloadJob | None:
        return self._job

    def is_running(self) -> bool:
        return self._job is not None and self._job.state is DownloadState.RUNNING

    async def start(self, alias: str, *, total_bytes: int | None) -> DownloadJob:
        async with self._lock:
            if self.is_running():
                raise DownloadError(
                    f"a download is already running ({self._job.alias}); "
                    "wait for it to finish or cancel it"
                )

            self._job = DownloadJob(alias=alias, total_bytes=total_bytes)
            self._output_tail = []

            try:
                process = await asyncio.create_subprocess_exec(
                    self._binary,
                    "pull",
                    alias,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    # Own process group so cancel reaches the whole tree:
                    # the pull spawns transfer workers, and signalling only
                    # the leader leaves them downloading with no parent.
                    start_new_session=True,
                )
            except OSError as exc:
                self._job.state = DownloadState.FAILED
                self._job.detail = str(exc)
                raise DownloadError(f"could not run {self._binary}: {exc}") from exc

            self._process = process
            self._task = asyncio.create_task(self._supervise(process))
            return self._job

    async def _supervise(self, process: asyncio.subprocess.Process) -> None:
        """Read progress until the child exits, then record the outcome."""
        assert process.stdout is not None
        job = self._job

        while True:
            try:
                raw = await process.stdout.readline()
            except (ValueError, OSError):
                # ValueError on an over-long line without a newline. Stop
                # reading rather than kill the drain — an unread pipe would
                # block the child on its next write.
                break
            if not raw:
                break

            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            progress = parse_progress(line)
            if progress is not None and job is not None:
                done, total = progress
                # Never let the bar go backwards: workers heartbeat
                # concurrently, so a stale line can arrive after a fresher.
                job.done_bytes = max(job.done_bytes, done)
                if total > 0:
                    job.total_bytes = total
                continue

            self._output_tail.append(line)
            if len(self._output_tail) > _OUTPUT_TAIL_LINES:
                del self._output_tail[:-_OUTPUT_TAIL_LINES]

        code = await process.wait()

        if job is not None:
            if job.state is DownloadState.CANCELLED:
                # Recorded by cancel(); a cancelled pull exits non-zero,
                # which must not be relabelled as a failure.
                pass
            elif code == 0:
                job.state = DownloadState.DONE
                # Snap to 100%: the last heartbeat can land short of the
                # total, leaving a bar stuck at 99%.
                if job.total_bytes:
                    job.done_bytes = job.total_bytes
            else:
                job.state = DownloadState.FAILED
                job.detail = self._tail_text() or f"pull exited with code {code}"

        self._process = None

    def _tail_text(self, lines: int = 6) -> str:
        return " | ".join(self._output_tail[-lines:])

    async def cancel(self) -> bool:
        """Stop the running pull. Returns False if none was running."""
        async with self._lock:
            process = self._process
            job = self._job
            if process is None or job is None or job.state is not DownloadState.RUNNING:
                return False

            # Mark before signalling so _supervise does not race and
            # relabel the non-zero exit as a failure.
            job.state = DownloadState.CANCELLED
            job.detail = "cancelled"

            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()

            try:
                await asyncio.wait_for(process.wait(), timeout=_TERM_GRACE_S)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
            return True

    async def shutdown(self) -> None:
        """Stop any running pull at process exit.

        A download left running would keep writing to the cache with
        nothing watching it, and no way to stop it short of the PID.
        """
        await self.cancel()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
