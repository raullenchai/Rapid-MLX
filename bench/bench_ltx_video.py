#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproducible LTX latency and unified-memory benchmark harness.

The controller launches a fresh process for every measured run, samples the
whole process tree, and writes both machine-readable JSON and a short Markdown
summary.  The worker mode is intentionally private: it keeps model/runtime
imports out of the controller so cold-start measurements include them.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import re
import secrets
import select
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_PREFIX = "LTX_BENCH_EVENT "
SWAP_RE = re.compile(r"used = ([0-9.]+)([MG])")
DEFAULT_PROMPT = (
    "A red fox trots across fresh snow at golden hour, cinematic tracking shot"
)


@dataclass
class Sample:
    elapsed_s: float
    rss_bytes: int
    swap_used_bytes: int | None


def _event_key() -> bytes:
    """Worker-side key, inherited through the child environment."""
    key = os.environ.get("LTX_BENCH_EVENT_KEY", "")
    return key.encode() if key else b""


def _sign(payload: str, key: bytes) -> str | None:
    if not key:
        # No key means an unauthenticated channel: fail closed.
        return None
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def _emit(kind: str, **fields: Any) -> None:
    body = {"kind": kind, "time_ns": time.monotonic_ns(), **fields}
    payload = json.dumps(body, sort_keys=True)
    print(
        EVENT_PREFIX
        + json.dumps(
            {"sig": _sign(payload, _event_key()), "payload": body}, sort_keys=True
        ),
        flush=True,
    )


def _artifact_ready(path: Path) -> bool:
    """TOCTOU-safe non-empty-file probe: a vanished artifact is "not yet"."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _artifact_size(path: Path) -> int:
    """TOCTOU-safe artifact size: a vanished artifact contributes zero."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _swap_used_bytes(timeout: float = 3.0) -> int | None:
    try:
        output = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            text=True,
            timeout=max(timeout, 0.05),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = SWAP_RE.search(output)
    if not match:
        return None
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    return round(float(match.group(1)) * scale)


def _tree_rss(pid: int) -> int:
    import psutil

    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.Error, OSError):
            pass
    return total


def _probe(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, timeout=10).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _machine_observation() -> dict[str, Any]:
    memory = _probe(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    return {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "hardware_model": _probe(["/usr/sbin/sysctl", "-n", "hw.model"]),
        "memory_bytes": int(memory) if memory and memory.isdigit() else None,
        "macos_version": _probe(["/usr/bin/sw_vers", "-productVersion"]),
        "macos_build": _probe(["/usr/bin/sw_vers", "-buildVersion"]),
    }


def _thermal_observation() -> str | None:
    return _probe(["/usr/bin/pmset", "-g", "therm"])


def _worker_mlx(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from rapid_mlx.runtime.video_lane import VideoEngine

    mx.reset_peak_memory()
    output = Path(args.output_video)
    _emit("worker_ready")
    started = time.monotonic_ns()
    try:
        VideoEngine(args.model).generate(
            prompt=args.prompt,
            output_path=output,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            fps=args.fps,
            seed=args.seed,
            image=None,
        )
    except SystemExit as exc:
        if exc.code not in (0, None):
            _emit("error", error_type="SystemExit", message=str(exc.code))
            return 1
    except BaseException as exc:
        _emit("error", error_type=type(exc).__name__, message=str(exc))
        return 1
    if not output.is_file() or output.stat().st_size == 0:
        _emit(
            "error",
            error_type="MissingArtifact",
            message=f"missing or empty artifact: {output}",
        )
        return 1
    _emit(
        "complete",
        generation_s=(time.monotonic_ns() - started) / 1e9,
        mlx_peak_bytes=mx.get_peak_memory(),
        output_bytes=output.stat().st_size,
    )
    return 0


class _MeasuredBar:
    """Iterator proxy that keeps the full tqdm object API alive.

    Replacing ``tqdm`` with a bare generator breaks callers that use
    ``update``, ``close``, or the context-manager protocol, so every
    attribute except the iteration protocol is delegated to the real bar.
    Manual ``tqdm(total=...)`` bars and batched updates are rejected
    outright: they cannot yield honest per-step timing samples.
    """

    def __init__(self, bar, stage_number: int, description: str, emit) -> None:
        self._bar = bar
        self._it = iter(bar)
        self._stage = stage_number
        self._description = description
        self._emit = emit
        self._step = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self._it)
        except StopIteration:
            if self._step:
                # Iterator exhaustion is the end of the denoise loop: a
                # boundary event here gives the final step a duration that
                # excludes downstream encoding.
                self._emit(
                    "stage_end",
                    stage=self._stage,
                    description=self._description,
                    step=self._step,
                    total=self._bar.total,
                )
            raise
        self._step += 1
        self._emit(
            "step_start",
            stage=self._stage,
            description=self._description,
            step=self._step,
            total=self._bar.total,
        )
        return item

    def update(self, n=1):
        # Pure delegation: step events come only from iteration boundaries,
        # so an explicit update can never double-count a step.
        return self._bar.update(n)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._bar, name)

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._bar.__exit__(*exc_info)


DIFFUSION_BAR_PREFIX = "Denoising"
"""Identity of diffusion-sampler progress bars in ltx-2-mlx.

Every sampler call site in the pinned fork creates its tqdm bar with a
desc beginning with "Denoising" (samplers.py:128/330/698/924 at commit
905efb23); any other tqdm bar belongs to downloads, loading, or encoding
and must not contribute step samples.
"""


def _wrap_progress_bar(wrapped, stage_number: int, description: str, emit):
    """Return a measured wrapper for iterable bars, or the bar untouched.

    Manual tqdm(total=...) bars reveal steps only after they finish: they
    are passed through unmeasured instead of fabricating per-step samples.
    """
    if not description.startswith(DIFFUSION_BAR_PREFIX):
        # Download/loading/encode bars share this tqdm module: measuring
        # them would pollute step medians with unrelated work.
        return wrapped
    if getattr(wrapped, "iterable", None) is None:
        return wrapped
    return _MeasuredBar(wrapped, stage_number, description, emit)


def _worker_ltx25(args: argparse.Namespace) -> int:
    """Run the pinned CLI while adding machine-readable per-step events."""
    import tqdm as tqdm_module

    original_tqdm = tqdm_module.tqdm
    stage_counter = 0

    def measured_tqdm(iterable=None, *pargs, **kwargs):
        nonlocal stage_counter
        wrapped = original_tqdm(iterable, *pargs, **kwargs)
        description = str(kwargs.get("desc", "step"))
        bar = _wrap_progress_bar(wrapped, stage_counter, description, _emit)
        if bar is wrapped:
            return wrapped
        stage_counter += 1
        return bar

    class _TqdmProxy:
        def __call__(self, *pargs, **kwargs):
            return measured_tqdm(*pargs, **kwargs)

        def __getattr__(self, name):
            return getattr(original_tqdm, name)

    tqdm_module.tqdm = _TqdmProxy()
    # The sampler binds tqdm at import time, so patch both module globals.
    from ltx_pipelines_mlx import cli
    from ltx_pipelines_mlx.utils import samplers

    samplers.tqdm = measured_tqdm
    mx = __import__("mlx.core", fromlist=["core"])
    mx.reset_peak_memory()
    output = Path(args.output_video)
    argv = [
        "ltx-2-mlx",
        "generate",
        "--model",
        args.model,
        "--distilled",
        "--low-ram",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--frames",
        str(args.frames),
        "--frame-rate",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--output",
        str(output),
        "--prompt",
        args.prompt,
    ]
    _emit("worker_ready")
    started = time.monotonic_ns()
    old_argv = sys.argv
    try:
        sys.argv = argv
        cli.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            _emit("error", error_type="SystemExit", message=str(exc.code))
            return 1
    except BaseException as exc:
        _emit("error", error_type=type(exc).__name__, message=str(exc))
        return 1
    finally:
        sys.argv = old_argv
    if not output.is_file() or output.stat().st_size == 0:
        _emit(
            "error",
            error_type="MissingArtifact",
            message=f"missing or empty artifact: {output}",
        )
        return 1
    _emit(
        "complete",
        generation_s=(time.monotonic_ns() - started) / 1e9,
        mlx_peak_bytes=mx.get_peak_memory(),
        output_bytes=output.stat().st_size,
    )
    return 0


def _worker(args: argparse.Namespace) -> int:
    return _worker_ltx25(args) if args.runtime == "ltx25" else _worker_mlx(args)


def _read_stream(
    stream,
    name: str,
    started_ns: int,
    events: list[dict[str, Any]],
    lines: list[str],
    stop_event: threading.Event,
    event_key: bytes,
    retained_bytes: list[int],
) -> None:
    """Drain one pipe on a bounded-wait select loop.

    Blocking readline() cannot be interrupted by closing the descriptor,
    so a wedged reader would hang the controller forever. select() with a
    timeout keeps every wait bounded and lets the controller stop the
    thread cooperatively.
    """
    fd = stream.fileno()
    pending = b""
    while not stop_event.is_set():
        try:
            readable, _, _ = select.select([fd], [], [], 0.5)
        except (OSError, ValueError):
            return
        if not readable:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return
        if not chunk:
            break
        pending += chunk
        if len(pending) > 16 * 1024 * 1024:
            # A runtime emitting endless unterminated output must not be
            # able to exhaust controller memory: flush it as one record.
            _handle_line(
                pending.decode("utf-8", "replace"),
                name,
                started_ns,
                events,
                lines,
                event_key,
                retained_bytes,
            )
            pending = b""
        *complete_lines, pending = pending.replace(b"\r", b"\n").split(b"\n")
        for raw in complete_lines:
            _handle_line(
                raw.decode("utf-8", "replace"),
                name,
                started_ns,
                events,
                lines,
                event_key,
                retained_bytes,
            )
    if pending.strip():
        # EOF without a trailing newline: still log and parse the tail.
        _handle_line(
            pending.decode("utf-8", "replace"),
            name,
            started_ns,
            events,
            lines,
            event_key,
            retained_bytes,
        )
    return


MAX_RETAINED_LINES = 20000
MAX_RETAINED_BYTES = 16 * 1024 * 1024
"""Bounds on retained output: an unbounded hour-long worker must not exhaust
controller memory; beyond either cap, lines are counted, not stored."""


def _handle_line(
    raw: str,
    name: str,
    started_ns: int,
    events: list[dict[str, Any]],
    lines: list[str],
    event_key: bytes,
    retained_bytes: list[int],
) -> None:
    try:
        elapsed = (time.monotonic_ns() - started_ns) / 1e9
        line = raw.rstrip("\r\n")
        record = f"{elapsed:.6f} {name} {line[:65536]}"
        if (
            len(lines) < MAX_RETAINED_LINES
            and retained_bytes[0] + len(record) <= MAX_RETAINED_BYTES
        ):
            lines.append(record)
            retained_bytes[0] += len(record)
        if line.startswith(EVENT_PREFIX):
            try:
                envelope = json.loads(line[len(EVENT_PREFIX) :])
                payload = json.dumps(envelope["payload"], sort_keys=True)
                expected = _sign(payload, event_key)
                if expected is None or not hmac.compare_digest(
                    expected, envelope.get("sig", "")
                ):
                    # An unauthenticated line (e.g. an echoed prompt) is
                    # logged but never trusted as an event.
                    return
                event = envelope["payload"]
                event["observed_elapsed_s"] = elapsed
                if "time_ns" in event:
                    # Worker-side monotonic timestamp: single clock, no
                    # reader-scheduling jitter.
                    event["worker_time_ns"] = event["time_ns"]
                events.append(event)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            return

    except (ValueError, OSError):
        return


def _stage_duration_samples(
    step_events: list[dict[str, Any]],
    stage_ends: dict[int, dict[str, Any]],
    same_stage_only: bool = True,
) -> dict[str, list[float]]:
    """Sample per-step durations for the stage medians.

    With ``same_stage_only`` (the only mode medians use) a step is timed to
    the next step start within the same stage, so stage transitions and
    output encoding never pollute the diffusion-step cadence. A stage's
    final step is timed to the ``stage_end`` boundary emitted when the
    sampler iterator is exhausted, which excludes downstream encoding; if
    no boundary event exists (older workers, interrupted runs), the final
    step is omitted rather than guessed.
    """
    stage_durations: dict[str, list[float]] = {}
    for current, following in zip(step_events, step_events[1:]):
        if same_stage_only and current.get("stage") != following.get("stage"):
            continue
        stage_durations.setdefault(str(current["stage"]), []).append(
            following["observed_elapsed_s"] - current["observed_elapsed_s"]
        )
    if step_events:
        last_of_stage: dict[int, dict[str, Any]] = {}
        for event in step_events:
            last_of_stage[event["stage"]] = event
        for stage, end in stage_ends.items():
            final = last_of_stage.get(stage)
            if final is not None:
                stage_durations.setdefault(str(stage), []).append(
                    end["observed_elapsed_s"] - final["observed_elapsed_s"]
                )
    return stage_durations


def annotate_step_boundaries(step_events: list[dict[str, Any]]) -> None:
    """Annotate adjacent step events in place.

    Same-stage neighbors record ``duration_to_next_step_s``; cross-stage
    neighbors record ``stage_transition_s`` instead, because that gap
    includes stage-switch overhead and must never be counted as a step.
    """
    for current, following in zip(step_events, step_events[1:]):
        gap = following["observed_elapsed_s"] - current["observed_elapsed_s"]
        if current.get("stage") == following.get("stage"):
            current["duration_to_next_step_s"] = gap
        else:
            current["stage_transition_s"] = gap


def _run_once(args: argparse.Namespace, run_index: int) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.runtime}-{args.frames}f-{args.invocation}-run{run_index}"
    output_video = output_dir / f"{stem}.mp4"
    # A reused filename must never let a stale artifact answer this run's
    # artifact_first_byte_s or survive a failed run.
    if output_video.exists():
        output_video.unlink()
    interpreter = args.ltx25_python if args.runtime == "ltx25" else sys.executable
    command = [
        interpreter,
        str(Path(__file__).resolve()),
        "--worker",
        "--runtime",
        args.runtime,
        "--model",
        args.resolved_model,
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--size",
        f"{args.width}x{args.height}",
        "--prompt",
        args.prompt,
        "--output-video",
        str(output_video),
        "--invocation",
        args.invocation,
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["LTX_BENCH_EVENT_KEY"] = args.event_key
    try:
        stat_now = _stat_fingerprint(Path(args.resolved_model))
    except OSError as exc:
        integrity_error = f"model fingerprint failed before the run: {exc}"
    else:
        integrity_error = None
        if stat_now != args.model_stat_fingerprint:
            integrity_error = (
                "model files changed since resolution (inode/size/mtime "
                "fingerprint mismatch); discarding the run"
            )
    if args.runtime_stat_fingerprint is not None:
        try:
            runtime_now = _source_stat_fingerprint(
                Path(args.runtime_module_file).parent
            )
        except OSError as exc:
            integrity_error = integrity_error or (
                f"runtime fingerprint failed before the run: {exc}"
            )
        else:
            if runtime_now != args.runtime_stat_fingerprint:
                integrity_error = integrity_error or (
                    "runtime files changed since resolution; discarding the run"
                )
    if integrity_error is not None:
        return {
            "run": run_index,
            "status": "failed",
            "deadline_exceeded": False,
            "returncode": None,
            "total_s": 0.0,
            "integrity_error": integrity_error,
            "cold_start_to_first_step_s": None,
            "artifact_first_byte_s": None,
            "step_median_s": None,
            "stage_step_median_s": {},
            "steps": [],
            "cleanup_s": 0.0,
            "peak_rss_basis": None,
            "peak_tree_rss_bytes": 0,
            "mlx_peak_bytes": None,
            "swap_start_bytes": None,
            "swap_end_bytes": None,
            "swap_growth_bytes": None,
            "swap_limited": False,
            "thermal_start": None,
            "thermal_end": None,
            "output": None,
            "output_bytes": 0,
            "samples": [],
            "events": [],
            "launch_error": None,
        }
    swap_start = _swap_used_bytes()
    thermal_start = _thermal_observation()
    started_ns = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
    except OSError as exc:
        result = {
            "run": run_index,
            "status": "failed",
            "deadline_exceeded": False,
            "returncode": None,
            "total_s": (time.monotonic_ns() - started_ns) / 1e9,
            "cold_start_to_first_step_s": None,
            "artifact_first_byte_s": None,
            "step_median_s": None,
            "stage_step_median_s": {},
            "steps": [],
            "cleanup_s": 0.0,
            "integrity_error": None,
            "peak_rss_basis": None,
            "peak_tree_rss_bytes": 0,
            "mlx_peak_bytes": None,
            "swap_start_bytes": swap_start,
            "swap_end_bytes": swap_start,
            "swap_growth_bytes": 0,
            "swap_limited": False,
            "thermal_start": thermal_start,
            "thermal_end": None,
            "output": None,
            "output_bytes": 0,
            "samples": [],
            "events": [],
            "launch_error": f"{type(exc).__name__}: {exc}",
        }
        (output_dir / f"{stem}.log").write_text("")
        (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    events: list[dict[str, Any]] = []
    lines: list[str] = []
    retained_bytes = [0]
    stop_event = threading.Event()
    readers_alive = False
    integrity_error = None
    event_key = (args.event_key or "").encode()
    threads = [
        threading.Thread(
            target=_read_stream,
            args=(
                stream,
                name,
                started_ns,
                events,
                lines,
                stop_event,
                event_key,
                retained_bytes,
            ),
            daemon=True,
        )
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
    ]
    for thread in threads:
        thread.start()
    samples: list[Sample] = []
    first_artifact_s = None
    deadline_exceeded = False
    next_sample_s = None

    terminated = False

    def _terminate_worker() -> bool:
        # A controller exception (Ctrl-C, probe failure) must never orphan the
        # worker's process group: descendants that inherited the group can
        # outlive the direct child while holding the GPU. Attempt group
        # termination even after the child itself has exited. POSIX only
        # reuses a PGID after the whole group dies: a surviving group under
        # this PGID is provably ours, and a vanished one makes killpg raise.
        nonlocal terminated
        if terminated:
            return True
        if process.poll() is not None:
            # The leader was reaped, but POSIX only reuses a PGID after the
            # whole group dies. So a group that still exists under this PGID
            # is provably ours (surviving descendants), and signaling it is
            # safe; a vanished group makes killpg raise and we are done.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                terminated = True
                return True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # The worker exited between the last poll() and termination:
            # still reap it so no zombie survives and returncode is set.
            terminated = True
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return True
        terminated = True
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # Descendants that inherited the group can outlive the reaped direct
        # child while holding the GPU: escalate only once the whole group is
        # gone (killpg(pid, 0) probes group existence).
        deadline_ns = time.monotonic_ns() + int(10 * 1e9)
        while time.monotonic_ns() < deadline_ns:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            events.append(
                {
                    "kind": "cleanup_failed",
                    "observed_elapsed_s": (time.monotonic_ns() - started_ns) / 1e9,
                }
            )
            return False
        # SIGKILL reaped the leader, but descendants can keep the group —
        # and its GPU allocations — alive: probe until a bounded deadline.
        deadline_ns = time.monotonic_ns() + int(10 * 1e9)
        while time.monotonic_ns() < deadline_ns:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.2)
        events.append(
            {
                "kind": "cleanup_failed",
                "observed_elapsed_s": (time.monotonic_ns() - started_ns) / 1e9,
            }
        )
        return False
        return True

    try:
        while process.poll() is None:
            now_s = (time.monotonic_ns() - started_ns) / 1e9
            if now_s > args.deadline:
                deadline_exceeded = True
                events.append(
                    {
                        "kind": "deadline_exceeded",
                        "observed_elapsed_s": now_s,
                    }
                )
                break
            samples.append(
                Sample(
                    now_s,
                    _tree_rss(process.pid),
                    _swap_used_bytes(timeout=args.deadline - now_s),
                )
            )
            if first_artifact_s is None and _artifact_ready(output_video):
                # The artifact may have appeared during the probes: stamp it
                # with a fresh clock reading, not the pre-probe sample time.
                first_artifact_s = (time.monotonic_ns() - started_ns) / 1e9
            # Re-read the clock after the probes, then sleep to the next
            # absolute sample boundary so the advertised sampling interval
            # excludes probe duration.
            now_s = (time.monotonic_ns() - started_ns) / 1e9
            if now_s > args.deadline:
                deadline_exceeded = True
                events.append(
                    {
                        "kind": "deadline_exceeded",
                        "observed_elapsed_s": now_s,
                    }
                )
                break
            next_sample_s = (next_sample_s or now_s) + args.sample_interval
            if next_sample_s < now_s:
                next_sample_s = now_s + args.sample_interval
            wake_s = min(next_sample_s, args.deadline)
            time.sleep(max(wake_s - now_s, 0))
        if not _terminate_worker():
            integrity_error = integrity_error or (
                "worker process group survived cleanup; GPU-holding "
                "descendants may remain alive"
            )
        # The worker is gone; a file flushed just before exit was never seen
        # by the polling loop, so record an explicit upper bound.
        if (
            first_artifact_s is None
            and output_video.is_file()
            and output_video.stat().st_size
        ):
            first_artifact_s = (time.monotonic_ns() - started_ns) / 1e9
        # Readers drain to EOF on their own after process exit. Only a
        # reader wedged behind a descendant's write end needs the bounded
        # pipe-release fallback; closing before a normal join could race
        # unread terminal events such as "complete".
        for thread in threads:
            thread.join(timeout=30)
        if any(thread.is_alive() for thread in threads):
            # Readers poll stop_event every 0.5 s, so they exit without any
            # blocking-read or buffered-lock deadlock.
            stop_event.set()
            for stream in (process.stdout, process.stderr):
                try:
                    os.close(stream.fileno())
                except OSError:
                    pass
            for thread in threads:
                thread.join(timeout=5)
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            for thread in threads:
                thread.join(timeout=5)
            readers_alive = any(thread.is_alive() for thread in threads)
            if readers_alive:
                integrity_error = (
                    "reader threads survived bounded cleanup; discarding "
                    "the run to avoid reading concurrently mutated state"
                )
    finally:
        if not _terminate_worker():
            integrity_error = integrity_error or (
                "worker process group survived cleanup; GPU-holding "
                "descendants may remain alive"
            )
    elapsed_s = (time.monotonic_ns() - started_ns) / 1e9
    try:
        stat_after = _stat_fingerprint(Path(args.resolved_model))
    except OSError as exc:
        integrity_error = integrity_error or (
            f"model fingerprint failed after the run: {exc}"
        )
    else:
        if stat_after != args.model_stat_fingerprint:
            integrity_error = integrity_error or (
                "model files changed during the run (inode/size/mtime "
                "fingerprint mismatch); discarding the run"
            )
    if args.runtime_stat_fingerprint is not None:
        try:
            runtime_after = _source_stat_fingerprint(
                Path(args.runtime_module_file).parent
            )
        except OSError as exc:
            integrity_error = integrity_error or (
                f"runtime fingerprint failed after the run: {exc}"
            )
        else:
            if runtime_after != args.runtime_stat_fingerprint:
                integrity_error = integrity_error or (
                    "runtime files changed during the run; discarding the run"
                )
    swap_end = _swap_used_bytes()
    thermal_end = _thermal_observation()
    step_events = sorted(
        (event for event in events if event.get("kind") == "step_start"),
        key=lambda event: event["observed_elapsed_s"],
    )
    complete = next(
        (event for event in events if event.get("kind") == "complete"), None
    )
    # Worker-emitted events carry a single-clock timestamp; project them onto
    # the controller timeline through the worker_ready anchor so step and
    # completion intervals are immune to reader-scheduling jitter.
    stage_end_events = [event for event in events if event.get("kind") == "stage_end"]
    ready = next(
        (event for event in events if event.get("kind") == "worker_ready"), None
    )
    if ready is not None and ready.get("worker_time_ns") is not None:
        anchored = [
            ready,
            *step_events,
            *stage_end_events,
            *([complete] if complete else []),
        ]
        if all(event.get("worker_time_ns") is not None for event in anchored):
            # time.monotonic_ns() is system-wide: worker timestamps compare
            # directly against the controller's started_ns, with no
            # reader-thread jitter in between.
            for event in anchored:
                event["observed_elapsed_s"] = (
                    event["worker_time_ns"] - started_ns
                ) / 1e9
    stage_ends = {event["stage"]: event for event in stage_end_events}
    annotate_step_boundaries(step_events)
    # Medians use adjacent step starts within the same stage only: that is
    # the clean diffusion-step cadence, unpolluted by stage transitions or
    # output encoding. A stage's final step is timed to the sampler's
    # stage_end boundary (pre-encoding); its full tail including encoding
    # is still recorded as duration_to_complete_s on the last event.
    stage_durations = _stage_duration_samples(step_events, stage_ends)
    durations = [duration for values in stage_durations.values() for duration in values]
    if step_events and complete:
        # step_start events for mlx23 arrive on the stderr reader thread
        # while "complete" arrives on stdout; each stream is timestamped by
        # its own reader, so the worker-ordered completion can only be
        # bounded from below here.
        last_duration = max(
            complete["observed_elapsed_s"] - step_events[-1]["observed_elapsed_s"],
            0.0,
        )
        step_events[-1]["duration_to_complete_s"] = last_duration
    first_step_s = step_events[0]["observed_elapsed_s"] if step_events else None
    if readers_alive:
        # Surviving daemon readers may still mutate shared state: publish
        # immutable snapshots and keep the JSON deterministic.
        events = list(events)
        lines = list(lines)
        step_events = list(step_events)
    swap_growth = (
        swap_end - swap_start
        if swap_start is not None and swap_end is not None
        else None
    )
    # A run that swapped hard and then released before exit must still be
    # classified by its peak sampled growth, not just the end-state delta.
    sampled_peak = max(
        (
            sample.swap_used_bytes
            for sample in samples
            if sample.swap_used_bytes is not None
        ),
        default=None,
    )
    if swap_start is not None and sampled_peak is not None:
        peak_growth = sampled_peak - swap_start
        if swap_growth is None or peak_growth > swap_growth:
            swap_growth = peak_growth
    worker_errored = any(event.get("kind") == "error" for event in events)
    run_status = (
        "completed"
        if process.returncode == 0
        and complete
        and not deadline_exceeded
        and not worker_errored
        else "failed"
    )
    total_s = elapsed_s
    cleanup_s = 0.0
    if (
        run_status == "completed"
        and complete is not None
        and complete.get("worker_time_ns") is not None
    ):
        # Generation latency ends at the worker's own completion event;
        # polling, group cleanup, and reader joins are reported separately.
        total_s = (complete["worker_time_ns"] - started_ns) / 1e9
        cleanup_s = elapsed_s - total_s
    if readers_alive:
        run_status = "failed"
    if integrity_error is not None:
        run_status = "failed"
    if run_status != "completed":
        output_video.unlink(missing_ok=True)
    result = {
        "run": run_index,
        "status": run_status,
        "deadline_exceeded": deadline_exceeded,
        "returncode": process.returncode,
        "cleanup_s": cleanup_s,
        "total_s": total_s,
        "integrity_error": integrity_error,
        "launch_error": None,
        "cold_start_to_first_step_s": first_step_s,
        "artifact_first_byte_s": first_artifact_s,
        "step_median_s": statistics.median(durations) if durations else None,
        "stage_step_median_s": {
            stage: statistics.median(values)
            for stage, values in stage_durations.items()
        },
        "steps": step_events,
        "peak_tree_rss_bytes": max((sample.rss_bytes for sample in samples), default=0),
        "peak_rss_basis": "sampled",
        "mlx_peak_bytes": complete.get("mlx_peak_bytes") if complete else None,
        "swap_start_bytes": swap_start,
        "swap_end_bytes": swap_end,
        "swap_growth_bytes": swap_growth,
        "swap_limited": bool(swap_growth is not None and swap_growth > 64 * 1024**2),
        "thermal_start": thermal_start,
        "thermal_end": thermal_end,
        "output": (
            str(output_video)
            if run_status == "completed" and _artifact_ready(output_video)
            else None
        ),
        "output_bytes": _artifact_size(output_video)
        if run_status == "completed" and _artifact_ready(output_video)
        else 0,
        "samples": [asdict(sample) for sample in samples],
        "events": events,
    }
    (output_dir / f"{stem}.log").write_text("\n".join(lines) + "\n")
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _median(results: list[dict[str, Any]], key: str) -> float | None:
    values = [result[key] for result in results if result.get(key) is not None]
    return statistics.median(values) if values else None


def _gib(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1024**3:.2f} GiB"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _frames_for_seconds(seconds: float, fps: int) -> int:
    return round((seconds * fps - 1) / 8) * 8 + 1


def longest_backtick_run(text: str) -> int:
    longest = current = 0
    for char in str(text):
        current = current + 1 if char == "`" else 0
        longest = max(longest, current)
    return longest


def _markdown(document: dict[str, Any]) -> str:
    results = document["results"]
    completed = [result for result in results if result["status"] == "completed"]
    comparable = [
        result for result in completed if not result.get("swap_limited", False)
    ]
    stages = sorted(
        {
            stage
            for result in comparable
            for stage in result.get("stage_step_median_s", {})
        }
    )
    stage_lines = []
    for stage in stages:
        values = [
            result["stage_step_median_s"][stage]
            for result in comparable
            if stage in result.get("stage_step_median_s", {})
        ]
        stage_lines.append(
            f"- Median stage {stage} diffusion step: {_seconds(statistics.median(values))}"
        )
    lines = [
        f"# LTX benchmark: {document['config']['model']}",
        "",
        f"- Host: {document['machine']['hostname']} "
        f"({document['machine']['hardware_model']}, {document['machine']['architecture']}, "
        f"{_gib(document['machine']['memory_bytes'])}, "
        f"macOS {document['machine']['macos_version']} / {document['machine']['macos_build']})",
        f"- Model identity: {document['config']['model_identity']}",
        f"- Runtime: {document['config']['runtime']} @ {document['config']['runtime_revision']}",
        f"- Workload: {document['config']['width']}x{document['config']['height']}, "
        f"{document['config']['frames']} frames at {document['config']['fps']} fps, "
        f"seed={document['config']['seed']}",
        "- Prompt:\n"
        + "`" * max(3, longest_backtick_run(document["config"]["prompt"]) + 1)
        + "\n"
        + str(document["config"]["prompt"])
        + "\n"
        + "`" * max(3, longest_backtick_run(document["config"]["prompt"]) + 1),
        f"- Runs: {len(completed)}/{len(results)} completed; "
        f"{len(comparable)} comparable without significant swap growth",
        f"- Median cold start to first diffusion step (process cold start; "
        f"the preflight digest cache-warms model files): "
        f"{_seconds(_median(comparable, 'cold_start_to_first_step_s'))}",
        f"- Median diffusion step: {_seconds(_median(comparable, 'step_median_s'))}",
        *stage_lines,
        f"- Median total: {_seconds(_median(comparable, 'total_s'))}",
        "- Peak sampled process-tree RSS (interval "
        + f"{document['config']['sample_interval_s']}s): "
        + _gib(max((r["peak_tree_rss_bytes"] for r in results), default=0)),
        f"- Peak MLX Metal allocator: {_gib(max((r['mlx_peak_bytes'] or 0 for r in results), default=0))}",
        "- Maximum swap growth (system-wide, all processes): "
        + _gib(max((max(r["swap_growth_bytes"] or 0, 0) for r in results), default=0)),
        "",
        "`artifact_first_byte_s` is the first observable MP4 byte, not a decoded-frame callback; "
        "the current runtimes do not expose the latter.",
    ]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--runtime",
        "--backend",
        choices=("mlx23", "ltx25"),
        default="mlx23",
        help="MLX backend/runtime adapter (MPS feasibility is a separate spike)",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", default="768x512")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=120)
    parser.add_argument("--sample-interval", type=float, default=1)
    parser.add_argument(
        "--deadline",
        type=float,
        default=3600,
        help="Per-run wall-clock deadline in seconds; a wedged worker is "
        "terminated at the deadline instead of hanging the sweep",
    )
    parser.add_argument("--output-dir", default="/private/tmp/LTX-benchmark/results")
    parser.add_argument("--ltx25-python", default="")
    parser.add_argument("--runtime-revision", default="unknown")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--invocation", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-video", default="", help=argparse.SUPPRESS)
    return parser


def _content_digest(path: Path) -> str:
    """Stream a SHA-256 over length-prefixed file names and contents.

    Records are ``<len(name)>:name<len(payload)>:payload`` so distinct
    directory layouts cannot collide by concatenation.
    """
    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        name = str(candidate.relative_to(path))
        encoded_name = name.encode()
        digest.update(b"f")
        digest.update(f"{len(encoded_name)}:".encode())
        digest.update(encoded_name)
        size = candidate.stat().st_size
        digest.update(f"{size}:".encode())
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(path: Path) -> str:
    """Cheap integrity check over inode/size/mtime_ns; reads no file bytes.

    Unlike :func:`_content_digest` this does not warm the filesystem cache,
    so it can run before every measured launch without distorting the
    cold-start measurement. It detects accidental rewrites; a deliberate
    mtime-preserving edit is out of scope for per-run checks.
    """
    digest = hashlib.sha256()
    if path.is_file():
        st = path.stat()
        digest.update(f".:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}:".encode())
        return digest.hexdigest()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        st = candidate.stat()
        digest.update(
            f"{candidate.relative_to(path)}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}:".encode()
        )
    return digest.hexdigest()


def _is_load_source(candidate: Path, rel: str) -> bool:
    if "__pycache__" in candidate.parts or rel.endswith((".pyc", ".pyo")):
        return False
    return rel.endswith(".py") or candidate.name in {
        "METADATA",
        "RECORD",
        "PKG-INFO",
    }


def _source_stat_fingerprint(path: Path) -> str:
    """Stat fingerprint over the same file set as :func:`_source_digest`."""
    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        rel = str(candidate.relative_to(path))
        if not _is_load_source(candidate, rel):
            continue
        st = candidate.stat()
        digest.update(f"{rel}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}:".encode())
    return digest.hexdigest()


def _source_digest(path: Path) -> str:
    """Content digest over load-bearing runtime source files.

    Excludes caches and generated artifacts (__pycache__, .pyc, dist-info)
    so the digest is stable across machines and ordinary imports.
    """
    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        rel = str(candidate.relative_to(path))
        if "__pycache__" in candidate.parts or rel.endswith((".pyc", ".pyo")):
            continue
        if not (
            rel.endswith(".py") or candidate.name in {"METADATA", "RECORD", "PKG-INFO"}
        ):
            continue
        digest.update(b"f")
        digest.update(f"{len(rel)}:".encode())
        digest.update(rel.encode())
        size = candidate.stat().st_size
        digest.update(f"{size}:".encode())
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_snapshot(model: str) -> tuple[str, str]:
    """Pin the weights once and return (identity, local_snapshot_path).

    Neither runtime adapter pins a remote revision during load, so the
    controller resolves the registry head, downloads that exact snapshot
    into the shared Hugging Face cache, and hands the worker an immutable
    local path. A local ``--model`` is used as-is. Registry failure fails
    closed: an unpinnable model cannot produce a reproducible report.
    """
    path = Path(model).expanduser()
    if path.is_file():
        raise SystemExit("--model must name a model snapshot directory, not a file")
    if path.exists():
        resolved = str(path.resolve())
        # The directory name alone proves nothing: any mutable directory can
        # wear it. Identity is a content digest over the snapshot, so an
        # in-place rewrite between runs changes the report.
        hf_sha = None
        if (
            len(path.name) == 40
            and path.parent.name == "snapshots"
            and all(c in "0123456789abcdef" for c in path.name)
        ):
            hf_sha = path.name
        refs = path / "refs" / "main"
        if hf_sha is None and refs.is_file():
            candidate = refs.read_text().strip()
            # The revision must have an actual snapshot; the worker cannot
            # load from the cache root, which holds no model files.
            if (
                len(candidate) == 40
                and all(c in "0123456789abcdef" for c in candidate)
                and (path / "snapshots" / candidate).is_dir()
            ):
                hf_sha = candidate
                resolved = str((path / "snapshots" / candidate).resolve())
        if hf_sha is None:
            # A cache root with several snapshots and no ref is ambiguous:
            # guessing a revision would falsely report an immutable identity.
            candidates = [
                p for p in path.glob("snapshots/*") if p.is_dir() and len(p.name) == 40
            ]
            if len(candidates) > 1:
                raise SystemExit(
                    f"{resolved} holds multiple snapshots and no refs/main: "
                    "pass one snapshots/<sha> directory explicitly"
                )
            if len(candidates) == 1:
                hf_sha = candidates[0].name
                resolved = str(candidates[0].resolve())
        if not any(p.is_file() for p in Path(resolved).rglob("*")):
            raise SystemExit(
                f"{resolved} contains no model files; pass a snapshot with weights"
            )
        stat_before = _stat_fingerprint(Path(resolved))
        digest = _content_digest(Path(resolved))
        stat_after = _stat_fingerprint(Path(resolved))
        if stat_before != stat_after:
            raise SystemExit(
                "model files changed while the content digest was computed; "
                "re-run against a stable snapshot"
            )
        revision = f"{hf_sha}+sha256:{digest}" if hf_sha else f"sha256:{digest}"
        return f"{resolved}@{revision}", resolved
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to pin a remote --model; pass a "
            "local snapshot directory instead"
        ) from exc
    try:
        sha = HfApi().model_info(model).sha
    except Exception as exc:
        raise SystemExit(
            f"could not resolve an immutable snapshot for {model!r}: {exc}"
        ) from exc
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
        raise SystemExit(f"{model!r} resolved to {sha!r}, not an immutable commit SHA")
    try:
        snapshot = snapshot_download(repo_id=model, revision=sha)
    except Exception as exc:
        # A report whose weights cannot be pinned is not reproducible: fail
        # closed instead of silently benchmarking a moving branch.
        raise SystemExit(
            f"could not resolve an immutable snapshot for {model!r}: {exc}"
        ) from exc
    snapshot_dir = Path(str(snapshot))
    if not snapshot_dir.is_dir() or not any(
        p.is_file() for p in snapshot_dir.rglob("*")
    ):
        raise SystemExit(
            f"{model!r} resolved to {sha!r} but the downloaded snapshot at "
            f"{snapshot_dir} is missing or empty"
        )
    stat_before = _stat_fingerprint(snapshot_dir)
    digest = _content_digest(snapshot_dir)
    stat_after = _stat_fingerprint(snapshot_dir)
    if stat_before != stat_after:
        raise SystemExit(
            "the downloaded snapshot changed while its content digest was "
            "computed; re-run to re-download a stable snapshot"
        )
    return f"{model}@{sha}+sha256:{digest}", str(snapshot_dir)


def _derive_runtime_revision(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None]:
    """Best-effort runtime version for the report; falls back to unknown."""
    try:
        if args.runtime == "ltx25":
            # The benchmarked code runs under --ltx25-python, not this
            # interpreter: query there or the report may record an unrelated
            # controller-side version.
            output = subprocess.run(
                [
                    args.ltx25_python,
                    "-c",
                    "import ltx_pipelines_mlx; "
                    "print(ltx_pipelines_mlx.__version__); "
                    "print(ltx_pipelines_mlx.__file__ or '')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            lines = output.stdout.strip().splitlines()
            version = lines[0] if lines else "unknown"
            module_file = lines[1] if len(lines) > 1 and lines[1].strip() else None
            if module_file is not None and not Path(module_file).is_file():
                module_file = None
            source_digest = None
            if module_file is not None:
                package_dir = Path(module_file).parent
                stat_before = _source_stat_fingerprint(package_dir)
                source_digest = _source_digest(package_dir)
                if stat_before != _source_stat_fingerprint(package_dir):
                    raise SystemExit(
                        "runtime source files changed while the digest was "
                        "computed; re-run against a stable runtime"
                    )
            return version, module_file, source_digest
        import rapid_mlx as runtime_module
        from rapid_mlx import __version__ as runtime_version

        module_file = getattr(runtime_module, "__file__", None)
        source_digest = None
        if module_file is not None and Path(module_file).is_file():
            package_dir = Path(module_file).parent
            stat_before = _source_stat_fingerprint(package_dir)
            source_digest = _source_digest(package_dir)
            if stat_before != _source_stat_fingerprint(package_dir):
                raise SystemExit(
                    "runtime source files changed while the digest was "
                    "computed; re-run against a stable runtime"
                )
        return runtime_version, module_file, source_digest
    except Exception:
        return "unknown", None, None


def main() -> int:
    args = _parser().parse_args()
    try:
        args.width, args.height = (
            int(part) for part in args.size.lower().split("x", 1)
        )
    except ValueError as exc:
        raise SystemExit("--size must use WIDTHxHEIGHT") from exc
    for numeric_name in ("seconds", "fps", "cooldown", "sample_interval", "deadline"):
        numeric_value = getattr(args, numeric_name)
        if numeric_value is not None and not math.isfinite(numeric_value):
            raise SystemExit(f"--{numeric_name.replace('_', '-')} must be finite")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--size dimensions must be positive")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.fps < 1:
        raise SystemExit("--fps must be at least 1")
    if args.cooldown < 0:
        raise SystemExit("--cooldown must not be negative")
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be positive")
    if args.deadline <= 0:
        raise SystemExit("--deadline must be positive")
    if not args.worker:
        try:
            import psutil  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "psutil is required for process-tree sampling; install the "
                "benchmark dependencies (pip install psutil)"
            ) from exc
    if args.worker:
        return _worker(args)
    if args.frames is None:
        if args.seconds is None:
            raise SystemExit("provide --frames or --seconds")
        args.frames = _frames_for_seconds(args.seconds, args.fps)
    elif args.seconds is not None:
        raise SystemExit("--frames and --seconds are mutually exclusive")
    if args.frames < 9 or args.frames % 8 != 1:
        raise SystemExit("LTX frame count must be 8n+1 and at least 9")
    if args.runtime == "ltx25" and not args.ltx25_python:
        raise SystemExit("--ltx25-python is required for the pinned research runtime")
    args.model_identity, args.resolved_model = _resolve_model_snapshot(args.model)
    # HF-pinned identities use "+sha256:"; bare local directories use
    # "@sha256:" without a registry revision.
    if "+sha256:" in args.model_identity:
        _, _, suffix = args.model_identity.partition("+sha256:")
    else:
        _, _, suffix = args.model_identity.partition("@sha256:")
    if not suffix:
        raise SystemExit("internal error: model identity lacks a content digest")
    args.model_stat_fingerprint = _stat_fingerprint(Path(args.resolved_model))
    args.runtime_revision_source = "user"
    if args.runtime_revision == "unknown":
        observed, args.runtime_module_file, args.runtime_source_digest = (
            _derive_runtime_revision(args)
        )
        args.runtime_revision = observed
        args.runtime_revision_source = "observed"
        if args.runtime_revision == "unknown":
            # A report whose runtime implementation cannot be identified is
            # not reproducible.
            raise SystemExit(
                "--runtime-revision is required: automatic version discovery "
                f"failed for runtime {args.runtime!r}"
            )
    else:
        # A user label alone does not identify the executed code: digest the
        # runtime implementation alongside the label, or fail closed.
        _, args.runtime_module_file, args.runtime_source_digest = (
            _derive_runtime_revision(args)
        )
        if args.runtime_source_digest is None:
            raise SystemExit(
                "--runtime-revision was given, but the runtime "
                f"implementation for {args.runtime!r} could not be located "
                "and digested"
            )
    args.invocation = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    )
    args.event_key = secrets.token_hex(16)
    args.runtime_stat_fingerprint = (
        _source_stat_fingerprint(Path(args.runtime_module_file).parent)
        if args.runtime_module_file
        else None
    )
    results = []
    for index in range(1, args.runs + 1):
        results.append(_run_once(args, index))
        if index < args.runs:
            time.sleep(args.cooldown)
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": _machine_observation(),
        "config": {
            "model": args.model,
            "model_identity": args.model_identity,
            "runtime": args.runtime,
            "runtime_revision": args.runtime_revision,
            "runtime_revision_source": args.runtime_revision_source,
            "runtime_module_file": args.runtime_module_file,
            "runtime_source_digest": args.runtime_source_digest,
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
            "seconds": args.frames / args.fps,
            "seed": args.seed,
            "prompt": args.prompt,
            "runs": args.runs,
            "cooldown_s": args.cooldown,
            "sample_interval_s": args.sample_interval,
            "deadline_s": args.deadline,
            "cold_start_scope": (
                "process: the preflight content digest warms the model "
                "files' page cache before run 1, so cold-start timings "
                "exclude filesystem cold-cache effects"
            ),
        },
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"summary-{args.runtime}-{args.frames}f"
    summary_stem = f"{stem}-{args.invocation}"
    (output_dir / f"{summary_stem}.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )
    markdown = _markdown(document)
    (output_dir / f"{summary_stem}.md").write_text(markdown)
    print(markdown, end="")
    return 0 if all(result["status"] == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
