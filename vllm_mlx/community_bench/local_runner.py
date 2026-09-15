# SPDX-License-Identifier: Apache-2.0
"""Local-only executors for registered Community Benchmark protocols."""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import contextvars
import copy
import inspect
import io
import logging
import math
import multiprocessing
import multiprocessing.process
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from multiprocessing.connection import Connection
from typing import Any

import requests

from .benchmark_contracts import public_prompt, registered_workload
from .hardware import collect, run_conditions

logger = logging.getLogger(__name__)

#: Where the measurement helpers deposit the "after" run-conditions snapshot.
#: It must be taken while the model is still resident — after the last
#: measured round but before the engine/server context tears down — or a
#: memory-saturated run would report normal pressure once its model is gone.
_CONDITIONS_AFTER: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("community_bench_conditions_after", default=None)
)


def _record_conditions_after() -> None:
    """Snapshot run conditions into the active capture, if one is armed."""
    capture = _CONDITIONS_AFTER.get()
    if capture is not None:
        capture["after"] = run_conditions()


from .run_builder import (
    build_run,
    consistent_model_identity,
    execution_config,
    unresolved_model_identity,
    utc_now,
)
from .workspace import LocalRunArchive, describe_case, model_is_cached, plan_for_alias

# Human-readable progress sink (one line per call, no trailing newline).
Progress = Callable[[str], None]

#: Where the text helper deposits the identity of the checkpoint the loader
#: pinned, so a run that fails after loading still archives the facts of
#: the artifact that was actually loaded.
_LOADED_IDENTITY: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("community_bench_loaded_identity", default=None)
)

_VIDEO_JOB_TIMEOUT_S = 3600.0
_VIDEO_POLL_INTERVAL_S = 1.0
_VIDEO_ARTIFACT_DOWNLOAD_TIMEOUT_S = 300.0
_VIDEO_ARTIFACT_PROBE_TIMEOUT_S = 120.0
_MAX_VIDEO_ARTIFACT_BYTES = 1024 * 1024 * 1024


def _raise_for_status(response: requests.Response, *, phase: str) -> None:
    """Preserve a bounded localhost API error instead of only its status line."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail: Any = None
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("error")
        except (requests.exceptions.JSONDecodeError, ValueError):
            detail = None
        if not isinstance(detail, str) or not detail.strip():
            detail = "local server rejected the request"
        detail = detail.strip()[:500]
        raise RuntimeError(
            f"{phase} failed with HTTP {response.status_code}: {detail}"
        ) from exc


class LocalBenchmarkError(RuntimeError):
    """A failed attempt plus any privacy-safe outcome available to the caller."""

    def __init__(
        self,
        message: str,
        run: dict[str, Any] | None,
        *,
        saved: bool,
    ):
        super().__init__(message)
        self.run = run
        self.saved = saved


class BenchmarkCancelledError(RuntimeError):
    """The local runtime reported a terminal user/system cancellation."""


def _failure_code(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, BenchmarkCancelledError):
        return "user_cancelled"
    if (
        isinstance(error, MemoryError)
        or "out of memory" in message
        or "memoryerror" in message
        or ("metal" in message and "alloc" in message)
    ):
        return "runtime_oom"
    if "unsupported" in message:
        return "unsupported_task"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "model" in message and ("invalid" in message or "not found" in message):
        return "invalid_model"
    return "runtime_error"


def _peak_memory_mib(base_url: str) -> int | None:
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        response.raise_for_status()
        peak = response.json().get("metal", {}).get("peak_memory_gb")
        value = float(peak)
        if not math.isfinite(value) or value <= 0:
            return None
        # `/status` reports decimal GB (bytes / 1e9); the contract stores MiB.
        return round(value * 1_000_000_000 / (1 << 20))
    except (requests.RequestException, TypeError, ValueError):
        return None


def _validated_image_count(result: dict[str, Any], *, width: int, height: int) -> int:
    from PIL import Image, UnidentifiedImageError

    data = result.get("data")
    if not isinstance(data, list):
        raise RuntimeError("image benchmark response has no artifact list")
    for item in data:
        encoded = item.get("b64_json") if isinstance(item, dict) else None
        if not isinstance(encoded, str):
            raise RuntimeError("image benchmark response has no base64 artifact")
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                actual_size = image.size
                image.verify()
        except (binascii.Error, OSError, UnidentifiedImageError, ValueError) as exc:
            raise RuntimeError("image benchmark returned an invalid artifact") from exc
        if actual_size != (width, height):
            raise RuntimeError(
                f"image benchmark returned {actual_size[0]}x{actual_size[1]}; "
                f"registered workload requires {width}x{height}"
            )
    return len(data)


def _probe_video_with_ffmpeg(
    path: str, ffmpeg: str, *, desktop_bundle: bool = False
) -> tuple[int, int, int, float]:
    """Probe an MP4 with the small FFmpeg binary shipped by Desktop.

    Desktop deliberately omits imageio/OpenCV. Decode and re-encode through its
    constrained VideoToolbox FFmpeg to prove every frame is readable without
    retaining a second media stack or a second artifact on disk.
    """

    try:
        sink = (
            [
                "-c:v",
                "h264_videotoolbox",
                "-movflags",
                "frag_keyframe+empty_moov",
                "-f",
                "mp4",
                "pipe:1",
            ]
            if desktop_bundle
            else ["-f", "null", "-"]
        )
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0", *sink],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
            timeout=_VIDEO_ARTIFACT_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("video benchmark returned an invalid MP4 artifact") from exc
    output = result.stderr
    stream = re.search(
        r"Stream #\S+.*Video:.*?\b(\d{2,5})x(\d{2,5})\b.*?\b([0-9.]+) fps\b",
        output,
    )
    frame_matches = re.findall(r"\bframe=\s*(\d+)\b", output)
    if result.returncode or stream is None or not frame_matches:
        raise RuntimeError("video benchmark returned an invalid MP4 artifact")
    return (
        int(stream.group(1)),
        int(stream.group(2)),
        int(frame_matches[-1]),
        float(stream.group(3)),
    )


def _is_sidecar_bundled_ffmpeg(ffmpeg: str) -> bool:
    """Return whether FFmpeg and this interpreter share the sidecar root."""

    try:
        ffmpeg_root = os.path.dirname(os.path.dirname(os.path.realpath(ffmpeg)))
        python_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
        )
        return ffmpeg_root == python_root
    except (OSError, ValueError):
        return False


def _probe_video_artifact_unbounded(path: str) -> tuple[int, int, int, float]:
    try:
        import imageio.v2 as imageio
    except ImportError:
        bundled_ffmpeg = os.environ.get("FFMPEG_BINARY")
        ffmpeg = bundled_ffmpeg or shutil.which("ffmpeg")
        if ffmpeg:
            return _probe_video_with_ffmpeg(
                path, ffmpeg, desktop_bundle=_is_sidecar_bundled_ffmpeg(ffmpeg)
            )
        raise RuntimeError("video artifact validation requires rapid-mlx[video]")

    try:
        reader = imageio.get_reader(path, format="ffmpeg")
        try:
            metadata = reader.get_meta_data()
            size = metadata.get("size")
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise RuntimeError("video artifact has no dimensions")
            frames = reader.count_frames()
            fps = float(metadata.get("fps"))
        finally:
            reader.close()
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError("video benchmark returned an invalid MP4 artifact") from exc
    return int(size[0]), int(size[1]), int(frames), fps


def _watch_parent_lifeline(lifeline: Connection, cleanup_path: str | None) -> None:
    """SIGKILL our detached group the instant the parent's lifeline drops.

    The lifeline write end lives only in the parent process, so the kernel
    closes it atomically when the parent exits for any reason — SIGTERM from
    the Desktop supervisor, SIGKILL, or a crash — and the blocking ``poll``
    wakes with EOF. Waking therefore proves the parent is gone: its local
    hard-deadline cleanup and its temporary-file teardown can no longer run,
    so this thread finishes both. Only our own process group is ever
    signalled, and only when we are its leader, so a reused pid or an
    unrelated group can never be hit.
    """

    try:
        lifeline.poll(None)
    except OSError:  # pragma: no cover - a torn lifeline still means gone
        pass
    if cleanup_path is not None:
        try:
            os.unlink(cleanup_path)
        except OSError:
            pass
    pid = os.getpid()
    try:
        if os.getpgid(0) == pid:
            os.killpg(pid, signal.SIGKILL)
    except OSError:  # pragma: no cover - never signal a group we do not own
        pass
    os._exit(1)


def _enter_worker_lifetime(
    lifeline: Connection, *, cleanup_path: str | None = None
) -> None:
    """Detach into an own process group whose lifetime is bound to the parent.

    ``setsid`` keeps the local hard-deadline contract: the parent can reap the
    blocked worker and its descendants (ffmpeg) with ``killpg`` without
    signalling itself. Detaching also escapes the externally supervised
    benchmark process group, so the inherited lifeline restores cancellation
    ownership: a daemon thread waits on it independently of the blocking
    probe/download work and destroys this group the moment the parent dies.
    """

    os.setsid()
    threading.Thread(
        target=_watch_parent_lifeline,
        args=(lifeline, cleanup_path),
        name="parent-lifeline-watchdog",
        daemon=True,
    ).start()


def _video_probe_worker(path: str, sender: Connection, lifeline: Connection) -> None:
    """Probe in its own parent-bound group so ffmpeg descendants are terminable."""

    try:
        _enter_worker_lifetime(lifeline, cleanup_path=path)
        sender.send(("ok", _probe_video_artifact_unbounded(path)))
    except BaseException as exc:
        message = (
            str(exc)
            if isinstance(exc, RuntimeError)
            else "video benchmark returned an invalid MP4 artifact"
        )
        try:
            sender.send(("error", message))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        sender.close()


def _terminate_worker_process(process: multiprocessing.process.BaseProcess) -> None:
    # ``BaseProcess`` is the shared supertype of every start-method Process
    # (spawn/fork/forkserver), so callers are not coupled to one context.
    pid = process.pid
    if pid is None:
        return
    try:
        owns_process_group = os.getpgid(pid) == pid
    except ProcessLookupError:
        owns_process_group = False
    if process.is_alive():
        if owns_process_group:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        if owns_process_group:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.join(timeout=1)


def _run_detached_worker(
    target: Callable[..., None],
    args: tuple[Any, ...],
    *,
    timeout_s: float,
    phase: str,
) -> Any:
    """Supervise a detached worker behind a hard deadline and a lifeline.

    The worker receives a result pipe plus the read end of a dedicated
    lifeline whose write end stays open here for the worker's whole life, so
    the worker can deterministically observe this process dying even though
    ``setsid`` moved it out of the externally supervised benchmark group.
    """

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    lifeline_receiver, lifeline_sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*args, sender, lifeline_receiver))
    process.start()
    sender.close()
    lifeline_receiver.close()
    try:
        if not receiver.poll(timeout_s):
            _terminate_worker_process(process)
            raise TimeoutError(f"video artifact {phase} exceeded its hard deadline")
        try:
            status, payload = receiver.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"video artifact {phase} exited without a result"
            ) from exc
    finally:
        receiver.close()
        process.join(timeout=1)
        if process.is_alive():
            _terminate_worker_process(process)
        # Deliberately outlives the worker: closing earlier would fire the
        # worker's parent-death watchdog during a normal shutdown. Once the
        # worker is reaped no watchdog exists, so closing is inert and never
        # signals a reused pid or group.
        lifeline_sender.close()
    if status != "ok":
        raise RuntimeError(str(payload))
    return payload


def _probe_video_artifact(
    path: str, *, timeout_s: float = _VIDEO_ARTIFACT_PROBE_TIMEOUT_S
) -> tuple[int, int, int, float]:
    """Probe an MP4 behind a hard deadline and reap the whole probe group."""

    payload = _run_detached_worker(
        _video_probe_worker, (path,), timeout_s=timeout_s, phase="probe"
    )
    return tuple(payload)


def _download_video_artifact_unbounded(
    base_url: str, job_id: str, destination_path: str
) -> None:
    """Download and size-check an artifact inside the terminable worker."""

    with requests.get(
        f"{base_url}/videos/{job_id}/content",
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        content_length = (getattr(response, "headers", {}) or {}).get("content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "video artifact has an invalid Content-Length"
                ) from exc
            if declared_bytes < 0:
                raise RuntimeError("video artifact has an invalid Content-Length")
            if declared_bytes > _MAX_VIDEO_ARTIFACT_BYTES:
                raise RuntimeError("video artifact exceeds the 1 GiB safety limit")
        size_bytes = 0
        with open(destination_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                next_size = size_bytes + len(chunk)
                if next_size > _MAX_VIDEO_ARTIFACT_BYTES:
                    raise RuntimeError("video artifact exceeds the 1 GiB safety limit")
                file.write(chunk)
                size_bytes = next_size
        if size_bytes == 0:
            raise RuntimeError("video benchmark returned an empty MP4 artifact")


def _video_download_worker(
    base_url: str,
    job_id: str,
    destination_path: str,
    sender: Connection,
    lifeline: Connection,
) -> None:
    try:
        _enter_worker_lifetime(lifeline, cleanup_path=destination_path)
        _download_video_artifact_unbounded(base_url, job_id, destination_path)
        sender.send(("ok", None))
    except BaseException as exc:
        message = (
            str(exc)
            if isinstance(exc, RuntimeError)
            else "video artifact download failed"
        )
        try:
            sender.send(("error", message))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        sender.close()


def _download_video_artifact(
    base_url: str,
    job_id: str,
    destination_path: str,
    *,
    timeout_s: float = _VIDEO_ARTIFACT_DOWNLOAD_TIMEOUT_S,
) -> None:
    """Download behind a wall-clock deadline immune to socket trickle."""

    _run_detached_worker(
        _video_download_worker,
        (base_url, job_id, destination_path),
        timeout_s=timeout_s,
        phase="download",
    )


def _validated_video_artifact(
    base_url: str,
    job_id: str,
    *,
    width: int,
    height: int,
    frames: int,
    fps: float,
) -> None:
    with tempfile.NamedTemporaryFile(prefix="rapid-benchmark-", suffix=".mp4") as file:
        _download_video_artifact(base_url, job_id, file.name)
        actual_width, actual_height, actual_frames, actual_fps = _probe_video_artifact(
            file.name
        )
    if (actual_width, actual_height) != (width, height):
        raise RuntimeError(
            f"video artifact is {actual_width}x{actual_height}; "
            f"registered workload requires {width}x{height}"
        )
    if actual_frames != frames:
        raise RuntimeError(
            f"video artifact has {actual_frames} frames; "
            f"registered workload requires {frames}"
        )
    if not math.isclose(actual_fps, fps, rel_tol=0, abs_tol=0.01):
        raise RuntimeError(
            f"video artifact is {actual_fps:g} fps; registered workload requires {fps:g}"
        )


def _format_duration(seconds: float) -> str:
    """``~45 s`` / ``~3 min 10 s`` for estimates and elapsed stage times."""

    whole = int(round(max(0.0, seconds)))
    if whole < 60:
        return f"{whole} s"
    minutes, rest = divmod(whole, 60)
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def _report(progress: Progress | None, line: str) -> None:
    """Hand one line to the progress sink; a failing sink is never fatal.

    Progress is presentation. A closed pipe, an encoding error, or a buggy
    sink must not abort a benchmark that is otherwise measuring correctly,
    let alone archive it as failed, so every sink call goes through here.
    """

    if progress is None:
        return
    try:
        progress(line)
    except Exception:
        pass


def _stage_started(progress: Progress | None) -> float | None:
    """Clock a display-only stage; reads no clock when nobody is watching."""

    return time.monotonic() if progress is not None else None


def _stage_finished(
    progress: Progress | None, started: float | None, what: str
) -> None:
    if progress is None or started is None:
        return
    _report(progress, f"{what} in {_format_duration(time.monotonic() - started)}")


def _run_image(
    alias: str,
    *,
    isolate_process_group: bool = True,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    from vllm_mlx.bench._server import serve

    workload = registered_workload("image_generation")
    case = workload["cases"][0]
    payload = {
        "model": alias,
        "prompt": public_prompt(case["case_id"]),
        "n": case["image_count"],
        "size": f"{case['width']}x{case['height']}",
        "response_format": "b64_json",
        "steps": case["steps"],
        "guidance": case["guidance_millionths"] / 1_000_000,
        "seed": case["seed"],
    }
    measurements: list[dict[str, Any]] = []
    _report(progress, f"Starting local image server for {alias} (loads the model)...")
    boot_started = _stage_started(progress)
    with serve(
        alias,
        boot_timeout_s=600,
        isolate_process_group=isolate_process_group,
    ) as server:
        _stage_finished(progress, boot_started, "Server ready")
        endpoint = f"{server['base_url']}/images/generations"
        total = case["warmup_rounds"] + case["measured_rounds"]
        for index in range(total):
            started = time.perf_counter()
            response = requests.post(endpoint, json=payload, timeout=3600)
            _raise_for_status(response, phase="image benchmark request")
            result = response.json()
            duration_ms = (time.perf_counter() - started) * 1000
            if result.get("cancelled", False):
                raise BenchmarkCancelledError("image benchmark was cancelled")
            if index < case["warmup_rounds"]:
                _report(
                    progress,
                    f"{case['case_id']:<16} warmup   "
                    f"{_format_duration(duration_ms / 1000)}",
                )
            else:
                _report(
                    progress,
                    f"{case['case_id']:<16} round "
                    f"{index - case['warmup_rounds'] + 1}/{case['measured_rounds']}  "
                    f"{_format_duration(duration_ms / 1000)}",
                )
            if index >= case["warmup_rounds"]:
                image_count = _validated_image_count(
                    result, width=case["width"], height=case["height"]
                )
                if image_count != case["image_count"]:
                    raise RuntimeError("image benchmark returned an incomplete batch")
                measurements.append(
                    {
                        "case_id": case["case_id"],
                        "round_index": index - case["warmup_rounds"] + 1,
                        "total_duration_ms": duration_ms,
                        "peak_active_memory_mib": _peak_memory_mib(server["base_url"]),
                        "completed": True,
                        "image_count": image_count,
                        "width": case["width"],
                        "height": case["height"],
                    }
                )
        # Still inside ``serve``: the model is resident, so this reflects
        # the state the measurements were produced under.
        _record_conditions_after()
    return measurements


def _run_video(
    alias: str,
    *,
    isolate_process_group: bool = True,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    from vllm_mlx.bench._server import serve

    workload = registered_workload("video_generation")
    case = workload["cases"][0]
    payload = {
        "model": alias,
        "prompt": public_prompt(case["case_id"]),
        "size": f"{case['width']}x{case['height']}",
        "frames": str(case["frames"]),
        "fps": str(case["fps_milli"] // 1000),
        "seed": str(case["seed"]),
        "guidance_scale": str(case["guidance_millionths"] / 1_000_000),
    }
    _report(progress, f"Starting local video server for {alias} (loads the model)...")
    boot_started = _stage_started(progress)
    with serve(
        alias,
        boot_timeout_s=900,
        # The registered protocol is a 20-step Wan workload. Wan otherwise
        # delegates to a backend default that may change between releases.
        extra_env={"RAPID_MLX_WAN_STEPS": str(case["steps"])},
        isolate_process_group=isolate_process_group,
    ) as server:
        _stage_finished(progress, boot_started, "Server ready")
        _report(progress, f"{case['case_id']:<16} round 1/1  generating...")
        started = time.perf_counter()
        response = requests.post(
            f"{server['base_url']}/videos", data=payload, timeout=30
        )
        _raise_for_status(response, phase="video benchmark request")
        job = response.json()
        deadline = time.monotonic() + _VIDEO_JOB_TIMEOUT_S
        active_statuses = {"queued", "running", "in_progress", "processing"}
        status = str(job.get("status", "")).lower()
        while status in active_statuses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"video benchmark timed out after {_VIDEO_JOB_TIMEOUT_S:g} seconds"
                )
            time.sleep(min(_VIDEO_POLL_INTERVAL_S, remaining))
            response = requests.get(
                f"{server['base_url']}/videos/{job['id']}",
                timeout=min(10, max(0.001, deadline - time.monotonic())),
            )
            _raise_for_status(response, phase="video benchmark status poll")
            job = response.json()
            status = str(job.get("status", "")).lower()
        if status in {"cancelled", "canceled"}:
            raise BenchmarkCancelledError(
                (job.get("error") or {}).get(
                    "message", "video generation was cancelled"
                )
            )
        if status != "completed":
            raise RuntimeError(
                (job.get("error") or {}).get(
                    "message", f"video generation ended with status {status!r}"
                )
            )
        duration_ms = (time.perf_counter() - started) * 1000
        _report(
            progress,
            f"{case['case_id']:<16} round 1/1  {_format_duration(duration_ms / 1000)}",
        )
        expected_size = f"{case['width']}x{case['height']}"
        if (
            job.get("size") != expected_size
            or type(job.get("frames")) is not int
            or job["frames"] != case["frames"]
            or type(job.get("fps")) is not int
            or job["fps"] * 1000 != case["fps_milli"]
        ):
            raise RuntimeError(
                "video benchmark artifact metadata does not match the registered workload"
            )
        _validated_video_artifact(
            server["base_url"],
            str(job["id"]),
            width=case["width"],
            height=case["height"],
            frames=case["frames"],
            fps=case["fps_milli"] / 1000,
        )
        # Still inside ``serve``: the model is resident, so this reflects
        # the state the measurement was produced under.
        _record_conditions_after()
        return [
            {
                "case_id": case["case_id"],
                "round_index": 1,
                "total_duration_ms": duration_ms,
                "peak_active_memory_mib": _peak_memory_mib(server["base_url"]),
                "completed": True,
                "frames": case["frames"],
                "width": case["width"],
                "height": case["height"],
            }
        ]


def _estimate_remaining_s(
    cases: list[dict[str, Any]],
    sample: Any,
    *,
    done_label: str,
) -> float | None:
    """Project total remaining wall time from the first completed round.

    Uses the observed prefill and decode rates of ``sample`` (a
    ``RoundResult``) to price every remaining round of every case, so the
    long case is weighted by its own token counts rather than assumed to
    cost the same as the short one. Returns ``None`` when rates are missing.
    """

    prefill_tps = getattr(sample, "prefill_tps", None)
    decode_tps = getattr(sample, "decode_tps", None)
    if not (
        isinstance(prefill_tps, int | float)
        and isinstance(decode_tps, int | float)
        and prefill_tps > 0
        and decode_tps > 0
    ):
        return None
    remaining = 0.0
    for case in cases:
        rounds = int(case.get("warmup_rounds", 0)) + int(case.get("measured_rounds", 0))
        if case.get("case_id") == done_label:
            rounds -= 1
        per_round = (
            case.get("target_prompt_tokens", 0) / prefill_tps
            + case.get("target_output_tokens", 0) / decode_tps
        )
        remaining += max(0, rounds) * per_round
    return remaining


def _text_round_observer(progress: Progress, cases: list[dict[str, Any]]):
    """Build the per-round progress callback for the text protocol."""

    estimated = False

    def on_round(label: str, phase: str, index: int, total: int, result: Any) -> None:
        nonlocal estimated
        if phase == "warmup":
            suffix = f" {index}/{total}" if total > 1 else ""
            _report(progress, f"{label:<16} warmup{suffix}")
            if not estimated:
                estimated = True
                remaining = _estimate_remaining_s(cases, result, done_label=label)
                if remaining is not None:
                    _report(
                        progress,
                        "Estimated time remaining: "
                        f"~{_format_duration(remaining)} (from the warmup rate)",
                    )
            return
        tps = getattr(result, "decode_tps", None)
        rate = f"  {tps:6.1f} tok/s" if isinstance(tps, int | float) else ""
        _report(progress, f"{label:<16} round {index}/{total}{rate}")

    return on_round


def _loader_target(model_name: str, repo_id: str) -> str:
    """What to hand ``load_model_with_fallback`` for ``model_name``.

    The loader only resolves an alias when that alias declares a subfolder
    (``lfm2.5-2.6b-4bit`` -> ``…/snapshots/<sha>/4bit``); a plain alias would
    reach the Hub verbatim and 404 (``serve`` resolves ``hf_path`` first for
    the same reason). So: alias when it pins a subfolder — explicit-alias
    precedence keeps the measured checkpoint equal to the recorded identity —
    and the resolved repo id otherwise.
    """
    from vllm_mlx.model_aliases import resolve_subfolder

    target = model_name if resolve_subfolder(model_name) else repo_id
    # The loader treats an existing path as authoritative before it consults
    # the registry, so a directory named like the alias (or the repo id) in
    # the working directory would be measured while the catalog identity is
    # recorded. Refuse rather than record numbers for unknown weights.
    if os.path.exists(target):
        raise RuntimeError(
            f"{target!r} is also a local path here; the benchmark measures the "
            "catalog checkpoint only. Run it from a directory without that path."
        )
    return target


def _uses_serving_benchmark_engine(target: str) -> bool:
    """Whether ``target`` needs the architecture-owned serving loader."""

    from vllm_mlx.api.utils import is_mllm_model

    return bool(is_mllm_model(target))


def _text_loader_needs_serving_fallback(exc: BaseException) -> bool:
    """Whether mlx-lm rejected an architecture owned by the serving runtime."""

    message = str(exc).strip().lower()
    missing_module = getattr(exc, "name", None)
    return (
        isinstance(missing_module, str) and missing_module.startswith("mlx_lm.models.")
    ) or bool(re.fullmatch(r"model type [a-z0-9_.-]+ not supported\.", message))


class _ServingBenchmarkAdapter:
    """Expose the standardized benchmark protocol over ``BatchedEngine``.

    Architecture-owned text backbones such as GLM-5 Next are loaded by the
    production multimodal lane rather than mlx-lm.  Keeping this adapter tiny
    lets the benchmark reuse that proven loader/scheduler while preserving the
    runner's exact-token and output-accounting contract.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._requests: dict[str, tuple[str | list[int], Any]] = {}

    async def __aenter__(self) -> _ServingBenchmarkAdapter:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def add_request(self, prompt: str | list[int], sampling_params: Any) -> str:
        if getattr(sampling_params, "stop_token_ids", None):
            raise ValueError(
                "the serving benchmark adapter does not support caller-supplied "
                "stop token IDs"
            )
        request_id = uuid.uuid4().hex
        self._requests[request_id] = (prompt, sampling_params)
        return request_id

    async def stream_outputs(
        self, request_id: str, timeout: float | None = None
    ) -> AsyncIterator[Any]:
        from vllm_mlx.request import RequestOutput

        prompt, sampling = self._requests.pop(request_id)
        output_token_ids: list[int] = []

        stream = self._engine.stream_generate(
            prompt,
            request_id=request_id,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            repetition_penalty=sampling.repetition_penalty,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            ignore_eos=sampling.ignore_eos,
            seed=sampling.seed,
            stop=sampling.stop,
        )

        def adapt(output: Any) -> Any:
            output_token_ids.extend(output.tokens)
            return RequestOutput(
                request_id=request_id,
                new_token_ids=list(output.tokens),
                new_text=output.new_text,
                output_token_ids=list(output_token_ids),
                output_text=output.text,
                finished=output.finished,
                finish_reason=output.finish_reason,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                cached_tokens=output.cached_tokens,
            )

        completed = False
        try:
            if timeout is None:
                async for output in stream:
                    yield adapt(output)
                completed = True
                return

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while True:
                try:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    output = await asyncio.wait_for(anext(stream), timeout=remaining)
                except StopAsyncIteration:
                    completed = True
                    return
                yield adapt(output)
        finally:
            if not completed:
                abort = getattr(self._engine, "abort_request", None)
                if abort is not None:
                    try:
                        result = abort(request_id)
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        logger.warning(
                            "benchmark request abort cleanup failed for %s",
                            request_id,
                            exc_info=True,
                        )
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.warning(
                        "benchmark request stream cleanup failed for %s",
                        request_id,
                        exc_info=True,
                    )


def _next_generation_chunk(iterator: Any) -> tuple[bool, Any | None]:
    """Advance a synchronous MLX generator without leaking StopIteration.

    ``StopIteration`` cannot cross an asyncio Future boundary: asyncio turns it
    into a TypeError.  A tagged result keeps every model step on the one MLX
    executor thread while preserving token-by-token timing at the async layer.
    """

    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _deepseek_v41_stream_generate(*args: Any, **kwargs: Any) -> Any:
    """Load the MLX-only generator at execution time.

    Keeping this boundary in the benchmark adapter lets Linux contract tests
    exercise scheduling and provenance without importing the large-model MLX
    implementation.  A real benchmark still resolves the exact product
    runtime here before its first token.
    """
    from vllm_mlx.models.deepseek_v41_native.serving import stream_generate

    return stream_generate(*args, **kwargs)


def _deepseek_v41_load_product_runtime(*args: Any, **kwargs: Any) -> Any:
    """Load the pinned product runtime behind the same no-MLX boundary."""
    from vllm_mlx.models.deepseek_v41_native.serving import load_product_runtime

    return load_product_runtime(*args, **kwargs)


def _deepseek_v41_input_limit() -> int:
    """Return the qualified product limit without importing MLX on Linux."""
    from vllm_mlx.models.deepseek_v41_native.serving import MAX_INPUT_TOKENS

    return MAX_INPUT_TOKENS


class _DeepSeekV41BenchmarkAdapter:
    """Registered-token benchmark facade for the qualified serial runtime."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        runtime: Any,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._runtime = runtime
        self._executor = executor
        self._requests: dict[str, tuple[list[int], Any]] = {}

    async def __aenter__(self) -> _DeepSeekV41BenchmarkAdapter:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def add_request(self, prompt: str | list[int], sampling_params: Any) -> str:
        if not isinstance(prompt, list) or not all(
            type(token) is int for token in prompt
        ):
            raise ValueError(
                "DeepSeek V4.1 Community Benchmark requires registered token IDs"
            )
        unsupported = (
            sampling_params.temperature != 0.0
            or sampling_params.top_p != 1.0
            or sampling_params.top_k != 0
            or sampling_params.min_p != 0.0
            or sampling_params.repetition_penalty != 1.0
            or sampling_params.presence_penalty != 0.0
            or sampling_params.frequency_penalty != 0.0
            or bool(sampling_params.stop)
            or bool(sampling_params.stop_token_ids)
            or not sampling_params.ignore_eos
        )
        if unsupported:
            raise ValueError(
                "DeepSeek V4.1 Community Benchmark supports only the registered "
                "greedy, ignore-EOS workload"
            )
        request_id = uuid.uuid4().hex
        self._requests[request_id] = (list(prompt), sampling_params)
        return request_id

    async def stream_outputs(
        self, request_id: str, timeout: float | None = None
    ) -> AsyncIterator[Any]:
        from vllm_mlx.request import RequestOutput

        prompt, sampling = self._requests.pop(request_id)
        iterator = _deepseek_v41_stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            runtime=self._runtime,
            max_tokens=sampling.max_tokens,
            ignore_eos=True,
        )
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        output_token_ids: list[int] = []
        output_text = ""
        while True:
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise asyncio.TimeoutError
            future = loop.run_in_executor(
                self._executor, _next_generation_chunk, iterator
            )
            if remaining is not None:
                has_chunk, chunk = await asyncio.wait_for(future, remaining)
            else:
                has_chunk, chunk = await future
            if not has_chunk:
                return
            assert chunk is not None
            output_token_ids.append(chunk.token)
            output_text += chunk.text
            finished = len(output_token_ids) == sampling.max_tokens
            yield RequestOutput(
                request_id=request_id,
                new_token_ids=[chunk.token],
                new_text=chunk.text,
                output_token_ids=list(output_token_ids),
                output_text=output_text,
                finished=finished,
                finish_reason="length" if finished else None,
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.generation_tokens,
                cached_tokens=0,
            )


def _is_deepseek_v41_benchmark_target(repo_id: str) -> bool:
    from vllm_mlx.models.deepseek_v41_native.artifacts import is_product_target

    return is_product_target(repo_id)


def _text_speculative_execution(repo_id: str) -> dict[str, Any] | None:
    if not _is_deepseek_v41_benchmark_target(repo_id):
        return None
    from vllm_mlx.models.deepseek_v41_native.artifacts import (
        mtp_model_identity_digest,
    )

    return {
        "method": "dspark",
        "max_draft_tokens": 5,
        "draft_model_identity_digest": mtp_model_identity_digest(),
    }


def _with_v41_sidecar_identity(
    primary: dict[str, Any], mtp_snapshot_path: str
) -> dict[str, Any]:
    """Record the external DSpark sidecar beside the target checkpoint."""
    from vllm_mlx.models.deepseek_v41_native.artifacts import MTP_REPO

    combined = copy.deepcopy(primary)
    sidecar = unresolved_model_identity(
        MTP_REPO, "text_generation", snapshot_path=mtp_snapshot_path
    )["components"][0]
    sidecar["component_id"] = "sidecar"
    sidecar["role"] = "other"
    combined["components"].append(sidecar)
    combined["components"].sort(key=lambda component: component["component_id"])
    return combined


async def _text_measurements(
    model_name: str,
    catalog_repo_id: str | None = None,
    *,
    progress: Progress | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any] | None]:
    """Measure ``model_name`` — the catalog alias, not the bare repo id.

    The loader resolves an explicit alias to that alias's checkpoint
    (repo + subfolder). A bare repo id would instead prefer whatever
    variant was last ``pull``ed, so a run labelled ``lfm2.5-2.6b-4bit``
    could silently measure the 8-bit checkpoint. Loading by alias keeps the
    measured artifact and the recorded identity the same thing.
    """
    from vllm_mlx.engine.batched import BatchedEngine
    from vllm_mlx.engine_core import (
        AsyncEngineCore,
        EngineConfig,
        _init_mlx_step_thread,
    )
    from vllm_mlx.model_aliases import resolve_model, resolve_subfolder
    from vllm_mlx.scheduler import SchedulerConfig
    from vllm_mlx.service.helpers import get_model_max_context
    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    from .runner import _reported_token_count, run_standardized_bench

    # The catalog's own repo id is authoritative: ``resolve_model`` may answer
    # with a same-named local directory or an extra-model-root path, which
    # would measure unrelated weights under the catalog identity.
    repo_id = catalog_repo_id or resolve_model(model_name)
    target = _loader_target(model_name, repo_id)
    from .workspace import benchmark_runtime_readiness

    runtime = benchmark_runtime_readiness(model_name, "text_generation")
    if runtime.get("status") == "unavailable":
        raise RuntimeError(runtime.get("message") or "benchmark runtime unavailable")

    # Resolve against the concrete catalog checkpoint, just like serve. Most
    # text-generation aliases (including DeepSeek V4) remain on the native
    # text lane. Architecture-owned backbones such as GLM-5 Next use the
    # production MLLM loader instead of failing in mlx-lm. Dedicated runtimes
    # such as DeepSeek V4.1 use their own adapter below.
    use_v41_runtime = _is_deepseek_v41_benchmark_target(repo_id)
    use_serving_engine = not use_v41_runtime and _uses_serving_benchmark_engine(target)

    workload = registered_workload("text_generation")
    executor = None
    serving_engine = None
    mtp_source: str | None = None
    try:
        if progress is not None:
            cached = model_is_cached(repo_id)
            if cached is True:
                source = "from the local Hugging Face cache"
            elif cached is False:
                source = "not cached yet; downloading from Hugging Face first"
            else:
                source = "cache state unknown; downloads anything missing"
            _report(progress, f"Loading {repo_id} ({source})...")
        load_started = _stage_started(progress)

        async def start_serving_engine() -> tuple[Any, Any]:
            scheduler = SchedulerConfig(
                max_num_seqs=1,
                max_concurrent_requests=1,
                prefill_batch_size=1,
                completion_batch_size=1,
                enable_prefix_cache=False,
                spec_decode="none",
            )
            candidate = BatchedEngine(
                target,
                scheduler_config=scheduler,
                force_mllm=True,
                profile_name=model_name,
            )
            try:
                await candidate.start()
                tokenizer = candidate.tokenizer
            except BaseException:
                try:
                    await candidate.stop()
                except Exception:
                    logger.warning(
                        "failed benchmark engine cleanup after startup error",
                        exc_info=True,
                    )
                raise
            return candidate, tokenizer

        if use_v41_runtime:
            from vllm_mlx.models.deepseek_v41_native.artifacts import (
                MTP_REPO,
                MTP_REVISION,
                TARGET_REVISION,
                download_mtp_snapshot,
                download_target_snapshot,
            )

            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mlx-step",
                initializer=_init_mlx_step_thread,
            )

            def load_v41_runtime() -> tuple[Any, Any, Any, str, str]:
                target_path = download_target_snapshot()
                mtp_path = download_mtp_snapshot()
                model, tokenizer, runtime = _deepseek_v41_load_product_runtime(
                    str(target_path),
                    str(mtp_path),
                    target_revision=TARGET_REVISION,
                    mtp_revision=MTP_REVISION,
                    mtp_identity=MTP_REPO,
                )
                return model, tokenizer, runtime, str(target_path), str(mtp_path)

            model, tokenizer, v41_runtime, loaded_source, mtp_source = executor.submit(
                load_v41_runtime
            ).result()
        elif use_serving_engine:
            serving_engine, tokenizer = await start_serving_engine()
            loaded_source, model = "", None
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mlx-step",
                initializer=_init_mlx_step_thread,
            )
            try:
                loaded = executor.submit(
                    load_model_with_fallback, target, return_source=True
                ).result()
            except (ValueError, ModuleNotFoundError) as exc:
                if not _text_loader_needs_serving_fallback(exc):
                    raise
                executor.shutdown(wait=False, cancel_futures=True)
                executor = None
                serving_engine, tokenizer = await start_serving_engine()
                loaded_source, model = "", None
            else:
                # ``return_source`` appends the concrete checkpoint source; a
                # test double may still answer with the plain pair.
                model, tokenizer = loaded[0], loaded[1]
                loaded_source = loaded[2] if len(loaded) > 2 else ""
        _stage_finished(progress, load_started, "Model loaded")
        # Identity of the checkpoint the loader just pinned, read now — before
        # minutes of measurement give another process time to move refs/main.
        loaded_identity = unresolved_model_identity(
            repo_id,
            "text_generation",
            resolve_subfolder(model_name),
            snapshot_path=(
                loaded_source
                if isinstance(loaded_source, str) and os.path.isdir(loaded_source)
                else None
            ),
        )
        if mtp_source is not None:
            loaded_identity = _with_v41_sidecar_identity(loaded_identity, mtp_source)
        capture = _LOADED_IDENTITY.get()
        if capture is not None:
            capture["identity"] = loaded_identity
        scheduler = SchedulerConfig(
            max_num_seqs=1,
            max_concurrent_requests=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            enable_prefix_cache=False,
            spec_decode="none",
        )
        config = EngineConfig(model_name=repo_id, scheduler_config=scheduler)
        engine_context: Any
        context_source: Any
        if use_v41_runtime:
            assert executor is not None
            engine_context = _DeepSeekV41BenchmarkAdapter(
                model, tokenizer, v41_runtime, executor
            )
            context_source = engine_context
        elif serving_engine is not None:
            engine_context = _ServingBenchmarkAdapter(serving_engine)
            # ``get_model_max_context`` intentionally accepts serving-engine
            # wrappers: it reads ``._model`` first and then the wrapper's
            # tokenizer/local config. BatchedEngine exposes both after start.
            context_source = serving_engine
        else:
            engine_context = AsyncEngineCore(
                model, tokenizer, config, executor=executor
            )
            context_source = None
        async with engine_context as engine:
            if use_v41_runtime:
                context_length = _deepseek_v41_input_limit()
            else:
                context_length = get_model_max_context(
                    context_source if context_source is not None else engine.engine
                )
            result = await run_standardized_bench(
                engine,
                tokenizer,
                sampling="greedy",
                registered_token_ids=True,
                on_round=(
                    _text_round_observer(progress, workload["cases"])
                    if progress is not None
                    else None
                ),
            )
            _record_conditions_after()
    finally:
        try:
            if serving_engine is not None:
                try:
                    await serving_engine.stop()
                except Exception:
                    logger.warning(
                        "benchmark serving-engine shutdown failed",
                        exc_info=True,
                    )
        finally:
            # ThreadPoolExecutor workers are non-daemon and Python joins them
            # again during interpreter shutdown. Leaving this executor live can
            # make the CLI appear finished while its process (and Desktop memory
            # lease) remains stuck. AsyncEngineCore has exited at this point, so
            # cancel work that never started and synchronously reap the owned
            # worker. The Desktop's outer CLI process group remains the hard
            # cancellation boundary for native MLX calls that cannot be
            # interrupted in-process.
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    buckets = (result.short, result.long)
    measurements: list[dict[str, Any]] = []
    peak = result.peak_ram_mb
    for case, bucket in zip(workload["cases"], buckets, strict=True):
        for index, round_result in enumerate(bucket.rounds_raw, start=1):
            decode_ms = (
                (case["target_output_tokens"] - 1) / round_result.decode_tps * 1000
            )
            measurements.append(
                {
                    "case_id": case["case_id"],
                    "round_index": index,
                    "total_duration_ms": round_result.ttft_ms + decode_ms,
                    "peak_active_memory_mib": peak,
                    "completed": True,
                    "prompt_tokens": _reported_token_count(
                        round_result.prompt_tokens, case["target_prompt_tokens"]
                    ),
                    "output_tokens": _reported_token_count(
                        round_result.output_tokens, case["target_output_tokens"]
                    ),
                    "ttft_ms": round_result.ttft_ms,
                    "decode_duration_ms": decode_ms,
                }
            )
    return measurements, context_length, loaded_identity


def _is_dedicated_process_group_leader() -> bool:
    """True when this process leads its own dedicated POSIX process group.

    ``inherit_process_group`` is an internal topology contract, not a
    privilege: the Desktop supervisor (``ProcessGroupChild.spawn``) launches
    the benchmark CLI via ``POSIX_SPAWN_SETPGROUP`` with pgroup 0, so the
    CLI's pid *is* its pgid and the supervisor owns exactly that group. A CLI
    that instead inherited a shell script's or another supervisor's group
    must not put the server into it: group teardown would then signal
    unrelated sibling jobs, or be impossible without doing so. ``os.getpgrp``
    does not exist on Windows; when the topology cannot be verified this
    fails closed.
    """

    getpgrp = getattr(os, "getpgrp", None)
    if getpgrp is None:
        return False
    try:
        return bool(os.getpid() == getpgrp())
    except OSError:  # pragma: no cover - kernel refused to report the group
        return False


def _announce_plan(progress: Progress | None, plan: dict[str, Any]) -> None:
    """Print what is about to run so a multi-minute run is never silent."""

    if progress is None:
        return
    model = plan["model"]
    cases = plan.get("workload", {}).get("cases", [])
    warmup = sum(int(case.get("warmup_rounds", 0)) for case in cases)
    measured = sum(int(case.get("measured_rounds", 0)) for case in cases)
    _report(
        progress,
        f"Benchmarking {model['alias']} ({model['task_type']}): "
        f"{len(cases)} case{'s' if len(cases) != 1 else ''}, "
        f"{warmup} warmup + {measured} measured rounds in total",
    )
    for case in cases:
        _report(progress, f"  {describe_case(case)}")


def run_local(
    alias: str,
    *,
    archive: LocalRunArchive | None = None,
    inherit_process_group: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run a registered protocol, validate it, and save it locally only.

    ``progress`` receives short human-readable status lines (model load,
    per-round throughput, time estimate). It is never given the result and
    nothing is written to stdout here, so ``--json`` callers stay clean.
    """

    if inherit_process_group and not _is_dedicated_process_group_leader():
        raise LocalBenchmarkError(
            "--inherit-process-group requires the benchmark CLI to be the "
            "leader of its own dedicated process group (the supervisor spawn "
            "topology); this process shares its parent's group, so the server "
            "tree could not be torn down safely. Re-run without the flag to "
            "keep the benchmark server in an isolated process group.",
            None,
            saved=False,
        )
    try:
        plan = plan_for_alias(alias)
    except Exception as exc:
        raise LocalBenchmarkError(str(exc), None, saved=False) from exc
    model = plan["model"]
    started_at = utc_now()
    task_type = model["task_type"]
    if task_type not in {"text_generation", "image_generation", "video_generation"}:
        raise ValueError(f"unsupported task type {task_type!r}")
    destination = archive or LocalRunArchive.default()
    _announce_plan(progress, plan)
    # The text helper deposits the identity of the checkpoint the loader
    # pinned; arm the capture for this run only and disarm on every path.
    loaded: dict[str, Any] = {}
    loaded_token = _LOADED_IDENTITY.set(loaded)
    try:
        return _run_local_measured(
            alias,
            plan,
            model,
            task_type,
            started_at,
            destination,
            inherit_process_group,
            loaded,
            progress=progress,
        )
    finally:
        _LOADED_IDENTITY.reset(loaded_token)


def _run_local_measured(
    alias: str,
    plan: dict[str, Any],
    model: dict[str, Any],
    task_type: str,
    started_at: str,
    destination: LocalRunArchive,
    inherit_process_group: bool,
    loaded: dict[str, Any],
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    context_length = None
    hardware = None
    software = None
    execution = None
    conditions_before = None
    conditions_after = None
    measurements_completed = False
    # Resolve the identity from the cache BEFORE loading: the loader pins a
    # snapshot at load time, and reading the config after the run could
    # describe a newer snapshot if another pull advanced refs/main meanwhile.
    model_identity = unresolved_model_identity(
        model["repo_id"], task_type, model.get("subfolder")
    )
    try:
        hardware, software = collect()
        # Snapshot the volatile machine state (power, thermal, memory
        # pressure) before the model is loaded and again right after the last
        # measurement, so a reader can tell a battery/throttled run apart.
        conditions_before = run_conditions()
        capture: dict[str, Any] = {}
        capture_token = _CONDITIONS_AFTER.set(capture)
        try:
            if task_type == "text_generation":
                measured = asyncio.run(
                    _text_measurements(alias, model["repo_id"], progress=progress)
                )
                measurements, context_length = measured[0], measured[1]
                # The identity read right after the loader pinned its snapshot
                # (3-tuple from the real helper; test doubles may return two).
                if len(measured) > 2 and measured[2] is not None:
                    model_identity = measured[2]
            elif task_type == "image_generation":
                measurements = _run_image(
                    alias,
                    isolate_process_group=not inherit_process_group,
                    progress=progress,
                )
            elif task_type == "video_generation":
                measurements = _run_video(
                    alias,
                    isolate_process_group=not inherit_process_group,
                    progress=progress,
                )
        finally:
            # Disarm on every path (success, failure, cancellation) so a
            # stray late call in this process can never mutate a stale
            # capture.
            _CONDITIONS_AFTER.reset(capture_token)
        measurements_completed = True
        # Taken by the helper before its engine/server context tore down. A
        # helper that never captured leaves ``after`` unknown; probing here,
        # after the model is gone, would misreport a memory-saturated run.
        conditions_after = capture.get("after")
        # ``_LOADED_IDENTITY`` is disarmed by ``run_local`` on every path. An
        # identity the helper deposited describes the snapshot the loader
        # actually pinned, so it stands even if refs/main moved during the
        # measurements. Only a pre-load cache read needs reconciling against a
        # post-run one, and only then does a moved snapshot degrade to unknown.
        if loaded.get("identity") is not None:
            model_identity = loaded["identity"]
        else:
            model_identity = consistent_model_identity(
                model_identity,
                unresolved_model_identity(
                    model["repo_id"], task_type, model.get("subfolder")
                ),
                model["repo_id"],
                task_type,
            )
        execution = execution_config(
            task_type,
            context_length=context_length,
            speculative_decoding=(
                _text_speculative_execution(model["repo_id"])
                if task_type == "text_generation"
                else None
            ),
        )
        run = build_run(
            repo_id=model["repo_id"],
            subfolder=model.get("subfolder"),
            model_identity=model_identity,
            task_type=task_type,
            hardware=hardware,
            software=software,
            started_at=started_at,
            measurements=measurements,
            context_length=context_length,
            execution=execution,
            conditions_before=conditions_before,
            conditions_after=conditions_after,
        )
    except (Exception, asyncio.CancelledError) as exc:
        # ``CancelledError`` is a BaseException: without naming it here a
        # cancelled benchmark would skip the archived cancellation record
        # (and its before-snapshot) entirely.
        # A run that failed after loading still describes the checkpoint
        # that was actually loaded.
        if loaded.get("identity") is not None:
            model_identity = loaded["identity"]
        if measurements_completed and execution is None:
            raise LocalBenchmarkError(
                f"benchmark completed but result could not be constructed: {exc}",
                None,
                saved=False,
            ) from exc
        if hardware is None or software is None:
            failure_code = "machine_probe_failed"
        elif isinstance(exc, asyncio.CancelledError):
            failure_code = "user_cancelled"
        else:
            failure_code = _failure_code(exc)
        try:
            if execution is None:
                execution = execution_config(
                    task_type,
                    context_length=context_length,
                    speculative_decoding=(
                        _text_speculative_execution(model["repo_id"])
                        if task_type == "text_generation"
                        else None
                    ),
                )
            failed = build_run(
                repo_id=model["repo_id"],
                subfolder=model.get("subfolder"),
                model_identity=model_identity,
                task_type=task_type,
                hardware=hardware,
                software=software,
                started_at=started_at,
                status=(
                    "cancelled"
                    if isinstance(exc, BenchmarkCancelledError | asyncio.CancelledError)
                    else "failed"
                ),
                failure_code=failure_code,
                context_length=context_length,
                execution=execution,
                conditions_before=conditions_before,
                conditions_after=conditions_after,
            )
        except Exception as envelope_exc:
            raise LocalBenchmarkError(
                f"{exc}; failed outcome could not be constructed: {envelope_exc}",
                None,
                saved=False,
            ) from exc
        try:
            destination.save(failed)
        except Exception as archive_exc:
            raise LocalBenchmarkError(
                f"{exc}; failed outcome could not be saved: {archive_exc}",
                failed,
                saved=False,
            ) from exc
        raise LocalBenchmarkError(str(exc), failed, saved=True) from exc

    try:
        destination.save(run)
    except Exception as exc:
        raise LocalBenchmarkError(
            f"benchmark completed but result could not be saved: {exc}",
            run,
            saved=False,
        ) from exc
    return run


__all__ = ["LocalBenchmarkError", "Progress", "run_local"]
