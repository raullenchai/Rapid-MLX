# SPDX-License-Identifier: Apache-2.0
"""Local-only executors for registered Community Benchmark protocols."""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import io
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
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import requests

from .benchmark_contracts import public_prompt, registered_workload
from .hardware import collect
from .run_builder import build_run, execution_config, utc_now
from .workspace import LocalRunArchive, plan_for_alias

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


def _run_image(
    alias: str, *, isolate_process_group: bool = True
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
    with serve(
        alias,
        boot_timeout_s=600,
        isolate_process_group=isolate_process_group,
    ) as server:
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
    return measurements


def _run_video(
    alias: str, *, isolate_process_group: bool = True
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
    with serve(
        alias,
        boot_timeout_s=900,
        # The registered protocol is a 20-step Wan workload. Wan otherwise
        # delegates to a backend default that may change between releases.
        extra_env={"RAPID_MLX_WAN_STEPS": str(case["steps"])},
        isolate_process_group=isolate_process_group,
    ) as server:
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


async def _text_measurements(repo_id: str) -> tuple[list[dict[str, Any]], int]:
    from vllm_mlx.engine_core import (
        AsyncEngineCore,
        EngineConfig,
        _init_mlx_step_thread,
    )
    from vllm_mlx.scheduler import SchedulerConfig
    from vllm_mlx.service.helpers import get_model_max_context
    from vllm_mlx.utils.tokenizer import load_model_with_fallback

    from .runner import _reported_token_count, run_standardized_bench

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="mlx-step", initializer=_init_mlx_step_thread
    )
    try:
        model, tokenizer = executor.submit(load_model_with_fallback, repo_id).result()
        scheduler = SchedulerConfig(
            max_num_seqs=1,
            max_concurrent_requests=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            enable_prefix_cache=False,
            spec_decode="none",
        )
        config = EngineConfig(model_name=repo_id, scheduler_config=scheduler)
        async with AsyncEngineCore(
            model, tokenizer, config, executor=executor
        ) as engine:
            context_length = get_model_max_context(engine.engine)
            result = await run_standardized_bench(
                engine,
                tokenizer,
                sampling="greedy",
                registered_token_ids=True,
            )
    finally:
        # ThreadPoolExecutor workers are non-daemon and Python joins them again
        # during interpreter shutdown. Leaving this executor live can make the
        # CLI appear finished while its process (and Desktop memory lease)
        # remains stuck. AsyncEngineCore has exited at this point, so cancel
        # work that never started and synchronously reap the owned worker. The
        # Desktop's outer CLI process group remains the hard cancellation
        # boundary for native MLX calls that cannot be interrupted in-process.
        executor.shutdown(wait=True, cancel_futures=True)

    workload = registered_workload("text_generation")
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
    return measurements, context_length


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


def run_local(
    alias: str,
    *,
    archive: LocalRunArchive | None = None,
    inherit_process_group: bool = False,
) -> dict[str, Any]:
    """Run a registered protocol, validate it, and save it locally only."""

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
    context_length = None
    hardware = None
    software = None
    execution = None
    measurements_completed = False
    destination = archive or LocalRunArchive.default()
    try:
        hardware, software = collect()
        if task_type == "text_generation":
            measurements, context_length = asyncio.run(
                _text_measurements(model["repo_id"])
            )
        elif task_type == "image_generation":
            measurements = _run_image(
                alias, isolate_process_group=not inherit_process_group
            )
        elif task_type == "video_generation":
            measurements = _run_video(
                alias, isolate_process_group=not inherit_process_group
            )
        measurements_completed = True
        execution = execution_config(task_type, context_length=context_length)
        run = build_run(
            repo_id=model["repo_id"],
            task_type=task_type,
            hardware=hardware,
            software=software,
            started_at=started_at,
            measurements=measurements,
            context_length=context_length,
            execution=execution,
        )
    except Exception as exc:
        if measurements_completed and execution is None:
            raise LocalBenchmarkError(
                f"benchmark completed but result could not be constructed: {exc}",
                None,
                saved=False,
            ) from exc
        failure_code = (
            "machine_probe_failed"
            if hardware is None or software is None
            else _failure_code(exc)
        )
        try:
            if execution is None:
                execution = execution_config(task_type, context_length=context_length)
            failed = build_run(
                repo_id=model["repo_id"],
                task_type=task_type,
                hardware=hardware,
                software=software,
                started_at=started_at,
                status=(
                    "cancelled"
                    if isinstance(exc, BenchmarkCancelledError)
                    else "failed"
                ),
                failure_code=failure_code,
                context_length=context_length,
                execution=execution,
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


__all__ = ["LocalBenchmarkError", "run_local"]
