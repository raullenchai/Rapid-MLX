# SPDX-License-Identifier: Apache-2.0
"""Local-only executors for registered Community Benchmark protocols."""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import io
import math
import tempfile
import time
from typing import Any

import requests

from .benchmark_contracts import public_prompt, registered_workload
from .hardware import collect
from .run_builder import build_run, utc_now
from .workspace import LocalRunArchive, plan_for_alias

_VIDEO_JOB_TIMEOUT_S = 3600.0
_VIDEO_POLL_INTERVAL_S = 1.0


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
    if isinstance(error, BenchmarkCancelledError) or "cancelled" in message:
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


def _probe_video_artifact(path: str) -> tuple[int, int, int, float]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - video extra owns this dependency
        raise RuntimeError(
            "video artifact validation requires rapid-mlx[video]"
        ) from exc

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


def _validated_video_artifact(
    base_url: str,
    job_id: str,
    *,
    width: int,
    height: int,
    frames: int,
    fps: float,
) -> None:
    response = requests.get(
        f"{base_url}/videos/{job_id}/content",
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(prefix="rapid-benchmark-", suffix=".mp4") as file:
        size_bytes = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)
            size_bytes += len(chunk)
        if size_bytes == 0:
            raise RuntimeError("video benchmark returned an empty MP4 artifact")
        file.flush()
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
            response.raise_for_status()
            result = response.json()
            duration_ms = (time.perf_counter() - started) * 1000
            if index >= case["warmup_rounds"]:
                if result.get("cancelled", False):
                    raise BenchmarkCancelledError("image benchmark was cancelled")
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
        response.raise_for_status()
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
            response.raise_for_status()
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
            context_length = get_model_max_context(engine)
            result = await run_standardized_bench(
                engine,
                tokenizer,
                sampling="greedy",
                registered_token_ids=True,
            )
    finally:
        executor.shutdown(wait=False)

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


def run_local(
    alias: str,
    *,
    archive: LocalRunArchive | None = None,
    inherit_process_group: bool = False,
) -> dict[str, Any]:
    """Run a registered protocol, validate it, and save it locally only."""

    try:
        plan = plan_for_alias(alias)
    except Exception as exc:
        raise LocalBenchmarkError(str(exc), None, saved=False) from exc
    model = plan["model"]
    started_at = utc_now()
    task_type = model["task_type"]
    context_length = None
    hardware = None
    software = None
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
        else:
            raise ValueError(f"unsupported benchmark task {task_type!r}")
        run = build_run(
            repo_id=model["repo_id"],
            task_type=task_type,
            hardware=hardware,
            software=software,
            started_at=started_at,
            measurements=measurements,
            context_length=context_length,
        )
        destination.save(run)
        return run
    except Exception as exc:
        failure_code = (
            "machine_probe_failed"
            if hardware is None or software is None
            else _failure_code(exc)
        )
        failed = build_run(
            repo_id=model["repo_id"],
            task_type=task_type,
            hardware=hardware,
            software=software,
            started_at=started_at,
            status=(
                "cancelled" if isinstance(exc, BenchmarkCancelledError) else "failed"
            ),
            failure_code=failure_code,
            context_length=context_length,
        )
        try:
            destination.save(failed)
        except Exception as archive_exc:
            raise LocalBenchmarkError(
                f"{exc}; failed outcome could not be saved: {archive_exc}",
                failed,
                saved=False,
            ) from exc
        raise LocalBenchmarkError(str(exc), failed, saved=True) from exc


__all__ = ["LocalBenchmarkError", "run_local"]
