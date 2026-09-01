# SPDX-License-Identifier: Apache-2.0
"""Local-only executors for registered Community Benchmark protocols."""

from __future__ import annotations

import asyncio
import concurrent.futures
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
    """A failed attempt whose privacy-safe outcome was archived locally."""

    def __init__(self, message: str, run: dict[str, Any]):
        super().__init__(message)
        self.run = run


def _failure_code(error: Exception) -> str:
    message = str(error).lower()
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


def _peak_memory_mib(base_url: str) -> int:
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        response.raise_for_status()
        peak = response.json().get("metal", {}).get("peak_memory_gb")
        return max(0, round(float(peak) * 1024)) if peak is not None else 0
    except (requests.RequestException, TypeError, ValueError):
        return 0


def _run_image(alias: str) -> list[dict[str, Any]]:
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
    with serve(alias, boot_timeout_s=600) as server:
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
                    raise RuntimeError("image benchmark was cancelled")
                if len(result.get("data", [])) != case["image_count"]:
                    raise RuntimeError("image benchmark returned an incomplete batch")
                measurements.append(
                    {
                        "case_id": case["case_id"],
                        "round_index": index - case["warmup_rounds"] + 1,
                        "total_duration_ms": duration_ms,
                        "peak_active_memory_mib": _peak_memory_mib(server["base_url"]),
                        "completed": True,
                        "image_count": len(result.get("data", [])),
                        "width": case["width"],
                        "height": case["height"],
                    }
                )
    return measurements


def _run_video(alias: str) -> list[dict[str, Any]]:
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
    ) as server:
        started = time.perf_counter()
        response = requests.post(
            f"{server['base_url']}/videos", data=payload, timeout=30
        )
        response.raise_for_status()
        job = response.json()
        deadline = time.monotonic() + _VIDEO_JOB_TIMEOUT_S
        while job["status"] not in {"completed", "failed"}:
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
        if job["status"] != "completed":
            raise RuntimeError(
                job.get("error", {}).get("message", "video generation failed")
            )
        duration_ms = (time.perf_counter() - started) * 1000
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

    from .runner import run_standardized_bench

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
            result = await run_standardized_bench(engine, tokenizer, sampling="greedy")
    finally:
        executor.shutdown(wait=False)

    workload = registered_workload("text_generation")
    buckets = (result.short, result.long)
    measurements: list[dict[str, Any]] = []
    peak = result.peak_ram_mb or 0
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
                    "prompt_tokens": round_result.prompt_tokens
                    or case["target_prompt_tokens"],
                    "output_tokens": round_result.output_tokens
                    or case["target_output_tokens"],
                    "ttft_ms": round_result.ttft_ms,
                    "decode_duration_ms": decode_ms,
                }
            )
    return measurements, context_length


def run_local(alias: str, *, archive: LocalRunArchive | None = None) -> dict[str, Any]:
    """Run a registered protocol, validate it, and save it locally only."""

    plan = plan_for_alias(alias)
    model = plan["model"]
    hardware, software = collect()
    started_at = utc_now()
    task_type = model["task_type"]
    context_length = None
    destination = archive or LocalRunArchive.default()
    try:
        if task_type == "text_generation":
            measurements, context_length = asyncio.run(
                _text_measurements(model["repo_id"])
            )
        elif task_type == "image_generation":
            measurements = _run_image(alias)
        elif task_type == "video_generation":
            measurements = _run_video(alias)
        else:
            raise ValueError(f"unsupported benchmark task {task_type!r}")
    except Exception as exc:
        failed = build_run(
            repo_id=model["repo_id"],
            task_type=task_type,
            hardware=hardware,
            software=software,
            started_at=started_at,
            status="failed",
            failure_code=_failure_code(exc),
            context_length=context_length,
        )
        destination.save(failed)
        raise LocalBenchmarkError(str(exc), failed) from exc
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


__all__ = ["LocalBenchmarkError", "run_local"]
