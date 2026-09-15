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
import json
import os
import platform
import re
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
STEP_RE = re.compile(r"STAGE:(\d+):STEP:(\d+):(\d+):")
SWAP_RE = re.compile(r"used = ([0-9.]+)([MG])")
PHASE_RE = re.compile(r"^\[([^]]+)] (\.\.\.|done in ([0-9.]+)s)$")
DEFAULT_PROMPT = (
    "A red fox trots across fresh snow at golden hour, cinematic tracking shot"
)


@dataclass
class Sample:
    elapsed_s: float
    rss_bytes: int
    swap_used_bytes: int | None


def _emit(kind: str, **fields: Any) -> None:
    print(
        EVENT_PREFIX
        + json.dumps({"kind": kind, "time_ns": time.monotonic_ns(), **fields}),
        flush=True,
    )


def _swap_used_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True, timeout=3
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

    from vllm_mlx.runtime.video_lane import VideoEngine

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
    except BaseException as exc:
        _emit("error", error_type=type(exc).__name__, message=str(exc))
        return 1
    _emit(
        "complete",
        generation_s=(time.monotonic_ns() - started) / 1e9,
        mlx_peak_bytes=mx.get_peak_memory(),
        output_bytes=output.stat().st_size,
    )
    return 0


def _worker_ltx25(args: argparse.Namespace) -> int:
    """Run the pinned CLI while adding machine-readable per-step events."""
    import tqdm as tqdm_module

    original_tqdm = tqdm_module.tqdm
    stage_counter = 0

    def measured_tqdm(iterable=None, *pargs, **kwargs):
        nonlocal stage_counter
        stage_counter += 1
        stage_number = stage_counter
        wrapped = original_tqdm(iterable, *pargs, **kwargs)
        description = str(kwargs.get("desc", "step"))

        def iterator():
            for index, item in enumerate(wrapped, start=1):
                _emit(
                    "step_start",
                    stage=stage_number,
                    description=description,
                    step=index,
                    total=wrapped.total,
                )
                yield item

        return iterator()

    tqdm_module.tqdm = measured_tqdm
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
    except BaseException as exc:
        _emit("error", error_type=type(exc).__name__, message=str(exc))
        return 1
    finally:
        sys.argv = old_argv
    _emit(
        "complete",
        generation_s=(time.monotonic_ns() - started) / 1e9,
        mlx_peak_bytes=mx.get_peak_memory(),
        output_bytes=output.stat().st_size if output.is_file() else 0,
    )
    return 0


def _worker(args: argparse.Namespace) -> int:
    return _worker_ltx25(args) if args.runtime == "ltx25" else _worker_mlx(args)


def _read_stream(
    stream, name: str, started_ns: int, events: list[dict[str, Any]], lines: list[str]
) -> None:
    for raw in iter(stream.readline, ""):
        elapsed = (time.monotonic_ns() - started_ns) / 1e9
        line = raw.rstrip("\r\n")
        lines.append(f"{elapsed:.6f} {name} {line}")
        if line.startswith(EVENT_PREFIX):
            try:
                event = json.loads(line[len(EVENT_PREFIX) :])
                event["observed_elapsed_s"] = elapsed
                events.append(event)
            except json.JSONDecodeError:
                pass
            continue
        match = STEP_RE.search(line)
        if match:
            events.append(
                {
                    "kind": "step_start",
                    "stage": int(match.group(1)),
                    "step": int(match.group(2)),
                    "total": int(match.group(3)),
                    "observed_elapsed_s": elapsed,
                }
            )
            continue
        if "TRANSFORMER:EVAL_COMPLETE" in line:
            events.append({"kind": "weights_ready", "observed_elapsed_s": elapsed})
            continue
        phase = PHASE_RE.match(line)
        if phase:
            phase_kind = "phase_start" if phase.group(2) == "..." else "phase_end"
            events.append(
                {
                    "kind": phase_kind,
                    "phase": phase.group(1),
                    "reported_duration_s": (
                        float(phase.group(3)) if phase.group(3) is not None else None
                    ),
                    "observed_elapsed_s": elapsed,
                }
            )
            if phase_kind == "phase_end" and phase.group(1).startswith(
                "Loading transformer"
            ):
                events.append({"kind": "weights_ready", "observed_elapsed_s": elapsed})


def _run_once(args: argparse.Namespace, run_index: int) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.runtime}-{args.frames}f-run{run_index}"
    output_video = output_dir / f"{stem}.mp4"
    interpreter = args.ltx25_python if args.runtime == "ltx25" else sys.executable
    command = [
        interpreter,
        str(Path(__file__).resolve()),
        "--worker",
        "--runtime",
        args.runtime,
        "--model",
        args.model,
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
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started_ns = time.monotonic_ns()
    swap_start = _swap_used_bytes()
    thermal_start = _thermal_observation()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
        start_new_session=True,
    )
    events: list[dict[str, Any]] = []
    lines: list[str] = []
    threads = [
        threading.Thread(
            target=_read_stream,
            args=(stream, name, started_ns, events, lines),
            daemon=True,
        )
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
    ]
    for thread in threads:
        thread.start()
    samples: list[Sample] = []
    first_artifact_s = None
    while process.poll() is None:
        now_s = (time.monotonic_ns() - started_ns) / 1e9
        samples.append(Sample(now_s, _tree_rss(process.pid), _swap_used_bytes()))
        if (
            first_artifact_s is None
            and output_video.is_file()
            and output_video.stat().st_size
        ):
            first_artifact_s = now_s
        time.sleep(args.sample_interval)
    for thread in threads:
        thread.join(timeout=5)
    elapsed_s = (time.monotonic_ns() - started_ns) / 1e9
    swap_end = _swap_used_bytes()
    thermal_end = _thermal_observation()
    step_events = sorted(
        (event for event in events if event.get("kind") == "step_start"),
        key=lambda event: event["observed_elapsed_s"],
    )
    for current, following in zip(step_events, step_events[1:]):
        current["duration_to_next_step_s"] = (
            following["observed_elapsed_s"] - current["observed_elapsed_s"]
        )
    stage_durations: dict[str, list[float]] = {}
    for current, following in zip(step_events, step_events[1:]):
        if current.get("stage") == following.get("stage"):
            stage_durations.setdefault(str(current["stage"]), []).append(
                current["duration_to_next_step_s"]
            )
    durations = [duration for values in stage_durations.values() for duration in values]
    complete = next(
        (event for event in events if event.get("kind") == "complete"), None
    )
    if step_events and complete:
        last_duration = (
            complete["observed_elapsed_s"] - step_events[-1]["observed_elapsed_s"]
        )
        step_events[-1]["duration_to_complete_s"] = last_duration
    first_step_s = step_events[0]["observed_elapsed_s"] if step_events else None
    weights_ready = next(
        (event for event in events if event.get("kind") == "weights_ready"), None
    )
    swap_growth = (
        swap_end - swap_start
        if swap_start is not None and swap_end is not None
        else None
    )
    result = {
        "run": run_index,
        "status": "completed" if process.returncode == 0 and complete else "failed",
        "returncode": process.returncode,
        "total_s": elapsed_s,
        "cold_start_to_first_step_s": first_step_s,
        "cold_start_to_weights_ready_s": (
            weights_ready["observed_elapsed_s"] if weights_ready else None
        ),
        "artifact_first_byte_s": first_artifact_s,
        "step_median_s": statistics.median(durations) if durations else None,
        "stage_step_median_s": {
            stage: statistics.median(values)
            for stage, values in stage_durations.items()
        },
        "steps": step_events,
        "peak_tree_rss_bytes": max((sample.rss_bytes for sample in samples), default=0),
        "mlx_peak_bytes": complete.get("mlx_peak_bytes") if complete else None,
        "swap_start_bytes": swap_start,
        "swap_end_bytes": swap_end,
        "swap_growth_bytes": swap_growth,
        "swap_limited": bool(swap_growth is not None and swap_growth > 64 * 1024**2),
        "thermal_start": thermal_start,
        "thermal_end": thermal_end,
        "output": str(output_video) if output_video.is_file() else None,
        "output_bytes": output_video.stat().st_size if output_video.is_file() else 0,
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
        f"- Model revision: {document['config']['model_revision']}",
        f"- Runtime: {document['config']['runtime']} @ {document['config']['runtime_revision']}",
        f"- Workload: {document['config']['width']}x{document['config']['height']}, "
        f"{document['config']['frames']} frames at {document['config']['fps']} fps, "
        f"seed={document['config']['seed']}",
        f"- Prompt: `{document['config']['prompt']}`",
        f"- Runs: {len(completed)}/{len(results)} completed; "
        f"{len(comparable)} comparable without swap growth",
        f"- Median cold start to weights ready: {_seconds(_median(comparable, 'cold_start_to_weights_ready_s'))}",
        f"- Median cold start to first diffusion step: {_seconds(_median(comparable, 'cold_start_to_first_step_s'))}",
        f"- Median diffusion step: {_seconds(_median(comparable, 'step_median_s'))}",
        *stage_lines,
        f"- Median total: {_seconds(_median(comparable, 'total_s'))}",
        f"- Peak process-tree RSS: {_gib(max((r['peak_tree_rss_bytes'] for r in results), default=0))}",
        f"- Peak MLX Metal allocator: {_gib(max((r['mlx_peak_bytes'] or 0 for r in results), default=0))}",
        f"- Maximum swap growth: {_gib(max((r['swap_growth_bytes'] or 0 for r in results), default=0))}",
        "",
        "`artifact_first_byte_s` is the first observable MP4 byte, not a decoded-frame callback; "
        "the current runtimes do not expose the latter.",
    ]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="unknown")
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
    parser.add_argument("--output-dir", default="/private/tmp/LTX-benchmark/results")
    parser.add_argument("--ltx25-python", default="")
    parser.add_argument("--runtime-revision", default="unknown")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-video", default="", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.width, args.height = (
            int(part) for part in args.size.lower().split("x", 1)
        )
    except ValueError as exc:
        raise SystemExit("--size must use WIDTHxHEIGHT") from exc
    if args.frames is None:
        if args.seconds is None:
            raise SystemExit("provide --frames or --seconds")
        args.frames = _frames_for_seconds(args.seconds, args.fps)
    if args.frames < 9 or args.frames % 8 != 1:
        raise SystemExit("LTX frame count must be 8n+1 and at least 9")
    if args.runtime == "ltx25" and not args.worker and not args.ltx25_python:
        raise SystemExit("--ltx25-python is required for the pinned research runtime")
    if args.worker:
        return _worker(args)

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
            "model_revision": args.model_revision,
            "runtime": args.runtime,
            "runtime_revision": args.runtime_revision,
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
        },
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"summary-{args.runtime}-{args.frames}f"
    (output_dir / f"{stem}.json").write_text(json.dumps(document, indent=2) + "\n")
    markdown = _markdown(document)
    (output_dir / f"{stem}.md").write_text(markdown)
    print(markdown, end="")
    return 0 if all(result["status"] == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
