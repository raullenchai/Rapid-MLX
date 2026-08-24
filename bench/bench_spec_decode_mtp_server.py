#!/usr/bin/env python3
"""End-to-end Rapid-MLX MTP benchmark against a running server.

This harness uses NVIDIA SPEED-Bench workloads but owns its Rapid-specific
transport, timing, metrics, and receipt logic.  Run baseline and MTP servers
separately, then compare their receipts; a single-server A/B would retain
controller and Metal state across arms and is therefore intentionally absent.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_REPO = "nvidia/SPEED-Bench"
DEFAULT_DATASET_REVISION = "487aa718444e816458d1a0a52bfce7a454285cf4"
MTP_METRIC_PREFIX = "rapid_mlx_spec_decode_"
MTP_CUMULATIVE_METRICS = {
    "rapid_mlx_spec_decode_accepts_total",
    "rapid_mlx_spec_decode_attempts_total",
    "rapid_mlx_spec_decode_k_chosen_rounds_total",
    "rapid_mlx_spec_decode_k_chosen_total",
    "rapid_mlx_spec_decode_park_total",
    "rapid_mlx_spec_decode_tokens_saved_total",
}
METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"])*)"')


@dataclass(frozen=True)
class Sample:
    sample_id: str
    category: str
    turns: tuple[str, ...]


@dataclass
class TurnResult:
    sample_id: str
    category: str
    turn_index: int
    ok: bool
    ttft_s: float | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    response_sha256: str | None
    error: str | None


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.endswith("/v1"):
        value = value[:-3]
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _unescape_label(value: str) -> str:
    return value.replace(r"\n", "\n").replace(r"\"", '"').replace(r"\\", "\\")


def parse_mtp_metrics(text: str) -> dict[str, float]:
    """Parse Rapid MTP samples into stable name+sorted-label keys."""
    result: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match or not match.group("name").startswith(MTP_METRIC_PREFIX):
            continue
        labels = {
            item.group("key"): _unescape_label(item.group("value"))
            for item in LABEL_RE.finditer(match.group("labels") or "")
        }
        suffix = ""
        if labels:
            suffix = (
                "{" + ",".join(f"{key}={labels[key]}" for key in sorted(labels)) + "}"
            )
        result[f"{match.group('name')}{suffix}"] = float(match.group("value"))
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(set(before) | set(after))
        if after.get(key, 0.0) != before.get(key, 0.0)
    }


def metric_total(metrics: dict[str, float], name: str) -> float:
    prefix = f"{name}{{"
    return sum(
        value for key, value in metrics.items() if key == name or key.startswith(prefix)
    )


def derived_mtp_metrics(counter_delta: dict[str, float]) -> dict[str, Any]:
    attempts = metric_total(counter_delta, "rapid_mlx_spec_decode_attempts_total")
    accepts = metric_total(counter_delta, "rapid_mlx_spec_decode_accepts_total")
    saved = metric_total(counter_delta, "rapid_mlx_spec_decode_tokens_saved_total")
    rounds = metric_total(counter_delta, "rapid_mlx_spec_decode_k_chosen_rounds_total")
    k_counts: dict[str, float] = {}
    for key, value in counter_delta.items():
        if not key.startswith("rapid_mlx_spec_decode_k_chosen_total{"):
            continue
        match = re.search(r"(?:\{|,)k=([^,}]+)", key)
        if match:
            k_counts[match.group(1)] = value
    return {
        "accept_ratio": accepts / attempts if attempts else None,
        "tokens_saved_per_attempt": saved / attempts if attempts else None,
        "k_counts": k_counts,
        "k_shares": {key: value / rounds for key, value in k_counts.items()}
        if rounds
        else {},
    }


def split_metric_observations(
    before: dict[str, float], after: dict[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    deltas = metric_delta(before, after)
    cumulative_delta = {
        key: value
        for key, value in deltas.items()
        if key.split("{", 1)[0] in MTP_CUMULATIVE_METRICS
    }
    gauges_after = {
        key: value
        for key, value in after.items()
        if key.split("{", 1)[0] not in MTP_CUMULATIVE_METRICS
    }
    return cumulative_delta, gauges_after


def extract_turns(row: dict[str, Any]) -> tuple[str, ...]:
    turns = row.get("turns")
    if not isinstance(turns, list):
        raise ValueError("missing turns list")
    clean = tuple(str(turn).strip() for turn in turns if str(turn).strip())
    if not clean:
        raise ValueError("empty turns list")
    return clean


def load_samples(
    bench: str,
    category: str | None,
    limit: int | None,
    sample_ids: set[str],
    dataset_revision: str = DEFAULT_DATASET_REVISION,
) -> list[Sample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "SPEED-Bench loading requires the optional 'datasets' package"
        ) from exc

    dataset = load_dataset(
        DATASET_REPO, name=bench, split="test", revision=dataset_revision
    )
    samples: list[Sample] = []
    for index, raw in enumerate(dataset):
        row = dict(raw)
        row_category = str(row.get("category", "unknown"))
        sample_id = str(row.get("id", row.get("question_id", index)))
        if category and row_category != category:
            continue
        if sample_ids and sample_id not in sample_ids:
            continue
        try:
            turns = extract_turns(row)
        except ValueError:
            continue
        samples.append(Sample(sample_id, row_category, turns))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError("no SPEED-Bench samples matched the requested filters")
    return samples


def _content_from_delta(delta: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("content", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str):
            parts.append(value)
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, sort_keys=True, separators=(",", ":")))
    return "".join(parts)


def stream_turn(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    sampling: dict[str, Any],
    timeout: float,
) -> tuple[str, TurnResult]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sampling,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_output: float | None = None
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or {}
            prompt_tokens = int(
                usage.get("prompt_tokens", prompt_tokens) or prompt_tokens
            )
            completion_tokens = int(
                usage.get("completion_tokens", completion_tokens) or completion_tokens
            )
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = _content_from_delta(delta)
                if content:
                    if first_output is None:
                        first_output = time.perf_counter()
                    chunks.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
    ended = time.perf_counter()
    text = "".join(chunks)
    result = TurnResult(
        sample_id="",
        category="",
        turn_index=-1,
        ok=True,
        ttft_s=None if first_output is None else first_output - started,
        latency_s=ended - started,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        response_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        error=None,
    )
    return text, result


def run_sample(
    sample: Sample,
    endpoint: str,
    model: str,
    max_tokens: int,
    sampling: dict[str, Any],
    timeout: float,
) -> list[TurnResult]:
    messages: list[dict[str, str]] = []
    results: list[TurnResult] = []
    for turn_index, prompt in enumerate(sample.turns):
        messages.append({"role": "user", "content": prompt})
        try:
            answer, result = stream_turn(
                endpoint, model, messages, max_tokens, sampling, timeout
            )
            result.sample_id = sample.sample_id
            result.category = sample.category
            result.turn_index = turn_index
            results.append(result)
            messages.append({"role": "assistant", "content": answer})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            results.append(
                TurnResult(
                    sample.sample_id,
                    sample.category,
                    turn_index,
                    False,
                    None,
                    0.0,
                    0,
                    0,
                    None,
                    None,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            break
    return results


def run_samples(
    samples: list[Sample],
    endpoint: str,
    model: str,
    max_tokens: int,
    sampling: dict[str, Any],
    timeout: float,
    concurrency: int,
) -> tuple[list[TurnResult], float]:
    results: list[TurnResult] = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_sample,
                sample,
                endpoint,
                model,
                max_tokens,
                sampling,
                timeout,
            )
            for sample in samples
        ]
        for future in futures:
            results.extend(future.result())
    return results, time.perf_counter() - started


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    results: list[TurnResult], wall_time_s: float | None = None
) -> dict[str, Any]:
    ok = [result for result in results if result.ok]
    ttfts = [result.ttft_s for result in ok if result.ttft_s is not None]
    latencies = [result.latency_s for result in ok]
    completion_tokens = sum(result.completion_tokens for result in ok)
    total_latency = sum(result.latency_s for result in ok)
    return {
        "requests": len(results),
        "successful_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "prompt_tokens": sum(result.prompt_tokens for result in ok),
        "completion_tokens": completion_tokens,
        "pooled_completion_tokens_per_s": (
            completion_tokens / total_latency if total_latency > 0 else None
        ),
        "aggregate_completion_tokens_per_s": (
            completion_tokens / wall_time_s if wall_time_s and wall_time_s > 0 else None
        ),
        "ttft_s": {
            "median": statistics.median(ttfts) if ttfts else None,
            "p95": percentile(ttfts, 0.95),
        },
        "latency_s": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
        },
    }


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": sys.version,
    }
    if sys.platform == "darwin":
        for key, sysctl_name in (
            ("cpu_brand", "machdep.cpu.brand_string"),
            ("memory_bytes", "hw.memsize"),
        ):
            result = subprocess.run(
                ["sysctl", "-n", sysctl_name], capture_output=True, text=True
            )
            if result.returncode == 0:
                value = result.stdout.strip()
                info[key] = int(value) if key == "memory_bytes" else value
    return info


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)
    digest = hashlib.sha256(encoded).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    base_url = normalize_base_url(args.base_url)
    endpoint = f"{base_url}/v1/chat/completions"
    models = fetch_json(f"{base_url}/v1/models", args.timeout)
    available = [str(item.get("id")) for item in models.get("data", [])]
    model = args.model or (available[0] if available else None)
    if not model:
        raise RuntimeError("server returned no model ID; pass --model")

    samples = load_samples(
        args.bench,
        args.category,
        args.limit,
        set(args.sample_id or []),
        args.dataset_revision,
    )
    if args.warmup_samples < 0 or args.warmup_samples > len(samples):
        raise RuntimeError("--warmup-samples must be between 0 and the sample count")
    if args.concurrency < 1:
        raise RuntimeError("--concurrency must be at least 1")
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    warmup_results: list[TurnResult] = []
    warmup_wall_time = 0.0
    if args.warmup_samples:
        warmup_results, warmup_wall_time = run_samples(
            samples[: args.warmup_samples],
            endpoint,
            model,
            args.max_tokens,
            sampling,
            args.timeout,
            min(args.concurrency, args.warmup_samples),
        )
    before = parse_mtp_metrics(fetch_text(f"{base_url}/metrics", args.timeout))
    results, wall_time = run_samples(
        samples,
        endpoint,
        model,
        args.max_tokens,
        sampling,
        args.timeout,
        args.concurrency,
    )
    after = parse_mtp_metrics(fetch_text(f"{base_url}/metrics", args.timeout))
    counter_delta, gauges_after = split_metric_observations(before, after)
    attempts = metric_total(counter_delta, "rapid_mlx_spec_decode_attempts_total")
    rounds = metric_total(counter_delta, "rapid_mlx_spec_decode_k_chosen_rounds_total")

    activity_passed = attempts > 0
    fallback_passed = attempts == 0
    validity = {
        "request_success": all(result.ok for result in results),
        "warmup_success": all(result.ok for result in warmup_results),
        "mtp_activity": {
            "required": args.require_mtp_activity,
            "passed": activity_passed,
            "attempts_delta": attempts,
        },
        "mtp_fallback": {
            "required": args.expect_mtp_fallback,
            "passed": fallback_passed,
            "reason": "Rapid currently routes generation batches larger than one to AR",
        },
    }
    passed = validity["request_success"] and validity["warmup_success"]
    if args.require_mtp_activity:
        passed = passed and activity_passed
    if args.expect_mtp_fallback:
        passed = passed and fallback_passed
    validity["passed"] = passed

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "rapid_git_sha": git_sha(),
        "benchmark_sha256": file_sha256(Path(__file__)),
        "server": {
            "base_url": base_url,
            "label": args.server_label,
            "model": model,
            "target_revision": args.target_revision,
            "available_models": available,
            "sidecar": args.sidecar,
        },
        "workload": {
            "dataset": DATASET_REPO,
            "dataset_revision": args.dataset_revision,
            "bench": args.bench,
            "category": args.category,
            "sample_ids": [sample.sample_id for sample in samples],
            "turns": sum(len(sample.turns) for sample in samples),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "warmup_samples": args.warmup_samples,
            "sampling": sampling,
        },
        "environment": hardware_info(),
        "wall_time_s": wall_time,
        "warmup": {
            "wall_time_s": warmup_wall_time,
            "summary": summarize(warmup_results, warmup_wall_time),
        },
        "summary": summarize(results, wall_time),
        "metrics": {
            "before": before,
            "after": after,
            "cumulative_delta": counter_delta,
            "gauges_after": gauges_after,
            "rounds_delta": rounds,
            "derived": derived_mtp_metrics(counter_delta),
        },
        "validity": validity,
        "requests": [asdict(result) for result in results],
    }
    write_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": receipt["summary"],
                "validity": validity,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _compare(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    mtp = json.loads(args.mtp.read_text(encoding="utf-8"))
    for field in (
        "dataset",
        "dataset_revision",
        "bench",
        "category",
        "sample_ids",
        "turns",
        "max_tokens",
        "concurrency",
        "warmup_samples",
        "sampling",
    ):
        if baseline["workload"].get(field) != mtp["workload"].get(field):
            raise RuntimeError(f"receipts differ in workload.{field}")
    if not baseline["validity"]["passed"] or not mtp["validity"]["passed"]:
        raise RuntimeError("both receipts must pass their declared validity gates")
    b_summary = baseline["summary"]
    m_summary = mtp["summary"]
    comparison = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline": str(args.baseline),
        "mtp": str(args.mtp),
        "workload": baseline["workload"],
        "ratios": {
            "aggregate_completion_tokens_per_s": _ratio(
                m_summary["aggregate_completion_tokens_per_s"],
                b_summary["aggregate_completion_tokens_per_s"],
            ),
            "pooled_completion_tokens_per_s": _ratio(
                m_summary["pooled_completion_tokens_per_s"],
                b_summary["pooled_completion_tokens_per_s"],
            ),
            "median_ttft": _ratio(
                m_summary["ttft_s"]["median"], b_summary["ttft_s"]["median"]
            ),
            "median_latency": _ratio(
                m_summary["latency_s"]["median"],
                b_summary["latency_s"]["median"],
            ),
        },
        "baseline_summary": b_summary,
        "mtp_summary": m_summary,
        "mtp_cumulative_metric_delta": mtp["metrics"]["cumulative_delta"],
        "mtp_gauges_after": mtp["metrics"]["gauges_after"],
    }
    write_receipt(args.output, comparison)
    print(json.dumps(comparison, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="capture one server arm")
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--model")
    run.add_argument("--target-revision", required=True)
    run.add_argument("--server-label", required=True)
    run.add_argument(
        "--sidecar", help="immutable MTP sidecar ID/revision or 'none'", required=True
    )
    run.add_argument("--bench", default="qualitative")
    run.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    run.add_argument("--category")
    run.add_argument("--sample-id", action="append")
    run.add_argument("--limit", type=int, default=8)
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument(
        "--warmup-samples",
        type=int,
        default=1,
        help="discard this many samples before capturing metrics/results",
    )
    run.add_argument("--temperature", type=float, default=0.6)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--timeout", type=float, default=600.0)
    activity = run.add_mutually_exclusive_group()
    activity.add_argument("--require-mtp-activity", action="store_true")
    activity.add_argument("--expect-mtp-fallback", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=_run)

    compare = subparsers.add_parser("compare", help="compare compatible receipts")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--mtp", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=_compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (RuntimeError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
