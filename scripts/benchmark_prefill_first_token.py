#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated HTTP experiment for prompt interference before the first token.

The candidate is installed only in this script's owned child process. Serving
code/defaults are unchanged. Uses private mlx-lm 0.31.3 prompt metadata; results
are research evidence, not a portable scheduler implementation.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import socket
import statistics as st
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

try:
    from scripts.bench_metadata import add_bench_metadata
except ModuleNotFoundError:
    from bench_metadata import add_bench_metadata

ARMS = (
    "fixed2048",
    "fixed512",
    "adaptive2048",
    "adaptive2048-1000ms",
    "adaptive2048-tail",
)


def first_token_chunk(
    selected: int,
    processing: Sequence,
    queued: int,
    slots: int,
    samples: Sequence[float],
    target_ms: float,
) -> int:
    """Bound work only when at least two prompt rows can share the next tick.

    Retains the configured checkpoint grid for every current row. The existing
    memory and decode bounds are applied first; this candidate can only shrink.
    A completed prompt awaiting promotion still needs protection: its first
    token won't be emitted until a subsequent generation step returns.
    """
    rows = len(processing) + min(queued, max(0, slots - len(processing)))
    if rows < 2 or target_ms <= 0:
        return selected
    if samples:
        budget = target_ms / 1000 / max(samples) / rows
    else:
        budget = 256 / rows  # Same total cold-start row-token allowance as decode.
    bounded = min(selected, max(64, int(min(selected, budget)) // 32 * 32))
    for row in processing:
        bounded = min(bounded, selected - int(row[1]) % selected)
    return bounded


def segment_tail_chunk(
    bound: int, processing: Sequence, queued: Sequence, slots: int
) -> int:
    """Avoid padding a finishing prompt segment to its long neighbour's chunk.

    Only shrink the existing cost/grid bound. A final single token is promoted
    rather than prefetched, so it must not force the remaining long row to 1.
    Queued rows count only if they can actually join this tick.
    """
    admitted = list(islice(queued, max(0, slots - len(processing))))
    if len(processing) + len(admitted) < 2:
        return bound
    segments = [row[0] for row in processing] + [row[1] for row in admitted]
    for parts in segments:
        if not parts or not parts[0] or (len(parts) == 1 and len(parts[0]) == 1):
            continue
        bound = min(bound, len(parts[0]))
    return bound


def serve_child(args):
    from rapid_mlx.cli import cli_entrypoint
    from rapid_mlx.scheduler import Scheduler

    apply_original = Scheduler._apply_adaptive_prefill_size
    record_original = Scheduler._record_prefill_cost
    trace = args.trace.open("w", buffering=1)

    def apply(self):
        started = time.perf_counter_ns()
        selected = apply_original(self)
        bg = self.batch_generator
        processing = bg._currently_processing
        queued = len(bg._unprocessed_sequences)
        decoding = len(bg._generation_batch)
        slots = min(bg.prefill_batch_size, bg.completion_batch_size - decoding)
        samples = list(self._prefill_cost_samples)
        dispatched = selected
        if args.arm.startswith("adaptive2048"):
            dispatched = first_token_chunk(
                selected, processing, queued, slots, samples, args.target_ms
            )
            if args.arm == "adaptive2048-tail":
                dispatched = segment_tail_chunk(
                    dispatched, processing, bg._unprocessed_sequences, slots
                )
            bg.prefill_step_size = dispatched
            bg._prompt_batch.prefill_step_size = dispatched
        self._first_token_probe_trace = dict(
            started_ns=started,
            selected=selected,
            dispatched=dispatched,
            processing=[
                {
                    "uid": uid,
                    "processed": row[1],
                    "total": row[2],
                    "segments": [len(seg) for seg in row[0]],
                }
                for uid, row in zip(bg._prompt_batch.uids, processing)
            ],
            queued=queued,
            decoding=decoding,
            samples=samples,
            target_ms=args.target_ms,
        )
        return dispatched

    def record(self, before):
        record_original(self, before)
        row = self._first_token_probe_trace
        row["ended_ns"] = time.perf_counter_ns()
        trace.write(json.dumps(row) + "\n")

    Scheduler._apply_adaptive_prefill_size = apply
    Scheduler._record_prefill_cost = record
    sys.argv = [
        sys.argv[0],
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "prefill-probe",
        "--no-mllm",
        "--no-spec-decode",
        "--disable-disk-caches",
        "--disable-model-downloads",
        "--prefill-step-size",
        "512" if args.arm == "fixed512" else "2048",
        "--prefill-batch-size",
        "8",
        "--log-level",
        "INFO",
    ]
    cli_entrypoint()


def receipt(lines, started_ns):
    first = None
    content, reasoning, usage, finish, done = [], [], {}, None, False
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        obj = json.loads(payload)
        if obj.get("error"):
            raise ValueError(obj["error"])
        usage = obj.get("usage") or usage
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            c = delta.get("content") or ""
            r = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (c or r) and first is None:
                first = time.perf_counter_ns()
            content.append(c)
            reasoning.append(r)
            finish = choice.get("finish_reason") or finish
    if not done or first is None or finish not in ("stop", "length"):
        raise ValueError("incomplete/error stream")
    if any(
        type(usage.get(k)) is not int or usage[k] <= 0
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        raise ValueError("invalid usage")
    text = "".join(content)
    return dict(
        started_ns=started_ns,
        first_ns=first,
        ended_ns=time.perf_counter_ns(),
        ttft_ms=(first - started_ns) / 1e6,
        output_text=text,
        reasoning="".join(reasoning),
        usage=usage,
        finish_reason=finish,
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        done=done,
    )


def validate_cold(row, expected=None):
    # The owned server omits prompt_tokens_details when cached_tokens == 0.
    details = row["usage"].get("prompt_tokens_details")
    cached = 0 if details is None else details.get("cached_tokens")
    if type(cached) is not int or cached != 0:
        raise ValueError(f"unexpected cache reuse: {cached}")
    if expected is not None:
        if row["output_text"].strip() != expected or row["finish_reason"] != "stop":
            raise ValueError(f"incorrect exact answer: {row['output_text']!r}")
        if row["reasoning"]:
            raise ValueError("unexpected reasoning with thinking disabled")


def validate_overlap(long, short):
    """A fast completed long request must not masquerade as prefill contention."""
    if not long["started_ns"] <= short["started_ns"] < long["first_ns"]:
        raise ValueError("short request did not arrive before long first token")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Existing local snapshot")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18348)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--cycles", type=int, default=2, help="Alternates forward/reverse arm order"
    )
    ap.add_argument("--target-ms", type=float, default=500)
    ap.add_argument(
        "--workloads",
        nargs="+",
        choices=["prose", "exact", "retrieval"],
        default=["prose", "retrieval"],
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS[:3]),
        help="Experimental arms; the 1000ms arm overrides --target-ms",
    )
    ap.add_argument("--trace", type=Path)
    args = ap.parse_args()
    if not args.model.is_dir():
        ap.error("model must already exist locally")
    if args.child:
        if not args.arm or not args.trace:
            ap.error("child requires arm and trace")
        return serve_child(args)
    if (
        args.repeat < 1
        or args.cycles < 1
        or not math.isfinite(args.target_ms)
        or args.target_ms <= 0
    ):
        ap.error("repeat, cycles and target must be positive")
    if len(set(args.arms)) != len(args.arms) or len(set(args.workloads)) != len(
        args.workloads
    ):
        ap.error("arms and workloads must not contain duplicates")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        ap.error("output already exists")
    for cycle in range(args.cycles):
        for arm in args.arms:
            for suffix in ("trace.jsonl", "server.log"):
                if (args.output.parent / f"cycle{cycle}-{arm}.{suffix}").exists():
                    ap.error(
                        "campaign trace/log already exists; use a new output directory"
                    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    def prompt(size, salt, workload):
        prefix = f"Independent session {salt}. Inspect these code records.\n"
        text = prefix + "".join(
            f"def record_{i:05d}(x): return (x * {i * 13 + 7}) % {i * 3 + 101}\n"
            for i in range(size)
        )
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if len(tokenizer.encode(text[:mid])) <= size:
                low = mid
            else:
                high = mid - 1
        suffix = (
            "\nIgnore the records. What is 23 + 19? Reply with only the number."
            if workload == "exact"
            else "\nExplain the purpose of these records in about 150 words."
        )
        body = text[:low]
        if workload == "retrieval":
            marker = "7319" if salt.startswith("long-") else "9283"
            cut = body.rfind("\n", 0, len(body) // 2) + 1
            body = body[:cut] + f"Archive verification code: {marker}.\n" + body[cut:]
            suffix = "\nWhat is the archive verification code? Reply with only its four digits."
        return [{"role": "user", "content": body + suffix}]

    base = f"http://127.0.0.1:{args.port}"
    result = dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version,
        packages={
            k: metadata.version(k)
            for k in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "numpy", "httpx")
        },
        model_revision=args.model.name,
        repeat=args.repeat,
        cycles=args.cycles,
        target_ms=args.target_ms,
        workloads=args.workloads,
        arms=[],
        rows=[],
    )

    def save():
        args.output.write_text(
            json.dumps(add_bench_metadata(result, __file__), indent=2) + "\n"
        )

    with httpx.Client(timeout=240, trust_env=False) as control:

        def status():
            r = control.get(base + "/v1/status")
            r.raise_for_status()
            return r.json()

        def clear():
            control.post(base + "/v1/cache/clear").raise_for_status()
            s = status()
            if s.get("num_running") or s.get("num_waiting"):
                raise ValueError("server not idle at clear")

        def ask(msg, budget):
            with httpx.Client(timeout=240, trust_env=False) as client:
                started = time.perf_counter_ns()
                with client.stream(
                    "POST",
                    base + "/v1/chat/completions",
                    json=dict(
                        model="prefill-probe",
                        messages=msg,
                        max_tokens=budget,
                        temperature=0,
                        stream=True,
                        stream_options={"include_usage": True},
                        chat_template_kwargs={"enable_thinking": False},
                        enable_thinking=False,
                    ),
                ) as response:
                    response.raise_for_status()
                    row = receipt(response.iter_lines(), started)
                if not 1 <= row["usage"]["completion_tokens"] <= budget:
                    raise ValueError("completion exceeded request budget")
                row["total_ms"] = (row["ended_ns"] - started) / 1e6
                return row

        arms = args.arms
        for cycle in range(args.cycles):
            for arm in arms if cycle % 2 == 0 else reversed(arms):
                target_ms = 1000.0 if arm == "adaptive2048-1000ms" else args.target_ms
                label = f"cycle{cycle}-{arm}"
                trace = args.output.parent / (label + ".trace.jsonl")
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--model",
                    str(args.model),
                    "--output",
                    str(args.output),
                    "--port",
                    str(args.port),
                    "--arm",
                    arm,
                    "--trace",
                    str(trace),
                    "--target-ms",
                    str(target_ms),
                ]
                env = {
                    k: os.environ[k]
                    for k in ("HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "PATH")
                    if k in os.environ
                }
                env.update(
                    PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    DO_NOT_TRACK="1",
                )
                info = dict(
                    label=label,
                    effective_target_ms=target_ms,
                    command=cmd,
                    controlled_env={
                        k: v
                        for k, v in env.items()
                        if k not in ("HOME", "TMPDIR", "PATH", "PYTHONPATH")
                    },
                )
                result["arms"].append(info)
                with (args.output.parent / (label + ".server.log")).open("w") as log:
                    proc = subprocess.Popen(
                        cmd, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        deadline = time.monotonic() + 180
                        while time.monotonic() < deadline:
                            if proc.poll() is not None:
                                raise RuntimeError(f"{label}: server exited; see log")
                            try:
                                if control.get(base + "/health").status_code == 200:
                                    break
                            except httpx.TransportError:
                                pass
                            time.sleep(0.2)
                        else:
                            raise TimeoutError("server readiness")
                        ask(prompt(128, "warmup", "exact"), 32)
                        for workload in args.workloads:
                            budget = 96 if workload == "prose" else 32
                            expected_long = {"exact": "42", "retrieval": "7319"}.get(
                                workload
                            )
                            expected_short = {"exact": "42", "retrieval": "9283"}.get(
                                workload
                            )
                            # Same payload across arms and cycles; clear all cache between waves.
                            long = prompt(16384, "long-" + workload, workload)
                            short = prompt(256, "short-" + workload, workload)
                            clear()
                            long_solo = ask(long, budget)
                            validate_cold(long_solo, expected_long)
                            for rep in range(args.repeat):
                                clear()
                                solo = ask(short, budget)
                                validate_cold(solo, expected_short)
                                clear()
                                with cf.ThreadPoolExecutor(max_workers=2) as pool:
                                    wave_started = time.perf_counter_ns()
                                    f_long = pool.submit(ask, long, budget)
                                    deadline = time.monotonic() + 30
                                    while True:
                                        s = status()
                                        if int(s.get("num_running") or 0) > 0:
                                            break
                                        if time.monotonic() > deadline or f_long.done():
                                            raise RuntimeError(
                                                "failed to observe active long request"
                                            )
                                        time.sleep(0.01)
                                    f_short = pool.submit(ask, short, budget)
                                    pair = [f_long.result(), f_short.result()]
                                    validate_overlap(*pair)
                                    wave_ms = (
                                        time.perf_counter_ns() - wave_started
                                    ) / 1e6
                                for row, expected in zip(
                                    pair, (expected_long, expected_short)
                                ):
                                    validate_cold(row, expected)
                                item = dict(
                                    arm=arm,
                                    cycle=cycle,
                                    workload=workload,
                                    repeat=rep,
                                    long_solo=long_solo,
                                    short_solo=solo,
                                    long=pair[0],
                                    short=pair[1],
                                    wave_ms=wave_ms,
                                    observed_status=s,
                                )
                                result["rows"].append(item)
                                save()
                                print(
                                    json.dumps(
                                        dict(
                                            arm=arm,
                                            cycle=cycle,
                                            workload=workload,
                                            repeat=rep,
                                            short_ttft_ms=pair[1]["ttft_ms"],
                                            long_total_ms=pair[0]["total_ms"],
                                        )
                                    ),
                                    flush=True,
                                )
                        info["final_status"] = status()
                    finally:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                        info["exit_code"] = proc.returncode
                        save()
    if any(arm.get("exit_code") not in (0, -15) for arm in result["arms"]):
        raise RuntimeError("unclean child shutdown; campaign is incomplete")
    result["summary"] = {
        arm: {
            workload: {
                key: st.median(
                    row[which][metric]
                    for row in result["rows"]
                    if row["arm"] == arm and row["workload"] == workload
                )
                for key, which, metric in [
                    ("short_ttft_ms", "short", "ttft_ms"),
                    ("short_solo_ttft_ms", "short_solo", "ttft_ms"),
                    ("long_total_ms", "long", "total_ms"),
                    ("long_solo_total_ms", "long_solo", "total_ms"),
                ]
            }
            for workload in args.workloads
        }
        for arm in arms
    }
    result["complete"] = True
    save()
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
