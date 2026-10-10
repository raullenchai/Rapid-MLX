"""Isolated streaming/cache/concurrency probe; no model downloads."""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

ROOT = Path(os.environ.get("AGENT_PROBE_OUTPUT_DIR", Path(__file__).parent))
ROOT.mkdir(parents=True, exist_ok=True)
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3"
)
URL = os.environ.get("AGENT_PROBE_URL", "http://127.0.0.1:18347")
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def messages(size, salt):
    # Independent beginnings prevent accidental prefix reuse in cold trials.
    prefix = f"Independent session {salt}. Inspect these deterministic code records.\n"
    lines = [
        f"def record_{i:05d}(x): return (x * {i * 13 + 7}) % {i * 3 + 101}\n"
        for i in range(size)
    ]
    text = prefix + "".join(lines)
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if len(tokenizer.encode(text[:mid])) <= size:
            low = mid
        else:
            high = mid - 1
    return [
        {
            "role": "user",
            "content": text[:low]
            + "\nExplain the purpose of these records in about 150 words.",
        }
    ]


def request(msg, max_tokens=96, barrier=None, started_event=None):
    payload = dict(
        model="agent-probe",
        messages=msg,
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
        enable_thinking=False,
    )
    with httpx.Client(timeout=180) as client:
        if barrier:
            barrier.wait()
        start = time.perf_counter()
        times, pieces, usage, finish, done = [], [], {}, None, False
        if started_event:
            started_event.set()
        with client.stream("POST", URL + "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    continue
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        times.append(time.perf_counter())
                        pieces.append(content)
                    finish = choice.get("finish_reason") or finish
        end = time.perf_counter()
    assert done and finish and times and usage.get("completion_tokens", 0) > 0, (
        done,
        finish,
        usage,
    )
    intervals = [(b - a) * 1000 for a, b in zip(times, times[1:])]

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(int((len(xs) - 1) * q), len(xs) - 1)] if xs else None

    return dict(
        ttft_ms=(times[0] - start) * 1000,
        total_ms=(end - start) * 1000,
        mean_tpot_ms=(times[-1] - times[0])
        * 1000
        / max(usage["completion_tokens"] - 1, 1),
        sse_gap_p95_ms=pct(intervals, 0.95),
        sse_gap_max_ms=max(intervals, default=0),
        usage=usage,
        finish_reason=finish,
        done=done,
        output_sha256=hashlib.sha256("".join(pieces).encode()).hexdigest(),
        output_text="".join(pieces),
    )


def status():
    with httpx.Client(timeout=30) as client:
        return client.get(URL + "/v1/status").json()


def clear():
    with httpx.Client(timeout=30) as client:
        client.post(URL + "/v1/cache/clear").raise_for_status()
    s = status()
    assert not s.get("num_running") and not s.get("num_waiting"), s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--contention-only", action="store_true")
    args = parser.parse_args()
    out = dict(
        label=args.label,
        model=str(MODEL),
        started=time.time(),
        runs=[],
        initial_status=status(),
    )

    def record(kind, rows, wall=None, **extra):
        item = dict(kind=kind, rows=rows, **extra)
        if wall:
            item.update(
                wall_s=wall,
                output_tps=sum(x["usage"]["completion_tokens"] for x in rows) / wall,
            )
        out["runs"].append(item)
        (ROOT / (args.label + ".json")).write_text(json.dumps(out, indent=2))
        print(
            json.dumps(
                dict(
                    label=args.label,
                    kind=kind,
                    ttft_ms=[round(x["ttft_ms"], 1) for x in rows],
                    cached=[
                        x["usage"]
                        .get("prompt_tokens_details", {})
                        .get("cached_tokens", 0)
                        for x in rows
                    ],
                    output_tps=item.get("output_tps"),
                )
            ),
            flush=True,
        )

    request(messages(128, "warmup"), 16)
    for rep in range(0 if args.contention_only else args.repeat):
        clear()
        msg = messages(4096, f"cache-{rep}")
        cold = request(msg)
        warm = request(msg)
        partial = [
            dict(
                msg[0],
                content=msg[0]["content"] + "\nAlso mention the modulo operation.",
            )
        ]
        tail = request(partial)
        record("cache", [cold, warm, tail], repeat=rep)
    for concurrency in [] if args.contention_only else [1, 2, 4]:
        for rep in range(args.repeat):
            clear()
            prompts = [
                messages(2048, f"concurrent-{concurrency}-{rep}-{i}")
                for i in range(concurrency)
            ]
            barrier = threading.Barrier(concurrency)
            with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
                start = time.perf_counter()
                futures = [pool.submit(request, m, 96, barrier) for m in prompts]
                rows = [f.result() for f in futures]
                wall = time.perf_counter() - start
            record("concurrency", rows, wall, concurrency=concurrency, repeat=rep)
    for rep in range(args.repeat):
        clear()
        long = messages(16384, f"long-{rep}")
        short = messages(256, f"short-{rep}")
        solo = request(short)
        clear()
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            start = time.perf_counter()
            f_long = pool.submit(request, long)
            deadline = time.monotonic() + 30
            while True:
                s = status()
                if int(s.get("num_running") or 0) > 0:
                    break
                if time.monotonic() > deadline or f_long.done():
                    raise RuntimeError("Failed to observe long request in scheduler")
                time.sleep(0.01)
            f_short = pool.submit(request, short)
            rows = [f_long.result(), f_short.result()]
            wall = time.perf_counter() - start
        record("contention", rows, wall, repeat=rep, short_solo=solo, observed_status=s)
    out["final_status"] = status()
    out["finished"] = time.time()
    (ROOT / (args.label + ".json")).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
