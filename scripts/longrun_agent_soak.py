"""Opt-in, long-running live-server soak; never collected by unit pytest.

Run on a model-capable host. The harness attaches to an existing loopback server
and does not load a model. Outputs minute samples and per-error JSONL records.
"""

import argparse
import asyncio
import csv
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import httpx

SHARED_PREFIX = (
    "You are a coding assistant. Answer briefly and use a tool when requested. "
    "Project rules: verify inputs, explain uncertainty, keep changes small. " * 12
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_symbol",
            "description": "Find a symbol in a source tree",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
]
FIELDS = (
    "minute",
    "elapsed_s",
    "requests",
    "successes",
    "errors",
    "disconnects",
    "stream",
    "nonstream",
    "tool",
    "long_prompt",
    "health_ok",
    "models_ok",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "rss_mb",
    "threads",
    "open_files",
    "metal_active_gb",
    "metal_cache_gb",
    "metal_peak_gb",
    "running",
    "waiting",
)


def percentile(values, fraction):
    if not values:
        return ""
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 1)


def process_stats(pid):
    result = subprocess.run(
        ["ps", "-o", "rss=,thcount=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    parts = result.stdout.split()
    rss = round(int(parts[0]) / 1024, 1) if len(parts) >= 2 else ""
    threads = int(parts[1]) if len(parts) >= 2 else ""
    files = subprocess.run(
        ["lsof", "-p", str(pid), "-Fn"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    open_files = sum(line.startswith("n") for line in files.stdout.splitlines())
    return rss, threads, open_files


class Soak:
    def __init__(self, args):
        self.args = args
        self.start = time.monotonic()
        self.stop = self.start + args.duration
        self.random = random.Random(args.seed)
        self.sequence = 0
        self.events = []
        self.errors = open(args.output / "errors.jsonl", "w", encoding="utf-8")
        self.csv_file = open(
            args.output / "minutes.csv", "w", newline="", encoding="utf-8"
        )
        self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.totals = Counter()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(args.timeout),
            limits=httpx.Limits(max_connections=args.concurrency + 4),
        )

    def record(self, kind, started, error=None, disconnected=False):
        event = {
            "kind": kind,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "error": error,
            "disconnected": disconnected,
        }
        self.events.append(event)
        self.totals[kind] += 1
        self.totals["requests"] += 1
        self.totals["errors" if error else "successes"] += 1
        if disconnected:
            self.totals["disconnects"] += 1
        if error:
            self.errors.write(
                json.dumps(
                    {"elapsed_s": round(time.monotonic() - self.start, 1), **event}
                )
                + "\n"
            )
            self.errors.flush()

    async def request(self, worker):
        sequence = self.sequence
        self.sequence += 1
        kind = (
            "disconnect"
            if sequence % 13 == 12
            else "tool"
            if sequence % 5 == 4
            else "long_prompt"
            if sequence % 17 == 16
            else "stream"
            if sequence % 2
            else "nonstream"
        )
        prompt = f"Worker {worker} turn {sequence}: "
        if kind == "tool":
            prompt += "Use lookup_symbol to locate the function named parse_request."
        elif kind == "long_prompt":
            prompt += (
                "Review this repeated source note for a potential bug: "
                + "the request must release resources after cancellation. " * 75
            )
        else:
            prompt += "Explain one safe way to handle a cancelled request."
        payload = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": SHARED_PREFIX},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 80 if kind != "disconnect" else 512,
            "temperature": 0,
            "stream": kind in {"stream", "disconnect", "long_prompt"},
        }
        if kind == "tool":
            payload["tools"] = TOOLS
            payload["tool_choice"] = "required"
        started = time.monotonic()
        error = None
        disconnected = False
        try:
            if payload["stream"]:
                chunks = 0
                done = False
                async with self.client.stream(
                    "POST", self.args.url + "/v1/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                done = True
                            else:
                                json.loads(line[6:])
                                chunks += 1
                        if kind == "disconnect" and chunks >= 3:
                            disconnected = True
                            break
                if kind == "disconnect" and not disconnected:
                    error = "disconnect stream ended before 3 chunks"
                elif not disconnected and not done:
                    error = "stream missing [DONE]"
            else:
                response = await self.client.post(
                    self.args.url + "/v1/chat/completions", json=payload
                )
                response.raise_for_status()
                body = response.json()
                if not body.get("choices"):
                    error = "no choices"
                elif kind == "tool" and not body["choices"][0]["message"].get(
                    "tool_calls"
                ):
                    error = "required tool call absent"
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:240]}"
        self.record(kind, started, error, disconnected)

    async def worker(self, index):
        while time.monotonic() < self.stop:
            await self.request(index)
            await asyncio.sleep(self.args.pause)

    async def sample(self, minute):
        elapsed = round(time.monotonic() - self.start, 1)
        events, self.events = self.events, []
        counts = Counter(event["kind"] for event in events)
        latencies = [event["latency_ms"] for event in events if not event["error"]]
        row = {key: "" for key in FIELDS}
        row.update(
            minute=minute,
            elapsed_s=elapsed,
            requests=len(events),
            successes=sum(not event["error"] for event in events),
            errors=sum(bool(event["error"]) for event in events),
            disconnects=sum(event["disconnected"] for event in events),
            stream=counts["stream"],
            nonstream=counts["nonstream"],
            tool=counts["tool"],
            long_prompt=counts["long_prompt"],
            latency_p50_ms=percentile(latencies, 0.5),
            latency_p95_ms=percentile(latencies, 0.95),
            latency_p99_ms=percentile(latencies, 0.99),
        )
        for path, key in (("/health", "health_ok"), ("/v1/models", "models_ok")):
            try:
                response = await self.client.get(self.args.url + path, timeout=10)
                row[key] = int(response.status_code == 200)
            except httpx.HTTPError:
                row[key] = 0
        try:
            response = await self.client.get(self.args.url + "/v1/status", timeout=10)
            response.raise_for_status()
            status = response.json()
            metal = status.get("metal") or {}
            row.update(
                metal_active_gb=metal.get("active_memory_gb"),
                metal_cache_gb=metal.get("cache_memory_gb"),
                metal_peak_gb=metal.get("peak_memory_gb"),
                running=status.get("num_running"),
                waiting=status.get("num_waiting"),
            )
        except (httpx.HTTPError, ValueError):
            pass
        row["rss_mb"], row["threads"], row["open_files"] = await asyncio.to_thread(
            process_stats, self.args.pid
        )
        self.writer.writerow(row)
        self.csv_file.flush()
        print(json.dumps(row), flush=True)
        if row["rss_mb"] and row["rss_mb"] > self.args.max_rss_mb:
            raise RuntimeError(f"RSS budget exceeded: {row['rss_mb']} MB")

    async def run(self):
        try:
            await self.sample(0)
            workers = [
                asyncio.create_task(self.worker(i))
                for i in range(self.args.concurrency)
            ]
            minute = 1
            while time.monotonic() < self.stop:
                await asyncio.sleep(
                    max(0, min(self.stop, self.start + minute * 60) - time.monotonic())
                )
                await self.sample(minute)
                minute += 1
            await asyncio.gather(*workers)
        finally:
            await self.client.aclose()
            self.errors.close()
            self.csv_file.close()
            (self.args.output / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_s": round(time.monotonic() - self.start, 1),
                        "server_pid": self.args.pid,
                        "model": self.args.model,
                        "totals": dict(self.totals),
                    },
                    indent=2,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18130")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--duration", type=int, default=10800)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-rss-mb", type=float, default=12288)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(Soak(args).run())


if __name__ == "__main__":
    main()
