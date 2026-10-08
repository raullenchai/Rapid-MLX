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
    "cancellations",
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
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    parts = result.stdout.split()
    if result.returncode or not parts:
        raise RuntimeError(f"server PID {pid} is missing or RSS is unreadable")
    rss = round(int(parts[0]) / 1024, 1)
    thread_result = subprocess.run(
        ["ps", "-M", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if thread_result.returncode:
        raise RuntimeError(f"thread count unavailable for server PID {pid}")
    threads = max(0, len(thread_result.stdout.splitlines()) - 1)
    files = subprocess.run(
        ["lsof", "-p", str(pid), "-Fn"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if files.returncode:
        raise RuntimeError(f"open-file count unavailable for server PID {pid}")
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
        self.writer = csv.DictWriter(
            self.csv_file, fieldnames=FIELDS, lineterminator="\n"
        )
        self.writer.writeheader()
        self.totals = Counter()
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(args.timeout),
            limits=httpx.Limits(max_connections=args.concurrency + 4),
        )

    def record(
        self,
        kind,
        started,
        worker,
        sequence,
        stage,
        error=None,
        disconnected=False,
        cancelled=False,
    ):
        event = {
            "kind": kind,
            "worker": worker,
            "sequence": sequence,
            "stage": stage,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "error": error,
            "disconnected": disconnected,
            "cancelled": cancelled,
        }
        self.events.append(event)
        self.totals[kind] += 1
        self.totals["requests"] += 1
        self.totals["errors" if error else "successes"] += 1
        if disconnected:
            self.totals["disconnects"] += 1
        if cancelled:
            self.totals["cancellations"] += 1
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
            "cancel"
            if sequence % 23 == 22
            else "disconnect"
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
            prompt += self.random.choice(
                (
                    "Explain one safe way to handle a cancelled request.",
                    "Name one check to make before retrying an agent tool call.",
                    "Summarize how to release a cached resource after a request.",
                )
            )
        payload = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": SHARED_PREFIX},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512 if kind in {"disconnect", "cancel"} else 80,
            "temperature": 0,
            "stream": kind in {"stream", "disconnect", "cancel", "long_prompt"},
        }
        if kind == "tool":
            payload["tools"] = TOOLS
            payload["tool_choice"] = "required"
        started = time.monotonic()
        try:
            disconnected, cancelled, stage, error = await asyncio.wait_for(
                self.send(kind, payload), timeout=self.args.timeout
            )
        except TimeoutError:
            disconnected, cancelled, stage, error = (
                False,
                False,
                "deadline",
                "request deadline exceeded",
            )
        except httpx.HTTPStatusError as exc:
            disconnected, cancelled, stage, error = (
                False,
                False,
                "http",
                f"HTTP {exc.response.status_code}: {exc.response.text[:240]}",
            )
        except Exception as exc:
            disconnected, cancelled, stage, error = (
                False,
                False,
                "request",
                f"{type(exc).__name__}: {str(exc)[:240]}",
            )
        self.record(
            kind, started, worker, sequence, stage, error, disconnected, cancelled
        )

    async def send(self, kind, payload):
        disconnected = False
        cancelled = False
        requested_cancel = False
        stage = "initial"
        error = None
        try:
            if payload["stream"]:
                chunks = 0
                done = False
                finished = False
                async with self.client.stream(
                    "POST", self.args.url + "/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        detail = bytearray()
                        async for chunk in response.aiter_bytes():
                            detail.extend(chunk[: max(0, 240 - len(detail))])
                            if len(detail) >= 240:
                                break
                        raise ValueError(
                            f"HTTP {response.status_code}: "
                            f"{detail.decode(errors='replace')}"
                        )
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                done = True
                            else:
                                item = json.loads(line[6:])
                                if item.get("error"):
                                    raise ValueError(
                                        f"stream error: {str(item['error'])[:240]}"
                                    )
                                if not isinstance(item.get("choices"), list):
                                    raise ValueError("stream chunk missing choices")
                                finished |= any(
                                    choice.get("finish_reason") is not None
                                    for choice in item["choices"]
                                )
                                chunks += 1
                        if kind == "disconnect" and chunks >= 3:
                            disconnected = True
                            break
                        if kind == "cancel" and chunks >= 3:
                            requested_cancel = True
                            asyncio.current_task().cancel()
                            await asyncio.sleep(0)
                if kind == "disconnect" and not disconnected:
                    error = "disconnect stream ended before 3 chunks"
                elif kind == "cancel":
                    error = "cancel stream ended before 3 chunks"
                elif not disconnected and not (done and finished):
                    error = "stream missing completion marker"
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
                elif kind == "tool":
                    assistant = body["choices"][0]["message"]
                    replies = [
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Found parse_request in server.py.",
                        }
                        for call in assistant["tool_calls"]
                    ]
                    followup = {
                        **payload,
                        "messages": payload["messages"] + [assistant, *replies],
                        "tool_choice": "none",
                        "max_tokens": 40,
                    }
                    stage = "tool_followup"
                    followup_response = await self.client.post(
                        self.args.url + "/v1/chat/completions", json=followup
                    )
                    followup_response.raise_for_status()
                    if not followup_response.json().get("choices"):
                        error = "tool follow-up has no choices"
        except asyncio.CancelledError:
            if kind == "cancel" and requested_cancel:
                cancelled = True
            else:
                raise
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}: {exc.response.text[:240]}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:240]}"
        return disconnected, cancelled, stage, error

    async def worker(self, index):
        while time.monotonic() < self.stop:
            await self.request(index)
            await asyncio.sleep(self.args.pause)

    def event_row(self, minute, events):
        elapsed = round(time.monotonic() - self.start, 1)
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
            cancellations=sum(event["cancelled"] for event in events),
            stream=counts["stream"],
            nonstream=counts["nonstream"],
            tool=counts["tool"],
            long_prompt=counts["long_prompt"],
            latency_p50_ms=percentile(latencies, 0.5),
            latency_p95_ms=percentile(latencies, 0.95),
            latency_p99_ms=percentile(latencies, 0.99),
        )
        return row

    async def sample(self, minute):
        events = self.events[:]
        row = self.event_row(minute, events)
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
        except (httpx.HTTPError, ValueError) as exc:
            self.totals["telemetry_errors"] += 1
            status_error = str(exc)[:240]
        else:
            status_error = None
        row["rss_mb"], row["threads"], row["open_files"] = await asyncio.to_thread(
            process_stats, self.args.pid
        )
        self.writer.writerow(row)
        self.csv_file.flush()
        del self.events[: len(events)]
        print(json.dumps(row), flush=True)
        self.totals["probe_errors"] += 2 - row["health_ok"] - row["models_ok"]
        if (
            status_error
            or row["metal_active_gb"] is None
            or row["metal_cache_gb"] is None
        ):
            raise RuntimeError(f"Metal telemetry unavailable: {status_error}")
        if row["rss_mb"] > self.args.max_rss_mb:
            raise RuntimeError(f"RSS budget exceeded: {row['rss_mb']} MB")
        metal_total = row["metal_active_gb"] + row["metal_cache_gb"]
        if metal_total > self.args.max_metal_gb:
            raise RuntimeError(f"Metal budget exceeded: {metal_total} GB")

    async def run(self):
        workers = []
        minute = 0
        failure = None
        finished_normally = False
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
            await asyncio.wait_for(
                asyncio.gather(*workers),
                timeout=self.args.timeout + self.args.pause + 2,
            )
            await self.sample(minute)
            finished_normally = True
        except BaseException as exc:
            failure = exc
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if self.events:
                self.writer.writerow(self.event_row(minute + 1, self.events))
                self.csv_file.flush()
                self.events.clear()
            await self.client.aclose()
            self.errors.close()
            self.csv_file.close()
            required = (
                "stream",
                "nonstream",
                "tool",
                "long_prompt",
                "disconnect",
                "cancel",
            )
            passed = (
                failure is None
                and finished_normally
                and time.monotonic() - self.start >= self.args.duration
                and self.totals["errors"] == 0
                and self.totals["probe_errors"] == 0
                and self.totals["telemetry_errors"] == 0
                and all(self.totals[kind] > 0 for kind in required)
                and self.totals["disconnects"] > 0
                and self.totals["cancellations"] > 0
            )
            (self.args.output / "summary.json").write_text(
                json.dumps(
                    {
                        "duration_s": round(time.monotonic() - self.start, 1),
                        "server_pid": self.args.pid,
                        "model": self.args.model,
                        "totals": dict(self.totals),
                        "passed": passed,
                        "failure": str(failure) if failure else None,
                    },
                    indent=2,
                )
                + "\n"
            )
        if failure:
            raise failure
        if not passed:
            raise RuntimeError("soak failed; see summary.json and errors.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18130")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--duration", type=int, default=10800)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-rss-mb", type=float, default=11444)
    parser.add_argument("--max-metal-gb", type=float, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.timeout <= 0 or args.concurrency <= 0:
        parser.error("duration, timeout, and concurrency must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(Soak(args).run())


if __name__ == "__main__":
    main()
