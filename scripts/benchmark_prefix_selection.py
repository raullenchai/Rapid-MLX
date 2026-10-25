#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serial, same-inventory A/B replay of prefix selection against a baseline file.

Run under scripts/large-model-run.py. The baseline is a trusted memory_cache.py
exported from git; only its probe/fetch selection methods are used. All model
execution uses the current checkout. This local benchmark is not a server API.
Each case supplies seeds (chat request bodies), a query, repeats, and optionally
an expected JSON value. No query result is retained between measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def install_replay(args):
    from rapid_mlx import memory_cache as current

    name = "rapid_mlx._benchmark_baseline_cache"
    spec = importlib.util.spec_from_file_location(name, args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    cls = current.MemoryAwarePrefixCache
    methods = {
        "baseline": (
            baseline.MemoryAwarePrefixCache._probe_locked,
            baseline.MemoryAwarePrefixCache._fetch_locked,
        ),
        "candidate": (cls._probe_locked, cls._fetch_locked),
    }
    methods["long-only"] = methods["baseline"]
    original_probe, original_fetch = cls.probe, cls.fetch
    snapshots, seen = {}, {}
    events = []

    def prepare(cache):
        control = json.loads((args.output / "control.json").read_text())
        identity = id(cache)
        if seen.get(identity) == control["id"]:
            return
        seen[identity] = control["id"]
        cls._probe_locked, cls._fetch_locked = methods[control["mode"]]
        if control.get("clear"):
            cache.clear()
            snapshots.pop(identity, None)
        if control.get("restore"):
            if identity not in snapshots:
                snapshots[identity] = [
                    (list(e.tokens), e.cache) for e in cache._entries.values()
                ]
            cache.clear()
            inventory = snapshots[identity]
            if control["mode"] == "long-only":
                # Positive control: existing restoration of the same long entry,
                # without a competing shorter prefix. Not part of A/B timings.
                inventory = [max(inventory, key=lambda entry: len(entry[0]))]
            for tokens, layers in inventory:
                if not cache.store(tokens, layers, evict_prefixes=False):
                    raise RuntimeError("Cannot restore the seed inventory")
            events.append(
                {
                    "id": control["id"],
                    "lengths": [len(e.tokens) for e in cache._entries.values()],
                }
            )
            write_json(args.output / "inventories.json", events)

    def probe(cache, *a, **kw):
        prepare(cache)
        return original_probe(cache, *a, **kw)

    def fetch(cache, *a, **kw):
        prepare(cache)
        return original_fetch(cache, *a, **kw)

    cls.probe, cls.fetch = probe, fetch


def measure(url, body):
    start = time.perf_counter()
    first, usage, finish = None, None, None
    pieces, done = [], False
    with (
        httpx.Client(timeout=900, trust_env=False) as client,
        client.stream("POST", url + "/v1/chat/completions", json=body) as response,
    ):
        if response.is_error:
            raise RuntimeError(response.read().decode())
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                content = choice.get("delta", {}).get("content") or ""
                if content:
                    if first is None:
                        first = time.perf_counter()
                    pieces.append(content)
    end = time.perf_counter()
    if (
        not done
        or finish != "stop"
        or first is None
        or not usage
        or "completion_tokens" not in usage
    ):
        raise RuntimeError(
            f"Incomplete/truncated response: done={done}, finish={finish}, usage={usage}"
        )
    text = "".join(pieces)
    return dict(
        ttft_s=first - start,
        elapsed_s=end - start,
        usage=usage,
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18646)
    parser.add_argument("--serve-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("model", "baseline_source", "cases", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if os.environ.get("RAPID_LARGE_MODEL_LOCK_HELD") != "1":
        parser.error("Run under scripts/large-model-run.py")
    root = Path(__file__).resolve().parents[1]
    if args.serve_child:
        import runpy

        install_replay(args)
        sys.argv = [
            "rapid-mlx",
            "serve",
            str(args.model),
            "--served-model-name",
            "prefix-bench",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--disable-disk-caches",
            "--disable-model-downloads",
            "--yes",
            "--kv-cache-dtype",
            "bf16",
            "--spec-decode",
            "mtp",
            "--mtp-sidecar",
            str(args.model),
            "--mtp-num-draft-tokens",
            "3",
            "--mtp-max-k",
            "3",
            "--prefill-step-size",
            "2048",
        ]
        runpy.run_module("rapid_mlx.cli", run_name="__main__")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    write_json(args.output / "cases.json", cases)
    write_json(args.output / "control.json", dict(id="init", mode="baseline"))
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--serve-child",
    ]
    write_json(
        args.output / "command.json",
        dict(
            argv=command,
            baseline_sha256=hashlib.sha256(
                args.baseline_source.read_bytes()
            ).hexdigest(),
            candidate_sha256=hashlib.sha256(
                (root / "rapid_mlx/memory_cache.py").read_bytes()
            ).hexdigest(),
        ),
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        RAPID_MLX_TELEMETRY="0",
        DO_NOT_TRACK="1",
    )
    url = f"http://127.0.0.1:{args.port}"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    rows = []

    def request(case, body, mode, identifier, seed=False, clear=False):
        write_json(
            args.output / "control.json",
            dict(
                id=identifier,
                mode=mode,
                clear=clear,
                restore=not seed and bool(case["seeds"]),
            ),
        )
        payload = dict(
            model="prefix-bench",
            temperature=0,
            max_tokens=256,
            chat_template_kwargs={"enable_thinking": False},
        )
        payload.update(body)
        payload.update(stream=True, stream_options={"include_usage": True})
        row = dict(
            case=case["name"],
            id=identifier,
            mode=mode,
            seed=seed,
            **measure(url, payload),
        )
        if not seed and "expected" in case:
            try:
                parsed = json.loads(row["text"])
            except ValueError:
                parsed = None
            row["strict_json_ok"] = parsed == case["expected"]
        rows.append(row)
        write_json(args.output / "results.json", rows)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)

    with (args.output / "server.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(300):
                if child.poll() is not None:
                    raise RuntimeError("Benchmark server exited")
                try:
                    if (
                        httpx.get(
                            url + "/v1/models", timeout=1, trust_env=False
                        ).status_code
                        == 200
                    ):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Benchmark server startup")
            for case in cases:
                for index, seed in enumerate(case["seeds"]):
                    request(
                        case,
                        seed,
                        "baseline",
                        f"{case['name']}-seed-{index}",
                        seed=True,
                        clear=index == 0,
                    )
                for repeat in range(case.get("repeats", 3)):
                    modes = case.get("modes", ["baseline", "candidate"])
                    if repeat % 2:
                        modes = list(reversed(modes))
                    for mode in modes:
                        request(
                            case,
                            case["query"],
                            mode,
                            f"{case['name']}-{mode}-{repeat}",
                            clear=not case["seeds"],
                        )
            write_json(args.output / "complete.json", dict(requests=len(rows)))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    main()
