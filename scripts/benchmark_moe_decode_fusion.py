#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ABAB streaming serve benchmark in two isolated source trees.

A runs main/ and B runs wt/ under the same venv/. Logs and JSONL stay in the
explicit scratch directory. Only the server subprocess started here is stopped.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import socket
import subprocess
import time
import urllib.request


def main() -> None:
    from mlx_lm.utils import load_tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument(
        "--workdir",
        type=pathlib.Path,
        required=True,
        help="Scratch directory containing main/, wt/, and venv/",
    )
    p.add_argument("--port", type=int, default=18961)
    p.add_argument("--contexts", default="256,4096")
    p.add_argument("--order", default="ABAB")
    p.add_argument("--out", default="serve-results.jsonl")
    p.add_argument("--concurrency", type=int, default=1)
    a = p.parse_args()
    if not pathlib.Path(a.model).is_dir():
        p.error("model must be an existing local checkpoint directory")
    if a.concurrency < 1 or set(a.order) - {"A", "B"}:
        p.error("positive concurrency and an order containing only A/B are required")
    if not 18950 <= a.port <= 18999:
        p.error("benchmark ports must be in 18950-18999")
    root = a.workdir.resolve()
    tok = load_tokenizer(pathlib.Path(a.model))
    port = a.port
    text = (
        "The history of scientific discovery shows how careful observation, reproducible experiments, and clear mathematical reasoning help people understand the natural world. Consider the development of astronomy, physics, biology and computing. Each field depends on evidence and on testing predictions against measurements. "
        * 200
    )
    ids = tok.encode(text)

    def req(ctx, n=128):
        prompt = tok.decode(ids[:ctx])
        data = json.dumps(
            dict(
                model="default",
                prompt=prompt,
                max_tokens=n,
                temperature=0,
                top_p=1,
                top_k=0,
                min_p=0,
                repetition_penalty=1,
                stream=True,
                stream_options={"include_usage": True},
            )
        ).encode()
        start = time.perf_counter()
        stamps = []
        output = ""
        usage = {}
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data,
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as resp:
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                j = json.loads(line[6:])
                usage = j.get("usage") or usage
                for c in j.get("choices", []):
                    if c.get("text"):
                        stamps.append(time.perf_counter())
                        output += c["text"]
        end = time.perf_counter()
        nt = usage.get("completion_tokens", len(tok.encode(output)))
        # SSE text events need not correspond one-to-one with generated tokens.
        # Decode timing is unavailable when events coalesce tokens or text is empty.
        interval = stamps[-1] - stamps[0] if len(stamps) > 1 else 0
        tps = (nt - 1) / interval if len(stamps) == nt and interval > 0 else None
        return dict(
            ctx=ctx,
            tokens=nt,
            tps=tps,
            ttft=stamps[0] - start if stamps else None,
            elapsed=end - start,
            output=output,
            usage=usage,
        )

    for run, v in enumerate(a.order):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port {port} is already in use")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / ("main" if v == "A" else "wt"))
        env["RAPID_MLX_MOE_DECODE_FUSION"] = "0" if v == "A" else "1"
        cmd = [
            str(root / "venv/bin/rapid-mlx"),
            "serve",
            a.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-prefix-cache",
            "--disable-disk-caches",
            "--no-spec-decode",
            "--max-tokens",
            "128",
        ]
        with (root / f"server-{run}-{v}.log").open("w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(240):
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exit {proc.returncode}")
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/health", timeout=2
                        ):
                            pass
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise TimeoutError("server did not become ready")
                req(256, 16)
                for ctx in map(int, a.contexts.split(",")):
                    for rep in range(3):
                        t0 = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=a.concurrency
                        ) as pool:
                            rows = list(pool.map(req, [ctx] * a.concurrency))
                        duration = time.perf_counter() - t0
                        r = rows[0]
                        r["requests"] = rows[1:]
                        r["concurrency"] = a.concurrency
                        r["aggregate_tps"] = sum(z["tokens"] for z in rows) / duration
                        try:
                            r["status"] = json.load(
                                urllib.request.urlopen(
                                    f"http://127.0.0.1:{port}/v1/status", timeout=5
                                )
                            )
                        except Exception as exc:
                            r["status_error"] = str(exc)
                        r.update(variant=v, run=run, rep=rep, model=a.model)
                        with (root / a.out).open("a") as result_file:
                            result_file.write(json.dumps(r) + "\n")
                        print(
                            json.dumps(
                                {
                                    k: x
                                    for k, x in r.items()
                                    if k
                                    not in ("output", "usage", "requests", "status")
                                }
                            ),
                            flush=True,
                        )
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
