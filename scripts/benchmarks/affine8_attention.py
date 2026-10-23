"""Reproduce the 32K affine8 MTP service A/B on a local model (no downloads).

Run through scripts/large-model-run.py with the appropriate working-set size.
Writes prompts, raw streaming timings, output hashes and dispatch counts to
--output. The service binds port 18643 and is stopped when the run finishes.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def serve():
    import runpy
    import threading

    from rapid_mlx.kernels import affine8_attention as attention

    directory = Path(os.environ["RAPID_AFFINE8_BENCH_DIR"])
    counts = {}
    original = attention.try_affine8_attention

    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        key = f"{os.environ.get('RAPID_MLX_AFFINE8_ATTENTION')}:{result is not None}"
        counts[key] = counts.get(key, 0) + 1
        return result

    attention.try_affine8_attention = measured

    def monitor():
        while True:
            mode = (directory / "mode.txt").read_text().strip()
            os.environ["RAPID_MLX_AFFINE8_ATTENTION"] = "1" if mode == "on" else "0"
            temporary = directory / "receipt.tmp"
            temporary.write_text(json.dumps({"mode": mode, "counts": counts.copy()}))
            temporary.replace(directory / "receipt.json")
            time.sleep(0.1)

    threading.Thread(target=monitor, daemon=True).start()
    sys.argv.remove("--serve")
    runpy.run_module("rapid_mlx.cli", run_name="__main__")


if "--serve" in sys.argv:
    serve()
    sys.exit(0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
R = args.output.resolve()
R.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
MODEL = args.model.resolve()
URL = "http://127.0.0.1:18643"
LABEL = "code"
DTYPE = "int8"


def dump(n, v):
    (R / n.replace("e2e", LABEL)).write_text(
        json.dumps(v, indent=2, ensure_ascii=False) + "\n"
    )


def request(prompt, mode, case, rep, warm=False):
    (R / "mode.txt").write_text(mode)
    time.sleep(0.4)
    receipt = json.loads((R / "receipt.json").read_text())
    assert receipt["mode"] == mode
    body = {
        "model": "kv-screen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first = None
    usage = {}
    chunks = []
    start = time.perf_counter()
    with (
        httpx.Client(timeout=900, trust_env=False) as client,
        client.stream("POST", URL + "/v1/chat/completions", json=body) as resp,
    ):
        if resp.is_error:
            raise RuntimeError(resp.read().decode())
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            x = json.loads(line[6:])
            if x.get("error"):
                raise RuntimeError(x["error"])
            if x.get("usage"):
                usage = x["usage"]
            for c in x.get("choices", []):
                d = c.get("delta", {})
                v = d.get("content") or d.get("reasoning_content") or ""
                if v:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(v)
    end = time.perf_counter()
    tokens = usage.get("completion_tokens")
    assert first and tokens
    out = "".join(chunks)
    row = {
        "mode": mode,
        "case": case,
        "repeat": rep,
        "warmup": warm,
        "usage": usage,
        "ttft_s": first - start,
        "elapsed_s": end - start,
        "decode_tps": (tokens - 1) / (end - first),
        "text": out,
        "hash": hashlib.sha256(out.encode()).hexdigest(),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    print(
        json.dumps(
            {k: v for k, v in row.items() if k not in ("text", "hash", "prompt_hash")}
        ),
        flush=True,
    )
    return row


assert os.environ.get("RAPID_LARGE_MODEL_LOCK_HELD") == "1"
(R / "mode.txt").write_text("off")
cmd = [
    PY,
    str(Path(__file__).resolve()),
    "--serve",
    "serve",
    str(MODEL),
    "--served-model-name",
    "kv-screen",
    "--host",
    "127.0.0.1",
    "--port",
    "18643",
    "--disable-disk-caches",
    "--disable-model-downloads",
    "--yes",
    "--kv-cache-dtype",
    DTYPE,
    "--spec-decode",
    "mtp",
    "--mtp-sidecar",
    str(MODEL),
    "--mtp-num-draft-tokens",
    "3",
    "--mtp-max-k",
    "3",
    "--mtp-disable-auto-k",
    "--prefill-step-size",
    "2048",
]
env = dict(
    os.environ,
    RAPID_AFFINE8_BENCH_DIR=str(R),
    PYTHONPATH=str(ROOT),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    RAPID_MLX_TELEMETRY="0",
    DO_NOT_TRACK="1",
)
dump("e2e-command.json", {"argv": cmd})
rows = []
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 18643))
with (R / f"{LABEL}-server.log").open("w") as log:
    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        for _ in range(240):
            if p.poll() is not None:
                raise RuntimeError("server exited")
            try:
                if (
                    httpx.get(
                        URL + "/v1/models", timeout=1, trust_env=False
                    ).status_code
                    == 200
                ):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("server startup")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        filler = "Reference log entry: the lighthouse lens was cleaned, the tide was measured, and the keeper checked the radio.\n"
        unit = tok.encode(filler, add_special_tokens=False)
        size = 32768
        ids = (unit * ((size // len(unit)) + 1))[: size - 120]
        prompt = (
            "Use this background log only as atmosphere:\n"
            + tok.decode(ids)
            + "\nNow implement a Python LRU cache with a doubly linked list and a dictionary, with type hints. Output only Python code."
        )
        dump("e2e-prompts.json", {str(size): prompt})
        for mode in ("on", "off"):
            rows.append(request(prompt, mode, str(size), -1, True))
            dump("e2e-requests.json", rows)
        for rep in range(3):
            for mode in ("on", "off") if rep % 2 == 0 else ("off", "on"):
                rows.append(request(prompt, mode, str(size), rep))
                dump("e2e-requests.json", rows)
        time.sleep(0.3)
        receipt = json.loads((R / "receipt.json").read_text())
        assert receipt["counts"].get("1:True", 0) > 0, "Fused path never engaged"
        dump("e2e-complete.json", {"requests": len(rows), "receipt": receipt})
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
