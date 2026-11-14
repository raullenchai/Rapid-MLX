#!/usr/bin/env python3
"""Train a small cached-model LoRA and compare served greedy tokens with fusion.

Run with the contributor venv: python scripts/check_lora_adapter.py
All weights/data/logs stay in --scratch. No model downloads are permitted.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path("/private/tmp/rapid-mlx-lora-adapter-scratch"),
    )
    parser.add_argument("--port", type=int, default=8613)
    args = parser.parse_args()
    root = args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    from huggingface_hub import snapshot_download

    model = snapshot_download("mlx-community/Qwen3-0.6B-4bit", local_files_only=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    for split in ("train", "valid"):
        (data / f"{split}.jsonl").write_text(
            "\n".join(
                json.dumps({"text": "The secret code is violet marmalade. " * 8})
                for _ in range(16)
            )
            + "\n"
        )
    adapter, fused = root / "adapter", root / "fused"
    commands = [
        [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--model",
            model,
            "--train",
            "--data",
            str(data),
            "--adapter-path",
            str(adapter),
            "--iters",
            "8",
            "--num-layers",
            "4",
            "--batch-size",
            "1",
            "--learning-rate",
            "0.001",
            "--steps-per-report",
            "1",
            "--steps-per-eval",
            "8",
            "--val-batches",
            "1",
            "--max-seq-length",
            "128",
            "--seed",
            "7",
        ],
        [
            sys.executable,
            "-m",
            "mlx_lm.fuse",
            "--model",
            model,
            "--adapter-path",
            str(adapter),
            "--save-path",
            str(fused),
            "--dequantize",
        ],
    ]
    for index, command in enumerate(commands):
        print("RUN", subprocess.list2cmdline(command), flush=True)
        with (root / f"step-{index}.log").open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    # Probe before spawning: never attach to or kill another user's server.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command = [
        str(Path(sys.executable).parent / "rapid-mlx"),
        "serve",
        model,
        "--adapter",
        str(adapter),
        "--port",
        str(args.port),
        "--no-spec-decode",
        "--max-tokens",
        "16",
    ]
    print("RUN", subprocess.list2cmdline(command), flush=True)
    base_url = f"http://127.0.0.1:{args.port}"
    prompt = "The secret code is"
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 240
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited {server.returncode}; see {root / 'server.log'}"
                    )
                try:
                    with urllib.request.urlopen(
                        base_url + "/v1/models", timeout=2
                    ) as response:
                        identity = json.load(response)["data"][0]["id"]
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("adapter server startup timed out")
                    time.sleep(0.5)
            request = urllib.request.Request(
                base_url + "/v1/completions",
                data=json.dumps(
                    {
                        "model": identity,
                        "prompt": prompt,
                        "temperature": 0,
                        "max_tokens": 12,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                served = json.load(response)["choices"][0]["text"]
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    outputs = {}
    for name, path in (("base", model), ("fused", str(fused))):
        mx.random.seed(7)
        network, tokenizer = load(path)
        outputs[name] = generate(
            network,
            tokenizer,
            prompt=prompt,
            max_tokens=12,
            sampler=make_sampler(temp=0),
            verbose=False,
        )
        del network, tokenizer
        mx.clear_cache()
    result = {
        "identity": identity,
        "prompt": prompt,
        "served": served,
        **outputs,
        "matches_fused": served == outputs["fused"],
        "differs_from_base": served != outputs["base"],
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert ":lora:" in identity, "server did not publish the adapter identity"
    assert result["matches_fused"], "served adapter differs from fused model"
    assert result["differs_from_base"], "adapter did not change greedy output"


if __name__ == "__main__":
    main()
