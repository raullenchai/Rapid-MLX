# SPDX-License-Identifier: Apache-2.0
"""Development-only, plain singleton decode ablations on cached checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_state():
    def read(cmd):
        return subprocess.check_output(cmd, text=True, timeout=15)

    cpu = read(["top", "-l", "2", "-n", "0", "-s", "1"]).split("Processes:")[-1]
    match = re.search(r"CPU usage: [\d.]+% user, [\d.]+% sys, ([\d.]+)% idle", cpu)
    if match is None:
        raise RuntimeError("Cannot read CPU utilization; refusing an ungated sample")
    gpu = []
    for _ in range(3):
        match_gpu = re.search(
            r'"Device Utilization %"=(\d+)',
            read(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"]),
        )
        if match_gpu is None:
            raise RuntimeError(
                "Cannot read GPU utilization; refusing an ungated sample"
            )
        gpu.append(int(match_gpu.group(1)))
        time.sleep(0.5)
    vm_cpu = sum(
        float(line.split(None, 1)[0])
        for line in read(["ps", "-axo", "pcpu=,comm="]).splitlines()
        if "Virtualization.VirtualMachine" in line
    )
    return {
        "cpu": round(100 - float(match.group(1)), 2),
        "gpu": min(gpu),
        "vm_cpu": vm_cpu,
    }


def is_idle(state):
    return state["cpu"] <= 15 and state["gpu"] <= 12 and state["vm_cpu"] <= 20


def wait_idle(timeout):
    start = time.monotonic()
    while True:
        state = load_state()
        if is_idle(state):
            return state
        print(json.dumps({"event": "wait_idle", **state}), flush=True)
        if time.monotonic() - start > timeout:
            raise TimeoutError("Idle gate timed out; no competing process was stopped")
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local snapshot or already-cached repository ID"
    )
    parser.add_argument("--revision", help="Cached checkpoint revision")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="User-message text; contents are never printed",
    )
    parser.add_argument(
        "--arm",
        default="stock",
        choices=[
            "stock",
            "gu",
            "both",
            "all",
            "resnorm",
            "fullnorm",
            "allfull",
            "compile",
            "ladder1",
            "ladder4",
            "wired0",
            "wiredfit",
            "cap128",
            "cap512",
            "gdn",
            "gdnfull",
        ],
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.reps < 1 or args.tokens < 1:
        parser.error("reps and tokens must be positive")
    if args.arm.startswith("cap"):
        # MLX reads this at backend creation; do not initialize MLX earlier.
        os.environ["MLX_MAX_MB_PER_BUFFER"] = args.arm[3:]

    from scripts.pr_validate.gpu_lock import gpu_lock_path, hold_gpu_lock

    def log(message):
        print(json.dumps({"event": "lock", "message": message}), flush=True)

    with hold_gpu_lock(gpu_lock_path(), log=log):
        wait_idle(args.idle_timeout)
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.utils import load_tokenizer
        from mlx_vlm import load

        from bench.dense_decode.projections import install

        path = Path(args.model).expanduser()
        if not path.is_dir():
            path = Path(
                snapshot_download(
                    args.model, revision=args.revision, local_files_only=True
                )
            )
        text = args.prompt_file.read_text()
        tokenizer = load_tokenizer(path)
        prompt = mx.array(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        model, _ = load(str(path), trust_remote_code=False)
        model = model.language_model
        arm = "stock" if args.arm.startswith("cap") else args.arm
        enrolled = install(model, arm)
        limit = mx.device_info()["max_recommended_working_set_size"]
        if arm == "wired0":
            limit = 0
        elif arm == "wiredfit":
            mx.clear_cache()
            limit = mx.get_active_memory()
        previous_wired = mx.set_wired_limit(limit)

        def run(n):
            cache = make_prompt_cache(model)

            def logits(output):
                return output.logits if hasattr(output, "logits") else output

            token = mx.argmax(
                logits(model(prompt[None], cache=cache))[:, -1, :], axis=-1
            )
            mx.eval(token, [c.state for c in cache])
            output, times = [], [0.0, 0.0, 0.0]
            start = time.perf_counter()
            for _ in range(n):
                before = time.perf_counter()
                nxt = mx.argmax(
                    logits(model(token[:, None], cache=cache))[:, -1, :], axis=-1
                )
                built = time.perf_counter()
                mx.async_eval(nxt)
                submitted = time.perf_counter()
                output.append(token.item())
                read = time.perf_counter()
                token = nxt
                for index, elapsed in enumerate(
                    (built - before, submitted - built, read - submitted)
                ):
                    times[index] += elapsed
            mx.synchronize()
            elapsed = time.perf_counter() - start
            return {
                "tps": n / elapsed,
                "ms_per_token": 1000 * elapsed / n,
                "boundaries_ms": dict(
                    zip(
                        ("build", "submit", "read_previous"),
                        (1000 * t / n for t in times),
                    )
                ),
                "token_sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                "active_bytes": mx.get_active_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }

        try:
            print(
                json.dumps(
                    {
                        "event": "metadata",
                        "model": str(path),
                        "arm": args.arm,
                        "mlx": mx.__version__,
                        "device": mx.device_info(),
                        "enrolled": enrolled,
                        "prompt_positions": prompt.size,
                        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "decode_positions": args.tokens,
                    }
                ),
                flush=True,
            )
            run(16)
            samples = []
            for rep in range(args.reps):
                for attempt in range(8):
                    before = wait_idle(args.idle_timeout)
                    sample = run(args.tokens)
                    after = load_state()
                    valid = is_idle(after)
                    print(
                        json.dumps(
                            {
                                "event": "sample",
                                "rep": rep,
                                "attempt": attempt,
                                "valid": valid,
                                "before": before,
                                "after": after,
                                **sample,
                            }
                        ),
                        flush=True,
                    )
                    if valid:
                        samples.append(sample)
                        break
                else:
                    raise RuntimeError("Eight contaminated repetitions")
            print(
                json.dumps(
                    {
                        "event": "summary",
                        "median_tps": statistics.median(r["tps"] for r in samples),
                        "min_tps": min(r["tps"] for r in samples),
                        "max_tps": max(r["tps"] for r in samples),
                        "distinct_token_digests": len(
                            {r["token_sha256"] for r in samples}
                        ),
                    }
                ),
                flush=True,
            )
        finally:
            mx.synchronize()
            mx.set_wired_limit(previous_wired)


if __name__ == "__main__":
    main()
