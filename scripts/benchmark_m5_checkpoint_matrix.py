#!/usr/bin/env python3
"""Real HTTP qualification of hybrid checkpoints with stock/fast prefill.

Owns a loopback-only child server per arm. Never connects to an existing
service or downloads weights. A cold execution of every edited request is
the correctness reference for its warm execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8617)
    args = ap.parse_args()
    if not args.model.is_dir() or args.rounds < 2:
        ap.error("local snapshot and >=2 rounds required")
    # Fail before any HTTP request if another service already owns the port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    result = {
        "model": str(args.model),
        "rounds": args.rounds,
        "rows": [],
        "arms": [],
        "correctness": "SHA256 of concatenated content/reasoning fields; HTTP does not expose token IDs",
        "other_flags": {
            "fused_gdn_decode": False,
            "compiled_decode": False,
            "speculation": False,
        },
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2))

    with httpx.Client(timeout=240, trust_env=False) as client:

        def clear():
            r = client.post(base + "/v1/cache/clear")
            r.raise_for_status()

        def ask(messages):
            t = time.perf_counter()
            first = None
            content = []
            reasoning = []
            usage = {}
            finish = None
            with client.stream(
                "POST",
                base + "/v1/chat/completions",
                json={
                    "model": "m5-qualification",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 32,
                    "enable_thinking": False,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    obj = json.loads(line[5:])
                    usage = obj.get("usage") or usage
                    for choice in obj.get("choices", []):
                        d = choice.get("delta") or {}
                        c = d.get("content") or ""
                        q = d.get("reasoning_content") or d.get("reasoning") or ""
                        if (c or q) and first is None:
                            first = time.perf_counter()
                        content.append(c)
                        reasoning.append(q)
                        finish = choice.get("finish_reason") or finish
            if first is None or not isinstance(usage.get("prompt_tokens"), int):
                raise RuntimeError("missing visible output or usage")
            return {
                "ttft_s": first - t,
                "elapsed_s": time.perf_counter() - t,
                "usage": usage,
                "finish_reason": finish,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                ),
                "output_sha256": hashlib.sha256(
                    json.dumps(
                        ["".join(content), "".join(reasoning)], ensure_ascii=False
                    ).encode()
                ).hexdigest(),
            }

        # Counterbalance checkpoint order within each prefill mode.
        for prefill, checkpoint in [(0, 0), (0, 4), (1, 4), (1, 0)]:
            name = f"prefill{prefill}-checkpoint{checkpoint}"
            log = args.output.parent / (name + ".server.log")
            env = os.environ.copy()
            env.update(
                RAPID_MLX_HYBRID_CHECKPOINT_MAX=str(checkpoint),
                RAPID_MLX_GDN_PREFILL=str(prefill),
                RAPID_MLX_QWEN35_FUSED_GDN_DECODE="0",
                RAPID_MLX_COMPILED_DECODE="0",
                RAPID_MLX_LANE_MATMUL="off",
                RAPID_MLX_PROMPT_HOST_CACHE="0",
                RAPID_MLX_PREFIX_CACHE_MAX_BYTES=str(2**31),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                RAPID_MLX_TELEMETRY="0",
            )
            cmd = [
                sys.executable,
                "-m",
                "rapid_mlx.cli",
                "serve",
                str(args.model),
                "--served-model-name",
                "m5-qualification",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--no-mllm",
                "--no-spec-decode",
                "--enable-prefix-cache",
                "--hybrid-cache-entries",
                "8",
                "--prefill-step-size",
                "2048",
                "--log-level",
                "INFO",
            ]
            with log.open("w") as stream:
                proc = subprocess.Popen(
                    cmd, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                arm = {"name": name, "command": cmd, "log": log.name}
                result["arms"].append(arm)
                save()
                try:
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        if proc.poll() is not None:
                            raise RuntimeError(
                                f"{name}: server exited {proc.returncode}; see {log}"
                            )
                        try:
                            if client.get(base + "/health").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(0.2)
                    else:
                        raise TimeoutError("server readiness timeout")
                    ask([{"role": "user", "content": "Say READY."}])
                    for rd in range(args.rounds):
                        rng = random.Random(7000 + rd)
                        words = [
                            "river",
                            "stone",
                            "window",
                            "market",
                            "signal",
                            "harbor",
                            "copper",
                            "garden",
                            "thread",
                            "engine",
                            "valley",
                            "mirror",
                            "ladder",
                            "candle",
                            "forest",
                            "bridge",
                            "packet",
                            "anchor",
                            "silver",
                            "meadow",
                        ]
                        sections = [
                            "Section "
                            + str(i)
                            + ": "
                            + " ".join(
                                rng.choice(words) + str(rng.randrange(1000))
                                for _ in range(22)
                            )
                            for i in range(70)
                        ]

                        def messages(parts, document_id=rd):
                            return [
                                {
                                    "role": "system",
                                    "content": "You are a concise local benchmark assistant.",
                                },
                                {
                                    "role": "user",
                                    "content": "Document "
                                    + str(document_id)
                                    + ":\n"
                                    + "\n".join(parts)
                                    + "\nSummarize three specific facts from the document.",
                                },
                            ]

                        for case, index in [
                            ("late_edit", 60),
                            ("mid_edit", 35),
                            ("head_edit", 0),
                        ]:
                            edited = list(sections)
                            edited[index] = (
                                edited[index].replace("Section", "Revised section", 1)
                                + " Updated record: cedar824."
                            )
                            clear()
                            seed = ask(messages(sections))
                            warm = ask(messages(edited))
                            clear()
                            cold = ask(messages(edited))
                            row = {
                                "arm": name,
                                "round": rd,
                                "case": case,
                                "seed": seed,
                                "warm": warm,
                                "cold": cold,
                                "output_exact": warm["output_sha256"]
                                == cold["output_sha256"],
                                "ttft_speedup_vs_same_arm_cold": cold["ttft_s"]
                                / warm["ttft_s"],
                            }
                            result["rows"].append(row)
                            save()
                            print(json.dumps(row), flush=True)
                except Exception as exc:
                    arm["error"] = repr(exc)
                    save()
                    raise
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    arm["server_exit"] = proc.returncode
                    save()
    result["all_outputs_exact"] = all(r["output_exact"] for r in result["rows"])
    save()
    print(
        json.dumps(
            {
                "event": "summary",
                "pairs": len(result["rows"]),
                "all_outputs_exact": result["all_outputs_exact"],
            }
        ),
        flush=True,
    )
    if not result["all_outputs_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
