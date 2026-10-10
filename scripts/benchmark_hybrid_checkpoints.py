#!/usr/bin/env python3
"""Real HTTP qualification of hybrid checkpoints with stock/fast prefill.

Owns a loopback-only child server per arm. Never connects to an existing
service or downloads weights. Reports incremental checkpoint-on/off and warm/cold correctness separately.
The default cold contract requires both; --contract incremental explicitly
qualifies only the checkpoint change under the same request history.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import httpx

CASES = {"late_edit": (60, 4096), "mid_edit": (35, 2048), "head_edit": (0, 0)}
ARMS = {f"prefill{p}-checkpoint{c}" for p in (0, 1) for c in (0, 4)}


def stream_receipt(
    lines: Iterable[str], started: float, *, require_full_budget: bool = True
) -> dict:
    """Reject truncated/error streams even when they emitted some valid text."""
    first = None
    content, reasoning = [], []
    usage, finish = {}, None
    done = False
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        obj = json.loads(payload)
        if obj.get("error"):
            raise ValueError(f"server stream error: {obj['error']}")
        usage = obj.get("usage") or usage
        choices = obj.get("choices", [])
        if len(choices) > 1:
            raise ValueError("unexpected additional choice")
        for choice in choices:
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected additional choice")
            delta = choice.get("delta") or {}
            c = delta.get("content") or ""
            q = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (c or q) and first is None:
                first = time.perf_counter()
            content.append(c)
            reasoning.append(q)
            finish = choice.get("finish_reason") or finish
    if first is None or not done or finish is None:
        raise ValueError("missing visible output, finish reason or [DONE]")
    receipt = {
        "ttft_s": first - started,
        "elapsed_s": time.perf_counter() - started,
        "usage": usage,
        "finish_reason": finish,
        "stream_done": done,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        ),
        "output_sha256": hashlib.sha256(
            json.dumps(
                ["".join(content), "".join(reasoning)], ensure_ascii=False
            ).encode()
        ).hexdigest(),
    }
    validate_receipt(receipt, require_full_budget=require_full_budget)
    return receipt


def validate_receipt(receipt: dict, *, require_full_budget: bool = True) -> None:
    usage = receipt["usage"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int or usage[key] <= 0:
            raise ValueError(f"invalid usage {key}")
    if not 1 <= usage["completion_tokens"] <= 32 or receipt["finish_reason"] not in (
        "length",
        "stop",
    ):
        raise ValueError("invalid completion budget or finish reason")
    if require_full_budget and (
        usage["completion_tokens"] != 32 or receipt["finish_reason"] != "length"
    ):
        raise ValueError(
            "benchmark requires a complete 32-token length-capped response"
        )
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("inconsistent usage totals")
    cached = receipt["cached_tokens"]
    if type(cached) is not int or not 0 <= cached < usage["prompt_tokens"]:
        raise ValueError("invalid cached-token receipt")
    if cached != (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0):
        raise ValueError("cached-token receipt disagrees with usage")
    for key in ("ttft_s", "elapsed_s"):
        value = receipt[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid {key}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {key}")
    if receipt["ttft_s"] > receipt["elapsed_s"]:
        raise ValueError("TTFT exceeds elapsed time")
    if receipt.get("stream_done") is not True:
        raise ValueError("missing stream terminal marker")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt["output_sha256"]):
        raise ValueError("invalid output digest")


def same_output(a: dict, b: dict) -> bool:
    return (
        a["output_sha256"] == b["output_sha256"]
        and all(
            a["usage"][k] == b["usage"][k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        and a["finish_reason"] == b["finish_reason"]
    )


def controlled_environment(prefill: int, checkpoint: int) -> dict[str, str]:
    """Only launcher-controlled flags; never retain ambient credentials."""
    return {
        "RAPID_MLX_HYBRID_CHECKPOINT_MAX": str(checkpoint),
        "RAPID_MLX_HYBRID_CHECKPOINT_STRIDE": "2048",
        "RAPID_MLX_PREFIX_CACHE_AUTOLOAD": "0",
        "RAPID_MLX_DISABLE_DISK_CACHES": "1",
        "RAPID_MLX_GDN_PREFILL": str(prefill),
        "RAPID_MLX_QWEN35_FUSED_GDN_DECODE": "0",
        "RAPID_MLX_COMPILED_DECODE": "0",
        "RAPID_MLX_LANE_MATMUL": "off",
        "RAPID_MLX_PROMPT_HOST_CACHE": "0",
        "RAPID_MLX_PREFIX_CACHE_MAX_BYTES": str(2**31),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "RAPID_MLX_TELEMETRY": "0",
    }


def child_environment(prefill: int, checkpoint: int) -> dict[str, str]:
    """Keep native process essentials; exclude ambient inference/import knobs.

    An absolute Python executable and local snapshot make PATH, PYTHONPATH,
    virtualenv activation, model credentials and cache overrides unnecessary.
    HOME retains the single default HF cache; system temp/locale values are
    retained for macOS/Python runtime operation, never serialized as evidence.
    """
    env = {
        key: os.environ[key]
        for key in ("HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE")
        if key in os.environ
    }
    env.update(controlled_environment(prefill, checkpoint))
    return env


def prefill_evidence(log_text: str, prefill: int) -> list[str]:
    """Require the server's install/disable log, not an arm label alone."""
    lines = [line for line in log_text.splitlines() if "[gdn_prefill]" in line]
    installed = any("blocked-seq GDN prefill kernel installed" in x for x in lines)
    disabled = any("disabled via RAPID_MLX_GDN_PREFILL=0" in x for x in lines)
    if (installed, disabled) != (bool(prefill), not bool(prefill)):
        raise ValueError("server prefill evidence disagrees with arm")
    return lines


def summarize(result: dict) -> dict:
    """Join by experiment identity, never by row order or saved pass booleans."""
    rounds = result["rounds"]
    if type(rounds) is not int or rounds < 2:
        raise ValueError("at least two complete rounds required")
    contract = result["contract"]
    threshold = result["min_speedup"]
    if (
        contract not in ("incremental", "cold")
        or not math.isfinite(threshold)
        or threshold <= 1
    ):
        raise ValueError("invalid qualification contract or speed threshold")
    arms = result["arms"]
    if len(arms) != len(ARMS) or {a["name"] for a in arms} != ARMS:
        raise ValueError("missing or duplicate server arm")
    for arm in arms:
        if arm.get("error") or arm.get("server_exit") not in (0, -15):
            raise ValueError("incomplete or unclean server arm")
        if arm["name"] != f"prefill{arm['prefill']}-checkpoint{arm['checkpoint_max']}":
            raise ValueError("arm configuration disagrees with identity")
        if arm.get("controlled_env") != controlled_environment(
            arm["prefill"], arm["checkpoint_max"]
        ):
            raise ValueError("missing or inconsistent controlled environment")
        prefill_evidence("\n".join(arm.get("prefill_evidence", [])), arm["prefill"])
    expected = {(a, r, c) for a in ARMS for r in range(rounds) for c in CASES}
    rows = {}
    for row in result["rows"]:
        key = (row["arm"], row["round"], row["case"])
        if type(row["round"]) is not int or key not in expected or key in rows:
            raise ValueError("unexpected or duplicate experiment row")
        for phase in ("seed", "warm", "cold"):
            validate_receipt(row[phase])
        checkpoint_on = row["arm"].endswith("checkpoint4")
        for phase in ("seed", "warm", "cold"):
            wanted = CASES[row["case"]][1] if checkpoint_on and phase == "warm" else 0
            if row[phase]["cached_tokens"] != wanted:
                raise ValueError(
                    f"{key}: {phase} did not report expected cached tokens {wanted}"
                )
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("incomplete matrix")
    incremental, groups = [], []
    for prefill in (0, 1):
        for case in CASES:
            gains = []
            for rd in range(rounds):
                off = rows[(f"prefill{prefill}-checkpoint0", rd, case)]
                on = rows[(f"prefill{prefill}-checkpoint4", rd, case)]
                comparisons = {
                    phase + "_exact": same_output(off[phase], on[phase])
                    for phase in ("seed", "warm", "cold")
                }
                gain = off["warm"]["ttft_s"] / on["warm"]["ttft_s"]
                gains.append(gain)
                incremental.append(
                    {
                        "prefill": prefill,
                        "round": rd,
                        "case": case,
                        **comparisons,
                        "ttft_speedup": gain,
                    }
                )
            groups.append(
                {
                    "prefill": prefill,
                    "case": case,
                    "median_paired_speedup": statistics.median(gains),
                    "positive_pairs": sum(g > 1 for g in gains),
                    "pairs": rounds,
                }
            )
    incremental_ok = all(
        p[k] for p in incremental for k in ("seed_exact", "warm_exact", "cold_exact")
    )
    cold_exact = sum(same_output(r["warm"], r["cold"]) for r in rows.values())
    performance_ok = all(
        g["median_paired_speedup"] >= threshold
        for g in groups
        if g["case"] != "head_edit"
    )
    return {
        "contract": contract,
        "rows": len(rows),
        "incremental_pairs": incremental,
        "incremental_passed": incremental_ok,
        "cold_exact_pairs": cold_exact,
        "cold_total_pairs": len(rows),
        "cold_passed": cold_exact == len(rows),
        "groups": groups,
        "performance_passed": performance_ok,
        "passed": incremental_ok
        and performance_ok
        and (contract == "incremental" or cold_exact == len(rows)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8617)
    ap.add_argument("--contract", choices=("cold", "incremental"), default="cold")
    ap.add_argument("--min-speedup", type=float, default=1.1)
    args = ap.parse_args()
    if not args.model.is_dir() or args.rounds < 2:
        ap.error("local snapshot and >=2 rounds required")
    if not math.isfinite(args.min_speedup) or args.min_speedup <= 1:
        ap.error("--min-speedup must be finite and greater than 1")
    args.model = args.model.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        ap.error("output already exists; use a new campaign directory")
    if any(args.output.parent.iterdir()):
        ap.error("campaign directory must be empty to preserve earlier logs")
    # Fail before any HTTP request if another service already owns the port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    result = {
        "schema_version": 1,
        "contract": args.contract,
        "min_speedup": args.min_speedup,
        "model": str(args.model),
        "rounds": args.rounds,
        "provenance": {
            "engine_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "worktree_status": subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "mlx",
                    "mlx-lm",
                    "mlx-vlm",
                    "transformers",
                    "numpy",
                    "httpx",
                )
            },
        },
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

        def ask(messages, *, warmup=False):
            t = time.perf_counter()
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
                return stream_receipt(r.iter_lines(), t, require_full_budget=not warmup)

        # Reverse checkpoint order between prefill modes; each mode is
        # measured in one process order, so this is not within-mode counterbalancing.
        for prefill, checkpoint in [(0, 0), (0, 4), (1, 4), (1, 0)]:
            name = f"prefill{prefill}-checkpoint{checkpoint}"
            log = args.output.parent / (name + ".server.log")
            env = child_environment(prefill, checkpoint)
            controlled_env = controlled_environment(prefill, checkpoint)
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
                arm = {
                    "name": name,
                    "command": cmd,
                    "controlled_env": controlled_env,
                    "log": log.name,
                    "prefill": prefill,
                    "checkpoint_max": checkpoint,
                }
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
                    ask([{"role": "user", "content": "Say READY."}], warmup=True)
                    arm["prefill_evidence"] = prefill_evidence(log.read_text(), prefill)
                    save()
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

                        for case, (index, _) in CASES.items():
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
    result["summary"] = summarize(result)
    save()
    print(json.dumps({"event": "summary", **result["summary"]}), flush=True)
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
