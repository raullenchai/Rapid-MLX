"""Qualify the Nemotron provider's long-chat cache against a cache-off run.

Run each cache mode in its own process under scripts/large-model-run.py.
Uses the qualified local HF snapshot only; never downloads weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

PROFILE = "nemotron-3.5-lightning-tensorfold"
ROOT = Path(__file__).resolve().parents[1]


def qualification_identity() -> dict:
    from rapid_mlx.speculative.tensorfold_families import PROFILES
    from rapid_mlx.speculative.tensorfold_runtime import (
        SUPPORTED_MLX_VERSION,
        SUPPORTED_REVISION,
        SUPPORTED_VERSION,
    )

    return {
        "profile": PROFILE,
        "revision": PROFILES[PROFILE].target_revision,
        "runtime": SUPPORTED_VERSION,
        "runtime_revision": SUPPORTED_REVISION,
        "mlx": SUPPORTED_MLX_VERSION,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def source_identity() -> dict:
    """Bind an untracked MVP probe and clean tracked serving code to a run."""
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "rapid_mlx"], cwd=ROOT
    ).returncode:
        raise RuntimeError(
            "commit serving changes before capturing qualification evidence"
        )
    return {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def serving_tree(commit: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{commit}:rapid_mlx"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("unresolvable serving source commit") from exc


def measure(provider, messages: list[dict[str, str]]) -> dict:
    """Measure one fully drained provider request, including its terminal usage."""
    app = provider.backend._app
    scheduler = app.scheduler
    original = scheduler.submit
    jobs = []

    def submit(job):
        jobs.append(job)
        return original(job)

    prompt = app.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    provider.last_outputs = []
    scheduler.submit = submit
    started = time.perf_counter()
    first = None
    try:
        for output in provider._outputs(
            prompt,
            messages=messages,
            enable_thinking=False,
            temperature=0,
            max_tokens=32,
        ):
            if first is None and output.new_token_ids:
                first = time.perf_counter() - started
        elapsed = time.perf_counter() - started
    finally:
        scheduler.submit = original
    if len(jobs) != 1 or first is None:
        raise RuntimeError(
            "request produced no tokens or did not submit exactly one job"
        )
    terminal = provider.last_outputs[-1] if provider.last_outputs else None
    if (
        terminal is None
        or not terminal.finished
        or not terminal.output_token_ids
        or not terminal.output_text.strip()
    ):
        raise RuntimeError("request ended without a completed token sequence")
    job = jobs[0]
    prefill = job.prefilled_at - job.started_at
    if prefill <= 0 or not 0 <= terminal.cached_tokens < terminal.prompt_tokens:
        raise RuntimeError("invalid timing or cached-token count")
    return {
        "request_sha256": digest(messages),
        "prompt_tokens": terminal.prompt_tokens,
        "cached_tokens": terminal.cached_tokens,
        "token_ids": terminal.output_token_ids,
        "answer": terminal.output_text,
        "prefill_seconds": prefill,
        "ttft_seconds": first,
        "total_seconds": elapsed,
    }


def conversation(provider) -> list[dict]:
    context = "\n".join(
        f"Record {i}: service worker_{i} has queue capacity {i % 19 + 1}, "
        f"retry limit {i % 5 + 1}, and owner team_{i % 23}."
        for i in range(650)
    )
    system = {
        "role": "system",
        "content": "Read the supplied service inventory. Answer briefly, without reasoning.",
    }
    messages = [
        system,
        {
            "role": "user",
            "content": context + "\nAcknowledge the inventory in one sentence.",
        },
    ]
    rows = []

    def run(case, history):
        row = measure(provider, history)
        row["case"] = case
        rows.append(row)
        return [*history, {"role": "assistant", "content": row["answer"]}]

    first = run("initial", messages)
    second = run("continue", [*first, _question(17)])
    run("continue_again", [*second, _question(34)])
    edited = [*second, _question(35)]
    run("edit", edited)
    run("regenerate", edited)
    run("branch", [*first, _question(51)])
    changed = [
        system,
        {
            "role": "user",
            "content": "A different inventory: worker_17 has retry limit 99.",
        },
        _question(17),
    ]
    run("different_inventory", changed)
    return rows


def _question(worker: int) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"What is the retry limit for worker_{worker}? Reply with the number.",
    }


def compare(warm: dict, cold: dict) -> None:
    """Fail on mismatches or absent reuse; speed is reported, never a noisy gate."""
    for key, expected in qualification_identity().items():
        if warm[key] != expected or cold[key] != expected:
            raise ValueError(f"unqualified provenance: {key}")
    # Evidence survives docs-only commits, but cannot qualify changed serving code.
    expected_tree = serving_tree(source_identity()["source_commit"])
    for artifact in (warm, cold):
        if serving_tree(artifact["source_commit"]) != expected_tree:
            raise ValueError("artifact does not qualify current serving sources")
    for key in (
        "profile",
        "revision",
        "runtime",
        "mlx",
        "source_commit",
        "probe_sha256",
        "host",
    ):
        if warm[key] != cold[key]:
            raise ValueError(f"different provenance: {key}")
    if warm["cache"] != "on" or cold["cache"] != "off":
        raise ValueError("expected cache-on and cache-off artifacts")
    cases = [
        "initial",
        "continue",
        "continue_again",
        "edit",
        "regenerate",
        "branch",
        "different_inventory",
    ]
    if [r["case"] for r in warm["rows"]] != cases or [
        r["case"] for r in cold["rows"]
    ] != cases:
        raise ValueError("missing or reordered cases")
    for a, b in zip(warm["rows"], cold["rows"], strict=True):
        case = a["case"]
        if (
            not a["token_ids"]
            or not a["answer"].strip()
            or not 0 <= a["cached_tokens"] < a["prompt_tokens"]
        ):
            raise ValueError(f"invalid tokens or cached-token count: {case}")
        if any(
            a[k] != b[k]
            for k in ("request_sha256", "prompt_tokens", "token_ids", "answer")
        ):
            raise ValueError(f"cache-on/off mismatch: {case}")
        if b["cached_tokens"] != 0:
            raise ValueError(f"cache-off run reused state: {case}")
        if case in cases[1:6] and a["cached_tokens"] / a["prompt_tokens"] < 0.95:
            raise ValueError(f"insufficient long-history reuse: {case}")
        # Retain the long input; checkpoint boundaries may exclude a few tokens.
        if case in cases[1:6] and a["prompt_tokens"] < warm["rows"][0]["prompt_tokens"]:
            raise ValueError(f"resumed prompt lost its long history: {case}")
    if warm["rows"][0]["cached_tokens"] != 0:
        raise ValueError("initial request was not cold")
    if warm["rows"][0]["prompt_tokens"] < 19000:
        raise ValueError("fixture did not exercise a long chat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", choices=("on", "off"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("ON", "OFF"))
    args = parser.parse_args()
    if args.compare:
        compare(*(json.loads(p.read_text()) for p in args.compare))
        print("PASS: all seven cases agree; all five resumed cases reuse state")
        return
    if args.cache is None or args.output is None:
        parser.error("--cache and --output are required for a model run")
    source = source_identity()
    from huggingface_hub import snapshot_download

    from rapid_mlx.speculative.tensorfold_families import PROFILES, backend_class_for
    from rapid_mlx.speculative.tensorfold_qwen27_server import TensorFoldRequestProvider

    profile = PROFILES[PROFILE]
    target = snapshot_download(
        profile.target, revision=profile.target_revision, local_files_only=True
    )
    backend = backend_class_for(profile).load(
        target, served_name=PROFILE, context_window=32768, max_tokens=32
    )
    try:
        if args.cache == "off":
            backend._app.scheduler.checkpoints = None
        rows = conversation(TensorFoldRequestProvider(backend))
    finally:
        backend.close()
    artifact = {
        **qualification_identity(),
        "runtime": importlib.metadata.version("tensorfold"),
        "mlx": importlib.metadata.version("mlx"),
        **source,
        "host": {
            "platform": platform.platform(),
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "ram_bytes": int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ),
        },
        "cache": args.cache,
        "rows": rows,
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    for row in rows:
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("token_ids", "answer")}
            )
        )


if __name__ == "__main__":
    main()
