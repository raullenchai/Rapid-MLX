#!/usr/bin/env python3
"""Run paired real tasks through the DeepSeek V4.1 product generator."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tasks() -> list[tuple[str, str, int]]:
    context = "\n".join(
        f"Case {i}: request {i} used a cache key and returned a normal response."
        for i in range(90)
    )
    return [
        (
            "code",
            "Return only Python code defining merge_intervals(intervals), which merges "
            "overlapping integer intervals. Include two asserts for empty input and "
            "adjacent intervals.",
            160,
        ),
        (
            "reasoning",
            "A train travels 180 km at 60 km/h, waits 35 minutes, then travels "
            "120 km at 80 km/h. Explain the total elapsed time in hours and minutes.",
            128,
        ),
        (
            "structured",
            'Return only a JSON object with keys "risk", "cause", and "next_action". '
            "A checkout API returned HTTP 503 after its inventory dependency timed out.",
            128,
        ),
        (
            "chinese",
            "请用简洁的中文解释数据库事务的原子性，并给出一个转账失败时应该回滚的例子。",
            128,
        ),
        (
            "long_context_retrieval",
            "The incident report contains many routine records. The unique recovery "
            "code is ZEBRA-4417.\n"
            + context
            + "\nWhat is the unique recovery code? Answer with only the code.",
            48,
        ),
    ]


def render_prompt(user_prompt: str) -> str:
    return (
        "<｜begin▁of▁sentence｜><｜User｜>"
        + user_prompt
        + "<｜Assistant｜></think>"
    )


def task_success(task_id: str, reply: str) -> bool:
    if task_id == "code":
        code = reply.strip().removeprefix("```python").removesuffix("```").strip()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return any(isinstance(node, ast.FunctionDef) and node.name == "merge_intervals" for node in tree.body)
    if task_id == "reasoning":
        return bool(re.search(r"5\s*(hours?|h)\b", reply, re.I)) and bool(
            re.search(r"5\s*(minutes?|min|m)\b", reply, re.I)
        )
    if task_id == "structured":
        try:
            value = json.loads(reply)
        except ValueError:
            return False
        return isinstance(value, dict) and set(value) == {"risk", "cause", "next_action"}
    if task_id == "chinese":
        return "原子" in reply and ("回滚" in reply or "撤销" in reply)
    if task_id == "long_context_retrieval":
        return reply.strip() == "ZEBRA-4417"
    return False


def host_gate(min_available_gib: float) -> dict[str, float]:
    import psutil

    available = psutil.virtual_memory().available / 1024**3
    swap = psutil.swap_memory().used / 1024**3
    facts = {"available_gib": round(available, 2), "used_swap_gib": round(swap, 2)}
    if available < min_available_gib or swap > 0:
        raise RuntimeError(
            f"unsafe large-model test host: {facts}; need at least "
            f"{min_available_gib:.0f} GiB available and zero used swap"
        )
    return facts


def set_page_read_ahead(model, enabled: bool) -> int:
    from rapid_mlx.models.deepseek_v41_native.engram import (
        DiskQuantizedEngramEmbedding,
    )

    count = 0
    for layer in model.layers:
        embedding = getattr(getattr(layer, "engram", None), "embed", None)
        if isinstance(embedding, DiskQuantizedEngramEmbedding):
            embedding.read_ahead = enabled
            count += 1
    if count != 2:
        raise RuntimeError(f"expected two SSD-backed Engram tables; found {count}")
    return count


def generate_task(model, tokenizer, runtime, prompt: str, limit: int) -> dict:
    from rapid_mlx.models.deepseek_v41_native.serving import stream_generate

    started = time.perf_counter()
    first = None
    text = []
    tokens = []
    for chunk in stream_generate(
        model, tokenizer, prompt, runtime=runtime, max_tokens=limit
    ):
        if first is None:
            first = time.perf_counter() - started
        text.append(chunk.text)
        tokens.append(chunk.token)
    elapsed = time.perf_counter() - started
    reply = "".join(text)
    return {
        "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "completion_tokens": len(tokens),
        "ttft_seconds": round(first, 3) if first is not None else None,
        "total_seconds": round(elapsed, 3),
        "output_sha256": hashlib.sha256(
            b"".join(int(token).to_bytes(4, "little") for token in tokens)
        ).hexdigest(),
        "reply": reply,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-available-gib", type=float, default=215)
    args = parser.parse_args()
    try:
        facts = host_gate(args.min_available_gib)
    except RuntimeError as exc:
        print(json.dumps({"event": "host_gate_failed", "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps({"event": "host_gate_passed", **facts}), flush=True)

    from rapid_mlx.models.deepseek_v41_native.artifacts import (
        MTP_REPO,
        MTP_REVISION,
        TARGET_REPO,
        TARGET_REVISION,
        verify_mtp_snapshot,
    )
    from rapid_mlx.models.deepseek_v41_native.serving import load_product_runtime

    hub = Path.home() / ".cache/huggingface/hub"

    def local_snapshot(repo: str, revision: str) -> Path:
        path = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"pinned snapshot missing from default cache: {path}")
        return path

    target = local_snapshot(TARGET_REPO, TARGET_REVISION)
    mtp = verify_mtp_snapshot(local_snapshot(MTP_REPO, MTP_REVISION))
    model, tokenizer, runtime = load_product_runtime(
        str(target),
        str(mtp),
        target_revision=TARGET_REVISION,
        mtp_revision=MTP_REVISION,
    )
    results = []
    for index, (task_id, user_prompt, limit) in enumerate(tasks()):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair = []
        for enabled in order:
            set_page_read_ahead(model, enabled)
            result = generate_task(
                model, tokenizer, runtime, render_prompt(user_prompt), limit
            )
            result.update(
                task_id=task_id,
                page_read_ahead=enabled,
                task_success=task_success(task_id, result["reply"]),
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            pair.append(result)
            results.append(result)
        if pair[0]["output_sha256"] != pair[1]["output_sha256"]:
            raise RuntimeError(f"token mismatch for task {task_id}")
        print(json.dumps({"event": "task_pair_equal", "task_id": task_id}), flush=True)
    summary = {
        "event": "dogfood_summary",
        "task_count": len(tasks()),
        "task_pass_count": sum(
            result["task_success"] for result in results if not result["page_read_ahead"]
        ),
        "paired_outputs_equal": True,
        "median_ttft_seconds": {
            str(enabled): statistics.median(
                result["ttft_seconds"]
                for result in results
                if result["page_read_ahead"] == enabled
            )
            for enabled in (False, True)
        },
        "timing_limit": "paired prompts warm file pages; use cold-prompt capture for speed claims",
    }
    print(json.dumps(summary), flush=True)
    if summary["task_pass_count"] != summary["task_count"]:
        raise SystemExit("one or more real-task checks failed")


if __name__ == "__main__":
    main()
