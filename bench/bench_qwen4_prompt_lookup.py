#!/usr/bin/env python3
"""Measure prompt-lookup decoding on realistic high-overlap editing tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def _english_source() -> str:
    return "\n\n".join(
        f"def normalize_{index:03d}(value: str) -> str:\n"
        f"    # Stable transformation {index:03d}.\n"
        "    return value.strip().lower()"
        for index in range(28)
    )


def _json_manifest() -> str:
    return json.dumps(
        {
            "services": [
                {
                    "name": f"worker-{index:03d}",
                    "enabled": True,
                    "retries": 3,
                    "region": "us-west",
                }
                for index in range(22)
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _chinese_document() -> str:
    return "\n".join(
        f"第{index:03d}条：服务必须记录请求编号，并在失败后保留审计信息。"
        for index in range(1, 41)
    )


def scenarios() -> dict[str, dict[str, Any]]:
    source = _english_source()
    manifest = _json_manifest()
    zh_doc = _chinese_document()
    return {
        "code_copy": {
            "messages": [
                {"role": "system", "content": "Output only the requested file."},
                {
                    "role": "user",
                    "content": f"Return this file verbatim, without fences.\nBEGIN\n{source}\nEND",
                },
            ],
            "expected": source,
            "validation": "exact",
            "max_tokens": 1024,
        },
        "code_edit": {
            "messages": [
                {"role": "system", "content": "Output only the complete updated file."},
                {
                    "role": "user",
                    "content": (
                        "Change only transformation 000 from lower() to upper(); preserve "
                        f"everything else exactly.\nBEGIN\n{source}\nEND"
                    ),
                },
            ],
            "expected": source.replace(
                "# Stable transformation 000.\n    return value.strip().lower()",
                "# Stable transformation 000.\n    return value.strip().upper()",
                1,
            ),
            "validation": "code",
            "max_tokens": 1024,
        },
        "json_copy": {
            "messages": [
                {"role": "system", "content": "Output only valid JSON."},
                {
                    "role": "user",
                    "content": f"Return this JSON verbatim.\nBEGIN\n{manifest}\nEND",
                },
            ],
            "expected": manifest,
            "validation": "json",
            "max_tokens": 1024,
        },
        "chinese_copy": {
            "messages": [
                {"role": "system", "content": "只输出要求的完整文档。"},
                {
                    "role": "user",
                    "content": f"逐字返回以下文档，不要添加代码围栏。\n开始\n{zh_doc}\n结束",
                },
            ],
            "expected": zh_doc,
            "validation": "chinese_document",
            "max_tokens": 1024,
        },
        "multi_turn_edit": {
            "messages": [
                {"role": "system", "content": "Output only the complete updated file."},
                {"role": "user", "content": "Remember this file for the next edit."},
                {"role": "assistant", "content": source},
                {
                    "role": "user",
                    "content": "Change only transformation 000 from lower() to upper().",
                },
            ],
            "expected": source.replace(
                "# Stable transformation 000.\n    return value.strip().lower()",
                "# Stable transformation 000.\n    return value.strip().upper()",
                1,
            ),
            "validation": "code",
            "max_tokens": 1024,
        },
    }


def stream_request(
    client: httpx.Client,
    url: str,
    model: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    first = None
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    with client.stream(
        "POST",
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": case["messages"],
            "max_tokens": case["max_tokens"],
            "temperature": 0.0,
            "enable_thinking": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            visible = (choices[0].get("delta") or {}).get("content") or ""
            if visible:
                first = first or time.perf_counter()
                chunks.append(visible)
    finished = time.perf_counter()
    text = "".join(chunks)
    expected = case["expected"]
    comparable = text
    if case["validation"] == "code" and text.startswith("```python\n"):
        comparable = text.removeprefix("```python\n").removesuffix("\n```")
    elif case["validation"] == "chinese_document":
        comparable = text.removeprefix("开始\n").removesuffix("\n结束")
    if case["validation"] == "json":
        try:
            valid = json.loads(text) == json.loads(expected)
        except json.JSONDecodeError:
            valid = False
    else:
        valid = comparable == expected
    decode_seconds = max(finished - (first or finished), 1e-9)
    return {
        "exact": text == expected,
        "valid": valid,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_ms": round(((first or finished) - started) * 1000, 3),
        "total_ms": round((finished - started) * 1000, 3),
        "decode_tokens_per_second": round(
            int(usage.get("completion_tokens") or 0) / decode_seconds, 3
        ),
        "first_difference": next(
            (
                index
                for index, (actual, wanted) in enumerate(zip(text, expected))
                if actual != wanted
            ),
            None if len(text) == len(expected) else min(len(text), len(expected)),
        ),
        "actual_length": len(text),
        "expected_length": len(expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8465")
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = scenarios()
    results = {}
    with httpx.Client(timeout=3600) as client:
        model = client.get(f"{args.url}/v1/models").json()["data"][0]["id"]
        for name, case in cases.items():
            rows = []
            for run in range(1, args.runs + 1):
                client.post(f"{args.url}/v1/cache/clear").raise_for_status()
                row = stream_request(client, args.url, model, case)
                row["run"] = run
                rows.append(row)
                print(
                    f"{name} run={run} valid={row['valid']} exact={row['exact']} "
                    f"decode={row['decode_tokens_per_second']:.2f} tok/s "
                    f"tokens={row['completion_tokens']} diff={row['first_difference']}"
                )
            results[name] = {
                "all_valid": all(row["valid"] for row in rows),
                "all_exact": all(row["exact"] for row in rows),
                "median_decode_tokens_per_second": statistics.median(
                    row["decode_tokens_per_second"] for row in rows
                ),
                "rows": rows,
            }
    args.output.write_text(
        json.dumps({"label": args.label, "model": model, "results": results}, indent=2)
        + "\n"
    )
    return 0 if all(row["all_valid"] for row in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
