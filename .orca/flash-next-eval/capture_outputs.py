#!/usr/bin/env python3
"""Capture deterministic Flash-Next outputs for baseline/MTP byte comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_service_prefill import (  # noqa: E402
    clear_prefix_cache,
    make_messages,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8465/v1")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    client = httpx.Client(timeout=3600)
    api = args.url.rstrip("/")
    root = api.removesuffix("/v1")
    model = client.get(f"{api}/models").json()["data"][0]["id"]
    captures = []
    for target in (128, 2048):
        messages = make_messages(
            tokenizer,
            target,
            suffix="\nReply with a detailed but direct explanation.",
        )
        clear_prefix_cache(client, root)
        response = client.post(
            f"{api}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.0,
                "enable_thinking": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"].get("content") or ""
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        captures.append(
            {
                "target_prompt_tokens": target,
                "finish_reason": body["choices"][0]["finish_reason"],
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "content": content,
                "token_ids": token_ids,
                "usage": body.get("usage"),
            }
        )
    args.output.write_text(
        json.dumps({"label": args.label, "captures": captures}, indent=2) + "\n"
    )
    for item in captures:
        print(
            item["target_prompt_tokens"],
            item["finish_reason"],
            item["usage"],
            item["content_sha256"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
