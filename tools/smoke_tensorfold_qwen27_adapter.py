#!/usr/bin/env python3
"""Private direct smoke for the experimental Qwen3.8-27B adapter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import time

adapter_path = os.environ.get("RAPID_TF_ADAPTER_PATH")
if adapter_path:
    spec = importlib.util.spec_from_file_location("rapid_tensorfold_qwen27_smoke", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    TensorFoldQwen27Backend = module.TensorFoldQwen27Backend
else:
    from rapid_mlx.speculative.tensorfold_qwen27 import TensorFoldQwen27Backend


async def one(backend, request_id: str, prompt: list[int], draft: bool) -> dict:
    deltas = []
    terminal = None
    started = time.perf_counter()
    first = None
    async for event in backend.stream(
        request_id,
        prompt,
        max_tokens=128,
        sampling={"temperature": 0.0, "seed": 1234, "draft": draft},
    ):
        if event.delta is not None:
            first = first or time.perf_counter()
            deltas.append(event.delta)
        if event.error is not None:
            raise event.error
        if event.reply is not None:
            terminal = event.reply
    if terminal is None:
        raise RuntimeError("adapter emitted no terminal reply")
    content = terminal["content"]
    return {
        "draft": draft,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "finish_reason": terminal["finish_reason"],
        "prompt_tokens": terminal["prompt_tokens"],
        "completion_tokens": terminal["completion_tokens"],
        "cached_tokens": terminal["cached_tokens"],
        "delta_count": len(deltas),
        "speculative": terminal["speculative"],
        "ttft_seconds": None if first is None else first - started,
        "total_seconds": time.perf_counter() - started,
    }


async def cancel_once(backend, prompt: list[int]) -> dict:
    saw_delta = False
    terminal_error = None
    async for event in backend.stream(
        "cancel", prompt, max_tokens=512,
        sampling={"temperature": 0.0, "seed": 1234, "draft": True},
    ):
        if event.delta is not None and not saw_delta:
            saw_delta = True
            backend.cancel("cancel")
        if event.error is not None:
            terminal_error = type(event.error).__name__
    return {"saw_delta": saw_delta, "terminal_error": terminal_error,
            "active_after": sorted(backend._active)}


async def main(args) -> int:
    backend = TensorFoldQwen27Backend.load(
        args.target, args.drafter, served_name="qwen27-adapter-smoke",
        context_window=2048, max_tokens=128,
    )
    try:
        tokenizer = backend._app.tokenizer
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        serial = await one(backend, "serial", list(prompt), False)
        drafted = await one(backend, "drafted", list(prompt), True)
        cancelled = await cancel_once(backend, list(prompt))
    finally:
        await asyncio.get_running_loop().run_in_executor(None, backend.close)
    exact = all(serial[key] == drafted[key] for key in (
        "content_sha256", "finish_reason", "completion_tokens"
    ))
    passed = (exact and serial["finish_reason"] == "stop"
              and (drafted["speculative"] or {}).get("drafted", 0) > 0
              and cancelled["saw_delta"] and cancelled["terminal_error"]
              and not cancelled["active_after"])
    print(json.dumps({"passed": passed, "exact": exact, "serial": serial,
                      "drafted": drafted, "cancelled": cancelled}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--prompt", default="Reply with exactly these three words: adapter smoke passed")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
