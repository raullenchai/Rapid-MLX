# SPDX-License-Identifier: Apache-2.0
"""DDTree server — experimental single-user mode.

This mirrors the DFlash single-user boundary: ``--enable-ddtree`` bypasses
BatchedEngine and routes generation through the optional external
``dtree_mlx`` runtime. The MVP prioritizes a clean end-to-end validation
surface over breadth of OpenAI features.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import functools
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from vllm_mlx.api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
)

from .eligibility import DDTreeUnavailable, have_runtime
from .runtime import DDTreeRuntime, load_runtime

logger = logging.getLogger(__name__)

_ddtree_lock = asyncio.Lock()
_ddtree_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ddtree-worker"
)


@atexit.register
def _shutdown_ddtree_executor() -> None:
    _ddtree_executor.shutdown(wait=False, cancel_futures=True)


def _build_app(
    *,
    runtime: DDTreeRuntime,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    no_thinking: bool = False,
) -> FastAPI:
    app = FastAPI(title="Rapid-MLX (DDTree)")
    from ...middleware.exception_handlers import install_exception_handlers

    install_exception_handlers(app)
    if cors_origins:
        wildcard = "*" in cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=not wildcard,
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Rapid-MLX-Internal"],
            max_age=3600,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "ddtree",
            "mode": "experimental-single-user-serial",
            "drafter": runtime.drafter_repo,
            "speculative_tokens": runtime.speculative_tokens,
            "tree_budget": runtime.tree_budget,
        }

    @app.get("/v1/models")
    async def list_models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=served_model_name,
                    created=int(time.time()),
                    owned_by="rapid-mlx",
                )
            ]
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        _validate_request(request)
        prompt = _render_prompt(runtime, request, no_thinking=no_thinking)
        max_tokens = (
            request.max_tokens if request.max_tokens is not None else default_max_tokens
        )
        temperature = request.temperature if request.temperature is not None else 0.0

        if request.stream:
            return StreamingResponse(
                _stream_completion(
                    runtime=runtime,
                    prompt=prompt,
                    served_model_name=served_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                media_type="text/event-stream",
            )

        return await _non_stream_completion(
            runtime=runtime,
            prompt=prompt,
            served_model_name=served_model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return app


def _validate_request(request: ChatCompletionRequest) -> None:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if request.n is not None and request.n > 1:
        raise HTTPException(status_code=400, detail="n > 1 is not supported")
    if request.tools:
        raise HTTPException(
            status_code=400,
            detail=(
                "Tool calling is not supported in DDTree mode (MVP limitation). "
                "Restart without --enable-ddtree to use tools."
            ),
        )
    if request.logprobs:
        raise HTTPException(
            status_code=400,
            detail=(
                "logprobs is not supported in DDTree mode. Restart without "
                "--enable-ddtree."
            ),
        )
    if request.response_format is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "response_format (structured output) is not supported in "
                "DDTree mode. Restart without --enable-ddtree."
            ),
        )
    if request.top_p is not None and request.top_p != 1.0:
        raise HTTPException(
            status_code=400,
            detail="top_p is not supported in DDTree mode; use top_p=1.",
        )
    if request.stop is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "custom stop sequences are not supported in DDTree mode; "
                "the model's EOS/chat-template stops still apply."
            ),
        )


def _render_prompt(
    runtime: DDTreeRuntime,
    request: ChatCompletionRequest,
    *,
    no_thinking: bool = False,
) -> str:
    tokenizer = runtime.generator.target.tokenizer
    messages = []
    for m in request.messages:
        content = m.content
        if isinstance(content, list):
            text_pieces = []
            dropped_kinds: list[str] = []
            for part in content:
                part_type = part.type if hasattr(part, "type") else part.get("type", "")
                if part_type == "text":
                    text_pieces.append(
                        part.text if hasattr(part, "text") else part.get("text", "")
                    )
                elif part_type:
                    dropped_kinds.append(part_type)
            if dropped_kinds:
                logger.warning(
                    "DDTree server is text-only; dropped %d non-text content "
                    "part(s) of type(s) %s.",
                    len(dropped_kinds),
                    sorted(set(dropped_kinds)),
                )
            content = "".join(text_pieces)
        messages.append({"role": m.role, "content": content})

    enable_thinking = False if no_thinking else _extract_thinking_from_request(request)
    effective_thinking = True if enable_thinking is None else enable_thinking
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=effective_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def _extract_thinking_from_request(request: ChatCompletionRequest) -> bool | None:
    ctk = getattr(request, "chat_template_kwargs", None)
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        v = ctk["enable_thinking"]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            lowered = v.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
    return getattr(request, "enable_thinking", None)


def _run_generate(
    runtime: DDTreeRuntime,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
):
    return runtime.generator.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        speculative_tokens=runtime.speculative_tokens,
        decode_mode="dtree",
        tree_budget=runtime.tree_budget,
        verify_mode="parallel-greedy-argmax",
    )


def _usage_from_result(result) -> tuple[int, int]:
    metrics = getattr(result, "metrics", {}) or {}
    prompt_tokens = int(metrics.get("num_input_tokens", 0))
    completion_tokens = len(getattr(result, "generated_tokens", []) or [])
    return prompt_tokens, completion_tokens


def _finish_reason(result, max_tokens: int) -> str:
    _, completion_tokens = _usage_from_result(result)
    return "length" if completion_tokens >= max_tokens else "stop"


async def _stream_completion(
    *,
    runtime: DDTreeRuntime,
    prompt: str,
    served_model_name: str,
    max_tokens: int,
    temperature: float,
) -> AsyncIterator[bytes]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model_name,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first)}\n\n".encode()

    async with _ddtree_lock:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _ddtree_executor,
            functools.partial(
                _run_generate,
                runtime,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    if getattr(result, "text", ""):
        piece = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": result.text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(piece)}\n\n".encode()

    prompt_tokens, completion_tokens = _usage_from_result(result)
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": _finish_reason(result, max_tokens),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _non_stream_completion(
    *,
    runtime: DDTreeRuntime,
    prompt: str,
    served_model_name: str,
    max_tokens: int,
    temperature: float,
) -> ChatCompletionResponse:
    async with _ddtree_lock:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _ddtree_executor,
            functools.partial(
                _run_generate,
                runtime,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    created = int(time.time())
    prompt_tokens, completion_tokens = _usage_from_result(result)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=created,
        model=served_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(role="assistant", content=result.text),
                finish_reason=_finish_reason(result, max_tokens),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def run_ddtree_server(
    *,
    main_model_repo: str,
    drafter_repo: str,
    speculative_tokens: int,
    tree_budget: int,
    host: str,
    port: int,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    uvicorn_log_level: str,
    no_thinking: bool = False,
) -> None:
    if not have_runtime():
        raise RuntimeError(
            "DDTree server requires the experimental dtree-mlx runtime — install with "
            "``pip install 'dtree-mlx @ git+https://github.com/DrHB/dtree-mlx.git'``."
        )
    if not drafter_repo:
        raise DDTreeUnavailable(
            "DDTree requires a non-empty drafter_repo — pass a matching DFlash "
            "drafter HF path (e.g. 'z-lab/Qwen3.5-9B-DFlash')."
        )
    if speculative_tokens <= 0 or tree_budget <= 0:
        raise DDTreeUnavailable(
            "DDTree requires positive speculative_tokens and tree_budget values."
        )

    import uvicorn

    def _load_all():
        return load_runtime(
            main_model_repo=main_model_repo,
            drafter_repo=drafter_repo,
            speculative_tokens=speculative_tokens,
            tree_budget=tree_budget,
        )

    logger.info("DDTree: loading runtime for %s", main_model_repo)
    runtime = _ddtree_executor.submit(_load_all).result()

    app = _build_app(
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        no_thinking=no_thinking,
    )

    print()
    host_display = "localhost" if host == "0.0.0.0" else host
    print(f"  Ready: http://{host_display}:{port}/v1  (DDTree experimental mode)")
    print(f"  Docs:  http://{host_display}:{port}/docs")
    print()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )
