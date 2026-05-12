# SPDX-License-Identifier: Apache-2.0
"""DFlash server — dedicated single-user mode that bypasses BatchedEngine.

When ``--enable-dflash`` is set, the CLI launches this server instead of
the standard ``vllm_mlx.server.app``. It hosts a minimal OpenAI-compatible
surface (``/healthz``, ``/v1/models``, ``/v1/chat/completions``) and routes
generation through mlx-vlm's ``stream_generate`` with the loaded DFlash
drafter.

Why a separate server (not a fork of the standard route)?
  - mlx-vlm's ``generate_step`` is a per-request Python generator with its
    own ``prompt_cache`` argument. BatchedEngine merges per-request KV
    caches into a ``BatchKVCache``. Grafting one onto the other would
    invent batched-DFlash that doesn't exist upstream and would risk
    regressing the non-DFlash path under attention layout changes.
  - DFlash today only validates on B=1 anyway (see PoC: 1.83-2.18× on
    Qwen3.5-27B-8bit; no batched-DFlash kernel exists in mlx-vlm 0.5.0).
  - A separate, opt-in server is a clean blast-radius boundary: turning
    on DFlash can never break a request that doesn't use it.

v1 limitations (documented in README + ``rapid-mlx info``):
  - Single-user serial. Concurrent requests queue on an ``asyncio.Lock``.
  - No tool calling, MCP, embeddings, or audio in this server (the
    standard server handles those).
  - No prefix cache (per-request KV cache built fresh each call).

These limitations are deliberate for v1 — the target user is someone
running ``rapid-mlx serve qwen3.5-27b-8bit --enable-dflash`` to get a
~2× speedup on code/long-form completions on a single Apple Silicon box.
"""

from __future__ import annotations

import asyncio
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

from .eligibility import have_runtime
from .runtime import DFlashRuntime, load_runtime

logger = logging.getLogger(__name__)


# Global serial lock — DFlash is single-stream by design (mlx-vlm doesn't
# expose a batched DFlash kernel in 0.5.0). The second concurrent request
# waits its turn; this matches the PoC reality.
_dflash_lock = asyncio.Lock()


def _build_app(
    *,
    model: Any,
    processor: Any,
    runtime: DFlashRuntime,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
) -> FastAPI:
    """Create the FastAPI application for DFlash mode.

    All globals (``model``, ``processor``, ``runtime``) are captured by
    closure here rather than module-level so a future caller could host
    multiple DFlash instances side-by-side without state collision.
    """
    app = FastAPI(title="rapid-mlx (DFlash)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "dflash",
            "mode": "single-user-serial",
            "drafter": runtime.drafter_repo,
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
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if request.n is not None and request.n > 1:
            raise HTTPException(status_code=400, detail="n > 1 is not supported")
        if request.tools:
            # DFlash server doesn't run a tool-call parser. Surface this so
            # users don't think their tools "silently worked" when in fact
            # the model just emitted free-form text.
            raise HTTPException(
                status_code=400,
                detail=(
                    "Tool calling is not supported in DFlash mode (v1 "
                    "limitation). Restart without --enable-dflash to use "
                    "tools."
                ),
            )

        # Render chat messages into a single prompt string via mlx-vlm's
        # processor. We pass through the model's chat template so the
        # tokenizer-side reasoning/tool markers match what the model was
        # trained on; no rapid-mlx-side prompt mutation happens here.
        prompt = _render_prompt(processor, model, request)

        max_tokens = (
            request.max_tokens if request.max_tokens is not None else default_max_tokens
        )
        temperature = request.temperature if request.temperature is not None else 0.0
        top_p = request.top_p if request.top_p is not None else 1.0

        gen_kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            draft_model=runtime.drafter,
            draft_kind=runtime.kind,
        )

        if request.stream:
            return StreamingResponse(
                _stream_completion(
                    prompt=prompt,
                    request=request,
                    served_model_name=served_model_name,
                    gen_kwargs=gen_kwargs,
                    model=model,
                    processor=processor,
                ),
                media_type="text/event-stream",
            )

        return await _non_stream_completion(
            prompt=prompt,
            request=request,
            served_model_name=served_model_name,
            gen_kwargs=gen_kwargs,
            model=model,
            processor=processor,
        )

    return app


def _render_prompt(processor: Any, model: Any, request: ChatCompletionRequest) -> str:
    """Apply the model's chat template via mlx-vlm's helper.

    mlx-vlm's ``apply_chat_template`` mirrors mlx-lm's but accepts the
    multimodal kwargs the VLM models need (we pass ``num_images=0`` since
    DFlash-eligible aliases are text-only Qwen3.5/3.6 variants today).
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = []
    for m in request.messages:
        content = m.content
        if isinstance(content, list):
            # Multimodal payload — DFlash server is text-only. Collapse
            # text parts; ignore image/video parts (a 400 might surprise
            # users sending mixed content; better to silently degrade).
            text_pieces = []
            for part in content:
                part_type = part.type if hasattr(part, "type") else part.get("type", "")
                if part_type == "text":
                    text_pieces.append(
                        part.text if hasattr(part, "text") else part.get("text", "")
                    )
            content = "".join(text_pieces)
        messages.append({"role": m.role, "content": content})

    return apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=0,
        num_audios=0,
        enable_thinking=True,
    )


async def _stream_completion(
    *,
    prompt: str,
    request: ChatCompletionRequest,
    served_model_name: str,
    gen_kwargs: dict[str, Any],
    model: Any,
    processor: Any,
) -> AsyncIterator[bytes]:
    """Stream OpenAI-format chunks. Generation happens under the serial
    lock; chunks are forwarded as ``data: ...\\n\\n`` SSE events."""
    from mlx_vlm import stream_generate

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # First chunk — role marker
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

    finish_reason = "stop"
    total_completion_tokens = 0
    prompt_tokens = 0

    async with _dflash_lock:
        # mlx-vlm's stream_generate is a sync generator — run it in a
        # thread pool so we don't block the FastAPI event loop. Iterate
        # by polling with ``run_in_executor`` per chunk.
        loop = asyncio.get_event_loop()
        gen = stream_generate(model, processor, prompt, **gen_kwargs)

        def _next_chunk():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                break
            if not chunk.text:
                continue
            total_completion_tokens = chunk.generation_tokens
            prompt_tokens = chunk.prompt_tokens
            piece = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(piece)}\n\n".encode()

    # Final chunk — finish_reason + usage
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": served_model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }
    yield f"data: {json.dumps(final)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _non_stream_completion(
    *,
    prompt: str,
    request: ChatCompletionRequest,
    served_model_name: str,
    gen_kwargs: dict[str, Any],
    model: Any,
    processor: Any,
) -> ChatCompletionResponse:
    """Run generation under the serial lock, return one
    ``ChatCompletionResponse`` containing the full text."""
    from mlx_vlm import generate

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    async with _dflash_lock:
        loop = asyncio.get_event_loop()
        # mlx-vlm's ``generate`` blocks; offload to thread to keep the
        # FastAPI event loop free for concurrent (queued) requests.
        result = await loop.run_in_executor(
            None,
            lambda: generate(model, processor, prompt, **gen_kwargs),
        )

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=served_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(role="assistant", content=result.text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.generation_tokens,
            total_tokens=result.prompt_tokens + result.generation_tokens,
        ),
    )


def run_dflash_server(
    *,
    main_model_repo: str,
    drafter_repo: str,
    host: str,
    port: int,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    uvicorn_log_level: str,
) -> None:
    """Load the model + DFlash drafter via mlx-vlm and start uvicorn.

    The mlx-vlm load path is mandatory: the DFlash hooks
    (``capture_layer_ids``, ``_dflash_rounds``) live on the mlx-vlm
    model classes, not mlx-lm's. Loading via ``mlx_lm.load`` would give
    us a model without the hooks and DFlash would silently fall back to
    AR — exactly the kind of "silent regression" the eligibility gate
    is meant to prevent. We surface a clear error if mlx-vlm is missing
    or too old.
    """
    if not have_runtime():
        raise RuntimeError(
            "DFlash server requires mlx-vlm 0.5.0+ — install with "
            "``pip install 'rapid-mlx[dflash]'``."
        )

    import uvicorn
    from mlx_vlm import load

    logger.info("DFlash: loading main model via mlx-vlm: %s", main_model_repo)
    t0 = time.perf_counter()
    model, processor = load(main_model_repo)
    logger.info("DFlash: main model loaded in %.1fs", time.perf_counter() - t0)

    runtime = load_runtime(drafter_repo)

    app = _build_app(
        model=model,
        processor=processor,
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
    )

    print()
    host_display = "localhost" if host == "0.0.0.0" else host
    print(f"  Ready: http://{host_display}:{port}/v1  (DFlash mode)")
    print(f"  Docs:  http://{host_display}:{port}/docs")
    print()

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        timeout_keep_alive=30,
    )
