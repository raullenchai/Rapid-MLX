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
import atexit
import concurrent.futures
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


# Dedicated single-thread executor so every mlx-vlm call (drafter loading,
# generate, stream_generate's ``next``) executes on ONE thread for the
# lifetime of the process. Reason: mlx-lm 0.31.3+ keeps the GPU Stream
# in thread-local storage; iterating a generator across threads (which
# would happen if we used the default ThreadPoolExecutor with N workers)
# trips "There is no Stream(gpu, N) in current thread" mid-stream. Pinning
# to one worker preserves thread affinity and matches the serial-only
# contract enforced by ``_dflash_lock``.
_dflash_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="dflash-worker"
)


@atexit.register
def _shutdown_dflash_executor() -> None:
    """Drain the DFlash worker on interpreter exit. Python registers an
    implicit atexit for ThreadPoolExecutor, but registering ours
    explicitly makes shutdown order deterministic and silences
    "unfinished thread" warnings during graceful uvicorn termination."""
    _dflash_executor.shutdown(wait=False, cancel_futures=True)


def _build_app(
    *,
    model: Any,
    processor: Any,
    runtime: DFlashRuntime,
    served_model_name: str,
    default_max_tokens: int,
    cors_origins: list[str],
    no_thinking: bool = False,
) -> FastAPI:
    """Create the FastAPI application for DFlash mode.

    Per-app state (``model``, ``processor``, ``runtime``,
    ``served_model_name``) is captured by closure so a single Python
    process could in principle host multiple ``_build_app`` instances
    against different models without per-request state collision.

    Note: ``_dflash_lock`` and ``_dflash_executor`` are *module-level*
    by design — every DFlash invocation must serialise through the
    same single-thread worker because mlx's GPU Stream is thread-local
    (see the ``_dflash_executor`` docstring at module top). A future
    multi-model deployment would still share that worker; one model
    can't run while another's generator is mid-step.
    """
    app = FastAPI(title="Rapid-MLX (DFlash)")
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
        # Surface unsupported params explicitly rather than silently
        # ignoring — silent-drop is the bug class that makes users think
        # they got logprobs / JSON-schema / etc. when they didn't.
        if request.logprobs:
            raise HTTPException(
                status_code=400,
                detail=(
                    "logprobs is not supported in DFlash mode. Restart "
                    "without --enable-dflash."
                ),
            )
        if request.response_format is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format (structured output) is not supported "
                    "in DFlash mode. Restart without --enable-dflash."
                ),
            )

        # Render chat messages into a single prompt string via mlx-vlm's
        # processor. We pass through the model's chat template so the
        # tokenizer-side reasoning/tool markers match what the model was
        # trained on; no rapid-mlx-side prompt mutation happens here.
        #
        # Resolve enable_thinking (#387). The dflash app captures its own
        # ``no_thinking`` by closure rather than going through the
        # ServerConfig singleton, so we apply that override first then
        # delegate the request-side precedence (chat_template_kwargs >
        # request.enable_thinking > None) to the shared extractor — same
        # source of truth as the OpenAI/anthropic helper, but without the
        # ``cfg.no_thinking`` consult that doesn't apply to dflash.
        from ...service.helpers import _extract_thinking_from_request

        if no_thinking:
            enable_thinking: bool | None = False
        else:
            enable_thinking = _extract_thinking_from_request(request)
        prompt = _render_prompt(
            processor, model, request, enable_thinking=enable_thinking
        )

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


def _render_prompt(
    processor: Any,
    model: Any,
    request: ChatCompletionRequest,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply the model's chat template via mlx-vlm's helper.

    mlx-vlm's ``apply_chat_template`` mirrors mlx-lm's but accepts the
    multimodal kwargs the VLM models need (we pass ``num_images=0`` since
    DFlash-eligible aliases are text-only Qwen3.5/3.6 variants today).

    ``enable_thinking`` resolution (caller-side; we just thread through):
      None  → defer to mlx-vlm default (Qwen3 family = True).
      True  → force chain-of-thought on.
      False → force chain-of-thought off (server --no-thinking or per-
              request ``enable_thinking=false`` body field).
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    messages = []
    for m in request.messages:
        content = m.content
        if isinstance(content, list):
            # Multimodal payload — DFlash server is text-only. Collapse
            # text parts; non-text parts (image/audio/video) are
            # dropped. A 400 would surprise users mid-prompt, but a
            # silent drop hides "why is my model ignoring the image?"
            # debugging — so we degrade with a visible WARN log per
            # request that hits this path.
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
                    "DFlash server is text-only; dropped %d non-text "
                    "content part(s) of type(s) %s. The request will be "
                    "served using text parts only — switch to the standard "
                    "server (no --enable-dflash) for full multimodal "
                    "support.",
                    len(dropped_kinds),
                    sorted(set(dropped_kinds)),
                )
            content = "".join(text_pieces)
        messages.append({"role": m.role, "content": content})

    # Preserve historic default (enable_thinking=True) when neither the
    # server-level --no-thinking nor a per-request body override is set,
    # to keep behaviour stable for callers that never opt out.
    effective_thinking = True if enable_thinking is None else enable_thinking
    return apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=0,
        num_audios=0,
        enable_thinking=effective_thinking,
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
    # Track the last token id to disambiguate "hit max_tokens but the
    # final token was actually EOS" — without this we'd falsely flag a
    # natural-stop response as truncated when it lands on exactly the
    # budget. None means "no token observed yet".
    last_token_id: int | None = None

    # Track max_tokens so we can report ``finish_reason="length"`` when
    # generation was truncated (OpenAI clients distinguish "stop"
    # = natural end / stop sequence from "length" = token-budget hit;
    # presenting "stop" for a truncated reply misleads downstream tools).
    _max_tokens = gen_kwargs.get("max_tokens")

    # Resolve the model's EOS token id (best-effort). Used by the
    # length-vs-stop disambiguation below; falls back to None when the
    # processor doesn't expose a tokenizer (the heuristic then degrades
    # to pure token-count comparison).
    _eos_ids: set[int] = set()
    _tok = getattr(processor, "tokenizer", processor)
    _eos = getattr(_tok, "eos_token_id", None)
    if isinstance(_eos, int):
        _eos_ids.add(_eos)
    elif isinstance(_eos, (list, tuple, set)):
        _eos_ids.update(int(t) for t in _eos if isinstance(t, int))

    error_message: str | None = None

    async with _dflash_lock:
        # mlx-vlm's stream_generate is a sync generator — run it in a
        # thread pool so we don't block the FastAPI event loop. Iterate
        # by polling with ``run_in_executor`` per chunk. We're already
        # inside a coroutine, so use ``get_running_loop`` (the 3.10+
        # idiom; ``get_event_loop`` is deprecated for in-coroutine use).
        # The executor MUST be ``_dflash_executor`` (single-thread) so
        # consecutive ``next(gen)`` calls land on the same worker —
        # mlx's GPU Stream is thread-local and a hand-off across worker
        # threads would crash mid-generation.
        loop = asyncio.get_running_loop()

        # Create the generator on the same worker that will drive it,
        # not on the event-loop thread — otherwise the first ``next``
        # crosses a thread boundary just like the rest.
        #
        # Wrap construction in a sentinel pattern too: if
        # ``stream_generate`` raises at setup time (OOM, missing kernel,
        # bad arg) the exception would otherwise propagate out of the
        # async generator and leave the SSE client hanging without a
        # ``[DONE]``. Surfacing it as an error SSE keeps the contract
        # the same as the mid-stream error path below.
        def _make_gen():
            try:
                return stream_generate(model, processor, prompt, **gen_kwargs)
            except Exception as e:  # noqa: BLE001 — surface upstream; outer code converts to error SSE
                return e

        gen_or_err = await loop.run_in_executor(_dflash_executor, _make_gen)
        if isinstance(gen_or_err, Exception):
            logger.exception(
                "DFlash stream_generate raised at construction: %s",
                gen_or_err,
                exc_info=gen_or_err,
            )
            error_message = f"{type(gen_or_err).__name__}: {gen_or_err}"
            # OpenAI ChatCompletion only accepts {stop, length, tool_calls,
            # content_filter, function_call}. The error block on the final
            # SSE chunk carries the abort details for clients.
            finish_reason = "length"
            gen = None
        else:
            gen = gen_or_err

        # Sentinels distinguish "generator exhausted" (None) from
        # "generator raised mid-stream" (an Exception instance). Catching
        # only StopIteration would let any other mlx-vlm error propagate
        # through run_in_executor, abort the response coroutine, and
        # leave the SSE client hanging without a final ``[DONE]`` — the
        # client then either times out or holds the connection forever.
        def _next_chunk():
            try:
                return next(gen)
            except StopIteration:
                return None
            except Exception as e:  # noqa: BLE001 — surface upstream; loop converts to error SSE
                return e

        try:
            while gen is not None:
                chunk = await loop.run_in_executor(_dflash_executor, _next_chunk)
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    logger.exception(
                        "DFlash stream_generate raised mid-stream: %s",
                        chunk,
                        exc_info=chunk,
                    )
                    error_message = f"{type(chunk).__name__}: {chunk}"
                    # See above for OpenAI spec literal-set rationale.
                    finish_reason = "length"
                    break
                # Always sync token counts from the chunk — even when text
                # is empty (mlx-vlm occasionally emits trailing flush
                # chunks carrying the final token counters but no
                # incremental text). Skipping the update would leave the
                # final usage block with stale numbers.
                total_completion_tokens = chunk.generation_tokens
                prompt_tokens = chunk.prompt_tokens
                _ct = getattr(chunk, "token", None)
                if isinstance(_ct, int):
                    last_token_id = _ct
                if not chunk.text:
                    continue
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
        finally:
            # On client disconnect (CancelledError) or any other exit
            # from the loop, drain the generator so its GPU state is
            # released *before* the lock unblocks the next request.
            # Without this the abandoned generator's KV cache lingers
            # until GC runs, transiently doubling memory and risking a
            # mid-step crash if the next request triggers reallocation.
            # Routed through ``_dflash_executor`` for thread affinity.
            if gen is not None:
                _gen_to_close = gen

                def _close_gen():
                    try:
                        _gen_to_close.close()
                    except Exception:  # noqa: BLE001 — cleanup is best-effort
                        logger.debug(
                            "DFlash generator close raised; ignoring",
                            exc_info=True,
                        )

                # ``run_in_executor`` may itself observe cancellation;
                # shield so the cleanup completes before propagating
                # the cancellation up.
                try:
                    await asyncio.shield(
                        loop.run_in_executor(_dflash_executor, _close_gen)
                    )
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

    # Length-truncation detection — mlx-vlm's GenerationResult has no
    # ``finish_reason`` field, so we infer "length" by comparing the
    # completion token count to the budget. Only set when we exited the
    # loop normally (StopIteration), not when the generator errored or
    # produced fewer tokens (natural stop).
    #
    # Subtle case: if the model emitted EOS exactly at ``max_tokens``,
    # the stop was natural and reporting "length" would mislead clients
    # into auto-continuing (only to get an immediate EOS again). Check
    # the last token id against the resolved EOS set to keep the
    # classification honest in this edge case.
    if (
        finish_reason == "stop"
        and _max_tokens is not None
        and total_completion_tokens >= _max_tokens
        and last_token_id not in _eos_ids
    ):
        finish_reason = "length"

    # Final chunk — finish_reason + usage. If we broke out of the loop
    # because the underlying generator raised, attach an OpenAI-style
    # error block so the client gets a readable failure instead of
    # silent truncation.
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
    if error_message is not None:
        final["error"] = {"type": "dflash_runtime_error", "message": error_message}
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
        loop = asyncio.get_running_loop()

        # mlx-vlm's ``generate`` blocks; offload to the dedicated
        # single-thread DFlash executor so every mlx-vlm call lands on
        # the same worker (matches ``_stream_completion`` — see the
        # _dflash_executor comment at module top).
        #
        # Wrap in a sentinel pattern so generate-time errors (OOM, bad
        # arg, drafter mismatch) come back as a clean HTTP 500 with a
        # readable detail string rather than a raw stack trace. Mirrors
        # the stream path's error handling.
        def _generate_safely():
            try:
                return generate(model, processor, prompt, **gen_kwargs)
            except Exception as e:  # noqa: BLE001 — surface as HTTPException below
                return e

        result = await loop.run_in_executor(_dflash_executor, _generate_safely)

    if isinstance(result, Exception):
        logger.exception(
            "DFlash non-stream generate raised: %s", result, exc_info=result
        )
        raise HTTPException(
            status_code=500,
            detail=f"DFlash runtime error: {type(result).__name__}: {result}",
        )

    # OpenAI distinguishes "stop" (natural end / stop sequence) from
    # "length" (token-budget hit). mlx-vlm doesn't surface that on
    # GenerationResult, so infer from token-count vs requested budget.
    #
    # Known v1 limitation: unlike the streaming path which can read
    # ``chunk.token`` and check against EOS, ``mlx_vlm.generate``
    # returns only the concatenated text + token counts. If the model
    # emits EOS at exactly ``max_tokens`` the non-stream response will
    # still report ``finish_reason="length"`` (false truncation). A
    # client that auto-continues will issue one more request that
    # immediately returns EOS — annoying but not corrupt. Fix requires
    # an upstream mlx-vlm change to expose the final token id; tracked
    # as a v2 follow-up.
    _max_tokens = gen_kwargs.get("max_tokens")
    finish_reason = (
        "length"
        if _max_tokens is not None and result.generation_tokens >= _max_tokens
        else "stop"
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
                finish_reason=finish_reason,
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
    no_thinking: bool = False,
) -> None:
    """Load the model + DFlash drafter via mlx-vlm and start uvicorn.

    The mlx-vlm load path is mandatory: the DFlash hooks
    (``capture_layer_ids``, ``_dflash_rounds``) live on the mlx-vlm
    model classes, not mlx-lm's. Loading via ``mlx_lm.load`` would give
    us a model without the hooks and DFlash would silently fall back to
    AR — exactly the kind of "silent regression" the eligibility gate
    is meant to prevent. We surface a clear error if mlx-vlm is missing
    or too old.

    Eligibility re-check: even though the CLI's ``serve_command`` gates
    on the alias before calling here, a *programmatic* caller (e.g. a
    notebook or test harness) can bypass the CLI entirely. We re-run
    the path-detectable gates (4-bit quant via repo-name heuristic;
    non-empty drafter). MoE detection requires the AliasProfile (an
    ``is_moe`` flag aliases.json maintains by hand) and is therefore
    only enforced via the CLI entrypoint — callers serving an
    arbitrary ``main_model_repo`` programmatically are responsible for
    not pointing it at a MoE model. Documented in CALLERS.md.
    """
    if not have_runtime():
        raise RuntimeError(
            "DFlash server requires mlx-vlm 0.5.0+ — install with "
            "``pip install 'rapid-mlx[dflash]'``."
        )

    # Belt-and-suspenders eligibility re-check for programmatic callers
    # (the CLI's serve_command already gates on the alias upstream, but
    # we don't want to depend on it being the only entrypoint).
    from .eligibility import (
        DFlashUnavailable,
        _looks_like_4bit,  # noqa: PLC2701 — internal helper
    )

    if _looks_like_4bit(main_model_repo):
        raise DFlashUnavailable(
            f"DFlash cannot run on a 4-bit quantized model "
            f"(main_model_repo={main_model_repo!r}); upstream PoC measured "
            "regression to 0.63-0.96× on Qwen3.5-4B-MLX-4bit. Use the "
            "8-bit variant."
        )
    if not drafter_repo:
        raise DFlashUnavailable(
            "DFlash requires a non-empty drafter_repo — pass the DFlash "
            "drafter HF path (e.g. 'z-lab/Qwen3.5-27B-DFlash')."
        )

    import uvicorn
    from mlx_vlm import load

    # CRITICAL: load model + drafter on the dedicated DFlash executor
    # thread (not the main thread). mlx-lm 0.31.3+ keeps GPU streams in
    # thread-local storage, so weights loaded on thread A cannot be
    # evaluated on thread B — generate() raises ``RuntimeError: There
    # is no Stream(gpu, N) in current thread``. By pinning load AND all
    # subsequent generate() calls to the same single-worker executor,
    # streams stay reachable for the lifetime of the process.
    def _load_all():
        t0 = time.perf_counter()
        m, p = load(main_model_repo)
        logger.info("DFlash: main model loaded in %.1fs", time.perf_counter() - t0)
        rt = load_runtime(drafter_repo)
        return m, p, rt

    logger.info("DFlash: loading main model via mlx-vlm: %s", main_model_repo)
    model, processor, runtime = _dflash_executor.submit(_load_all).result()

    app = _build_app(
        model=model,
        processor=processor,
        runtime=runtime,
        served_model_name=served_model_name,
        default_max_tokens=default_max_tokens,
        cors_origins=cors_origins,
        no_thinking=no_thinking,
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
