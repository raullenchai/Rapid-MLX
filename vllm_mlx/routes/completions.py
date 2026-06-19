# SPDX-License-Identifier: Apache-2.0
"""Text completion endpoints — /v1/completions."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    PromptTokensDetails,
    Usage,
)
from ..config import get_config
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _check_admission_or_503,
    _disconnect_guard,
    _release_admission_unless_committed,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    enforce_context_length_for_prompt,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Create a text completion."""
    _validate_model_name(request.model)
    if request.suffix:
        raise HTTPException(
            status_code=400,
            detail=(
                "FIM (fill-in-the-middle) 'suffix' is not supported by this "
                "server. Use the chat completions API or omit 'suffix'."
            ),
        )
    engine = get_engine(request.model)

    # Pre-flight admission gate (C4). Reservation is released by the
    # ``finally`` block below; on the streaming path we flip
    # ``_admission_committed`` to True so ``_disconnect_guard`` owns
    # the release once the SSE generator closes. Closes the codex R3
    # leak where any HTTPException between this call and the
    # streaming/non-streaming helper pinned the slot until restart.
    _check_admission_or_503(engine)
    _admission_committed = False
    try:
        # Handle single prompt or list of prompts
        prompts = (
            request.prompt if isinstance(request.prompt, list) else [request.prompt]
        )

        # --- Detailed request logging ---
        prompt_preview = prompts[0][:200] if prompts else "(empty)"
        prompt_len = sum(len(p) for p in prompts)
        logger.info(
            f"[REQUEST] POST /v1/completions stream={request.stream} "
            f"max_tokens={request.max_tokens} temp={request.temperature} "
            f"prompt_chars={prompt_len} prompt_preview={prompt_preview!r}"
        )

        # Context-length pre-check — same DoS gate the chat/anthropic/
        # responses routes enforce. Raw-prompt API skips chat templating
        # but the prompt-token budget still applies. Iterate the list
        # form because each entry hits prefill independently. See
        # ``service/helpers.py::enforce_context_length_for_prompt``.
        _resolved_max = _resolve_max_tokens(request.max_tokens)
        for _p in prompts:
            enforce_context_length_for_prompt(
                engine, _p, max_tokens=_resolved_max
            )

        if request.stream:
            _admission_committed = True
            return StreamingResponse(
                _disconnect_guard(
                    stream_completion(engine, prompts[0], request),
                    raw_request,
                    engine=engine,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming response with timing and timeout
        start_time = time.perf_counter()
        timeout = request.timeout or get_config().default_timeout
        choices = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_cached_tokens = 0

        extended_kwargs = build_extended_sampling_kwargs(request)

        for i, prompt in enumerate(prompts):
            output = await _wait_with_disconnect(
                engine.generate(
                    prompt=prompt,
                    max_tokens=_resolve_max_tokens(request.max_tokens),
                    temperature=_resolve_temperature(request.temperature),
                    top_p=_resolve_top_p(request.top_p),
                    stop=request.stop,
                    **extended_kwargs,
                ),
                raw_request,
                timeout=timeout,
            )
            if output is None:
                return Response(status_code=499)  # Client closed request

            choices.append(
                CompletionChoice(
                    index=i,
                    text=output.text,
                    finish_reason=output.finish_reason,
                )
            )
            total_completion_tokens += output.completion_tokens
            total_prompt_tokens += (
                output.prompt_tokens if hasattr(output, "prompt_tokens") else 0
            )
            total_cached_tokens += getattr(output, "cached_tokens", 0) or 0

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Completion: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        comp_response = CompletionResponse(
            model=_resolve_model_name(request.model),
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=total_cached_tokens)
                    if total_cached_tokens
                    else None
                ),
            ),
        )
        return Response(
            content=comp_response.model_dump_json(exclude_none=True),
            media_type="application/json",
        )
    finally:
        _release_admission_unless_committed(engine, _admission_committed)


async def stream_completion(
    engine,
    prompt: str,
    request: CompletionRequest,
) -> AsyncIterator[str]:
    """Stream completion response."""
    extended_kwargs = build_extended_sampling_kwargs(request)

    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=_resolve_max_tokens(request.max_tokens),
        temperature=_resolve_temperature(request.temperature),
        top_p=_resolve_top_p(request.top_p),
        stop=request.stop,
        **extended_kwargs,
    ):
        data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _resolve_model_name(request.model),
            "choices": [
                {
                    "index": 0,
                    "text": output.new_text,
                    "finish_reason": output.finish_reason if output.finished else None,
                }
            ],
        }
        if output.finished:
            data["usage"] = get_usage(output).model_dump(exclude_none=True)
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"
