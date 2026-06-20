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
    LegacyCompletionLogProbs,
    PromptTokensDetails,
    Usage,
)
from ..config import get_config
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _check_admission_or_503,
    _disconnect_guard,
    _extract_streaming_token_logprobs,
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
    # F-152: legacy completions params that have NO implementation on
    # Rapid-MLX must fail loudly instead of returning 200 with a single
    # completion (the silent-compat lie SDKs port broken from). The
    # canonical chat-completions handler already rejects ``n > 1``;
    # mirror that here and extend to ``best_of`` (a top-k rerank knob
    # we don't implement at all). ``n == 1`` and ``best_of == 1`` are
    # the OpenAI defaults — accept them silently so well-behaved
    # clients passing the documented default don't see a 400.
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "n > 1 is not supported on /v1/completions. Rapid-MLX "
                "generates one completion per request — send "
                "individual requests if you need multiple samples."
            ),
        )
    if request.best_of is not None and request.best_of > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "best_of > 1 is not supported on /v1/completions. "
                "Rapid-MLX has no server-side reranker — send "
                "individual requests and rerank client-side."
            ),
        )
    # F-153: ``logprobs`` on legacy completions is an INTEGER (top-k
    # count, 0..5 per OpenAI spec). The pydantic ``mode="before"``
    # validator on ``CompletionRequest`` already rejects the
    # chat-shape ``bool`` form with a 422; here we enforce the spec
    # range with a 400 so ``logprobs=20`` (chat-shape ``top_logprobs``
    # ceiling) doesn't slip through and DoS the server with
    # top-of-vocab work.
    if request.logprobs is not None and (
        request.logprobs < 0 or request.logprobs > 5
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "logprobs must be between 0 and 5 on /v1/completions "
                "(OpenAI legacy spec)."
            ),
        )
    # F-152 follow-up (codex r1 BLOCKING): legacy clients use
    # ``echo:true + logprobs:N`` SPECIFICALLY to score prompt tokens
    # (lm-evaluation-harness, ``openai.Completion.create`` with
    # ``echo=True``). Producing logprobs arrays that cover only the
    # generated tokens — with a leading prompt prefix in ``text`` —
    # would mis-align the ``text_offset`` cursor against ``tokens``
    # and silently corrupt every prompt-conditioned score. We don't
    # replay the prompt through the sampler (no per-token
    # distributions available without a dedicated prefill-with-
    # logprobs path), so reject the combination with a clear 400
    # instead of returning partial-but-wrong data. Either knob
    # alone keeps working; only the combination is rejected.
    if request.echo and request.logprobs is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                "`echo` combined with `logprobs` is not supported on "
                "/v1/completions: Rapid-MLX does not replay the prompt "
                "through the sampler, so we cannot return per-token "
                "logprobs for the echoed prefix. Send `echo` and "
                "`logprobs` in separate requests (the `echo` request "
                "returns the prompt-prefixed text; the `logprobs` "
                "request returns the generated-token distributions)."
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
            enforce_context_length_for_prompt(engine, _p, max_tokens=_resolved_max)

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

        # F-152/F-153: ``logprobs`` is an integer (top-k count). When
        # non-None we route through ``stream_generate`` to accumulate
        # per-token distributions chunk by chunk (the same pattern
        # ``routes/chat.py`` uses for ``logprobs=true, top_logprobs=K``
        # — the non-streaming ``generate`` path doesn't surface the
        # per-step ``mx.array`` distributions a top-k logprobs payload
        # needs). ``logprobs=0`` is a valid OpenAI request that asks
        # for the sampled-token logprob WITHOUT alternatives — we
        # still need ``_extract_streaming_token_logprobs`` to surface
        # the sampled probability, but pass ``effective_top_k=1`` to
        # avoid the ``argpartition(-0)[-0:]``-returns-full-vocab
        # pre-existing footgun in ``_extract_token_logprob`` (chat
        # route side-steps this by gating ``logprobs && top_logprobs``;
        # we have to handle ``top_k=0`` explicitly). The resulting
        # ``top_logprobs`` dict is stripped to ``{}`` below so the
        # response shape stays spec-correct.
        want_logprobs = request.logprobs is not None
        top_k_logprobs = request.logprobs or 0
        effective_top_k = max(1, top_k_logprobs)

        for i, prompt in enumerate(prompts):
            token_logprobs_list = []
            if want_logprobs:
                # Accumulate streaming chunks; the engine emits one
                # GenerationOutput per generated token (or per flush
                # under ``stream_interval > 1`` — the helper's per-step
                # iteration handles both shapes; see
                # ``service/helpers.py::_extract_streaming_token_logprobs``).
                output = None
                _accum_text_parts: list[str] = []
                stream_iter = engine.stream_generate(
                    prompt=prompt,
                    max_tokens=_resolve_max_tokens(request.max_tokens),
                    temperature=_resolve_temperature(request.temperature),
                    top_p=_resolve_top_p(request.top_p),
                    stop=request.stop,
                    **extended_kwargs,
                )

                async def _drain_stream(it=stream_iter):
                    nonlocal output
                    async for chunk in it:
                        output = chunk
                        _accum_text_parts.append(chunk.new_text or "")
                        token_logprobs_list.extend(
                            _extract_streaming_token_logprobs(
                                chunk, engine.tokenizer, effective_top_k
                            )
                        )
                    return output

                output = await _wait_with_disconnect(
                    _drain_stream(), raw_request, timeout=timeout
                )
                if output is None:
                    return Response(status_code=499)  # Client closed request
                # The stream's last chunk carries the aggregate text on
                # the LLM engine path (it's accumulated by
                # ``RequestOutput.output_text``), but the MLLM
                # scheduler historically populated only the per-chunk
                # ``new_text`` — fold the accumulated parts to cover
                # both paths.
                final_text = output.text or "".join(_accum_text_parts)
            else:
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
                final_text = output.text

            # F-152: ``echo:true`` prepends the prompt to the response
            # text (legacy OpenAI behaviour — used by eval harnesses
            # like ``lm-evaluation-harness`` to score prompt-conditioned
            # token log-probs). Cheap to implement (just a string
            # concat); without it the silent-drop pre-fix made every
            # eval that depends on the prompt prefix score garbage.
            if request.echo:
                final_text = prompt + final_text

            # Build the legacy logprobs payload per OpenAI spec: four
            # parallel arrays keyed positionally per generated token.
            # ``echo + logprobs`` is rejected upstream (see route
            # entry) so ``offset`` always starts at 0 here.
            choice_logprobs = None
            if want_logprobs:
                tokens_arr: list[str] = []
                token_lps: list[float] = []
                top_lps: list[dict[str, float]] = []
                text_offset: list[int] = []
                offset = 0  # echo+logprobs is rejected upstream
                for entry in token_logprobs_list:
                    tokens_arr.append(entry.token)
                    token_lps.append(entry.logprob)
                    # ``logprobs=0`` per OpenAI spec asks for the
                    # sampled-token logprob WITHOUT any alternatives.
                    # ``effective_top_k=1`` above means
                    # ``entry.top_logprobs`` carries a single-element
                    # list; strip it so the response shape matches
                    # what a real OpenAI call returns (``{}``).
                    if top_k_logprobs == 0:
                        top_lps.append({})
                    else:
                        top_lps.append(
                            {
                                tl.token: tl.logprob
                                for tl in (entry.top_logprobs or [])
                            }
                        )
                    text_offset.append(offset)
                    offset += len(entry.token)
                choice_logprobs = LegacyCompletionLogProbs(
                    tokens=tokens_arr,
                    token_logprobs=token_lps,
                    top_logprobs=top_lps,
                    text_offset=text_offset,
                )

            choices.append(
                CompletionChoice(
                    index=i,
                    text=final_text,
                    finish_reason=output.finish_reason,
                    logprobs=choice_logprobs,
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

    # F-152: ``echo`` on the streaming path emits the prompt as the
    # FIRST SSE chunk, then continues with generated tokens. Without
    # this initial chunk, the streaming branch ignored ``echo`` even
    # after the non-streaming branch was fixed — a silent split-brain
    # SDK clients would discover only at runtime.
    model_name = _resolve_model_name(request.model)
    if request.echo:
        echo_data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": prompt,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(echo_data)}\n\n"

    # F-152: ``logprobs`` on streaming surfaces per-chunk top-k
    # alternatives in the spec-correct legacy shape (four parallel
    # arrays). Each SSE chunk represents one (or a few) generated
    # tokens; emit a fresh ``logprobs`` object per chunk keyed to
    # those token(s) only. Cumulative ``text_offset`` is preserved
    # across chunks so client-side accumulators can concat directly.
    want_logprobs = request.logprobs is not None
    top_k_logprobs = request.logprobs or 0
    effective_top_k = max(1, top_k_logprobs)  # see non-stream branch
    text_offset_cursor = 0  # echo+logprobs is rejected upstream

    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=_resolve_max_tokens(request.max_tokens),
        temperature=_resolve_temperature(request.temperature),
        top_p=_resolve_top_p(request.top_p),
        stop=request.stop,
        **extended_kwargs,
    ):
        choice = {
            "index": 0,
            "text": output.new_text,
            "finish_reason": output.finish_reason if output.finished else None,
        }
        if want_logprobs:
            entries = _extract_streaming_token_logprobs(
                output, engine.tokenizer, effective_top_k
            )
            if entries:
                tokens_arr = []
                token_lps = []
                top_lps = []
                text_offsets = []
                for entry in entries:
                    tokens_arr.append(entry.token)
                    token_lps.append(entry.logprob)
                    if top_k_logprobs == 0:
                        top_lps.append({})
                    else:
                        top_lps.append(
                            {
                                tl.token: tl.logprob
                                for tl in (entry.top_logprobs or [])
                            }
                        )
                    text_offsets.append(text_offset_cursor)
                    text_offset_cursor += len(entry.token)
                choice["logprobs"] = {
                    "tokens": tokens_arr,
                    "token_logprobs": token_lps,
                    "top_logprobs": top_lps,
                    "text_offset": text_offsets,
                }
        # NOTE: distinct per-chunk ``cmpl-XXXX`` ids are F-154 — fixing
        # that here would broaden this PR's scope (F-152/F-153 only).
        # Mint a fresh id per chunk to preserve current behaviour and
        # leave F-154 as a separate one-line follow-up.
        data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
        }
        if output.finished:
            data["usage"] = get_usage(output).model_dump(exclude_none=True)
        yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"
