# SPDX-License-Identifier: Apache-2.0
"""Text completion endpoints — /v1/completions."""

import inspect
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
    SSE_RESPONSE_HEADERS,
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


def _engine_supports_completion_logprobs(engine) -> bool:
    capability = getattr(engine, "supports_completion_logprobs", None)
    if capability is not None:
        if callable(capability):
            value = capability()
            if inspect.isawaitable(value):
                close = getattr(value, "close", None)
                if callable(close):
                    close()
                return False
            return value if isinstance(value, bool) else False
        return capability if isinstance(capability, bool) else False
    return getattr(engine, "tokenizer", None) is not None and callable(
        getattr(engine, "stream_generate", None)
    )


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
    if request.logprobs is not None and (request.logprobs < 0 or request.logprobs > 5):
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
        # R-05 (PyPI 0.8.6 dogfood, liang-r2 L2-001): INFO-level logs
        # MUST NOT carry user prompt content. Pre-fix this line emitted
        # ``prompt_preview="my secret password is hunter2"`` straight to
        # the INFO stream — anyone with log-aggregator read access could
        # harvest credentials. The chat / anthropic / responses lanes
        # already split metadata (counts) at INFO and the content
        # preview at DEBUG; legacy completions just skipped the parity.
        # Match the chat lane: INFO carries counts only, DEBUG carries
        # a 300-char preview behind the operator-controlled log-level
        # dial. (Server's default level is INFO, so the preview no
        # longer reaches production log aggregators without an explicit
        # opt-in.)
        n_prompts = len(prompts)
        prompt_len = sum(len(p) for p in prompts)
        logger.info(
            f"[REQUEST] POST /v1/completions stream={request.stream} "
            f"model={request.model!r} max_tokens={request.max_tokens} "
            f"temp={request.temperature} n_prompts={n_prompts} "
            f"prompt_chars={prompt_len}"
        )
        prompt_preview = prompts[0][:300] if prompts else "(empty)"
        logger.debug(f"[REQUEST] prompt preview: {prompt_preview!r}")

        # Context-length pre-check — same DoS gate the chat/anthropic/
        # responses routes enforce. Raw-prompt API skips chat templating
        # but the prompt-token budget still applies. Iterate the list
        # form because each entry hits prefill independently. See
        # ``service/helpers.py::enforce_context_length_for_prompt``.
        _resolved_max = _resolve_max_tokens(request.max_tokens)
        for _p in prompts:
            enforce_context_length_for_prompt(engine, _p, max_tokens=_resolved_max)

        # Codex r2/r3 BLOCKING: engine capability guard for ``logprobs``
        # applies to BOTH streaming and non-streaming paths. Without
        # this check on the streaming branch, the AttributeError fires
        # inside the SSE generator (already-committed
        # ``StreamingResponse``) and clients see an EventSource that
        # disconnects after the first chunk instead of a controlled
        # 501. Lift to the top so both branches are covered.
        _want_logprobs = request.logprobs is not None
        if _want_logprobs and not _engine_supports_completion_logprobs(engine):
            raise HTTPException(
                status_code=501,
                detail=(
                    "logprobs requested but this engine does not expose "
                    "the streaming logprobs extraction path "
                    "(``stream_generate`` + ``tokenizer``). Reissue "
                    "without ``logprobs`` or use a model that supports "
                    "per-token distributions."
                ),
            )

        if request.stream:
            _admission_committed = True
            # C-01 force-abort: holder list the engine populates with
            # the admitted scheduler request id; the disconnect_guard
            # reads it and force-calls scheduler.abort_request on
            # client disconnect (closes the Astrid r3 hang where the
            # generator-close cascade alone left ~6144 tokens of
            # runaway generation eating GPU after the client RST'd).
            _completion_rid_holder: list[str | None] = [None]
            return StreamingResponse(
                _disconnect_guard(
                    stream_completion(
                        engine,
                        prompts[0],
                        request,
                        request_id_holder=_completion_rid_holder,
                    ),
                    raw_request,
                    engine=engine,
                    request_id_holder=_completion_rid_holder,
                ),
                media_type="text/event-stream",
                headers=SSE_RESPONSE_HEADERS,
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

        # Engine capability guard for ``logprobs`` is enforced earlier
        # (top of the route, covers both streaming + non-streaming).

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
                _stream_yielded = False
                stream_iter = engine.stream_generate(
                    prompt=prompt,
                    max_tokens=_resolve_max_tokens(request.max_tokens),
                    temperature=_resolve_temperature(request.temperature),
                    top_p=_resolve_top_p(request.top_p),
                    stop=request.stop,
                    **extended_kwargs,
                )

                async def _drain_stream(it=stream_iter):
                    nonlocal output, _stream_yielded
                    async for chunk in it:
                        _stream_yielded = True
                        output = chunk
                        # B023 is a false positive here: the closure is
                        # invoked synchronously inside the same loop
                        # iteration via ``await _wait_with_disconnect``
                        # below, so ``_accum_text_parts`` /
                        # ``token_logprobs_list`` always reference the
                        # current iteration's bindings. Suppress so the
                        # ruff baseline stays clean.
                        _accum_text_parts.append(chunk.new_text or "")  # noqa: B023
                        token_logprobs_list.extend(  # noqa: B023
                            _extract_streaming_token_logprobs(
                                chunk, engine.tokenizer, effective_top_k
                            )
                        )
                    return output

                output = await _wait_with_disconnect(
                    _drain_stream(), raw_request, timeout=timeout
                )
                # Codex r2/r3 BLOCKING: ``_wait_with_disconnect``
                # returns ``None`` on client disconnect, on timeout,
                # OR when ``_drain_stream`` exits cleanly without
                # yielding any chunks. Disambiguate three cases:
                #   1. ``_stream_yielded`` — at least one chunk
                #      reached us; bailing now is a mid-flight
                #      disconnect or timeout → 499.
                #   2. ``await raw_request.is_disconnected()`` — the
                #      client closed BEFORE the first chunk; also
                #      499. (Codex r3 BLOCKING #2: without this
                #      check, a disconnect-before-first-token was
                #      misclassified as a successful empty
                #      completion.)
                #   3. Otherwise — engine genuinely returned an empty
                #      stream; synthesize an empty ``GenerationOutput``
                #      so the response matches OpenAI's empty-
                #      completion shape.
                if output is None:
                    if _stream_yielded:
                        return Response(status_code=499)
                    try:
                        client_gone = await raw_request.is_disconnected()
                    except Exception:
                        client_gone = False
                    if client_gone:
                        return Response(status_code=499)
                    from ..engine.base import GenerationOutput

                    output = GenerationOutput(
                        text="",
                        finish_reason="stop",
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
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
            #
            # Codex r2 BLOCKING #3: ``text_offset`` is documented as
            # offsets into ``choices[0].text``. We compute offsets
            # cumulatively from ``len(decoded_token)`` so they
            # ALWAYS align with ``"".join(tokens_arr)``. When
            # ``output.text`` differs from that concatenation
            # (whitespace normalization in ``clean_output_text``,
            # tokenizer-side cleanup) the spec-correct move is to
            # surface the token-concatenated text as ``text`` so
            # ``text_offset`` is byte-exact against the field it
            # references. Falls back to ``output.text`` only when no
            # token entries were captured (empty stream / engine
            # quirk).
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
                            {tl.token: tl.logprob for tl in (entry.top_logprobs or [])}
                        )
                    text_offset.append(offset)
                    offset += len(entry.token)
                choice_logprobs = LegacyCompletionLogProbs(
                    tokens=tokens_arr,
                    token_logprobs=token_lps,
                    top_logprobs=top_lps,
                    text_offset=text_offset,
                )
                # Pin ``final_text`` to the token concatenation so
                # ``text_offset`` is byte-exact. Skip when no tokens
                # were captured (e.g. engine quirk that yielded
                # ``new_text`` without ``new_token_ids``) — keep the
                # accumulated raw text in that case.
                if tokens_arr:
                    final_text = "".join(tokens_arr)

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
    *,
    request_id_holder: list | None = None,
) -> AsyncIterator[str]:
    """Stream completion response.

    Args:
        request_id_holder: C-01 force-abort plumbing. When provided,
            forwarded as a kwarg to ``engine.stream_generate`` so the
            engine writes the admitted scheduler request id into
            ``holder[0]``. The route's ``_disconnect_guard`` reads the
            same holder to force-call ``scheduler.abort_request`` on
            client disconnect. ``None`` (default) is a no-op.
    """
    extended_kwargs = build_extended_sampling_kwargs(request)
    # C-01: pass the holder through so the engine can publish the
    # scheduler request id without changing every engine signature.
    if request_id_holder is not None:
        extended_kwargs["request_id_holder"] = request_id_holder

    # F-152: ``echo`` on the streaming path emits the prompt as the
    # FIRST SSE chunk, then continues with generated tokens. Without
    # this initial chunk, the streaming branch ignored ``echo`` even
    # after the non-streaming branch was fixed — a silent split-brain
    # SDK clients would discover only at runtime.
    model_name = _resolve_model_name(request.model)
    # F-154: every SSE chunk in a single streamed completion shares
    # one ``cmpl-XXXX`` id (per OpenAI legacy /v1/completions spec).
    # Pre-fix this branch minted a fresh id at every yield point —
    # client-side aggregators that key on ``id`` to correlate chunks
    # to a request treated each chunk as a separate response, making
    # cross-chunk assembly impossible. ``/v1/chat/completions``
    # already shares one ``chatcmpl-XXXX`` across all chunks; this
    # change brings the legacy route to parity. ``created`` is also
    # captured once so all chunks report the same start timestamp.
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created_ts = int(time.time())
    if request.echo:
        echo_data = {
            "id": completion_id,
            "object": "text_completion",
            "created": created_ts,
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

    # D-SSE-USAGE: capture the engine-reported usage from the final
    # ``GenerationOutput`` so the dedicated trailing usage chunk (only
    # emitted when ``stream_options.include_usage=true``) can read it
    # AFTER the per-token loop has finished. Pre-fix this code attached
    # usage to the finish chunk unconditionally — see comment further
    # down at the chunk-build site.
    _final_usage = None

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
                            {tl.token: tl.logprob for tl in (entry.top_logprobs or [])}
                        )
                    text_offsets.append(text_offset_cursor)
                    text_offset_cursor += len(entry.token)
                choice["logprobs"] = {
                    "tokens": tokens_arr,
                    "token_logprobs": token_lps,
                    "top_logprobs": top_lps,
                    "text_offset": text_offsets,
                }
                # Codex r3 BLOCKING #3: pin chunk ``text`` to the
                # token concatenation so ``text_offset`` is byte-
                # exact against the field it references (matches the
                # non-streaming path's alignment fix). Without this
                # rebind, ``output.new_text`` may differ from
                # ``"".join(tokens_arr)`` after tokenizer cleanup
                # or whitespace normalization, and clients that
                # slice ``chunk.text[offset:offset+len(token)]``
                # would read garbage.
                choice["text"] = "".join(tokens_arr)
        # F-154: reuse ``completion_id`` / ``created_ts`` minted once
        # above so every chunk shares the same id+timestamp — matches
        # the OpenAI legacy /v1/completions spec and brings parity with
        # the ``/v1/chat/completions`` SSE path.
        data = {
            "id": completion_id,
            "object": "text_completion",
            "created": created_ts,
            "model": model_name,
            "choices": [choice],
        }
        # D-SSE-USAGE: capture usage from the engine but DO NOT attach
        # it to this finish chunk — the OpenAI streaming spec says
        # ``usage`` is opt-in via ``stream_options.include_usage=true``
        # and, when opted in, MUST appear ONLY on a dedicated trailing
        # chunk with empty ``choices``. Pre-fix this branch ALWAYS
        # attached usage to the finish chunk, double-counting on
        # aggregating clients (LangChain / AI-SDK / vercel-ai-stream).
        if output.finished:
            _final_usage = get_usage(output)
        yield f"data: {json.dumps(data)}\n\n"

    # Dedicated trailing usage chunk (OpenAI spec — empty ``choices``,
    # populated ``usage``). Only emitted when the caller opted in via
    # ``stream_options.include_usage=true``; otherwise the field is
    # omitted from the wire entirely. Mirrors the trailing usage chunk
    # on ``/v1/chat/completions`` so SDK shape is identical across
    # both endpoints.
    if (
        _final_usage is not None
        and request.stream_options is not None
        and request.stream_options.include_usage
    ):
        usage_data = {
            "id": completion_id,
            "object": "text_completion",
            "created": created_ts,
            "model": model_name,
            "choices": [],
            "usage": _final_usage.model_dump(exclude_none=True),
        }
        yield f"data: {json.dumps(usage_data)}\n\n"

    yield "data: [DONE]\n\n"
