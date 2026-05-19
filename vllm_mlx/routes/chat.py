# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import gc
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceLogProbs,
    TokenLogProb,
    Usage,
)
from ..api.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_schema_for_guided,
    parse_json_output,
)
from ..api.utils import (
    clean_output_text,
    decode_inline_tool_call_arguments,
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_USE_SYSTEM_SUFFIX,
    _build_usage,
    _check_admission_or_503,
    _disconnect_guard,
    _extract_streaming_token_logprobs,
    _finalize_content_and_reasoning,
    _inject_json_instruction,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _release_admission_unless_committed,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Matches a single backslash directly followed by a non-ASCII codepoint.
# ``lm-format-enforcer``'s grammar permits ``\\`` followed by any codepoint
# as a valid JSON escape, so a model emitting JSON with CJK / emoji content
# can produce strings like ``"\\빠\\르\\게"`` — valid JSON, but the decoded
# value carries literal backslashes. Strip them so clients see clean text.
#
# Scope / known tradeoff: this is applied only on the ``response_format``
# json-output path (see line ~632 below), not to tool-call arguments or
# regular text content. The cleanup is unconditional within that path,
# matching upstream waybarrios#525. A JSON object that LEGITIMATELY
# contains a backslash before a non-ASCII codepoint (e.g. a Windows path
# ``"C:\\사용자\\file.txt"`` in a response_format=json_object reply) will
# be mutated to ``"C:사용자file.txt"``. We accept this tradeoff because:
#  (a) the lm-format-enforcer bug is the overwhelming source of these
#      sequences in JSON-output responses; the file-path case is rare,
#  (b) gating the cleanup on a heuristic ("looks like enforcer output")
#      would be fragile and only catch the obvious patterns,
#  (c) clients that need raw backslash + non-ASCII can fall back to
#      ``response_format=text`` and parse the JSON themselves.
# If a user reports the false-positive in practice, revisit by adding a
# config flag (``--no-strip-spurious-backslashes``) rather than a heuristic.
_BACKSLASH_BEFORE_UNICODE = re.compile(r"\\([^\x00-\x7F])")


def _strip_backslash_before_unicode(obj: object) -> object:
    if isinstance(obj, dict):
        # Clean both keys and values: ``lm-format-enforcer`` can produce
        # ``"\\한\\글": "value"`` (valid JSON, ugly key). Stripping only
        # values would leak the bug into client-visible object keys.
        cleaned: dict[object, object] = {}
        for k, v in obj.items():
            new_key = _strip_backslash_before_unicode(k)
            new_val = _strip_backslash_before_unicode(v)
            if new_key in cleaned:
                # Two distinct dirty keys can collapse to the same clean
                # key (e.g. ``"\\한"`` and ``"한"`` both → ``"한"``). Keep
                # the first occurrence and surface the collision rather
                # than silently dropping a field.
                logger.warning(
                    "JSON key collision after backslash strip: %r dropped "
                    "in favor of earlier value (cleaned key=%r)",
                    k,
                    new_key,
                )
                continue
            cleaned[new_key] = new_val
        return cleaned
    if isinstance(obj, list):
        return [_strip_backslash_before_unicode(v) for v in obj]
    if isinstance(obj, str):
        return _BACKSLASH_BEFORE_UNICODE.sub(r"\1", obj)
    return obj


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create a chat completion (supports multimodal content for VLM models).

    OpenAI-compatible multimodal format for images:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
    ```

    Video support:
    ```json
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }]
    ```

    Structured output (JSON mode):
    ```json
    response_format={"type": "json_object"}
    ```

    Structured output (JSON Schema):
    ```json
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "my_schema",
            "schema": {"type": "object", "properties": {...}}
        }
    }
    ```
    """
    _validate_model_name(request.model)
    engine = get_engine(request.model)

    # Pre-flight admission gate (C4). Runs BEFORE returning the
    # StreamingResponse so a backpressured streaming request surfaces
    # as HTTP 503 (Retry-After) instead of a 200 stream with an
    # error-data chunk. The reservation is released by the route-level
    # ``finally`` (calls ``_release_admission_unless_committed``);
    # on the streaming path ``_commit_state[0]`` is flipped to True so
    # ``_disconnect_guard`` (via its ``engine=`` kwarg) owns the
    # release once the SSE generator closes. Closes the codex R3 leak
    # where validation errors raised between the reservation and the
    # helper used to pin the slot until restart — after
    # ``max_concurrent_requests`` such bad requests the cap was
    # permanently exhausted.
    _check_admission_or_503(engine)
    _commit_state = [False]
    try:
        return await _create_chat_completion_impl(
            request, raw_request, engine, _commit_state
        )
    finally:
        _release_admission_unless_committed(engine, _commit_state[0])


async def _create_chat_completion_impl(
    request: ChatCompletionRequest,
    raw_request: Request,
    engine,
    _commit_state: list[bool],
):
    """Inner impl for ``create_chat_completion`` after admission has
    been acquired by the wrapper. ``_commit_state[0] = True`` defers
    the slot release to ``_disconnect_guard`` for streaming paths."""
    # Validate messages is non-empty
    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    # Validate message roles
    _valid_roles = {"system", "user", "assistant", "tool", "developer"}
    for msg in request.messages:
        if msg.role not in _valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{msg.role}'. Must be one of: {', '.join(sorted(_valid_roles))}",
            )

    # Validate n parameter (only n=1 supported)
    if request.n is not None and request.n > 1:
        raise HTTPException(
            status_code=400,
            detail="n > 1 is not supported. Rapid-MLX generates one completion per request.",
        )

    # Validate max_tokens. Lower bound: must be positive. Upper bound: a
    # hard sanity ceiling so a buggy client passing 999_999_999 cannot
    # combine with unbounded admission to OOM the Metal allocator.
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )
    if request.max_tokens is not None and request.max_tokens > 1_000_000:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at most 1000000",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_p range (OpenAI spec: (0, 1]). Without this, top_p=2.0
    # is silently accepted while sister field `temperature` is checked,
    # so clients with a bug see no signal.
    if request.top_p is not None and (request.top_p <= 0 or request.top_p > 1):
        raise HTTPException(
            status_code=400,
            detail="top_p must be in (0, 1]",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
        )

    # Reject non-empty logit_bias with a clear 400 rather than silently
    # dropping it. We accept {} so defensive clients that always include
    # the field don't break.
    if request.logit_bias:
        raise HTTPException(
            status_code=400,
            detail="logit_bias is not supported on this server",
        )

    # --- Detailed request logging ---
    n_msgs = len(request.messages)
    msg_roles = [m.role for m in request.messages]
    total_chars = 0
    last_user_preview = ""
    for m in request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    has_tools = bool(request.tools)
    n_tools = len(request.tools) if request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/chat/completions stream={request.stream} "
        f"model={request.model!r} max_tokens={request.max_tokens} "
        f"temp={request.temperature} msgs={n_msgs} roles={msg_roles} "
        f"total_chars={total_chars} tools={n_tools} "
        f"response_format={request.response_format}"
    )
    logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

    cfg = get_config()

    # Save original messages (clean dicts) for cloud routing BEFORE
    # local mutations (extract_multimodal_content, developer→system, suffix injection).
    if cfg.cloud_router:
        _cloud_original_messages = [
            (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else {k: v for k, v in dict(msg).items() if v is not None}
            )
            for msg in request.messages
        ]
    else:
        _cloud_original_messages = None

    # Reject image/video content when the loaded model has no vision
    # head. Without this guard ``extract_multimodal_content`` silently
    # drops the image parts on the text-only path and the model
    # hallucinates a confident description of an image it never saw
    # (R9P1: 600M text model returned "a red rose" for arbitrary images).
    if not engine.is_mllm:
        for _msg in request.messages:
            _content = (
                _msg.content if hasattr(_msg, "content") else _msg.get("content", "")
            )
            if isinstance(_content, list):
                for _item in _content:
                    _item_type = (
                        _item.type
                        if hasattr(_item, "type")
                        else (_item.get("type", "") if isinstance(_item, dict) else "")
                    )
                    if _item_type in ("image_url", "image", "video", "video_url"):
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Model '{cfg.model_name}' does not support "
                                "image or video inputs."
                            ),
                        )

    # For MLLM models, keep original messages with embedded images
    if engine.is_mllm:
        messages = []
        for msg in request.messages:
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump(exclude_none=True)
            else:
                raw = dict(msg)
                msg_dict = {k: v for k, v in raw.items() if v is not None}
            messages.append(msg_dict)
        images, videos = [], []
        # The non-MLLM branch decodes tool_call.function.arguments from JSON
        # string to dict inside extract_multimodal_content() so chat templates
        # that iterate args via .items() (e.g. GLM-4.6V) don't crash. The
        # MLLM branch bypasses that helper, so call the shared decoder here.
        if engine.preserve_native_tool_format:
            decode_inline_tool_call_arguments(messages)
        logger.debug(f"MLLM: Processing {len(messages)} messages")
    else:
        messages, images, videos = extract_multimodal_content(
            request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

    has_media = bool(images or videos)
    if engine.is_mllm and not has_media:
        for msg in request.messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else (item.get("type", "") if isinstance(item, dict) else "")
                    )
                    if item_type in ("image_url", "image", "video", "video_url"):
                        has_media = True
                        break
            if has_media:
                break

    # Normalize "developer" role to "system"
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "developer":
            if isinstance(m, dict):
                messages[i]["role"] = "system"
            else:
                m.role = "system"

    # Auto-inject system prompt suffix for tool use and/or reasoning control
    _inject_suffix = None
    if request.tools and cfg.tool_call_parser:
        _inject_suffix = _TOOL_USE_SYSTEM_SUFFIX
    elif cfg.reasoning_parser_name == "minimax":
        _inject_suffix = (
            "\n\nDo NOT think out loud or show your reasoning process. "
            "Give direct answers only — no preamble like 'The user asks...' or "
            "'We should respond...' or 'Let me think...'. Be concise."
        )

    if _inject_suffix:
        has_system = any(
            (m.get("role") if isinstance(m, dict) else getattr(m, "role", None))
            == "system"
            for m in messages
        )
        if has_system:
            for i, m in enumerate(messages):
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                if role == "system":
                    if isinstance(m, dict):
                        messages[i] = {**m, "content": m["content"] + _inject_suffix}
                    else:
                        messages[i]["content"] = m["content"] + _inject_suffix
                    break
        else:
            system_msg = {"role": "system", "content": _inject_suffix.strip()}
            messages = [system_msg] + list(messages)

    # Auto-pin system prompt prefix cache blocks
    if cfg.pin_system_prompt:
        _maybe_pin_system_prompt(messages)

    # Handle response_format - inject system prompt if needed
    response_format = request.response_format
    if response_format:
        try:
            json_instruction = build_json_system_prompt(response_format)
        except Exception as e:
            logger.warning(f"Failed to build JSON system prompt: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid response_format schema: {e}",
            )
        if json_instruction:
            messages = _inject_json_instruction(messages, json_instruction)

    # Resolve enable_thinking once and reuse — drives both the
    # max_tokens default (thinking models need more headroom) and the
    # chat_template kwarg below. (#387)
    resolved_thinking = _resolve_enable_thinking(request)

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_tokens, resolved_thinking),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }

    # Extended sampling params — resolve through the request → CLI →
    # alias → generation_config cascade. Only forwards values the
    # cascade actually produced.
    chat_kwargs.update(build_extended_sampling_kwargs(request))

    # Add multimodal content
    if has_media:
        chat_kwargs["images"] = images if images else None
        chat_kwargs["videos"] = videos if videos else None
        if request.video_fps:
            chat_kwargs["video_fps"] = request.video_fps
        if request.video_max_frames:
            chat_kwargs["video_max_frames"] = request.video_max_frames

    # Add tools if provided
    if request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(request.tools)

    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

    # Cloud routing: offload large-context requests to cloud LLM
    if cfg.cloud_router and not engine.is_mllm and hasattr(engine, "build_prompt"):
        try:
            prompt = engine.build_prompt(messages, tools=request.tools)
            total_tokens, new_tokens = engine.model.estimate_new_tokens(prompt)
            if cfg.cloud_router.should_route_to_cloud(new_tokens):
                logger.info(
                    f"[CLOUD ROUTE] {new_tokens} new tokens (total {total_tokens}) "
                    f"> threshold {cfg.cloud_router.threshold}, "
                    f"routing to {cfg.cloud_router.cloud_model}"
                )
                cloud_messages = _cloud_original_messages
                cloud_kwargs = {
                    "temperature": chat_kwargs.get("temperature"),
                    "max_tokens": chat_kwargs.get("max_tokens"),
                    "top_p": chat_kwargs.get("top_p"),
                }
                if request.stop:
                    cloud_kwargs["stop"] = request.stop
                if request.tool_choice is not None:
                    cloud_kwargs["tool_choice"] = request.tool_choice
                if request.response_format:
                    rf = request.response_format
                    cloud_kwargs["response_format"] = (
                        rf.model_dump() if hasattr(rf, "model_dump") else rf
                    )
                if request.tools:
                    cloud_kwargs["tools"] = [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in request.tools
                    ]
                if request.stream:
                    _commit_state[0] = True
                    return StreamingResponse(
                        _disconnect_guard(
                            cfg.cloud_router.stream_completion(
                                cloud_messages,
                                model_name=cfg.model_name or "cloud",
                                **cloud_kwargs,
                            ),
                            raw_request,
                            engine=engine,
                        ),
                        media_type="text/event-stream",
                    )
                else:
                    result = await _wait_with_disconnect(
                        cfg.cloud_router.completion(cloud_messages, **cloud_kwargs),
                        raw_request,
                        timeout=request.timeout or cfg.default_timeout,
                    )
                    if result is None:
                        return Response(status_code=499, content="Client disconnected")
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                    )
            else:
                logger.info(
                    f"[LOCAL] {new_tokens} new tokens (total {total_tokens}) "
                    f"<= threshold {cfg.cloud_router.threshold}, using local inference"
                )
        except Exception as e:
            logger.warning(
                f"[CLOUD ROUTE] Error during routing check: {e}, falling back to local"
            )

    if request.stream:
        # Validate chat template eagerly so template errors return 400
        if hasattr(engine, "build_prompt") and not engine.is_mllm:
            try:
                engine.build_prompt(
                    messages,
                    tools=chat_kwargs.get("tools"),
                    enable_thinking=chat_kwargs.get("enable_thinking"),
                )
            except Exception as e:
                err_msg = str(e)
                err_type = type(e).__name__
                if (
                    "TemplateError" in err_type
                    or "template" in err_msg.lower()
                    or ("user" in err_msg.lower() and "found" in err_msg.lower())
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Chat template error: {err_msg}",
                    )
                raise
        _commit_state[0] = True
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(engine, messages, request, **chat_kwargs),
                raw_request,
                engine=engine,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response with timing and timeout
    start_time = time.perf_counter()
    timeout = request.timeout or cfg.default_timeout

    # Disable GC during generation to avoid latency spikes
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    # Determine if we need per-token logprobs
    want_logprobs = request.logprobs and request.top_logprobs
    top_k_logprobs = request.top_logprobs or 0
    token_logprobs_list: list[TokenLogProb] = []

    # Check if we should use guided generation for JSON schema
    use_guided = False
    json_schema = None
    if response_format and not request.tools:
        json_schema = extract_json_schema_for_guided(response_format)
        if json_schema and hasattr(engine, "supports_guided_generation"):
            use_guided = engine.supports_guided_generation
            if use_guided:
                logger.info("Using guided generation for JSON schema enforcement")

    try:
        if want_logprobs and not use_guided:
            output = None
            async for chunk in engine.stream_chat(messages=messages, **chat_kwargs):
                output = chunk
                token_logprobs_list.extend(
                    _extract_streaming_token_logprobs(
                        chunk, engine.tokenizer, top_k_logprobs
                    )
                )
            if output is None:
                return Response(status_code=499)
        elif use_guided and json_schema:
            try:
                output = await _wait_with_disconnect(
                    engine.generate_with_schema(
                        messages=messages,
                        json_schema=json_schema,
                        **chat_kwargs,
                    ),
                    raw_request,
                    timeout=timeout,
                )
            except Exception as guided_err:
                logger.warning(
                    f"Guided generation failed, falling back to standard: {guided_err}"
                )
                logger.debug(f"Problematic schema: {json_schema}")
                # Fallback runs under the outer admission reservation
                # still held by the wrapper's ``finally`` — no
                # re-acquire needed (the helper does not release on
                # its own now that release lives at the route level).
                output = await _wait_with_disconnect(
                    engine.chat(messages=messages, **chat_kwargs),
                    raw_request,
                    timeout=timeout,
                )
        else:
            output = await _wait_with_disconnect(
                engine.chat(messages=messages, **chat_kwargs),
                raw_request,
                timeout=timeout,
            )
    except HTTPException:
        raise
    except Exception as e:
        from ..request import InferenceAbortedError

        err_msg = str(e)
        err_type = type(e).__name__
        if isinstance(e, InferenceAbortedError):
            # Engine aborted the request (e.g. Metal runtime error caught
            # in the engine loop). 503 — the server is still up and a
            # smaller request may succeed (#353).
            raise HTTPException(status_code=503, detail=err_msg)
        if (
            "TemplateError" in err_type
            or "template" in err_msg.lower()
            or ("user" in err_msg.lower() and "found" in err_msg.lower())
        ):
            raise HTTPException(
                status_code=400, detail=f"Chat template error: {err_msg}"
            )
        raise
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()

    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Chat completion: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls from output using configured parser
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(output.text, request)

    # Validate tool call parameter values against schemas
    if tool_calls and request.tools:
        _validate_tool_call_params(tool_calls, request.tools)

    # Extract reasoning content. extract_reasoning() is stateless (pure regex
    # on full text), so the singleton is safe here unlike the streaming variant.
    # The tool_calls vs no-tool_calls split is encapsulated in
    # _finalize_content_and_reasoning so the regression test suite can exercise
    # the same orchestration without re-implementing it.
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=output.text,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=cfg.reasoning_parser,
    )

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                parsed_json = _strip_backslash_before_unicode(parsed_json)
                # ``ensure_ascii=False`` keeps non-ASCII characters as
                # raw UTF-8 rather than escaping them to ``\uXXXX``. This
                # is the standard recommendation for JSON-over-HTTP with
                # international content (matches OpenAI's own response
                # encoding); FastAPI emits this body as UTF-8 anyway, so
                # the on-wire bytes are smaller and clients don't have to
                # un-escape user-visible CJK / emoji a second time.
                cleaned_text = json.dumps(parsed_json, ensure_ascii=False)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")
        except Exception as e:
            logger.warning(f"JSON output parsing failed: {e}")

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    # Clean and strip thinking tags from content
    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)
        if response_format and final_content:
            final_content = extract_json_from_response(final_content)

    # Build logprobs for response if requested
    choice_logprobs = None
    if want_logprobs and token_logprobs_list:
        choice_logprobs = ChoiceLogProbs(content=token_logprobs_list)

    chat_response = ChatCompletionResponse(
        model=_resolve_model_name(request.model),
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=choice_logprobs,
            )
        ],
        usage=_build_usage(output, reasoning_text),
    )
    return Response(
        content=chat_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


async def stream_chat_completion(
    engine,
    messages: list,
    request: ChatCompletionRequest,
    **kwargs,
) -> AsyncIterator[str]:
    """Stream chat completion response.

    Uses StreamingPostProcessor for reasoning/tool/sanitization pipeline.
    SSE formatting stays inline for performance (fast path bypasses Pydantic).
    """
    from ..service.postprocessor import StreamingPostProcessor

    cfg = get_config()
    gc_was_enabled = gc.isenabled()
    if cfg.gc_control and gc_was_enabled:
        gc.disable()

    try:
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Check if we should include usage in the final chunk
        include_usage = request.stream_options and request.stream_options.include_usage

        # Logprobs configuration
        want_logprobs = request.logprobs and request.top_logprobs
        top_k_logprobs = request.top_logprobs or 0

        def _build_chunk_logprobs(output: GenerationOutput) -> ChoiceLogProbs | None:
            """Build ChoiceLogProbs for a streaming chunk if logprobs requested."""
            if not want_logprobs:
                return None
            entries = _extract_streaming_token_logprobs(
                output, engine.tokenizer, top_k_logprobs
            )
            return ChoiceLogProbs(content=entries) if entries else None

        # Pre-compute SSE template parts that don't change per-token.
        _sse_created = int(time.time())
        _model_escaped = json.dumps(_resolve_model_name(request.model))
        _sse_prefix = (
            f'data: {{"id":"{response_id}","object":"chat.completion.chunk",'
            f'"created":{_sse_created},"model":{_model_escaped},'
            f'"choices":[{{"index":0,"delta":{{'
        )
        _sse_suffix = "}}]}\n\n"

        def _fast_sse_chunk(text: str, field: str = "content") -> str:
            """Build SSE chunk JSON directly, bypassing Pydantic serialization."""
            escaped = json.dumps(text)
            return f'{_sse_prefix}"{field}":{escaped}{_sse_suffix}'

        # First chunk with role
        _first_sse = f'{_sse_prefix}"role":"assistant"{_sse_suffix}'
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"[SSE-ROLE] {_first_sse.strip()[:200]}")
        yield _first_sse

        # Initialize post-processor.
        # request_dict carries `tools` so streaming parsers (qwen3_coder etc.)
        # can do schema-driven type conversion (#171).
        request_dict = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else None
        )
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=bool(request.tools),
            # `kwargs` is the **kwargs from this function's signature; the
            # route handler unpacks chat_kwargs (which sets
            # "enable_thinking" when request.enable_thinking is not None
            # or cfg.no_thinking is set). Pulled through as a name so
            # StreamingPostProcessor can short-circuit the reasoning
            # parser when the client explicitly disabled thinking
            # (closes the empty-content streaming bug from PR #208).
            enable_thinking=kwargs.get("enable_thinking"),
            json_mode=bool(
                request.response_format
                and getattr(request.response_format, "type", "text") != "text"
            ),
            request=request_dict,
        )
        processor.set_thinking_model(request.model)
        processor.reset()

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0

        # Stream content — PostProcessor handles reasoning/tool/sanitize
        async for output in engine.stream_chat(messages=messages, **kwargs):
            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens

            for event in processor.process_chunk(output):
                if event.type == "content":
                    if not want_logprobs:
                        _sse = _fast_sse_chunk(event.content, "content")
                        if _sse:
                            yield _sse
                    else:
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            model=_resolve_model_name(request.model),
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        content=event.content,
                                    ),
                                    logprobs=_build_chunk_logprobs(output),
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                elif event.type == "reasoning":
                    yield _fast_sse_chunk(event.reasoning, "reasoning_content")

                elif event.type == "tool_call":
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    tool_calls=event.tool_calls,
                                ),
                                finish_reason=event.finish_reason,
                            )
                        ],
                        usage=get_usage(output) if output.finished else None,
                    )
                    _tc_sse = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    logger.info(f"[SSE-TC] {_tc_sse.strip()[:300]}")
                    yield _tc_sse

                elif event.type == "finish":
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=_resolve_model_name(request.model),
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    content=event.content,
                                    reasoning_content=event.reasoning,
                                ),
                                finish_reason=event.finish_reason,
                                logprobs=_build_chunk_logprobs(output),
                            )
                        ],
                        usage=get_usage(output) if output.finished else None,
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        # Fallback tool call detection
        for event in processor.finalize():
            if event.type == "tool_call":
                tool_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=event.tool_calls,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                _fb_sse = f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                logger.info(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
                yield _fb_sse

        # Log throughput
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Chat completion (stream): {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Send final chunk with usage if requested
        if include_usage:
            usage_chunk = ChatCompletionChunk(
                id=response_id,
                model=_resolve_model_name(request.model),
                choices=[],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        if cfg.gc_control and gc_was_enabled:
            gc.enable()
            gc.collect()
