# SPDX-License-Identifier: Apache-2.0
"""Chat completion endpoints — /v1/chat/completions."""

import gc
import json
import logging
import re
import time
import uuid
from collections import Counter
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
    extract_json_from_response,
    extract_multimodal_content,
    sanitize_output,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import GenerationOutput
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _TOOL_CALL_REQUIRED_RETRY_PROMPT,
    _TOOL_CONTINUATION_RETRY_PROMPT,
    _TOOL_USE_SYSTEM_SUFFIX,
    _append_tool_continuation_prompt,
    _build_usage,
    _disconnect_guard,
    _extract_token_logprob,
    _inject_json_instruction,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    get_engine,
    get_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_REPETITION_WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")
_TOOL_TEXT_REPETITION_MIN_WORDS = 32
_TOOL_TEXT_REPETITION_MIN_COUNT = 24
_TOOL_TEXT_REPETITION_RATIO = 0.60


def _tool_choice_allows_tool_calls(tool_choice) -> bool:
    if tool_choice == "none":
        return False
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "none":
        return False
    return True


def _is_repetitive_tool_text(text: str) -> bool:
    words = _REPETITION_WORD_RE.findall(text.lower())
    if len(words) < _TOOL_TEXT_REPETITION_MIN_WORDS:
        return False

    tail = words[-96:]
    most_common_count = Counter(tail).most_common(1)[0][1]
    if (
        most_common_count >= _TOOL_TEXT_REPETITION_MIN_COUNT
        and most_common_count / len(tail) >= _TOOL_TEXT_REPETITION_RATIO
    ):
        return True

    for size in (2, 3, 4):
        if len(tail) < size * 8:
            continue
        ngrams = [" ".join(tail[i : i + size]) for i in range(len(tail) - size + 1)]
        ngram_count = Counter(ngrams).most_common(1)[0][1]
        if ngram_count >= 8 and (ngram_count * size) / len(tail) >= 0.50:
            return True

    return False


def _buffered_stream_text(buffered_events: list[tuple]) -> str:
    parts = []
    for event, _ in buffered_events:
        parts.append(getattr(event, "content", None) or "")
        parts.append(getattr(event, "reasoning", None) or "")
    return "".join(parts)


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

    # Validate max_tokens (must be positive)
    if request.max_tokens is not None and request.max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be at least 1",
        )

    # Validate temperature range (OpenAI spec: 0-2)
    if request.temperature is not None and (
        request.temperature < 0 or request.temperature > 2
    ):
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    # Validate top_logprobs range (OpenAI spec: 0-20)
    if request.top_logprobs is not None and (
        request.top_logprobs < 0 or request.top_logprobs > 20
    ):
        raise HTTPException(
            status_code=400,
            detail="top_logprobs must be between 0 and 20",
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

    messages, added_tool_continuation_prompt = _append_tool_continuation_prompt(
        list(messages),
        bool(request.tools),
    )

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

    # Prepare kwargs
    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(request.max_tokens, request.enable_thinking),
        "temperature": _resolve_temperature(request.temperature),
        "top_p": _resolve_top_p(request.top_p),
        "stop": request.stop,
    }

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

    # Pass through enable_thinking if explicitly set by the client
    if request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = request.enable_thinking
    elif cfg.no_thinking or (
        added_tool_continuation_prompt and cfg.tool_call_parser == "qwen3_coder_xml"
    ):
        chat_kwargs["enable_thinking"] = False

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
                    return StreamingResponse(
                        _disconnect_guard(
                            cfg.cloud_router.stream_completion(
                                cloud_messages,
                                model_name=cfg.model_name or "cloud",
                                **cloud_kwargs,
                            ),
                            raw_request,
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
        return StreamingResponse(
            _disconnect_guard(
                stream_chat_completion(
                    engine,
                    messages,
                    request,
                    tool_continuation_retry=added_tool_continuation_prompt,
                    **chat_kwargs,
                ),
                raw_request,
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
                if chunk.logprobs is not None and chunk.new_text:
                    token_id = chunk.tokens[-1] if chunk.tokens else 0
                    token_logprobs_list.append(
                        _extract_token_logprob(
                            chunk.logprobs, token_id, engine.tokenizer, top_k_logprobs
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
        err_msg = str(e)
        err_type = type(e).__name__
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

    # Extract reasoning content FIRST.
    # Note: extract_reasoning() is stateless (pure regex on full text),
    # so using the singleton is safe here unlike the streaming variant.
    reasoning_text = None
    if cfg.reasoning_parser:
        text_to_parse = cleaned_text or output.text
        reasoning_text, cleaned_text = cfg.reasoning_parser.extract_reasoning(
            text_to_parse
        )

    # Process response_format if specified (after reasoning parser cleaned the text)
    if response_format and not tool_calls:
        json_input = cleaned_text or output.text
        try:
            _, parsed_json, is_valid, error = parse_json_output(
                json_input, response_format
            )
            if parsed_json is not None:
                cleaned_text = json.dumps(parsed_json)
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
    tool_continuation_retry: bool = False,
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
            if not want_logprobs or output.logprobs is None:
                return None
            token_id = output.tokens[-1] if output.tokens else 0
            token_lp = _extract_token_logprob(
                output.logprobs, token_id, engine.tokenizer, top_k_logprobs
            )
            return ChoiceLogProbs(content=[token_lp])

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

        # Track token counts for usage reporting
        prompt_tokens = 0
        completion_tokens = 0

        def _format_stream_event(event, output: GenerationOutput) -> list[str]:
            if event.type == "content":
                if not want_logprobs:
                    _sse = _fast_sse_chunk(event.content, "content")
                    return [_sse] if _sse else []
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
                return [f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"]

            if event.type == "reasoning":
                return [_fast_sse_chunk(event.reasoning, "reasoning_content")]

            if event.type == "tool_call":
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
                return [_tc_sse]

            if event.type == "finish":
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
                return [f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"]

            return []

        retry_attempts = 0
        active_messages = messages
        tool_retries_enabled = bool(request.tools) and _tool_choice_allows_tool_calls(
            request.tool_choice
        )
        max_tool_continuation_retries = (
            2 if (tool_continuation_retry or tool_retries_enabled) else 0
        )
        retry_prompt = (
            _TOOL_CONTINUATION_RETRY_PROMPT
            if tool_continuation_retry
            else _TOOL_CALL_REQUIRED_RETRY_PROMPT
        )

        while True:
            # Initialize post-processor
            processor = StreamingPostProcessor(
                cfg,
                tools_requested=bool(request.tools),
                json_mode=bool(
                    request.response_format
                    and getattr(request.response_format, "type", "text") != "text"
                ),
                request_dict=request.model_dump(),
            )
            processor.set_thinking_model(request.model)
            processor.reset()

            buffered_events: list[tuple] = []
            deferred_finish: tuple | None = None
            emitted_tool_call = False
            last_output = None
            retry_reason = None

            # Stream content — PostProcessor handles reasoning/tool/sanitize
            async for output in engine.stream_chat(messages=active_messages, **kwargs):
                last_output = output
                if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                    prompt_tokens = output.prompt_tokens
                if hasattr(output, "completion_tokens") and output.completion_tokens:
                    completion_tokens = output.completion_tokens

                retry_window = (
                    retry_attempts < max_tool_continuation_retries
                    and not emitted_tool_call
                )

                for event in processor.process_chunk(output):
                    if retry_window and event.type in ("content", "reasoning"):
                        buffered_events.append((event, output))
                        if _is_repetitive_tool_text(
                            _buffered_stream_text(buffered_events)
                        ):
                            retry_reason = "repetitive text before tool call"
                            logger.info(
                                "[tool-continuation] detected repetitive text "
                                "before tool call; aborting current stream"
                            )
                            break
                        continue

                    if (
                        retry_window
                        and event.type == "finish"
                        and event.finish_reason == "stop"
                    ):
                        deferred_finish = (event, output)
                        continue

                    if event.type == "tool_call":
                        emitted_tool_call = True
                        for buffered_event, buffered_output in buffered_events:
                            for _sse in _format_stream_event(
                                buffered_event, buffered_output
                            ):
                                yield _sse
                        buffered_events.clear()

                    for _sse in _format_stream_event(event, output):
                        yield _sse

                if retry_reason:
                    break

            # Fallback tool call detection
            if not retry_reason:
                for event in processor.finalize():
                    if event.type == "tool_call":
                        emitted_tool_call = True
                        for buffered_event, buffered_output in buffered_events:
                            for _sse in _format_stream_event(
                                buffered_event, buffered_output
                            ):
                                yield _sse
                        buffered_events.clear()

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
                        _fb_sse = (
                            f"data: {tool_chunk.model_dump_json(exclude_none=True)}\n\n"
                        )
                        logger.info(f"[SSE-FALLBACK-TC] {_fb_sse.strip()[:300]}")
                        yield _fb_sse

            if not retry_reason and deferred_finish and not emitted_tool_call:
                retry_reason = "text-only stop"
            elif not retry_reason and buffered_events and not emitted_tool_call:
                retry_reason = "stream exhausted without tool call"

            if retry_reason and retry_attempts < max_tool_continuation_retries:
                retry_attempts += 1
                logger.info(
                    "[tool-continuation] retrying after %s following "
                    "tool result (attempt %d/%d)",
                    retry_reason,
                    retry_attempts,
                    max_tool_continuation_retries,
                )
                active_messages = list(messages) + [
                    {"role": "user", "content": retry_prompt}
                ]
                continue

            if retry_reason and buffered_events:
                logger.warning(
                    "[tool-continuation] suppressing buffered %s after retry "
                    "budget exhausted",
                    retry_reason,
                )
                buffered_events.clear()
                deferred_finish = None
                finish_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=_resolve_model_name(request.model),
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(),
                            finish_reason=(
                                last_output.finish_reason if last_output else "stop"
                            )
                            or "stop",
                        )
                    ],
                    usage=get_usage(last_output) if last_output else None,
                )
                yield f"data: {finish_chunk.model_dump_json(exclude_none=True)}\n\n"

            for buffered_event, buffered_output in buffered_events:
                for _sse in _format_stream_event(buffered_event, buffered_output):
                    yield _sse
            if deferred_finish:
                event, output = deferred_finish
                for _sse in _format_stream_event(event, output):
                    yield _sse

            break

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
