# SPDX-License-Identifier: Apache-2.0
"""Anthropic Messages API endpoints — /v1/messages."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.anthropic_adapter import anthropic_to_openai, openai_to_anthropic
from ..api.anthropic_models import AnthropicRequest
from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import (
    StreamingThinkRouter,
    StreamingToolCallFilter,
    clean_output_text,
    extract_multimodal_content,
    strip_special_tokens,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import BaseEngine
from ..service.helpers import (
    _build_usage,
    _disconnect_guard,
    _parse_tool_calls_with_parser,
    _resolve_max_tokens,
    _validate_model_name,
    _wait_with_disconnect,
    get_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/messages")
async def create_anthropic_message(
    request: Request,
):
    """
    Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI format, runs inference
    through the existing engine, and converts the response back.
    """
    body = await request.json()
    anthropic_request = AnthropicRequest(**body)

    _validate_model_name(anthropic_request.model)
    engine = get_engine(anthropic_request.model)

    # --- Detailed request logging ---
    n_msgs = len(anthropic_request.messages)
    total_chars = 0
    last_user_preview = ""
    for m in anthropic_request.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        if m.role == "user":
            last_user_preview = content[:300]
    sys_chars = len(anthropic_request.system) if anthropic_request.system else 0
    n_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
    logger.info(
        f"[REQUEST] POST /v1/messages (anthropic) stream={anthropic_request.stream} "
        f"model={anthropic_request.model!r} max_tokens={anthropic_request.max_tokens} "
        f"msgs={n_msgs} total_chars={total_chars} system_chars={sys_chars} "
        f"tools={n_tools}"
    )
    logger.debug(f"[REQUEST] last user message preview: {last_user_preview!r}")

    # Convert Anthropic request -> OpenAI request
    openai_request = anthropic_to_openai(anthropic_request)

    if anthropic_request.stream:
        return StreamingResponse(
            _disconnect_guard(
                _stream_anthropic_messages(engine, openai_request, anthropic_request),
                request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming: run inference through existing engine
    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(
            openai_request.max_tokens,
            getattr(openai_request, "enable_thinking", None),
        ),
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
    }

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
    cfg = get_config()
    if openai_request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = openai_request.enable_thinking
    elif cfg.no_thinking:
        chat_kwargs["enable_thinking"] = False

    start_time = time.perf_counter()
    timeout = cfg.default_timeout

    try:
        output = await _wait_with_disconnect(
            engine.chat(messages=messages, **chat_kwargs),
            request,
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
    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages: {output.completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    # Parse tool calls
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text, openai_request
    )

    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))

    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    openai_response = ChatCompletionResponse(
        model=cfg.model_name or openai_request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_build_usage(output, None),
    )

    anthropic_response = openai_to_anthropic(
        openai_response, cfg.model_name or anthropic_request.model
    )
    return Response(
        content=anthropic_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


@router.post("/v1/messages/count_tokens")
async def count_anthropic_tokens(request: Request):
    """Count tokens for an Anthropic Messages API request."""
    body = await request.json()

    engine = get_engine()
    tokenizer = engine.tokenizer

    total_tokens = 0

    # System message
    system = body.get("system", "")
    if isinstance(system, str) and system:
        total_tokens += len(tokenizer.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    total_tokens += len(tokenizer.encode(text))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            if content:
                total_tokens += len(tokenizer.encode(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total_tokens += len(tokenizer.encode(text))
                    if block.get("input"):
                        total_tokens += len(
                            tokenizer.encode(json.dumps(block["input"]))
                        )
                    sub_content = block.get("content", "")
                    if isinstance(sub_content, str) and sub_content:
                        total_tokens += len(tokenizer.encode(sub_content))
                    elif isinstance(sub_content, list):
                        for item in sub_content:
                            if isinstance(item, dict):
                                item_text = item.get("text", "")
                                if item_text:
                                    total_tokens += len(tokenizer.encode(item_text))

    # Tools
    for tool in body.get("tools", []):
        name = tool.get("name", "")
        if name:
            total_tokens += len(tokenizer.encode(name))
        desc = tool.get("description", "")
        if desc:
            total_tokens += len(tokenizer.encode(desc))
        if tool.get("input_schema"):
            total_tokens += len(tokenizer.encode(json.dumps(tool["input_schema"])))

    return {"input_tokens": total_tokens}


def _emit_content_pieces(
    pieces: list[tuple[str, str]],
    current_block_type: str | None,
    block_index: int,
) -> tuple[list[str], str | None, int]:
    """Emit Anthropic SSE events for content pieces from the think router."""
    events = []
    for block_type, text in pieces:
        if block_type != current_block_type:
            if current_block_type is not None:
                events.append(
                    f"event: content_block_stop\ndata: "
                    f"{json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                )
                block_index += 1
            current_block_type = block_type
            content_block = (
                {"type": block_type, "text": ""}
                if block_type == "text"
                else {"type": block_type, "thinking": ""}
            )
            events.append(
                f"event: content_block_start\ndata: "
                f"{json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': content_block})}\n\n"
            )
        delta_key = "thinking" if block_type == "thinking" else "text"
        delta_type = "thinking_delta" if block_type == "thinking" else "text_delta"
        delta_event = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": delta_type, delta_key: text},
        }
        events.append(
            f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
        )
    return events, current_block_type, block_index


async def _stream_anthropic_messages(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    anthropic_request: AnthropicRequest,
) -> AsyncIterator[str]:
    """Stream Anthropic Messages API SSE events."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.perf_counter()

    messages, images, videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(
            openai_request.max_tokens,
            getattr(openai_request, "enable_thinking", None),
        ),
        "temperature": openai_request.temperature,
        "top_p": openai_request.top_p,
    }

    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
    cfg = get_config()
    if openai_request.enable_thinking is not None:
        chat_kwargs["enable_thinking"] = openai_request.enable_thinking
    elif cfg.no_thinking:
        chat_kwargs["enable_thinking"] = False

    # Emit message_start
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": cfg.model_name or anthropic_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    accumulated_text = ""
    accumulated_raw = ""
    tool_filter = StreamingToolCallFilter()
    _tokenizer = engine.tokenizer if hasattr(engine, "tokenizer") else None
    _chat_template = ""
    if _tokenizer and hasattr(_tokenizer, "chat_template"):
        _chat_template = _tokenizer.chat_template or ""
    _starts_thinking = (
        "<think>" in _chat_template and "add_generation_prompt" in _chat_template
    )
    think_router = StreamingThinkRouter(start_in_thinking=_starts_thinking)
    prompt_tokens = 0
    completion_tokens = 0

    current_block_type = None
    block_index = 0

    if cfg.reasoning_parser:
        cfg.reasoning_parser.reset_state()

    async for output in engine.stream_chat(messages=messages, **chat_kwargs):
        delta_text = output.new_text

        if hasattr(output, "prompt_tokens") and output.prompt_tokens:
            prompt_tokens = output.prompt_tokens
        if hasattr(output, "completion_tokens") and output.completion_tokens:
            completion_tokens = output.completion_tokens

        if delta_text:
            accumulated_text += delta_text
            content = None

            if cfg.reasoning_parser:
                previous_raw = accumulated_raw
                accumulated_raw += delta_text
                delta_msg = cfg.reasoning_parser.extract_reasoning_streaming(
                    previous_raw, accumulated_raw, delta_text
                )
                if delta_msg is not None:
                    content = delta_msg.content
            else:
                content = strip_special_tokens(delta_text)

            if content:
                content = strip_special_tokens(content)

            if content:
                filtered = tool_filter.process(content)
                if not filtered:
                    continue
                pieces = think_router.process(filtered)
                events, current_block_type, block_index = _emit_content_pieces(
                    pieces, current_block_type, block_index
                )
                for event in events:
                    yield event

    # Flush remaining from both filters
    remaining = tool_filter.flush()
    if remaining:
        events, current_block_type, block_index = _emit_content_pieces(
            think_router.process(remaining), current_block_type, block_index
        )
        for event in events:
            yield event

    flush_pieces = think_router.flush()
    if flush_pieces:
        events, current_block_type, block_index = _emit_content_pieces(
            flush_pieces, current_block_type, block_index
        )
        for event in events:
            yield event

    # Close final content block
    if current_block_type is not None:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
        block_index += 1

    # Handle reasoning parser finalization
    if cfg.reasoning_parser and accumulated_raw:
        final_msg = (
            cfg.reasoning_parser.finalize_streaming(accumulated_raw)
            if hasattr(cfg.reasoning_parser, "finalize_streaming")
            else None
        )
        if final_msg and final_msg.content:
            content = strip_special_tokens(final_msg.content)
            if content:
                accumulated_text = content
                yield (
                    f"event: content_block_start\ndata: "
                    f"{json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                )
                delta_event = {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                block_index += 1
        cfg.reasoning_parser.reset_state()

    # Check for tool calls in accumulated text
    _, tool_calls = _parse_tool_calls_with_parser(accumulated_text, openai_request)

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tool_index = block_index + i
            try:
                tool_input = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                tool_input = {}

            tool_block_start = {
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": {},
                },
            }
            yield f"event: content_block_start\ndata: {json.dumps(tool_block_start)}\n\n"

            input_json = json.dumps(tool_input)
            input_delta = {
                "type": "content_block_delta",
                "index": tool_index,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            }
            yield f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tool_index})}\n\n"

    stop_reason = "tool_use" if tool_calls else "end_turn"

    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Anthropic messages (stream): prompt={prompt_tokens} + completion={completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
    )

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
