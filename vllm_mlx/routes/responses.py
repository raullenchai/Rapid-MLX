# SPDX-License-Identifier: Apache-2.0
"""OpenAI Responses API endpoint — /v1/responses.

Stateless shim that lets Codex CLI (and any other Responses-API client)
talk to rapid-mlx as if it were OpenAI. Translates Responses → Chat,
runs inference through the existing engine, translates back into the
seven SSE events Codex CLI parses (``response.created``,
``response.output_item.added``, ``response.output_text.delta``,
``response.function_call_arguments.delta``, ``response.output_item.done``,
``response.completed``, ``response.failed``).

Statelessness: ``previous_response_id`` returns 400. Codex CLI doesn't
use that field (openai/codex#3841) — it re-sends the full conversation
history every turn in ``input``.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..api.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from ..api.responses_adapter import openai_to_responses, responses_to_openai
from ..api.responses_models import ResponsesRequest
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import (
    StreamingThinkRouter,
    StreamingToolCallFilter,
    clean_output_text,
    extract_multimodal_content,
    sanitize_output,
    strip_special_tokens,
    strip_thinking_tags,
)
from ..config import get_config
from ..engine import BaseEngine
from ..middleware.auth import check_rate_limit, verify_api_key
from ..service.helpers import (
    _build_usage,
    _check_admission_or_503,
    _disconnect_guard,
    _effective_enable_thinking,
    _finalize_content_and_reasoning,
    _parse_tool_calls_with_parser,
    _release_admission_unless_committed,
    _resolve_enable_thinking,
    _resolve_max_tokens,
    _resolve_temperature,
    _resolve_top_p,
    _validate_model_name,
    _wait_with_disconnect,
    build_extended_sampling_kwargs,
    get_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolved_sampling_kwargs(openai_request: ChatCompletionRequest) -> dict:
    """Resolve sampling params through the 4-layer cascade.

    Mirrors the helper in routes/anthropic.py so ``/v1/responses`` users
    get the same alias / generation_config defaults as ``/v1/messages``
    and ``/v1/chat/completions``.
    """
    out = {
        "temperature": _resolve_temperature(openai_request.temperature),
        "top_p": _resolve_top_p(openai_request.top_p),
        "stop": getattr(openai_request, "stop", None),
    }
    out.update(build_extended_sampling_kwargs(openai_request))
    return out


def _should_start_in_thinking(chat_template: str, enable_thinking: bool | None) -> bool:
    """Same heuristic as routes/anthropic.py: stream that starts inside an
    implicit ``<think>`` block should be routed as reasoning until the
    closing tag. Bypass when thinking is explicitly disabled."""
    if enable_thinking is False:
        return False
    return "<think>" in chat_template and "add_generation_prompt" in chat_template


@router.post(
    "/v1/responses",
    dependencies=[
        Depends(verify_api_key),
        Depends(check_rate_limit),
    ],
)
async def create_response(request: Request):
    """OpenAI Responses API entry point.

    Codex CLI hardcodes ``stream: true`` and sends the full
    conversation history in ``input[]`` each turn, so the streaming
    path is the hot path.
    """
    body = await request.json()
    responses_request = ResponsesRequest(**body)

    # Statelessness gate — see module docstring. Codex CLI does not set
    # this field; clients that DO use it would get silent prompt loss
    # on retries because we have no response store, so 400 loudly.
    if responses_request.previous_response_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "previous_response_id is not supported by this server — "
                "rapid-mlx is a stateless Responses API shim. Re-send the "
                "full conversation history in the `input` field each turn."
            ),
        )

    # Reuse the Claude-Code / Codex bypass from #557: ``claude-*``,
    # ``gpt-*`` model names pass through to the loaded engine instead of
    # 404'ing on _validate_model_name. Codex sends ``gpt-5``,
    # ``gpt-5-codex``, etc. — none of which match a local alias.
    if not (responses_request.model or "").startswith(("claude-", "gpt-")):
        _validate_model_name(responses_request.model)
    engine = get_engine(responses_request.model)

    # Pre-flight admission — same C4 reservation shape the other two
    # routes use. ``_admission_committed`` flips to True when the
    # streaming path takes over so ``_disconnect_guard`` owns release.
    _check_admission_or_503(engine)
    _admission_committed = False
    try:
        _log_request(responses_request)

        cfg_for_log = get_config()
        if (
            responses_request.model
            and cfg_for_log.model_name
            and responses_request.model != cfg_for_log.model_name
        ):
            logger.info(
                "Responses /v1/responses: request model=%r served by loaded engine=%r",
                responses_request.model,
                cfg_for_log.model_name,
            )

        openai_request = responses_to_openai(responses_request)

        if responses_request.stream:
            _admission_committed = True
            return StreamingResponse(
                _disconnect_guard(
                    _stream_responses(engine, openai_request, responses_request),
                    request,
                    engine=engine,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return await _non_stream(engine, openai_request, responses_request, request)
    finally:
        _release_admission_unless_committed(engine, _admission_committed)


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------


async def _non_stream(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
    request: Request,
) -> Response:
    cfg = get_config()
    created_at = int(time.time())

    messages, _images, _videos = extract_multimodal_content(
        openai_request.messages,
        preserve_native_format=engine.preserve_native_tool_format,
    )

    chat_kwargs = {
        "max_tokens": _resolve_max_tokens(
            openai_request.max_tokens,
            _resolve_enable_thinking(openai_request),
        ),
        **_resolved_sampling_kwargs(openai_request),
    }
    if openai_request.tools:
        chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)

    resolved_thinking = _resolve_enable_thinking(openai_request)
    if resolved_thinking is not None:
        chat_kwargs["enable_thinking"] = resolved_thinking

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
    except Exception as e:  # noqa: BLE001 — match other routes' error shape
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
        if "Failed to process image" in err_msg or "Failed to process video" in err_msg:
            raise HTTPException(status_code=400, detail=err_msg)
        raise

    if output is None:
        return Response(status_code=499)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = output.completion_tokens / elapsed if elapsed > 0 else 0
    logger.info(
        f"Responses: {output.completion_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )

    engine_tool_calls = getattr(output, "tool_calls", None)
    cleaned_text, tool_calls = _parse_tool_calls_with_parser(
        output.text, openai_request, structured_tool_calls=engine_tool_calls
    )
    cleaned_text, reasoning_text = _finalize_content_and_reasoning(
        raw_text=output.raw_text or output.text,
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        reasoning_parser=cfg.reasoning_parser,
        engine_reasoning_text=getattr(output, "reasoning_text", "") or "",
        enable_thinking=_effective_enable_thinking(
            resolved_thinking, cfg.model_path or cfg.model_name
        ),
        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # Forwarded from ``ResponsesRequest.reasoning_max_tokens`` via
        # the Responses → OpenAI adapter. None → no cap (back-compat).
        reasoning_max_tokens=getattr(openai_request, "reasoning_max_tokens", None),
    )

    final_content = None
    if cleaned_text:
        final_content = strip_thinking_tags(clean_output_text(cleaned_text))
        final_content = sanitize_output(final_content)

    finish_reason = "tool_calls" if tool_calls else output.finish_reason

    openai_response = ChatCompletionResponse(
        model=cfg.model_name or openai_request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(
                    content=final_content,
                    reasoning_content=reasoning_text,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_build_usage(output, reasoning_text),
    )

    responses_response = openai_to_responses(
        openai_response,
        model=cfg.model_name or responses_request.model,
        request=responses_request,
        created_at=created_at,
    )
    return Response(
        content=responses_response.model_dump_json(exclude_none=True),
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# Streaming path — emits the 7 SSE events Codex CLI parses
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event in Responses-API shape.

    Codex parses ``event: <name>\\ndata: <json>\\n\\n`` framing — same as
    chat-completions and Anthropic streams. No ``data: [DONE]`` here;
    that sentinel is chat-completions-only.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_responses(
    engine: BaseEngine,
    openai_request: ChatCompletionRequest,
    responses_request: ResponsesRequest,
) -> AsyncIterator[str]:
    """Stream a Responses-API SSE event sequence Codex CLI can parse.

    Event order Codex expects:
      1. ``response.created`` — once, before any deltas
      2. ``response.output_item.added`` (message item) — when first text
         delta arrives
      3. ``response.output_text.delta`` — each chunk of assistant text
      4. ``response.output_item.done`` (message item) — when text ends,
         before any tool_calls
      5. For each tool call:
         ``response.output_item.added`` (function_call item) +
         ``response.function_call_arguments.delta`` (full JSON args) +
         ``response.output_item.done`` (function_call item)
      6. ``response.completed`` — terminal event, carries final usage

    Errors emit ``response.failed`` then close. Codex treats
    stream-close-without-``response.completed`` as a hard failure, so
    we always finalize.
    """
    cfg = get_config()
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    start_time = time.perf_counter()
    served_model = cfg.model_name or responses_request.model

    # response.created — Codex needs this before any deltas.
    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": served_model,
                "output": [],
            },
        },
    )
    try:
        messages, _images, _videos = extract_multimodal_content(
            openai_request.messages,
            preserve_native_format=engine.preserve_native_tool_format,
        )

        chat_kwargs = {
            "max_tokens": _resolve_max_tokens(
                openai_request.max_tokens,
                _resolve_enable_thinking(openai_request),
            ),
            **_resolved_sampling_kwargs(openai_request),
        }
        if openai_request.tools:
            chat_kwargs["tools"] = convert_tools_for_template(openai_request.tools)
        resolved_thinking = _resolve_enable_thinking(openai_request)
        if resolved_thinking is not None:
            chat_kwargs["enable_thinking"] = resolved_thinking

        accumulated_text = ""
        accumulated_raw = ""
        accumulated_structured_tool_calls: list[dict] = []
        tool_filter = StreamingToolCallFilter()

        _tokenizer = engine.tokenizer
        _chat_template = ""
        if _tokenizer and hasattr(_tokenizer, "chat_template"):
            _chat_template = _tokenizer.chat_template or ""
        _starts_thinking = _should_start_in_thinking(
            _chat_template, chat_kwargs.get("enable_thinking")
        )
        think_router = StreamingThinkRouter(start_in_thinking=_starts_thinking)

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0

        # Lazy message-item state. We do NOT emit the message
        # output_item.added until we have actual user-facing text to stream
        # — a turn that is pure tool_calls should not emit a phantom empty
        # message item.
        message_item_id: str | None = None
        message_output_index: int | None = None
        message_open = False

        # Per-request reasoning parser instance (matches anthropic.py).
        reasoning_parser = None
        if cfg.reasoning_parser_name:
            try:
                from ..reasoning import get_parser

                reasoning_parser = get_parser(cfg.reasoning_parser_name)()
            except Exception:
                pass
        if chat_kwargs.get("enable_thinking") is False:
            reasoning_parser = None
        if reasoning_parser:
            reasoning_parser.reset_state()

        # Per-request reasoning cap (upstream vLLM PR #20859 backport).
        # Responses SSE drops reasoning to the floor (Codex doesn't read
        # ``response.reasoning_text.delta`` in v1) so the cap's primary
        # job here is to RECLASSIFY: once the budget is exhausted, any
        # further reasoning bytes become ``response.output_text.delta``
        # so the user actually sees a reply instead of an infinite
        # silent thinking block.
        _reasoning_cap = getattr(responses_request, "reasoning_max_tokens", None)
        _reasoning_tokens_emitted = 0
        _reasoning_cap_hit = False
        _reasoning_close_injected = False

        def _account_for_reasoning(text: str) -> tuple[str, str, bool]:
            """Returns ``(kept_reasoning, overflow_content, just_hit)``.

            Approximates token count via the chars-÷4 heuristic so the
            cap matches what the OpenAI route's StreamingPostProcessor
            does and the same effective budget applies regardless of
            which API surface the client uses.
            """
            nonlocal _reasoning_tokens_emitted, _reasoning_cap_hit
            if _reasoning_cap is None or not text:
                return text, "", False
            if _reasoning_cap_hit:
                return "", text, False
            delta = max(1, len(text) // 4)
            new_total = _reasoning_tokens_emitted + delta
            if new_total < _reasoning_cap:
                _reasoning_tokens_emitted = new_total
                return text, "", False
            if new_total == _reasoning_cap:
                # Exact-boundary latch (codex round-2 BLOCKING #3):
                # keep this chunk as reasoning but mark the cap hit so
                # the NEXT chunk routes through the overflow branch
                # (and the text-parser path injects ``</think>``).
                _reasoning_tokens_emitted = new_total
                _reasoning_cap_hit = True
                return text, "", True
            remaining = _reasoning_cap - _reasoning_tokens_emitted
            keep_chars = max(0, remaining * 4)
            _reasoning_tokens_emitted = _reasoning_cap
            _reasoning_cap_hit = True
            return text[:keep_chars], text[keep_chars:], True

        async def _open_message_item() -> str:
            """Emit response.output_item.added for the assistant message.

            Returns the event string so callers can yield it; the bookkeeping
            for ``message_open`` lives here so the open/close pair stays
            symmetric.
            """
            nonlocal message_item_id, message_output_index, message_open
            message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
            message_output_index = 0
            message_open = True
            return _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": {
                        "type": "message",
                        "id": message_item_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )

        async def _emit_text_delta(delta: str) -> AsyncIterator[str]:
            """Yield the message item-added event (lazily) + a text delta."""
            nonlocal accumulated_text
            if not delta:
                return
            if not message_open:
                yield await _open_message_item()
            accumulated_text += delta
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta,
                },
            )

        async for output in engine.stream_chat(messages=messages, **chat_kwargs):
            delta_text = output.new_text
            # Accumulate the RAW model output (pre-filter, pre-router) so the
            # post-loop tool_call parser can see `<tool_call>...</tool_call>`
            # XML that tool_filter rightly suppresses from the user-facing
            # text channel. Without this, `accumulated_text` is empty in the
            # tool-calling case and no `response.function_call` SSE event
            # gets emitted — Codex sees turn.completed with zero output
            # items and the agent loop silently ends. The chat-completions
            # route avoids this by parsing `output.text` (the full
            # non-streamed text) directly; the streaming path needs an
            # explicit raw accumulator.
            if delta_text:
                accumulated_raw += delta_text

            if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if hasattr(output, "completion_tokens") and output.completion_tokens:
                completion_tokens = output.completion_tokens
            if hasattr(output, "cached_tokens") and output.cached_tokens:
                cached_tokens = output.cached_tokens

            engine_tool_calls = getattr(output, "tool_calls", None) or []
            if engine_tool_calls:
                accumulated_structured_tool_calls.extend(engine_tool_calls)
                continue

            if not delta_text:
                continue

            # Channel-routed engines (harmony / gemma4) — honor the
            # channel directly. ``reasoning`` channel drops here
            # because Responses-API streams don't have a reasoning
            # delta event Codex parses (Codex maps it from a separate
            # ``response.reasoning_text.delta`` we omit in v1).
            output_channel = getattr(output, "channel", None)
            if output_channel is not None:
                if output_channel in ("content", "tool_call"):
                    content = strip_special_tokens(delta_text)
                    if content:
                        filtered = tool_filter.process(content)
                        if filtered:
                            async for ev in _emit_text_delta(filtered):
                                yield ev
                elif output_channel == "reasoning":
                    # Reasoning-cap reclassification: once the per-request
                    # cap fires, route the overflow portion of this and
                    # every subsequent reasoning chunk to ``content`` so
                    # the user actually sees a reply instead of an
                    # unending silent reasoning stream. Without the cap
                    # the chunk drops as before (v1 Responses contract).
                    _, overflow, _ = _account_for_reasoning(delta_text)
                    if overflow:
                        content = strip_special_tokens(overflow)
                        if content:
                            filtered = tool_filter.process(content)
                            if filtered:
                                async for ev in _emit_text_delta(filtered):
                                    yield ev
                # ``reasoning`` and unknown channels are dropped for v1.
                continue

            if reasoning_parser:
                # accumulated_raw already updated above; pass current/previous
                # to the parser's streaming extractor.
                previous_raw = (
                    accumulated_raw[: -len(delta_text)]
                    if delta_text
                    else accumulated_raw
                )
                # Text-parser path: once the cap fires, splice ``</think>``
                # in front of the next chunk so the parser flips to
                # content. Idempotent — only fires once per request.
                #
                # Codex round-1 BLOCKING #1: the prior implementation
                # mutated ``delta_text`` to ``"</think>" + delta_text`` but
                # appended ``"</think>"`` AFTER the original delta in
                # ``accumulated_raw``. That left the parser's
                # ``current`` argument out of sync with the (previous,
                # delta) pair (``current != previous + delta``) and the
                # parser saw the forced close marker AFTER the chunk's
                # body instead of before it. Rebuild ``accumulated_raw``
                # so the close marker sits between ``previous_raw`` and
                # the original delta — symmetric with the postprocessor
                # path that injects via ``_maybe_inject_reasoning_close``.
                if _reasoning_cap_hit and not _reasoning_close_injected:
                    delta_text = "</think>" + delta_text
                    accumulated_raw = previous_raw + delta_text
                    _reasoning_close_injected = True
                delta_msg = reasoning_parser.extract_reasoning_streaming(
                    previous_raw, accumulated_raw, delta_text
                )
                if delta_msg is None:
                    continue
                if delta_msg.reasoning:
                    # Account for reasoning bytes. Overflow becomes
                    # content; the next stream iteration will emit the
                    # forged ``</think>`` so the parser cleanly transitions.
                    _, overflow, _ = _account_for_reasoning(delta_msg.reasoning)
                    if overflow:
                        # Promote overflow to content alongside any content
                        # the parser produced this delta.
                        delta_msg.content = (delta_msg.content or "") + overflow
                if delta_msg.content:
                    content = strip_special_tokens(delta_msg.content)
                    if content:
                        filtered = tool_filter.process(content)
                        if filtered:
                            async for ev in _emit_text_delta(filtered):
                                yield ev
                # delta_msg.reasoning intentionally dropped — see above.
                continue

            # Default path: text-only stream with think_router stripping
            # ``<think>...</think>`` from the text channel.
            content = strip_special_tokens(delta_text)
            if not content:
                continue
            filtered = tool_filter.process(content)
            if not filtered:
                continue
            pieces = think_router.process(filtered)
            for block_type, piece in pieces:
                if block_type == "text" and piece:
                    async for ev in _emit_text_delta(piece):
                        yield ev
                # block_type == "thinking" intentionally dropped.

        # Flush filters
        remaining = tool_filter.flush()
        if remaining:
            if reasoning_parser:
                async for ev in _emit_text_delta(remaining):
                    yield ev
            else:
                for block_type, piece in think_router.process(remaining):
                    if block_type == "text" and piece:
                        async for ev in _emit_text_delta(piece):
                            yield ev

        if not reasoning_parser:
            for block_type, piece in think_router.flush():
                if block_type == "text" and piece:
                    async for ev in _emit_text_delta(piece):
                        yield ev

        # Codex round-3 BLOCKING #3: if the reasoning cap latched on the
        # last engine chunk of the stream (terminal exact-boundary case
        # OR the model stopped immediately after overflow), the
        # ``</think>`` close marker was never spliced into the parser —
        # so any held content past the cap stays buffered and the
        # client sees a silent reasoning-only response with no
        # ``output_text.delta`` ever emitted. Force the injection here
        # so a terminal cap-hit flips the parser to content and any
        # trailing bytes are promoted to ``response.output_text.delta``.
        # Idempotent via ``_reasoning_close_injected``.
        terminal_injection_emitted = False
        if (
            reasoning_parser is not None
            and _reasoning_cap_hit
            and not _reasoning_close_injected
        ):
            _reasoning_close_injected = True
            previous_raw = accumulated_raw
            injected_delta = "</think>"
            accumulated_raw = previous_raw + injected_delta
            try:
                final_inject = reasoning_parser.extract_reasoning_streaming(
                    previous_raw, accumulated_raw, injected_delta
                )
            except Exception as e:
                # Codex round-5 BLOCKING #3: an earlier draft emitted a
                # diagnostic string ``"[reasoning cap hit — parser
                # flush failed]"`` as ``response.output_text.delta``,
                # which fabricates assistant content from an INTERNAL
                # server failure — clients see an "answer" that the
                # model never produced. Log the parser failure and
                # leave the assistant content empty. The route's
                # existing 5xx / disconnect-guard semantics handle
                # truly catastrophic failures upstream; a single
                # reasoning-cap parser bug must not invent text.
                logger.warning(
                    "responses terminal close-marker injection raised on %r: %s — "
                    "trailing reasoning content (if any) will not be "
                    "promoted to output_text.delta for this request",
                    type(reasoning_parser).__name__,
                    e,
                )
                final_inject = None
            if final_inject is not None and getattr(final_inject, "content", None):
                content = strip_special_tokens(final_inject.content)
                if content:
                    filtered = tool_filter.process(content)
                    if filtered:
                        async for ev in _emit_text_delta(filtered):
                            yield ev
                        terminal_injection_emitted = True

        # Codex round-4 BLOCKING #1: when the terminal injection above
        # emitted content, the parser already flushed its held buffer
        # for the spliced-in ``</think>``. Running ``finalize_streaming``
        # next on the now-mutated ``accumulated_raw`` (which includes
        # the forged close marker) risks re-emitting the SAME bytes a
        # second time — the qwen3 / deepseek parsers' ``finalize_streaming``
        # is a non-stream re-parse of the whole accumulated buffer and
        # cannot distinguish "content already streamed" from "content
        # still held". Skip the finalize pass when terminal injection
        # already produced output. (When the injection fell through
        # without emitting — buggy parser, no held content — the
        # finalize path still runs as a fallback.)
        if reasoning_parser and accumulated_raw and not terminal_injection_emitted:
            final_msg = (
                reasoning_parser.finalize_streaming(accumulated_raw)
                if hasattr(reasoning_parser, "finalize_streaming")
                else None
            )
            if final_msg and final_msg.content:
                content = strip_special_tokens(final_msg.content)
                if content:
                    async for ev in _emit_text_delta(content):
                        yield ev

        # Close the message item if we ever opened it.
        if message_open:
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": {
                        "type": "message",
                        "id": message_item_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": accumulated_text,
                                "annotations": [],
                            }
                        ],
                    },
                },
            )

        # Emit function_call items for every tool call we saw.
        # Pass `accumulated_raw` (pre-filter model output) not
        # `accumulated_text` (post-filter user-visible text) — tool_filter
        # rightly suppresses `<tool_call>...</tool_call>` XML from
        # `accumulated_text`, but the post-loop parser needs that XML
        # to extract structured tool_calls. Without this swap, the
        # text-parser path returned zero tool_calls and Codex's agent
        # loop silently terminated with no items emitted.
        _, tool_calls = _parse_tool_calls_with_parser(
            accumulated_raw,
            openai_request,
            structured_tool_calls=accumulated_structured_tool_calls or None,
        )

        tool_output_index = (message_output_index + 1) if message_open else 0
        for tc in tool_calls or []:
            fc_id = f"fc_{uuid.uuid4().hex[:24]}"
            yield _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": tool_output_index,
                    "item": {
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
            # Codex CLI accepts the args as a single delta — we don't
            # have token-by-token streaming for tool_call arguments in
            # the underlying engine yet, so emit the whole JSON string
            # at once. Codex concatenates these the same way regardless
            # of chunk count.
            yield _sse(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": fc_id,
                    "output_index": tool_output_index,
                    "delta": tc.function.arguments or "",
                },
            )
            yield _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": tool_output_index,
                    "item": {
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "",
                        "status": "completed",
                    },
                },
            )
            tool_output_index += 1

        # response.completed — terminal event. Codex treats a missing
        # one as a hard failure (it logs "stream closed before
        # response.completed").
        cached_tokens_clamped = min(cached_tokens, prompt_tokens)
        usage_payload = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cached_tokens_clamped:
            usage_payload["input_tokens_details"] = {
                "cached_tokens": cached_tokens_clamped
            }
        yield _sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "completed",
                    "model": served_model,
                    "usage": usage_payload,
                },
            },
        )

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"Responses (stream): prompt={prompt_tokens} + "
            f"completion={completion_tokens} tokens in {elapsed:.2f}s "
            f"({tokens_per_sec:.1f} tok/s)"
        )

    except Exception as e:  # noqa: BLE001
        # response.failed gives Codex a clean shutdown signal instead of
        # a half-stream-then-EOF; matches how the OpenAI cloud
        # Responses API closes errored streams.
        logger.exception("Responses stream failed: %s", e)
        yield _sse(
            "response.failed",
            {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "status": "failed",
                    "error": {
                        "code": "internal_error",
                        "message": str(e),
                    },
                },
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_request(req: ResponsesRequest) -> None:
    """One-line request log mirroring the other route surfaces."""
    if isinstance(req.input, str):
        n_items = 1
        total_chars = len(req.input)
    else:
        n_items = len(req.input)
        total_chars = 0
        for item in req.input:
            if isinstance(item.content, str):
                total_chars += len(item.content)
            elif item.content:
                for c in item.content:
                    if c.text:
                        total_chars += len(c.text)
            if item.arguments:
                total_chars += len(item.arguments)
    n_tools = len(req.tools) if req.tools else 0
    instr_chars = len(req.instructions) if req.instructions else 0
    logger.info(
        f"[REQUEST] POST /v1/responses (codex) stream={req.stream} "
        f"model={req.model!r} max_output_tokens={req.max_output_tokens} "
        f"input_items={n_items} total_chars={total_chars} "
        f"instructions_chars={instr_chars} tools={n_tools}"
    )
