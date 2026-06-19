# SPDX-License-Identifier: Apache-2.0
"""
Adapter for converting between Anthropic Messages API and OpenAI Chat Completions API.

Handles translation of:
- Requests: Anthropic → OpenAI format
- Responses: OpenAI → Anthropic format
- Messages: Content blocks, tool calls, tool results
"""

import json
import re
import uuid

from .anthropic_models import (
    ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS,
    AnthropicMessage,
    AnthropicOutputConfig,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicResponseContentBlock,
    AnthropicToolDef,
    AnthropicUsage,
)
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolDefinition,
)


class AnthropicOutputConfigError(ValueError):
    """Raised when ``output_config`` on a /v1/messages request is malformed.

    Adapter-layer error type — the route layer (``routes/anthropic.py``)
    converts this into ``HTTPException(400)``. Kept distinct from a plain
    ``ValueError`` so the route can match on type without sniffing the
    message string, and to make grep-for-callers trivial. Codex review
    flagged the message string as the validation surface; subclassing
    here gives both ergonomic typing AND a stable string identity.
    """


def anthropic_to_openai(request: AnthropicRequest) -> ChatCompletionRequest:
    """
    Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    Handles:
    - system field → system message
    - Content blocks → OpenAI message format
    - tool_use/tool_result → OpenAI tool_calls/tool messages
    - Anthropic tools → OpenAI tools

    Args:
        request: Anthropic Messages API request

    Returns:
        OpenAI ChatCompletionRequest
    """
    messages = []

    # Convert system to system message
    if request.system:
        if isinstance(request.system, str):
            system_text = request.system
        elif isinstance(request.system, list):
            # System can be a list of content blocks
            parts = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            system_text = "\n".join(parts)
        else:
            system_text = str(request.system)
        # Strip per-request billing/tracking headers injected by some
        # clients (e.g. Claude Code).  These contain a per-request hash
        # that prevents prefix-cache reuse across turn boundaries.
        system_text = re.sub(r"x-anthropic-billing-header:[^\n]*\n?", "", system_text)
        messages.append(Message(role="system", content=system_text))

    # Convert each message
    for msg in request.messages:
        converted = _convert_message(msg)
        messages.extend(converted)

    # Convert tools
    tools = None
    if request.tools:
        tools = [_convert_tool(t) for t in request.tools]

    # Convert tool_choice
    tool_choice = None
    if request.tool_choice:
        tool_choice = _convert_tool_choice(request.tool_choice)

    # Translate ``output_config.format = json_schema`` (Anthropic shape,
    # upstream vLLM PR #42396) into the OpenAI ``response_format`` shape
    # the chat-completions guided-decode pipeline already understands.
    # Adapter-layer validation: invalid shapes raise
    # ``AnthropicOutputConfigError``; the route converts that to HTTP 400.
    response_format = _convert_output_config(request.output_config)

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        # Forward None when the Anthropic client omits the field so the
        # server-side sampling cascade (request > CLI > alias overlay >
        # generation_config.json > fallback) can fire. Hard-coding 0.7
        # / 0.9 here would short-circuit the cascade at layer 1 and rob
        # Anthropic-compat clients of the model author's curated defaults.
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream,
        stop=request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        # Pick 1 (this PR) — upstream vLLM PR #20859 + #42396 backport.
        # Translates ``output_config.effort`` (or legacy
        # ``thinking.budget_tokens``) into a per-request reasoning cap
        # on the OpenAI surface.
        reasoning_max_tokens=_resolve_reasoning_max_tokens(request),
    )


def _resolve_reasoning_max_tokens(request: AnthropicRequest) -> int | None:
    """Pick the reasoning cap from the Anthropic-side fields.

    Precedence (first wins):
      1. ``output_config.effort`` — newer Anthropic SDK shape (v0.22,
         upstream vLLM PR #42396). ``max`` and unset both mean "no cap".
      2. ``thinking.budget_tokens`` — legacy v0.20 shape (upstream vLLM
         PR #20859). Verbatim integer budget.
      3. ``None`` — no cap, model decides.

    Returning ``None`` keeps the OpenAI-side request unchanged so the
    existing global ``cfg.thinking_token_budget`` semantic (additive
    max_tokens headroom for reasoning models) keeps applying — these
    two budgets are independent dials.
    """
    if request.output_config is not None and request.output_config.effort is not None:
        # ``max`` → None (no cap) via the canonical mapping; other
        # values resolve to a concrete integer cap.
        return ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS.get(
            request.output_config.effort
        )
    if isinstance(request.thinking, dict):
        budget = request.thinking.get("budget_tokens")
        if isinstance(budget, int) and budget >= 1:
            return budget
    return None


def openai_to_anthropic(
    response: ChatCompletionResponse,
    model: str,
    *,
    reasoning_enabled: bool = True,
) -> AnthropicResponse:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages API format.

    Args:
        response: OpenAI ChatCompletionResponse
        model: Model name for the response
        reasoning_enabled: Whether the served alias is configured with a
            ``reasoning_parser`` (i.e. structurally capable of producing
            reasoning text). When False, the ``thinking`` block is never
            emitted regardless of what ``reasoning_content`` carries —
            matches Anthropic's public API where non-extended-thinking
            models never emit a ``thinking`` block. Defaults to True so
            external callers that don't pass the flag keep their existing
            behavior (pre-issue #702).

    Returns:
        Anthropic Messages API response
    """
    content = []
    choice = response.choices[0] if response.choices else None

    if choice:
        # Issue #702: emit a ``thinking`` block iff the alias is
        # reasoning-capable AND the reasoning text is genuinely distinct
        # from the answer text.
        #
        # Without this gate, two failure modes leak into Anthropic clients:
        #   (1) An alias with ``reasoning_parser: null`` whose OpenAI-side
        #       response happens to carry ``reasoning_content`` would
        #       still get a ``thinking`` block. Anthropic's public API
        #       never emits one for non-extended-thinking models, so any
        #       client branching on ``content[0].type == "thinking"``
        #       mis-detects capability.
        #   (2) The ``_rescue_silent_drop_from_reasoning`` helper (#569)
        #       deliberately promotes a stuck reasoning trace into
        #       ``content`` so the OpenAI-side message isn't silently
        #       empty. The adapter has no other way to know
        #       ``reasoning_content == content`` is a rescue artifact, so
        #       it would dutifully emit BOTH blocks carrying the same
        #       string — Claude Code / claude-cli / langchain-anthropic
        #       render the same paragraph twice.
        #
        # Both cases collapse to "emit text only" under the same
        # predicate: the reasoning channel must be enabled AND the
        # reasoning bytes must differ from the content bytes (and be
        # non-empty AND non-whitespace). When the predicate fails we
        # still surface the answer as ``text`` — silent drop is the
        # worse failure mode (#569). The whitespace-only guard mirrors
        # ``_rescue_silent_drop_from_reasoning`` which treats
        # ``"   \n"`` as semantically empty — without this gate the
        # adapter would emit a leading ``thinking`` block of pure
        # whitespace that Claude Code surfaces as a blank thought.
        # Codex r1 NIT on PR #705.
        reasoning_text = choice.message.reasoning_content
        text = choice.message.content
        emit_thinking = (
            reasoning_enabled
            and bool(reasoning_text)
            and reasoning_text.strip() != ""
            and reasoning_text != text
        )
        # Add thinking block FIRST so it appears before the answer text,
        # matching Anthropic's extended-thinking SDK convention. Without
        # this block ``<think>...</think>`` reasoning would silently
        # disappear from the non-streaming response — issue #413.
        if emit_thinking:
            content.append(
                AnthropicResponseContentBlock(
                    type="thinking",
                    thinking=reasoning_text,
                )
            )

        # Add text content
        if text:
            content.append(
                AnthropicResponseContentBlock(
                    type="text",
                    text=text,
                )
            )

        # Add tool use blocks
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}

                content.append(
                    AnthropicResponseContentBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=tool_input,
                    )
                )

        stop_reason = _convert_stop_reason(choice.finish_reason)
    else:
        stop_reason = "end_turn"

    # If no content blocks, add empty text
    if not content:
        content.append(AnthropicResponseContentBlock(type="text", text=""))

    # Map the OpenAI prefix-cache field onto Anthropic's usage shape.
    # Per Anthropic's prompt-caching docs the three input fields are
    # mutually exclusive and satisfy
    #     total_input_tokens
    #         = input_tokens
    #         + cache_read_input_tokens
    #         + cache_creation_input_tokens
    # so ``input_tokens`` is "the non-cached share", NOT the whole
    # prompt. We only populate ``cache_read_input_tokens`` — the prefix
    # served from the local KV cache — and leave
    # ``cache_creation_input_tokens`` unset: Anthropic's "creation"
    # specifically means tokens being written between explicit
    # ``cache_control`` breakpoints (billed 1.25x), which has no
    # analog on a local engine that auto-caches every prefix without
    # a billing dimension. Cache fields stay ``None`` when the engine
    # didn't report a hit so clients can keep distinguishing "engine
    # doesn't report" from "engine reported a hit".
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    cached_tokens = 0
    if response.usage and response.usage.prompt_tokens_details is not None:
        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
    # Clamp once so cache_read + input_tokens cannot exceed prompt_tokens —
    # a defensive guard against an upstream over-report (e.g. prefix-cache
    # bookkeeping bug) that would otherwise emit an impossible Anthropic
    # usage block where cache_read_input_tokens > total prompt tokens.
    cached_tokens = min(cached_tokens, prompt_tokens)
    cache_read = cached_tokens if cached_tokens else None
    input_tokens = prompt_tokens - cached_tokens
    return AnthropicResponse(
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
    )


def _convert_message(msg: AnthropicMessage) -> list[Message]:
    """
    Convert an Anthropic message to one or more OpenAI messages.

    Anthropic tool_result blocks (sent as user messages) need to be
    split into separate OpenAI tool messages.

    Args:
        msg: Anthropic message

    Returns:
        List of OpenAI messages
    """
    # Simple string content
    if isinstance(msg.content, str):
        return [Message(role=msg.role, content=msg.content)]

    # Content is a list of blocks
    messages = []
    text_parts = []
    tool_calls_for_assistant = []
    tool_results = []

    for block in msg.content:
        if block.type == "text":
            text_parts.append(block.text or "")

        elif block.type == "tool_use":
            # Assistant message with tool calls
            tool_input = block.input or {}
            tool_calls_for_assistant.append(
                {
                    "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": json.dumps(tool_input),
                    },
                }
            )

        elif block.type == "tool_result":
            # Tool result → OpenAI tool message
            result_content = block.content
            if isinstance(result_content, list):
                # Extract text from content blocks
                parts = []
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                result_content = "\n".join(parts)
            elif result_content is None:
                result_content = ""

            tool_results.append(
                Message(
                    role="tool",
                    content=str(result_content),
                    tool_call_id=block.tool_use_id or "",
                )
            )

    # Build the messages
    if msg.role == "assistant":
        combined_text = "\n".join(text_parts) if text_parts else None
        if tool_calls_for_assistant:
            messages.append(
                Message(
                    role="assistant",
                    content=combined_text or "",
                    tool_calls=tool_calls_for_assistant,
                )
            )
        elif combined_text is not None:
            messages.append(Message(role="assistant", content=combined_text))
        else:
            messages.append(Message(role="assistant", content=""))
    elif msg.role == "user":
        # User messages: collect text parts, then add tool results separately
        if text_parts:
            combined_text = "\n".join(text_parts)
            messages.append(Message(role="user", content=combined_text))

        # Tool results become separate tool messages
        messages.extend(tool_results)

        # If no text and no tool results, add empty user message
        if not text_parts and not tool_results:
            messages.append(Message(role="user", content=""))
    else:
        # Other roles
        combined_text = "\n".join(text_parts) if text_parts else ""
        messages.append(Message(role=msg.role, content=combined_text))

    return messages


def _convert_tool(tool: AnthropicToolDef) -> ToolDefinition:
    """
    Convert an Anthropic tool definition to OpenAI format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    return ToolDefinition(
        type="function",
        function={
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    )


def _convert_tool_choice(tool_choice: dict) -> str | dict | None:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}
    OpenAI: "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
    """
    choice_type = tool_choice.get("type", "auto")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    elif choice_type == "none":
        return "none"

    return "auto"


def _convert_output_config(
    output_config: AnthropicOutputConfig | None,
) -> ResponseFormat | None:
    """Translate Anthropic ``output_config`` → OpenAI ``response_format``.

    Backport of upstream vLLM PR #42396. Only ``format.type == "json_schema"``
    is supported on this surface today; downstream of this call the existing
    chat-completions guided-decode pipeline (``api/guided.py`` + outlines)
    runs unchanged.

    ``output_config.effort`` is intentionally NOT translated here — see the
    docstring on ``AnthropicOutputConfig``. The field is accepted by the
    Pydantic model but Pick 1 (a concurrent PR) owns wiring it through.

    Raises:
        AnthropicOutputConfigError: when ``format.type`` is not
            ``"json_schema"`` or when the ``schema`` field is missing /
            not a JSON object. The route layer converts this to HTTP 400.
    """
    if output_config is None or output_config.format is None:
        return None

    fmt = output_config.format
    fmt_type = fmt.type
    if fmt_type != "json_schema":
        # Mirror the message style of routes/chat.py's 400 responses so
        # error strings on the two surfaces look like siblings.
        raise AnthropicOutputConfigError(
            f"output_config.format.type={fmt_type!r} is not supported on "
            "/v1/messages; only 'json_schema' is accepted. See upstream "
            "vLLM PR #42396 for the backport contract."
        )

    schema = fmt.schema_
    if schema is None:
        raise AnthropicOutputConfigError(
            "output_config.format.schema is required when "
            "output_config.format.type == 'json_schema' on /v1/messages."
        )
    if not isinstance(schema, dict):
        # Pydantic would have already coerced strings/lists away here for
        # the dict-typed field, but guard explicitly so the message stays
        # informative if a future schema type widens.
        raise AnthropicOutputConfigError(
            "output_config.format.schema must be a JSON object "
            f"(got {type(schema).__name__})."
        )

    # ResponseFormatJsonSchema requires ``name`` — default to "response"
    # to match the existing OpenAI surface's behavior when the field is
    # absent (see api/tool_calling.build_json_system_prompt fallback).
    return ResponseFormat(
        type="json_schema",
        json_schema=ResponseFormatJsonSchema(
            name=fmt.name or "response",
            description=fmt.description,
            schema=schema,
            strict=fmt.strict if fmt.strict is not None else False,
        ),
    )


def _convert_stop_reason(openai_reason: str | None) -> str:
    """
    Convert OpenAI finish_reason to Anthropic stop_reason.

    OpenAI: "stop" | "tool_calls" | "length" | "content_filter"
    Anthropic: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
    """
    if openai_reason is None:
        return "end_turn"

    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(openai_reason, "end_turn")
