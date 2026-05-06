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
    AnthropicMessage,
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
    ToolDefinition,
)


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

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature if request.temperature is not None else 0.7,
        top_p=request.top_p if request.top_p is not None else 0.9,
        stream=request.stream,
        stop=request.stop_sequences,
        tools=tools,
        tool_choice=tool_choice,
    )


def openai_to_anthropic(
    response: ChatCompletionResponse,
    model: str,
) -> AnthropicResponse:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages API format.

    Args:
        response: OpenAI ChatCompletionResponse
        model: Model name for the response

    Returns:
        Anthropic Messages API response
    """
    content = []
    choice = response.choices[0] if response.choices else None

    if choice:
        # Add text content
        if choice.message.content:
            content.append(
                AnthropicResponseContentBlock(
                    type="text",
                    text=choice.message.content,
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

    return AnthropicResponse(
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
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
