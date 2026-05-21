# SPDX-License-Identifier: Apache-2.0
"""Safety normalization for messages before Jinja chat-template rendering."""

import json
from typing import Any


def _close_dangling_think_before_tool_call(content: str) -> str:
    """Keep raw tool XML out of an unterminated ``<think>`` section.

    Qwen 3.6 can produce assistant history where ``<think>`` is opened and a
    raw ``<tool_call>`` follows before ``</think>``. Rendering that history as-is
    conditions the next turn as though the tool call is still reasoning. Close
    the dangling thinking span immediately before the first tool call.

    This mirrors the template-side repair described by Cheuk-Yiu Chan:
    https://allanchan339.github.io/bug-fixes/2026/05/02/Qwen36-27B-updated-jinja.html
    """
    if "<tool_call>" not in content or "<think>" not in content:
        return content

    last_think = content.rfind("<think>")
    last_close = content.rfind("</think>")
    tool_pos = content.find("<tool_call>")
    if last_close >= last_think and last_close != -1:
        return content
    if tool_pos > last_think:
        return content[:tool_pos] + "</think>" + content[tool_pos:]
    return content + "</think>"


def _message_to_dict(message: Any) -> dict[str, Any] | Any:
    """Convert OpenAI message model objects without stringifying them."""
    if isinstance(message, dict):
        return dict(message)
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return {
            key: value
            for key, value in model_dump(exclude_none=True).items()
            if value is not None
        }
    legacy_dict = getattr(message, "dict", None)
    if callable(legacy_dict):
        return {k: v for k, v in legacy_dict().items() if v is not None}
    return message


def normalize_messages_for_chat_template(messages: list[Any]) -> list[dict]:
    """Return a JSON-safe copy of messages for chat-template rendering.

    Normalizations:
    - close dangling ``<think>`` spans before raw ``<tool_call>`` XML in
      assistant content
    - convert OpenAI tool-call argument JSON strings to mappings for templates
      that iterate argument keys
    """
    normalized = json.loads(
        json.dumps([_message_to_dict(message) for message in messages], default=str)
    )
    for message in normalized:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _close_dangling_think_before_tool_call(content)

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments: Any = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed = {"value": arguments}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            function["arguments"] = parsed
    return normalized
