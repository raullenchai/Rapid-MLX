# SPDX-License-Identifier: Apache-2.0
"""
Chat template application logic for BatchedEngine.

Handles enable_thinking, tools, and fallback logic for chat template rendering.
"""

import copy
import json
import logging

logger = logging.getLogger(__name__)


def _build_tool_injection_text(tools: list[dict]) -> str:
    """Build a compact tool definition string for system prompt injection.

    When a chat template doesn't support the ``tools`` parameter natively,
    we inject tool definitions into the system message so the model can
    still see them.

    Args:
        tools: List of tool definitions in OpenAI function-calling format.

    Returns:
        A formatted string describing available tools and calling format.
    """
    lines = ["# Available Tools", ""]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        lines.append(f"## {name}")
        if desc:
            lines.append(f"{desc}")
        if props:
            lines.append(f"Parameters: {json.dumps(props, ensure_ascii=False)}")
        if required:
            lines.append(f"Required: {json.dumps(required)}")
        lines.append("")

    lines.append(
        "When you need to use a tool, respond with a JSON object "
        'containing "name" and "arguments" keys.'
    )

    return "\n".join(lines)


def _inject_tools_into_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Inject tool definitions into the system message.

    If the first message has role ``system``, append to its content.
    Otherwise, prepend a new system message with the tool definitions.

    Args:
        messages: Original messages (not mutated).
        tools: Tool definitions to inject.

    Returns:
        A shallow copy of messages with tool definitions injected.
    """
    injection = _build_tool_injection_text(tools)
    msgs = copy.copy(messages)

    if msgs and msgs[0].get("role") == "system":
        first = dict(msgs[0])
        existing = first.get("content", "")
        # Handle content parts format (multimodal messages)
        if isinstance(existing, list):
            # Append as a new text part
            first["content"] = list(existing) + [
                {"type": "text", "text": "\n\n" + injection}
            ]
        else:
            first["content"] = str(existing) + "\n\n" + injection
        msgs[0] = first
    else:
        msgs.insert(0, {"role": "system", "content": injection})

    return msgs


def apply_chat_template(
    template_applicator,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    model_name: str = "",
) -> str:
    """Apply a chat template to messages with consistent fallback behavior.

    Applies a chat template with consistent fallback for ``enable_thinking``
    and ``tools`` parameters.

    Args:
        template_applicator: Object with ``apply_chat_template`` method
            (tokenizer or processor).
        messages: List of chat messages in OpenAI format.
        tools: Converted tool definitions for the template, or None.
        enable_thinking: Whether to enable thinking mode.
            - True/False: explicit control
            - None: auto-detect (True except for coder models)
        model_name: Model name string, used for auto-detection of
            ``enable_thinking`` when set to None.

    Returns:
        The formatted prompt string.  Falls back to a plain
        ``role: content`` format if the applicator has no
        ``apply_chat_template`` method.
    """
    if not hasattr(template_applicator, "apply_chat_template"):
        # Fallback for models without apply_chat_template.
        # Inject tools into the system prompt so the model still sees
        # function schemas — same treatment as the TypeError fallback
        # below.  Fixes #120.
        if tools:
            messages = _inject_tools_into_messages(messages, tools)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return prompt + "\nassistant:"

    if enable_thinking is None:
        enable_thinking = "coder" not in model_name.lower()

    template_kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        template_kwargs["tools"] = tools

    try:
        return template_applicator.apply_chat_template(messages, **template_kwargs)
    except TypeError as e:
        # Step 1: retry without enable_thinking (many templates don't support it)
        logger.debug("Chat template TypeError, retrying without enable_thinking: %s", e)
        template_kwargs.pop("enable_thinking", None)
        try:
            return template_applicator.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            pass

        # Step 2: template also rejects tools — fall back to prompt injection.
        # Restore enable_thinking: the step-1 pop removed it because we
        # didn't know yet whether the failure was about enable_thinking
        # or about tools.  Now we know it was tools, so re-add
        # enable_thinking for the final retry so thinking-capable models
        # (Qwen, DeepSeek) don't silently lose that feature.  Fixes #122.
        template_kwargs.pop("tools", None)
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if tools:
            logger.info(
                "Chat template doesn't support tools param — "
                "injecting %d tool definitions into system prompt",
                len(tools),
            )
            injected = _inject_tools_into_messages(messages, tools)
            try:
                return template_applicator.apply_chat_template(
                    injected, **template_kwargs
                )
            except TypeError:
                # enable_thinking also unsupported after all — drop it
                template_kwargs.pop("enable_thinking", None)
                return template_applicator.apply_chat_template(
                    injected, **template_kwargs
                )

        return template_applicator.apply_chat_template(messages, **template_kwargs)
