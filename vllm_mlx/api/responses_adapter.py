# SPDX-License-Identifier: Apache-2.0
"""
Adapter for converting between OpenAI Responses API and OpenAI Chat
Completions API.

Handles translation of:
- Requests: Responses (with polymorphic ``input`` items) → Chat
- Responses: Chat → Responses ``output[]`` (message + function_call items)

This is a stateless conversion — the route layer enforces statelessness
by 400'ing when ``previous_response_id`` is set. Codex CLI never sends
that field (openai/codex#3841), so the resulting shim covers the real
hot path despite the simplification.
"""

import json
import uuid

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolDefinition,
)
from .responses_models import (
    ResponsesContentItem,
    ResponsesInputItem,
    ResponsesOutputContent,
    ResponsesOutputItem,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)


def responses_to_openai(request: ResponsesRequest) -> ChatCompletionRequest:
    """
    Convert a Responses-API request to an OpenAI Chat Completions request.

    Translation rules:
    - ``instructions`` → system message (prepended)
    - ``input`` (bare string) → single user message
    - ``input[]`` items:
        - ``message`` → assistant or user message (joined text content)
        - ``function_call`` → assistant message with ``tool_calls``
        - ``function_call_output`` → tool-role message with ``tool_call_id``
        - ``reasoning`` → dropped (encrypted blobs we can't replay)
    - ``tools`` (Responses-flat) → Chat-nested ``{type, function:{name, ...}}``
    - ``text.format`` (JSON-schema output) → ``response_format``
    - ``max_output_tokens`` → ``max_tokens``
    - ``parallel_tool_calls`` / ``tool_choice`` (string form) carried through
    """
    messages: list[Message] = []

    if request.instructions:
        messages.append(Message(role="system", content=request.instructions))

    if isinstance(request.input, str):
        messages.append(Message(role="user", content=request.input))
    else:
        for item in request.input:
            converted = _convert_input_item(item)
            messages.extend(converted)

    # Codex 0.136.0 sends BOTH `instructions` (the big system prompt)
    # AND `developer`-role items interleaved with the user turns.
    # After role mapping both become `system`. Qwen / Llama / Gemma
    # chat templates require:
    #   - at most ONE system message
    #   - at position 0
    # …otherwise `raise_exception('System message must be at the
    # beginning.')` fires mid-stream and Codex sees "stream
    # disconnected before completion".
    #
    # Concatenate every system message into a single one at index 0,
    # preserving their relative order so the per-turn `developer`
    # instructions sit *after* `instructions` (where Codex puts them
    # semantically — the per-turn directive refines the base system
    # prompt).
    messages = _merge_system_messages(messages)

    tools = _convert_tools(request.tools)
    tool_choice = _convert_tool_choice(request.tool_choice)
    response_format = _convert_text_format(request.text)

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        # Mirror Anthropic adapter: forward None so the server-side
        # sampling cascade (request > CLI > alias > generation_config >
        # fallback) can fire. Hard-coding here would short-circuit it
        # at the first layer and rob Responses-compat clients of the
        # model author's curated defaults.
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
        stream=request.stream,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=response_format,
        # Forward the per-request reasoning cap so the same enforcement
        # path used by /v1/chat/completions and /v1/messages applies on
        # /v1/responses (upstream vLLM PR #20859 backport).
        reasoning_max_tokens=request.reasoning_max_tokens,
    )


def _merge_system_messages(messages: list[Message]) -> list[Message]:
    """Collapse all system messages into one at index 0.

    Codex 0.136.0 sends BOTH ``instructions`` (the big system prompt)
    AND ``developer``-role items interleaved with user turns. After role
    mapping both become ``system``. Qwen / Llama / Gemma chat templates
    require at most ONE system message at position 0 — otherwise
    ``raise_exception('System message must be at the beginning.')``
    fires mid-stream and Codex sees "stream disconnected".

    Defensive coercion: today every system message reaches this point
    with a string content (``_message_item_to_chat`` joins structured
    content parts), so the join would be safe for current callers. The
    explicit ``_to_text`` guard defends against future paths or hand-
    crafted ``ChatCompletionRequest`` mutations that leave a list / dict
    in ``content`` — without it, ``"\\n\\n".join([list, list])`` would
    raise ``TypeError: sequence item 0: expected str instance, list
    found`` mid-conversion.
    """

    def _to_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("text") or ""
        if isinstance(value, list):
            return "\n".join(_to_text(v) for v in value)
        return ""

    # Branch on role presence, not on whether the merged text is truthy.
    # An empty / unsupported-shape `developer` item still appears as a
    # system-role message after `_message_item_to_chat`, so leaving the
    # list untouched when `system_texts` is empty would let a non-leading
    # system message reach Qwen / Llama / Gemma — the exact template
    # failure this function exists to prevent (codex_review BLOCKING).
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        return messages
    system_texts = [
        t for t in (_to_text(m.content) for m in messages if m.role == "system") if t
    ]
    non_system = [m for m in messages if m.role != "system"]
    if not system_texts:
        # System messages existed but contributed no usable text. Drop
        # them entirely rather than emit an empty system message, which
        # some templates also reject.
        return non_system
    merged = Message(role="system", content="\n\n".join(system_texts))
    return [merged] + non_system


def openai_to_responses(
    response: ChatCompletionResponse,
    model: str,
    request: ResponsesRequest,
    created_at: int,
) -> ResponsesResponse:
    """
    Convert an OpenAI Chat Completions response to a Responses-API
    response shape.

    The ``output`` array is built in Codex CLI's expected order:
    first a ``message`` item with the assistant text, then one
    ``function_call`` item per tool call. An empty assistant turn (only
    tool calls, no text) emits no ``message`` item — matches the
    public Responses API.
    """
    output: list[ResponsesOutputItem] = []
    choice = response.choices[0] if response.choices else None

    if choice:
        text = choice.message.content or ""
        if text:
            output.append(
                ResponsesOutputItem(
                    type="message",
                    id=f"msg_{uuid.uuid4().hex[:24]}",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponsesOutputContent(type="output_text", text=text),
                    ],
                )
            )

        for tc in choice.message.tool_calls or []:
            output.append(
                ResponsesOutputItem(
                    type="function_call",
                    id=f"fc_{uuid.uuid4().hex[:24]}",
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "",
                    status="completed",
                )
            )

    status = _convert_status(choice.finish_reason if choice else None)

    usage = _build_responses_usage(response)

    return ResponsesResponse(
        created_at=created_at,
        model=model,
        status=status,
        output=output,
        usage=usage,
        parallel_tool_calls=bool(request.parallel_tool_calls),
        tool_choice=request.tool_choice or "auto",
        tools=request.tools or [],
        metadata=request.metadata,
        instructions=request.instructions,
    )


# ---------------------------------------------------------------------------
# Internal: input-item conversion
# ---------------------------------------------------------------------------


def _convert_input_item(item: ResponsesInputItem) -> list[Message]:
    """Translate one Responses-API input item to 0+ Chat messages."""
    if item.type == "message":
        return [_message_item_to_chat(item)]
    if item.type == "function_call":
        return [_function_call_to_chat(item)]
    if item.type == "function_call_output":
        return [_function_call_output_to_chat(item)]
    if item.type == "reasoning":
        # The encrypted_content payload is opaque to non-OpenAI backends;
        # dropping reasoning items is the documented fallback. Codex
        # tolerates the absence — it doesn't re-display them anyway.
        return []
    # Unknown item types (local_shell_call, tool_search_call, etc.) are
    # OpenAI-side features Codex won't send to a third-party backend.
    # Silently drop them rather than 400 — defensive against future
    # additions on the OpenAI side.
    return []


_RESPONSES_TO_CHAT_ROLE = {
    # Responses-API "developer" is the new high-priority instruction role
    # (Codex CLI uses it for the system prompt). Qwen / Llama chat
    # templates only know system/user/assistant/tool, so the unmapped
    # "developer" raises `jinja2.TemplateError: Unexpected message role.`
    # mid-stream — visible to Codex as "stream disconnected".
    "developer": "system",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


def _message_item_to_chat(item: ResponsesInputItem) -> Message:
    raw_role = item.role or "user"
    role = _RESPONSES_TO_CHAT_ROLE.get(raw_role, raw_role)
    content = item.content

    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        parts = []
        for c in content:
            if isinstance(c, ResponsesContentItem):
                # input_text and output_text both render as plain text.
                # input_image is dropped here — vision passthrough is a
                # follow-up and Codex CLI does not send images today.
                if c.type in ("input_text", "output_text") and c.text:
                    parts.append(c.text)
            elif isinstance(c, dict):
                # Defensive: client may have sent a raw dict that slipped
                # past Pydantic if validators are loosened later.
                ctype = c.get("type")
                if ctype in ("input_text", "output_text"):
                    t = c.get("text")
                    if t:
                        parts.append(t)
        text = "\n".join(parts)

    return Message(role=role, content=text)


def _function_call_to_chat(item: ResponsesInputItem) -> Message:
    """Replay a prior assistant tool_call. ``call_id`` becomes the OpenAI
    tool_call_id, ``name`` + ``arguments`` populate the function payload.

    Arguments are kept as the original JSON string (the engine never
    re-parses tool_call arguments). Missing args fall back to ``{}``.
    """
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": item.call_id or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.name or "",
                    "arguments": item.arguments or "{}",
                },
            }
        ],
    )


def _function_call_output_to_chat(item: ResponsesInputItem) -> Message:
    """Replay a tool result. Coerce structured output to JSON string."""
    out = item.output
    if isinstance(out, (dict, list)):
        text = json.dumps(out)
    elif out is None:
        text = ""
    else:
        text = str(out)
    return Message(
        role="tool",
        content=text,
        tool_call_id=item.call_id or "",
    )


# ---------------------------------------------------------------------------
# Internal: tools, tool_choice, response_format
# ---------------------------------------------------------------------------


def _convert_tools(tools: list[dict] | None) -> list[ToolDefinition] | None:
    """Convert Responses-flat tool shape to Chat-nested.

    Responses: ``{type: "function", name, description, parameters}``
    Chat:      ``{type: "function", function: {name, description, parameters}}``

    Non-function tool types (web_search, image_generation, code_interpreter,
    file_search, computer_use, etc.) are silently dropped — Codex CLI
    won't send them to a third-party backend, and they have no analog
    on a local engine.
    """
    if not tools:
        return None
    converted: list[ToolDefinition] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        # OpenAI's Responses-flat shape sometimes nests parameters under
        # ``parameters`` and sometimes alongside a ``strict`` flag; we
        # carry both through verbatim — engine layer expects nested.
        name = t.get("name") or t.get("function", {}).get("name", "")
        if not name:
            continue
        converted.append(
            ToolDefinition(
                type="function",
                function={
                    "name": name,
                    "description": t.get("description")
                    or t.get("function", {}).get("description", ""),
                    "parameters": t.get("parameters")
                    or t.get("function", {}).get("parameters")
                    or {"type": "object", "properties": {}},
                },
            )
        )
    return converted or None


def _convert_tool_choice(tool_choice: str | dict | None) -> str | dict | None:
    """Carry through string tool_choice; convert object shape to OpenAI's.

    Responses string values: ``"auto"`` | ``"none"`` | ``"required"`` —
    the same set OpenAI Chat expects, so they pass straight through.

    Object form on Responses is ``{type: "function", name: "..."}``;
    OpenAI Chat wants ``{type: "function", function: {name: "..."}}``.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" and "name" in tool_choice:
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
    return None


def _convert_text_format(text: dict | None) -> ResponseFormat | None:
    """Map Responses ``text.format`` → Chat ``response_format``.

    ``text.format.type`` values:
    - ``"text"`` (default) → no response_format needed; return None
    - ``"json_schema"`` → ResponseFormat with the embedded schema
    - ``"json_object"`` → ResponseFormat type=json_object

    Anything else is silently passed through as None — the engine then
    runs unconstrained, matching what Codex would have got from OpenAI
    if it asked for an unsupported format type.
    """
    if not text:
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None
    ftype = fmt.get("type")
    if ftype == "json_object":
        return ResponseFormat(type="json_object")
    if ftype == "json_schema":
        schema = fmt.get("schema") or fmt.get("json_schema")
        name = fmt.get("name") or "response"
        if not isinstance(schema, dict):
            return None
        return ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name=name,
                description=fmt.get("description"),
                schema=schema,
                strict=bool(fmt.get("strict", False)),
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Internal: response building
# ---------------------------------------------------------------------------


def _convert_status(openai_finish_reason: str | None) -> str:
    """Map OpenAI ``finish_reason`` to Responses ``status``.

    ``"length"`` is the only one Codex CLI reads specially — it
    surfaces as a follow-up prompt to extend. The rest are folded
    into ``"completed"``.
    """
    if openai_finish_reason == "length":
        return "incomplete"
    return "completed"


def _build_responses_usage(response: ChatCompletionResponse) -> ResponsesUsage:
    if not response.usage:
        return ResponsesUsage()
    prompt = response.usage.prompt_tokens
    completion = response.usage.completion_tokens
    cached = 0
    if response.usage.prompt_tokens_details is not None:
        cached = response.usage.prompt_tokens_details.cached_tokens or 0
    cached = min(cached, prompt)
    reasoning = 0
    if response.usage.completion_tokens_details is not None:
        reasoning = response.usage.completion_tokens_details.reasoning_tokens or 0
    return ResponsesUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=prompt + completion,
        input_tokens_details=({"cached_tokens": cached} if cached else None),
        output_tokens_details=({"reasoning_tokens": reasoning} if reasoning else None),
    )
