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

from fastapi import HTTPException

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

# Yuki F13 (0.8.5 dogfood) — allowlist of tool types accepted by the
# /v1/responses lane. Single source of truth: route, adapter, and tests
# all consult this set so the contract can't drift per-call site.
#
#   ``function``          — the standard local-model tool surface
#   ``computer_20251022`` — OpenAI Computer-Use tool input shape
#                           (Ana C-06); routed to UI-TARS when the
#                           loaded model is UI-TARS, otherwise this
#                           route accepts the type but the request
#                           will surface a 400 when the model can't
#                           fulfil it.
#
# Anything else (``web_search``, ``file_search``, ``code_interpreter``,
# ``image_generation``, …) is rejected with a 400 listing the supported
# types — silent acceptance was the pre-0.8.5 behaviour and led to
# clients believing their tool was being invoked (Yuki F13).
SUPPORTED_RESPONSES_TOOL_TYPES: frozenset[str] = frozenset(
    {
        "function",
        "computer_20251022",
    }
)


def _raise_unsupported_tool_type(tool_type: str) -> None:
    """Single source of truth for the F13 envelope (Yuki R1 0.8.5 dogfood).

    Raised by the adapter when an incoming tool entry has a ``type``
    that is not in :data:`SUPPORTED_RESPONSES_TOOL_TYPES`. Routes that
    want to short-circuit before the adapter runs (e.g. SSE prelude
    has already started) call :func:`validate_responses_tool_types`
    directly.
    """
    supported = sorted(SUPPORTED_RESPONSES_TOOL_TYPES)
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    f"Tool type {tool_type!r} is not supported by this "
                    f"server. Supported types: {supported}. "
                    "``computer_20251022`` requires a UI-TARS model to "
                    "be loaded (other vision+tool-calling models may "
                    "not fulfil the request)."
                ),
                "type": "invalid_request_error",
                "code": "unsupported_tool_type",
                "param": "tools",
            }
        },
    )


def validate_responses_tool_types(tools: list[dict] | None) -> None:
    """Raise 400 if any ``tools[i].type`` falls outside the allowlist.

    Idempotent — safe to call from both the route entry point and from
    the adapter's own ``responses_to_openai`` path. The route gate fires
    BEFORE we touch the engine so unsupported requests don't admit a
    scheduler slot.
    """
    if not tools:
        return
    for t in tools:
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        if ttype and ttype not in SUPPORTED_RESPONSES_TOOL_TYPES:
            _raise_unsupported_tool_type(ttype)


def _is_computer_use_tool(tool: dict) -> bool:
    """True iff the tool entry is the OpenAI Computer-Use shape.

    Documented input shape (Ana C-06):
        {"type":"computer_20251022","name":"computer",
         "display_width":1280,"display_height":800,"environment":"linux"}

    Only the ``type`` is load-bearing; ``name``, ``display_*``, and
    ``environment`` are passed through as part of the converted
    function-tool's parameters so a Computer-Use-aware model (UI-TARS)
    sees the screen geometry hints.
    """
    return isinstance(tool, dict) and tool.get("type") == "computer_20251022"


def request_uses_computer_use(request: ResponsesRequest) -> bool:
    """True iff any submitted tool has ``type == "computer_20251022"``.

    Surfaces to the route + adapter so we know when to translate
    ``function_call`` items with ``name=="computer"`` into the
    Computer-Use ``computer_call`` output-item shape (Ana C-06).
    """
    return bool(request.tools) and any(_is_computer_use_tool(t) for t in request.tools)


def validate_responses_tool_choice(
    tool_choice: str | dict | None, tools: list[dict] | None
) -> None:
    """Reject malformed tool_choice up-front (Yuki F6 0.8.5 dogfood).

    The chat-completions lane gates ``tool_choice`` at the prompt
    rendering layer (``routes/chat.py``); /v1/responses skipped this
    check, so ``tool_choice="required"`` and the named-function form
    silently degraded to ``auto``. Mirror the chat-route gate here so
    the contract is enforced at the SAME point in the request
    lifecycle on both surfaces.

    Validation rules:
      * ``"required"`` requires a non-empty ``tools`` array (otherwise
        the model has nothing to choose from).
      * ``{type:"function","name":X}`` requires a tool named ``X`` in
        ``tools``.
      * ``"auto"`` / ``"none"`` / object-without-name pass through.
    """
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice == "required" and not tools:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "tool_choice='required' but the request has "
                            "no 'tools' array — the model has nothing to "
                            "choose from. Either drop tool_choice or "
                            "add at least one tool definition."
                        ),
                        "type": "invalid_request_error",
                        "code": "tool_choice_required_without_tools",
                        "param": "tool_choice",
                    }
                },
            )
        return
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            target = tool_choice.get("name") or (
                (tool_choice.get("function") or {}).get("name")
            )
            if not target:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                "tool_choice.type='function' requires a "
                                "non-empty 'name' field."
                            ),
                            "type": "invalid_request_error",
                            "code": "tool_choice_missing_name",
                            "param": "tool_choice.name",
                        }
                    },
                )
            tool_names = _submitted_tool_names(tools)
            if target not in tool_names:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                f"tool_choice references function "
                                f"{target!r} which is not present in the "
                                "'tools' array. Add the tool definition "
                                "or pick one of the submitted names."
                            ),
                            "type": "invalid_request_error",
                            "code": "tool_choice_unknown_function",
                            "param": "tool_choice.name",
                        }
                    },
                )


def _submitted_tool_names(tools: list[dict] | None) -> set[str]:
    """Extract the set of tool names from the Responses-flat ``tools``
    list. ``computer_20251022`` always maps to ``"computer"`` (see
    ``_convert_tools``), so include that synthetic name too.
    """
    names: set[str] = set()
    if not tools:
        return names
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            n = t.get("name") or (t.get("function") or {}).get("name")
            if n:
                names.add(n)
        elif t.get("type") == "computer_20251022":
            # Force canonical name to match the ``_convert_tools``
            # contract — see Codex r2 BLOCKING fix in that function.
            names.add("computer")
    return names


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
        # H-11: forward the per-request seed so the Responses surface
        # honours determinism the same way /v1/chat/completions does.
        # Without this, the seed declared on ResponsesRequest would
        # parse, validate, and stop here — the ChatCompletionRequest
        # the rest of the pipeline reads would carry None.
        seed=request.seed,
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

    The ``output`` array is built in OpenAI Responses spec order:
        1. ``reasoning`` (when the model produced reasoning content)
        2. ``message`` (assistant text reply)
        3. ``function_call`` / ``computer_call`` (one per tool call)

    Yuki F4 / R10 (0.8.5 dogfood): the prior shim dropped
    ``message.reasoning_content`` entirely even though the
    chat-completions lane returned it for the SAME model + prompt. The
    top-level ``reasoning`` item closes the cross-lane parity gap so a
    Responses-API client walking ``output[i].type == "reasoning"`` finds
    the model's chain-of-thought summary.

    Ana C-06 (0.8.5 dogfood): when the request supplied
    ``tools=[{"type":"computer_20251022",...}]`` and the parser surfaced
    a ``function.name == "computer"`` call, emit a ``computer_call``
    item instead of ``function_call`` so the OpenAI Computer-Use SDK
    contract is honoured.
    """
    output: list[ResponsesOutputItem] = []
    choice = response.choices[0] if response.choices else None

    uses_computer_use = request_uses_computer_use(request)

    if choice:
        # Yuki F4 / R10: emit ``reasoning`` BEFORE ``message`` so
        # walkers that consume ``output[]`` in order see the
        # spec-compliant sequence.
        reasoning_text = getattr(choice.message, "reasoning_content", None) or ""
        if reasoning_text:
            output.append(_build_reasoning_output_item(reasoning_text))

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
            output.append(_build_tool_call_output_item(tc, uses_computer_use))

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
        # Yuki R6 / R7: echo truncation + service_tier on the envelope.
        # Truncation is a no-op at the engine level (see ResponsesRequest
        # docstring) but is echoed so migrating clients see the field
        # round-trip. ``service_tier`` is echoed as the requested value.
        truncation=request.truncation,
        service_tier=request.service_tier,
    )


def _build_reasoning_output_item(reasoning_text: str) -> ResponsesOutputItem:
    """Build the top-level ``reasoning`` output item (Yuki F4 / R10).

    Spec shape (OpenAI Responses):
        {"type":"reasoning","id":"rs_<hex>",
         "summary":[{"type":"summary_text","text":"..."}]}

    rapid-mlx does not run a separate summarization model — the
    ``summary_text`` carries the raw reasoning chain-of-thought
    verbatim. Large reasoning blobs are chunk-capped at the engine
    level via ``reasoning_max_tokens`` (upstream vLLM PR #20859),
    which already runs upstream of this adapter call.
    """
    return ResponsesOutputItem(
        type="reasoning",
        id=f"rs_{uuid.uuid4().hex[:24]}",
        status="completed",
        summary=[{"type": "summary_text", "text": reasoning_text}],
    )


def _build_tool_call_output_item(
    tool_call, uses_computer_use: bool
) -> ResponsesOutputItem:
    """Translate one OpenAI ``ToolCall`` to a Responses-API output item.

    When the request used Computer-Use AND the tool call's function
    name is ``"computer"`` (the canonical UI-TARS function name —
    every emitted UI-TARS tool_call uses that name), emit a
    ``computer_call`` envelope per Ana C-06. The ``action`` field is
    parsed from the JSON arguments string and surfaced in the
    OpenAI-documented shape ``{"type": <verb>, ...kwargs}``.
    """
    if uses_computer_use and (tool_call.function.name or "") == "computer":
        action = _parse_computer_action(tool_call.function.arguments or "")
        return ResponsesOutputItem(
            type="computer_call",
            id=f"cu_{uuid.uuid4().hex[:24]}",
            call_id=tool_call.id,
            status="completed",
            action=action,
            pending_safety_checks=[],
        )
    return ResponsesOutputItem(
        type="function_call",
        id=f"fc_{uuid.uuid4().hex[:24]}",
        call_id=tool_call.id,
        name=tool_call.function.name,
        arguments=tool_call.function.arguments or "",
        status="completed",
    )


def _parse_computer_action(arguments: str) -> dict:
    """Translate the UI-TARS canonical JSON arguments to a Responses
    ``computer_call.action`` envelope.

    UI-TARS parser emits arguments like::

        {"action": "click", "start_box": [128, 128]}

    The OpenAI Computer-Use spec uses ``type`` for the verb instead
    of ``action``. Map ``action`` → ``type`` and pass through the
    remaining kwargs verbatim so a downstream Computer-Use runtime
    can dispatch on ``action.type``.

    Defensive: invalid JSON, non-dict arguments, AND empty / missing
    arguments all degrade to a ``{"type": "unknown", "raw": "..."}``
    envelope. Returning ``{}`` would produce a ``computer_call.action``
    without a ``type`` field — harder for SDK dispatchers than the
    sentinel shape (codex r1 NIT on PR #817).
    """
    if not arguments:
        return {"type": "unknown", "raw": arguments}
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return {"type": "unknown", "raw": arguments}
    if not isinstance(parsed, dict):
        return {"type": "unknown", "raw": arguments}
    out = dict(parsed)
    if "action" in out and "type" not in out:
        out["type"] = out.pop("action")
    # Even with valid JSON, a missing verb is still ambiguous — fall
    # back to the sentinel so the dispatcher can detect the gap.
    if "type" not in out:
        return {"type": "unknown", "raw": arguments}
    # R6-M2: OpenAI's Responses Computer-Use spec uses ``coordinate``
    # for the single-point verbs and ``path=[{"x":x,"y":y}, ...]`` for
    # ``drag`` (two-point). The UI-TARS parser emits the canonical
    # ``point`` / ``start_point`` / ``end_point`` keys (PR #812
    # contract — chat-completions OpenAI lane stays on those for
    # bytes-faithfulness with downstream OpenAI tool_call shape). The
    # Responses-lane translator folds the point-pair into the spec
    # ``path`` array shape. Anthropic's lane uses a different drag
    # shape (``start_coordinate`` + ``coordinate``); the per-lane
    # translators live next to the parser so the two adapters can't
    # drift.
    from ..tool_parsers.ui_tars_tool_parser import (
        translate_to_responses_spec_keys,
    )

    return translate_to_responses_spec_keys(out)


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

    Computer-Use (``computer_20251022``) is translated to a synthetic
    ``function`` tool named ``"computer"`` so the UI-TARS tool parser
    (which always emits ``function.name == "computer"``) sees a matching
    entry in the request. The original ``display_width`` /
    ``display_height`` / ``environment`` hints are placed in the
    function parameters' ``properties`` so they reach the chat template
    and the model can ground its actions to the actual screen geometry
    instead of guessing 0-1000 normalized coords.

    Other non-function tool types (web_search, image_generation,
    code_interpreter, file_search, …) trigger the F13 400 envelope
    — silent acceptance was the pre-0.8.5 behaviour and led migrating
    clients to believe their tool was being invoked.
    """
    if not tools:
        return None
    converted: list[ToolDefinition] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        ttype = t.get("type")
        if ttype == "function":
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
        elif ttype == "computer_20251022":
            # Ana C-06: translate Computer-Use to the canonical
            # ``computer`` function tool. The UI-TARS tool parser ALWAYS
            # emits ``function.name == "computer"`` regardless of which
            # ``name`` field the caller supplied. We force the canonical
            # name here so the post-parse ``computer_call`` translation
            # in ``_build_tool_call_output_item`` (which keys off
            # ``function.name == "computer"``) lights up — even when
            # the caller submitted e.g. ``{type:"computer_20251022",
            # name:"screen"}``. Codex r2 BLOCKING (PR #817): keying the
            # downstream check on the original ``type`` instead of the
            # function name would require threading the tool-type
            # metadata through the engine surface; forcing the name at
            # the boundary is simpler and keeps the parser → adapter
            # contract intact. Screen geometry hints stay in the
            # parameters so the chat template / system prompt can
            # ground the model.
            geometry: dict = {
                "type": "object",
                "properties": {
                    "display_width": {"type": "integer"},
                    "display_height": {"type": "integer"},
                    "environment": {"type": "string"},
                },
                "_computer_use": {
                    "display_width": t.get("display_width"),
                    "display_height": t.get("display_height"),
                    "environment": t.get("environment"),
                },
            }
            converted.append(
                ToolDefinition(
                    type="function",
                    function={
                        # NB: hard-coded ``"computer"`` — see comment above.
                        "name": "computer",
                        "description": "Computer-Use (UI-TARS) GUI action tool",
                        "parameters": geometry,
                    },
                )
            )
        else:
            # Anything outside the allowlist — surface the F13 envelope.
            # The route-level ``validate_responses_tool_types`` normally
            # fires before we reach here, but keep this as defense-in-
            # depth so a future bypass still 400s instead of silently
            # dropping the tool entry.
            _raise_unsupported_tool_type(ttype or "<missing>")
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
