# SPDX-License-Identifier: MIT
"""DeepSeek-V4-Flash-0731 prompt encoding.

This is a compact serving adapter for the dedicated encoder published with
``deepseek-ai/DeepSeek-V4-Flash-0731``.  The checkpoint deliberately ships no
Jinja chat template, so treating it as generic ChatML produces invalid prompts.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
LATEST_REMINDER = "<｜latest_reminder｜>"
THINK_START = "<think>"
THINK_END = "</think>"
DSML = "｜DSML｜"
_CONTROL_LITERAL_START = re.compile(r"<(?=(?:/?(?:think|reasoning)>|/?｜DSML｜))")
_EXISTING_CONTROL_ESCAPE = re.compile(
    r"<(?P<esc>\u200b+)(?=(?:/?(?:think|reasoning)>|/?｜DSML｜))"
)


def _escape_string_parameter(value: str) -> str:
    """Frame control-looking source text safely inside a DSML string value."""
    value = _EXISTING_CONTROL_ESCAPE.sub(
        lambda match: "<" + (match.group("esc") * 2), value
    )
    return _CONTROL_LITERAL_START.sub("<\u200b", value)


def _escape_argument_value(value: Any) -> Any:
    """Recursively frame every string leaf, including JSON object keys."""
    if isinstance(value, str):
        return _escape_string_parameter(value)
    if isinstance(value, list):
        return [_escape_argument_value(item) for item in value]
    if isinstance(value, dict):
        return {
            _escape_string_parameter(key) if isinstance(key, str) else key: (
                _escape_argument_value(item)
            )
            for key, item in value.items()
        }
    return value


# Keep the small, universal engineering surface near the beginning of large
# agent tool catalogs.  Codex can submit well over one hundred connector and
# plugin tools; sorting every name alphabetically buried ``exec_command`` deep
# in that prompt even when the task explicitly asked the model to inspect the
# repository.  The order remains canonical (and therefore prefix-cacheable),
# but gives the model's most frequently needed local tools the strongest prompt
# position.
_CORE_AGENT_TOOL_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "exec_command",
            "write_stdin",
            "apply_patch",
            "view_image",
            "request_user_input",
        )
    )
}


def _json(value: Any) -> str:
    # Match the checkpoint's published encoder, including its whitespace.
    return json.dumps(value, ensure_ascii=False)


def _tool_schemas(tools: list[dict]) -> str:
    # Only function-schema fields affect model tool calling. Agent clients may
    # attach runtime connector metadata (for example a changing nested
    # ``tools`` inventory on Codex's GitHub gateway); serialising that metadata
    # into the prompt invalidates a 30K+ prefix without changing semantics.
    definitions = [
        {
            key: copy.deepcopy(value)
            for key, value in t.get("function", t).items()
            if key in {"name", "description", "parameters", "strict"}
        }
        for t in tools
    ]
    for definition in definitions:
        description = definition.get("description")
        if isinstance(description, str):
            paragraphs = description.split("\n\n")
            if (
                paragraphs
                and "Required for some features such as Codex" in paragraphs[0]
            ):
                description = "\n\n".join(paragraphs[1:])
            # Codex decorates connector tools with this redundant suffix after
            # the connector is activated.  Removing it keeps the otherwise
            # identical 30K+ system/tool prefix cacheable across tool rounds.
            description = re.sub(
                r" This tool is part of plugin `[^`]+`\.", "", description
            )
            # The decorator also inserts a sentence-delimiter period when the
            # source description did not already have one.  Canonicalising a
            # final period is semantically neutral and keeps Codex's 30K+
            # system/tool prefix stable after connector activation.
            definition["description"] = description.rstrip().removesuffix(".")
    # Agent clients may reorder otherwise-identical tools (and JSON object
    # keys) between turns.  Since the schemas live in the system prefix, that
    # turns a harmless wire-order change into a 30K-token prefix-cache miss.
    # Tool order has no semantic meaning, so canonicalise both levels.
    definitions.sort(
        key=lambda item: (
            _CORE_AGENT_TOOL_ORDER.get(
                str(item.get("name", "")), len(_CORE_AGENT_TOOL_ORDER)
            ),
            str(item.get("name", "")),
        )
    )
    schemas = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) for item in definitions
    )
    return f"""## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{DSML}tool_calls>" block like the following:

<{DSML}tool_calls>
<{DSML}invoke name="$TOOL_NAME">
<{DSML}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{DSML}parameter>
...
</{DSML}invoke>
<{DSML}invoke name="$TOOL_NAME2">
...
</{DSML}invoke>
</{DSML}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

Inside a `string="true"` parameter, escape literal control-looking text by inserting one invisible U+200B immediately after its opening `<`. This applies to `<think>`, `</think>`, `<reasoning>`, `</reasoning>`, and any literal `<｜DSML｜...>` or `</｜DSML｜...>` text in the parameter value. If repository text already has one or more U+200B bytes there, double their count instead. Never escape the structural DSML wrapper tags themselves. The tool parser reverses this framing before invoking the tool, so the tool receives the repository's exact literal bytes.

If thinking_mode is enabled (triggered by {THINK_START}), you MUST output your complete reasoning inside {THINK_START}...{THINK_END} BEFORE any tool calls or final response.

Otherwise, output directly after {THINK_END} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
Before closing an invoke, emit every parameter listed in that tool schema's `required` array. Never emit an empty invoke for a tool that has required parameters.
"""


def _encode_call(call: dict) -> str:
    fn = call.get("function", call)
    name = fn.get("name", "")
    arguments = fn.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"arguments": arguments}
    params = []
    for key, value in arguments.items():
        is_string = isinstance(value, str)
        escaped_value = _escape_argument_value(value)
        rendered = escaped_value if is_string else _json(escaped_value)
        params.append(
            f'<{DSML}parameter name="{key}" string="{str(is_string).lower()}">'
            f"{rendered}</{DSML}parameter>"
        )
    body = "\n".join(params)
    return f'<{DSML}invoke name="{name}">\n{body}\n</{DSML}invoke>'


def _merge_tool_messages(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for original in messages:
        message = copy.deepcopy(original)
        if message.get("role") != "tool":
            merged.append(message)
            continue
        block = f"<tool_result>{message.get('content') or ''}</tool_result>"
        if merged and merged[-1].get("role") == "user":
            prior = merged[-1].get("content") or ""
            merged[-1]["content"] = f"{prior}\n\n{block}" if prior else block
        else:
            merged.append({"role": "user", "content": block})
    return merged


def encode_messages(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    enable_thinking: bool = True,
    add_generation_prompt: bool = True,
) -> str:
    """Encode the OpenAI serving subset using the official 0731 wire format."""
    work = _merge_tool_messages(messages)
    if tools:
        tool_text = _tool_schemas(tools)
        if work and work[0].get("role") == "system":
            content = work[0].get("content") or ""
            work[0]["content"] = f"{content}\n\n{tool_text}" if content else tool_text
        else:
            work.insert(0, {"role": "system", "content": tool_text})

    parts = [BOS]
    last_user = max(
        (
            i
            for i, message in enumerate(work)
            if message.get("role") in {"user", "developer"}
        ),
        default=-1,
    )
    preserve_history_thinking = bool(tools)
    for index, message in enumerate(work):
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            parts.append(content)
        elif role == "latest_reminder":
            parts.extend((LATEST_REMINDER, content))
        elif role in {"user", "developer"}:
            parts.extend((USER, content))
        elif role == "assistant":
            parts.append(ASSISTANT)
            reasoning = message.get("reasoning_content") or ""
            include_reasoning = enable_thinking and (
                preserve_history_thinking or index > last_user
            )
            if include_reasoning:
                parts.extend((THINK_START, reasoning, THINK_END))
            else:
                parts.append(THINK_END)
            parts.append(content)
            calls = message.get("tool_calls") or []
            if calls:
                rendered = "\n".join(_encode_call(c) for c in calls)
                parts.append(f"\n\n<{DSML}tool_calls>\n{rendered}\n</{DSML}tool_calls>")
            parts.append(EOS)
        else:
            raise ValueError(f"Unsupported DeepSeek-V4-0731 message role: {role!r}")
    if (
        add_generation_prompt
        and work
        and work[-1].get("role")
        in {
            "user",
            "developer",
        }
    ):
        parts.extend((ASSISTANT, THINK_START if enable_thinking else THINK_END))
    return "".join(parts)


def is_deepseek_v4_0731(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", "-")
    return "deepseek-v4-flash-0731" in normalized
