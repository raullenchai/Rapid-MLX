# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for the Raon-1 chat template (``--tool-call-parser raon_1``).

The chat template renders each call as::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

String values are written as-is and other values as JSON; arguments are
returned as a JSON object whose values follow the request's tool schema.
Streaming sends the name first and the arguments as they become stable.

Adapted from the vendor parser. The state machines and schema conversions
are retained; request and output objects use Rapid parser interfaces.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from ..reasoning.raon_1_parser import (
    DISCARD,
    FUNCTION_CLOSE,
    FUNCTION_OPEN,
    PARAMETER_CLOSE,
    PARAMETER_OPEN,
    TAG_CLOSE,
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    Piece,
    RaonMarkupStream,
    RaonMarkupTokens,
    _TextTokenizer,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


@dataclass
class DeltaFunctionCall:
    name: str | None = None
    arguments: str | None = None


@dataclass
class DeltaToolCall:
    index: int
    id: str | None = None
    type: str | None = None
    function: DeltaFunctionCall | None = None


@dataclass
class DeltaMessage:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


@dataclass
class ToolCall:
    id: str
    function: DeltaFunctionCall


FunctionCall = DeltaFunctionCall


@dataclass
class _ExtractedToolCalls:
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


def make_tool_call_id(**kwargs):
    return "call_" + uuid.uuid4().hex[:24]


def _wire(delta):
    if delta is None:
        return None

    def prune(value):
        if isinstance(value, dict):
            return {k: prune(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [prune(v) for v in value]
        return value

    return prune(asdict(delta))


def _request(request):
    if isinstance(request, dict):
        return SimpleNamespace(**request)
    return request or SimpleNamespace()


# Parser states. Text outside a call is content; text between <tool_call> and
# <function= or after </function> belongs to no field and is dropped.
_BEFORE, _CONTENT, _CALL, _NAME, _ARGS, _AFTER = (
    "before",
    "content",
    "call",
    "name",
    "args",
    "after",
)

# Parameter tags, tolerant of whitespace inside the tag.
_PARAMETER = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_NAMED_PARAMETER = re.compile(r"<\s*parameter\s*=\s*([^>]+)>")
_PARAMETER_END = re.compile(r"<\s*/\s*parameter\s*>")
_PARAMETER_NEXT = re.compile(r"<\s*parameter\s*=")
_JSON_NAME = re.compile(r'"name"\s*:\s*"([^"]*)"')

_TYPE_NAMES = {
    "string": ("str", "text", "varchar", "char", "enum"),
    "integer": (
        "int",
        "int32",
        "int64",
        "uint",
        "uint32",
        "uint64",
        "long",
        "short",
        "unsigned",
    ),
    "number": ("float", "float32", "float64", "double"),
    "boolean": ("bool",),
    "object": ("dict",),
    "array": ("arr", "list", "sequence"),
}
_CANONICAL_TYPE = {
    alias: name for name, aliases in _TYPE_NAMES.items() for alias in aliases
}
_TYPE_ORDER = ("null", "integer", "number", "boolean", "object", "array", "string")


def _unwrap(value: str) -> str:
    """Remove the newline the template writes on each side of a value."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _parameters_json(raw: str, partial: bool) -> str:
    """Parameters of one call as a JSON object of strings.

    A parameter ends at ``</parameter>`` or where the next ``<parameter=``
    begins. With ``partial``, a final parameter still being written is
    included.
    """
    params: dict[str, Any] = {}
    kept: list[str] = []
    position = cursor = 0
    while True:
        opening = _PARAMETER.search(raw, position)
        if opening is None:
            break
        closing = _PARAMETER_END.search(raw, opening.end())
        following = _PARAMETER_NEXT.search(raw, opening.end())
        if closing is None and following is None:
            position = opening.start() + 1
            continue
        if following is not None and (
            closing is None or following.start() < closing.start()
        ):
            body_end = end = following.start()
        else:
            assert closing is not None
            body_end, end = closing.start(), closing.end()
        params[opening.group(1)] = _unwrap(raw[opening.end() : body_end])
        kept.append(raw[cursor : opening.start()])
        cursor = position = end
    if partial:
        rest = "".join(kept) + raw[cursor:]
        pending = _NAMED_PARAMETER.search(rest)
        if pending is not None and pending.group(1):
            params[pending.group(1)] = _unwrap(rest[pending.end() :])
    return json.dumps(params, ensure_ascii=False)


def _schema_types(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return {"string"}
    types: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, list):
        types.update(item for item in declared if isinstance(item, str))
    enum = schema.get("enum")
    if isinstance(enum, list):
        for item in enum:
            if item is None:
                types.add("null")
            elif isinstance(item, bool):
                types.add("boolean")
            elif isinstance(item, int):
                types.add("integer")
            elif isinstance(item, float):
                types.add("number")
            elif isinstance(item, str):
                types.add("string")
            elif isinstance(item, list):
                types.add("array")
            elif isinstance(item, dict):
                types.add("object")
    for key in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(key)
        if isinstance(choices, list):
            for choice in choices:
                types |= _schema_types(choice)
    return types or {"string"}


def _finite_json(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return True


def _cast(text: str, types: set[str]) -> Any:
    """Convert a parameter written as text to the first schema type it fits."""
    wanted = {
        _CANONICAL_TYPE.get(name.strip().lower(), name.strip().lower())
        for name in types
    }
    for name in _TYPE_ORDER:
        if name not in wanted:
            continue
        if name == "string":
            return text
        if name == "null":
            if text.lower() == "null":
                return None
        elif name == "integer":
            try:
                return int(text)
            except (ValueError, TypeError):
                pass
        elif name == "number":
            try:
                number = float(text)
            except (ValueError, TypeError):
                continue
            if math.isfinite(number):
                return number if number != int(number) else int(number)
        elif name == "boolean":
            flag = text.lower().strip()
            if flag in ("true", "1"):
                return True
            if flag in ("false", "0"):
                return False
        else:
            try:
                decoded = json.loads(text)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _finite_json(decoded):
                return decoded
    try:
        decoded = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text
    return decoded if _finite_json(decoded) else text


def _conform(value: Any, schema: dict) -> tuple[Any, bool]:
    if isinstance(value, str):
        cast = _cast(value, _schema_types(schema))
        return cast, cast is not value
    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            return value, _conform_object(value, properties)
        return value, False
    if isinstance(value, list):
        items = schema.get("items")
        changed = False
        if isinstance(items, dict):
            for index, item in enumerate(value):
                cast, item_changed = _conform(item, items)
                if item_changed:
                    value[index] = cast
                    changed = True
        return value, changed
    cast = _cast(json.dumps(value, ensure_ascii=False), _schema_types(schema))
    return cast, type(cast) is not type(value) or cast != value


def _conform_object(arguments: dict, properties: dict) -> bool:
    changed = False
    for key, value in arguments.items():
        schema = properties.get(key)
        if not isinstance(schema, dict):
            continue
        cast, value_changed = _conform(value, schema)
        if value_changed:
            arguments[key] = cast
            changed = True
    return changed


def _stable_prefix(arguments: str, string_keys: set[str] | None) -> str:
    """Prefix of the arguments JSON that later text cannot change.

    The last value is left out unless it is an open string of a key that stays
    a string under the schema.
    """
    colon, key, pending_key = -1, None, None
    in_string = escaped = False
    string_start, depth = -1, 0
    for index, char in enumerate(arguments):
        if escaped:
            escaped = False
        elif in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if depth == 1 and string_start >= 0:
                    pending_key = arguments[string_start + 1 : index]
        elif char == '"':
            in_string, string_start = True, index
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == ":" and depth == 1:
            colon, key, pending_key = index, pending_key, None
    if colon < 0:
        return ""
    end = colon + 1
    while end < len(arguments) and arguments[end] in " \t\n\r":
        end += 1
    if end >= len(arguments) or arguments[end] != '"':
        return arguments[:end]
    if string_keys is not None and key not in string_keys:
        return arguments[:end]
    escaped = False
    for index in range(end + 1, len(arguments)):
        char = arguments[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return arguments[:index]
    return arguments


class _Call:
    __slots__ = ("id", "name", "raw", "named", "string_keys", "sent")

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.raw = ""
        self.named = False
        self.string_keys: set[str] | None = None
        self.sent = ""


class RaonToolParser(ToolParser):
    """Parses ``<tool_call>`` blocks into OpenAI ``tool_calls``."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("tool_call_xml_body",)
    engine_based_streaming = True

    def __init__(self, tokenizer: Any = None, tools: list | None = None) -> None:
        super().__init__(tokenizer)
        tokenizer = tokenizer or _TextTokenizer()
        self.markup = RaonMarkupTokens(tokenizer)
        self._stream = RaonMarkupStream(self.markup)
        self._request_tools = tools
        self._suppressed = False
        self._started = False
        self._next_id = 0
        self._reset(_BEFORE)

    # -- state ----------------------------------------------------------------

    def _reset(self, state: str) -> None:
        self._stream.reset()
        self._state = state
        self._index = -1
        self._calls: list[_Call] = []
        self._held_content = ""
        self._content_started = False

    def _use_request(self, request: Any) -> None:
        request = _request(request)
        tools = getattr(request, "tools", None)
        if tools:
            self._request_tools = tools
        if (
            not self._suppressed
            and getattr(request, "tool_choice", None) == "none"
            and tools
        ):
            self._suppressed = True

    def _text(self, value: str) -> list[tuple[str, str, int]]:
        state = self._state
        if state in (_CONTENT, _BEFORE):
            return [("content" if state == _CONTENT else "reasoning", value, -1)]
        if state == _NAME:
            return [("name", value, self._index)]
        if state == _ARGS:
            return [("args", value, self._index)]
        return []

    def _open_call(self, state: str, *events: str) -> list[tuple[str, str, int]]:
        self._index += 1
        self._state = state
        return [(event, "", self._index) for event in events] + [
            ("open", "", self._index)
        ]

    def _events(self, pieces: list[Piece]) -> list[tuple[str, str, int]]:
        events: list[tuple[str, str, int]] = []
        for piece in pieces:
            kind, value, state = piece.kind, piece.value, self._state
            if kind is None:
                if value or piece.always:
                    events += self._text(value)
            elif kind == DISCARD:
                continue
            elif state == _BEFORE and kind in (THINK_START, THINK_END, TOOL_CALL_START):
                if kind == THINK_END:
                    self._state = _CONTENT
                    events.append(("reasoning_end", "", -1))
                elif kind == TOOL_CALL_START:
                    events += self._open_call(_CALL, "reasoning_end")
            elif state == _CONTENT and kind in (
                THINK_END,
                TOOL_CALL_START,
                FUNCTION_OPEN,
            ):
                if kind == TOOL_CALL_START:
                    events += self._open_call(_CALL, "reasoning_end")
                elif kind == FUNCTION_OPEN:
                    events += self._open_call(_NAME)
            elif state == _CALL and kind in (TOOL_CALL_END, FUNCTION_OPEN):
                if kind == TOOL_CALL_END:
                    self._state = _CONTENT
                    events.append(("close", "", self._index))
                else:
                    self._state = _NAME
            elif state == _NAME and kind in (TAG_CLOSE, FUNCTION_CLOSE):
                if kind == TAG_CLOSE:
                    self._state = _ARGS
                else:
                    self._state = _AFTER
                    events.append(("close", "", self._index))
            elif state == _ARGS and kind in (
                FUNCTION_CLOSE,
                PARAMETER_OPEN,
                PARAMETER_CLOSE,
            ):
                if kind == FUNCTION_CLOSE:
                    self._state = _AFTER
                    events.append(("close", "", self._index))
                else:
                    events.append(("args", value, self._index))
            elif state == _AFTER and kind in (
                TOOL_CALL_END,
                TOOL_CALL_START,
                FUNCTION_OPEN,
            ):
                if kind == TOOL_CALL_END:
                    self._state = _CONTENT
                elif kind == TOOL_CALL_START:
                    events += self._open_call(_CALL)
                else:
                    events += self._open_call(_NAME)
            else:
                events += self._text(value)
        return events

    def _finish_events(self) -> list[tuple[str, str, int]]:
        events = self._events(self._stream.finish())
        if self._state in (_CALL, _NAME, _ARGS, _AFTER):
            if self._index >= 0:
                events.append(("close", "", self._index))
            self._state = _CONTENT
        elif self._state == _BEFORE:
            self._state = _CONTENT
            events.append(("reasoning_end", "", -1))
        return events

    # -- tool schema ------------------------------------------------------------

    def _properties(self, name: str) -> dict:
        for tool in self._request_tools or ():
            if hasattr(tool, "model_dump"):
                tool = tool.model_dump()
            if not isinstance(tool, dict):
                continue
            function = tool.get("function", tool)
            if function.get("name") == name:
                properties: dict = (function.get("parameters") or {}).get(
                    "properties", {}
                )
                return properties
        return {}

    def _typed(self, arguments: str, name: str) -> str:
        if not self._request_tools or not name:
            return arguments
        try:
            decoded = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return arguments
        if not isinstance(decoded, dict):
            return arguments
        properties = self._properties(name)
        if not properties:
            return arguments
        return (
            json.dumps(decoded, ensure_ascii=False)
            if _conform_object(decoded, properties)
            else arguments
        )

    def _string_keys(self, name: str) -> set[str] | None:
        properties = self._properties(name)
        if not properties:
            return None
        return {
            key
            for key, schema in properties.items()
            if _schema_types(schema) == {"string"}
        }

    # -- tool call deltas -------------------------------------------------------

    def _assign_id(self, call: _Call, name: str) -> None:
        if not call.id:
            call.id = make_tool_call_id(
                id_type="random", func_name=name, idx=self._next_id
            )
            self._next_id += 1

    def _json_name(self, call: _Call) -> str | None:
        match = _JSON_NAME.search(call.raw)
        return match.group(1) if match and match.group(1) else None

    def _announce(
        self,
        index: int,
        deltas: list[DeltaToolCall],
        name: str | None,
        arguments: str | None = None,
    ) -> bool:
        if not name:
            return False
        call = self._calls[index]
        call.name, call.named = name, True
        call.string_keys = self._string_keys(name)
        self._assign_id(call, name)
        function = (
            DeltaFunctionCall(name=name)
            if arguments is None
            else DeltaFunctionCall(name=name, arguments=arguments)
        )
        deltas.append(
            DeltaToolCall(index=index, id=call.id, type="function", function=function)
        )
        return True

    def _argument_delta(self, call: _Call) -> str | None:
        current = _parameters_json(call.raw, True)
        if call.name:
            current = self._typed(current, call.name)
        stable = _stable_prefix(current, call.string_keys)
        if (
            not stable
            or stable == call.sent
            or (call.sent and not stable.startswith(call.sent))
        ):
            return None
        delta, call.sent = stable[len(call.sent) :], stable
        return delta

    def _argument_rest(self, call: _Call) -> str | None:
        final = self._typed(_parameters_json(call.raw, False), call.name)
        if len(final) <= len(call.sent) or (
            call.sent and not final.startswith(call.sent)
        ):
            return None
        rest, call.sent = final[len(call.sent) :], final
        return rest

    def _on_arguments(
        self, index: int, value: str, deltas: list[DeltaToolCall]
    ) -> None:
        call = self._calls[index]
        if value:
            call.raw += value
        if not call.named:
            if call.name:
                self._announce(index, deltas, call.name)
            elif value:
                self._announce(index, deltas, self._json_name(call))
        elif value:
            delta = self._argument_delta(call)
            if delta:
                deltas.append(
                    DeltaToolCall(
                        index=index, function=DeltaFunctionCall(arguments=delta)
                    )
                )

    def _on_close(self, index: int, deltas: list[DeltaToolCall]) -> None:
        if index >= len(self._calls):
            return
        rest = self._argument_rest(self._calls[index])
        call = self._calls[index]
        if not call.named:
            name = call.name or self._json_name(call) or ""
            if self._announce(index, deltas, name, rest or ""):
                rest = None
        if rest and call.named:
            deltas.append(
                DeltaToolCall(index=index, function=DeltaFunctionCall(arguments=rest))
            )

    @staticmethod
    def _merge(deltas: list[DeltaToolCall]) -> list[DeltaToolCall]:
        merged: dict[int, DeltaToolCall] = {}
        for delta in deltas:
            first = merged.setdefault(delta.index, delta)
            if first is delta:
                continue
            if delta.id is not None and first.id is None:
                first.id = delta.id
            if delta.type is not None and first.type is None:
                first.type = delta.type
            if delta.function is None:
                continue
            if first.function is None:
                first.function = delta.function
                continue
            if delta.function.name is not None and first.function.name is None:
                first.function.name = delta.function.name
            if delta.function.arguments is not None:
                first.function.arguments = (
                    delta.function.arguments
                    if first.function.arguments is None
                    else first.function.arguments + delta.function.arguments
                )
        return deltas if len(merged) == len(deltas) else list(merged.values())

    def _delta(
        self, events: list[tuple[str, str, int]], finished: bool = False
    ) -> DeltaMessage | None:
        if not events and not self._held_content:
            return None
        deltas: list[DeltaToolCall] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        touched_calls = False
        for kind, value, index in events:
            if kind == "content":
                if touched_calls:
                    self._held_content += value
                else:
                    content_parts.append(value)
            elif kind == "reasoning":
                reasoning_parts.append(value)
            elif kind == "reasoning_end" or self._suppressed:
                continue
            else:
                touched_calls = True
                if kind == "open":
                    while len(self._calls) <= index:
                        self._calls.append(_Call())
                elif kind == "name":
                    self._calls[index].name += value
                elif kind == "args":
                    self._on_arguments(index, value, deltas)
                else:
                    self._on_close(index, deltas)
        if len(deltas) > 1:
            deltas = self._merge(deltas)
        if self._held_content and (not touched_calls or not deltas):
            content_parts.insert(0, self._held_content)
            self._held_content = ""
        content = "".join(content_parts)
        if not self._content_started and content:
            if content.strip():
                self._content_started = True
            elif self._calls:
                content = ""
            elif not finished:
                self._held_content, content = content, ""
        reasoning = "".join(reasoning_parts)
        if not (content or deltas or reasoning):
            return None
        fields: dict[str, Any] = {}
        if content:
            fields["content"] = content
        if reasoning:
            fields["reasoning"] = reasoning
        if deltas:
            fields["tool_calls"] = deltas
        return DeltaMessage(**fields)

    def _calls_found(self) -> list[ToolCall]:
        found: list[ToolCall] = []
        for call in self._calls:
            if not call.name and not call.raw:
                continue
            name, raw = call.name.strip(), call.raw
            if not name and raw.strip():
                name, arguments = self._json_call(raw)
            elif raw.strip():
                arguments = _parameters_json(raw, False)
            else:
                arguments = "{}"
            if name:
                self._assign_id(call, name)
                found.append(
                    ToolCall(
                        id=call.id,
                        function=FunctionCall(
                            name=name, arguments=self._typed(arguments, name)
                        ),
                    )
                )
        return found

    @staticmethod
    def _json_call(raw: str) -> tuple[str, str]:
        """A call body written as JSON ``{"name": ..., "arguments": ...}``."""
        raw = raw.strip()
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return "", raw
        if not isinstance(decoded, dict):
            return "", raw
        name = decoded.get("name", "")
        for key in ("arguments", "parameters"):
            if key in decoded:
                value = decoded[key]
                return name, value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False
                )
        return name, json.dumps(
            {key: value for key, value in decoded.items() if key != "name"},
            ensure_ascii=False,
        )

    # -- ToolParser interface ---------------------------------------------------

    def _extract_tool_calls(
        self, model_output: str, request: Any
    ) -> _ExtractedToolCalls:
        self._use_request(request)
        self._reset(_CONTENT)
        events = (
            self._events(self._stream.feed(model_output, [])) + self._finish_events()
        )
        self._delta(events, finished=True)
        calls = self._calls_found()
        # The complete-output API has no later delta/finish callback. Read
        # content from the ordered grammar events, as the reasoning parser
        # does, instead of a delta that may defer post-call text until EOF.
        content = "".join(value for kind, value, _ in events if kind == "content")
        if content and calls:
            content = content.strip()
        return _ExtractedToolCalls(
            tools_called=bool(calls), tool_calls=calls, content=content or None
        )

    def _extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: Any,
    ) -> DeltaMessage | None:
        if not self._started:
            self._started = True
            self._reset(_CONTENT)
        self._use_request(request)
        return self._delta(self._events(self._stream.feed(delta_text, delta_token_ids)))

    def finish_streaming(self) -> DeltaMessage | None:
        events = self._finish_events()
        if events or self._held_content:
            return self._delta(events, finished=True)
        return None

    def reset(self):
        super().reset()
        self._request_tools = None
        self._suppressed = False
        self._started = False
        self._next_id = 0
        self._reset(_CONTENT)

    def extract_tool_calls(self, model_output: str, request: dict | None = None):
        result = self._extract_tool_calls(model_output, request)
        calls = [
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
            for call in result.tool_calls
        ]
        return ExtractedToolCallInformation(result.tools_called, calls, result.content)

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids=None,
        current_token_ids=None,
        delta_token_ids=None,
        request=None,
    ):
        return _wire(
            self._extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids or [],
                current_token_ids or [],
                delta_token_ids or [],
                request,
            )
        )

    def flush_held_content(self, full_text: str) -> str:
        delta = self.finish_streaming()
        return delta.content or "" if delta is not None else ""


ToolParserManager.register_module("raon_1", RaonToolParser)
