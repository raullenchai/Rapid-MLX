# SPDX-License-Identifier: Apache-2.0
"""Tool parser for K2 Horizon's IFM JSON and XML call envelopes."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from ..api.tool_calling import _coerce_schema_value, validate_json_schema
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


@ToolParserManager.register_module("k2_horizon")
class K2HorizonToolParser(ToolParser):
    EXPECTED_WIRE_FORMATS = ("k2_ifm",)
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    SUPPORTED_FORMATS = frozenset({"json", "xml", "xml_typed"})
    GROUP_START = "<ifm|tool_calls>"
    GROUP_END = "</ifm|tool_calls>"
    ARG_KEY_START = "<ifm|arg_key>"
    ARG_RE = re.compile(
        r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
        r"(?:<ifm\|arg_type>(.*?)</ifm\|arg_type>\s*)?"
        r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
        re.DOTALL,
    )
    CALL_RE = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.DOTALL)
    REASONING_ENDS = (
        "</ifm|think>",
        "</ifm|think_fast>",
        "</ifm|think_faster>",
    )

    def __init__(self, tokenizer=None):
        self._input_reasoning_sanitized = False
        super().__init__(tokenizer)
        self.reset()

    def set_reasoning_sanitized(self, value: bool) -> None:
        """Declare that the upstream reasoning parser owns IFM redaction."""
        self._input_reasoning_sanitized = bool(value)

    def reset(self) -> None:
        super().reset()
        self._content_upto = 0
        self._next_tool_index = 0
        self._tool_group_seen = False
        self._post_tool_content_visible = False

    @staticmethod
    def _request_value(request: dict[str, Any] | None, key: str, default=None):
        if not isinstance(request, dict):
            return default
        return request.get(key, default)

    @classmethod
    def _tool_format(cls, request: dict[str, Any] | None) -> str:
        kwargs = cls._request_value(request, "chat_template_kwargs", {})
        value = (
            kwargs.get("tool_call_format", "xml") if isinstance(kwargs, dict) else "xml"
        )
        return (
            value
            if isinstance(value, str) and value in cls.SUPPORTED_FORMATS
            else "xml"
        )

    @classmethod
    def _declared_tools(cls, request: dict[str, Any] | None) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for tool in cls._request_value(request, "tools", []) or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                result[function["name"]] = function
        return result

    @classmethod
    def _named_choice(cls, request: dict[str, Any] | None) -> str | None:
        choice = cls._request_value(request, "tool_choice")
        if not isinstance(choice, dict):
            return None
        function = choice.get("function")
        return function.get("name") if isinstance(function, dict) else None

    @classmethod
    def _validate_name(cls, name: Any, request: dict[str, Any] | None) -> str:
        if not isinstance(name, str) or not name or any(ch.isspace() for ch in name):
            raise ValueError("invalid IFM tool name")
        declared = cls._declared_tools(request)
        if name not in declared:
            raise ValueError("unknown IFM tool name")
        named = cls._named_choice(request)
        if named is not None and name != named:
            raise ValueError("unexpected IFM tool name")
        return name

    @classmethod
    def _properties(cls, name: str, request: dict[str, Any] | None) -> dict:
        function = cls._declared_tools(request).get(name, {})
        parameters = function.get("parameters") if isinstance(function, dict) else None
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        return properties if isinstance(properties, dict) else {}

    @classmethod
    def _validate_argument_contract(
        cls,
        name: str,
        arguments: dict[str, Any],
        request: dict[str, Any] | None,
    ) -> None:
        function = cls._declared_tools(request).get(name, {})
        parameters = function.get("parameters") if isinstance(function, dict) else None
        if not isinstance(parameters, dict):
            return
        properties = parameters.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        if parameters.get("additionalProperties") is False:
            unknown = set(arguments) - set(properties)
            if unknown:
                raise ValueError("undeclared IFM tool argument")
        required = parameters.get("required", [])
        if isinstance(required, list):
            missing = {
                key for key in required if isinstance(key, str) and key not in arguments
            }
            if missing:
                raise ValueError("missing required IFM tool argument")
        valid, _error = validate_json_schema(arguments, parameters)
        if not valid:
            raise ValueError("IFM tool arguments violate tool schema")

    @classmethod
    def _parse_call(
        cls, body: str, request: dict[str, Any] | None, wire_format: str
    ) -> tuple[str, dict[str, Any]]:
        if wire_format == "json":
            payload = json.loads(body.strip())
            arguments = (
                payload.get("arguments", {}) if isinstance(payload, dict) else None
            )
            if not isinstance(payload, dict) or not isinstance(arguments, dict):
                raise ValueError("invalid IFM JSON call")
            name = cls._validate_name(payload.get("name"), request)
            props = cls._properties(name, request)
            coerced = {
                key: _coerce_schema_value(value, props.get(key))
                for key, value in arguments.items()
            }
            cls._validate_argument_contract(name, coerced, request)
            return name, coerced

        first_arg = body.find(cls.ARG_KEY_START)
        if first_arg < 0:
            return cls._validate_name(body.strip(), request), {}
        name = cls._validate_name(body[:first_arg].strip(), request)
        props = cls._properties(name, request)
        arguments: dict[str, Any] = {}
        cursor = first_arg
        for match in cls.ARG_RE.finditer(body, first_arg):
            if body[cursor : match.start()].strip():
                raise ValueError("malformed IFM argument tags")
            key = match.group(1).strip()
            explicit_type = (match.group(2) or "").strip()
            if not key or key in arguments:
                raise ValueError("missing or duplicate IFM argument name")
            if wire_format == "xml" and explicit_type:
                raise ValueError("unexpected IFM argument type")
            if wire_format == "xml_typed" and not explicit_type:
                raise ValueError("missing IFM argument type")
            schema = props.get(key)
            if explicit_type and isinstance(schema, dict):
                explicit_base = explicit_type.split("[", 1)[0].lower()
                declared_type = schema.get("type")
                declared_types = (
                    {declared_type}
                    if isinstance(declared_type, str)
                    else set(declared_type)
                    if isinstance(declared_type, list)
                    and all(isinstance(value, str) for value in declared_type)
                    else set()
                )
                if declared_types and explicit_base not in declared_types:
                    raise ValueError("IFM argument type contradicts tool schema")
            if schema is None and explicit_type:
                schema = {"type": explicit_type.split("[", 1)[0].lower()}
            arguments[key] = _coerce_schema_value(match.group(3), schema)
            cursor = match.end()
        if cursor == first_arg or body[cursor:].strip():
            raise ValueError("malformed IFM argument tags")
        cls._validate_argument_contract(name, arguments, request)
        return name, arguments

    @classmethod
    def _parse_group(
        cls, group: str, request: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        inner = group[len(cls.GROUP_START) : -len(cls.GROUP_END)]
        matches = list(cls.CALL_RE.finditer(inner))
        if not matches or cls.CALL_RE.sub("", inner).strip():
            raise ValueError("malformed IFM tool-call group")
        wire_format = cls._tool_format(request)
        calls = []
        for match in matches:
            name, arguments = cls._parse_call(match.group(1), request, wire_format)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, allow_nan=False
                    ),
                }
            )
        return calls

    @classmethod
    def _visible_prefix(cls, prefix: str) -> str:
        """Drop K2's prompt-primed reasoning before a native tool group.

        The opening think marker is part of the rendered assistant prefix and
        therefore is not present in generated text.  The closing marker is the
        only trustworthy boundary available to the non-streaming tool parser.
        Plain prose without a protocol closer remains visible.
        """
        boundaries = [prefix.rfind(marker) for marker in cls.REASONING_ENDS]
        boundary = max(boundaries, default=-1)
        if boundary < 0:
            return prefix
        marker = next(
            marker for marker in cls.REASONING_ENDS if prefix.rfind(marker) == boundary
        )
        return prefix[boundary + len(marker) :]

    @classmethod
    def _without_tool_groups(cls, text: str) -> str:
        """Remove native tool envelopes when this request forbids execution."""
        start = text.find(cls.GROUP_START)
        if start < 0:
            return text
        parts = [cls._visible_prefix(text[:start])]
        cursor = start
        while cursor >= 0:
            end = text.find(cls.GROUP_END, cursor + len(cls.GROUP_START))
            if end < 0:
                break
            end += len(cls.GROUP_END)
            next_start = text.find(cls.GROUP_START, end)
            parts.append(
                cls._visible_prefix(text[end : next_start if next_start >= 0 else None])
            )
            cursor = next_start
        return "".join(parts)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        suppress_calls = self._request_value(request, "tool_choice") == "none"
        start = model_output.find(self.GROUP_START)
        if start < 0:
            return ExtractedToolCallInformation(False, [], model_output)
        content_parts = [self._visible_prefix(model_output[:start])]
        calls: list[dict[str, Any]] = []
        cursor = start
        while cursor >= 0:
            end = model_output.find(self.GROUP_END, cursor + len(self.GROUP_START))
            if end < 0:
                if suppress_calls:
                    content = self._without_tool_groups(model_output)
                    return ExtractedToolCallInformation(False, [], content or None)
                return ExtractedToolCallInformation(
                    False, [], self._visible_prefix(model_output)
                )
            end += len(self.GROUP_END)
            try:
                calls.extend(self._parse_group(model_output[cursor:end], request))
            except (json.JSONDecodeError, TypeError, ValueError):
                if suppress_calls:
                    content = self._without_tool_groups(model_output)
                    return ExtractedToolCallInformation(False, [], content or None)
                return ExtractedToolCallInformation(
                    False, [], self._visible_prefix(model_output)
                )
            next_start = model_output.find(self.GROUP_START, end)
            content_parts.append(
                self._visible_prefix(
                    model_output[end : next_start if next_start >= 0 else None]
                )
            )
            cursor = next_start
        content = "".join(content_parts)
        if suppress_calls:
            return ExtractedToolCallInformation(False, [], content or None)
        return ExtractedToolCallInformation(True, calls, content or None)

    @staticmethod
    def _partial_overlap(text: str, marker: str) -> int:
        for size in range(min(len(text), len(marker) - 1), 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del (
            previous_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        suppress_calls = self._request_value(request, "tool_choice") == "none"

        start = current_text.find(self.GROUP_START, self._content_upto)
        if start < 0:
            if self._tool_group_seen:
                # K2 may open another private reasoning lane between tool
                # groups. Hold it until a reasoning closer establishes the
                # visible boundary, then resume ordinary incremental output.
                pending = current_text[self._content_upto :]
                if not (
                    self._input_reasoning_sanitized
                    or self._post_tool_content_visible
                ):
                    boundary = max(
                        (pending.rfind(marker) for marker in self.REASONING_ENDS),
                        default=-1,
                    )
                    if boundary < 0:
                        return None
                    marker = next(
                        marker
                        for marker in self.REASONING_ENDS
                        if pending.rfind(marker) == boundary
                    )
                    self._content_upto += boundary + len(marker)
                    self._post_tool_content_visible = True
                    pending = current_text[self._content_upto :]
                held = self._partial_overlap(pending, self.GROUP_START)
                end = len(current_text) - held
                addition = current_text[self._content_upto : end]
                self._content_upto = end
                return {"content": addition} if addition else None
            if not self._input_reasoning_sanitized:
                # Direct parser callers have not passed through K2's
                # implicit-reasoning parser. Hold the prefix until a closer
                # proves which bytes are visible; K2 always primes reasoning.
                return None
            pending = current_text[self._content_upto :]
            held = self._partial_overlap(pending, self.GROUP_START)
            end = len(current_text) - held
            addition = current_text[self._content_upto : end]
            self._content_upto = end
            return {"content": addition} if addition else None

        initial_upto = self._content_upto
        content_parts = [self._visible_prefix(current_text[self._content_upto : start])]
        calls: list[dict[str, Any]] = []
        cursor = start
        while cursor >= 0:
            end = current_text.find(self.GROUP_END, cursor + len(self.GROUP_START))
            if end < 0:
                self._content_upto = cursor
                break
            end += len(self.GROUP_END)
            try:
                calls.extend(self._parse_group(current_text[cursor:end], request))
            except (json.JSONDecodeError, TypeError, ValueError):
                if suppress_calls:
                    addition = self._without_tool_groups(current_text[initial_upto:])
                    self._content_upto = len(current_text)
                else:
                    addition = self._visible_prefix(current_text[initial_upto:end])
                    self._content_upto = end
                return {"content": addition} if addition else None

            next_start = current_text.find(self.GROUP_START, end)
            if next_start >= 0:
                content_parts.append(self._visible_prefix(current_text[end:next_start]))
                self._post_tool_content_visible = False
                cursor = next_start
                continue

            trailing = current_text[end:]
            if self._input_reasoning_sanitized:
                held = self._partial_overlap(trailing, self.GROUP_START)
                visible_end = len(current_text) - held
                content_parts.append(current_text[end:visible_end])
                self._content_upto = visible_end
                self._post_tool_content_visible = True
            else:
                boundary = max(
                    (trailing.rfind(marker) for marker in self.REASONING_ENDS),
                    default=-1,
                )
                if boundary >= 0:
                    marker = next(
                        marker
                        for marker in self.REASONING_ENDS
                        if trailing.rfind(marker) == boundary
                    )
                    visible_start = end + boundary + len(marker)
                    visible = current_text[visible_start:]
                    held = self._partial_overlap(visible, self.GROUP_START)
                    visible_end = len(current_text) - held
                    content_parts.append(current_text[visible_start:visible_end])
                    self._content_upto = visible_end
                    self._post_tool_content_visible = True
                else:
                    # The completed call is safe to emit, but trailing bytes
                    # are not public until a reasoning closer or EOF proves it.
                    self._content_upto = end
                    self._post_tool_content_visible = False
            break

        if not calls:
            prefix = "".join(content_parts)
            return {"content": prefix} if prefix else None

        content = "".join(content_parts)
        if suppress_calls:
            self._tool_group_seen = True
            return {"content": content} if content else None
        first_index = self._next_tool_index
        self._next_tool_index += len(calls)
        self._tool_group_seen = True
        return {
            "content": content or None,
            "tool_calls": [
                {
                    "index": first_index + index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for index, call in enumerate(calls)
            ],
        }

    def has_pending_tool_call(self, text: str) -> bool:
        start = text.rfind(self.GROUP_START)
        return start >= 0 and self.GROUP_END not in text[start:]

    def flush_held_content(self, full_text: str) -> str:
        if self.has_pending_tool_call(full_text):
            return self._visible_prefix(full_text[self._content_upto :])
        if self._tool_group_seen:
            return self._visible_prefix(full_text[self._content_upto :])
        if not self._input_reasoning_sanitized and not self._tool_group_seen:
            remaining = full_text[self._content_upto :]
            if not any(marker in remaining for marker in self.REASONING_ENDS):
                return ""
            return self._visible_prefix(remaining)
        held = self._partial_overlap(full_text[self._content_upto :], self.GROUP_START)
        return full_text[-held:] if held else ""
