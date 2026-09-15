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


class K2HorizonToolParser(ToolParser):
    EXPECTED_WIRE_FORMATS = ("k2_ifm",)
    REASONING_PROTOCOL = "k2_ifm"
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    SUPPORTED_FORMATS = frozenset({"json", "xml", "xml_typed"})
    GROUP_START = "<ifm|tool_calls>"
    GROUP_END = "</ifm|tool_calls>"
    CALL_START = "<ifm|tool_call>"
    CALL_END = "</ifm|tool_call>"
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
        self._pending_tool_start: int | None = None
        self._pre_tool_scan_upto = 0
        self._post_tool_scan_upto = 0
        self._json_group_start: int | None = None
        self._json_scan_upto = 0
        self._json_in_string = False
        self._json_escape = False
        self._suppress_calls = False

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
            if not isinstance(payload, dict):
                raise ValueError("invalid IFM JSON call")
            json_arguments = payload.get("arguments", {})
            if not isinstance(json_arguments, dict):
                raise ValueError("invalid IFM JSON call")
            name = cls._validate_name(payload.get("name"), request)
            props = cls._properties(name, request)
            coerced = {
                key: _coerce_schema_value(value, props.get(key))
                for key, value in json_arguments.items()
            }
            cls._validate_argument_contract(name, coerced, request)
            return name, coerced

        first_arg = body.find(cls.ARG_KEY_START)
        if first_arg < 0:
            name = cls._validate_name(body.strip(), request)
            empty_arguments: dict[str, Any] = {}
            cls._validate_argument_contract(name, empty_arguments, request)
            return name, empty_arguments
        name = cls._validate_name(body[:first_arg].strip(), request)
        props = cls._properties(name, request)
        xml_arguments: dict[str, Any] = {}
        cursor = first_arg
        for match in cls.ARG_RE.finditer(body, first_arg):
            if body[cursor : match.start()].strip():
                raise ValueError("malformed IFM argument tags")
            key = match.group(1).strip()
            explicit_type = (match.group(2) or "").strip()
            if not key or key in xml_arguments:
                raise ValueError("missing or duplicate IFM argument name")
            if wire_format == "xml" and explicit_type:
                raise ValueError("unexpected IFM argument type")
            if wire_format == "xml_typed" and not explicit_type:
                raise ValueError("missing IFM argument type")
            declared_schema = props.get(key)
            schema = declared_schema if isinstance(declared_schema, dict) else None
            if explicit_type and schema is not None:
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
            xml_arguments[key] = _coerce_schema_value(match.group(3), schema)
            cursor = match.end()
        if cursor == first_arg or body[cursor:].strip():
            raise ValueError("malformed IFM argument tags")
        cls._validate_argument_contract(name, xml_arguments, request)
        return name, xml_arguments

    @classmethod
    def _parse_group(
        cls, group: str, request: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        inner = group[len(cls.GROUP_START) : -len(cls.GROUP_END)]
        wire_format = cls._tool_format(request)
        bodies: list[str] = []
        if wire_format == "json":
            decoder = json.JSONDecoder()
            cursor = 0
            while True:
                cursor = cls._skip_whitespace(inner, cursor)
                if cursor == len(inner):
                    break
                if not inner.startswith(cls.CALL_START, cursor):
                    raise ValueError("malformed IFM tool-call group")
                cursor = cls._skip_whitespace(inner, cursor + len(cls.CALL_START))
                body_start = cursor
                try:
                    _payload, cursor = decoder.raw_decode(inner, cursor)
                except json.JSONDecodeError as exc:
                    raise ValueError("malformed IFM JSON call") from exc
                bodies.append(inner[body_start:cursor])
                cursor = cls._skip_whitespace(inner, cursor)
                if not inner.startswith(cls.CALL_END, cursor):
                    raise ValueError("malformed IFM tool-call group")
                cursor += len(cls.CALL_END)
        else:
            matches = list(cls.CALL_RE.finditer(inner))
            if not matches or cls.CALL_RE.sub("", inner).strip():
                raise ValueError("malformed IFM tool-call group")
            bodies = [match.group(1) for match in matches]
        if not bodies:
            raise ValueError("malformed IFM tool-call group")
        calls = []
        for body in bodies:
            name, arguments = cls._parse_call(body, request, wire_format)
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

    @staticmethod
    def _skip_whitespace(text: str, cursor: int) -> int:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    @classmethod
    def _find_group_end(
        cls,
        text: str,
        start: int,
        request: dict[str, Any] | None,
        search_from: int | None = None,
    ) -> int | None:
        """Find a complete group without mistaking JSON string bytes for tags."""
        if cls._tool_format(request) != "json":
            end = text.find(
                cls.GROUP_END,
                search_from
                if search_from is not None
                else start + len(cls.GROUP_START),
            )
            return end + len(cls.GROUP_END) if end >= 0 else None

        json_end, _cursor, _in_string, _escape = cls._scan_json_group_end(
            text,
            start + len(cls.GROUP_START),
            in_string=False,
            escape=False,
        )
        return json_end

    @classmethod
    def _scan_json_group_end(
        cls,
        text: str,
        cursor: int,
        *,
        in_string: bool,
        escape: bool,
    ) -> tuple[int | None, int, bool, bool]:
        """Scan once for a group closer outside JSON string literals."""
        while cursor < len(text):
            char = text[cursor]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                cursor += 1
                continue
            if char == '"':
                in_string = True
                cursor += 1
                continue
            if char == "<":
                remainder = text[cursor:]
                if remainder.startswith(cls.GROUP_END):
                    return (
                        cursor + len(cls.GROUP_END),
                        cursor + len(cls.GROUP_END),
                        False,
                        False,
                    )
                if cls.GROUP_END.startswith(remainder):
                    # Preserve a split delimiter for the next delta.
                    return None, cursor, False, False
            cursor += 1
        return None, cursor, in_string, escape

    def _find_streaming_group_end(
        self,
        text: str,
        start: int,
        request: dict[str, Any] | None,
        search_from: int,
    ) -> int | None:
        if self._tool_format(request) != "json":
            return self._find_group_end(
                text,
                start,
                request,
                search_from=search_from,
            )
        if self._json_group_start != start:
            self._json_group_start = start
            self._json_scan_upto = start + len(self.GROUP_START)
            self._json_in_string = False
            self._json_escape = False
        end, cursor, in_string, escape = self._scan_json_group_end(
            text,
            self._json_scan_upto,
            in_string=self._json_in_string,
            escape=self._json_escape,
        )
        self._json_scan_upto = cursor
        self._json_in_string = in_string
        self._json_escape = escape
        if end is not None:
            self._json_group_start = None
            self._json_scan_upto = 0
            self._json_in_string = False
            self._json_escape = False
        return end

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
    def _visible_post_tool_prefix(cls, text: str) -> str:
        """Return only text proven public by a post-tool reasoning closer."""
        if not any(marker in text for marker in cls.REASONING_ENDS):
            return ""
        return cls._visible_prefix(text)

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
                cls._visible_post_tool_prefix(
                    text[end : next_start if next_start >= 0 else None]
                )
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
            end = self._find_group_end(model_output, cursor, request)
            if end is None:
                if suppress_calls:
                    content = self._without_tool_groups(model_output)
                    return ExtractedToolCallInformation(False, [], content or None)
                return ExtractedToolCallInformation(
                    False, [], self._visible_prefix(model_output)
                )
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
                self._visible_post_tool_prefix(
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
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        suppress_calls = self._request_value(request, "tool_choice") == "none"
        self._suppress_calls = suppress_calls

        if self._pending_tool_start is not None:
            start = self._pending_tool_start
        else:
            start_search = self._content_upto
            if not self._input_reasoning_sanitized and not self._tool_group_seen:
                start_search = max(
                    start_search,
                    self._pre_tool_scan_upto - len(self.GROUP_START) + 1,
                )
            if self._tool_group_seen and not self._post_tool_content_visible:
                start_search = max(
                    start_search,
                    self._post_tool_scan_upto - len(self.GROUP_START) + 1,
                )
            start = current_text.find(self.GROUP_START, start_search)
        if start < 0:
            if self._tool_group_seen:
                if not self._post_tool_content_visible:
                    # K2 has no unambiguous opener for a repeated private
                    # reasoning lane. Keep all post-call bytes until EOF can
                    # select the suffix after the *last* closer. Advance only
                    # the marker-search cursor, retaining content_upto for the
                    # single linear final scan.
                    self._post_tool_scan_upto = len(current_text)
                    return None
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
                self._pre_tool_scan_upto = len(current_text)
                return None
            pending = current_text[self._content_upto :]
            held = self._partial_overlap(pending, self.GROUP_START)
            end = len(current_text) - held
            addition = current_text[self._content_upto : end]
            self._content_upto = end
            return {"content": addition} if addition else None

        initial_upto = self._content_upto
        initial = current_text[self._content_upto : start]
        content_parts = [
            self._visible_post_tool_prefix(initial)
            if self._tool_group_seen and not self._post_tool_content_visible
            else self._visible_prefix(initial)
        ]
        calls: list[dict[str, Any]] = []
        cursor = start
        while cursor >= 0:
            search_from = cursor + len(self.GROUP_START)
            if self._pending_tool_start == cursor:
                # The prior delta already proved there was no closer in the
                # accumulated prefix. Search only the new bytes plus enough
                # overlap for a marker split across the delta boundary.
                search_from = max(
                    search_from,
                    len(current_text) - len(delta_text) - len(self.GROUP_END) + 1,
                )
            group_end = self._find_streaming_group_end(
                current_text,
                cursor,
                request,
                search_from,
            )
            if group_end is None:
                self._content_upto = cursor
                self._pending_tool_start = cursor
                break
            self._pending_tool_start = None
            try:
                calls.extend(self._parse_group(current_text[cursor:group_end], request))
            except (json.JSONDecodeError, TypeError, ValueError):
                # A complete but invalid envelope is deliberately surfaced as
                # content. Latch the same consumed/content-visible state as a
                # completed group so later deltas are not stranded behind the
                # implicit-reasoning hold used before the first group.
                self._tool_group_seen = True
                self._post_tool_content_visible = True
                self._pending_tool_start = None
                if not suppress_calls:
                    content_parts.append(current_text[cursor:group_end])
                self._content_upto = group_end
                next_start = current_text.find(self.GROUP_START, group_end)
                if next_start >= 0:
                    # An invalid envelope is visible text, not a dispatched
                    # tool boundary. Preserve the intervening prose and keep
                    # scanning: a later complete group in this same delta may
                    # still be valid and must not leak as raw markup at EOF.
                    content_parts.append(current_text[group_end:next_start])
                    cursor = next_start
                    continue
                content = "".join(content_parts)
                return {"content": content} if content else None

            next_start = current_text.find(self.GROUP_START, group_end)
            if next_start >= 0:
                content_parts.append(
                    self._visible_post_tool_prefix(current_text[group_end:next_start])
                )
                self._post_tool_content_visible = False
                cursor = next_start
                continue

            # A valid call makes all following bytes ambiguous until EOF: K2
            # can begin another implicit reasoning lane without an opener.
            # Buffer once, then reveal only the suffix after the final closer.
            self._content_upto = group_end
            self._post_tool_scan_upto = len(current_text)
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
            # The parser independently holds/redacts post-call reasoning and
            # returns only bytes proven visible. Allow the postprocessor to
            # preserve those later content deltas after a call was emitted.
            "preserve_post_tool_content": True,
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
        if self._pending_tool_start is not None or self.has_pending_tool_call(
            full_text
        ):
            if self._suppress_calls:
                return ""
            return self._visible_prefix(full_text[self._content_upto :])
        if self._tool_group_seen:
            remaining = full_text[self._content_upto :]
            if self._post_tool_content_visible:
                return remaining
            return self._visible_post_tool_prefix(remaining)
        if not self._input_reasoning_sanitized and not self._tool_group_seen:
            remaining = full_text[self._content_upto :]
            if not any(marker in remaining for marker in self.REASONING_ENDS):
                return ""
            return self._visible_prefix(remaining)
        held = self._partial_overlap(full_text[self._content_upto :], self.GROUP_START)
        return full_text[-held:] if held else ""


# Direct registration keeps the manager's runtime contract while avoiding the
# intentionally broad decorator return type from obscuring this class's type.
ToolParserManager.register_module("k2_horizon", K2HorizonToolParser)
