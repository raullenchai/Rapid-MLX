# SPDX-License-Identifier: Apache-2.0
"""Pure conversion of one owned, complete tool body; never response recovery."""

from __future__ import annotations

import json
import re

from ..tool_call_scan import trim_wrapping_newlines


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(value):
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _value(text, schema):
    text = trim_wrapping_newlines(text)
    if schema.get("type") == "string":
        return text
    try:
        return _json(text)
    except ValueError:
        return text


def _parameters(payload, opener, props):
    """Segment once at sibling boundaries; retain literal inner closers."""
    pattern = re.compile(f"(?:{opener})|(?P<close></parameter>)")
    pairs = []
    name = None
    start = 0
    last_close = None
    for marker in pattern.finditer(payload):
        if marker.group("close") is not None:
            if name is None:
                raise ValueError("parameter closer without opener")
            last_close = marker.start()
            continue
        key = marker[1].strip()
        if name is None:
            if payload[: marker.start()].strip():
                raise ValueError("text before parameters")
        elif last_close is None:
            continue  # opener-shaped text inside the current value
        else:
            if props and key not in props:
                raise ValueError("undeclared parameter sibling")
            pairs.append((name, payload[start:last_close]))
        name, start, last_close = key, marker.end(), None
    if name is not None:
        if last_close is None or payload[last_close + len("</parameter>") :].strip():
            raise ValueError("incomplete parameter")
        pairs.append((name, payload[start:last_close]))
    elif payload.strip():
        raise ValueError("invalid XML arguments")
    return pairs


def decode_body(body: str, format_name: str, schemas: dict) -> tuple[str, object]:
    """Convert only the selected grammar's isolated body, once at completion."""
    stripped = body.strip()
    if format_name == "muse" and stripped.startswith("<atem:invoke"):
        from ..tool_parsers.muse_tool_parser import (
            _INVOKE_OPEN_RE,
            _convert_param_value,
            _walk_invoke,
        )

        opener = _INVOKE_OPEN_RE.match(stripped)
        walked = _walk_invoke(stripped, opener) if opener else None
        if not isinstance(walked, tuple) or walked[2] != len(stripped):
            raise ValueError("invalid invoke")
        name, params, _ = walked
        props = schemas.get(name, {}).get("properties", {})
        arguments = {}
        for key, value in params:
            if key not in arguments and (not props or key in props):
                arguments[key] = _convert_param_value(value, key, props)
        return name, arguments
    if stripped.startswith("{"):
        value = _json(stripped)
        if "function" in value and any(
            key in value for key in ("name", "arguments", "parameters")
        ):
            raise ValueError("conflicting nested tool envelope")
        function = value.get("function", value)
        if not isinstance(function, dict) or not any(
            key in function for key in ("arguments", "parameters")
        ):
            raise ValueError("missing tool arguments")
        if "arguments" in function and "parameters" in function:
            raise ValueError("ambiguous tool arguments")
        name = function.get("name")
        arguments = function.get("arguments", function.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = _json(arguments)
        if not isinstance(name, str) or not name:
            raise ValueError("missing tool name")
        return name, arguments
    function = re.fullmatch(r"<function=([^>]+)>(.*)</function>\s*", stripped, re.S)
    if function is None:
        function = re.fullmatch(
            r'<function name="([^"]+)">(.*)</function>\s*', stripped, re.S
        )
    if function is not None:
        name, payload = function.groups()
        name = name.strip()
        if payload.lstrip().startswith("{"):
            return name, _json(payload)
        props = schemas.get(name, {}).get("properties", {})
        named = '<parameter name="' in payload
        pattern = (
            r'<parameter name="([^"]{1,1024})">'
            if named
            else r"<parameter=([^>]{1,1024})>"
        )
        pairs = _parameters(payload, pattern, props)
        return name, {key: _value(value, props.get(key, {})) for key, value in pairs}
    function = re.fullmatch(
        r"<function>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</function>",
        stripped,
        re.S,
    )
    if function is not None:
        return function[1].strip(), _json(function[2])
    if format_name in {"glm47", "glm4"}:
        match = re.match(r"([^<\s]+)\s*", stripped)
        if match is None:
            raise ValueError("missing GLM name")
        name = match[1]
        payload = stripped[match.end() :]
        # A delayed reasoning closer is grammar control before arguments only.
        if payload.startswith("</think>"):
            payload = payload[len("</think>") :].lstrip()
        cursor = 0
        arguments = {}
        while cursor < len(payload):
            if not payload.startswith("<arg_key>", cursor):
                raise ValueError("invalid GLM argument boundary")
            key_start = cursor + len("<arg_key>")
            key_end = payload.find("</arg_key>", key_start)
            if key_end < 0:
                raise ValueError("incomplete GLM argument key")
            key = payload[key_start:key_end].strip()
            cursor = key_end + len("</arg_key>")
            while cursor < len(payload) and payload[cursor].isspace():
                cursor += 1
            if not payload.startswith("<arg_value>", cursor):
                raise ValueError("missing GLM argument value")
            value_start = cursor + len("<arg_value>")
            search = value_start
            while True:
                value_end = payload.find("</arg_value>", search)
                if value_end < 0:
                    raise ValueError("incomplete GLM argument value")
                after = value_end + len("</arg_value>")
                cursor = after
                while cursor < len(payload) and payload[cursor].isspace():
                    cursor += 1
                if cursor == len(payload) or payload.startswith("<arg_key>", cursor):
                    break
                search = after
            arguments[key] = _value(
                payload[value_start:value_end].strip(),
                schemas.get(name, {}).get("properties", {}).get(key, {}),
            )
        return name, arguments
    raise ValueError("unsupported tool body")
