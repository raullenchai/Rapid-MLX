"""Strict JSON-prefix recognition for provisional tool envelopes."""

import json
import re

_STRING = re.compile(r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"')
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*|(?:\.[0-9]+)?[eE][+-]?[0-9]*)?"
)


def tool_json_prefix(
    text: str, names: set[str] | None = None
) -> tuple[bool, int | None]:
    """Return viability and complete-document end offset.

    A viable prefix can still become an object with a declared name and
    arguments. Syntax and root fields are checked even before the object
    closes; nested strings may contain arbitrary protocol delimiters.
    """
    stack = ["root"]
    key = None
    fields = set()
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char in " \t\r\n":
            pos += 1
            continue
        state = stack[-1]
        if state == "done":
            return "name" in fields and "arguments" in fields, pos
        if state in ("object", "key"):
            if char == "}" and state == "object":
                stack.pop()
                pos += 1
                continue
            if char != '"':
                return False, None
            match = _STRING.match(text, pos)
            if match is None:
                return _string_prefix(text[pos:]), None
            if len(stack) == 2:
                key = json.loads(match.group())
                fields.add(key)
            stack[-1] = "colon"
            pos = match.end()
            continue
        if state == "colon":
            if char != ":":
                return False, None
            stack[-1] = "value"
            pos += 1
            continue
        if state in ("object_end", "array_end"):
            closer = "}" if state == "object_end" else "]"
            if char == closer:
                stack.pop()
            elif char == ",":
                stack[-1] = "key" if closer == "}" else "array_value"
            else:
                return False, None
            pos += 1
            continue
        if state == "array" and char == "]":
            stack.pop()
            pos += 1
            continue
        if state == "root" and char != "{":
            return False, None
        root_value = len(stack) == 2 and state == "value"
        if root_value and key == "name" and char != '"':
            return False, None
        if root_value and key == "arguments" and char not in '{"':
            return False, None
        stack[-1] = (
            "done"
            if state == "root"
            else ("array_end" if state in ("array", "array_value") else "object_end")
        )
        # Container kind must not depend on the outer object's current key.
        if char in "{[":
            stack.append("object" if char == "{" else "array")
            pos += 1
            continue
        if char == '"':
            match = _STRING.match(text, pos)
            if match is None:
                return _string_prefix(text[pos:]), None
            if root_value and key == "name":
                name = json.loads(match.group())
                if not name or names is not None and name not in names:
                    return False, None
            if root_value and key == "arguments":
                try:
                    arguments = json.loads(json.loads(match.group()))
                except json.JSONDecodeError:
                    return False, None
                if not isinstance(arguments, dict):
                    return False, None
            pos = match.end()
            continue
        literal = next(
            (v for v in ("true", "false", "null") if text.startswith(v, pos)), None
        )
        if literal:
            pos += len(literal)
            continue
        if any(v.startswith(text[pos:]) for v in ("true", "false", "null")):
            return True, None
        match = _NUMBER.match(text, pos)
        if match:
            end = match.end()
            if end < len(text) and text[end] in ".eE":
                return bool(_NUMBER_PREFIX.fullmatch(text[pos:])), None
            pos = end
            continue
        return text[pos:] == "-", None
    complete = stack == ["done"]
    return (not complete or "name" in fields and "arguments" in fields), (
        pos if complete else None
    )


def _string_prefix(text: str) -> bool:
    """Recognize unfinished JSON strings, including partial Unicode escapes."""
    pos = 1
    while pos < len(text):
        char = text[pos]
        if ord(char) < 32 or char == '"':
            return False
        if char == "\\":
            pos += 1
            if pos == len(text):
                return True
            if text[pos] == "u":
                digits = text[pos + 1 : pos + 5]
                if any(c not in "0123456789abcdefABCDEF" for c in digits):
                    return False
                if len(digits) < 4:
                    return True
                pos += 4
            elif text[pos] not in '"\\/bfnrt':
                return False
        pos += 1
    return True


class ToolJSONPrefix:
    """Incremental counterpart of tool_json_prefix for held streaming chunks.

    Payload strings and numbers are validated without retaining their text.
    Only root keys, names, and JSON-encoded argument strings need decoding.
    """

    def __init__(self, names: set[str] | None = None):
        self.names = names
        self.viable = True
        self.complete = False
        self.stack = ["root"]
        self.fields: set[str] = set()
        self.key = ""
        self.token = ""
        self.capture: list[str] | None = None
        self.string_role = ""
        self.escaped = False
        self.unicode_left = 0
        self.number = ""
        self.literal = ""

    def feed(self, text: str) -> int:
        """Consume a new chunk, stopping at a complete JSON object or failure."""
        pos = 0
        while pos < len(text) and self.viable and not self.complete:
            char = text[pos]
            if self.token == "string":
                self._string(char)
                pos += 1
                continue
            if self.token == "literal":
                if not self.literal.startswith(char):
                    self.viable = False
                else:
                    self.literal = self.literal[1:]
                    if not self.literal:
                        self.token = ""
                pos += 1
                continue
            if self.token == "number":
                if self._number(char):
                    pos += 1
                    continue
                self.token = ""
                if not self.viable:
                    break
                # A delimiter ends a number and belongs to its container.
                continue
            state = self.stack[-1]
            if char in " \t\r\n":
                pos += 1
                continue
            if state in ("object", "key"):
                if char == "}" and state == "object":
                    self._close()
                elif char == '"':
                    self.string_role = "key" if len(self.stack) == 2 else ""
                    self.capture = [] if self.string_role else None
                    self.token = "string"
                    self.stack[-1] = "colon"
                else:
                    self.viable = False
            elif state == "colon":
                self.viable = char == ":"
                self.stack[-1] = "value"
            elif state in ("object_end", "array_end"):
                closer = "}" if state == "object_end" else "]"
                if char == closer:
                    self._close()
                elif char == ",":
                    self.stack[-1] = "key" if closer == "}" else "array_value"
                else:
                    self.viable = False
            elif state == "array" and char == "]":
                self._close()
            else:
                root_value = len(self.stack) == 2 and state == "value"
                if (
                    state == "root"
                    and char != "{"
                    or root_value
                    and self.key == "name"
                    and char != '"'
                    or root_value
                    and self.key == "arguments"
                    and char not in '{"'
                ):
                    self.viable = False
                    break
                self.stack[-1] = (
                    "done"
                    if state == "root"
                    else "array_end"
                    if state in ("array", "array_value")
                    else "object_end"
                )
                if char in "{[":
                    self.stack.append("object" if char == "{" else "array")
                elif char == '"':
                    self.string_role = self.key if root_value else ""
                    self.capture = (
                        [] if self.string_role in ("name", "arguments") else None
                    )
                    self.token = "string"
                elif char in "-0123456789":
                    self.token = "number"
                    self.number = (
                        "minus" if char == "-" else "zero" if char == "0" else "int"
                    )
                elif char in "tfn":
                    self.token = "literal"
                    self.literal = {"t": "rue", "f": "alse", "n": "ull"}[char]
                else:
                    self.viable = False
            pos += 1
        return pos

    def _close(self) -> None:
        self.stack.pop()
        if self.stack == ["done"]:
            self.complete = True
            self.viable = {"name", "arguments"} <= self.fields

    def _string(self, char: str) -> None:
        if self.unicode_left:
            self.viable = char in "0123456789abcdefABCDEF"
            self.unicode_left -= 1
        elif self.escaped:
            self.escaped = False
            if char == "u":
                self.unicode_left = 4
            else:
                self.viable = char in '"\\/bfnrt'
        elif char == "\\":
            self.escaped = True
        elif char == '"':
            self.token = ""
            if self.capture is not None:
                value = json.loads('"' + "".join(self.capture) + '"')
                if self.string_role == "key":
                    self.key = value
                    self.fields.add(value)
                elif self.string_role == "name":
                    self.viable = bool(value) and (
                        self.names is None or value in self.names
                    )
                else:
                    try:
                        arguments = json.loads(value)
                    except json.JSONDecodeError:
                        self.viable = False
                    else:
                        self.viable = isinstance(arguments, dict)
            self.capture = None
            return
        elif ord(char) < 32:
            self.viable = False
        if self.capture is not None:
            self.capture.append(char)

    def _number(self, char: str) -> bool:
        state = self.number
        digit = char in "0123456789"
        if state == "minus" and digit:
            self.number = "zero" if char == "0" else "int"
        elif state in ("int", "frac", "expdigits") and digit:
            pass
        elif state in ("zero", "int") and char == ".":
            self.number = "dot"
        elif state in ("zero", "int", "frac") and char in "eE":
            self.number = "exp"
        elif state == "dot" and digit:
            self.number = "frac"
        elif state == "exp" and char in "+-":
            self.number = "expsign"
        elif state in ("exp", "expsign") and digit:
            self.number = "expdigits"
        else:
            self.viable = state in ("zero", "int", "frac", "expdigits")
            return False
        return True
