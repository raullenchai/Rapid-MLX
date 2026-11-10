# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python list-of-calls tool parser.

Ported from vLLM upstream (vllm/tool_parsers/pythonic_tool_parser.py
and its literal helpers in tool_parsers/utils.py).
Interfaces use Rapid-MLX dictionaries and optional request/token arguments.
Registration, native-template support, reset state and request allowlisting
(with drop accounting and contiguous client indices) follow local contracts.
The upstream regex timeout defaults to one second; use that value directly
because this runtime has no corresponding timeout environment setting.
Positional arguments are not supported, preserving upstream parity.
"""

import ast
import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any, cast

import regex as re

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)


class UnexpectedAstError(Exception):
    """Raised when the AST structure does not match the expected
    pythonic tool call format."""

    pass


_JSON_NAME_LITERALS = {
    "null": None,
    "true": True,
    "false": False,
}


def get_parameter_value(val: ast.expr) -> Any:
    """Extract a Python literal value from an AST expression node.

    Handles constants, dicts, lists, and JSON-style name literals
    (null, true, false) that some models produce instead of Python
    literals (None, True, False).

    Raises:
        UnexpectedAstError: If the AST node is not a supported literal type.

    """
    if isinstance(val, ast.Constant):
        if val.value is None or isinstance(val.value, (str, int, float)):
            return val.value
        # bytes/Ellipsis/complex constants have no JSON representation and
        # would otherwise surface as a TypeError deep inside json.dumps;
        # reject them explicitly like other unsupported nodes.
        logger.warning(
            "Non-JSON-representable constant in tool call arguments: %s",
            ast.dump(val),
        )
        raise UnexpectedAstError("Tool call arguments must be JSON values")
    elif isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in val.keys):
            logger.warning(
                "Dict argument keys are not all literals: %s",
                ast.dump(val),
            )
            raise UnexpectedAstError("Dict tool call arguments must have literal keys")
        return {
            k.value: get_parameter_value(v)  # type: ignore
            for k, v in zip(val.keys, val.values)
        }
    elif isinstance(val, ast.List):
        return [get_parameter_value(v) for v in val.elts]
    elif isinstance(val, ast.Tuple):
        # JSON has no tuple type; a tuple argument (e.g. ``size=(800, 600)``)
        # is treated as a list so it round-trips through ``json.dumps``.
        # Without this the whole call is dropped.
        return [get_parameter_value(v) for v in val.elts]
    elif isinstance(val, ast.Set):
        # JSON has no set type either; a set argument (e.g.
        # ``tags={'a', 'b'}``) is treated as a list, preserving source order.
        return [get_parameter_value(v) for v in val.elts]
    elif isinstance(val, ast.JoinedStr) and all(
        isinstance(part, ast.Constant) for part in val.values
    ):
        # An f-string without placeholders (``f'hello'``) is a plain string
        # constant, but Python parses it as JoinedStr rather than Constant;
        # without this branch the whole call is dropped. F-strings with real
        # placeholders still fall through to the raise below.
        return "".join(
            str(part.value)  # type: ignore
            for part in val.values
        )
    elif isinstance(val, ast.Name) and val.id in _JSON_NAME_LITERALS:
        return _JSON_NAME_LITERALS[val.id]
    elif isinstance(val, ast.UnaryOp) and isinstance(val.op, (ast.USub, ast.UAdd)):
        # A negative (or explicitly positive) number is parsed by Python as a
        # unary operation over a numeric constant, e.g. ``-1`` becomes
        # ``UnaryOp(USub, Constant(1))`` rather than a plain ``Constant(-1)``.
        # These are extremely common tool arguments (negative longitudes,
        # offsets, deltas); without this branch the whole call is dropped.
        # Restrict to numeric operands so ``not``/``~`` and other expressions
        # still raise below.
        operand = get_parameter_value(val.operand)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(val.op, ast.USub) else operand
        logger.warning(
            "Unsupported unary operand in tool call arguments: %s",
            ast.dump(val),
        )
        raise UnexpectedAstError("Tool call arguments must be literals")
    else:
        logger.warning(
            "Unsupported AST node type in tool call arguments: %s",
            ast.dump(val),
        )
        raise UnexpectedAstError("Tool call arguments must be literals")


def _ast_callable_dotted_name(node: ast.expr) -> str:
    """Return the dotted name for a call target, walking `ast.Attribute`
    chains so `a.b.c(...)` becomes `"a.b.c"`.

    Raises:
        UnexpectedAstError: If the chain does not bottom out in an
            `ast.Name` (e.g. subscript or call expression as receiver).

    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise UnexpectedAstError("Invalid tool call name")
    parts.append(current.id)
    return ".".join(reversed(parts))


def handle_single_tool(call: ast.Call) -> dict[str, Any]:
    """Convert a single AST function call node into a ToolCall object.

    Accepts both bare names (`foo(...)`) and dotted attribute chains
    (`a.b.c(...)`); the resulting tool call `name` field preserves the
    dotted form.

    Raises:
        UnexpectedAstError: If the call target is neither a simple name
            nor a chain of attribute accesses bottoming out in a name.

    """
    if not isinstance(call.func, (ast.Name, ast.Attribute)):
        logger.warning(
            "Tool call has non-simple function name: %s",
            ast.dump(call.func),
        )
        raise UnexpectedAstError("Invalid tool call name")
    function_name = _ast_callable_dotted_name(call.func)
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = get_parameter_value(keyword.value)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "name": function_name,
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


class PythonicToolParser(ToolParser):
    """Tool call parser for models that produce tool calls in a pythonic style,
    such as Llama 3.2 and Llama 4 models.

    Used when --enable-auto-tool-choice --tool-call-parser pythonic are all set
    """

    # TODO(mdepinet): Possible future improvements:
    #   1. Support text + tools separated by either <|python_tag|> or \n\n
    #   2. Support tools outside of a list (or separated by a semicolon).
    #      This depends on item 1 for consistent streaming.
    # Neither of these are necessary for e.g. ToolACE, but both would help make
    # Llama3.2 models more reliable.

    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("pythonic_bracket",)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.current_tool_id = 0
        self.streamed_args_for_tool: list[str] = []
        self._buffer: str = ""

    def reset(self) -> None:
        super().reset()
        self.current_tool_id = 0
        self.streamed_args_for_tool = []
        self._buffer = ""

    def _declared_calls(self, calls, request):
        # Match the existing parsers' request allowlist and drop accounting.
        names = {
            tool.get("function", {}).get("name")
            for tool in (request or {}).get("tools") or []
        }
        if not names:
            return calls
        accepted = [call for call in calls if call["name"] in names]
        self._undeclared_dropped = len(calls) - len(accepted)
        return accepted

    # Rename for readability. This is NOT a tool id.
    @property
    def current_tool_index(self) -> int:
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        self.current_tool_id = value

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract the tool calls from a complete model response."""
        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = (
                self.TOOL_CALL_REGEX.match(model_output, timeout=1.0) is not None
            )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug(
                "Regex timeout occurred when matching user input: %s", model_output
            )

        if not is_tool_call_pattern:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            module = ast.parse(model_output)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                calls = self._declared_calls(
                    [handle_single_tool(cast(ast.Call, e)) for e in parsed.elts],
                    request,
                )
                return ExtractedToolCallInformation(
                    tools_called=bool(calls), tool_calls=calls, content=None
                )
            else:
                raise UnexpectedAstError("Tool output must be a list of function calls")
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # Treat as regular text
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    @staticmethod
    def _find_matching_bracket(buffer: str, start: int) -> int:
        """Find the closing square bracket using nested bracket depth."""
        bracket_count = 0
        for i in range(start, len(buffer)):
            if buffer[i] == "[":
                bracket_count += 1
            elif buffer[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    return i
        return -1

    def flush_held_content(self, full_text: str) -> str:
        """Release an unterminated bracket segment at generation end."""
        held: str = self._buffer
        self._buffer = ""
        return held

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
        self._buffer += delta_text
        content = ""
        tool_deltas = []
        while self._buffer:
            start = self._buffer.find("[")
            if start == -1:
                content += self._buffer
                self._buffer = ""
                break

            content += self._buffer[:start]
            self._buffer = self._buffer[start:]
            end = self._find_matching_bracket(self._buffer, 0)
            if end == -1:
                break

            segment = self._buffer[: end + 1]
            self._buffer = self._buffer[end + 1 :]
            result = self.extract_tool_calls(segment, request)
            # A valid list containing only undeclared calls has no content.
            # Preserve that distinction so rejected tool wire stays dropped.
            if result.content is not None:
                content += result.content
            for call in result.tool_calls:
                tool_deltas.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "index": self.current_tool_index,
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                )
                self.current_tool_index += 1
                self.streamed_args_for_tool.append(call["arguments"])
                self.prev_tool_call_arr.append(call)

        delta: dict[str, Any] = {}
        if content:
            delta["content"] = content
        if tool_deltas:
            delta["tool_calls"] = tool_deltas
            delta["preserve_post_tool_content"] = True
        return delta or None


ToolParserManager.register_module("pythonic", PythonicToolParser)
