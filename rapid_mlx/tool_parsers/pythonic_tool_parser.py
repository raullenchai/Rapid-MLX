# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python list-of-calls tool parser.

Ported from vLLM upstream (vllm/tool_parsers/pythonic_tool_parser.py
and its literal, partial-Python and delta helpers in tool_parsers/utils.py).
Interfaces use Rapid-MLX dictionaries and optional request/token arguments.
Registration, native-template support, reset state and request allowlisting
(with drop accounting and contiguous client indices) follow local contracts.
The upstream regex timeout defaults to one second; use that value directly
because this runtime has no corresponding timeout environment setting.
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


def _is_escaped(text: str, index: int) -> bool:
    """Whether the character at `index` is backslash-escaped.

    A character is escaped iff it is preceded by an *odd* number of consecutive
    backslashes. Checking only the single preceding character is wrong for even
    runs: in `'ab\\'` the closing quote follows an escaped backslash (`\\\\`)
    and is therefore NOT escaped — it closes the string. Common in regex/code
    arguments such as `r'\\\\b'`.
    """
    backslashes = 0
    j = index - 1
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 1


def make_valid_python(text: str) -> tuple[str, str] | None:
    """Attempt to close all open brackets/quotes to make partial Python valid.

    Used during streaming to parse incomplete tool call expressions by
    appending the necessary closing characters.

    Returns:
        A tuple of (completed_text, added_suffix) if the text can be
        made valid, or None if the text is too incomplete to complete
        meaningfully (e.g. mid-parameter-name or mid-dict-key).

    Raises:
        UnexpectedAstError: If mismatched brackets or parentheses
            are detected.

    """
    bracket_stack: list[str] = []
    for index, char in enumerate(text):
        # Inside a string literal only an unescaped matching quote is
        # significant; brackets are literal text. Without this guard a bracket
        # in a string argument (e.g. `cmd='grep -F "]"'`) corrupts the bracket
        # stack and the whole tool call is rejected as mismatched.
        if bracket_stack and bracket_stack[-1] in {"'", '"'}:
            if char == bracket_stack[-1] and not _is_escaped(text, index):
                bracket_stack.pop()
            continue
        if char in {"[", "(", "{"}:
            bracket_stack.append(char)
        elif char == "]":
            if not bracket_stack or bracket_stack.pop() != "[":
                raise UnexpectedAstError("Mismatched square brackets")
        elif char == ")":
            if not bracket_stack or bracket_stack.pop() != "(":
                raise UnexpectedAstError("Mismatched parentheses")
        elif char == "}":
            if not bracket_stack or bracket_stack.pop() != "{":
                raise UnexpectedAstError("Mismatched curly braces")
        elif char in {"'", '"'}:
            bracket_stack.append(char)

    text = text.rstrip()
    if text.endswith("=") or text.endswith(":"):
        return None
    if bracket_stack and bracket_stack[-1] == "{":
        trailing_dict_text = text[: text.rfind("{")]
        num_keys = trailing_dict_text.count(":")
        num_values = trailing_dict_text.count(",")
        if num_keys <= num_values:
            return None
    if bracket_stack and bracket_stack[-1] == "(":
        trailing_params_text = text[: text.rfind("(")]
        num_full_param_names = trailing_params_text.count("=")
        num_full_param_values = trailing_params_text.count(",")
        if num_full_param_names <= num_full_param_values:
            return None
    if text.endswith(","):
        text = text[:-1]
    if (
        bracket_stack
        and bracket_stack[-1] == "["
        and not text.endswith("[")
        and not text.endswith(")")
    ):
        return None

    _CLOSING = {"[": "]", "(": ")", "{": "}", "'": "'", '"': '"'}
    added_text = ""
    for char in reversed(bracket_stack):
        added_text += _CLOSING[char]

    candidate = text + added_text

    # Streaming partial text can land in shapes the bracket-counting
    # heuristics above don't catch. Two failure modes:
    #   1. Mid-key inside a dict (`..., "k`) closes to `..., "k"}` — a
    #      syntactically invalid mixed dict/set.
    #   2. A bare string inside a dict (`{"k`) closes to `{"k"}` — valid
    #      Python but a *set* literal that is really a truncated dict.
    # Validate the candidate parses and has a body, and treat Set nodes as
    # incomplete — but only when this completion added a `}` itself; a set
    # already closed in the model text is a genuine set argument, not an
    # artifact. Callers whose models emit raw control chars inside string
    # arguments must escape them (see escape_ctrl_chars_in_strings) before
    # calling.
    try:
        module = ast.parse(candidate)
    except SyntaxError:
        return None
    if not module.body:
        return None
    if "}" in added_text:
        for node in ast.walk(module):
            if isinstance(node, ast.Set):
                return None

    return candidate, added_text


def compute_tool_delta(
    previously_sent_args: str,
    new_call: dict[str, Any],
    index: int,
    withheld_suffix: str,
) -> dict[str, Any] | None:
    """Compute the incremental delta between previously streamed arguments
    and the current tool call state.

    Returns:
        A DeltaToolCall with only the new argument characters, or None
        if there is no difference from what was previously sent.

    """
    new_call_args = new_call["arguments"]
    if withheld_suffix:
        if not new_call_args.endswith(withheld_suffix):
            msg = (
                f"Tool call arguments '{new_call_args}' do not end with "
                f"expected withheld suffix '{withheld_suffix}'"
            )
            logger.error(msg)
            raise ValueError(msg)
        new_call_args = new_call_args[: -len(withheld_suffix)]
    if not previously_sent_args:
        return {
            "id": new_call["id"],
            "type": "function",
            "index": index,
            "function": {"name": new_call["name"], "arguments": new_call_args},
        }
    arg_diff = new_call_args[len(previously_sent_args) :]
    return {"index": index, "function": {"arguments": arg_diff}} if arg_diff else None


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

    def reset(self) -> None:
        super().reset()
        self.current_tool_id = 0
        self.streamed_args_for_tool = []

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
        if not current_text.startswith("["):
            return {"content": delta_text}

        try:
            valid_and_added_text = make_valid_python(current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List) or not all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                raise UnexpectedAstError("Tool output must be a list of function calls")
            tool_calls = self._declared_calls(
                [handle_single_tool(cast(ast.Call, e)) for e in parsed.elts], request
            )

            tool_deltas = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = (
                    index < len(tool_calls) - 1 or ")]" not in added_text
                )
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-2] if not new_call_complete else ""
                if not new_call_complete and added_text[-2] == ")":
                    # Function call is incomplete. Withhold the closing bracket.
                    withheld_suffix = withheld_suffix + "}"
                # Strings get single quotes in the model-produced string.
                # JSON requires double quotes.
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = compute_tool_delta(
                    self.streamed_args_for_tool[index], new_call, index, withheld_suffix
                )

                if delta is not None:
                    tool_deltas.append(delta)
                    self.streamed_args_for_tool[index] += delta["function"]["arguments"]

            # HACK: serving_chat.py inspects the internal state of tool parsers
            # when determining its final streaming delta, automatically
            # adding autocompleted JSON.
            # These two lines avoid that nonsense while ensuring finish_reason
            # is set to tool_calls when at least one tool is called.
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return {"tool_calls": tool_deltas}
            elif not added_text and self.current_tool_id > 0:
                # Return an empty DeltaMessage once the tool calls are all done
                # so that finish_reason gets set.
                return {"content": ""}
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None


ToolParserManager.register_module("pythonic", PythonicToolParser)
