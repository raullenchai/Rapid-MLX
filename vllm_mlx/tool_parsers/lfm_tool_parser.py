# SPDX-License-Identifier: Apache-2.0
"""
LFM / Liquid tool call parser for vllm-mlx.

Handles Liquid AI's LFM model tool calling format:
- Bracketed pythonic format: [func_name(arg1=val1, arg2=val2)]
"""

import ast
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)

# ``[name(`` — the structural marker of a pythonic LFM call. Shared with
# AutoToolParser and the streaming postprocessor's plausible-markup
# pre-check so all three agree on what counts as LFM markup.
LFM_CALL_START = re.compile(r"\[\s*([A-Za-z_]\w*)\s*\(", re.DOTALL)
_LFM_PARTIAL_START = re.compile(r"\[\s*(?:[A-Za-z_]\w*\s*(?:\(.*)?)?$", re.DOTALL)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def eval_node(node: ast.AST) -> Any:
    """Safely evaluate AST nodes to Python values.

    Only ``ast.Constant`` and friends — never ``eval``. The deprecated
    ``ast.Num`` / ``ast.Str`` / ``ast.NameConstant`` aliases are NOT
    referenced here: they were removed in Python 3.14 and touching them
    raises ``AttributeError`` (constants have parsed as ``ast.Constant``
    since 3.8, so the aliases were dead code anyway).
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        # Bare names (``unit=celsius``) are treated as strings.
        return node.id
    if isinstance(node, ast.List):
        return [eval_node(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(eval_node(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {eval_node(k): eval_node(v) for k, v in zip(node.keys, node.values)}

    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)


def _find_lfm_call_start(text: str, start: int = 0) -> int:
    """Return the next ``[name(`` style LFM call start, or ``-1``."""
    match = LFM_CALL_START.search(text, start)
    return -1 if match is None else match.start()


def _extract_balanced_bracket_block(
    text: str, start_idx: int
) -> tuple[str | None, str]:
    """
    Return the balanced bracket block at ``start_idx`` and remaining text.

    Nested brackets and quoted strings are accounted for so values like
    ``items=[1, 2]`` or ``query="]"`` do not prematurely close the block.
    """
    depth = 0
    in_string = False
    string_char = None
    escaped = False

    for i in range(start_idx, len(text)):
        char = text[i]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if in_string:
            if char == string_char:
                in_string = False
            continue

        if char in ('"', "'"):
            in_string = True
            string_char = char
            continue

        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                bracket_block = text[start_idx : i + 1]
                remaining = text[:start_idx] + text[i + 1 :]
                return bracket_block, remaining

    return None, text


def _parse_call_block(block: str) -> list[dict[str, Any]]:
    """Parse one balanced ``[...]`` block into tool-call dicts.

    Returns an empty list when the block is not a clean LFM call list.
    A call carrying positional arguments rejects the WHOLE block:
    positional values cannot be mapped to named tool parameters, and
    emitting the call with empty/partial arguments would silently invoke
    the tool wrong. Rejected blocks stay in the content instead.
    """
    try:
        tree = ast.parse(block.strip())
    except SyntaxError:
        return []

    if not tree.body or not isinstance(tree.body[0], ast.Expr):
        return []
    node = tree.body[0].value
    if not isinstance(node, ast.List):
        return []

    calls: list[dict[str, Any]] = []
    for elt in node.elts:
        if not (isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name)):
            return []
        if elt.args:
            return []
        arguments = {}
        for kw in elt.keywords:
            if kw.arg is None:
                return []
            arguments[kw.arg] = eval_node(kw.value)
        calls.append(
            {
                "id": generate_tool_id(),
                "name": elt.func.id,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }
        )
    return calls


def parse_lfm_tool_calls(model_output: str) -> tuple[list[dict[str, Any]], str]:
    """Parse LFM pythonic tool calls and return ``(tool_calls, cleaned_text)``.

    Every ``[name(...)]`` block in the output is considered — LFM may emit
    several separate blocks, not just one list. Blocks that don't parse as
    clean call lists (prose, positional args) are left in the content.
    """
    tool_calls: list[dict[str, Any]] = []
    text = model_output
    search_from = 0

    while True:
        start = _find_lfm_call_start(text, search_from)
        if start == -1:
            break
        block, remaining = _extract_balanced_bracket_block(text, start)
        if block is None:
            break

        try:
            block_calls = _parse_call_block(block)
        except Exception as exc:
            logger.debug("Failed to parse LFM pythonic tool call: %s", exc)
            block_calls = []

        if block_calls:
            tool_calls.extend(block_calls)
            text = remaining
            search_from = start
        else:
            search_from = start + 1

    if not tool_calls:
        return [], model_output
    return tool_calls, text


@ToolParserManager.register_module(["lfm", "liquid"])
class LfmToolParser(ToolParser):
    """
    Tool call parser for Liquid's LFM models.

    Supports LFM bracket pythonic format:
    - [get_current_weather(location="Paris")]
    - [get_current_weather(location="New York", unit="celsius"), other_tool(arg=123)]
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = False
    EXPECTED_WIRE_FORMATS = ("pythonic_bracket",)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Once tool calls have been emitted for this stream, later ``]``
        # characters in trailing prose must not re-trigger extraction:
        # re-running ``extract_tool_calls`` re-emits the same call with a
        # fresh id at the same index, and OpenAI-delta clients concatenate
        # per-index ``arguments`` fragments into corrupt JSON. Parser
        # instances are per-request (see StreamingPostProcessor), so this
        # flag never leaks across streams.
        self._streaming_tools_emitted = False

    def reset(self) -> None:
        super().reset()
        self._streaming_tools_emitted = False

    def has_pending_tool_call(self, text: str) -> bool:
        return _find_unclosed_lfm_call_start(text) != -1

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete LFM model response."""
        tool_calls, cleaned_text = parse_lfm_tool_calls(model_output)

        if tool_calls:
            content = cleaned_text.strip()
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Return text safe to emit without leaking partial LFM markup."""
        start = _find_unclosed_lfm_call_start(text)
        if start != -1:
            return text[:start]

        # Hold a tail that could still become ``[func(`` once more tokens
        # arrive — but only while the bracket block is still unbalanced. A
        # closed block is either an already-extracted tool call or plain
        # content; holding it would suppress everything after a completed
        # call for the rest of the stream.
        last_bracket = text.rfind("[")
        if last_bracket != -1 and _LFM_PARTIAL_START.fullmatch(text[last_bracket:]):
            block, _ = _extract_balanced_bracket_block(text, last_bracket)
            if block is None:
                return text[:last_bracket]

        return text

    @classmethod
    def _emit_safe_content(
        cls, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        safe_current = cls._safe_content_prefix(current_text)
        safe_previous = cls._safe_content_prefix(previous_text)
        if len(safe_current) <= len(safe_previous):
            return None
        return {"content": safe_current[len(safe_previous) :]}

    def flush_held_content(self, full_text: str) -> str:
        """Release any held non-tool bracket prefix at stream end."""
        return full_text[len(self._safe_content_prefix(full_text)) :]

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
        """Extract tool calls from streaming LFM model output."""
        if "[" not in current_text:
            return {"content": delta_text}

        if (
            LFM_CALL_START.search(current_text) is not None
            and "]" in delta_text
            and not self._streaming_tools_emitted
        ):
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                self._streaming_tools_emitted = True
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        return self._emit_safe_content(previous_text, current_text)


def _find_unclosed_lfm_call_start(text: str) -> int:
    """Return the first plausible LFM call start without a matching ``]``."""
    search_from = 0
    while True:
        start = _find_lfm_call_start(text, search_from)
        if start == -1:
            return -1

        bracket_block, _ = _extract_balanced_bracket_block(text, start)
        if bracket_block is None:
            return start

        search_from = start + len(bracket_block)
