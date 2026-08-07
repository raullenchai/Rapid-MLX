# SPDX-License-Identifier: Apache-2.0
"""
LFM / Liquid tool call parser for vllm-mlx.

Handles Liquid AI's LFM model tool calling format:
- Bracketed pythonic format: [func_name(arg1=val1, arg2=val2)]
- Text-format envelope: [Calling tool: func_name({"arg1": "val1"})]

The second dialect is one rapid-mlx teaches the model itself. ``LfmToolParser``
reports ``SUPPORTS_NATIVE_TOOL_FORMAT = False`` (LFM chat templates drop
``message.tool_calls`` entirely, so preserving the native shape would erase
tool calls from the history), which makes ``api/utils.py`` serialise prior
assistant calls into the prompt as ``[Calling tool: name({args})]``. Models
imitate the transcript they are shown, so this form comes back on the wire
and has to be understood here — see the module note on
``_find_unclosed_markup_start``.
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

# ``[Calling tool: name({...})]`` — the text-format envelope described in
# the module docstring. Case-sensitive on purpose: it mirrors both the
# producer (``api/utils.py`` writes this exact literal) and the recovery
# parsers that already understand it (``AutoToolParser.QWEN_BRACKET_PATTERN``,
# ``api/tool_calling._iter_calling_tool_calls``). A looser matcher here
# than in the recovery path is precisely the divergence that let this
# markup stream to the client in the first place.
TEXT_CALL_START = re.compile(r"\[\s*Calling\s+tool\s*:", re.DOTALL)

# ``name(<json object>)`` — the envelope's payload.
_TEXT_CALL_BODY = re.compile(r"([A-Za-z_][\w.-]*)\s*\((.*)\)", re.DOTALL)


def _prefix_alternation(word: str) -> str:
    """Regex source matching any non-empty prefix of ``word``.

    ``"tool"`` becomes ``t(?:o(?:o(?:l)?)?)?`` — used to keep a partially
    streamed envelope opener held back until enough bytes arrive to tell
    markup from prose.
    """
    pattern = ""
    for char in reversed(word):
        pattern = re.escape(char) + (f"(?:{pattern})?" if pattern else "")
    return pattern


# Suffix of the accumulated text that may still GROW into ``TEXT_CALL_START``
# (or into its argument payload, once the ``:`` has landed).
_TEXT_CALL_PARTIAL_START = re.compile(
    r"\[\s*(?:"
    + _prefix_alternation("Calling")
    + r"(?:\s+(?:"
    + _prefix_alternation("tool")
    + r"(?:\s*(?::.*)?)?)?)?)?$",
    re.DOTALL,
)


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
    """Return the next LFM call start, or ``-1``.

    Both dialects count: the pythonic ``[name(`` and the text-format
    ``[Calling tool:`` envelope. Whichever comes first wins so a turn
    mixing the two is still walked left to right.
    """
    starts = [
        match.start()
        for match in (
            LFM_CALL_START.search(text, start),
            TEXT_CALL_START.search(text, start),
        )
        if match is not None
    ]
    return min(starts) if starts else -1


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


def _build_json_args_call(name: str, raw_arguments: str) -> list[dict[str, Any]]:
    """Build a one-element tool-call list from a JSON arguments payload.

    The payload MUST be parsed as JSON, never as a Python literal: the
    dialect writes ``true`` / ``false`` / ``null``, which ``ast`` reads as
    bare names and ``eval_node`` would turn into the strings ``"true"`` /
    ``"false"`` / ``"null"`` — a tool invoked with materially wrong
    arguments. Anything that is not a JSON object rejects the block.
    """
    raw = raw_arguments.strip()
    if not raw:
        arguments: Any = {}
    else:
        try:
            arguments = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(arguments, dict):
            return []

    return [
        {
            "id": generate_tool_id(),
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }
    ]


def _parse_text_format_block(block: str) -> list[dict[str, Any]]:
    """Parse a ``[Calling tool: name({...})]`` block into tool-call dicts.

    Returns an empty list for anything that is not exactly that shape
    (prose such as ``[Calling tool: see the docs]``), leaving the block
    in the content.
    """
    marker = TEXT_CALL_START.match(block)
    if marker is None:
        return []

    body = block[marker.end() :].strip()
    if not body.endswith("]"):
        return []
    call = _TEXT_CALL_BODY.fullmatch(body[:-1].strip())
    if call is None:
        return []
    return _build_json_args_call(call.group(1), call.group(2))


def _parse_bracket_json_block(block: str) -> list[dict[str, Any]]:
    """Parse ``[name({...})]`` — the envelope with its prefix dropped.

    Only a payload that already starts with ``{`` is claimed here; a
    pythonic call (``[f(x=1)]``, ``[index(0)]``) falls through to the AST
    path below, which keeps its stricter rules.
    """
    inner = block.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        return []
    call = _TEXT_CALL_BODY.fullmatch(inner[1:-1].strip())
    if call is None:
        return []
    if not call.group(2).strip().startswith("{"):
        return []
    return _build_json_args_call(call.group(1), call.group(2))


def _parse_call_block(block: str) -> list[dict[str, Any]]:
    """Parse one balanced ``[...]`` block into tool-call dicts.

    Three shapes are accepted: the ``[Calling tool: name({json})]``
    envelope, its prefix-less ``[name({json})]`` form, and the pythonic
    ``[name(arg=val), ...]`` list.

    Returns an empty list when the block is none of those. In the
    pythonic form a call carrying positional arguments rejects the WHOLE
    block: positional values cannot be mapped to named tool parameters,
    and emitting the call with empty/partial arguments would silently
    invoke the tool wrong. Rejected blocks stay in the content instead.
    """
    for parse in (_parse_text_format_block, _parse_bracket_json_block):
        calls = parse(block)
        if calls:
            return calls

    try:
        tree = ast.parse(block.strip())
    except SyntaxError:
        return []

    if not tree.body or not isinstance(tree.body[0], ast.Expr):
        return []
    node = tree.body[0].value
    if not isinstance(node, ast.List):
        return []

    calls = []
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


def _iter_call_blocks(text: str) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Return ``(start, end, calls)`` for every parsed call block, in order.

    Offsets are into ``text`` itself so callers can tell block bytes from
    content bytes — the streaming path needs that to emit prose that
    shares a delta with a completed call instead of dropping it.
    """
    blocks: list[tuple[int, int, list[dict[str, Any]]]] = []
    search_from = 0

    while True:
        start = _find_lfm_call_start(text, search_from)
        if start == -1:
            return blocks
        block, _ = _extract_balanced_bracket_block(text, start)
        if block is None:
            return blocks

        try:
            block_calls = _parse_call_block(block)
        except Exception as exc:
            logger.debug("Failed to parse LFM tool call: %s", exc)
            block_calls = []

        if block_calls:
            blocks.append((start, start + len(block), block_calls))
            search_from = start + len(block)
        else:
            search_from = start + 1


def parse_lfm_tool_calls(model_output: str) -> tuple[list[dict[str, Any]], str]:
    """Parse LFM tool calls and return ``(tool_calls, cleaned_text)``.

    Every call block in the output is considered — LFM may emit several
    separate blocks, not just one list. Blocks that don't parse as clean
    calls (prose, positional args) are left in the content.
    """
    blocks = _iter_call_blocks(model_output)
    if not blocks:
        return [], model_output

    tool_calls: list[dict[str, Any]] = []
    kept: list[str] = []
    cursor = 0
    for start, end, block_calls in blocks:
        tool_calls.extend(block_calls)
        kept.append(model_output[cursor:start])
        cursor = end
    kept.append(model_output[cursor:])
    return tool_calls, "".join(kept)


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
        # Count of tool calls already streamed to the client this turn.
        # LFM may emit several bracket blocks over the course of a stream
        # (``[f(x=1)] ... [g(y=2)]``); each ``]`` re-runs full-text
        # extraction, so we slice ``result.tool_calls[_emitted_tool_count:]``
        # to send only the newly-completed calls with continuing indices —
        # the same dedup pattern as Gemma4/Qwen. Re-running extraction on a
        # later ``]`` in trailing prose yields no NEW calls (slice is
        # empty), so the same call is never re-emitted with a fresh id
        # (which would corrupt OpenAI per-index ``arguments`` concat).
        # Parser instances are per-request (see StreamingPostProcessor), so
        # this counter never leaks across streams.
        self._emitted_tool_count = 0
        # Bytes of the accumulated text already accounted for — emitted as
        # content OR consumed as a tool-call block. Tracking one watermark
        # instead of diffing ``previous_text`` against ``current_text`` is
        # what lets the tool-call branch emit prose that shares a delta
        # with a completed block: after the call is built, everything from
        # the watermark up to the safe boundary that is not block bytes is
        # still content and must reach the client. The EOF flush is skipped
        # once a call fires, so anything not emitted here is lost.
        self._consumed_len = 0

    def reset(self) -> None:
        super().reset()
        self._emitted_tool_count = 0
        self._consumed_len = 0

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
        start = _find_unclosed_markup_start(text)
        return text if start == -1 else text[:start]

    def _emitted_boundary(self, previous_text: str) -> int:
        """Bytes of the accumulated text already accounted for.

        ``previous_text`` is authoritative — the postprocessor may seed it
        with a forced-``tool_choice`` assistant prefix the parser never
        saw as a delta (``seed_forced_assistant_prefix``), and those bytes
        must not be re-emitted as content. ``_consumed_len`` only raises
        the floor for what the tool-call branch consumed on top.
        """
        return max(len(self._safe_content_prefix(previous_text)), self._consumed_len)

    def _emit_safe_content(
        self, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        safe_current = self._safe_content_prefix(current_text)
        boundary = self._emitted_boundary(previous_text)
        if len(safe_current) <= boundary:
            return None
        self._consumed_len = len(safe_current)
        return {"content": safe_current[boundary:]}

    def _unconsumed_content(
        self,
        previous_text: str,
        current_text: str,
        blocks: list[tuple[int, int, Any]],
    ) -> str:
        """Content bytes not yet emitted and not part of a call block.

        Advances the watermark past everything it returns, so a later
        delta cannot re-emit the same bytes.
        """
        pieces: list[str] = []
        cursor = self._emitted_boundary(previous_text)
        for start, end, _ in blocks:
            if end <= cursor:
                continue
            if start > cursor:
                pieces.append(current_text[cursor:start])
            cursor = max(cursor, end)

        safe_end = len(self._safe_content_prefix(current_text))
        if safe_end > cursor:
            pieces.append(current_text[cursor:safe_end])
            cursor = safe_end

        self._consumed_len = max(self._consumed_len, cursor)
        return "".join(pieces)

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

        if _find_lfm_call_start(current_text) != -1 and "]" in delta_text:
            blocks = _iter_call_blocks(current_text)
            total_calls = sum(len(calls) for _, _, calls in blocks)
            if total_calls > self._emitted_tool_count:
                base = self._emitted_tool_count
                new_calls = [tc for _, _, calls in blocks for tc in calls][base:]
                self._emitted_tool_count = total_calls
                out: dict[str, Any] = {
                    "tool_calls": [
                        {
                            "index": base + i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(new_calls)
                    ]
                }
                # Prose that shares a delta with a completed call (batched
                # chunk / finalize / single-shot stream) was never emitted,
                # and once a tool fires the EOF ``flush_held_content`` path
                # is skipped — so it has to go out in this same return or
                # it is lost. The postprocessor renders the content event
                # before the tool-call event (the llama-parser precedent,
                # StreamingPostProcessor._detect_tool_calls).
                content = self._unconsumed_content(previous_text, current_text, blocks)
                if content.strip():
                    out["content"] = content
                return out

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


def _may_grow_into_markup(tail: str) -> bool:
    """True when ``tail`` (which begins at a ``[``) may still become markup."""
    return bool(
        _LFM_PARTIAL_START.fullmatch(tail) or _TEXT_CALL_PARTIAL_START.fullmatch(tail)
    )


def _unclosed_bracket_starts(text: str) -> list[int]:
    """Indices of every ``[`` still open at the end of ``text``, outermost first.

    One left-to-right pass, so this stays linear where a
    ``_extract_balanced_bracket_block`` call per ``[`` would be quadratic
    (``"[!" * n`` re-scans the whole suffix for every opener, and the
    streaming path runs this on every delta).

    Quote and escape state is tracked only INSIDE a block. At depth 0 an
    apostrophe is just prose — treating ``don't`` as an open string would
    swallow every ``[`` in the rest of the turn, including real markup.
    Inside a block the state matches ``_extract_balanced_bracket_block``
    exactly, which is what protects ``query="]"``.
    """
    stack: list[int] = []
    in_string = False
    string_char = ""
    escaped = False

    for i, char in enumerate(text):
        if not stack:
            if char == "[":
                stack.append(i)
            continue

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
            stack.append(i)
        elif char == "]":
            stack.pop()

    return stack


def _find_unclosed_markup_start(text: str) -> int:
    """Index of the outermost still-open ``[`` that may become tool markup.

    Everything from that index on is held back: emitting it would leak
    half-formed markup, and the bytes are cheap to keep until the block
    either closes (tool call, or plain content released in one piece) or
    the stream ends (``flush_held_content``).

    Two properties matter, and the LFM streaming leak came from missing
    both:

    * The scan runs OUTERMOST first. Anchoring on the LAST ``[`` (the old
      behaviour) breaks as soon as the argument payload contains one of
      its own — the opener is released while the rest of the span is
      still held, so the client sees a headless fragment.
    * The hold covers the WHOLE opener, including the multi-word
      ``[Calling tool:`` envelope. Releasing an opener the moment it
      stops looking like ``[name(`` is what let ``[Calling tool`` reach
      the wire, where the global sanitizer stripped exactly that span
      and left ``: browse({...})]`` in the visible message.

    An unbalanced ``[`` that is unmistakably prose (``a[i is fine``) is
    skipped: a ``[`` nested inside it may still be a real opener.
    """
    for idx in _unclosed_bracket_starts(text):
        if _may_grow_into_markup(text[idx:]):
            return idx
    return -1
