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

# Suffix that may still GROW into ``LFM_CALL_START``. Deliberately free of
# any ``.*``: it is evaluated once per ``[`` on every streamed delta, and a
# trailing ``.*`` made each test cost the length of the remaining text. The
# already-complete opener is matched by ``LFM_CALL_START`` itself.
_LFM_PARTIAL_START = re.compile(r"\[\s*(?:[A-Za-z_]\w*\s*\(?)?$", re.DOTALL)

# ``[Calling tool: name({...})]`` — the text-format envelope described in
# the module docstring. Case-sensitive on purpose: it mirrors both the
# producer (``api/utils.py`` writes this exact literal) and the recovery
# parsers that already understand it (``AutoToolParser.QWEN_BRACKET_PATTERN``,
# ``api/tool_calling._iter_calling_tool_calls``). A looser matcher here
# than in the recovery path is precisely the divergence that let this
# markup stream to the client in the first place.
TEXT_CALL_START = re.compile(r"\[\s*Calling\s+tool\s*:", re.DOTALL)

# Tool names in the text-format dialect follow the OpenAI function-name
# charset (``api/models._FUNCTION_NAME_PATTERN``) plus ``.``, matching what
# the cross-format recovery parser accepts — NOT Python identifier rules.
# ``my-tool`` and ``2fa`` are legal names a client may register, and the
# streaming path has to hold their markup back for the same reason as any
# other: otherwise the recovery path extracts the call at finalize while
# the raw span has already streamed to the user.
_TOOL_NAME = r"[A-Za-z0-9_.-]{1,64}"

# ``[name({`` — the envelope with its ``Calling tool:`` prefix dropped.
# The ``{`` is required so this can only claim the JSON-arguments dialect,
# never a pythonic call (``[index(0)]``) or ordinary prose.
JSON_CALL_START = re.compile(r"\[\s*" + _TOOL_NAME + r"\s*\(\s*\{", re.DOTALL)

# Suffix that may still grow into ``JSON_CALL_START``. Same no-``.*`` rule
# as the other partial patterns.
_JSON_CALL_PARTIAL_START = re.compile(
    r"\[\s*(?:" + _TOOL_NAME + r"\s*(?:\(\s*\{?)?)?$", re.DOTALL
)

# ``name(<json object>)`` — the envelope's payload.
_TEXT_CALL_BODY = re.compile(r"(" + _TOOL_NAME + r")\s*\((.*)\)", re.DOTALL)


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


# Suffix that may still GROW into ``TEXT_CALL_START``. Same no-``.*`` rule as
# ``_LFM_PARTIAL_START``: once the ``:`` has landed ``TEXT_CALL_START`` itself
# matches, so this only has to cover the opener still being typed.
_TEXT_CALL_PARTIAL_START = re.compile(
    r"\[\s*(?:"
    + _prefix_alternation("Calling")
    + r"(?:\s+(?:"
    + _prefix_alternation("tool")
    + r"\s*:?)?)?)?$",
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

    All three dialects count: the pythonic ``[name(``, the text-format
    ``[Calling tool:`` envelope, and its prefix-less ``[name({`` form.
    Whichever comes first wins so a turn mixing them is still walked left
    to right.
    """
    starts = [
        match.start()
        for match in (
            LFM_CALL_START.search(text, start),
            TEXT_CALL_START.search(text, start),
            JSON_CALL_START.search(text, start),
        )
        if match is not None
    ]
    return min(starts) if starts else -1


def _balanced_block_end(text: str, start_idx: int) -> int:
    """Index just past the balanced bracket block at ``start_idx``, or ``-1``.

    Nested brackets and quoted strings are accounted for so values like
    ``items=[1, 2]`` or ``query="]"`` do not prematurely close the block.
    Quote state starts fresh at ``start_idx`` — a stray quote earlier in the
    turn ("can't", an unfinished prose bracket) must never hide the markup
    that follows it.
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
                return i + 1

    return -1


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

    The walk NEVER descends into a block's payload, whether that block was
    accepted or rejected, and STOPS at the first opener with no matching
    ``]``. A payload can contain anything — including a literal
    ``[g({})]`` inside a JSON string argument — and dispatching a tool
    call out of another call's string data is irreversible: the
    count-based dedup below can never retract it once the real call
    arrives. So a rejected block resumes at its own end, and an
    unterminated opener (mid-stream, simply the call still being written)
    masks whatever follows. Both leave the text as content, the safe
    failure.
    """
    blocks: list[tuple[int, int, list[dict[str, Any]]]] = []
    search_from = 0

    while True:
        start = _find_lfm_call_start(text, search_from)
        if start == -1:
            return blocks
        end = _balanced_block_end(text, start)
        if end == -1:
            return blocks

        try:
            block_calls = _parse_call_block(text[start:end])
        except Exception as exc:
            logger.debug("Failed to parse LFM tool call: %s", exc)
            block_calls = []

        if block_calls:
            blocks.append((start, end, block_calls))
        # Sibling blocks after this span are still discovered; only its
        # own payload is off limits.
        search_from = end


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
    # ``calling_tool_text`` is not aspirational: because this parser reports
    # no native tool format, ``api/utils.py`` writes prior tool calls into
    # the prompt in exactly that shape and the model echoes it back.
    EXPECTED_WIRE_FORMATS = ("pythonic_bracket", "calling_tool_text")

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

        # No unterminated opener can precede an extracted block — the walk
        # in ``_iter_call_blocks`` stops at the first one — so the hold, if
        # any, starts after the blocks and this prefix is never short.
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
                if content:
                    # Every byte, whitespace included: ``_unconsumed_content``
                    # has already advanced the watermark past them, so
                    # dropping whitespace here would make the output depend
                    # on where the chunk boundary happened to fall.
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

        end = _balanced_block_end(text, start)
        if end == -1:
            return start

        search_from = end


def _may_grow_into_markup(text: str, idx: int) -> bool:
    """True when the suffix at ``idx`` (a ``[``) is or may become markup.

    Matched with ``pos`` rather than a slice: this runs once per ``[`` on
    every streamed delta, and slicing the tail would make each test cost
    the length of the remaining text.
    """
    return bool(
        LFM_CALL_START.match(text, idx)
        or TEXT_CALL_START.match(text, idx)
        or JSON_CALL_START.match(text, idx)
        or _LFM_PARTIAL_START.fullmatch(text, idx)
        or _TEXT_CALL_PARTIAL_START.fullmatch(text, idx)
        or _JSON_CALL_PARTIAL_START.fullmatch(text, idx)
    )


def _find_unclosed_markup_start(text: str) -> int:
    """Index of the first ``[`` that is markup and has no matching ``]``.

    Everything from that index on is held back: emitting it would leak
    half-formed markup, and the bytes are cheap to keep until the block
    either closes (tool call, or plain content released in one piece) or
    the stream ends (``flush_held_content``).

    Three properties matter, and the LFM streaming leak came from missing
    the first two:

    * The scan runs LEFT to RIGHT. Anchoring on the LAST ``[`` (the old
      behaviour) breaks as soon as the argument payload contains one of
      its own — the opener is released while the rest of the span is
      still held, so the client sees a headless fragment.
    * The hold covers the WHOLE opener, including the multi-word
      ``[Calling tool:`` envelope. Releasing an opener the moment it
      stops looking like ``[name(`` is what let ``[Calling tool`` reach
      the wire, where the global sanitizer stripped exactly that span
      and left ``: browse({...})]`` in the visible message.
    * Balance is judged per candidate, with quote state starting fresh at
      that ``[``. Carrying quote state across the whole turn would let an
      unfinished prose bracket (``See [note "unfinished``) hide the real
      opener that follows it inside the apparent string.

    Cost is linear in ``len(text)``: the markup test is bounded (no
    ``.*``), so the balance scan only runs for genuine candidates, and
    each balanced candidate advances the cursor past its own block.
    """
    search_from = 0
    while True:
        idx = text.find("[", search_from)
        if idx == -1:
            return -1
        if not _may_grow_into_markup(text, idx):
            # Unmistakably prose (``a[i is fine``). A ``[`` nested inside
            # it may still be a real opener, so keep looking.
            search_from = idx + 1
            continue
        end = _balanced_block_end(text, idx)
        if end == -1:
            return idx
        # Closed: either an already-extracted call or plain content.
        # Holding it would suppress the rest of the stream.
        search_from = end
