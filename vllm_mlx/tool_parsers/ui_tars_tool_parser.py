# SPDX-License-Identifier: Apache-2.0
"""UI-TARS Computer-Use action parser for rapid-mlx.

UI-TARS (ByteDance) is a Qwen2-VL / Qwen2.5-VL based GUI agent VLM that
takes a screenshot + instruction and emits one or more action calls in
the literal ``Action: <verb>(<args>)`` shape. The actions follow the
Anthropic Computer-Use idiom — ``click`` / ``drag`` / ``hotkey`` / ``type``
/ ``scroll`` / ``wait`` / ``finished`` / ``call_user`` plus mobile
variants ``long_press`` / ``open_app`` / ``press_home`` / ``press_back``.

Reference: https://github.com/bytedance/UI-TARS (``codes/ui_tars/prompt.py``
+ ``codes/ui_tars/action_parser.py``).

Wire format examples this parser is responsible for:

    Thought: Click the search button in the top-right.
    Action: click(point='<point>200 300</point>')

    Thought: Drag the slider from left to right.
    Action: drag(start_point='<point>100 500</point>', end_point='<point>800 500</point>')

    Action: type(content='hello world\\n')

    Action: hotkey(key='ctrl c')

This parser emits each ``Action: <verb>(<kwargs>)`` line as a single
OpenAI ``tool_call`` whose ``function.name`` is ``"computer"`` (mirroring
Anthropic's ``computer`` tool) and ``function.arguments`` is a JSON object
of the canonical shape ``{"action": "<verb>", ...kwargs}``. Coordinate
arguments are normalized from the model's ``'<point>x y</point>'`` string
to a 2-int list ``[x, y]`` — downstream consumers ``json.loads`` the
arguments string and get a structured point, not a UI-TARS-specific
template string.

The accompanying reasoning parser (``vllm_mlx.reasoning.ui_tars_parser``)
splits the leading ``Thought: ...`` (or ``Reflection: ... Action_Summary:
...``) preamble into ``reasoning_content`` so SDK consumers don't see the
chain-of-thought leak into ``content``.

The ``Action:`` lines are stripped out of the residual ``content`` field
so OpenAI/Anthropic SDK consumers don't double-render them as both
``tool_calls`` and prose.

Streaming: emit each completed action eagerly the moment its closing
``)`` arrives. Dedup against the count of actions already emitted on the
previous delta (mirrors the pattern in ``QwenToolParser`` / ``HermesToolParser``).
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


# Canonical OpenAI function name for every UI-TARS action. Mirrors the
# Anthropic ``computer`` / ``computer_20241022`` tool that Computer-Use
# clients are already prompting for, so a Claude SDK consumer can swap
# UI-TARS in without rewriting the tool-handler dispatch.
COMPUTER_TOOL_NAME = "computer"


# Verbs UI-TARS may emit. The set is a superset of the desktop
# (``COMPUTER_USE_DOUBAO``) and mobile (``MOBILE_USE_DOUBAO``) action
# spaces in ``codes/ui_tars/prompt.py``. Verbs not in this set are still
# parsed and surfaced verbatim — we intentionally don't gate on the list
# so a future UI-TARS revision that adds a verb won't silently drop calls
# during the upgrade window.
_KNOWN_VERBS: frozenset[str] = frozenset(
    {
        # Desktop / Computer-Use
        "click",
        "left_double",
        "right_single",
        "drag",
        "hotkey",
        "type",
        "scroll",
        "wait",
        "finished",
        "done",
        "call_user",
        # Mobile additions
        "long_press",
        "open_app",
        "press_home",
        "press_back",
    }
)


# Action line: ``Action: verb(kwargs)`` — verb identifier, parenthesized
# args body. Body may span newlines (e.g. ``type(content='line1\nline2')``)
# but must end on a balanced ``)``. We match minimally and use a manual
# brace-balanced consumer in ``_iter_actions`` for correctness on nested
# parens inside string args.
_ACTION_LINE = re.compile(r"\bAction:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

# Reasoning-channel preamble at the start of a UI-TARS response. When the
# tool parser is run on its own (no separate reasoning parser configured),
# we strip this preamble out of ``content`` so the same chain-of-thought
# doesn't surface twice (once in ``reasoning_content`` via the reasoning
# parser, once in ``content`` via the tool parser). The reasoning parser
# is configured to extract this same prefix; stripping here keeps the
# two surfaces aligned regardless of postprocessor invocation order.
_PREAMBLE_LEADING = re.compile(
    r"^\s*(?:Thought|Reflection|Action_Summary):.*?(?=\s*Action:|\Z)",
    re.DOTALL,
)

# Point/box body the model emits inside kwargs:
#   point='<point>x1 y1</point>'
#   start_point='<point>x1 y1</point>'
#   end_point='<point>x2 y2</point>'
#   start_box='<bbox>x1 y1 x2 y2</bbox>'  (UI-TARS-1.5 absolute-coord shape)
# Tolerant of single/double quotes, extra whitespace, and missing tag
# (some quantized checkpoints emit bare ``<x,y>`` without ``<point>`` —
# verified in 2026-06 community forks; preserve robustness).
_POINT_TAGGED = re.compile(
    r"<point>\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*</point>"
)
_POINT_BARE_ANGLE = re.compile(
    r"<\s*([-+]?\d+(?:\.\d+)?)\s*[,\s]\s*([-+]?\d+(?:\.\d+)?)\s*>"
)
_BBOX_TAGGED = re.compile(
    r"<bbox>\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*</bbox>"
)
# UI-TARS-1.5 sentinel-token format observed in live 4-bit checkpoint:
#   start_box='<|box_start|>(x,y)<|box_end|>'
#   start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>'  (4-tuple variant)
# Coords are absolute pixel offsets into the input image (see
# action_parser.py's qwen25vl branch in upstream UI-TARS repo).
_BOX_SENTINEL = re.compile(
    r"<\|box_start\|>\s*(\([^)]+\)(?:\s*,\s*\([^)]+\))?)\s*<\|box_end\|>"
)


def _generate_tool_id() -> str:
    """Mint an OpenAI-flavored tool-call id.

    Adapter layer (``vllm_mlx/api/anthropic_adapter.py``) rewrites the
    prefix to ``toolu_`` on the ``/v1/messages`` surface — coordinated
    with D-ANTHRO-SPEC-POLISH so all parsers stay on the OpenAI ``call_``
    convention and the per-surface prefix is owned by a single helper.
    """
    return f"call_{uuid.uuid4().hex[:8]}"


def _coerce_int(value: float) -> int | float:
    """Snap a UI-TARS coord to int if it's a whole number, else preserve float.

    UI-TARS-1.0 emits integer 0-1000 normalized coords; UI-TARS-1.5 emits
    absolute integer pixel coords. Both are integral. We keep the float
    type for the rare case that future variants emit subpixel decimals
    (e.g. for sub-grid drag interpolation) — round-tripping ``200.5`` →
    ``200`` would silently change semantics.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _parse_point(raw: str) -> list[float | int] | None:
    """Parse a point/box string to a coordinate list.

    Recognized shapes (in priority order):
      1. ``<point>x y</point>``     → ``[x, y]``
      2. ``<bbox>x1 y1 x2 y2</bbox>`` → ``[x1, y1, x2, y2]``
      3. ``<x, y>`` or ``<x y>``     → ``[x, y]``  (bare-angle fallback)
      4. ``x, y`` or ``[x, y]``      → ``[x, y]``  (Python-literal fallback)

    Returns ``None`` if no recognizable coordinate pattern is found so the
    caller can preserve the raw string in ``arguments`` (don't drop user
    intent on a malformed action).
    """
    if not isinstance(raw, str):
        return None

    s = raw.strip()
    m = _POINT_TAGGED.search(s)
    if m is not None:
        return [_coerce_int(float(m.group(1))), _coerce_int(float(m.group(2)))]

    m = _BBOX_TAGGED.search(s)
    if m is not None:
        return [_coerce_int(float(m.group(i))) for i in (1, 2, 3, 4)]

    # UI-TARS-1.5 sentinel — ``<|box_start|>(x,y)<|box_end|>`` (single point)
    # or ``<|box_start|>(x1,y1),(x2,y2)<|box_end|>`` (bbox). Split the inner
    # tuples cheaply via regex; coord parsing reuses the comma-split fallback
    # below by stripping parens.
    m = _BOX_SENTINEL.search(s)
    if m is not None:
        body = m.group(1)
        nums: list[int | float] = []
        for tup in re.findall(r"\(([^)]+)\)", body):
            for part in tup.split(","):
                try:
                    nums.append(_coerce_int(float(part.strip())))
                except ValueError:
                    continue
        if nums:
            return nums

    m = _POINT_BARE_ANGLE.search(s)
    if m is not None:
        return [_coerce_int(float(m.group(1))), _coerce_int(float(m.group(2)))]

    # Python-literal fallback: ``[200, 300]`` or ``(200, 300)`` or ``200,300``
    candidate = s.strip("[]()")
    parts = [p.strip() for p in candidate.split(",")]
    if 2 <= len(parts) <= 4:
        try:
            nums = [_coerce_int(float(p)) for p in parts]
        except ValueError:
            return None
        return nums

    return None


def _find_balanced_close(text: str, start: int) -> int:
    """Return the index of the ``)`` that closes the paren at ``text[start-1]``.

    Skips parens that appear inside Python string literals (single, double,
    or triple-quoted) so that ``type(content='hello (world)')`` is consumed
    as a single action. Returns ``-1`` if no balanced close is found before
    end-of-string — that signals a partial mid-stream action which the
    streaming path should NOT emit yet.

    Algorithm: scan forward, tracking quote state (none / single / double
    / triple-single / triple-double) and paren depth. Backslash escapes
    are honored inside quoted regions.
    """
    n = len(text)
    depth = 1
    i = start
    quote: str | None = None  # one of None, "'", '"', "'''", '"""'
    while i < n:
        ch = text[i]
        if quote is None:
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth -= 1
                if depth == 0:
                    return i
                i += 1
                continue
            if ch in ("'", '"'):
                # Probe for triple-quote
                if i + 2 < n and text[i + 1] == ch and text[i + 2] == ch:
                    quote = ch * 3
                    i += 3
                    continue
                quote = ch
                i += 1
                continue
            i += 1
            continue
        # Inside a quoted string.
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if len(quote) == 3:
            if i + 2 < n and text[i : i + 3] == quote:
                i += 3
                quote = None
                continue
            i += 1
            continue
        # Single-char quote.
        if ch == quote:
            quote = None
            i += 1
            continue
        i += 1
    return -1


def _parse_kwargs(body: str) -> dict[str, Any]:
    """Parse the kwargs body of an action call.

    UI-TARS emits Python kwargs syntax: ``key='value', key2='value2'``.
    We use ``ast.parse`` to handle escapes / quoted-paren correctly, with
    a regex fallback for malformed bodies (e.g. unclosed quotes from a
    truncated stream — keeps the parser graceful instead of crashing).

    Special-cases the upstream ``parse_action_to_structure_output`` rename
    where ``start_point`` / ``end_point`` are aliased to
    ``start_box`` / ``end_box`` and bare ``point`` becomes ``start_box``.
    We DON'T do that rename here — we preserve the verb-author's intent
    so a downstream consumer can dispatch on the original kwarg name.
    """
    body = body.strip()
    if not body:
        return {}
    try:
        # Wrap as a function call to use Python's grammar for kwargs.
        node = ast.parse(f"_({body})", mode="eval")
    except SyntaxError:
        return _parse_kwargs_lenient(body)

    call = node.body
    if not isinstance(call, ast.Call):
        return {}
    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs splat — unsupported in UI-TARS
            continue
        try:
            value: Any = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            try:
                value = ast.unparse(kw.value)
            except AttributeError:
                value = None
        kwargs[kw.arg] = value
    # Positional args (e.g. ``finished("done")`` instead of
    # ``finished(content="done")``) — UI-TARS doesn't emit these per the
    # upstream prompt, but quantized variants occasionally do. Bind them
    # by position to a best-effort name so they're not silently lost.
    for idx, pos in enumerate(call.args):
        try:
            kwargs.setdefault(f"arg{idx}", ast.literal_eval(pos))
        except (ValueError, SyntaxError):
            continue
    return kwargs


_KWARG_REGEX = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"|([^,]+))",
    re.DOTALL,
)


def _parse_kwargs_lenient(body: str) -> dict[str, Any]:
    """Regex fallback when ``ast.parse`` rejects a malformed body."""
    out: dict[str, Any] = {}
    for m in _KWARG_REGEX.finditer(body):
        key = m.group(1)
        sq, dq, bare = m.group(2), m.group(3), m.group(4)
        if sq is not None:
            out[key] = sq.encode().decode("unicode_escape", errors="replace")
        elif dq is not None:
            out[key] = dq.encode().decode("unicode_escape", errors="replace")
        else:
            val = (bare or "").strip()
            # Try Python literal first; fall back to raw string.
            try:
                out[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                out[key] = val
    return out


def _normalize_action(verb: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map a parsed (verb, kwargs) pair to the canonical computer-tool args.

    Output shape: ``{"action": "<verb>", ...normalized_kwargs}``.

    Point-bearing kwargs (``point``, ``start_point``, ``end_point``,
    ``start_box``, ``end_box``) are normalized to integer lists when the
    coordinate body parses cleanly. Other kwargs pass through untouched.
    Unknown verbs are NOT rejected — emit them verbatim so a future
    UI-TARS-2.0 verb doesn't get dropped during the upgrade window.
    """
    out: dict[str, Any] = {"action": verb}
    _COORD_KEYS = ("point", "start_point", "end_point", "start_box", "end_box")
    for key, value in kwargs.items():
        if key in _COORD_KEYS and isinstance(value, str):
            parsed = _parse_point(value)
            if parsed is not None:
                out[key] = parsed
                continue
        out[key] = value
    return out


def _iter_actions(text: str) -> list[tuple[int, int, str, dict[str, Any]]]:
    """Find every ``Action: verb(...)`` block in left-to-right order.

    Returns a list of ``(start, end, verb, kwargs)`` tuples. ``start`` /
    ``end`` are byte offsets into ``text`` for the full ``Action: ...``
    line so the caller can blank out the matched spans when computing the
    residual ``content``. Bodies are extracted via the balanced-paren
    scanner above so embedded ``)`` inside string args don't truncate.

    Action: sentinels that appear INSIDE an already-consumed action's
    body — e.g. ``Action: type(content='Action: wait()')`` — are skipped
    so the inner string-literal content is not re-parsed as a second
    tool call. The scanner advances past each matched ``)`` cursor.

    A partial action mid-stream (no balanced ``)`` yet) is NOT returned —
    the streaming path relies on this to avoid double-emitting once the
    rest of the body arrives.
    """
    actions: list[tuple[int, int, str, dict[str, Any]]] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        m = _ACTION_LINE.search(text, cursor)
        if m is None:
            break
        verb = m.group(1)
        body_start = m.end()
        close = _find_balanced_close(text, body_start)
        if close == -1:
            # Partial action — skip until the body finishes. The streaming
            # callsite re-scans on every delta, so we'll catch it later.
            break
        body = text[body_start:close]
        kwargs = _parse_kwargs(body)
        actions.append((m.start(), close + 1, verb, kwargs))
        cursor = close + 1
    return actions


@ToolParserManager.register_module(["ui_tars", "ui-tars", "uitars"])
class UiTarsToolParser(ToolParser):
    """Tool-call parser for UI-TARS (ByteDance) GUI-agent VLMs.

    Wire format: ``Action: <verb>(<kwargs>)`` lines (typically preceded
    by a ``Thought: ...`` chain-of-thought block which the matching
    reasoning parser handles separately).

    Each action becomes a single OpenAI ``tool_call`` with
    ``function.name = "computer"`` and ``function.arguments`` a JSON
    object of the form ``{"action": "<verb>", ...kwargs}`` — see
    ``_normalize_action`` for the kwarg-normalization contract.

    Used when ``--tool-call-parser ui_tars`` is set (or auto-wired by
    the ``ui-tars*`` regex in ``model_auto_config``).
    """

    EXPECTED_WIRE_FORMATS = ("ui_tars_action",)

    # Chat template for UI-TARS does NOT define ``role="tool"`` handling;
    # action results flow back through user messages with screenshots
    # rather than structured tool_result blocks. Leave the native-format
    # flag off — the engine will text-convert any historical tool calls.
    SUPPORTS_NATIVE_TOOL_FORMAT = False

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Parse a complete UI-TARS response.

        Empty / no-action input passes through with ``tools_called=False``
        and the original text as ``content`` so non-action prose
        (e.g. ``call_user()`` rejection messages) isn't lost.
        """
        if not model_output or "Action:" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        actions = _iter_actions(model_output)
        if not actions:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Build tool_calls + blank out matched spans for residual content.
        tool_calls: list[dict[str, Any]] = []
        residual_parts: list[str] = []
        cursor = 0
        for start, end, verb, kwargs in actions:
            residual_parts.append(model_output[cursor:start])
            cursor = end
            args = _normalize_action(verb, kwargs)
            tool_calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": COMPUTER_TOOL_NAME,
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
        residual_parts.append(model_output[cursor:])
        content = "".join(residual_parts)
        # Strip the standalone ``Thought:`` / ``Reflection:`` /
        # ``Action_Summary:`` preamble from ``content`` so the same chain
        # of thought doesn't surface twice (once in ``reasoning_content``
        # via the reasoning parser, once in ``content`` here). The
        # reasoning parser owns the channel; this strip is a defensive
        # mirror so a misconfigured server (tool parser ON, reasoning
        # parser OFF) still produces clean content.
        content = _PREAMBLE_LEADING.sub("", content).strip()
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content or None,
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
        """Streaming variant — emit each action exactly once when its body closes.

        Strategy: count balanced ``Action: ...)`` blocks in ``previous_text``
        vs ``current_text``. If the count increased, parse the newly
        completed actions and emit them with the right ``index`` offset.

        Edge cases:
        - Partial action mid-stream (no closing ``)`` yet) — ``_iter_actions``
          skips it, count stays unchanged, no emit. Once the ``)`` lands
          on a future delta, the action counts forward.
        - Mid-stream backslash-escaped close inside a string literal —
          ``_find_balanced_close`` honors quote state so we don't
          prematurely emit on ``type(content='oops)')``.
        - No ``Action:`` seen yet — passthrough delta as content so the
          ``Thought:`` preamble streams to the reasoning channel via the
          reasoning parser (which sees the same delta).
        """
        if "Action:" not in current_text:
            return {"content": delta_text}

        prev_actions = _iter_actions(previous_text) if previous_text else []
        cur_actions = _iter_actions(current_text)

        if len(cur_actions) <= len(prev_actions):
            return None

        new_actions = cur_actions[len(prev_actions) :]
        tool_calls = []
        for i, (_start, _end, verb, kwargs) in enumerate(new_actions):
            args = _normalize_action(verb, kwargs)
            tool_calls.append(
                {
                    "index": len(prev_actions) + i,
                    "id": _generate_tool_id(),
                    "type": "function",
                    "function": {
                        "name": COMPUTER_TOOL_NAME,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return {"tool_calls": tool_calls}

    def has_pending_tool_call(self, text: str) -> bool:
        """Return True if text contains an unfinished ``Action: verb(`` block.

        The ``finalize()`` postprocessor uses this to decide whether to
        hold delta bytes back vs. flush them as content at end-of-stream.
        For UI-TARS we conservatively say "pending" iff an ``Action:``
        token appears with no balanced ``)`` after it — that's the only
        case where a late-arriving byte could change parser output.
        """
        if "Action:" not in text:
            return False
        for m in _ACTION_LINE.finditer(text):
            close = _find_balanced_close(text, m.end())
            if close == -1:
                return True
        return False
