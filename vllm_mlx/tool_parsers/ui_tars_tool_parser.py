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


# Canonical UI-TARS Computer-Use action-API system prompt. Sourced
# verbatim (with the ``{language}`` / ``{instruction}`` placeholders
# stripped — we render only the static action-space contract so the
# user-supplied turn stays intact) from upstream
# https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py
# ``COMPUTER_USE_DOUBAO``. The model is post-trained on this exact
# wire format — without it the parser silently no-ops on raw output
# (dogfood C-05). Auto-prepended on the chat lane when the loaded
# alias's ``tool_call_parser == "ui_tars"`` AND the request didn't
# already include a UI-TARS preamble. Users CAN extend by supplying
# their own ``system`` message — the auto-injected sysprompt lands
# FIRST and the user's system content is appended (operator default:
# auto-injected sysprompt wins; user system is additive).
UI_TARS_COMPUTER_USE_SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, "
    "with screenshots. You need to perform the next action to complete the task.\n\n"
    "## Output Format\n"
    "```\n"
    "Thought: ...\n"
    "Action: ...\n"
    "```\n\n"
    "## Action Space\n"
    "click(point='<point>x1 y1</point>')\n"
    "left_double(point='<point>x1 y1</point>')\n"
    "right_single(point='<point>x1 y1</point>')\n"
    "drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')\n"
    "hotkey(key='ctrl c')\n"
    "type(content='xxx') # Use escape characters \\\\', \\\\\", and \\\\n in content "
    "part to ensure we can parse the content in normal python string format. "
    "If you want to submit your input, use \\\\n at the end of content.\n"
    "scroll(point='<point>x1 y1</point>', direction='down or up or right or left')\n"
    "wait() # Sleep for 5s and take a screenshot to check for any changes.\n"
    "finished(content='xxx') # Use escape characters \\\\', \\\\\", and \\\\n in content "
    "part to ensure we can parse the content in normal python string format.\n\n"
    "## Note\n"
    "- Use English in `Thought` part.\n"
    "- Write a small plan and finally summarize your next action (with its target "
    "element) in one sentence in `Thought` part."
)


# Sentinel substrings used to detect that an operator-supplied system
# message ALREADY carries (a variant of) the canonical UI-TARS
# sysprompt — in which case the auto-prepend is a no-op so we don't
# double-inject.
#
# Codex r2 BLOCKING (2026-06-21): the earlier draft accepted
# ``"## Output Format"`` or ``"You are a GUI agent"`` as evidence
# alone — both are too generic. A non-UI-TARS request whose system
# message happened to say "## Output Format: ...JSON..." would skip
# the inject and regress to raw prose / no tool calls.
#
# Tightened contract: detection requires BOTH a header-level marker
# (``"## Action Space"`` — present in every fork of the canonical
# UI-TARS prompt, and a phrase no general-purpose system message
# carries) OR a literal action-verb call signature
# (``click(point=``, ``click(start_box=``, ``drag(start_point=``)
# whose presence is mechanically the model-API kwarg shape.
# Header-alone hits + verb-alone hits both qualify; we don't AND
# them, because UI-TARS forks come in two flavors: full-prompt
# variants ship the ``## Action Space`` header; minimal variants
# ship only the verb-list table without the markdown headers.
_UI_TARS_SYSPROMPT_MARKERS: tuple[str, ...] = (
    # Header-level marker — unique to UI-TARS-class prompts; the
    # exact heading appears in every upstream fork of
    # ``codes/ui_tars/prompt.py``. Generic markdown formatting
    # instructions don't write this section heading.
    "## Action Space",
    # Action-verb call signatures: structural model-API kwarg
    # shapes that only a UI-TARS sysprompt would carry verbatim.
    # The space-and-paren forms make these robust against shuffled
    # whitespace.
    "click(point=",
    "click(start_box=",
    "drag(start_point=",
    "drag(start_box=",
    # Joint check — the canonical opener PLUS the next-line
    # ``Action`` keyword from the ``Output Format`` block. This
    # AND-shape catches forks that strip the ``## Action Space``
    # header but keep the ``Thought: ... / Action: ...`` skeleton.
    # (Implemented as substring of the joint phrase rather than a
    # boolean AND because string-contains is O(1) per marker and
    # keeps the detector single-pass.)
    "Thought: ...\nAction: ...",
)


def request_declares_computer_tool(tools: Any) -> bool:
    """Return True iff ``tools`` declares the Computer-Use ``computer`` tool.

    Dogfood r5-B C-09 root cause: sysprompt injection used to be
    **lane-coupled** — every ``/v1/chat/completions`` request hitting a
    UI-TARS-aliased model got the Computer-Use action-API system prompt
    prepended, even when the request had NO ``tools`` array. The model
    then dutifully emitted ``Action: click(point=...)`` for a "what is
    2+2?" prompt, the parser surfaced a phantom ``computer`` tool_call,
    and ``content`` came back ``null`` (F-R1-L). JSON mode broke the
    same way (F-R1-M: ``content="[]"``).

    The architectural fix is to make injection **tool-coupled**: the
    canonical UI-TARS sysprompt is only injected when the request
    actually declares a Computer-Use tool. Detector input is
    deliberately polymorphic (matches both ``ChatCompletionRequest``
    dict form and Anthropic / Responses-flat lists):

      * ``None`` / empty list → False (caller wants plain prose).
      * A list of pydantic ``ToolDefinition`` objects with
        ``function.name == "computer"`` → True (OpenAI chat shape).
      * A list of dicts ``{"function":{"name":"computer"}}`` → True
        (raw chat-completions tools array).
      * A list of dicts ``{"type":"computer_20251022", ...}`` → True
        (Responses-flat Computer-Use tool; mapped to ``computer`` in
        ``responses_adapter._convert_tools``).
      * A list of Anthropic-flat dicts ``{"name":"computer", ...}``
        (the ``/v1/messages`` tools shape) → True.
      * Anything else → False.

    Why ``"computer"`` specifically: the canonical UI-TARS function
    name (see ``COMPUTER_TOOL_NAME`` above) is the singleton name every
    UI-TARS lane emits. Custom user-supplied function tools named
    something else (``"search_screen"``, ``"summarize"``, …) do NOT
    trigger Computer-Use injection — those are vanilla function tools
    and should round-trip through the model untouched by the action-API
    contract.
    """
    if not tools:
        return False
    try:
        iterator = iter(tools)
    except TypeError:
        return False
    for t in iterator:
        # Pydantic ``ToolDefinition`` carries ``.function`` (dict on
        # OpenAI chat shape) plus ``.type``. Anthropic / Responses-flat
        # tools are plain dicts. Defensive: accept both.
        ttype = None
        name = None
        if isinstance(t, dict):
            ttype = t.get("type")
            # OpenAI-nested shape ``{"type":"function","function":{"name":...}}``
            fn = t.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
            # Anthropic / Responses-flat ``{"name":...}``
            if not name:
                name = t.get("name")
        else:
            ttype = getattr(t, "type", None)
            fn = getattr(t, "function", None)
            if isinstance(fn, dict):
                name = fn.get("name")
            elif fn is not None:
                name = getattr(fn, "name", None)
            if not name:
                name = getattr(t, "name", None)
        # Computer-Use input shape (Responses-flat ``computer_20251022``)
        # is canonical Computer-Use regardless of name. The adapter
        # rewrites ``name`` to ``"computer"`` downstream — match by
        # ``type`` here so the detector doesn't need to know about the
        # adapter's rewrite ordering.
        if ttype == "computer_20251022":
            return True
        if isinstance(name, str) and name == COMPUTER_TOOL_NAME:
            return True
    return False


def maybe_inject_ui_tars_system_prompt(
    messages: list,
    *,
    tool_call_parser: str | None,
    tool_choice: Any = None,
    tools: Any = None,
) -> list:
    """Auto-prepend the canonical UI-TARS sysprompt to ``messages`` when needed.

    Dogfood r5-B (C-09 / C-10 / C-11 / R-09) root cause: the prior
    contract was **lane-coupled** — every ``/v1/chat/completions``
    request hitting a UI-TARS-aliased model got the Computer-Use
    action-API sysprompt prepended, even when the request had no
    ``tools`` array at all. ``"what is 2+2?"`` came back as a phantom
    click; JSON mode degraded to ``"[]"``; ``/v1/responses`` was the
    mirror image (never injected → never emitted ``computer_call``).

    The architectural fix is **tool-coupled injection**: a single
    decision tree (this helper) reused by all 3 lanes — chat /
    messages / responses — that says "inject only when this UI-TARS
    request actually declares a Computer-Use tool." The signal lives
    in ``tools`` (per :func:`request_declares_computer_tool`), not in
    the route name. Three lanes, one rule::

        should_inject = (
            is_ui_tars_parser(parser)
            and tool_choice != "none"
            and request_declares_computer_tool(tools)
            and not has_ui_tars_system_prompt(messages)
        )

    Operator design choice: auto-injected sysprompt wins; a user-
    supplied ``system`` message is preserved as-is and APPENDED after
    the auto-injected one (additive, not overriding). This matches the
    PR #812 contract that "the alias just works" while leaving the
    operator a knob to extend / constrain the action space.

    Skip conditions (no injection):
    1. ``tool_call_parser != "ui_tars"`` — wrong model family.
    2. ``tool_choice == "none"`` — dogfood C-07 fix: the caller is
       requesting a text-only turn (e.g. for planning / debugging /
       asking the user). Prepending the action-API sysprompt would
       prime the model to emit ``Action: ...`` lines anyway, which
       the parser would then surface as a tool_call — violating
       OpenAI spec for ``tool_choice="none"``. Skipping the inject
       collapses the model into plain prose mode.
    3. ``tools`` does NOT declare a Computer-Use tool (r5-B C-09): the
       request isn't asking for actions — don't prime the model to
       emit them. This is the architectural fix that resolves F-R1-L
       (phantom clicks on "what is 2+2"), F-R1-M (JSON mode returning
       ``"[]"``), and via the same gate firing on ``/v1/responses``,
       F-R2-D / F-R2-I (cross-lane parity).
    4. The user already pasted (a variant of) the canonical
       sysprompt — ``has_ui_tars_system_prompt`` returns True. We
       respect the operator's preferred wording verbatim.

    Returns the (possibly prepended) ``messages`` list. The caller's
    reference is mutated only when a new message is inserted.
    """
    if tool_call_parser != "ui_tars":
        return messages
    # Codex r5 NIT #2: accept both raw string and request-dict shapes.
    # _is_tool_choice_none handles the request-dict path; mirror its
    # string-arm here so both call shapes converge on the same check.
    if tool_choice == "none" or _is_tool_choice_none(
        tool_choice if isinstance(tool_choice, dict) else None
    ):
        return messages
    # r5-B C-09: tool-coupled gate. NO Computer-Use tool declared →
    # caller wants plain prose / JSON / custom function call — skip
    # the action-API sysprompt entirely.
    if not request_declares_computer_tool(tools):
        return messages
    if has_ui_tars_system_prompt(messages):
        return messages

    # Codex r1 BLOCKING #1: build the inserted message in the SAME
    # shape as the existing list so we don't produce a mixed
    # ``dict``/Message-object collection downstream. Production code
    # path always normalizes ``messages`` to dicts before reaching
    # this helper (``extract_multimodal_content`` returns dicts on
    # both lanes), but the developer→system normalization in
    # ``routes/chat.py`` has a defensive object-handling branch —
    # we mirror that defense here. ``model_copy`` is preferred over
    # type construction so a pydantic subclass (custom Message
    # variant) round-trips its own type.
    sys_msg: Any
    if messages and not isinstance(messages[0], dict):
        first = messages[0]
        first_copy = getattr(first, "model_copy", None)
        if callable(first_copy):
            sys_msg = first_copy(
                update={"role": "system", "content": UI_TARS_COMPUTER_USE_SYSTEM_PROMPT}
            )
        else:
            sys_msg = {
                "role": "system",
                "content": UI_TARS_COMPUTER_USE_SYSTEM_PROMPT,
            }
    else:
        sys_msg = {"role": "system", "content": UI_TARS_COMPUTER_USE_SYSTEM_PROMPT}
    # Auto-injected sysprompt lands at index 0 so it primes the model
    # FIRST and any user-supplied system message extends (rather than
    # overrides) the action-API contract.
    return [sys_msg, *messages]


def has_ui_tars_system_prompt(messages: list) -> bool:
    """Return True if any ``system`` message already contains a UI-TARS preamble.

    Used by the chat / anthropic routes to decide whether to prepend the
    canonical sysprompt. The detector inspects ``str`` content directly
    and stringifies list-of-blocks content (Anthropic shape) so a
    multimodal system message with ``[{"type":"text","text":...}]``
    is still scanned.
    """
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "system":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, list):
            for block in content:
                text = (
                    block.get("text")
                    if isinstance(block, dict)
                    else getattr(block, "text", "")
                )
                if isinstance(text, str) and any(
                    marker in text for marker in _UI_TARS_SYSPROMPT_MARKERS
                ):
                    return True
        elif isinstance(content, str):
            if any(marker in content for marker in _UI_TARS_SYSPROMPT_MARKERS):
                return True
    return False


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

# Bare ``Action:`` token marker (no verb / paren required). Used by the
# streaming hold-back path to detect that an in-flight delta has STARTED
# the action prefix even though the verb/parens haven't arrived yet.
# Without this, ``has_pending_tool_call("Action: type")`` returned False
# (the strict ``_ACTION_LINE`` regex needs ``(``) and the postprocessor
# fast-path leaked the bare ``Action: <verb>`` bytes into ``delta.content``
# BEFORE the structured ``tool_calls`` flush (R6-C3, dogfood-087/R2-6).
_ACTION_PREFIX = re.compile(r"\bAction:", re.MULTILINE)

# Sentinel string the streaming parser hangs onto when a trailing
# delta tail is a STRICT prefix of ``Action:``. e.g. delta ends with
# ``"\nActi"`` → could become ``"\nAction:"`` on the next chunk, so we
# hold back those 4 bytes rather than leaking them as ``content``.
_ACTION_TOKEN = "Action:"

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


def _is_tool_choice_none(request: dict[str, Any] | None) -> bool:
    """Return True if ``request.tool_choice`` is OpenAI's ``"none"`` sentinel.

    The parser uses this to short-circuit tool emission for both the
    non-streaming and streaming paths so a model that still produced
    ``Action: ...`` text (e.g. operator-supplied UI-TARS sysprompt,
    quantization variance) doesn't bypass the OpenAI spec for
    ``tool_choice="none"``. Dogfood C-07.
    """
    if not isinstance(request, dict):
        return False
    return request.get("tool_choice") == "none"


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


# Coordinate kwargs the model may emit. Both upstream UI-TARS-1.0
# (``point`` / ``start_point`` / ``end_point``) and UI-TARS-1.5
# (``start_box`` / ``end_box``) are accepted on input; the parser
# normalizes everything to the SPEC keys on output (``point`` for
# single-point verbs, ``start_point`` / ``end_point`` for two-point
# verbs). This decouples downstream consumers from per-checkpoint
# kwarg drift — same OpenAI/Anthropic tool_call shape regardless of
# which UI-TARS variant produced the bytes.
_COORD_KEYS: tuple[str, ...] = (
    "point",
    "start_point",
    "end_point",
    "start_box",
    "end_box",
)

# Verbs that take a SINGLE point argument. All single-point variants
# (whether the model wrote ``point=`` or ``start_box=``) collapse to
# the spec ``point`` key. Verbs not in this set and not in the
# two-point set below preserve whatever key the model emitted (after
# the box→point rename below).
_SINGLE_POINT_VERBS: frozenset[str] = frozenset(
    {
        "click",
        "left_double",
        "right_single",
        "scroll",
        "hover",
        "tap",
        "long_press",
    }
)

# Verbs that take a TWO-point (start + end) argument. Both
# ``start_box`` / ``end_box`` and ``start_point`` / ``end_point``
# inputs collapse to the spec ``start_point`` / ``end_point`` keys.
_TWO_POINT_VERBS: frozenset[str] = frozenset({"drag", "select", "swipe"})

# Known chord-modifier prefixes. ``_normalize_action`` uses this set
# to decide whether to rewrite ``hotkey(key='X Y')`` → ``"X+Y"``.
# Only ``"<modifier> <key>"`` forms get the rewrite; single key names
# that contain a space (``"page down"``, ``"arrow up"``, ``"caps
# lock"``) are pass-through. Codex r5 NIT #2.
_HOTKEY_MODIFIERS: frozenset[str] = frozenset(
    {
        "ctrl",
        "control",
        "shift",
        "alt",
        "option",
        "opt",
        "cmd",
        "command",
        "meta",
        "win",
        "super",
        "fn",
    }
)


def _spec_key_for(verb: str, model_key: str) -> str:
    """Translate a model-emitted coord kwarg name to the SPEC key.

    The model speaks ``start_box`` / ``end_box`` (UI-TARS-1.5) OR
    ``point`` / ``start_point`` / ``end_point`` (UI-TARS-1.0). The
    spec — and PR #812's contract — is ``point`` / ``start_point`` /
    ``end_point``. We rename based on the verb so the OpenAI/Anthropic
    consumer sees a single canonical shape regardless of model
    variant.

    Renames applied (verb-aware):
    - Single-point verbs (``click`` / ``scroll`` / …):
      ``start_box`` → ``point``; ``point`` and ``start_point`` are
      kept as ``point``; ``end_box`` / ``end_point`` are dropped to
      ``point`` only if no single ``point`` was already present
      (defensive — these shouldn't appear on single-point verbs).
    - Two-point verbs (``drag`` / ``select`` / …):
      ``start_box`` → ``start_point``; ``end_box`` → ``end_point``;
      ``point`` → ``start_point`` (lone-point form for two-point
      verbs — model variant; rare).
    - Other / unknown verbs: preserve the model's choice (rare
      future-proofing — a UI-TARS-2.0 verb we haven't enumerated
      won't get its kwargs silently renamed).
    """
    if verb in _SINGLE_POINT_VERBS:
        # Codex r1 BLOCKING #2: single-point verbs ALWAYS emit the
        # spec ``point`` key. The earlier draft mapped
        # ``end_box``/``end_point`` to ``end_point`` for single-point
        # verbs — but the spec says single-point verbs MUST NOT carry
        # a two-point key. Coalesce every coord kwarg to ``point``;
        # the caller's first-write-wins guard then keeps the most
        # informative value (model rarely emits both, but if it does
        # the FIRST kwarg in dict-iteration order wins — typically
        # ``point`` > ``start_box`` > ``end_box``).
        return "point"
    if verb in _TWO_POINT_VERBS:
        if model_key == "start_box":
            return "start_point"
        if model_key == "end_box":
            return "end_point"
        # Codex r3 NIT: do NOT silently rename a lone ``point=`` on a
        # two-point verb (``drag``, ``select``) to ``start_point``.
        # That would turn a malformed ``drag(point='...')`` (missing
        # the end target) into a valid-looking partial drag, masking
        # an upstream validation opportunity. Preserve ``point`` so
        # the downstream consumer can detect the malformed shape and
        # surface a clear error.
        return model_key  # ``start_point`` / ``end_point`` / ``point`` pass through.
    # Unknown verb: emit whatever the model said.
    return model_key


def _normalize_action(verb: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map a parsed (verb, kwargs) pair to the canonical computer-tool args.

    Output shape: ``{"action": "<verb>", ...normalized_kwargs}``.

    Point-bearing kwargs (``point``, ``start_point``, ``end_point``,
    ``start_box``, ``end_box``) are normalized to integer lists when the
    coordinate body parses cleanly. Coord KEYS are renamed via
    ``_spec_key_for`` so single-point verbs emit ``point`` and two-point
    verbs emit ``start_point`` / ``end_point`` — independent of which
    UI-TARS checkpoint generated the wire bytes (PR #812 contract).

    The ``hotkey`` ``key`` argument is also normalized: UI-TARS emits
    space-separated chords (``"ctrl c"``) but the documented spec — and
    every downstream computer-use runtime — expects plus-separated
    chords (``"ctrl+c"``). The translation is lossless and reversible
    so a runtime that ALSO accepts the space form keeps working.

    Other kwargs pass through untouched. Unknown verbs are NOT
    rejected — emit them verbatim so a future UI-TARS-2.0 verb doesn't
    get dropped during the upgrade window.
    """
    out: dict[str, Any] = {"action": verb}
    # Track which spec keys have already been written so a model that
    # emits BOTH ``point=...`` and ``start_box=...`` on a single-point
    # verb doesn't double-write the same canonical key (first-wins).
    written: set[str] = set()
    for key, value in kwargs.items():
        if key in _COORD_KEYS:
            spec_key = _spec_key_for(verb, key)
            if spec_key in written:
                # Already populated by an earlier (typically more
                # canonical) kwarg. Don't clobber.
                continue
            if isinstance(value, str):
                parsed = _parse_point(value)
                if parsed is not None:
                    out[spec_key] = parsed
                    written.add(spec_key)
                    continue
            out[spec_key] = value
            written.add(spec_key)
            continue
        # ``hotkey.key``: rewrite space-separated chord to plus-form
        # so downstream computer-use runtimes (xdotool, pyautogui,
        # the Anthropic computer-tool harness) receive the documented
        # shape. The model trained on space form, so normalization
        # happens at the parser boundary, not on the model side.
        #
        # codex r5 NIT #2: only rewrite when the first whitespace
        # token is a known modifier — ``"ctrl c"``, ``"shift tab"``,
        # ``"alt f4"``, etc. — so single-key names that contain a
        # space (``"page down"``, ``"arrow up"``, ``"caps lock"``)
        # are passed through unchanged. The chord form ALWAYS leads
        # with a modifier; single key names never do.
        if verb == "hotkey" and key == "key" and isinstance(value, str):
            stripped = value.strip()
            if stripped and "+" not in stripped:
                tokens = stripped.split()
                if len(tokens) > 1 and tokens[0].lower() in _HOTKEY_MODIFIERS:
                    # Collapse runs of whitespace into a single ``+``
                    # (matches how xdotool / pyautogui name keys).
                    out[key] = "+".join(tokens)
                    continue
        out[key] = value
    return out


def _trailing_action_prefix_len(text: str) -> int:
    """Return how many trailing bytes of ``text`` could be a prefix of ``Action:``.

    Used by the streaming hold-back path. If ``text`` ends with a
    non-empty strict prefix of the literal ``"Action:"`` token (e.g.
    ``"...thing.\\nAc"`` → trailing ``"Ac"`` is the 2-byte prefix of
    ``"Action:"``), return that prefix length so the streaming path
    can keep those trailing bytes buffered until the next delta
    resolves the token. Returns 0 if no such overlap exists, OR if
    ``text`` contains the full ``"Action:"`` token already (in which
    case the full-action consumer / strict-prefix branch handles it).

    This is the symmetric counterpart to the reasoning parser's
    ``_compute_partial_action_hold`` — same purpose (don't ship a
    candidate-opener prefix until disambiguation), but operates on the
    tool-parser side of the hand-off so a leak that snuck past the
    reasoning parser's gate (e.g. via the postprocessor's fast-path
    short-circuit on ``<``/``[``-free deltas) is still suppressed.
    """
    if not text:
        return 0
    # If the full token is already present anywhere in the buffer,
    # the strict-prefix check is moot — the streaming consumer should
    # handle the completed-action accounting and emit the residual.
    if _ACTION_TOKEN in text:
        return 0
    # Longest non-empty suffix of ``text`` that matches a strict prefix
    # of ``_ACTION_TOKEN``. We scan from longest-to-shortest so that a
    # single delta containing ``"....Acti"`` returns 4, not 1.
    max_overlap = len(_ACTION_TOKEN) - 1  # 6 bytes (we already excluded full token)
    for k in range(min(max_overlap, len(text)), 0, -1):
        if _ACTION_TOKEN.startswith(text[-k:]):
            return k
    return 0


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


# Spec-key translation map for the Anthropic ``/v1/messages`` and OpenAI
# ``/v1/responses`` lanes. The UI-TARS parser emits its CANONICAL key set
# (``point`` / ``start_point`` / ``end_point``) so the chat-completions
# lane stays bytes-faithful to PR #812's contract; the adapters then
# remap to each spec's documented key.
#
# Anthropic Computer-Use spec
# (https://docs.anthropic.com/en/docs/agents-and-tools/computer-use):
# ``coordinate=[x, y]`` for single-point verbs; two-point verbs are
# expressed via separate ``start_coordinate`` / ``end_coordinate``
# fields (matches the spec's ``screenshot`` / ``cursor_position`` shape).
#
# OpenAI Responses Computer-Use spec
# (https://platform.openai.com/docs/api-reference/responses-streaming/output_item):
# ``computer_call.action.coordinate=[x, y]`` for single-point; two-point
# ``drag`` action uses ``path=[{x, y}, ...]`` — but a pragmatic translation
# emits ``start_coordinate`` / ``end_coordinate`` mirroring the Anthropic
# shape so a single key-mapper covers both surfaces. The Responses lane
# would otherwise have NO drag shape because the parser emits the
# Anthropic-compatible point-pair, not OpenAI's path list. Centralized
# here so the two adapters can't drift on key naming.
_UI_TARS_TO_SPEC_KEY_MAP: dict[str, str] = {
    "point": "coordinate",
    "start_point": "start_coordinate",
    "end_point": "end_coordinate",
}


def translate_to_spec_coordinate_keys(args: dict[str, Any]) -> dict[str, Any]:
    """Translate UI-TARS canonical coord keys to Anthropic/Responses spec keys.

    The parser output (``arguments`` JSON for the ``computer`` tool call)
    uses UI-TARS native keys ``point`` / ``start_point`` / ``end_point``
    because that's the contract PR #812 documented and the chat-completions
    OpenAI lane is already on. But the Anthropic ``/v1/messages`` lane
    (``tool_use.input``) and the OpenAI ``/v1/responses`` lane
    (``computer_call.action``) both follow specs that use ``coordinate``
    (and ``start_coordinate`` / ``end_coordinate`` for two-point verbs).

    Returns a shallow-copied dict with the keys remapped per
    ``_UI_TARS_TO_SPEC_KEY_MAP``; non-coord keys (``action``, ``content``,
    ``key``, ``direction``, …) pass through verbatim. Caller mutates only
    the copy.

    Idempotent: applying twice produces the same result (already-translated
    keys aren't in the map and are passed through).
    """
    if not args:
        return args
    out: dict[str, Any] = {}
    for k, v in args.items():
        out[_UI_TARS_TO_SPEC_KEY_MAP.get(k, k)] = v
    return out


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

        Dogfood C-07: when ``request.tool_choice == "none"`` the caller
        is opting OUT of tool emission. Even if the model still
        produced an ``Action:`` line (e.g. because the operator pasted
        the UI-TARS sysprompt and the route-level skip didn't fire),
        suppress the tool_call shape and surface the raw bytes as
        ``content`` so the OpenAI spec contract — "no tool_calls on
        tool_choice=none" — holds.
        """
        # Codex r3 BLOCKING #2: ``_is_tool_choice_none`` check MUST
        # run BEFORE the ``"Action:" not in model_output`` early
        # return — otherwise a case-variant ``action:`` (lowercase)
        # or a malformed marker the early-return branch already
        # accepts as "no tool calls" would skip the no-tools
        # contract and slip through the original semantics rather
        # than explicitly enforcing it. Promote the C-07 short-
        # circuit to the FIRST decision so the contract holds for
        # every shape of model output.
        if _is_tool_choice_none(request):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output or "",
            )

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

        Dogfood C-07: ``tool_choice=none`` short-circuits to
        passthrough — never emit a streaming ``tool_calls`` chunk so
        the OpenAI streaming spec for ``tool_choice=none`` ("no
        tool_calls in any delta") holds even if the model still emits
        ``Action:`` bytes.
        """
        if _is_tool_choice_none(request):
            return {"content": delta_text}
        if "Action:" not in current_text:
            # R6-C3 hold-back: the trailing bytes of ``current_text``
            # might be a STRICT prefix of the ``Action:`` token (e.g.
            # ``current_text`` ends with ``"\nAc"``). If we emit
            # ``delta_text`` verbatim those candidate-opener bytes leak
            # into ``delta.content`` BEFORE the structured tool_calls
            # flush — the dogfood R2-6 leak. Hold back any trailing
            # bytes of THIS delta that fall inside the partial-opener
            # window; the next delta that resolves the token (either
            # by completing ``Action:`` or by emitting a disambiguating
            # byte) releases them through the appropriate channel.
            held = _trailing_action_prefix_len(current_text)
            if held == 0:
                return {"content": delta_text}
            # Compute how much of THIS delta is in the held window vs.
            # the safely-emittable prefix.
            delta_start_in_current = len(previous_text)
            held_window_start = len(current_text) - held
            if held_window_start <= delta_start_in_current:
                # Every byte of this delta is in the partial-opener
                # window — emit nothing. The reasoning-parser side
                # already buffered any prior held prefix bytes; the
                # next disambiguating delta releases them all.
                return None
            split = held_window_start - delta_start_in_current
            return {"content": delta_text[:split]}

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

        # codex r2 BLOCKING #1: a single delta can contain a completed
        # action plus trailing/leading non-action text — e.g. delta
        # ``Action: wait() done`` where `` done`` is regular content the
        # model emitted after the action. Without explicit handling, the
        # original implementation only returned ``{"tool_calls": ...}``
        # and the trailing bytes were silently dropped from the response.
        #
        # Recover those bytes by computing the residual portion of THIS
        # delta that falls outside any newly completed action span. We
        # operate on offsets into ``current_text`` and clip to the slice
        # that this specific delta contributed.
        delta_start_in_current = len(previous_text)
        residual_pieces: list[str] = []
        cursor = delta_start_in_current
        for start, end, _verb, _kwargs in new_actions:
            # Bytes from the prior cursor up to the action's start that
            # fall inside this delta's window are residual content.
            if start > cursor:
                # Clip to [delta_start_in_current, len(current_text))
                piece_start = max(cursor, delta_start_in_current)
                piece_end = min(start, len(current_text))
                if piece_end > piece_start:
                    residual_pieces.append(current_text[piece_start:piece_end])
            cursor = end
        # Trailing bytes after the last newly completed action that fall
        # inside this delta's window.
        if cursor < len(current_text):
            piece_start = max(cursor, delta_start_in_current)
            piece_end = len(current_text)
            if piece_end > piece_start:
                residual_pieces.append(current_text[piece_start:piece_end])

        residual = "".join(residual_pieces)
        result: dict[str, Any] = {"tool_calls": tool_calls}
        if residual:
            result["content"] = residual
        return result

    def has_pending_tool_call(self, text: str) -> bool:
        """Return True if text contains (or could complete to) an unfinished action.

        Used by the streaming postprocessor's fast-path: when ``False``
        the postprocessor short-circuits the delta into ``content``;
        when ``True`` the full streaming path is engaged so the parser
        can buffer / suppress / emit structured ``tool_calls`` instead
        of leaking raw bytes.

        Pre-r6-B contract was "Action: verb( with no balanced )" only.
        The R6-C3 dogfood leak (R2-6) showed that an in-flight delta
        carrying the bare ``Action: <verb>`` token (no ``(`` yet) hit
        the False arm and leaked into ``delta.content`` BEFORE the
        structured ``tool_calls`` arrived. Tightened contract — three
        cases now return True:

        1. The strict ``_ACTION_LINE`` regex (``Action: verb(``) matched
           and has no balanced close — the original case.
        2. The bare ``Action:`` token appears in ``text`` (verb /
           parens not yet emitted) — a few more decode steps will
           complete the action signature, so hold the buffer.
        3. ``text`` ends with a non-empty STRICT prefix of ``Action:``
           (e.g. ``"...thing.\\nAc"``) — the next delta might land the
           rest of the token, so the trailing bytes must NOT be flushed
           through the fast-path; the streaming parser's hold-back
           logic clips them.

        Pure-False only when the buffer has no in-flight action signal
        and no trailing partial-opener candidate.
        """
        # Case 1: full ``Action: verb(`` line whose ``)`` hasn't arrived.
        # Note: when an action HAS completed (balanced close found) we
        # still return False here — the streaming consumer will surface
        # the completed tool_call on the next delta-resolving step; we
        # don't claim "pending" for an already-resolved action.
        line_starts: list[int] = []
        for m in _ACTION_LINE.finditer(text):
            line_starts.append(m.start())
            close = _find_balanced_close(text, m.end())
            if close == -1:
                return True
        # Case 2: bare ``Action:`` token whose verb / open-paren hasn't
        # arrived yet (i.e. an ``Action:`` occurrence that is NOT the
        # head of any matched ``_ACTION_LINE``). The classic dogfood
        # leak shape is ``"Action: type"`` where the ``(`` is in a
        # pending decode step — the postprocessor's fast-path used
        # to short-circuit the delta as plain ``content`` because
        # ``_ACTION_LINE.search`` requires the open paren. Holding
        # the buffer keeps the bare prefix bytes off the wire until
        # the verb + body resolve on subsequent deltas.
        for pm in _ACTION_PREFIX.finditer(text):
            if pm.start() not in line_starts:
                return True
        # Case 3: trailing strict-prefix of ``Action:`` at the buffer
        # tail (e.g. ``"...end.\\nAc"``). Holding back keeps the
        # candidate-opener bytes off the wire until the next delta
        # either completes the token or disambiguates them as content.
        return _trailing_action_prefix_len(text) > 0
