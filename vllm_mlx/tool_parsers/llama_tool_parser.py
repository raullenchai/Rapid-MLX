# SPDX-License-Identifier: Apache-2.0
"""
Llama tool call parser for rapid-mlx.

Handles three wire formats that real Llama checkpoints emit:

1. ``<function=name>{"arg": "value"}</function>`` — the Llama 4 /
   vLLM-style XML wrapper. Kept for compatibility with hosts and
   quantizations that follow that convention.
2. ``<|python_tag|>{"name": "X", "parameters": {...}}`` — Llama 3.1's
   ipython-mode prefix when the chat template surfaces the
   ``Environment: ipython`` system header.
3. Bare ``{"name": "X", "parameters": {...}}`` JSON with no wrapper —
   the **default** assistant tool-call shape baked into the official
   ``meta-llama/Llama-3.1-8B-Instruct`` (and 3.2 sibling) chat
   template. This is what ``Meta-Llama-3.1-8B-Instruct-4bit`` actually
   produces and what previously leaked into ``message.content`` because
   the parser only matched shape 1 (issue #700, "F-008").

For shapes 2 and 3 we accept either ``parameters`` (the Llama 3.1/3.2
canonical key) or ``arguments`` (OpenAI-style) — some quantizations
drift to the OpenAI key under fine-tuning.
"""

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

# Llama 3.1 ipython-mode prefix. The chat template emits this only when the
# system message advertises ``Environment: ipython``; we still accept it
# anywhere so quantizations that drift don't lose tool routing.
PYTHON_TAG = "<|python_tag|>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"


@dataclass
class _StreamState:
    """The "what should have been emitted by now" snapshot computed
    from a model-output prefix during streaming.

    Used by ``LlamaToolParser._emitted_state`` to derive a stateless
    diff between consecutive ``previous_text`` / ``current_text``
    streaming calls.
    """

    emitted_content_end: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _normalize_arguments(args: Any) -> str:
    """Render a parsed arguments value back to the OpenAI ``arguments``
    string contract.

    The OpenAI ``tool_calls[i].function.arguments`` field is a string of
    JSON, not a dict. Both callers (the dict-shape and the str-shape
    fallback paths) funnel through here so we keep the contract in one
    place.
    """
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def _build_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    return {
        "id": generate_tool_id(),
        "name": name.strip(),
        "arguments": _normalize_arguments(arguments),
    }


def _parse_json_tool_call(json_str: str) -> dict[str, Any] | None:
    """Parse one ``{"name": ..., "parameters"|"arguments": ...}`` blob.

    Returns ``None`` if the blob is not a Llama-style tool-call object
    (e.g. plain prose JSON the model happened to emit, or a malformed
    fragment). Caller treats ``None`` as "leave as content, not a tool
    call" — false positives here would silently swallow user-visible
    assistant text.

    Disambiguation rule: we require BOTH a non-empty ``name`` AND one of
    ``parameters``/``arguments`` to be present. The Llama 3.1/3.2 chat
    template always emits the ``parameters`` key (even ``{}`` for no-arg
    calls — ``"parameters": " + tool_call.arguments | tojson``), so this
    is a tight fit for the trained shape. Prose like
    ``Here is an example: {"name": "Alice"}`` no longer false-matches as
    a tool call named ``Alice``.
    """
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    # Llama 3.1/3.2 canonical key is "parameters"; accept "arguments" as
    # an OpenAI-style alias for fine-tunes that drifted to OpenAI's key.
    if "parameters" in obj:
        args = obj["parameters"]
    elif "arguments" in obj:
        args = obj["arguments"]
    else:
        # No args key — this is prose JSON that happens to have a
        # ``name`` field (user / object literal in assistant text).
        # Refuse to route as a tool call.
        return None
    return _build_tool_call(name, args)


def _find_top_level_json_object(text: str, start: int = 0) -> tuple[int, int] | None:
    """Find the first balanced ``{...}`` block starting at or after ``start``.

    Returns ``(begin, end_exclusive)`` of the JSON span, or ``None`` if
    no balanced object is found. Brace-counting is string-literal aware
    so a ``"}"`` inside a JSON string doesn't terminate early. We don't
    try to be a full JSON parser — ``json.loads`` validates the span
    once the brackets balance.
    """
    n = len(text)
    i = start
    while i < n and text[i] != "{":
        i += 1
    if i >= n:
        return None

    depth = 0
    in_str = False
    escape = False
    begin = i
    while i < n:
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return begin, i + 1
        i += 1
    return None


def _find_xml_function_call(
    text: str, start: int = 0
) -> tuple[int, int, str, str] | None:
    """Locate one ``<function=NAME>{...}</function>`` block from ``start``.

    Returns ``(begin, end_exclusive, name, args_json)`` or ``None``.

    Unlike a single regex (which is delimiter-unsafe — a ``}</function>``
    inside a JSON string terminates early and corrupts both the args and
    the trailing content — codex r3 MAJOR), we locate the opener, parse
    the name, then use the string-aware balanced-brace scanner to find
    the JSON span, and finally require ``</function>`` to follow with
    optional whitespace.
    """
    n = len(text)
    i = start
    while True:
        open_at = text.find(FUNCTION_OPEN, i)
        if open_at == -1:
            return None
        name_start = open_at + len(FUNCTION_OPEN)
        name_end = text.find(">", name_start)
        if name_end == -1:
            return None
        name = text[name_start:name_end]
        # ``>`` is the opener delimiter and ``<`` inside the name
        # indicates malformed wire text; reject either.
        if "<" in name or not name.strip():
            i = open_at + 1
            continue
        json_span = _find_top_level_json_object(text, name_end + 1)
        if json_span is None:
            return None
        args_begin, args_end = json_span
        # ``</function>`` must follow, possibly after whitespace only.
        tail = args_end
        while tail < n and text[tail] in " \t\r\n":
            tail += 1
        if not text.startswith(FUNCTION_CLOSE, tail):
            # Not a balanced wrapper at this position — skip past the
            # opener and keep scanning. (A genuine unbalanced earlier
            # opener falls through to plain content downstream.)
            i = open_at + 1
            continue
        return open_at, tail + len(FUNCTION_CLOSE), name, text[args_begin:args_end]


@ToolParserManager.register_module(["llama", "llama3", "llama4"])
class LlamaToolParser(ToolParser):
    """
    Tool call parser for Llama models (3.1, 3.2, 4 and quantized
    derivatives).
    """

    # Llama 3+ chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    # Shapes handled: XML wrapper (Llama 4 / vLLM style), the 3.1
    # ipython python-tag prefix, and bare JSON (Llama 3.1/3.2 default).
    EXPECTED_WIRE_FORMATS = ("function_bare", "llama_python_tag", "raw_json")

    # Streaming sentinels whose proper prefixes (``<``, ``<f``, ``<|p``,
    # ``<|python_ta``, ...) MUST be held back so per-char SSE doesn't
    # leak ``<|python`` to the client before the full tag arrives
    # (codex r3 MAJOR). Hermes/harmony use the same pattern.
    _STREAMING_SENTINELS = (FUNCTION_OPEN, PYTHON_TAG)

    def has_pending_tool_call(self, text: str) -> bool:
        if FUNCTION_OPEN in text or PYTHON_TAG in text:
            return True
        # Bare-JSON path: a streamed tool call begins with the very first
        # ``{`` token. We MUST treat the response as pending from that
        # first token onward — otherwise the partial ``{`` / ``{"na`` /
        # ``{"name"`` prefix leaks as assistant content before the JSON
        # closes (streaming false-negative — P1 / codex r1).
        #
        # We also recognise ``{"name"`` anywhere in the text so a prose
        # preface followed by the tool-call JSON (``Let me check.
        # {"name": ...}``) is correctly suppressed once the JSON opener
        # arrives — P2 / codex r2.
        if text.lstrip().startswith("{"):
            return True
        return '{"name"' in text

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Strip the longest tool-call-sentinel prefix off ``text``'s tail.

        Returns the portion of ``text`` that is safe to emit as content
        right now. The trimmed suffix is the longest non-empty proper
        prefix of any ``_STREAMING_SENTINELS`` entry that also forms a
        suffix of ``text`` — so a delta whose tail is ``"<|python"`` is
        held until either the full tag arrives (claimed by the tool-call
        branch) or a non-matching char arrives (released as content).
        """
        max_hold = 0
        for sentinel in cls._STREAMING_SENTINELS:
            for length in range(min(len(text), len(sentinel) - 1), 0, -1):
                if text.endswith(sentinel[:length]):
                    if length > max_hold:
                        max_hold = length
                    break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    def flush_held_content(self, full_text: str) -> str:
        """Release the prefix-held sentinel suffix at stream end.

        The streaming path holds back partial sentinel suffixes (``<``,
        ``<|p``, ``<func``...) until either the full opener arrives or a
        non-matching char arrives. When the stream ends with bytes still
        held AND no tool call fired, those bytes are ordinary content
        and must be released — otherwise a response ending in ``abc<``
        would surface as ``abc`` to the client (mirror of Hermes /
        harmony — codex r3 MAJOR).
        """
        return full_text[len(self._safe_content_prefix(full_text)) :]

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Llama model response.
        """
        tool_calls: list[dict[str, Any]] = []
        # Scan left-to-right: at each position pick the earliest of
        # ``<function=`` (XML wrapper), ``<|python_tag|>`` (ipython
        # prefix), or ``{`` (bare JSON). Whichever wins defines the next
        # candidate span; validation decides tool-call vs content.
        out_parts: list[str] = []
        n = len(model_output)
        i = 0
        while i < n:
            xml_idx = model_output.find(FUNCTION_OPEN, i)
            tag_idx = model_output.find(PYTHON_TAG, i)
            brace_idx = model_output.find("{", i)
            candidates = [idx for idx in (xml_idx, tag_idx, brace_idx) if idx != -1]
            if not candidates:
                out_parts.append(model_output[i:])
                break
            next_idx = min(candidates)

            if next_idx == xml_idx:
                xml = _find_xml_function_call(model_output, xml_idx)
                if xml is None:
                    # Unbalanced opener — leak it as residual content
                    # past the ``<``. (A trailing ``<function=`` with no
                    # close stays in the buffer until stream end.)
                    out_parts.append(model_output[i : xml_idx + 1])
                    i = xml_idx + 1
                    continue
                begin, end, name, args_str = xml
                out_parts.append(model_output[i:begin])
                try:
                    arguments = json.loads(args_str)
                    tool_calls.append(_build_tool_call(name, arguments))
                except json.JSONDecodeError:
                    # Keep the raw arguments string so callers see
                    # *something* rather than dropping the call entirely.
                    tool_calls.append(_build_tool_call(name, args_str))
                i = end
                continue

            if next_idx == tag_idx:
                json_search_start = tag_idx + len(PYTHON_TAG)
                span = _find_top_level_json_object(model_output, json_search_start)
                if span is None:
                    # Stray tag with no JSON payload — drop the tag and
                    # keep scanning past it. It's not user-visible text.
                    out_parts.append(model_output[i:tag_idx])
                    i = json_search_start
                    continue
                preceding_text = model_output[i:tag_idx]
            else:
                span = _find_top_level_json_object(model_output, brace_idx)
                if span is None:
                    out_parts.append(model_output[i:])
                    break
                preceding_text = model_output[i : span[0]]

            begin, end = span
            tool = _parse_json_tool_call(model_output[begin:end])
            if tool is None:
                # Not a tool call — keep the JSON as residual content
                # and move past it. The python_tag opener (if any) is
                # silently dropped — it was a stray control token.
                out_parts.append(model_output[i:end])
                i = end
                continue

            out_parts.append(preceding_text)
            tool_calls.append(tool)
            i = end

        cleaned_text = "".join(out_parts).strip()
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    #
    # The postprocessor invariant is:
    #   tool_accumulated_text  == previous_text + delta_text == current_text
    # i.e. ``previous_text`` is the *full* model output up to (but not
    # including) ``delta_text``. It is NOT a record of what the parser
    # has already sent the client.
    #
    # To remain stateless we re-derive "what should already have been
    # sent" from ``previous_text`` via ``_emitted_state`` and compute
    # the delta against ``_emitted_state(current_text)``. That keeps
    # the parser idempotent (calling it twice on the same prefix yields
    # the same emission) and survives postprocessor retries.

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
        """Diff-of-emitted-prefix streaming tool-call extractor.

        Handles all three Llama wire formats incrementally without
        instance state. Compares the "should-have-emitted" prefix of
        ``previous_text`` and ``current_text`` and returns the delta
        between them on the appropriate channel (content vs tool_calls).
        """
        prev_state = self._emitted_state(previous_text)
        cur_state = self._emitted_state(current_text)

        # New tool calls discovered in ``current_text`` that weren't
        # already in ``previous_text``. Emit them once and stop —
        # subsequent calls on the same prefix won't re-emit because the
        # ``previous_text`` state will have caught up. This is the
        # codex-r3 BLOCKING fix: previously every later delta re-ran
        # ``_buffered_region_start`` from position 0 and re-emitted the
        # already-flushed tool call (or duplicated it).
        start_index = len(prev_state.tool_calls)
        new_tool_calls = cur_state.tool_calls[start_index:]
        if new_tool_calls:
            return self._format_tool_calls_delta(
                new_tool_calls, start_index=start_index
            )

        # New content bytes — slice the safely-emittable region of
        # ``current_text`` against the same region in ``previous_text``.
        if cur_state.emitted_content_end > prev_state.emitted_content_end:
            new_content = current_text[
                prev_state.emitted_content_end : cur_state.emitted_content_end
            ]
            if new_content:
                return {"content": new_content}

        # Nothing to emit yet — still buffering an open anchor or a
        # held sentinel prefix.
        return None

    # ------------------------------------------------------------------
    # _emitted_state — the heart of the streaming state machine.
    # ------------------------------------------------------------------

    def _emitted_state(self, text: str) -> "_StreamState":
        """Walk ``text`` left-to-right and decide what the parser
        *should* have emitted up to this point.

        Returns a ``_StreamState`` describing:
          - ``emitted_content_end``: the index in ``text`` such that
            ``text[:emitted_content_end]`` has been streamed to the
            client as content. Bytes between this index and the end
            of ``text`` are either (a) inside an open buffered anchor
            (held pending the close decision) or (b) the sentinel-
            prefix held suffix.
          - ``tool_calls``: the complete list of tool calls that have
            been (or should have been) emitted in tool_calls channel
            events. Order matches their appearance in ``text``.

        Idempotent: calling ``_emitted_state`` twice on the same ``text``
        returns identical state (modulo per-call ``call_<uuid>`` IDs —
        we accept that drift; clients identify tool calls by ``index``).
        """
        n = len(text)
        cursor = 0  # bytes already classified (either emitted content or tool span)
        emitted_content_end = 0
        tool_calls: list[dict[str, Any]] = []
        while cursor < n:
            anchor = self._next_anchor_from(text, cursor)
            if anchor is None:
                # No more anchors. Everything from cursor onward is
                # safe content modulo the sentinel-prefix hold.
                safe = self._safe_content_prefix(text[cursor:])
                emitted_content_end = cursor + len(safe)
                cursor = n
                break

            # The bytes between ``cursor`` and ``anchor`` are plain
            # content (no anchor matched in there). They are safe to
            # emit because the anchor IS the next sentinel — bytes
            # before it can't be the start of a longer opener.
            if anchor > cursor:
                emitted_content_end = anchor

            closed_span = self._anchor_span(text, anchor)
            if closed_span is None:
                # Open anchor — buffer everything from ``anchor`` to
                # end of text. ``emitted_content_end`` stays at
                # ``anchor`` (any leading prose has already been
                # accounted for above).
                cursor = n
                break

            span_begin, span_end = closed_span
            tool = self._tool_from_span(text, anchor, closed_span)

            if tool is None:
                # Buffered region closed but isn't a tool call —
                # flush the whole span as content. The python_tag
                # opener / unbalanced wrapper bytes are included so
                # the client doesn't lose them.
                emitted_content_end = span_end
            else:
                # Tool call decision. The preface (cursor..anchor) was
                # already added to ``emitted_content_end`` above —
                # those bytes are emitted as content in a prior round.
                # The span itself (anchor..span_end) is emitted via the
                # tool_calls channel, NOT the content channel. Advance
                # the content "next-byte-to-emit" cursor past the span
                # so subsequent prose deltas (in this same scan or in
                # later ``extract_tool_calls_streaming`` calls) are
                # diffed against this boundary, not against a stale
                # pre-span boundary — codex r3 BLOCKING.
                emitted_content_end = span_end
                tool_calls.append(tool)

            cursor = span_end

        return _StreamState(
            emitted_content_end=emitted_content_end,
            tool_calls=tool_calls,
        )

    def _next_anchor_from(self, text: str, start: int) -> int | None:
        """Return the index of the next tool-call anchor at or after
        ``start`` in ``text``, or ``None`` if no anchor is found.

        Anchors are (precedence at the same position):
          - ``<function=...>`` XML opener
          - ``<|python_tag|>`` ipython prefix
          - ``{`` that begins a tool-call-shaped JSON object (closed
            with a ``"name"`` key) OR an unclosed ``{``

        Closed prose JSON without a ``name`` key is skipped past so
        successive prose JSON objects don't each create a buffered
        region.
        """
        n = len(text)
        i = start
        while i < n:
            xml_idx = text.find(FUNCTION_OPEN, i)
            tag_idx = text.find(PYTHON_TAG, i)
            brace_idx = text.find("{", i)
            candidates = [idx for idx in (xml_idx, tag_idx, brace_idx) if idx != -1]
            if not candidates:
                return None
            next_idx = min(candidates)
            if next_idx in (xml_idx, tag_idx):
                return next_idx
            # ``{`` — disambiguate prose JSON from tool-call JSON.
            span = _find_top_level_json_object(text, next_idx)
            if span is None:
                # Unclosed ``{`` — could still grow into a tool call.
                return next_idx
            window = text[next_idx : span[1]]
            if '"name"' in window:
                return next_idx
            # Closed JSON object with no ``name`` key — prose, not a
            # tool call. Skip past it.
            i = span[1]
        return None

    def _anchor_span(self, text: str, anchor: int) -> tuple[int, int] | None:
        """Return the closed span anchored at ``anchor`` in ``text``,
        or ``None`` if the span hasn't closed yet.

        ``anchor`` may point at ``<function=`` (span runs through
        ``</function>``), ``<|python_tag|>`` (span runs from the tag
        through the JSON close), or ``{`` (bare JSON span).
        """
        if text.startswith(FUNCTION_OPEN, anchor):
            xml = _find_xml_function_call(text, anchor)
            if xml is None:
                return None
            begin, end, _name, _args = xml
            return begin, end
        if text.startswith(PYTHON_TAG, anchor):
            json_start = anchor + len(PYTHON_TAG)
            brace = text.find("{", json_start)
            if brace == -1:
                return None
            span = _find_top_level_json_object(text, brace)
            if span is None:
                return None
            return anchor, span[1]
        return _find_top_level_json_object(text, anchor)

    def _tool_from_span(
        self,
        text: str,
        anchor: int,
        closed_span: tuple[int, int],
    ) -> dict[str, Any] | None:
        """Build a tool-call dict from a closed anchor span, or return
        ``None`` if the span turned out not to be a Llama tool call."""
        if text.startswith(FUNCTION_OPEN, anchor):
            xml = _find_xml_function_call(text, anchor)
            if xml is None:
                return None
            _b, _e, name, args_str = xml
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = args_str
            return _build_tool_call(name, arguments)
        # Bare JSON or python_tag prefix.
        if text.startswith(PYTHON_TAG, anchor):
            json_start = anchor + len(PYTHON_TAG)
        else:
            json_start = anchor
        json_span = _find_top_level_json_object(text, json_start)
        if json_span is None:
            return None
        jb, je = json_span
        return _parse_json_tool_call(text[jb:je])

    def _format_tool_calls_delta(
        self,
        tool_calls: list[dict[str, Any]],
        start_index: int = 0,
    ) -> dict[str, Any]:
        return {
            "tool_calls": [
                {
                    "index": start_index + i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
        }
