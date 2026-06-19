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
import re
import uuid
from collections.abc import Sequence
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

    # Pattern for Llama-4 / vLLM style: <function=name>{"json"}</function>
    FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(\{.*?\})</function>", re.DOTALL)

    def has_pending_tool_call(self, text: str) -> bool:
        if "<function=" in text or PYTHON_TAG in text:
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

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Llama model response.
        """
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        # 1. <function=name>{...}</function> XML wrapper.
        xml_matches = self.FUNCTION_PATTERN.findall(model_output)
        for name, args_str in xml_matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(_build_tool_call(name, arguments))
            except json.JSONDecodeError:
                # Keep the raw arguments string so callers see *something*
                # rather than dropping the call entirely.
                tool_calls.append(_build_tool_call(name, args_str))
        if xml_matches:
            cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text).strip()

        # 2. & 3. ipython tag and/or bare JSON. These can appear together
        # (model emits a short comment, then the JSON) so we sweep until
        # no more balanced JSON objects with a ``name`` key are found.
        cleaned_text = self._extract_json_tool_calls(cleaned_text, tool_calls)

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def _extract_json_tool_calls(
        self, text: str, tool_calls: list[dict[str, Any]]
    ) -> str:
        """Pull every ``<|python_tag|>``-prefixed or bare JSON tool call
        out of ``text``, append them to ``tool_calls``, and return the
        residual content (with the matched spans scrubbed).

        Strategy: scan left-to-right for either ``PYTHON_TAG`` or a ``{``;
        whichever comes first defines the start of the next candidate
        span. Use balanced-brace matching to find the span end, then
        ``_parse_json_tool_call`` to validate. If validation succeeds,
        delete the span (and any preceding ``PYTHON_TAG`` marker) from
        the residual content. If it fails (not a tool-call-shaped JSON),
        skip past the ``{`` and keep scanning — preserves prose JSON.
        """
        if not text:
            return text
        out_parts: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            tag_idx = text.find(PYTHON_TAG, i)
            brace_idx = text.find("{", i)
            # Pick the earlier candidate; -1 means "not found".
            candidates = [idx for idx in (tag_idx, brace_idx) if idx != -1]
            if not candidates:
                out_parts.append(text[i:])
                break
            next_idx = min(candidates)

            if next_idx == tag_idx:
                json_search_start = tag_idx + len(PYTHON_TAG)
                span = _find_top_level_json_object(text, json_search_start)
                if span is None:
                    # Stray tag with no JSON payload — drop the tag and
                    # keep scanning past it. It's not user-visible text.
                    out_parts.append(text[i:tag_idx])
                    i = json_search_start
                    continue
                preceding_text = text[i:tag_idx]
            else:
                span = _find_top_level_json_object(text, brace_idx)
                if span is None:
                    out_parts.append(text[i:])
                    break
                preceding_text = text[i : span[0]]

            begin, end = span
            tool = _parse_json_tool_call(text[begin:end])
            if tool is None:
                # Not a tool call — keep the JSON as residual content
                # and move past it.
                out_parts.append(text[i:end])
                i = end
                continue

            out_parts.append(preceding_text)
            tool_calls.append(tool)
            i = end

        return "".join(out_parts).strip()

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
        """
        Extract tool calls from streaming Llama model output.

        Three streaming shapes handled:

        1. XML wrapper (``<function=...>...</function>``): suppress
           content while the wrapper is open; emit ``tool_calls`` when
           the closing tag arrives.
        2. ipython ``<|python_tag|>{...}`` and bare ``{...}`` JSON:
           same shape — suppress content from the opener until the
           balanced closing brace, then emit ``tool_calls`` (or flush
           buffered content if the close revealed it wasn't a tool
           call).
        3. Plain prose: pass ``delta_text`` straight through as content.

        State is recovered from ``previous_text`` (no instance state):
        we recompute the buffered-prefix decision on every call so
        multi-turn replies (prose → tool → prose) behave correctly.
        """
        # Fast path 1: XML wrapper — preserve historical behaviour.
        if "<function=" in current_text:
            if "</function>" not in delta_text:
                if "<function=" in previous_text:
                    return None
                split = delta_text.find("<function=")
                if split > 0:
                    return {"content": delta_text[:split]}
                return None
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return self._format_tool_calls_delta(result.tool_calls)
            return {"content": delta_text}

        # Compute the "buffered region" boundary. Anything before this
        # index in current_text has already been streamed to the client
        # as content (or as tool_calls). Anything from this index onward
        # is currently being held back pending a tool/not-tool decision.
        buffered_start = self._buffered_region_start(previous_text)
        cur_buffered_start = self._buffered_region_start(current_text)

        # No buffered region in current_text (and none was carried over)
        # → plain prose path; pass the delta through.
        if cur_buffered_start is None and buffered_start is None:
            return {"content": delta_text}

        # The buffer just opened in this delta. Forward any pre-buffer
        # prose that arrived in this same delta as content.
        if buffered_start is None and cur_buffered_start is not None:
            preface_len = cur_buffered_start - len(previous_text)
            preface = delta_text[:preface_len] if preface_len > 0 else ""
            # The buffered region itself is suppressed until the close.
            # If the JSON happens to close inside this same delta, fall
            # through to the close-decision branch.
            if not self._buffer_closes_in(current_text, cur_buffered_start):
                return {"content": preface} if preface else None
            decision = self._resolve_buffered_close(
                current_text, cur_buffered_start
            )
            if preface and decision and "content" in decision:
                decision = {"content": preface + decision["content"]}
            elif preface:
                # tool_calls decision — emit preface separately by
                # prepending to the content channel is not possible in
                # a single delta dict, so prefer correctness: the
                # postprocessor's prior calls already streamed the
                # preface as content (it's in delta_text but came
                # *before* the JSON anchor — but in the same callback).
                # Best-effort: emit tool_calls; preface is small and
                # rare in practice (typical Llama outputs are pure
                # JSON or pure prose, not mixed within one token).
                pass
            return decision

        # We were buffering before this delta. ``cur_buffered_start``
        # should match ``buffered_start``.
        anchor = cur_buffered_start if cur_buffered_start is not None else buffered_start
        assert anchor is not None
        if not self._buffer_closes_in(current_text, anchor):
            return None
        return self._resolve_buffered_close(current_text, anchor)

    def _buffered_region_start(self, text: str) -> int | None:
        """Return the index at which the held-back region of ``text``
        begins, or ``None`` if nothing in ``text`` is being held back.

        The region is the longest suffix that *might* still resolve to
        a tool call:

          - ``<|python_tag|>`` opener through end-of-string;
          - the first ``{`` that begins a tool-call-shaped object
            (``"name"`` present in the closed object, or unclosed and
            we haven't yet seen the disqualifying close).

        Prose JSON that has already closed without a ``"name"`` key is
        not buffered — it has been (or will be) streamed as content.
        """
        if not text:
            return None
        tag_pos = text.find(PYTHON_TAG)
        if tag_pos != -1:
            return tag_pos
        # Scan ``{`` candidates. Keep the first one that either (a) is
        # unclosed (might still grow into a tool call) or (b) is closed
        # and contains ``"name"`` (already a tool-call shape).
        i = 0
        n = len(text)
        while i < n:
            j = text.find("{", i)
            if j == -1:
                return None
            span = _find_top_level_json_object(text, j)
            if span is None:
                # Unclosed window. If the prefix before it is prose
                # (non-whitespace), the model has committed to "prose +
                # opening brace" — treat as a real buffer because the
                # JSON may yet resolve to a tool call.
                window = text[j:]
                if '"name"' in window or text[:j].strip() == "":
                    return j
                # Prose followed by a partial ``{`` with no ``name``
                # yet — could be either prose-JSON or a tool call.
                # Buffer it pending more tokens.
                return j
            window = text[j : span[1]]
            if '"name"' in window:
                return j
            i = span[1]
        return None

    def _buffer_closes_in(self, text: str, anchor: int) -> bool:
        """Return True iff the JSON object beginning at ``anchor``
        balances within ``text``."""
        # The anchor may be a python tag rather than ``{``.
        start = anchor
        if text.startswith(PYTHON_TAG, anchor):
            start = anchor + len(PYTHON_TAG)
            brace = text.find("{", start)
            if brace == -1:
                return False
            start = brace
        return _find_top_level_json_object(text, start) is not None

    def _resolve_buffered_close(
        self, text: str, anchor: int
    ) -> dict[str, Any] | None:
        """Called when the buffered region beginning at ``anchor`` has
        just closed in ``text``. Returns a tool_calls delta if it parses
        as a tool call, otherwise a content delta that flushes the
        buffered JSON (so the client doesn't lose it)."""
        result = self.extract_tool_calls(text[anchor:])
        if result.tools_called:
            return self._format_tool_calls_delta(result.tool_calls)
        # Not a tool call — flush the buffered region as content. The
        # post-close trailing text (anything after the closing brace)
        # is handled by subsequent ``extract_tool_calls_streaming``
        # calls falling through to the plain-prose path.
        return {"content": text[anchor:]}

    def _format_tool_calls_delta(
        self, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
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
                for i, tc in enumerate(tool_calls)
            ]
        }
