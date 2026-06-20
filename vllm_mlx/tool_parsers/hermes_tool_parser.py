# SPDX-License-Identifier: Apache-2.0
"""
Hermes/Nous tool call parser for rapid-mlx.

Handles Hermes-style tool calling format used by NousResearch models.
"""

import ast
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


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _parse_function_body(body: str) -> dict[str, Any]:
    """Parse the body of any ``<function=name>...</function>`` block.

    Called both for wrapped (``<tool_call><function=...>...``) and
    bare (no ``<tool_call>`` wrapper) Nemotron-shape blocks. Two wire
    formats coexist for the inner body (issue #448 BUG-2):

    * **Nemotron XML** — body holds ``<parameter=p>v</parameter>`` tags.
      Iterate via ``PARAM_PATTERN`` and decode each value through
      ``_parse_param_value``.
    * **Qwen3-Coder JSON** — body holds a JSON object
      ``{"key": "value", ...}`` directly. Try ``json.loads`` first;
      fall through to the XML path on failure.

    Discriminate cheaply on the first non-whitespace char: ``{`` ⇒
    likely JSON. Anything else ⇒ XML. We still attempt the other
    path as a fallback so malformed bodies degrade gracefully to an
    empty dict rather than crashing.
    """
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    arguments: dict[str, Any] = {}
    for p_name, p_value in HermesToolParser.PARAM_PATTERN.findall(body):
        arguments[p_name.strip()] = _parse_param_value(p_value.strip())
    return arguments


def _parse_param_value(val: str) -> Any:
    """Parse a tool call parameter value, handling both JSON and Python literals.

    Tries json.loads first. If that fails, falls back to ast.literal_eval
    for Python literal syntax (single quotes, True/False, None). Converts
    sets to lists and rejects types that are not JSON-serializable (complex,
    bytes) to avoid crashes during json.dumps later.
    """
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        python_val = ast.literal_eval(val)
        if isinstance(python_val, set):
            python_val = sorted(python_val, key=str)
        if isinstance(python_val, (complex, bytes)):
            return val
        json.dumps(python_val)
        return python_val
    except (ValueError, SyntaxError, TypeError):
        return val


@ToolParserManager.register_module(["hermes", "nous", "qwen3_coder"])
class HermesToolParser(ToolParser):
    """
    Tool call parser for Hermes/Nous models.

    Supports Hermes tool call format:
    - <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Sometimes with additional reasoning in <tool_call_reasoning>
    - Fallback: raw JSON {"name": "func", "arguments": {...}} (for models that omit tags)

    Used when --enable-auto-tool-choice --tool-call-parser hermes are set.
    """

    # Qwen3 / Hermes chat templates handle role="tool" and tool_calls natively.
    # Without this, tool history is converted to "[Calling tool: ...]" text,
    # which causes the model to mimic that text format instead of producing
    # proper <tool_call> XML after a few rounds of tool use.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Hermes is the most flexible parser — handles JSON body inside
    # <tool_call>, the Nemotron-style XML body fallback (covers vanilla
    # Qwen3.6), bare <function=...> blocks, the VibeThinker
    # ``<function><name>...</name><arguments>...</arguments></function>``
    # XML-named shape (F-042 redo), raw JSON tool calls, and the
    # [Calling tool:] text-fallback for low-quant degradation.
    EXPECTED_WIRE_FORMATS = (
        "tool_call_json",
        "tool_call_xml_body",
        "function_bare",
        "function_xml_named",
        "raw_json",
        "calling_tool_text",
    )

    # Standard format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    # Lenient format: <tool_call or <tool_call> followed by JSON (handles malformed tags)
    TOOL_CALL_LENIENT_PATTERN = re.compile(
        r'<tool_call[^{]*(\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\})', re.DOTALL
    )
    # Nemotron XML: <tool_call><function=name><parameter=p>v</parameter></function></tool_call>
    NEMOTRON_PATTERN = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
    )
    PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
    REASONING_PATTERN = re.compile(
        r"<tool_call_reasoning>(.*?)</tool_call_reasoning>", re.DOTALL
    )
    # Fallback pattern for raw JSON tool calls (without tags)
    RAW_JSON_TOOL_PATTERN = re.compile(
        r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}', re.DOTALL
    )
    # Bare Nemotron XML: <function=name>...</function> without <tool_call> wrapper
    BARE_FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
    # VibeThinker XML-named shape (F-042 redo): the model emits the call as
    # ``<function><name>NAME</name><arguments>JSON</arguments></function>``
    # under ``tool_choice="auto"`` despite the chat template specifying
    # ``<tool_call>{"name":...,"arguments":...}</tool_call>``. Whitespace-
    # tolerant (newlines between tags, indented inner JSON). The arguments
    # body is parsed via the same ``_parse_function_body`` helper as the
    # ``<function=...>`` Nemotron path — accepts JSON object or XML
    # parameter shape.
    FUNCTION_XML_NAMED_PATTERN = re.compile(
        r"<function>\s*<name>\s*(.*?)\s*</name>\s*<arguments>\s*(.*?)\s*</arguments>\s*</function>",
        re.DOTALL,
    )

    @classmethod
    def _scan_tool_call_shapes(
        cls, text: str
    ) -> tuple[list[tuple[int, int, str, str]], str]:
        """Unified left-to-right scan over the three wire shapes.

        Returns ``(matches, residual_text)`` where ``matches`` is a list of
        ``(start, end, name, arguments_json)`` tuples in **wire order** and
        ``residual_text`` is the input with each matched span replaced by
        whitespace (preserving offsets for any downstream content emit).

        The three shapes considered at each scan position are:

        1. ``<tool_call>{json}</tool_call>`` (Hermes JSON body)
        2. ``<tool_call><function=NAME>...</function></tool_call>`` (Nemotron wrap)
        3. ``<function=NAME>...</function>`` (bare Nemotron)
        4. ``<function><name>NAME</name><arguments>JSON</arguments></function>``
           (VibeThinker named-XML — F-042)

        Single left-to-right scan: at each position, pick whichever of the
        four shape openers appears EARLIEST and consume its full block,
        then continue from the end of that block. This is the structural
        fix for F-042 r4: the prior cascade ran shapes additively, which
        broke wire order whenever both shape #1 and shape #4 coexisted in
        one response (e.g. ``<function>...</function><tool_call>...`` came
        out as ``[#1, #4]`` instead of ``[#4, #1]``).
        """
        matches: list[tuple[int, int, str, str]] = []
        cursor = 0
        n = len(text)
        residual_parts: list[str] = []

        # Pre-compile lightweight openers for cheap earliest-match probing.
        # We use the existing class patterns for the actual parse (they
        # already handle the body grammar) — these locators just tell us
        # which shape's full pattern to anchor on next.
        while cursor < n:
            # Find the next occurrence of each opener after ``cursor``.
            candidates: list[tuple[int, str]] = []
            tc_pos = text.find("<tool_call>", cursor)
            if tc_pos != -1:
                candidates.append((tc_pos, "tool_call"))
            # ``<function=`` opens both the Nemotron-wrapped (#2) and bare
            # (#3) shapes — we'll discriminate after matching.
            fb_pos = text.find("<function=", cursor)
            if fb_pos != -1:
                candidates.append((fb_pos, "function_eq"))
            # ``<function>`` (no ``=``) opens the named-XML shape (#4).
            # ``str.find`` would match ``<function=`` too, so guard with a
            # regex anchored on the closing ``>`` directly after.
            fn_match = re.compile(r"<function>").search(text, cursor)
            if fn_match is not None:
                candidates.append((fn_match.start(), "function_open"))

            if not candidates:
                residual_parts.append(text[cursor:])
                break

            earliest_pos, shape = min(candidates, key=lambda x: x[0])
            # Emit prefix between cursor and earliest_pos as residual.
            residual_parts.append(text[cursor:earliest_pos])

            consumed = cls._try_consume_at(text, earliest_pos, shape)
            if consumed is None:
                # Opener at ``earliest_pos`` didn't form a complete block.
                # For a ``<tool_call>`` opener that didn't close yet, skip
                # past its matching ``</tool_call>`` if present, otherwise
                # skip to end-of-text — so bare-function or named-XML
                # shapes nested INSIDE an unclosed ``<tool_call>`` wrapper
                # aren't transiently double-counted during streaming (their
                # ``</function>`` arrives before the wrapping ``</tool_call>``).
                # Any other shape's failed open: advance one byte so we
                # don't infinite-loop and let the bytes fall through as
                # content.
                if shape == "tool_call":
                    closing = text.find("</tool_call>", earliest_pos)
                    if closing == -1:
                        residual_parts.append(text[earliest_pos:])
                        break
                    residual_parts.append(
                        text[earliest_pos : closing + len("</tool_call>")]
                    )
                    cursor = closing + len("</tool_call>")
                    continue
                residual_parts.append(text[earliest_pos])
                cursor = earliest_pos + 1
                continue

            end_pos, name, arguments_json = consumed
            matches.append((earliest_pos, end_pos, name, arguments_json))
            # Replace the matched span with same-length whitespace in
            # residual so any downstream positional logic stays aligned.
            residual_parts.append(" " * (end_pos - earliest_pos))
            cursor = end_pos

        residual_text = "".join(residual_parts)
        return matches, residual_text

    @classmethod
    def _try_consume_at(
        cls, text: str, pos: int, shape: str
    ) -> tuple[int, str, str] | None:
        """Anchor the given shape's full pattern at ``pos`` and parse.

        Returns ``(end_offset, name, arguments_json)`` on a clean parse,
        ``None`` if the opener at ``pos`` doesn't form a valid block (we
        let the scanner skip past it as content).
        """
        if shape == "tool_call":
            # Shape #1: <tool_call>{json}</tool_call>
            m = cls.TOOL_CALL_PATTERN.match(text, pos)
            if m is not None:
                body = m.group(1)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    name = data.get("name", "")
                    if name:
                        arguments = data.get("arguments", {})
                        return (
                            m.end(),
                            name,
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments),
                        )
            # Shape #2: <tool_call><function=NAME>...</function></tool_call>
            m = cls.NEMOTRON_PATTERN.match(text, pos)
            if m is not None:
                name = m.group(1).strip()
                args = _parse_function_body(m.group(2))
                return (m.end(), name, json.dumps(args, ensure_ascii=False))
            return None
        if shape == "function_eq":
            # Shape #3: <function=NAME>...</function>
            m = cls.BARE_FUNCTION_PATTERN.match(text, pos)
            if m is not None:
                name = m.group(1).strip()
                args = _parse_function_body(m.group(2))
                return (m.end(), name, json.dumps(args, ensure_ascii=False))
            return None
        if shape == "function_open":
            # Shape #4: <function><name>NAME</name><arguments>JSON</arguments></function>
            m = cls.FUNCTION_XML_NAMED_PATTERN.match(text, pos)
            if m is not None:
                name = m.group(1).strip()
                args_body = m.group(2)
                args = _parse_function_body(args_body)
                if not args and args_body.strip().startswith("{"):
                    # Body looked like JSON but failed both paths — keep
                    # raw to avoid silently dropping argument content.
                    return (m.end(), name, args_body.strip())
                return (m.end(), name, json.dumps(args, ensure_ascii=False))
            return None
        return None

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Hermes model response.
        """
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(cleaned_text)

        # Remove reasoning tags first (keep for content)
        reasoning_matches = self.REASONING_PATTERN.findall(cleaned_text)
        cleaned_text = self.REASONING_PATTERN.sub("", cleaned_text)

        # Unified left-to-right scan over the three structured wire
        # shapes — emits tool calls in WIRE ORDER (F-042 redo). The
        # residual text is what's left when each matched span is blanked
        # out; we collapse whitespace below to recover the content
        # surface.
        scan_matches, residual = self._scan_tool_call_shapes(cleaned_text)
        for _start, _end, name, args_json in scan_matches:
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": args_json,
                }
            )
        if scan_matches:
            # Collapse any runs of whitespace-only residual (left over from
            # blanked spans) so callers don't see chunks of empty space
            # in the content surface.
            cleaned_text = re.sub(r"\s+", " ", residual).strip()

        # Fallback: try lenient pattern for malformed tags like <tool_call without >
        if not tool_calls:
            lenient_matches = self.TOOL_CALL_LENIENT_PATTERN.findall(cleaned_text)
            for match in lenient_matches[:1]:  # Only first to avoid hallucinations
                try:
                    data = json.loads(match)
                    name = data.get("name", "")
                    arguments = data.get("arguments", {})
                    if name:
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": name,
                                "arguments": (
                                    json.dumps(arguments, ensure_ascii=False)
                                    if isinstance(arguments, dict)
                                    else str(arguments)
                                ),
                            }
                        )
                        cleaned_text = self.TOOL_CALL_LENIENT_PATTERN.sub(
                            "", cleaned_text, count=1
                        ).strip()
                except json.JSONDecodeError:
                    continue

        # Fallback: try raw JSON format if no tagged tool calls found
        # Only parse the FIRST valid tool call to avoid hallucinated multiple calls
        if not tool_calls:
            raw_matches = self.RAW_JSON_TOOL_PATTERN.findall(cleaned_text)
            if raw_matches:
                name, args_str = raw_matches[0]
                try:
                    arguments = json.loads(args_str)
                    valid_tool = True
                    if request and "tools" in request:
                        tool_names = [
                            t.get("function", {}).get("name", "")
                            for t in request.get("tools", [])
                            if isinstance(t, dict)
                        ]
                        valid_tool = name in tool_names

                    if valid_tool and name:
                        tool_calls.append(
                            {
                                "id": generate_tool_id(),
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            }
                        )
                        cleaned_text = self.RAW_JSON_TOOL_PATTERN.sub(
                            "", cleaned_text, count=1
                        ).strip()
                except json.JSONDecodeError:
                    pass

        # Include reasoning in content if present
        if reasoning_matches:
            reasoning_text = " ".join(reasoning_matches)
            if cleaned_text:
                cleaned_text = f"{cleaned_text}\n\n(Reasoning: {reasoning_text})"
            else:
                cleaned_text = f"(Reasoning: {reasoning_text})"

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

    @staticmethod
    def _format_streaming_tool_calls(
        tool_calls: list[dict], start_index: int = 0
    ) -> dict[str, Any]:
        """Format tool calls for streaming response."""
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

    # Tool-call sentinels that the streaming branch must hold-back
    # partial prefixes of (issue #448 + F-042 redo). When a char-level
    # delivery emits ``<``, ``<f``, ``<fun``... ahead of the full opener,
    # those partial bytes used to fall through to ``{"content": delta_text}``
    # and leak as content. The ``_safe_content_prefix`` helper trims the
    # longest sentinel-prefix suffix off any candidate emission so partial
    # sentinels are held back until either (a) the full sentinel arrives
    # and the tool-call branch claims it or (b) a non-matching char
    # arrives and the held bytes are released as ordinary content.
    #
    # ``<function>`` (no ``=``) is included so the F-042 VibeThinker shape
    # ``<function><name>...`` is held back like the other openers. To avoid
    # holding ordinary prose mentioning a literal ``<function>``, the
    # streaming branch only enters the named-XML state when ``<function>``
    # is followed by ``<name`` (the ``_FUNCTION_XML_OPENER_RE`` guard).
    _STREAMING_SENTINELS = ("<tool_call>", "<function=", "<function>")

    # Discriminator: a bare ``<function>`` is NOT a tool-call opener; only
    # ``<function>`` immediately followed (after optional whitespace) by
    # ``<name>`` qualifies as the VibeThinker named-XML opener. Without
    # this guard the streaming branch would suppress prose explaining the
    # tag in plain text.
    _FUNCTION_XML_OPENER_RE = re.compile(r"<function>\s*<name>")

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Strip the longest tool-call-sentinel prefix off ``text``'s tail.

        Returns the portion of ``text`` that is safe to emit as content
        right now. The trimmed suffix is the longest non-empty proper
        prefix of any ``_STREAMING_SENTINELS`` entry that also forms a
        suffix of ``text``. Returns an empty string when the entire
        ``text`` is a sentinel prefix (e.g. ``"<"`` ⇒ hold everything).
        """
        max_hold = 0
        for sentinel in cls._STREAMING_SENTINELS:
            for length in range(min(len(text), len(sentinel) - 1), 0, -1):
                if text.endswith(sentinel[:length]):
                    if length > max_hold:
                        max_hold = length
                    break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    @classmethod
    def _emit_safe_content(
        cls, previous_text: str, current_text: str
    ) -> dict[str, Any] | None:
        """Emit the new content delta with sentinel prefixes held back.

        Computes the safe-to-emit prefix of ``current_text`` and
        ``previous_text`` and returns only the diff. When the diff is
        empty (everything new is a held sentinel prefix), returns
        ``None`` so no content event fires this round.
        """
        safe_current = cls._safe_content_prefix(current_text)
        safe_previous = cls._safe_content_prefix(previous_text)
        if len(safe_current) <= len(safe_previous):
            return None
        return {"content": safe_current[len(safe_previous) :]}

    def flush_held_content(self, full_text: str) -> str:
        """Release the prefix-held suffix at stream end.

        The streaming branch holds back partial sentinel suffixes
        (``<``, ``<f``, ``<fu``...) until either the full opener
        arrives or a non-matching char arrives. When the stream ends
        with bytes still held, those bytes are ordinary content and
        must be released — otherwise a model response ending in
        ``abc<`` would surface as ``abc`` to the user (codex round-3
        CRITICAL).
        """
        return full_text[len(self._safe_content_prefix(full_text)) :]

    @classmethod
    def _has_incomplete_structured_block(cls, text: str) -> bool:
        """Heuristic: is the text currently inside an unclosed structured
        block of any of the three wire shapes? Used by the streaming
        branch to decide whether to suppress emit while a block is
        being assembled."""
        # Shape #1 + #2 (<tool_call> wrapper)
        if text.count("<tool_call>") > text.count("</tool_call>"):
            return True
        # Shape #3: <function=NAME> openers without matching </function>.
        # We must not double-count the ``</function>`` inside a closed
        # Nemotron-wrapped <tool_call> block, but if the <tool_call> is
        # already closed those inner ``</function>`` tags are already
        # consumed in the open/close balance check above.
        fn_eq_open = text.count("<function=")
        fn_named_open = len(cls._FUNCTION_XML_OPENER_RE.findall(text))
        fn_close = text.count("</function>")
        # Nemotron-wrapped blocks contribute 1 ``<function=`` and 1
        # ``</function>`` each — they balance, so we can subtract both
        # sides equally without affecting the open>close test.
        if fn_eq_open + fn_named_open > fn_close:
            return True
        return False

    @classmethod
    def _completed_structured_tool_calls(cls, text: str) -> int:
        """Count completed structured tool calls in ``text``.

        Returns the count of *fully-closed* structured blocks across
        all three wire shapes by running the same left-to-right scan
        ``extract_tool_calls`` uses but ignoring everything except the
        match count. This is the streaming branch's source of truth
        for deciding whether to emit ``tool_calls`` deltas.

        Using the scan rather than counting close-tag tokens
        (``</tool_call>`` / ``</function>``) handles the awkward
        intermediate state where ``</function>`` arrives BEFORE its
        enclosing ``</tool_call>`` (Nemotron-wrapped streaming) —
        the scan correctly waits for the wrapper close, while raw
        tag counting would briefly count a Nemotron inner close as a
        standalone bare-function close.
        """
        matches, _ = cls._scan_tool_call_shapes(text)
        return len(matches)

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
        Extract tool calls from streaming Hermes model output.

        Routes ALL three structured shapes (``<tool_call>``,
        ``<function=NAME>``, ``<function><name>``) through a single
        streaming branch that re-runs the unified left-to-right scan
        from ``extract_tool_calls`` whenever a new block closes. This
        is the streaming counterpart of the F-042 redo: a stream that
        carries shape #1 followed by shape #4 now routes BOTH through
        ``_format_streaming_tool_calls`` instead of leaking the second
        as content (P2-2 fix).

        Partial tool-call sentinels (``<tool_``, ``<function``…) are
        held back via ``_emit_safe_content`` so per-char streaming
        doesn't leak them as content deltas before the full opener
        arrives (issue #448 / BUG-3 family-wide leak).
        """
        has_any_opener = (
            "<tool_call>" in current_text
            or "<function=" in current_text
            or self._FUNCTION_XML_OPENER_RE.search(current_text) is not None
        )

        if has_any_opener:
            if self._has_incomplete_structured_block(current_text):
                # Inside an incomplete structured block — suppress output.
                return None

            # Count COMPLETED structured-shape blocks so the streaming
            # branch only emits when a structured block actually closes
            # in this delta. Counting close tags directly (rather than
            # re-running the unified scan on previous_text) avoids
            # spuriously treating an in-flight ``<tool_call>{...``
            # whose inner JSON is already complete as a finished call —
            # the unified scan would skip the unclosed opener and pick
            # up bare shapes nested inside (e.g. mid-stream Nemotron
            # XML reaches ``</function>\n`` before its outer
            # ``</tool_call>``).
            prev_completed = self._completed_structured_tool_calls(previous_text)
            cur_completed = self._completed_structured_tool_calls(current_text)
            if cur_completed > prev_completed:
                # Re-run the source-of-truth scan on current_text to
                # get the WIRE-ORDERED list of completed tool calls in
                # the dict form the streaming emitter expects, then
                # slice past the count already emitted.
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called and len(result.tool_calls) > prev_completed:
                    new_calls = result.tool_calls[prev_completed:]
                    if new_calls:
                        return self._format_streaming_tool_calls(
                            new_calls, start_index=prev_completed
                        )

            # All current tool calls already emitted; emit post-call
            # content with prefix-hold applied so any partial sentinel
            # at the new tail doesn't leak.
            return self._emit_safe_content(previous_text, current_text)

        # Fallback: check for raw JSON tool calls (detect closing brace pattern)
        if '{"name":' in current_text and '"arguments":' in current_text:
            if delta_text.rstrip().endswith("}"):
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    return self._format_streaming_tool_calls(result.tool_calls)
            return None

        return self._emit_safe_content(previous_text, current_text)
