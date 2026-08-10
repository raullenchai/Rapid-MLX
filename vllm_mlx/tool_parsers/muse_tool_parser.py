# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Meta Muse Glimmer's ATEM wire format.

Muse Glimmer (meta-models/Muse-Glimmer-30B, 2026-08-10) renders tool
calls as an Anthropic-style XML block inside a Harmony-style channel
message::

    <|start|>assistant to=get_weather<|message|><atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Paris</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eot|>

Everything the parser relies on is pinned by the model's own chat
template (``chat_template.jinja`` on the HF repo):

* String and scalar parameter values are rendered **as is** — no JSON
  quoting, and "spaces for string values are not stripped". Lists and
  objects are rendered with ``tojson``. So value typing comes from the
  request's tool schema, not from the wire.
* The template itself warns: "The output is not expected to be valid
  XML and is parsed with regular expressions." A string value may
  therefore contain XML-ish text, including a literal
  ``</atem:parameter>`` — the same delimiter-ambiguity class as the
  Qwen3-Coder fixes in #1730. Two defenses here: a parameter's value
  runs to the LAST closer before the next opener (not the first), and
  when the request declares the tool's parameter names, an opener whose
  name is not declared is treated as literal value text
  (``declared_parameter_names``, same rationale as the qwen scanner).
* The tool name is duplicated in the channel header (``to=NAME``) and
  the invoke tag; the invoke tag is authoritative — the header is
  routing plumbing and is stripped with the other channel tokens.

The companion ``MuseReasoningParser`` routes ``to=self`` segments to
``reasoning_content`` BEFORE this parser sees the stream, so on the
production path this parser receives channel-clean content plus ATEM
blocks. It still strips channel tokens itself so that standalone use
(no reasoning parser configured) does not leak plumbing into content.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    declared_parameter_names,
)

_BLOCK_OPEN = "<atem:function_calls>"
_BLOCK_CLOSE = "</atem:function_calls>"
_PARAM_CLOSE = "</atem:parameter>"

_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="(?P<name>[^"]+)">(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_PARAM_OPEN_RE = re.compile(r'<atem:parameter\s+name="(?P<name>[^"]+)">')

# Muse channel plumbing (Harmony-flavored, but a distinct token set:
# <|eot|>/<|eom|> terminators instead of <|end|>/<|return|>/<|call|>,
# and recipient routing via ``to=`` in the <|start|> header).
_CHANNEL_HEADER_RE = re.compile(r"<\|start\|>assistant(?:\s+to=\S+?)?\s*<\|message\|>")
# The FIRST segment of a generation has an implicit header: the prompt
# ends with ``<|start|>assistant``, so output begins directly with
# `` to=recipient<|message|>`` (or a bare ``<|message|>``).
_IMPLICIT_HEADER_RE = re.compile(r"^\s?(?:to=\S+?)?\s*<\|message\|>")
_SELF_SEGMENT_RE = re.compile(
    r"(?:<\|start\|>assistant\s+|^\s?)to=self<\|message\|>.*?(?:<\|eom\|>|<\|eot\|>|$)",
    re.DOTALL,
)
_CONTROL_TOKENS = ("<|start|>", "<|message|>", "<|eot|>", "<|eom|>")

# Sentinels whose partial prefixes must be held back from streamed
# content so per-char streaming doesn't leak ``<``, ``<a``, ``<atem:``…
# before the full opener arrives (same bug class as #444/#480).
_STREAMING_SENTINELS: tuple[str, ...] = (
    _BLOCK_OPEN,
    "<|start|>",
    "<|message|>",
    "<|eot|>",
    "<|eom|>",
)


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _schema_properties(tool_name: str, request: dict[str, Any] | None) -> dict:
    """The declared JSON-schema ``properties`` for ``tool_name``, or {}."""
    if not isinstance(request, dict):
        return {}
    for tool in request.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == tool_name:
            props = (fn.get("parameters") or {}).get("properties")
            return props if isinstance(props, dict) else {}
    return {}


def _convert_param_value(raw: str, param_name: str, props: dict) -> Any:
    """Type a raw ATEM value using the tool schema.

    ATEM's contract (from the chat template): strings and scalars are
    rendered bare, lists/objects as JSON. So:

    * declared string  -> raw, verbatim (whitespace preserved — the
      template says so explicitly, and #1444's ``_decode_json_like``
      whitespace-eating bug is the cautionary tale);
    * declared boolean/integer/number -> parsed scalar (raw on failure,
      so strict schema validation downstream rejects it visibly);
    * declared object/array -> ``json.loads`` (raw on failure);
    * ``null`` for a nullable non-string -> None;
    * undeclared/unknown -> JSON only when it unambiguously parses to a
      container (the template only emits JSON for containers); any other
      shape stays a raw string. A bare ``5`` or ``true`` with no schema
      stays a string — guessing would corrupt legitimate string values.
    """
    cfg = props.get(param_name)
    declared = None
    if isinstance(cfg, dict):
        declared = cfg.get("type")
    if declared == "string":
        return raw
    if raw == "null" and declared is not None:
        return None
    if declared == "boolean":
        if raw in ("true", "false"):
            return raw == "true"
        return raw
    if declared == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw
    if declared == "number":
        try:
            return float(raw)
        except ValueError:
            return raw
    if declared in ("object", "array"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    # Undeclared or exotic type: containers self-identify as JSON.
    stripped = raw.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return raw


def _scan_parameters(body: str, declared: set[str] | None) -> list[tuple[str, str]]:
    """Extract ``(name, raw_value)`` pairs from an invoke body.

    A value runs from its opener to the LAST ``</atem:parameter>``
    before the next accepted opener (or the body end). Combined with
    the declared-name filter, this survives values that contain a
    literal closer or an XML-ish fake opener — the wire is genuinely
    ambiguous there (the template says regex, not XML), and the schema
    is the only disambiguator, exactly as in the qwen scanner.
    """
    openers = [
        m
        for m in _PARAM_OPEN_RE.finditer(body)
        if declared is None or m.group("name") in declared
    ]
    params: list[tuple[str, str]] = []
    for i, m in enumerate(openers):
        seg_end = openers[i + 1].start() if i + 1 < len(openers) else len(body)
        segment = body[m.end() : seg_end]
        close = segment.rfind(_PARAM_CLOSE)
        value = segment[:close] if close >= 0 else segment
        params.append((m.group("name"), value))
    return params


def _strip_channel_plumbing(text: str) -> str:
    """Remove Muse channel tokens and ``to=self`` reasoning segments."""
    # Reasoning segments first (they carry their own header/terminator).
    result = _SELF_SEGMENT_RE.sub("", text)
    result = _CHANNEL_HEADER_RE.sub("", result)
    result = _IMPLICIT_HEADER_RE.sub("", result)
    for token in _CONTROL_TOKENS:
        result = result.replace(token, "")
    return result


@ToolParserManager.register_module(["muse"])
class MuseToolParser(ToolParser):
    """Tool-call parser for Muse Glimmer's ATEM function-calls blocks."""

    # The Muse chat template natively renders ``tool_calls`` (render_atem)
    # and ``role="tool"`` results (<tool_output>) — without this flag the
    # engine would flatten tool history into "[Calling tool: ...]" text
    # the model never saw in training (#1593's bug class).
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    EXPECTED_WIRE_FORMATS = ("muse_atem",)

    def reset(self) -> None:
        super().reset()
        self._stream_calls_emitted = 0

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if _BLOCK_OPEN not in model_output:
            return ExtractedToolCallInformation(
                False, [], _strip_channel_plumbing(model_output) or None
            )

        calls: list[dict[str, str]] = []
        for match in _INVOKE_RE.finditer(model_output):
            name = match.group("name")
            props = _schema_properties(name, request)
            declared = declared_parameter_names(name, request)
            arguments = {
                pname: _convert_param_value(raw, pname, props)
                for pname, raw in _scan_parameters(match.group("body"), declared)
            }
            calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        # Content = whatever sits outside the ATEM blocks, channel-clean.
        outside = re.sub(
            re.escape(_BLOCK_OPEN) + r".*?" + re.escape(_BLOCK_CLOSE),
            "",
            model_output,
            flags=re.DOTALL,
        )
        content = _strip_channel_plumbing(outside).strip() or None

        if not calls:
            # An opener with no parseable invoke (malformed / truncated):
            # keep the raw text as content rather than dropping bytes.
            return ExtractedToolCallInformation(
                False, [], _strip_channel_plumbing(model_output) or None
            )
        return ExtractedToolCallInformation(True, calls, content)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @classmethod
    def _safe_content_prefix(cls, text: str) -> str:
        """Hold back any tail that could grow into a sentinel."""
        first_open = text.find(_BLOCK_OPEN)
        if first_open >= 0:
            return text[:first_open]
        # A ``<|start|>`` whose ``<|message|>`` has not arrived is a
        # channel header still in flight — hold from it so its plain-word
        # bytes ("assistant to=x") cannot leak once tokens are stripped.
        pending = text.rfind("<|start|>")
        if pending >= 0 and "<|message|>" not in text[pending:]:
            return text[:pending]
        # Same for the IMPLICIT first-segment header: while the entire
        # output is still a growing `` to=recipient`` (no ``<|message|>``
        # yet), classification is impossible and emitting would leak
        # header bytes as content. Once any non-header shape appears the
        # rule stops matching and normal emission resumes.
        if "<|message|>" not in text and re.fullmatch(r"\s?(?:t|to|to=\S*)?", text):
            return ""
        max_hold = 0
        for sentinel in _STREAMING_SENTINELS:
            for length in range(min(len(text), len(sentinel) - 1), 0, -1):
                if text.endswith(sentinel[:length]):
                    max_hold = max(max_hold, length)
                    break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    def has_pending_tool_call(self, text: str) -> bool:
        return _BLOCK_OPEN in text or self._safe_content_prefix(text) != text

    def flush_held_content(self, full_text: str) -> str:
        if _BLOCK_OPEN in full_text:
            return ""
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
        if not hasattr(self, "_stream_calls_emitted"):
            self.reset()

        # A block close just completed — surface any not-yet-emitted calls.
        prev_closes = previous_text.count(_BLOCK_CLOSE)
        curr_closes = current_text.count(_BLOCK_CLOSE)
        if curr_closes > prev_closes:
            result = self.extract_tool_calls(current_text, request)
            if result.tools_called:
                fresh = result.tool_calls[self._stream_calls_emitted :]
                if fresh:
                    offset = self._stream_calls_emitted
                    self._stream_calls_emitted = len(result.tool_calls)
                    return {
                        "tool_calls": [
                            {
                                "index": offset + i,
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"],
                                },
                            }
                            for i, call in enumerate(fresh)
                        ]
                    }
            return None

        # Inside (or before) a block: emit only channel-clean safe content.
        if _BLOCK_OPEN in current_text:
            return None
        prev_safe = _strip_channel_plumbing(self._safe_content_prefix(previous_text))
        curr_safe = _strip_channel_plumbing(self._safe_content_prefix(current_text))
        if len(curr_safe) > len(prev_safe):
            return {"content": curr_safe[len(prev_safe) :]}
        return None
