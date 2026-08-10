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


def _completed_blocks(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, body)`` for each completed ATEM block, in order.

    A block's close is the first ``</atem:function_calls>`` at which the
    body's invoke structure is BALANCED (opens <= closes). A literal
    block closer inside a parameter value sits between an
    ``<atem:invoke`` and its ``</atem:invoke>``, leaves the count
    unbalanced, and is skipped over instead of truncating the real call
    (codex r3 #1 — the block-level twin of the parameter scan's
    last-closer rule).

    Prefix-stable: whether a block is complete, and where it closes,
    depends only on text up to that close — so as a stream grows,
    already-completed blocks never change. The streaming path's
    ``_stream_blocks_emitted`` cursor relies on this.
    """
    blocks: list[tuple[int, int, str]] = []
    pos = 0
    while True:
        open_idx = text.find(_BLOCK_OPEN, pos)
        if open_idx < 0:
            break
        body_start = open_idx + len(_BLOCK_OPEN)
        search = body_start
        close_idx = -1
        while True:
            candidate = text.find(_BLOCK_CLOSE, search)
            if candidate < 0:
                break
            body = text[body_start:candidate]
            if body.count("<atem:invoke") <= body.count("</atem:invoke>"):
                close_idx = candidate
                break
            search = candidate + len(_BLOCK_CLOSE)
        if close_idx < 0:
            break
        end = close_idx + len(_BLOCK_CLOSE)
        blocks.append((open_idx, end, text[body_start:close_idx]))
        pos = end
    return blocks


def _declared_type(cfg: Any) -> str | None:
    """Normalize a property schema to one simple type name, or None.

    Handles the shapes real tool schemas use for nullability (codex r3
    #3): ``{"type": ["integer", "null"]}`` and single-non-null
    ``anyOf``/``oneOf`` unions both normalize to the non-null type.
    Ambiguous unions return None (value falls to the as-is rules).
    """
    if not isinstance(cfg, dict):
        return None
    declared = cfg.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        non_null = [t for t in declared if t != "null"]
        if len(non_null) == 1 and isinstance(non_null[0], str):
            return non_null[0]
        return None
    for key in ("anyOf", "oneOf"):
        subs = cfg.get(key)
        if isinstance(subs, list):
            types = {_declared_type(sub) for sub in subs}
            types.discard(None)
            types.discard("null")
            if len(types) == 1:
                return next(iter(types))
    return None


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
    declared = _declared_type(cfg)
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
        self._stream_blocks_emitted = 0
        # Chars of the visible-content view already sent to the client.
        # An absolute cursor (not a prev-vs-curr diff): a delta that both
        # completes a block AND carries trailing text returns tool_calls
        # for that call, and the trailing text is picked up on the next
        # call because the cursor did not advance past it (codex r2 #1).
        self._content_emitted = 0

    @staticmethod
    def _parse_block(body: str, request: dict[str, Any] | None) -> list[dict[str, str]]:
        """Parse the invokes of ONE completed block body.

        Scoping the invoke scan to completed block bodies (rather than the
        whole response) is load-bearing: invoke-shaped text OUTSIDE a
        block — the model quoting an example, documentation in prose —
        must stay literal content, never become an executable call
        (codex r1 BLOCKING #2; the #44 injection-surface concern).
        """
        calls: list[dict[str, str]] = []
        for match in _INVOKE_RE.finditer(body):
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
        return calls

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        calls = [
            call
            for _, _, body in _completed_blocks(model_output)
            for call in self._parse_block(body, request)
        ]
        # One content rule for BOTH modes (codex r2 #2, r3 #2): the
        # ``hold_tail=False`` visible view — parseable blocks removed,
        # malformed blocks and truncated openers kept as literal bytes,
        # channel plumbing stripped, no ``.strip()``.
        content = self._visible_content(model_output, hold_tail=False) or None
        return ExtractedToolCallInformation(bool(calls), calls, content)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @classmethod
    def _visible_content(cls, text: str, *, hold_tail: bool = True) -> str:
        """Channel-clean content emittable for ``text`` so far.

        Completed ATEM blocks are removed (their calls surface on the
        tool_calls channel), the region from any UNMATCHED opener on is
        held, and — when ``hold_tail`` — the usual in-flight holds apply:
        a growing channel header and partial sentinels. Content between
        and after blocks therefore streams normally instead of being
        swallowed once the first opener appears (codex r1 BLOCKING #1).

        Monotonic in ``text`` by construction: completed blocks are
        prefix-stable (see ``_completed_blocks``), a parseable block's
        removal never exposes earlier bytes, and a malformed block's
        bytes appear in one piece at its close. That is what lets
        callers emit from an absolute cursor.
        """
        blocks = _completed_blocks(text)
        pieces: list[str] = []
        pos = 0
        for start, end, body in blocks:
            pieces.append(text[pos:start])
            if _INVOKE_RE.search(body) is None:
                # Completed but malformed — no parseable invoke, so the
                # bytes are model output the client must see (codex r3
                # #2; non-stream applies the identical rule above).
                pieces.append(text[start:end])
            pos = end
        tail = text[pos:]
        first_open = tail.find(_BLOCK_OPEN)
        if first_open >= 0:
            if hold_tail:
                tail = tail[:first_open]
            # hold_tail=False (end of stream): the opener can no longer
            # complete, so its literal bytes are content — matching the
            # non-streaming path (codex r2 #3; the #1766 principle).
        elif hold_tail:
            # In-flight holds apply to the TAIL only — bytes before a
            # completed block are already-visible history and truncating
            # them would break monotonicity.
            #
            # A ``<|start|>`` whose ``<|message|>`` has not arrived is a
            # channel header still in flight — hold from it so its
            # plain-word bytes ("assistant to=x") cannot leak once
            # tokens are stripped.
            pending = tail.rfind("<|start|>")
            if pending >= 0 and "<|message|>" not in tail[pending:]:
                tail = tail[:pending]
            # The IMPLICIT first-segment header: while the ENTIRE output
            # is still a growing `` to=recipient`` (no ``<|message|>``
            # yet), classification is impossible. Includes the prefixes
            # of ``to=`` itself (`` t``, `` to``). Only possible at the
            # very start of the generation — i.e. before any block.
            elif (
                not blocks
                and "<|message|>" not in tail
                and re.fullmatch(r"\s?(?:t|to|to=\S*)?", tail)
            ):
                tail = ""
            else:
                tail = cls._hold_partial_sentinel(tail)
        return _strip_channel_plumbing("".join(pieces) + tail)

    @classmethod
    def _hold_partial_sentinel(cls, text: str) -> str:
        max_hold = 0
        for sentinel in _STREAMING_SENTINELS:
            for length in range(min(len(text), len(sentinel) - 1), 0, -1):
                if text.endswith(sentinel[:length]):
                    max_hold = max(max_hold, length)
                    break
        return text if max_hold == 0 else text[: len(text) - max_hold]

    def has_pending_tool_call(self, text: str) -> bool:
        blocks = _completed_blocks(text)
        after = blocks[-1][1] if blocks else 0
        return _BLOCK_OPEN in text[after:]

    def flush_held_content(self, full_text: str) -> str:
        """Release held-but-safe bytes at end of stream.

        The unheld view (partial sentinels and unmatched-opener text
        released — the stream is over, they can no longer grow into
        markup) minus what the per-delta path already emitted, tracked
        by the ``_content_emitted`` cursor.
        """
        final = self._visible_content(full_text, hold_tail=False)
        cursor = getattr(self, "_content_emitted", 0)
        return final[cursor:]

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

        # A block close just completed — parse ONLY the newly completed
        # blocks (already-emitted blocks keep their ids and are never
        # re-scanned; codex r1 #4). ``_completed_blocks`` is
        # prefix-stable, so the cursor into its result is safe.
        prev_closes = previous_text.count(_BLOCK_CLOSE)
        curr_closes = current_text.count(_BLOCK_CLOSE)
        if curr_closes > prev_closes:
            blocks = _completed_blocks(current_text)
            new_blocks = blocks[self._stream_blocks_emitted :]
            self._stream_blocks_emitted = len(blocks)
            fresh = [
                call
                for _, _, body in new_blocks
                for call in self._parse_block(body, request)
            ]
            if fresh:
                offset = self._stream_calls_emitted
                self._stream_calls_emitted += len(fresh)
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

        # Content channel: everything outside completed blocks, held at
        # any in-flight opener/header/sentinel. Emitted from an absolute
        # cursor, not a prev/curr diff, so bytes that arrived in the same
        # delta as a block close are not skipped (codex r2 #1).
        visible = self._visible_content(current_text)
        if len(visible) > self._content_emitted:
            new_content = visible[self._content_emitted :]
            self._content_emitted = len(visible)
            return {"content": new_content}
        return None
