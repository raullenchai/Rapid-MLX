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
  therefore contain arbitrary XML-ish text — literal closers, literal
  openers, whole fake structures (#1730's bug class, and the #44
  injection-surface concern).

The defense is a POSITIONAL structural scan, not substring counting: a
closing tag only closes a structure when the text after it (skipping
whitespace) is a legal continuation of the grammar — the next opener,
the enclosing structure's closer, or end of input. Structural tags
INSIDE a parameter value are never examined, because the scanner is in
the value state until a boundary-valid closer appears. This is the
same "canonical closing line" discipline the Qwen3-Coder series
converged on. The residual ambiguity (a value whose literal text ends
with a boundary-valid closer sequence) is irreducible on this wire;
the failure mode is always "unparseable stays visible content", never
a wrong or truncated executable call.

The tool name is duplicated in the channel header (``to=NAME``) and
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
import math
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
_INVOKE_CLOSE = "</atem:invoke>"

_INVOKE_OPEN_RE = re.compile(r'<atem:invoke\s+name="(?P<name>[^"]+)">')
_PARAM_OPEN_RE = re.compile(r'<atem:parameter\s+name="(?P<name>[^"]+)">')
_WS_RE = re.compile(r"[ \t\r\n]*")

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

# A completed block: (start_offset, end_offset, invokes) where invokes
# is [(tool_name, [(param_name, raw_value), ...]), ...].
_Block = tuple[int, int, list[tuple[str, list[tuple[str, str]]]]]


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _walk_invoke(
    text: str, opener: re.Match[str]
) -> tuple[str, list[tuple[str, str]], int] | None:
    """Walk one invoke starting at ``opener``; None if malformed/incomplete.

    Positional grammar walk: after the invoke opener, the only legal
    tokens (whitespace-separated) are parameter openers and the invoke
    closer. Inside a parameter value, NOTHING is structural until a
    ``</atem:parameter>`` whose lookahead is a legal continuation — the
    next parameter opener, the invoke closer, or end of input. Literal
    tags inside values are therefore never mistaken for structure
    (codex r4 #3 / r5 #1–#3), and a parameter that never closes makes
    the whole invoke unparseable rather than an executable call with a
    truncated value.
    """
    params: list[tuple[str, str]] = []
    cursor = opener.end()
    while True:
        j = _WS_RE.match(text, cursor).end()
        pm = _PARAM_OPEN_RE.match(text, j)
        if pm is not None:
            value_start = pm.end()
            pos = value_start
            closed = False
            while True:
                idx = text.find(_PARAM_CLOSE, pos)
                if idx < 0:
                    break
                after = idx + len(_PARAM_CLOSE)
                k = _WS_RE.match(text, after).end()
                if (
                    k >= len(text)
                    or _PARAM_OPEN_RE.match(text, k) is not None
                    or text.startswith(_INVOKE_CLOSE, k)
                ):
                    params.append((pm.group("name"), text[value_start:idx]))
                    cursor = after
                    closed = True
                    break
                pos = after
            if not closed:
                return None
            continue
        if text.startswith(_INVOKE_CLOSE, j):
            return (opener.group("name"), params, j + len(_INVOKE_CLOSE))
        # Unexpected bytes (or end of input) inside the invoke.
        return None


def _completed_blocks(text: str) -> list[_Block]:
    """Every completed, structurally valid ATEM block in ``text``.

    Prefix-stable: a completed block's geometry depends only on the
    bytes up to its close (every lookahead that validated a closer is
    internal to the block once the block close itself validates), so as
    a stream grows, already-completed blocks never change. The
    streaming path's ``_stream_blocks_emitted`` cursor relies on this.

    A block whose interior fails the walk is treated as in-flight /
    malformed: scanning stops at its opener, and the bytes surface as
    visible content instead of calls.
    """
    blocks: list[_Block] = []
    pos = 0
    while True:
        open_idx = text.find(_BLOCK_OPEN, pos)
        if open_idx < 0:
            break
        invokes: list[tuple[str, list[tuple[str, str]]]] = []
        cursor = open_idx + len(_BLOCK_OPEN)
        completed = False
        while True:
            j = _WS_RE.match(text, cursor).end()
            im = _INVOKE_OPEN_RE.match(text, j)
            if im is not None:
                walked = _walk_invoke(text, im)
                if walked is None:
                    break
                name, params, end = walked
                invokes.append((name, params))
                cursor = end
                continue
            if text.startswith(_BLOCK_CLOSE, j):
                blocks.append((open_idx, j + len(_BLOCK_CLOSE), invokes))
                pos = j + len(_BLOCK_CLOSE)
                completed = True
                break
            break
        if not completed:
            break
    return blocks


def _allows_null(cfg: Any) -> bool:
    """Whether the property schema explicitly permits ``null``."""
    if not isinstance(cfg, dict):
        return False
    declared = cfg.get("type")
    if declared == "null":
        return True
    if isinstance(declared, list) and "null" in declared:
        return True
    return any(
        any(_allows_null(sub) for sub in cfg[key])
        for key in ("anyOf", "oneOf")
        if isinstance(cfg.get(key), list)
    )


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
    * declared boolean/integer/number -> parsed scalar (raw on failure
      or non-finite floats, so strict schema validation downstream
      rejects it visibly and ``json.dumps`` never emits NaN/Infinity
      tokens — codex r4 #1 / r5 #4);
    * declared object/array -> ``json.loads`` (raw on failure);
    * ``null`` -> None only when the schema explicitly permits null;
    * undeclared/unknown -> JSON only when it unambiguously parses to a
      container (the template only emits JSON for containers); any other
      shape stays a raw string. A bare ``5`` or ``true`` with no schema
      stays a string — guessing would corrupt legitimate string values.
    """
    cfg = props.get(param_name)
    declared = _declared_type(cfg)
    if declared == "string":
        return raw
    if raw == "null" and _allows_null(cfg):
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
            value = float(raw)
        except ValueError:
            return raw
        return value if math.isfinite(value) else raw
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
    def _convert_invokes(
        invokes: list[tuple[str, list[tuple[str, str]]]],
        request: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        """Turn structurally captured invokes into wire tool calls.

        The declared-name filter drops parameters the tool's schema does
        not declare (#1551's ambiguity class); a duplicated name keeps
        the FIRST occurrence so later text can never overwrite an
        already-captured value (codex r4 #2).
        """
        calls: list[dict[str, str]] = []
        for name, params in invokes:
            props = _schema_properties(name, request)
            declared = declared_parameter_names(name, request)
            arguments: dict[str, Any] = {}
            for pname, raw in params:
                if declared is not None and pname not in declared:
                    continue
                if pname in arguments:
                    continue
                arguments[pname] = _convert_param_value(raw, pname, props)
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
            for _, _, invokes in _completed_blocks(model_output)
            for call in self._convert_invokes(invokes, request)
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

        Completed blocks WITH invokes are removed (their calls surface
        on the tool_calls channel); a completed block with none is model
        output the client must see (codex r3 #2). The region from any
        unconsumed opener on is held while the stream runs, and — when
        ``hold_tail`` — the usual in-flight holds apply: a growing
        channel header and partial sentinels. Content between and after
        blocks therefore streams normally (codex r1 #1).

        Monotonic in ``text`` by construction: completed blocks are
        prefix-stable (see ``_completed_blocks``), a parseable block's
        removal never exposes earlier bytes, and an empty block's bytes
        appear in one piece at its close. That is what lets callers
        emit from an absolute cursor.
        """
        blocks = _completed_blocks(text)
        pieces: list[str] = []
        pos = 0
        for start, end, invokes in blocks:
            pieces.append(text[pos:start])
            if not invokes:
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
                for _, _, invokes in new_blocks
                for call in self._convert_invokes(invokes, request)
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
