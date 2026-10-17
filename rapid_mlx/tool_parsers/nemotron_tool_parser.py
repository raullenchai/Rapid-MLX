# SPDX-License-Identifier: Apache-2.0
"""
Nemotron tool call parser for rapid-mlx.

Handles NVIDIA Nemotron models' tool calling format:
- <tool_call><function=name><parameter=p>v</parameter></function></tool_call>

Supports Nemotron-3-Nano-30B-A3B and similar models.
"""

import json
import logging
import re
import uuid
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Any

from ..tool_call_scan import (
    declared_tool_names as _declared_tool_names,
)
from ..tool_call_scan import (
    split_marked_calls,
    split_marked_parameters,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    declared_parameter_names,
)

logger = logging.getLogger(__name__)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["nemotron", "nemotron3"])
class NemotronToolParser(ToolParser):
    """
    Tool call parser for NVIDIA Nemotron models.

    Supports Nemotron's tool call format:
    <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>

    Also supports JSON arguments:
    <tool_call><function=get_weather>{"city": "Paris"}</function></tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser nemotron are set.
    """

    EXPECTED_WIRE_FORMATS = ("tool_call_xml_body",)

    # Nemotron's tool template accepts structured assistant calls and tool
    # results, rendering its native function/parameter XML transcript.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # How much of ``current_text`` this turn has already been accounted for on
    # the wire — forwarded as prose, consumed into an emitted call, or released
    # after a refusal. The refusal release below sends only what lies past it.
    #
    # A watermark rather than a "released once" boolean: a turn can hold more
    # than one refused block, and a boolean drops every block after the first.
    # Class-level default so an instance used without a preceding ``reset()``
    # still has the attribute.
    _content_upto = 0
    _stream_started = False
    _post_call_wrapped = False
    _wrapped_whitespace: tuple[int, int] | None = None
    _last_spans: (
        tuple[str, frozenset[str] | None, list[tuple[str, str, int, int]]] | None
    ) = None

    # Pattern for Nemotron-style with parameters.
    #
    # The load-bearing signature of a call is ``<function=NAME>...</function>``;
    # the ``<tool_call>``/``</tool_call>`` wrapper is treated as optional /
    # decorative so that observed degradations still parse:
    #   (a) a missing/truncated ``</tool_call>``,
    #   (b) a bare ``<function=..>..</function>`` with no wrapper at all,
    #   (d) stray text between ``</function>`` and ``</tool_call>``,
    #   (e) prose between ``<tool_call>`` and ``<function=``.
    # Prose without ``<function=..>..</function>`` still never matches, so
    # this cannot manufacture a tool call out of plain text.
    TOOL_CALL_PATTERN = re.compile(
        r"<function=([^>]+)>(.*?)</function>",
        re.DOTALL,
    )

    # Residual bare wrapper tags left behind after the function bodies have
    # been stripped from ``content``; removed so they don't leak as text.
    RESIDUAL_WRAPPER_PATTERN = re.compile(r"</?tool_call>")
    _TAIL_OPENER_PATTERN = re.compile(r"<tool_call>|<function=")
    _TAIL_PAIR_PATTERN = re.compile(r"</function>\s*</tool_call>")
    _TAIL_PARTIAL_PATTERN = re.compile(r"</function>\s*(<[^>]*|)?$")
    _PARAMETER_CLOSE_PATTERN = re.compile(r"</parameter>\s*</function>$")

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    def reset(self) -> None:
        """Clear per-request state.

        The base reset owns the tool-id counters; the content watermark is
        just as per-request. Relying on the ``not previous_text`` branch alone
        is not enough — the postprocessor can forward a new turn's opening
        prose through its own fast path before this parser sees a delta, so
        the first call can arrive with ``previous_text`` already non-empty and
        a stale watermark from the turn before.
        """
        super().reset()
        self._content_upto = 0
        self._stream_started = False
        self._post_call_wrapped = False
        self._wrapped_whitespace = None
        self._last_spans = None
        self._undeclared_logged = False
        self._opener_scan_upto = 0
        self._opener_scan_result = False

    def flush_held_content(self, full_text: str) -> str:
        return self._release_held_content(full_text, final=True)

    def _release_held_content(self, full_text: str, *, final: bool = False) -> str:
        # Completed calls are already accounted for by the streaming watermark.
        # Inspect only the held range: old markers elsewhere in the response
        # must not hide trailing prose. A full opener or ambiguous suffix still
        # belongs to markup and must never be flushed onto the content channel.
        whitespace = self._wrapped_whitespace
        if (
            not final
            and self._post_call_wrapped
            and whitespace is not None
            and whitespace[0] == self._content_upto
            and len(full_text) >= whitespace[1]
        ):
            added = full_text[whitespace[1] :]
            if not added or added.isspace():
                # A held function close plus whitespace still cannot decide
                # whether a wrapper close follows. Check only new whitespace.
                self._wrapped_whitespace = (self._content_upto, len(full_text))
                return ""
        self._wrapped_whitespace = None
        held = full_text[self._content_upto :]
        if self._post_call_wrapped:
            consumed, pending_pair = self._wrapped_tail_boundary(held)
            self._content_upto += consumed
            held = held[consumed:]
            if pending_pair is not None:
                candidate = held[pending_pair:]
                if candidate.startswith("</function>"):
                    gap = candidate[len("</function>") :]
                    if not gap or gap.isspace():
                        self._wrapped_whitespace = (
                            self._content_upto,
                            len(full_text),
                        )
        else:
            pending_pair = None
        pending = self._pending_markup_index(
            held,
            include_closers=self._has_unmatched_opener(full_text, self._content_upto),
        )
        # At EOF a partial continuation cannot become a pair: release it as
        # prose. Complete openers and ambiguous opener prefixes stay held.
        if pending_pair is not None and not final:
            pending = 0
        safe_end = len(held) if pending is None else pending
        self._content_upto += safe_end
        return held[:safe_end]

    @staticmethod
    def _wrapped_tail_boundary(
        held: str, start: int = 0, stop: int | None = None
    ) -> tuple[int, int | None]:
        """Consume continuation pairs and hold a possible final pair prefix."""
        if stop is None:
            opener = NemotronToolParser._TAIL_OPENER_PATTERN.search(held, start)
            stop = opener.start() if opener else len(held)
        consumed = start
        # Buffered wrapped calls end at the LAST pair. Even prose containing
        # such a pair is payload by that rule; an emitted call cannot retract,
        # but its still-held continuation must never leak as visible content.
        for pair in NemotronToolParser._TAIL_PAIR_PATTERN.finditer(held, start, stop):
            consumed = pair.end()
        if stop < len(held):
            return consumed - start, None
        partial = NemotronToolParser._TAIL_PARTIAL_PATTERN.search(held, consumed, stop)
        if partial:
            suffix = partial.group(1) or ""
            if "</tool_call>".startswith(suffix):
                return consumed - start, partial.start() - consumed
        for size in range(1, len("</function>")):
            if held.endswith("</function>"[:size], consumed, stop):
                return consumed - start, stop - consumed - size
        return consumed - start, None

    def _has_unmatched_opener(self, text: str, end: int | None = None) -> bool:
        """Whether accounted markup still needs a function or wrapper close.

        Skip complete function spans using the same literal-safe scanner as
        extraction. Counting textual tags would let wrapper-shaped payload
        close a real wrapper, or leave a payload opener spuriously pending.
        """
        end = len(text) if end is None else end
        upto = getattr(self, "_opener_scan_upto", 0)
        if upto and end == upto:
            return self._opener_scan_result
        if upto and end >= upto:
            # Narration cannot change opener state, even with a wrapper still
            # open. Include bounded lookbehind for tags split at watermarks.
            markers: tuple[str, ...] = ("<function=", "<tool_call>")
            if self._opener_scan_result:
                markers += ("</function>", "</tool_call>")
            start = max(0, upto - max(map(len, markers)))
            if not any(text.find(marker, start, end) != -1 for marker in markers):
                self._opener_scan_upto = end
                return self._opener_scan_result
        cached = self._last_spans
        if (
            end == len(text)
            and cached is not None
            and cached[0] is text
            and cached[1] is None
        ):
            result = self._scan_unmatched_opener(text, cached[2])
        else:
            result = self._scan_unmatched_opener(text[:end])
        self._opener_scan_upto = end
        self._opener_scan_result = result
        return result

    @classmethod
    def _scan_unmatched_opener(cls, prefix: str, calls=None) -> bool:
        if calls is None:
            calls = cls._function_spans(prefix)
        cursor = 0
        wrappers = 0
        for start, end in [(a, b) for _, _, a, b in calls] + [
            (len(prefix), len(prefix))
        ]:
            outside = prefix[cursor:start]
            if "<function=" in outside:
                return True
            for marker in cls.RESIDUAL_WRAPPER_PATTERN.finditer(outside):
                if marker.group() == "<tool_call>":
                    wrappers += 1
                else:
                    wrappers = max(0, wrappers - 1)
            cursor = end
        return wrappers > 0

    @staticmethod
    def _is_wrapped(
        text: str, start: int, previous_end: int = 0, wrapped: bool = False
    ) -> bool:
        """Update wrapper state only from text outside completed payloads."""
        opener = text.rfind("<tool_call>", previous_end, start)
        closer = text.rfind("</tool_call>", previous_end, start)
        return opener > closer if max(opener, closer) >= 0 else wrapped

    @classmethod
    def _function_spans(cls, text: str, valid_names=None):
        """Bound shared literal-safe spans before post-call narration."""
        calls = split_marked_calls(
            text, r"<function=([^>]+)>", "</function>", valid_names=valid_names
        )
        bounded = []
        # A single forward scan bounds every tail, including literal openers
        # inside payloads. Never copy the whole remainder for each call.
        openers = [m.start() for m in cls._TAIL_OPENER_PATTERN.finditer(text)]
        previous_end = 0
        wrapped = False
        for name, body, start, end in calls:
            wrapped = cls._is_wrapped(text, start, previous_end, wrapped)
            body_start = text.index(">", start) + 1
            close = None
            if wrapped:
                next_opener = bisect_left(openers, body_start)
                stop = openers[next_opener] if next_opener < len(openers) else len(text)
                consumed, _ = cls._wrapped_tail_boundary(text, body_start, stop)
                if consumed and cls._PARAMETER_CLOSE_PATTERN.search(
                    text, body_start + consumed, end
                ):
                    # A later complete parameter payload already closed the
                    # function, even if its wrapper close is still arriving.
                    close = end - len("</function>")
                elif consumed:
                    close = text.rfind("</function>", body_start, body_start + consumed)
            if close is None:
                # Bare calls end at the first complete payload, matching the
                # streaming path. Missing wrapper closes retain that fallback.
                if body.lstrip().startswith("{"):
                    try:
                        _, offset = json.JSONDecoder().raw_decode(body.lstrip())
                        offset += body_start + len(body) - len(body.lstrip())
                        suffix = re.match(r"\s*(</function>)", text[offset:end])
                        if suffix:
                            close = offset + suffix.start(1)
                    except json.JSONDecodeError:
                        pass
                else:
                    payload = re.search(
                        r"</parameter>\s*(</function>)", text[body_start:end]
                    )
                    if payload:
                        close = body_start + payload.start(1)
            if close is not None:
                body, end = text[body_start:close], close + len("</function>")
            bounded.append((name, body, start, end))
            previous_end = end
        return bounded

    def _cached_function_spans(self, text: str, valid_names: frozenset[str] | None):
        # Keep the text alive and compare identity so neither equal-length
        # replacements nor recycled object ids can reuse another input's spans.
        cached = self._last_spans
        if cached is not None and cached[0] is text and cached[1] == valid_names:
            return cached[2]
        return self._function_spans(text, valid_names=valid_names)

    def _undeclared_spans(self, text, request, accepted):
        """Consume rejected siblings in bounded, unfenced call wrappers."""
        from .qwen3coder_tool_parser import _MarkdownCodeTracker

        declared = _declared_tool_names(request)
        if not declared:
            return []
        tracker = _MarkdownCodeTracker()
        spans = []
        wrapper_closes = [m.start() for m in re.finditer(r"</tool_call>", text)]
        accepted_ends = [b for _, _, _, b in accepted]
        consumed = 0
        for wrapper in re.finditer(r"<tool_call>", text):
            start = wrapper.start()
            if start < consumed:
                continue
            # Bound the search BEFORE looking for a close. Repeated unclosed
            # openers must not each rescan the entire remaining response.
            limit = min(len(text), start + 32768)
            close = text.find("</function>", wrapper.end(), limit)
            if close == -1:
                continue
            function_end = close + len("</function>")
            first_calls = split_marked_calls(
                text[wrapper.end() : function_end],
                r"<function=([^>]+)>",
                "</function>",
            )
            if not first_calls:
                continue
            first_start = wrapper.end() + first_calls[0][2]
            # A closed decorative wrapper cannot claim a later bare function.
            # A literal wrapper close *inside* the function remains payload.
            if text.find("</tool_call>", wrapper.end(), first_start) != -1:
                continue
            close_index = bisect_left(wrapper_closes, function_end)
            wrapper_close = (
                wrapper_closes[close_index] if close_index < len(wrapper_closes) else -1
            )
            next_wrapper = text.find("<tool_call>", function_end, limit)
            wrapped = wrapper_close != -1 and (
                next_wrapper == -1 or wrapper_close < next_wrapper
            )
            if wrapped and wrapper_close + len("</tool_call>") > limit:
                continue
            if tracker.inside(text, start):
                continue
            boundary = (
                wrapper_close
                if wrapped
                else (next_wrapper if next_wrapper != -1 else limit)
            )
            # Recognize every sibling with the same nested-opener rules as
            # accepted calls. Never cut a rejected-looking span out of an
            # accepted argument: marker-shaped payload must remain payload.
            calls = split_marked_calls(
                text[wrapper.end() : boundary],
                r"<function=([^>]+)>",
                "</function>",
            )
            rejected = []
            for name, _, a, b in calls:
                a += wrapper.end()
                b += wrapper.end()
                accepted_index = bisect_right(accepted_ends, a)
                overlaps = (
                    accepted_index < len(accepted) and accepted[accepted_index][2] < b
                )
                if name not in declared and not overlaps:
                    rejected.append((a, b))
            if not rejected:
                continue
            # Formatting-only whitespace belongs to the wrapper; narration
            # between the same boundaries remains ordinary assistant content.
            a = calls[0][2] + wrapper.end()
            b = calls[-1][3] + wrapper.end()
            prefix_end = a if text[wrapper.end() : a].isspace() else wrapper.end()
            spans.append((start, prefix_end))
            spans.extend(rejected)
            if wrapped:
                suffix_start = b if text[b:wrapper_close].isspace() else wrapper_close
                spans.append((suffix_start, wrapper_close + len("</tool_call>")))
            consumed = wrapper_close + len("</tool_call>") if wrapped else b
        return spans

    def _visible_content_between(
        self,
        text: str,
        start: int,
        end: int,
        request: dict[str, Any] | None,
    ) -> str:
        """Project a raw range onto the non-tool content stream.

        Valid function spans are executable calls and must stay off the content
        channel. Wrapped undeclared attempts are consumed; bare refused forms
        remain prose. When at least one valid call exists, the decorative outer wrappers are removed
        too, matching :meth:`extract_tool_calls`.

        Keeping source offsets here is important: a single model delta can
        finish a refused block and a later valid call. A plain byte watermark
        advanced for the valid call used to swallow the refused range.
        """
        if end <= start:
            return ""

        valid_spans = [
            (span_start, span_end)
            for _name, _body, span_start, span_end in self._cached_function_spans(
                text,
                valid_names=_declared_tool_names(request),
            )
        ]
        accepted = [("", "", a, b) for a, b in valid_spans]
        removed = list(valid_spans) + self._undeclared_spans(text, request, accepted)
        if valid_spans:
            # Only paired wrapper closers are markup. Orphan closers after a
            # completed call are literal prose, including in buffered replies.
            pending_wrappers = 0
            span_index = 0
            for match in self.RESIDUAL_WRAPPER_PATTERN.finditer(text):
                while (
                    span_index < len(valid_spans)
                    and valid_spans[span_index][1] <= match.start()
                ):
                    span_index += 1
                if (
                    span_index < len(valid_spans)
                    and valid_spans[span_index][0] <= match.start()
                ):
                    continue
                if match.group() == "<tool_call>":
                    pending_wrappers += 1
                elif pending_wrappers:
                    pending_wrappers -= 1
                else:
                    continue
                removed.append((match.start(), match.end()))
        # Accepted calls, rejected siblings and wrappers can overlap. Project
        # their union in source order so no earlier markup becomes a gap.
        merged: list[tuple[int, int]] = []
        for span_start, span_end in sorted(removed):
            if merged and span_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], span_end))
            else:
                merged.append((span_start, span_end))
        removed = merged

        parts: list[str] = []
        cursor = start
        removed_before = any(span_end == start for _, span_end in removed)
        for span_start, span_end in removed:
            if span_end <= start or span_start >= end:
                continue
            clipped_start = max(start, span_start)
            clipped_end = min(end, span_end)
            if cursor < clipped_start:
                gap = text[cursor:clipped_start]
                # Pretty-printed wire uses newlines between the decorative
                # wrapper and function body. If both neighboring spans are
                # removed, that formatting is markup too, not assistant text.
                if not (removed_before and gap.isspace()):
                    parts.append(gap)
            cursor = max(cursor, clipped_end)
            removed_before = True
        if cursor < end:
            parts.append(text[cursor:end])
        return "".join(parts)

    @staticmethod
    def _pending_markup_index(
        text: str, *, include_closers: bool = False
    ) -> int | None:
        """Return where recognized complete/partial markup begins, if any.

        A bare ``<`` in assistant prose (for example ``2 < 3``) is content,
        not sufficient evidence of a tool-call opener. Only suppress a suffix
        that is already a known marker or can still grow into one.
        """
        markers: tuple[str, ...] = ("<function=", "<tool_call>")
        # Only an actual preceding opener makes a closing tag markup. Orphan
        # closers in ordinary prose must pass through, including byte streams.
        if include_closers:
            markers += ("</function>", "</tool_call>")
        candidates: list[int] = []
        for marker in markers:
            full = text.find(marker)
            if full != -1:
                candidates.append(full)
            for prefix_len in range(1, len(marker)):
                if text.endswith(marker[:prefix_len]):
                    candidates.append(len(text) - prefix_len)
        return min(candidates) if candidates else None

    def _safe_accounting_boundary(self, text: str) -> int:
        """Return the raw offset safe to account after a close event.

        Plain trailing text is safe through the end. If another markup opener
        has started, stop at the latest complete close so its partial bytes can
        still be released or parsed on a later delta.
        """
        latest_close = 0
        for tag in ("</function>", "</tool_call>"):
            idx = text.rfind(tag)
            if idx != -1:
                latest_close = max(latest_close, idx + len(tag))
        if latest_close == 0:
            return 0
        tail = text[latest_close:]
        # Whitespace after a function may be wrapper formatting or the start
        # of narration. Hold it until the next byte resolves that ambiguity.
        if tail.isspace() and text.rfind("<tool_call>") > text.rfind("</tool_call>"):
            return latest_close
        pending = self._pending_markup_index(
            tail,
            include_closers=self._has_unmatched_opener(text, latest_close),
        )
        if pending is not None and tail[:pending].isspace():
            return latest_close
        return len(text) if pending is None else latest_close + pending

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Nemotron model output.
        """
        if "<tool_call>" not in model_output and "<function=" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        cleaned_text = model_output

        # Positional scan, not ``TOOL_CALL_PATTERN.findall``. The compiled
        # non-greedy patterns above stop at the FIRST closing marker, so a
        # literal ``</function>`` or ``</parameter>`` inside an argument
        # truncated the call and dropped the rest in silence — the defect
        # ``rapid_mlx/api/tool_calling.py`` had, fixed there, and still live
        # here because this is a second implementation of the same format.
        # Both now share ``rapid_mlx/tool_call_scan``; the patterns are kept
        # only as the format's documentation.
        # Declared tool names, same as the ``api/tool_calling`` path. This is
        # the second implementation of this wire format, so a gate added only
        # there leaves the other door open: argument text containing
        # ``</function><function=delete_everything>`` would still fabricate an
        # executable call here.
        valid_names = _declared_tool_names(request)
        matches = self._function_spans(
            model_output,
            valid_names=valid_names,
        )
        self._last_spans = (model_output, valid_names, matches)
        rejected = self._undeclared_spans(model_output, request, matches)
        self._undeclared_dropped = sum(
            model_output.startswith("<function=", a) for a, _ in rejected
        )
        if rejected and not getattr(self, "_undeclared_logged", False):
            logger.warning("nemotron tool parser: dropped undeclared tool attempt")
            self._undeclared_logged = True
        for func_name, content, _span_start, _span_end in matches:
            func_name = func_name.strip()

            # Try to parse content as JSON first
            content = content.strip()
            if content.startswith("{"):
                try:
                    json.loads(content)
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": content,
                        }
                    )
                    continue
                except json.JSONDecodeError:
                    pass

            # Parse parameter tags
            params = split_marked_parameters(
                content,
                r"<parameter=([^>]+)>",
                "</parameter>",
                valid_names=declared_parameter_names(func_name, request),
                reject_undeclared_siblings=True,
            )
            if params is None:
                continue
            if params:
                arguments = {}
                for param_name, param_value in params:
                    # Try to parse value as JSON (for nested objects)
                    try:
                        arguments[param_name] = json.loads(param_value)
                    except json.JSONDecodeError:
                        arguments[param_name] = param_value

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            elif content:
                # Raw content without parameter tags
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": content,
                    }
                )

        # Clean the text: drop the function bodies and any residual bare
        # <tool_call>/</tool_call> wrapper tags so they don't leak as content.
        if matches or rejected:
            # Share streaming's projection, including formatting-only gaps
            # between removed siblings and overlapping accepted payload spans.
            cleaned_text = self._visible_content_between(
                model_output, 0, len(model_output), request
            ).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        elif rejected:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text or None
            )
        else:
            # Diagnostic: a tool-call marker was present but nothing parsed
            # out — i.e. an as-yet-unhandled wire variant. Emit only a
            # STRUCTURAL SUMMARY of the shape, never the raw payload: this is
            # the normal degraded-wire path, and model_output can carry user
            # prompts, tool arguments, or credentials. The counts below are
            # enough to triage the unhandled variant without leaking content.
            has_tool_call_marker = "<tool_call>" in model_output
            function_tag_count = model_output.count("<function=")
            logger.warning(
                "nemotron tool parser: tool-call marker present but no tool "
                "call extracted (possible unhandled variant); "
                "<tool_call> present=%s, %d <function= tags, 0 parseable",
                has_tool_call_marker,
                function_tag_count,
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    @staticmethod
    def _close_tag_count(text: str) -> int:
        """Number of completed tool-call close tags present in ``text``.

        A call may legitimately close on either ``</function>`` (a truncated
        variant that never emits the wrapper) or ``</tool_call>``. Used by the
        streaming path to detect that a NEW call finished in the latest delta
        (count went up) rather than re-parsing on every chunk.
        """
        return text.count("</function>") + text.count("</tool_call>")

    @staticmethod
    def _clean_trailing_content(
        current_text: str, *, include_closers: bool
    ) -> str | None:
        """The plain-content tail of ``current_text``, or ``None`` if none.

        The caller supplies cached opener state on the no-new-close path.

        "Tail" = everything after the last COMPLETE tool-call close tag
        (``</function>`` / ``</tool_call>``). It is content-safe only if it
        contains no ``<`` at all — the moment a ``<`` appears we are (possibly)
        building the next call and must suppress, so no tag (complete or a
        partial fragment like ``"<fun"`` / ``"</fun"``) can ever leak into
        user-visible content.

        Returns:
          * ``None``  — still inside markup (no call closed yet, or a new
            ``<`` has started after the last close): suppress.
          * ``""``    — a call has closed and nothing (yet) follows it.
          * ``str``   — the safe trailing content after the last close.
        """
        end = 0
        for tag in ("</function>", "</tool_call>"):
            idx = current_text.rfind(tag)
            if idx != -1:
                end = max(end, idx + len(tag))
        if end == 0:
            # No close tag yet → we are still inside the (first) call's markup.
            return None
        tail = current_text[end:]
        if tail.isspace() and current_text.rfind("<tool_call>") > current_text.rfind(
            "</tool_call>"
        ):
            return None
        pending = NemotronToolParser._pending_markup_index(
            tail,
            include_closers=include_closers,
        )
        return None if pending is not None else tail

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
        Extract tool calls from streaming Nemotron model output.
        """
        if not self._stream_started or not previous_text:
            # The postprocessor's plain-text fast path can forward an opening
            # prefix without invoking this parser. On the first parser call of
            # a reset turn, that prefix is therefore already on the wire and
            # ``previous_text`` is non-empty. Start the watermark there rather
            # than replaying it. ``not previous_text`` also covers callers that
            # reuse an instance without honoring reset().
            self._content_upto = len(previous_text)
            self._opener_scan_upto = 0
            self._post_call_wrapped = False
            self._wrapped_whitespace = None
            self._stream_started = True
        if "<tool_call>" not in current_text and "<function=" not in current_text:
            # Ordinary prose, forwarded verbatim — already on the wire, so a
            # later refusal must not send it again.
            pending = self._pending_markup_index(current_text)
            safe_end = len(current_text) if pending is None else pending
            content = current_text[self._content_upto : safe_end]
            self._content_upto = safe_end
            return {"content": content} if content else None

        held = current_text[self._content_upto :]
        unmatched_opener = bool(self._content_upto) and self._has_unmatched_opener(
            current_text, self._content_upto
        )
        if (
            self._content_upto
            and "<function=" not in held
            and "<tool_call>" not in held
            and not unmatched_opener
        ):
            # Inspect only the held tail: wrapped continuation pairs stay off
            # the wire, while disambiguated lone closers remain literal prose.
            content = self._release_held_content(current_text)
            return {"content": content} if content else None

        # The XML parameter format has no escaping rule. A literal
        # ``</function>`` inside a value can appear before the structural
        # close; the outer wrapper is the first unambiguous boundary.
        function_open = current_text.rfind("<function=")
        wrapper_open = current_text.rfind("<tool_call>", 0, function_open)
        wrapper_before_function = current_text.rfind("</tool_call>", 0, function_open)
        if (
            wrapper_open > wrapper_before_function
            and current_text.rfind("</parameter>")
            > current_text.rfind("</function>")
            >= 0
        ):
            return None

        # Trigger from the COMPLETION STATE of current_text, NOT from a close
        # tag appearing inside a single delta_text. We fire only when a NEW
        # close tag finished in this delta — i.e. the close-tag count in
        # current_text ticked up versus previous_text.
        #
        # Counting the delta (rather than testing membership in delta_text) is
        # what makes a close tag split across chunks work: the tokenizer can
        # emit "</fun" then "ction>", so no single delta ever contains the
        # whole "</function>" — but the accumulated current_text does once both
        # fragments arrive, and only then does the count go up. Gating on the
        # *increase* also means we never re-parse current_text on the many
        # trailing deltas after a call has closed (avoiding O(n^2) re-parsing
        # and repeated fail-open WARNINGs on a trailing unparseable marker).
        #
        # extract_tool_calls re-parses current_text and returns ALL complete
        # calls; we de-dupe against the number already streamed (tracked in
        # current_tool_id, which reset() zeroes per request) so each completed
        # call is emitted exactly once even when </function> and </tool_call>
        # arrive in separate deltas (each bumps the count → one re-parse each,
        # but the second finds nothing new to emit).
        if self._close_tag_count(current_text) > self._close_tag_count(previous_text):
            # Account for completed calls immediately; only their tail may be held.
            # ``request`` matters here, not just to the non-streaming caller:
            # ``extract_tool_calls`` derives its declared-name gate from it, so
            # omitting it let a name the caller never offered through on the
            # streaming path while the same text was correctly refused when
            # buffered. Agents stream, so the gate was off where it counts.
            result = self.extract_tool_calls(current_text, request)
            safe_end = self._safe_accounting_boundary(current_text)
            if result.tools_called:
                spans = self._cached_function_spans(
                    current_text, valid_names=_declared_tool_names(request)
                )
                previous_end = 0
                wrapped = False
                for _, _, start, end in spans:
                    wrapped = self._is_wrapped(
                        current_text, start, previous_end, wrapped
                    )
                    previous_end = end
                self._post_call_wrapped = wrapped
                if self._post_call_wrapped:
                    wrapper = re.match(r"\s*</tool_call>", current_text[end:])
                    if wrapper:
                        end += wrapper.end()
                    consumed, pending_pair = self._wrapped_tail_boundary(
                        current_text[end:]
                    )
                    if pending_pair is not None:
                        safe_end = min(safe_end, end + consumed)
            unsent_content = self._visible_content_between(
                current_text, self._content_upto, safe_end, request
            )
            # The source-offset projection includes safe trailing assistant
            # text from this same delta while excluding executable spans.
            if result.tools_called:
                already_emitted = self.current_tool_id + 1
                total = len(result.tool_calls)
                if total > already_emitted:
                    new_calls = result.tool_calls[already_emitted:]
                    self.current_tool_id = total - 1
                    out: dict[str, Any] = {
                        # A function can finish before its held suffix resolves
                        # into narration. Keep that later prose on the wire.
                        "preserve_post_tool_content": True,
                        "tool_calls": [
                            {
                                "index": already_emitted + i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(new_calls)
                        ],
                    }
                    if unsent_content:
                        out["content"] = unsent_content
                    # Account only through the latest complete/safe boundary.
                    # A partial following opener still belongs to a future
                    # block and must remain releasable.
                    self._content_upto = safe_end
                    return out
            elif safe_end > self._content_upto:
                # Advance past a consumed attempt or a refused bare form.
                # The projection keeps narration and bare prose, excluding
                # wrapped undeclared markup even at the first function close.
                self._content_upto = safe_end
                if unsent_content:
                    return {"content": unsent_content}
            # Close tag but no NEW call to emit (e.g. the second of </function>
            # + </tool_call> for a call already streamed). Still surface any
            # trailing content that rode in on this delta.
            if safe_end > self._content_upto:
                # The projection is a view of the raw range after the watermark,
                # so a refusal released above cannot be sent twice.
                self._content_upto = safe_end
                if unsent_content:
                    return {"content": unsent_content}
            # Consumed markup is progress even without visible output: clear
            # the postprocessor suppression budget before narration resumes.
            return {"content": ""}

        # No new call closed in this delta. If we are past all tool-call markup
        # (a call has closed and no new "<" has started since), the delta is
        # trailing assistant content and must pass through instead of being
        # silently dropped. _clean_trailing_content being non-None guarantees no
        # partial or complete tag can leak, so we never emit "<function=",
        # "</function>", or a fragment like "</fun" as user-visible content. We
        # Emit only the source range after the watermark, so already-streamed
        # trailing content is not re-sent while previously withheld partial
        # opener bytes can be recovered when they become ordinary prose.
        if (
            self._clean_trailing_content(current_text, include_closers=unmatched_opener)
            is not None
        ):
            # Release the whole unaccounted visible range, not only this delta.
            # A prior suffix such as ``<fun`` may have been withheld while it
            # could still become ``<function=``; once a later byte turns it
            # into ordinary prose (``<funx``), those earlier bytes belong on
            # the wire too.
            # Everything after the watermark is already known to be outside
            # completed tool spans on this no-new-close path. Slice directly
            # instead of reparsing the full accumulated response for every
            # prose delta (which would make long streams quadratic).
            content = current_text[self._content_upto :]
            self._content_upto = len(current_text)
            return {"content": content} if content else None

        return None
