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
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
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

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

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

        matches = self.TOOL_CALL_PATTERN.findall(model_output)
        for func_name, content in matches:
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
            params = self.PARAM_PATTERN.findall(content)
            if params:
                arguments = {}
                for param_name, param_value in params:
                    # Try to parse value as JSON (for nested objects)
                    try:
                        arguments[param_name.strip()] = json.loads(param_value.strip())
                    except json.JSONDecodeError:
                        arguments[param_name.strip()] = param_value.strip()

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
        if matches:
            cleaned_text = self.TOOL_CALL_PATTERN.sub("", cleaned_text)
            cleaned_text = self.RESIDUAL_WRAPPER_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            # Diagnostic: a genuine ``<tool_call>`` WRAPPER was present but
            # nothing parsed out — i.e. an as-yet-unhandled degraded-wire
            # variant worth capturing. We deliberately do NOT warn on a bare
            # unmatched ``<function=`` with no wrapper: that is far more likely
            # ordinary assistant prose mentioning an XML-ish example than a real
            # tool call, so warning on it would be noise. Emit only a STRUCTURAL
            # SUMMARY, never the raw payload — this path can carry user prompts,
            # tool arguments, or credentials.
            if "<tool_call>" in model_output:
                function_tag_count = model_output.count("<function=")
                logger.warning(
                    "nemotron tool parser: <tool_call> marker present but no "
                    "tool call extracted (possible unhandled variant); "
                    "%d <function= tags, 0 parseable",
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

    # Markup tokens the content scanner recognises. ``<function=`` opens a tool
    # body (suppress until ``</function>``); the wrapper/close tags are stripped
    # from the content stream. A ``<`` that matches none of these — even
    # partially — is ordinary prose (e.g. ``"2 < 3"``) and streams through.
    _CONTENT_OPEN_CALL = "<function="
    _CONTENT_STRIP_TAGS = ("<tool_call>", "</tool_call>", "</function>")
    _CONTENT_HOLD_TOKENS = ("<function=", "<tool_call>", "</tool_call>", "</function>")
    _CONTENT_CLOSE_CALL = "</function>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Incremental content-channel scanner state (see _advance_content).
        self._c_cursor = 0
        self._c_inside_call = False
        # Chars already emitted on the streaming content channel this request
        # (used by flush_held_content to release a held-but-never-closed tail).
        self._content_emitted_len = 0
        # Count of tool calls ALREADY emitted on the streaming tool_calls
        # channel this request. Updated only from emitted calls (never from the
        # raw close-tag count), so malformed close tags can't shift indices.
        self._stream_calls_emitted = 0

    def reset(self) -> None:
        super().reset()
        self._c_cursor = 0
        self._c_inside_call = False
        self._content_emitted_len = 0
        self._stream_calls_emitted = 0

    def _advance_content(self, current_text: str) -> str:
        """Return assistant-content chars newly emittable since the last call.

        A single forward scan over the accumulated ``current_text`` using a
        persistent cursor (so the whole response is scanned O(n) total, not
        O(n^2) per token). Rules:

          * ``<function=..>..</function>`` bodies are suppressed (they are the
            ``tool_calls`` channel, never content).
          * bare ``<tool_call>`` / ``</tool_call>`` wrapper tags and any stray
            ``</function>`` are stripped.
          * a trailing ``<`` that is a partial prefix of one of those tokens
            (``"<fun"``, ``"</too"``, a lone ``"<"``) is HELD until the next
            delta resolves it, so no partial markup ever leaks as content.
          * every other ``<`` (``"2 < 3"``) is ordinary prose and streams.

        The caller (:meth:`extract_tool_calls_streaming`) restarts the scan
        whenever the cursor is ahead of the stream, so ``current_text`` is
        always an extension of what was already consumed here.
        """
        n = len(current_text)
        out: list[str] = []
        i = self._c_cursor
        hold = max(len(t) for t in self._CONTENT_HOLD_TOKENS) - 1
        while i < n:
            if self._c_inside_call:
                close = current_text.find(self._CONTENT_CLOSE_CALL, i)
                if close == -1:
                    # Not closed yet; suppress everything but keep a small tail
                    # unconsumed in case the close tag is split across deltas.
                    i = max(i, n - hold)
                    break
                i = close + len(self._CONTENT_CLOSE_CALL)
                self._c_inside_call = False
                continue

            lt = current_text.find("<", i)
            if lt == -1:
                out.append(current_text[i:n])
                i = n
                break
            if lt > i:
                out.append(current_text[i:lt])
            rest = current_text[lt:]
            if rest.startswith(self._CONTENT_OPEN_CALL):
                self._c_inside_call = True
                i = lt + len(self._CONTENT_OPEN_CALL)
                continue
            stripped = next(
                (t for t in self._CONTENT_STRIP_TAGS if rest.startswith(t)), None
            )
            if stripped is not None:
                i = lt + len(stripped)
                continue
            if any(tok.startswith(rest) for tok in self._CONTENT_HOLD_TOKENS):
                # Ambiguous partial tag at the tail → hold, decide next delta.
                i = lt
                break
            # A "<" that cannot be any tool tag → it is content.
            out.append("<")
            i = lt + 1

        self._c_cursor = i
        return "".join(out)

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
        # Guard the incremental content scanner against a parser reused for a
        # new/independent stream without reset(): within one stream the cursor
        # is always <= len(previous_text) (it was set while scanning that very
        # text). If it is ahead, previous_text belongs to a different (shorter)
        # stream — including a fresh stream where previous_text == "" — so
        # restart the scan to avoid skipping the new stream's leading content.
        if self._c_cursor > len(previous_text):
            self._c_cursor = 0
            self._c_inside_call = False

        # --- tool_calls channel ------------------------------------------
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
        # extract_tool_calls re-parses current_text and returns ALL complete,
        # PARSEABLE calls; we de-dupe against how many we have ALREADY emitted
        # (self._stream_calls_emitted, advanced only by emitted calls — never by
        # the raw close-tag count) so each completed call is emitted exactly
        # once, indices stay contiguous, and a malformed earlier close tag
        # (close-count bump with no parseable call) can't shift later indices.
        tool_calls_payload: list[dict[str, Any]] | None = None
        if self._close_tag_count(current_text) > self._close_tag_count(previous_text):
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                already_emitted = self._stream_calls_emitted
                total = len(result.tool_calls)
                if total > already_emitted:
                    new_calls = result.tool_calls[already_emitted:]
                    self._stream_calls_emitted += len(new_calls)
                    tool_calls_payload = [
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
                    ]

        # --- content channel ---------------------------------------------
        # Assistant prose outside tool markup — before, between, and after
        # calls — must stream through instead of being dropped. _advance_content
        # scans incrementally (O(n) total), holds in-flight openers so no
        # partial markup ("<fun") leaks, keeps genuine "<" prose ("2 < 3"), and
        # never re-emits. Trailing prose in the SAME delta as a close tag rides
        # out alongside tool_calls via the combined return the postprocessor
        # already supports.
        new_content = self._advance_content(current_text)
        self._content_emitted_len += len(new_content)

        out: dict[str, Any] = {}
        if tool_calls_payload is not None:
            out["tool_calls"] = tool_calls_payload
        if new_content:
            out["content"] = new_content
        return out or None

    def flush_held_content(self, full_text: str) -> str:
        """Release content the streaming scanner suppressed but that was, in
        hindsight, ordinary prose.

        The postprocessor calls this in ``finalize()`` only when NO tool call
        fired. In that case an unclosed ``<function=..`` the scanner optimist-
        ically entered suppression on never became a real call, so — matching
        the non-streaming fail-open in ``extract_tool_calls`` — its text is
        plain content. We return the fail-open content that was not yet emitted;
        an empty string when a real call did parse (nothing to release) or when
        everything was already streamed.
        """
        info = self.extract_tool_calls(full_text)
        if info.tools_called:
            return ""
        full_content = info.content or ""
        if self._content_emitted_len >= len(full_content):
            return ""  # everything already streamed
        return full_content[self._content_emitted_len :]
