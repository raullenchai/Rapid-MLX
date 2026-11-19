# SPDX-License-Identifier: Apache-2.0
"""Scanning for marker-delimited tool calls using upstream XML boundaries.

Parameters end at the first close, next parameter opener, function close, or
end of input. Calls end at the first function close or end of input. See
SGLang python/sglang/srt/function_call/qwen3_coder_detector.py and the former
vLLM Python qwen3coder parser. XML marker text has no escaping layer.

The undeclared-parameter sibling rejection (#1541) is retained separately
from segmentation. This module lives outside api/tool_parsers to avoid their
import cycle.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "segment_by_next_opener",
    "declared_tool_names",
    "split_marked_calls",
    "split_marked_parameters",
    "trim_wrapping_newlines",
]


def trim_wrapping_newlines(value: str) -> str:
    r"""Remove the ONE newline the wire puts on each side of a value.

    The Qwen3-Coder chat template renders a parameter as
    ``<parameter=NAME>\n`` + value + ``\n</parameter>\n``, so exactly one
    newline per side is markup and everything else -- including the
    indentation of the FIRST line -- is payload. Stripping all surrounding
    whitespace instead silently de-indented the opening line of every value,
    which for a coding agent's ``new_string`` means the edit no longer
    compiles or, worse, changes scope while still parsing (#3401).

    This matches the two mature engines that serve this wire format: vLLM's
    ``_trim_wrapping_newlines`` (``vllm/parser/qwen3.py``) and SGLang's
    ``qwen3_coder_detector`` both drop one leading and one trailing ``\n``
    and nothing else, in their streaming and non-streaming paths alike.
    ``\r\n`` is deliberately NOT treated as a two-byte wrapper. The template
    frames with ``\n``, so under LF framing a payload that ends in ``\r``
    arrives as ``"\ntext\r\n"`` -- indistinguishable from a CRLF-framed
    ``"text"``. Consuming both bytes would delete a payload byte to tidy up
    markup that this wire does not actually emit. Removing one ``\n`` leaves
    ``"text\r"``, which is correct under LF framing and merely leaves a
    visible stray ``\r`` under the CRLF framing nothing here produces. When
    the two readings cannot be told apart, keep the byte.

    Same lesson as ``_decode_json_like`` in ``api/tool_calling.py``: on this
    wire, whitespace around a value is only safe to remove where the format
    says that whitespace is markup.
    """
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def declared_tool_names(request: dict[str, Any] | None) -> frozenset[str] | None:
    """Tool names the request actually declared, or ``None`` if it declared none.

    ``None`` rather than an empty set, because the two mean different things
    to the scanners: an empty set would reject every opener, while ``None``
    selects the position-only rules. A request with no tools cannot execute
    anything anyway, so there is nothing to protect there and no reason to
    change how its text is parsed.
    """
    if not isinstance(request, dict):
        return None
    tools = request.get("tools")
    if not isinstance(tools, list):
        return None
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return frozenset(names) or None


def segment_by_next_opener(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
) -> tuple[str, int]:
    """Return a call body through the first closer, or the remaining input.

    The upstream function regex consumes nested-looking openers as body text.
    Declared names do not change wire boundaries.
    """
    start = openers[index].end()
    cut = text.find(closer, start)
    if cut < 0:
        return text[start:], len(text)
    return text[start:cut], cut + len(closer)


def split_marked_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None = None,
    valid_names: frozenset[str] | set[str] | None = None,
) -> list[tuple[str, str, int, int]]:
    """Return name, body, start and end using the upstream first-close rule.

    An immediately following outer wrapper close is absorbed into the span.
    Missing closes are tolerated, as in the upstream function regex.
    """
    if "<function=" in opener:
        calls = []
        previous_end = 0
        for match in re.finditer(
            r"<function=(.*?)</function>|<function=(.*)$", text, re.DOTALL
        ):
            function = match.group(1) if match.group(1) is not None else match.group(2)
            name, separator, body = function.partition(">")
            if not separator:
                continue
            start, end = match.span()
            if opener.startswith("<tool_call>"):
                wrapper = re.search(r"<tool_call>\s*$", text[previous_end:start])
                if wrapper is None:
                    continue
                start = previous_end + wrapper.start()
            if outer:
                tail = re.match(r"\s*" + re.escape(outer), text[end:])
                if tail:
                    end += tail.end()
            calls.append((name, body, start, end))
            previous_end = end
        return calls

    openers = list(re.finditer(opener, text, re.DOTALL))
    calls = []
    consumed = 0
    for i, match in enumerate(openers):
        if match.start() < consumed:
            continue
        body, end = segment_by_next_opener(text, openers, i, closer)
        calls.append((match.group(1).strip(), body, match.start(), end))
        consumed = end
    return calls


def split_marked_parameters(
    block: str,
    opener: str,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
    *,
    reject_undeclared_siblings: bool = False,
) -> list[tuple[str, str]] | None:
    """Split parameters at the earliest upstream boundary, trimming one LF.

    Keep #1541: an undeclared sibling after a parameter close rejects the
    entire call instead of splicing its markup into the preceding value.
    Schema membership never changes the segmentation itself.
    """
    if opener.startswith("<parameter="):
        # Match the complete upstream expression rather than requiring a
        # well-formed next opener: even a partial <parameter= ends the value.
        matches = list(
            re.finditer(
                r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
                block,
                re.DOTALL,
            )
        )
        parsed: list[tuple[str, str]] = []
        first_close = block.find(closer)
        for match in matches:
            name, separator, value = match.group(1).partition(">")
            if not separator:
                continue
            name = name.strip()
            if (
                reject_undeclared_siblings
                and valid_names is not None
                and parsed
                and name not in valid_names
                and first_close >= 0
                and match.start() >= first_close + len(closer)
            ):
                return None
            parsed.append((name, trim_wrapping_newlines(value)))
        return parsed

    openers = list(re.finditer(opener, block, re.DOTALL))
    out = []
    for index, match in enumerate(openers):
        start = match.end()
        boundaries = [block.find(closer, start)]
        if index + 1 < len(openers):
            boundaries.append(openers[index + 1].start())
        end = min([pos for pos in boundaries if pos >= 0], default=len(block))
        out.append((match.group(1).strip(), trim_wrapping_newlines(block[start:end])))
    return out
