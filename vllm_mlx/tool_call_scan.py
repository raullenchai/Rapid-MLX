# SPDX-License-Identifier: Apache-2.0
"""Literal-safe scanning for marker-delimited tool-call wire formats.

XML-ish tool-call formats — Nemotron's ``<function=…><parameter=…>``,
Qwen 3.6's ``<tool_call>`` body, MiniCPM's ``<param name=…>`` — carry no
escaping layer. A value is whatever sits between two markers, which makes
the obvious implementation wrong in the same way every time::

    re.findall(r"<parameter=(\\w+)>(.*?)</parameter>", text, re.DOTALL)

The non-greedy ``(.*?)`` stops at the FIRST ``</parameter>``. When that
marker also occurs *inside* a value — an agent writing documentation about
tool calling, a diff of a parser, a shell heredoc — the match ends early,
the remaining JSON is truncated, and the call is dropped with no error and
no log line. That is jundot/omlx#2507; the silence is omlx#2420.

The resolution these formats actually use is positional: an element ends at
the LAST closing marker before the next sibling opens. This module is that
rule, written once.

It lives at package top level with no imports beyond ``re`` on purpose.
``vllm_mlx/api/tool_calling.py`` and the parsers under
``vllm_mlx/tool_parsers/`` both need it, and those two packages already
import each other (``api/__init__`` pulls in ``tool_calling``;
``tool_parsers/__init__`` eagerly loads ``qwen3coder`` which imports back
into ``api``). Anything shared between them has to sit outside both, or
the cycle closes.

Before this existed the rule was reimplemented per format, which is how
the same defect could be fixed in ``tool_calling.py`` and still be live in
``nemotron_tool_parser.py``.
"""

from __future__ import annotations

import re

__all__ = [
    "segment_by_next_opener",
    "split_marked_calls",
    "split_marked_parameters",
]


def _next_sibling(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
    used_names: set[str] | None = None,
) -> int:
    """Index of the next opener that is a real SIBLING of ``openers[index]``.

    Three filters, because each alone leaves a hole:

    * a ``closer`` must appear between the two openers — an element cannot
      begin before the previous one ends;
    * with ``valid_names`` given, the opener's name must be one the request
      actually declared.

    The name check is what resolves the genuinely ambiguous ordering
    ``<parameter=a>before </parameter> text <parameter=fake> after
    </parameter>``. Syntax alone cannot tell that apart from two real
    parameters — the closer *is* there before the second opener — so
    position-only rules fabricate a ``fake`` argument. Nothing in the wire
    format disambiguates it; only the schema does.

    ``used_names`` closes the last gap: a literal opener whose name IS
    declared — ``<parameter=body>before </parameter> <parameter=body>
    after</parameter>`` — passes the schema filter, and treating it as a
    sibling both truncates the real value and emits a second ``body`` that
    overwrites the first. These formats carry one value per name, so a
    repeat is payload by construction.

    Getting this wrong is worse than truncating: a phantom or overwritten
    argument is handed to the tool as though the model had asked for it.

    Returns ``len(openers)`` when no sibling follows.
    """
    for j in range(index + 1, len(openers)):
        name = openers[j].group(1).strip()
        if valid_names is not None and name not in valid_names:
            continue  # not a declared name — literal text inside the value
        if used_names is not None and name in used_names:
            continue  # already emitted — a repeat is payload, not an element
        if closer in text[openers[index].end() : openers[j].start()]:
            return j
    return len(openers)


def segment_by_next_opener(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
    used_names: set[str] | None = None,
) -> tuple[str, int] | None:
    """``(body, end_offset)`` for ``openers[index]``, or ``None`` if unclosed.

    Bounded by the next *sibling* opener, then trimmed at the LAST ``closer``
    in that span. The two rules cover the two ways these escaping-free
    formats can be misread:

    * ending at the first ``closer`` truncates a value that legitimately
      contains one (jundot/omlx#2507);
    * ending at the next *textual* opener truncates a value that contains
      one and fabricates an element.

    ``None`` for an element with no ``closer`` at all: a truncated or
    malformed emission is rejected rather than silently swallowing the rest
    of the buffer as its value. The regexes this replaced also required the
    closing marker to match, so accepting them would be a behaviour change,
    not a fix.
    """
    sibling = _next_sibling(text, openers, index, closer, valid_names, used_names)
    end = openers[sibling].start() if sibling < len(openers) else len(text)
    body = text[openers[index].end() : end]
    cut = body.rfind(closer)
    if cut == -1:
        return None
    return body[:cut], openers[index].end() + cut + len(closer)


def split_marked_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None = None,
    valid_names: frozenset[str] | set[str] | None = None,
) -> list[tuple[str, str, int, int]]:
    """``(name, body, span_start, span_end)`` for each call in ``text``.

    ``opener`` must capture the call name in group 1. ``outer``, when given,
    is an enclosing marker (e.g. ``</tool_call>`` around ``</function>``)
    that is absorbed into the span so callers can excise the whole
    invocation from surrounding prose in one step — a second, differently
    truncating regex pass over the same text is how tag fragments leak into
    user-visible content.
    """
    openers = list(re.finditer(opener, text, re.DOTALL))
    calls: list[tuple[str, str, int, int]] = []
    i = 0
    while i < len(openers):
        # NO name deduplication here. That rule belongs to parameters, where
        # a repeat within one call is payload because each name carries one
        # value. Calls are the opposite: invoking the same tool twice in a
        # turn is ordinary agent behaviour — read_file /a then read_file /b —
        # and suppressing the second opener merges both into one body and
        # silently drops a call the model asked for.
        segmented = segment_by_next_opener(text, openers, i, closer, valid_names)
        sibling = _next_sibling(text, openers, i, closer, valid_names)
        if segmented is not None:
            body, span_end = segmented
            emit = True
            if outer:
                limit = (
                    openers[sibling].start() if sibling < len(openers) else len(text)
                )
                tail = re.match(r"\s*" + re.escape(outer), text[span_end:limit])
                if tail:
                    span_end += tail.end()
                else:
                    # ``outer`` is REQUIRED when requested, not opportunistic.
                    # The regex this replaced matched only with the enclosing
                    # marker present, so accepting a call whose wrapper never
                    # arrived would newly admit truncated emissions. Callers
                    # that tolerate a missing wrapper (nemotron_tool_parser
                    # documents that case) simply do not pass ``outer``.
                    emit = False
            if emit:
                calls.append(
                    (openers[i].group(1).strip(), body, openers[i].start(), span_end)
                )
        # See split_marked_parameters: skip openers swallowed by this value.
        i = sibling if sibling > i else i + 1
    return calls


def split_marked_parameters(
    block: str,
    opener: str,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
) -> list[tuple[str, str]]:
    """``(name, value)`` for each parameter in ``block``.

    ``opener`` must capture the parameter name in group 1. Values are
    stripped of surrounding whitespace: in these formats the newlines and
    indentation around a value are layout, not payload. Whitespace *inside*
    a value is preserved.
    """
    openers = list(re.finditer(opener, block))
    out: list[tuple[str, str]] = []
    used: set[str] = set()
    i = 0
    while i < len(openers):
        seen = used | {openers[i].group(1).strip()}
        segmented = segment_by_next_opener(block, openers, i, closer, valid_names, seen)
        sibling = _next_sibling(block, openers, i, closer, valid_names, seen)
        if segmented is not None:
            name = openers[i].group(1).strip()
            used.add(name)
            out.append((name, segmented[0].strip()))
        # Jump to the sibling, not i+1: the openers in between are literal
        # text inside the value just consumed. Advancing one at a time is
        # what turns a value containing "<parameter=x>" into a phantom
        # parameter named x, passed to the tool as if the model sent it.
        i = sibling if sibling > i else i + 1
    return out
