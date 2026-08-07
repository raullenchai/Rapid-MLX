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
from typing import Any


def payload_spans(text: str, opener: str, closer: str) -> list[tuple[int, int]]:
    """Half-open ranges covering ``opener…closer`` element bodies.

    Used to tell an element that CONTAINS marker-shaped text apart from a
    real sibling. A tool name inside a parameter value is prose the model
    quoted — a file it read, a page it fetched, a previous tool result —
    not a call it made, and emitting it as one turns any content an agent
    ingests into an execution channel.

    Deliberately conservative: the range ends at the FIRST closer, so a
    value that itself contains a literal closer under-covers rather than
    over-covers. Under-covering leaves the pre-existing behaviour; the
    reverse would silently drop calls the model really did make.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        a = text.find(opener, i)
        if a == -1:
            return spans
        b = text.find(closer, a + len(opener))
        if b == -1:
            return spans
        spans.append((a, b + len(closer)))
        i = b + len(closer)


__all__ = [
    "payload_spans",
    "segment_by_next_opener",
    "declared_tool_names",
    "split_marked_calls",
    "split_marked_parameters",
]


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


def _next_sibling(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
) -> int:
    """Index of the next opener that is a real SIBLING of ``openers[index]``.

    Two filters:

    * a ``closer`` must appear between the two openers — an element cannot
      begin before the previous one ends;
    * with ``valid_names`` given, the opener's name must be one the request
      declared.

    The name check resolves what syntax alone cannot. In
    ``<parameter=a>before </parameter> text <parameter=fake> after
    </parameter>`` the closer really is there before the second opener, so
    a position-only reading fabricates ``fake``. Nothing in these
    escaping-free formats distinguishes that from two real elements; only
    the schema does. Concretely, the case this exists for::

        <parameter=code>if x: print("</parameter><parameter=evil>")</parameter>

    An agent writing code or documentation ABOUT tool calling emits exactly
    that. With the filter, ``code`` keeps its whole value. Without it the
    value is truncated at ``print("`` and a phantom ``evil`` is handed to
    the tool as though the model had asked for it.

    Repeated names are NOT filtered. Two real ``<parameter=body>`` elements
    are the caller's to resolve — dict assignment makes it last-value-wins,
    which is the established behaviour — and suppressing the second opener
    instead merges its wire markup into the first value, which corrupts
    both.

    Returns ``len(openers)`` when no sibling follows.
    """
    for j in range(index + 1, len(openers)):
        if valid_names is not None and openers[j].group(1).strip() not in valid_names:
            continue  # not a declared name — literal text inside the value
        if closer in text[openers[index].end() : openers[j].start()]:
            return j
    return len(openers)


def segment_by_next_opener(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
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
    sibling = _next_sibling(text, openers, index, closer, valid_names)
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
    not_inside: list[tuple[int, int]] | None = None,
) -> list[tuple[str, str, int, int]]:
    """``(name, body, span_start, span_end)`` for each call in ``text``.

    ``opener`` must capture the call name in group 1. ``outer``, when given,
    is an enclosing marker (e.g. ``</tool_call>`` around ``</function>``)
    that is absorbed into the span so callers can excise the whole
    invocation from surrounding prose in one step — a second, differently
    truncating regex pass over the same text is how tag fragments leak into
    user-visible content.

    ``not_inside`` lists ranges whose contents are payload — typically the
    parameter bodies from ``payload_spans``. An opener starting inside one
    is text the model quoted, not a call it made. Without this filter a
    tool name appearing in an argument value is emitted as an executable
    invocation, which turns any content an agent ingests (a file it reads,
    a page it fetches, a previous tool result) into an execution channel.

    ``valid_names`` cannot cover this: the dangerous names — run_shell,
    write_file — are exactly the ones the request DID declare, so a
    name-based filter passes them through.
    """
    openers = list(re.finditer(opener, text, re.DOTALL))
    if not_inside:
        openers = [
            m for m in openers if not any(a <= m.start() < b for a, b in not_inside)
        ]
    calls: list[tuple[str, str, int, int]] = []
    i = 0
    while i < len(openers):
        # NO name deduplication here. That rule belongs to parameters, where
        # a repeat within one call is payload because each name carries one
        # value. Calls are the opposite: invoking the same tool twice in a
        # turn is ordinary agent behaviour — read_file /a then read_file /b —
        # and suppressing the second opener merges both into one body and
        # silently drops a call the model asked for.
        # The gate applies to the opener being EMITTED, not only to the
        # search for the next one. Filtering siblings alone left the hole
        # wide open at index 0: a standalone ``<function=delete_everything>``
        # was still returned as a call, so the authorisation check could be
        # skipped entirely by putting the undeclared opener first.
        if valid_names is not None and openers[i].group(1).strip() not in valid_names:
            i += 1
            continue
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
    *,
    reject_undeclared_siblings: bool = False,
) -> list[tuple[str, str]] | None:
    """``(name, value)`` for each parameter in ``block``.

    ``opener`` must capture the parameter name in group 1. Values are
    stripped of surrounding whitespace: in these formats the newlines and
    indentation around a value are layout, not payload. Whitespace *inside*
    a value is preserved.
    """
    openers = list(re.finditer(opener, block))
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(openers):
        if reject_undeclared_siblings and valid_names is not None:
            current_end = openers[i].end()
            for candidate in openers[i + 1 :]:
                if candidate.group(1).strip() in valid_names:
                    continue
                if closer in block[current_end : candidate.start()]:
                    # Syntax cannot distinguish an undeclared sibling from
                    # marker-shaped payload after a close.  Continuing would
                    # splice the entire undeclared element into the previous
                    # value (#1541).  Refuse the call rather than execute a
                    # silently rewritten argument.
                    return None
        segmented = segment_by_next_opener(block, openers, i, closer, valid_names)
        sibling = _next_sibling(block, openers, i, closer, valid_names)
        if segmented is not None:
            out.append((openers[i].group(1).strip(), segmented[0].strip()))
        # Jump to the sibling, not i+1: the openers in between are literal
        # text inside the value just consumed. Advancing one at a time is
        # what turns a value containing "<parameter=x>" into a phantom
        # parameter named x, passed to the tool as if the model sent it.
        i = sibling if sibling > i else i + 1
    return out
