# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #448 — hermes streaming leaks `<function` prefix.

Reported 2026-05-23 on ``mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit``
(aliased via ``qwen3-coder-30b`` to the hermes tool parser). Qwen3-Coder
emits the legacy ``<function=name>{...}</function>`` wire format. The
hermes non-stream regex handles both ``<tool_call>...`` and bare
``<function=...`` blocks, but the streaming branch only short-circuits
on ``<function=`` *after* the substring has fully arrived
(``hermes_tool_parser.py:320``). The leading ``<`` and ``function``
tokens fall through to the final ``return {"content": delta_text}`` at
line 350 before the pattern matches — leaking as content deltas before
the tool call fires.

Same family-wide pattern as #444 / #455 / #480: every parser's stream
path is missing the prefix-hold logic its non-stream path implies. The
SGLang-style ``prefix_hold(text, prefixes)`` primitive is the canonical
fix (see ``docs/parser_testing_patterns`` for the upstream pattern).

Scope caveat (per round-1 codex review of the #444 template): this
file exercises the ``HermesToolParser`` streaming entry in isolation,
not the end-to-end route. The bug is parser-internal so the parser-only
test is sufficient — unlike #444 / #447 where router or postprocessor
fixes also contribute.

**Additional bug surfaced while writing this test (NOT in the original
issue body):** the hermes *non-stream* path also drops arguments on the
``<function=name>{json}</function>`` wire format. The non-stream regex
extracts ``name`` correctly but then runs ``PARAM_PATTERN`` (designed
for Nemotron's ``<parameter=p>v</parameter>`` XML inner format) against
the JSON body, producing zero param matches and silently emitting an
empty arguments dict. The #448 reporter declared "non-stream is clean"
because tool_calls fired with the right name and content was null — but
the arguments were ``{}``. Captured here as additional xfails; the
cluster fix should cover both non-stream and streaming for this wire
format. Calls this a "drop", not a "leak", but it lives in the same
parser-path family (``hermes_tool_parser.py:165-181``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

from ..dispatch import run_tool_extraction
from ..token_delta_splitter import batch_deltas_with_stream_interval


@dataclass
class _Case:
    id: str
    raw: str
    expected_name: str
    expected_args: dict


# Bare-function (Qwen3-Coder) wire format — broken in BOTH non-stream
# and streaming. The cluster fix must repair both branches.
BARE_FUNCTION_CASES: list[_Case] = [
    # #448 verbatim — Qwen3-Coder format. Leading `<function` must
    # never appear as content before the tool call emits.
    _Case(
        id="issue_448_qwen3coder_bare_function",
        raw='<function=read_file>{"path": "/tmp/example.py"}</function>',
        expected_name="read_file",
        expected_args={"path": "/tmp/example.py"},
    ),
    # Multi-arg bare-function shape — pins arg-order preservation
    # alongside the prefix-hold + JSON-args fix.
    _Case(
        id="bare_function_multi_arg",
        raw=('<function=search>{"query": "rapid mlx", "limit": 10}</function>'),
        expected_name="search",
        expected_args={"query": "rapid mlx", "limit": 10},
    ),
]

# Classic <tool_call>{json}</tool_call> wire format — works today on
# both non-stream and streaming paths. Sanity-pinned so the cluster
# fix doesn't accidentally break the working format while repairing
# the bare-function one.
CLASSIC_TOOL_CALL_CASES: list[_Case] = [
    _Case(
        id="classic_tool_call_tag",
        raw=(
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n'
            "</tool_call>"
        ),
        expected_name="get_weather",
        expected_args={"city": "Tokyo"},
    ),
]

# Leak markers specific to hermes wire formats. Subset rather than a
# canonical-list helper because the hermes parser doesn't expose a
# single source-of-truth control-token list the way harmony does (its
# pattern matching is regex-based).
HERMES_LEAK_MARKERS: tuple[str, ...] = (
    "<function=",
    "<tool_call>",
    "</tool_call>",
)


@pytest.fixture
def parser() -> HermesToolParser:
    return HermesToolParser()


def _split_into_char_deltas(text: str, stream_interval: int) -> list[str]:
    """Char-level split + stream_interval batching.

    Same rationale as the #444 file — the parser doesn't need a
    tokenizer through this code path and char-level subsumes any
    reasonable tokenization for triggering boundary leaks. The
    specific bug here surfaces *because* the leading ``<`` and
    ``function`` arrive in early per-token deltas before the full
    ``<function=`` pattern is visible.
    """
    per_char = list(text)
    return batch_deltas_with_stream_interval(per_char, stream_interval)


def _assert_no_hermes_marker_leak(content: str | None, *, context: str) -> None:
    """Assert no hermes wire-format marker leaked into content.

    Inline rather than imported because hermes lacks a single
    authoritative control-token list (it's regex-based); the three
    markers here cover every wire format the parser handles.
    """
    if content is None or content == "":
        return
    leaked = [m for m in HERMES_LEAK_MARKERS if m in content]
    assert not leaked, (
        f"Hermes wire-format marker(s) leaked into content delta: "
        f"{leaked!r}. context={context!r} content={content!r}"
    )


# ----- Classic <tool_call> format: pin working behavior -----------------


@pytest.mark.parametrize("case", CLASSIC_TOOL_CALL_CASES, ids=lambda c: c.id)
def test_hermes_classic_tool_call_non_stream(case: _Case, parser):
    """Pin: classic <tool_call> non-stream extraction is correct today."""
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    assert content in (None, ""), (
        f"Expected non-stream extraction to consume all input as a tool "
        f"call; got leftover content={content!r}"
    )
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


@pytest.mark.parametrize("case", CLASSIC_TOOL_CALL_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_hermes_classic_tool_call_streaming(case: _Case, stream_interval: int, parser):
    """Pin: classic <tool_call> streaming is correct today.

    Regression sentinel — the cluster fix adds a ``<function=`` prefix
    hold to the streaming branch. If that helper is overzealous it
    could also defer ``<tool_call>`` prefixes; this test catches that
    sibling regression.
    """
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    _assert_no_hermes_marker_leak(
        content,
        context=f"case={case.id} stream_interval={stream_interval}",
    )

    assert len(tool_calls) == 1, (
        f"Expected 1 tool call after stream reassembly, got {len(tool_calls)}: "
        f"{tool_calls!r}. stream_interval={stream_interval}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


# ----- Bare-function format: non-stream argument drop (BUG-2) -----------


@pytest.mark.parametrize("case", BARE_FUNCTION_CASES, ids=lambda c: c.id)
@pytest.mark.xfail(
    reason=(
        "Hermes non-stream path drops arguments on the bare "
        "`<function=name>{json}</function>` wire format. Root cause: the "
        "regex extracts ``name`` and ``params_block`` correctly, but the "
        "body parser runs ``PARAM_PATTERN`` (designed for Nemotron "
        "``<parameter=p>v</parameter>`` XML) against the JSON body and "
        "extracts zero params, silently emitting ``arguments={}``. "
        "Surfaced while writing the #448 streaming regression — the "
        "issue reporter declared non-stream 'clean' because tool_calls "
        "fired with the right name and content was null. Not in #448's "
        "issue body but lives in the same parser-path family; the cluster "
        "fix repairs both. Flip to passing once the fix lands."
    ),
    strict=True,
)
def test_hermes_bare_function_non_stream(case: _Case, parser):
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    assert content in (None, ""), (
        f"Expected non-stream extraction to consume all input as a tool "
        f"call; got leftover content={content!r}"
    )
    assert len(tool_calls) == 1, (
        f"Expected exactly one tool call, got {len(tool_calls)}: {tool_calls!r}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


# ----- Bare-function format: streaming prefix leak (BUG-1, #448) --------


@pytest.mark.parametrize("case", BARE_FUNCTION_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
@pytest.mark.xfail(
    reason=(
        "Issue #448 — hermes streaming leaks `<` and `function` content "
        "deltas before the `<function=` pattern fully arrives. The non-"
        "stream regex catches the bare-function format but the streaming "
        "branch only short-circuits after the substring is complete. Fix: "
        "add SGLang-style prefix_hold for the partial `<function` prefix. "
        "Flip to passing once the cluster fix lands."
    ),
    strict=True,
)
def test_hermes_bare_function_streaming(case: _Case, stream_interval: int, parser):
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    _assert_no_hermes_marker_leak(
        content,
        context=f"case={case.id} stream_interval={stream_interval}",
    )

    assert len(tool_calls) == 1, (
        f"Expected 1 tool call after stream reassembly, got {len(tool_calls)}: "
        f"{tool_calls!r}. stream_interval={stream_interval}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args
