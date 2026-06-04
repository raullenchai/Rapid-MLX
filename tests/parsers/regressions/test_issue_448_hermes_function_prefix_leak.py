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

**Two bugs surfaced while writing this test, both colocated here
because they share the parser-path family:**

1. **Non-stream argument drop on bare ``<function=...{json}...>``.** The
   non-stream regex extracts ``name`` and ``params_block`` correctly,
   but the body parser runs ``PARAM_PATTERN`` (designed for Nemotron's
   ``<parameter=p>v</parameter>`` XML inner format) against the JSON
   body, producing zero param matches and silently emitting an empty
   arguments dict. The #448 reporter declared "non-stream is clean"
   because tool_calls fired with the right name and content was null —
   but the arguments were ``{}``. ``hermes_tool_parser.py:165-181``.

2. **Streaming prefix leak applies to BOTH wire formats, not just
   bare-function.** Round-2 codex strict review of this file's first
   draft surfaced this: classic ``<tool_call>{...}</tool_call>`` is
   ALSO vulnerable under char-level streaming — the parser emits the
   partial ``<tool_call`` opener (no closing ``>``) as content before
   the substring match fires. In production the bug is masked because
   tokenizers typically emit ``<tool_call>`` as a single chat-template
   special token; under char-level streaming (the test boundary fuzzer
   here) both formats leak. Same root cause as the bare-function bug;
   the cluster fix must add prefix-hold to BOTH wire formats.
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


BARE_FUNCTION_CASES: list[_Case] = [
    # #448 verbatim — Qwen3-Coder format. Leading `<function` must
    # never appear as content before the tool call emits.
    _Case(
        id="issue_448_qwen3coder_bare_function",
        raw='<function=read_file>{"path": "/tmp/example.py"}</function>',
        expected_name="read_file",
        expected_args={"path": "/tmp/example.py"},
    ),
    # Multi-arg bare-function shape — guards the JSON-body arg parser
    # alongside the prefix-hold fix. (Dict equality is unordered so
    # this does NOT pin field order; the cluster fix should also add
    # a separate order-preservation test if that's a requirement.)
    _Case(
        id="bare_function_multi_arg",
        raw=('<function=search>{"query": "rapid mlx", "limit": 10}</function>'),
        expected_name="search",
        expected_args={"query": "rapid mlx", "limit": 10},
    ),
]

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


@pytest.fixture
def parser() -> HermesToolParser:
    return HermesToolParser()


def _split_into_char_deltas(text: str, stream_interval: int) -> list[str]:
    """Char-level split + stream_interval batching.

    Same rationale as the #444 file — the parser doesn't need a
    tokenizer through this code path and char-level fuzzes byte-
    boundary leaks (any reasonable tokenization split is a subset of
    these splits). This is a boundary fuzzer, not an end-to-end
    tokenizer-equivalence proof.
    """
    per_char = list(text)
    return batch_deltas_with_stream_interval(per_char, stream_interval)


def _assert_content_clean(content: str | None, *, context: str) -> None:
    """Assert no chat content leaked into the streaming output.

    Stricter than a marker-substring check: these test cases contain
    *only* a tool call with no surrounding chat, so any non-empty
    ``content`` is a parser leak — including partial control-token
    prefixes like ``<func`` or ``<tool_cal`` that wouldn't match a
    full-sentinel substring scan (codex round-2 finding).
    """
    assert content in (None, ""), (
        f"Expected no chat content (test input is tool-call-only); got "
        f"content={content!r}. context={context!r}"
    )


# ----- Classic <tool_call> format ---------------------------------------


@pytest.mark.parametrize("case", CLASSIC_TOOL_CALL_CASES, ids=lambda c: c.id)
def test_hermes_classic_tool_call_non_stream(case: _Case, parser):
    """Pin: classic <tool_call> non-stream extraction is correct today.

    Regression sentinel — the cluster fix's prefix-hold helper must
    not regress the working non-stream path for the classic wire
    format.
    """
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    _assert_content_clean(content, context=f"case={case.id}")

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


@pytest.mark.parametrize("case", CLASSIC_TOOL_CALL_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_hermes_classic_tool_call_streaming(case: _Case, stream_interval: int, parser):
    # Flipped from xfail strict → passing by the cluster fix's
    # ``HermesToolParser._safe_content_prefix`` / ``_emit_safe_content``
    # prefix-hold helpers. Partial sentinel prefixes
    # (``<tool_call`` without ``>``, ``<function`` without ``=``) are
    # now held back until either the full opener arrives (tool-call
    # branch claims them) or a non-matching char releases them as
    # ordinary content.
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    _assert_content_clean(
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
def test_hermes_bare_function_non_stream(case: _Case, parser):
    # Flipped from xfail strict → passing by the cluster fix's
    # ``_parse_bare_function_body`` body-parser helper that
    # discriminates Qwen3-Coder JSON bodies (``{...}``) from Nemotron
    # XML bodies (``<parameter=p>v</parameter>``) on the first
    # non-whitespace char.
    content, tool_calls = run_tool_extraction(parser, [case.raw], streaming=False)

    _assert_content_clean(content, context=f"case={case.id}")

    assert len(tool_calls) == 1, (
        f"Expected exactly one tool call, got {len(tool_calls)}: {tool_calls!r}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args


# ----- Bare-function format: streaming prefix leak (BUG-1, #448) --------


@pytest.mark.parametrize("case", BARE_FUNCTION_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_hermes_bare_function_streaming(case: _Case, stream_interval: int, parser):
    # Flipped from xfail strict → passing by the combined cluster fix:
    # prefix-hold (BUG-1 `<function` partial leak) +
    # ``_parse_bare_function_body`` (BUG-2 JSON args drop on the same
    # wire format). Both fixes land together because the streaming
    # path delegates body extraction back to ``extract_tool_calls``.
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    _assert_content_clean(
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
