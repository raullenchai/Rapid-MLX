# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #444 + #480 — harmony streaming tool-call leak.

Two cluster issues with the same root cause (router + postprocessor
streaming-pipeline gap on harmony commentary channel) and the same
parser-level surface:

  * #444 (2026-05-22, gpt-oss-20b-MXFP4-Q8, ``tool_choice="auto"``):
    raw body ``...to=functions.get_weather...{"city": "Tokyo"}...``
    leaks into ``delta.content``.

  * #480 (2026-05-28, gpt-oss-20b-mxfp4-q8, ``tool_choice="auto"``): raw body
    ``commentary to=functions.get_weather json{"city":"Paris"}``
    leaks into ``delta.content``. User-facing duplicate of #444 with
    a different prompt — covered here to ensure the fix applies
    uniformly across prompts, not just the #444 repro.

Non-streaming on the same prompts works in both reports.

Per the issue, two bugs compound (router + postprocessor). The parser-
level streaming entry point is exercised here in isolation:

  * Non-stream: ``HarmonyToolParser.extract_tool_calls`` should pick up
    the commentary block and return ``tools_called=True`` with the
    parsed function name + arguments.

  * Streaming: ``HarmonyToolParser.extract_tool_calls_streaming`` is
    expected to emit a single ``{"tool_calls": [...]}`` delta when
    ``<|call|>`` arrives, and nothing before that (per the parser's
    own docstring lines 165-167).

Test cases sourced verbatim from the issue body's repro section.

Scope caveat (post-live-verification on gpt-oss-20b-mxfp4-q8 2026-06-04):
this file exercises the ``HarmonyToolParser`` streaming entry point
in isolation against the FULL markered text (`<|channel|>commentary
to=functions.X<|message|>{body}<|call|>`). The parser-level fix
(prefix-hold + count-based `<|call|>` detection) is necessary but
not sufficient for the production user-visible bug: live test
revealed the engine's ``OutputRouter`` consumes the `<|channel|>` /
`<|message|>` / `<|call|>` markers BEFORE the parser sees them
(commentary is also multi-token — `comment` + `ary` — defeating
single-token channel-type matching). The end-to-end fix requires a
router redesign (marker-preserving TOOL_CALL_TEXT state) tracked as
a separate followup. This file pins the parser contract so the
streaming-prefix-hold fix is verified independently and remains
correct once the router fix lands and feeds the parser full
markered text.

Convention: vLLM's ``test_*_failure_case_bug_NNNNN`` (e.g.
``tests/tool_parsers/test_hermes_tool_parser.py:78``). Each entry
states the wire format up front so failures are obvious in the
parametrize id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

from .._harmony_markers import assert_no_harmony_marker_leak
from ..dispatch import run_tool_extraction
from ..token_delta_splitter import batch_deltas_with_stream_interval


@dataclass
class _Case:
    """A single (id, raw_model_output, expected_name, expected_args) row.

    ``expected_args`` is the literal JSON object the parser should
    surface — match the model's actual emission, not the tool schema
    (the model in the repro emitted ``{"city": "Tokyo"}`` even though
    the schema declared ``location``; the parser must not "correct"
    that — its job is to extract what the model emitted).
    """

    id: str
    raw: str
    expected_name: str
    expected_args: dict


# Verbatim from the repro in issue #444 (and adjacent harmony commentary
# formats observed on gpt-oss-20b-mxfp4-q8). Each case is the FULL model output
# from the ``<|channel|>commentary`` token through the closing ``<|call|>``,
# i.e. what the model emits when invoking exactly one tool.
TEST_CASES: list[_Case] = [
    _Case(
        id="simple_single_arg",
        raw=(
            "<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city": "Tokyo"}<|call|>'
        ),
        expected_name="get_weather",
        expected_args={"city": "Tokyo"},
    ),
    _Case(
        id="multi_arg_object",
        raw=(
            "<|channel|>commentary to=functions.search "
            '<|constrain|>json<|message|>{"query": "rapid mlx", '
            '"limit": 10}<|call|>'
        ),
        expected_name="search",
        expected_args={"query": "rapid mlx", "limit": 10},
    ),
    _Case(
        id="empty_args_object",
        raw=(
            "<|channel|>commentary to=functions.list_models "
            "<|constrain|>json<|message|>{}<|call|>"
        ),
        expected_name="list_models",
        expected_args={},
    ),
    # #480 verbatim — same shape as the simple-arg #444 case but
    # different prompt and arg value, kept separately so a fix that
    # accidentally hardcodes the #444 prompt path can't pass.
    _Case(
        id="issue_480_paris",
        raw=(
            "<|channel|>commentary to=functions.get_weather "
            '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'
        ),
        expected_name="get_weather",
        expected_args={"city": "Paris"},
    ),
]


@pytest.fixture
def parser() -> HarmonyToolParser:
    return HarmonyToolParser()


def _split_into_char_deltas(text: str, stream_interval: int) -> list[str]:
    """Char-level split + stream_interval batching.

    We split per-character rather than per-token because the harmony
    test parser doesn't carry a real tokenizer through this code path
    and the bug surface is byte-boundary-sensitive (markers straddle
    chunks). Char-level subsumes any reasonable tokenization for the
    purpose of triggering boundary leaks.
    """
    per_char = list(text)
    return batch_deltas_with_stream_interval(per_char, stream_interval)


# ----- Non-streaming -----------------------------------------------------


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.id)
def test_harmony_tool_extraction_non_stream(case: _Case, parser):
    """Non-stream path works today (issue #444 only affects streaming).
    Pinning it so a regression here is loud."""
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


# ----- Streaming (THE BUG) ----------------------------------------------


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("stream_interval", [1, 2, 3, 5, 8])
def test_harmony_tool_extraction_streaming(case: _Case, stream_interval: int, parser):
    # Flipped from xfail strict → passing by the cluster fix's harmony
    # parser changes: (a) prefix-hold via ``_safe_content_prefix`` for
    # partial ``<|...|>`` openers, and (b) tool-call completion is now
    # detected by comparing ``previous_text.count("<|call|>")`` vs
    # ``current_text.count("<|call|>")`` instead of checking
    # ``"<|call|>" in delta_text`` (which never matched under
    # char-level streaming where the sentinel spans 8 deltas).
    deltas = _split_into_char_deltas(case.raw, stream_interval)

    content, tool_calls = run_tool_extraction(parser, deltas, streaming=True)

    # Invariant: no harmony control marker leaks into the content delta.
    # Uses the full canonical marker list (#444 issue body called out only
    # 3 markers, but the parser strips 7; a fix that's strict on 3 but
    # leaky on the other 4 would silently pass without this).
    assert_no_harmony_marker_leak(
        content,
        context=f"case={case.id} stream_interval={stream_interval}",
    )

    # Stricter than marker-only: these test inputs are tool-call-only
    # (no surrounding chat), so ANY non-empty content is a leak —
    # including raw JSON body fragments (``{"city":``) that wouldn't
    # match the harmony-marker check above (codex round-5 NIT).
    assert content in (None, ""), (
        f"Expected no chat content (tool-call-only input); got "
        f"content={content!r}. stream_interval={stream_interval} "
        f"case={case.id}"
    )

    # Invariant: exactly one tool call, fully assembled.
    assert len(tool_calls) == 1, (
        f"Expected 1 tool call after stream reassembly, got {len(tool_calls)}: "
        f"{tool_calls!r}. stream_interval={stream_interval}"
    )
    tc = tool_calls[0]
    assert tc.name == case.expected_name
    assert json.loads(tc.arguments) == case.expected_args
