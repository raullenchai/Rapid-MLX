# SPDX-License-Identifier: Apache-2.0
"""Offline Hy3 (Hunyuan 3) parser-level integration test — CI-runnable.

The always-on-CI value-add for the Hy3 Tier-1 family. Every *live* Hy3
matrix cell in ``test_agents_matrix.py`` / ``test_frameworks_matrix.py``
is strict-xfail'd because the only shipped SKU (``hy3-preview-4bit``) is
166 GB / ~156 GB peak and single-node-infeasible in per-PR CI — real
inference for Hy3 lives in the weekly Golden Path job (see
``conftest.py`` ``_HY3_XFAIL_REASON``).

This file fills the CI gap. It exercises the Hy3 parsing path
END-TO-END at the OpenAI-API-shape level **without booting the 166 GB
model**: captured Hy3 wire strings (the exact
``<tool_call:opensource>…<end_of_tool_call:opensource>`` and
``<think:opensource>…</think:opensource>`` shapes the 4-bit checkpoint
emits, harvested in the 2026-07-09 ``pipenetwork/Hy3-REAP50/75-MLX-4bit``
spike that seeded the PR-2 parser unit tests) are fed through the two
parsers the ``hy3-preview-4bit`` alias wires
(``tool_call_parser="hy_v3"`` + ``reasoning_parser="hy_v3"``) and the
resulting API-shape objects are asserted:

* tool_calls array is well-formed (OpenAI ``tool_call`` dict shape,
  JSON-parseable ``arguments``) — reuses the shared
  ``assert_tool_call_shape`` helper the live matrix cells use;
* ``<think>`` reasoning content is routed to the reasoning channel, NOT
  leaked into visible content — reuses ``assert_no_think_tag_leak``;
* no analysis/think-tag markers leak into either channel.

Unlike the PR-2 *unit* tests (``tests/test_hy_v3_tool_parser.py`` /
``tests/test_hy3_reasoning_parser.py``), which assert parser internals in
isolation, THIS file asserts the composed *API-shape contract* a real
agent/framework client would observe on the wire — the same contract the
live matrix cells assert against a booted server, minus the boot. PR-2
(#1070) is now merged; these fixtures were re-verified green against the
merged parser at rebase time (5 codex rounds of literal-close / reasoning
partial-close / brace-depth fixes landed after this file was first
written, and the captured wire still round-trips byte-exact — no
assertion change was needed).

Runs in the normal ``pytest tests/`` sweep — no server, no model, no
Docker. Pure-Python, sub-second.
"""

from __future__ import annotations

import json

from tests.integrations.conftest import (
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    assert_tool_call_shape,
)

# --------------------------------------------------------------------------- #
# Captured Hy3 wire fixtures
# --------------------------------------------------------------------------- #
#
# These are the canonical shapes emitted by the 4-bit ``hy3-preview-4bit``
# checkpoint, harvested from the ``pipenetwork/Hy3-REAP50-MLX-4bit`` +
# ``Hy3-REAP75-MLX-4bit`` BFCL spike (2026-07-09). They match the fixtures
# that seed the PR-2 unit suites, so if PR-2 changes the parser contract
# these strings are the single place to re-sync at rebase time.

# Canonical single tool call — the chat-template default emission.
_WIRE_SINGLE_TOOL_CALL = (
    "<tool_call:opensource>get_weather"
    '<tool_sep:opensource>{"city": "Tokyo"}'
    "<end_of_tool_call:opensource>"
)

# Two tool calls in one assistant turn (parallel tool-calling wire).
_WIRE_MULTI_TOOL_CALL = (
    "<tool_call:opensource>get_weather"
    '<tool_sep:opensource>{"city": "Tokyo"}'
    "<end_of_tool_call:opensource>"
    "<tool_call:opensource>get_time"
    '<tool_sep:opensource>{"tz": "Asia/Tokyo"}'
    "<end_of_tool_call:opensource>"
)

# 4-bit numerical-noise malformed close — the model skips the JSON body and
# jumps straight to ``</arg_value>``. The non-streaming path must still
# surface the tool name with empty args rather than dropping the call.
_WIRE_MALFORMED_CLOSE = "<tool_call:opensource>get_weather</arg_value:opensource>"

# Reasoning span then answer — the canonical Hy3 think emission.
_WIRE_REASONING_THEN_ANSWER = (
    "<think:opensource>The user wants the weather in Tokyo. "
    "I should call the get_weather tool.</think:opensource>"
    "The weather in Tokyo is sunny."
)

# Reasoning span with no visible answer after it (all reasoning).
_WIRE_REASONING_ONLY = (
    "<think:opensource>Let me work through this step by step.</think:opensource>"
)

# Plain content, no tool call, no reasoning — must pass straight through.
_WIRE_PLAIN_CONTENT = "The answer is 42."


_HY3_ALIAS = "hy3-preview-4bit"


def _hy3_alias_profile():
    """Resolve the production ``hy3-preview-4bit`` alias profile.

    Resolving through the real alias loader (not a hard-coded parser
    name) means this test also guards the **alias → parser wiring**: if
    the alias ever stops declaring ``tool_call_parser`` /
    ``reasoning_parser``, or points them at a different parser, the
    assertions below fail instead of silently passing on a stale literal.
    """
    from vllm_mlx.model_aliases import resolve_profile

    profile = resolve_profile(_HY3_ALIAS)
    assert profile is not None, f"{_HY3_ALIAS!r} alias not found in aliases.json"
    return profile


def _tool_parser():
    """Construct the Hy3 tool parser the ``hy3-preview-4bit`` alias wires.

    The parser name is read FROM the alias config (``tool_call_parser``),
    not hard-coded, so a change that unwires the alias from ``hy_v3`` is
    caught here. The import + registry lookup are HARD (no ``pytest.skip``):
    the ``hy_v3`` parser is merged (PR-2, #1070) and permanent, so an
    import-time regression or accidental deletion must FAIL this always-on
    test rather than skip it green — the exact failure mode this file
    exists to catch.
    """
    from vllm_mlx.tool_parsers import HyV3ToolParser, ToolParserManager

    parser_name = _hy3_alias_profile().tool_call_parser
    assert parser_name == "hy_v3", (
        f"{_HY3_ALIAS} tool_call_parser changed to {parser_name!r}; "
        "update this offline test to match the alias wiring"
    )
    # Resolve through the registry the way the server wires it from the
    # alias's ``tool_call_parser`` field.
    resolved = ToolParserManager.get_tool_parser(parser_name)
    assert resolved is HyV3ToolParser
    return resolved()


def _reasoning_parser():
    """Construct the Hy3 reasoning parser the alias wires (``reasoning_parser``).

    Same discipline as ``_tool_parser``: the parser name is read from the
    alias config and the import is HARD — a missing/broken import or an
    unwired alias must fail, not skip.
    """
    from vllm_mlx.reasoning import get_parser
    from vllm_mlx.reasoning.hy3_parser import Hy3ReasoningParser

    parser_name = _hy3_alias_profile().reasoning_parser
    assert parser_name == "hy_v3", (
        f"{_HY3_ALIAS} reasoning_parser changed to {parser_name!r}; "
        "update this offline test to match the alias wiring"
    )
    resolved = get_parser(parser_name)
    assert resolved is Hy3ReasoningParser
    return resolved()


def _tool_calls_to_openai_shape(extracted) -> list[dict]:
    """Project the parser's ``ExtractedToolCallInformation`` into the
    OpenAI ``tool_call`` dict list a chat-completions client observes."""
    out: list[dict] = []
    for idx, tc in enumerate(extracted.tool_calls):
        # The parser emits ``{"name": ..., "arguments": <json str>}`` dicts.
        out.append(
            {
                "id": tc.get("id") or f"call_hy3_{idx}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Tool-call path — API-shape contract
# --------------------------------------------------------------------------- #


class TestHy3ToolCallWireOffline:
    """Feed captured Hy3 tool-call wire through the parser; assert the
    OpenAI-shape ``tool_calls`` array a real agent client would see."""

    def test_single_tool_call_wellformed(self) -> None:
        parser = _tool_parser()
        res = parser.extract_tool_calls(_WIRE_SINGLE_TOOL_CALL)
        assert res.tools_called is True, res
        assert res.content is None, (
            f"content leaked alongside tool call: {res.content!r}"
        )

        tool_calls = _tool_calls_to_openai_shape(res)
        assert len(tool_calls) == 1, tool_calls
        tc = tool_calls[0]
        # Reuse the exact shared helper the live matrix cells use.
        assert_tool_call_shape(tc)
        assert tc["function"]["name"] == "get_weather", tc
        args = json.loads(tc["function"]["arguments"])
        assert args == {"city": "Tokyo"}, args

    def test_multiple_tool_calls_each_wellformed(self) -> None:
        parser = _tool_parser()
        res = parser.extract_tool_calls(_WIRE_MULTI_TOOL_CALL)
        assert res.tools_called is True, res

        tool_calls = _tool_calls_to_openai_shape(res)
        assert len(tool_calls) == 2, tool_calls
        for tc in tool_calls:
            assert_tool_call_shape(tc)
        assert [tc["function"]["name"] for tc in tool_calls] == [
            "get_weather",
            "get_time",
        ]
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
        assert json.loads(tool_calls[1]["function"]["arguments"]) == {
            "tz": "Asia/Tokyo"
        }

    def test_malformed_close_still_surfaces_call(self) -> None:
        """4-bit numerical-noise close — the tool name must still surface as
        a well-formed OpenAI tool_call with empty args, so a client sees a
        real call instead of an empty array and thinks the model refused."""
        parser = _tool_parser()
        res = parser.extract_tool_calls(_WIRE_MALFORMED_CLOSE)
        assert res.tools_called is True, res

        tool_calls = _tool_calls_to_openai_shape(res)
        assert len(tool_calls) == 1, tool_calls
        tc = tool_calls[0]
        assert_tool_call_shape(tc)  # arguments="{}" is still JSON-parseable
        assert tc["function"]["name"] == "get_weather", tc
        assert json.loads(tc["function"]["arguments"]) == {}

    def test_plain_content_is_not_a_tool_call(self) -> None:
        """A plain reply with no ``<tool_call>`` opener routes to content,
        not a phantom tool call — and the content is channel-clean."""
        parser = _tool_parser()
        res = parser.extract_tool_calls(_WIRE_PLAIN_CONTENT)
        assert res.tools_called is False, res
        assert res.tool_calls == [], res.tool_calls
        assert res.content == _WIRE_PLAIN_CONTENT
        assert_no_think_tag_leak(res.content)
        assert_no_analysis_channel_leak(res.content)


# --------------------------------------------------------------------------- #
# Reasoning path — think content routed, not leaked
# --------------------------------------------------------------------------- #


class TestHy3ReasoningWireOffline:
    """Feed captured Hy3 ``<think:opensource>`` wire through the reasoning
    parser; assert reasoning is routed to its own channel and never leaks
    into the visible content a chat client renders."""

    def test_reasoning_routed_content_clean(self) -> None:
        parser = _reasoning_parser()
        reasoning, content = parser.extract_reasoning(_WIRE_REASONING_THEN_ANSWER)
        assert reasoning is not None and reasoning.strip(), reasoning
        # The reasoning channel carries the extracted think TEXT verbatim,
        # with the wrapping ``<think:opensource>…</think:opensource>`` tags
        # (and the ``:opensource`` suffix) stripped — assert the exact
        # payload, not just a substring, so a parser that left the raw tags
        # in ``reasoning`` would fail here (not only in ``content``).
        assert reasoning == (
            "The user wants the weather in Tokyo. I should call the get_weather tool."
        ), reasoning
        assert_no_think_tag_leak(reasoning)
        assert ":opensource" not in reasoning, reasoning
        assert content == "The weather in Tokyo is sunny.", content
        # The visible content the client renders must be think-tag-clean.
        assert_no_think_tag_leak(content)
        assert_no_analysis_channel_leak(content)
        # And the raw ``:opensource`` suffix must never survive into content.
        assert ":opensource" not in content, content

    def test_reasoning_only_no_content_leak(self) -> None:
        parser = _reasoning_parser()
        reasoning, content = parser.extract_reasoning(_WIRE_REASONING_ONLY)
        assert reasoning is not None and reasoning.strip(), reasoning
        # Reasoning channel holds the stripped think text — no raw tags/suffix.
        assert reasoning == "Let me work through this step by step.", reasoning
        assert_no_think_tag_leak(reasoning)
        assert ":opensource" not in reasoning, reasoning
        # Pure reasoning span — content is empty / None, never the raw tags.
        assert not (content or "").strip(), content
        assert_no_think_tag_leak(content or "")

    def test_plain_content_no_reasoning(self) -> None:
        """A reply with no ``<think>`` span → reasoning is None, content
        passes through verbatim."""
        parser = _reasoning_parser()
        reasoning, content = parser.extract_reasoning(_WIRE_PLAIN_CONTENT)
        assert reasoning is None, reasoning
        assert content == _WIRE_PLAIN_CONTENT, content


# --------------------------------------------------------------------------- #
# Composed contract — reasoning + tool call in one turn
# --------------------------------------------------------------------------- #


class TestHy3ComposedWireOffline:
    """The realistic Hy3 turn: a ``<think>`` span THEN a tool call. The
    server pipeline routes the reasoning to its channel first, then the
    tool parser extracts the call from the residual content. Assert the
    composed API-shape contract end-to-end, offline."""

    _WIRE_REASON_THEN_TOOL = (
        "<think:opensource>The user asked for the weather in Tokyo; "
        "I'll call get_weather.</think:opensource>"
        "<tool_call:opensource>get_weather"
        '<tool_sep:opensource>{"city": "Tokyo"}'
        "<end_of_tool_call:opensource>"
    )

    def test_reasoning_then_tool_call_both_wellformed(self) -> None:
        reasoning_parser = _reasoning_parser()
        tool_parser = _tool_parser()

        # Stage 1: reasoning parser splits off the think span; the residual
        # is what the tool parser sees (mirrors the server pipeline order).
        reasoning, residual = reasoning_parser.extract_reasoning(
            self._WIRE_REASON_THEN_TOOL
        )
        assert reasoning is not None and "get_weather" in reasoning, reasoning
        assert_no_think_tag_leak(residual or "")
        assert (
            ":opensource"
            not in (
                # only the reasoning-tag suffix should be gone; the tool-call
                # tokens legitimately remain in the residual for stage 2.
                (residual or "").split("<tool_call")[0]
            )
        ), residual

        # Stage 2: tool parser extracts the OpenAI-shape call from residual.
        res = tool_parser.extract_tool_calls(residual or "")
        assert res.tools_called is True, res
        tool_calls = _tool_calls_to_openai_shape(res)
        assert len(tool_calls) == 1, tool_calls
        assert_tool_call_shape(tool_calls[0])
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
