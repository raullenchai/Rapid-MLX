# SPDX-License-Identifier: Apache-2.0
"""Promoted tool calls retain reasoning and the actual answer across routes."""

import pytest

from tests.test_reasoning_boundary_routes import route

CALL = (
    '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "raw,reason,answer",
    [(CALL, "", ""), ("<think>Secret." + CALL + "</think>Answer", "Secret.", "Answer")],
)
def test_tool_reasoning_answer(surface, raw, reason, answer):
    expected = {
        "content": answer,
        "reasoning": reason,
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route("qwen3", "hermes", raw, surface, False) == expected
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        assert route("qwen3", "hermes", raw, surface, True, chunks) == expected, chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("reasoning_parser", ["qwen3", "glm5"])
def test_native_orphan_arguments_keep_answer(surface, thinking, reasoning_parser):
    raw = "<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>Answer"
    expected = {"content": "Answer", "reasoning": "", "calls": []}
    assert (
        route(reasoning_parser, "glm47", raw, surface, False, thinking=thinking)
        == expected
    )
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        assert (
            route(
                reasoning_parser, "glm47", raw, surface, True, chunks, thinking=thinking
            )
            == expected
        ), chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_foreign_malformed_envelope_is_preserved_once(surface):
    raw = '<tool_call>{"name":"run_shell_command","arguments":BROKEN}</tool_call>'
    expected = {"content": raw, "reasoning": "", "calls": []}
    assert route("qwen3", "glm47", raw, surface, False) == expected
    for chunks in [[raw], list(raw)]:
        assert route("qwen3", "glm47", raw, surface, True, chunks) == expected


def test_base_implicit_reasoning_promotes_tool_only_wire():
    from rapid_mlx.reasoning import get_parser

    assert get_parser("deepseek_r1")().extract_reasoning(
        CALL, enable_thinking=True
    ) == (None, CALL)


def test_no_declared_tools_preserve_original_output():
    from rapid_mlx.config import reset_config
    from rapid_mlx.reasoning import get_parser
    from rapid_mlx.service.helpers import _parse_tool_calls_with_parser

    cfg = reset_config()
    cfg.reasoning_parser = get_parser("qwen3")()
    raw = "<think>Secret." + CALL + "</think>Answer"
    try:
        assert _parse_tool_calls_with_parser(raw) == (raw, None)
    finally:
        reset_config()


def test_shared_base_implicit_tool_promotion():
    from rapid_mlx.reasoning import get_parser
    from rapid_mlx.reasoning.think_parser import BaseThinkingReasoningParser

    assert BaseThinkingReasoningParser.extract_reasoning(
        get_parser("qwen3")(), CALL, enable_thinking=True
    ) == (None, CALL)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_tool_arguments_can_contain_literal_closers(surface):
    import json

    arguments = {"command": 'echo "</tool_call>"'}
    raw = (
        "<tool_call>"
        + json.dumps({"name": "run_shell_command", "arguments": arguments})
        + "</tool_call>"
    )
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", arguments]],
    }
    assert route("qwen3", "hermes", raw, surface, False) == expected
    for chunks in [[raw], list(raw)]:
        assert route("qwen3", "hermes", raw, surface, True, chunks) == expected


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("reasoning_parser", ["qwen3", "glm5"])
def test_native_named_call_is_not_reasoning(surface, reasoning_parser):
    raw = "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route(reasoning_parser, "glm47", raw, surface, False) == expected
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        assert (
            route(reasoning_parser, "glm47", raw, surface, True, chunks) == expected
        ), chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_implicit_json_tool_only_other_reasoning_family(surface):
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route("deepseek_r1", "hermes", CALL, surface, False) == expected
    for chunks in [[CALL], list(CALL)] + [
        [CALL[:i], CALL[i:]] for i in range(1, len(CALL))
    ]:
        assert (
            route("deepseek_r1", "hermes", CALL, surface, True, chunks) == expected
        ), chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_native_header_does_not_enter_implicit_reasoning(surface):
    raw = (
        "<|tool_calls_section_begin|><|tool_call_begin|>run_shell_command:0"
        '<|tool_call_argument_begin|>{"command":"pwd"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route("deepseek_r1", "kimi", raw, surface, False) == expected
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        assert route("deepseek_r1", "kimi", raw, surface, True, chunks) == expected, (
            chunks
        )


@pytest.mark.parametrize("parser_name", ["qwen3", "glm5", "deepseek_r1"])
def test_incomplete_native_candidate_is_preserved_at_eof(parser_name):
    from rapid_mlx.reasoning import get_parser

    parser = get_parser(parser_name)()
    parser.configure_request(enable_thinking=True, tool_parser_name="glm47")
    assert parser.extract_reasoning_streaming("", "<arg", "<arg") is None
    assert parser.finish_stream().content == "<arg"
    assert parser.finish_stream() is None


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_suffixed_native_call_is_not_buffered_reasoning(surface):
    raw = '<tool_call:opensource>run_shell_command<tool_sep:opensource>{"command":"pwd"}<end_of_tool_call:opensource>'
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route("hy_v3", "hy_v3", raw, surface, False) == expected
    for chunks in [[raw], list(raw)]:
        assert route("hy_v3", "hy_v3", raw, surface, True, chunks) == expected


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "reasoning_parser,tool_parser", [("qwen3", "hermes"), ("deepseek_r1", "kimi")]
)
def test_reasoning_after_native_call_keeps_boundaries(
    surface, reasoning_parser, tool_parser
):
    wire = (
        CALL
        if tool_parser == "hermes"
        else (
            "<|tool_calls_section_begin|><|tool_call_begin|>run_shell_command:0"
            '<|tool_call_argument_begin|>{"command":"pwd"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
    )
    raw = wire + "<think>Secret.</think>Answer"
    expected = {
        "content": "Answer",
        "reasoning": "Secret.",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route(reasoning_parser, tool_parser, raw, surface, False) == expected
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        assert (
            route(reasoning_parser, tool_parser, raw, surface, True, chunks) == expected
        ), chunks


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_named_native_call_inside_reasoning_keeps_answer(surface):
    wire = "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
    raw = "<think>Secret." + wire + "</think>Answer"
    expected = {
        "content": "Answer",
        "reasoning": "Secret.",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route("qwen3", "glm47", raw, surface, False) == expected
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        assert route("qwen3", "glm47", raw, surface, True, chunks) == expected, chunks


def test_threshold_transition_replays_held_literal_prefix():
    from rapid_mlx.reasoning import get_parser

    parser = get_parser("deepseek_r1")()
    parser.configure_request(enable_thinking=True)
    previous = "x" * 62 + "<"
    assert (
        parser.extract_reasoning_streaming("", previous, previous).reasoning == "x" * 62
    )
    assert (
        parser.extract_reasoning_streaming(previous, previous + "z", "z").content
        == "<z"
    )


def test_terminal_reasoning_after_tool_retains_both_channels(monkeypatch):
    from rapid_mlx.engine.base import GenerationOutput
    from tests.test_reasoning_boundary_routes import Engine

    async def stream_chat(self, **kwargs):
        text = ""
        for i, chunk in enumerate(self.chunks):
            text += chunk
            yield GenerationOutput(
                text=text,
                new_text=chunk,
                finished=i == len(self.chunks) - 1,
                finish_reason="stop" if i == len(self.chunks) - 1 else None,
            )

    monkeypatch.setattr(Engine, "stream_chat", stream_chat)
    tail = "<think>Secret.</think>Answer"
    assert route("qwen3", "hermes", CALL + tail, "chat", True, [CALL, tail]) == {
        "content": "Answer",
        "reasoning": "Secret.",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_native_argument_markers_are_literal_data(surface):
    import json

    arguments = {"command": "echo '<|tool_calls_section_end|><think>literal</think>'"}
    raw = (
        "<|tool_calls_section_begin|><|tool_call_begin|>run_shell_command:0<|tool_call_argument_begin|>"
        + json.dumps(arguments)
        + "<|tool_call_end|><|tool_calls_section_end|>"
    )
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", arguments]],
    }
    assert route("deepseek_r1", "kimi", raw, surface, False) == expected
    for chunks in [[raw], list(raw)]:
        assert route("deepseek_r1", "kimi", raw, surface, True, chunks) == expected


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
def test_native_xml_tool_only_is_not_reasoning(surface):
    raw = '<minimax:tool_call><invoke name="run_shell_command"><parameter name="command">pwd</parameter></invoke></minimax:tool_call>'
    expected = {
        "content": "",
        "reasoning": "",
        "calls": [["run_shell_command", {"command": "pwd"}]],
    }
    assert route("qwen3", "minimax", raw, surface, False) == expected
    for chunks in [[raw], list(raw)]:
        assert route("qwen3", "minimax", raw, surface, True, chunks) == expected


def test_literal_partial_tool_name_stays_in_reasoning():
    from rapid_mlx.reasoning import get_parser

    parser = get_parser("qwen3")()
    parser.configure_request(enable_thinking=True, tool_parser_name="glm47")
    chunks = ["<think>Discuss <tool_cal", "</think>Answer"]
    previous = reasoning = content = ""
    for chunk in chunks:
        result = parser.extract_reasoning_streaming(previous, previous + chunk, chunk)
        previous += chunk
        if result:
            reasoning += result.reasoning or ""
            content += result.content or ""
    assert reasoning == "Discuss <tool_cal"
    assert content == "Answer"
