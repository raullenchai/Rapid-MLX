"""Release contracts: strict failures must disappear when their owner is migrated."""

import itertools
import json
from pathlib import Path

import pytest

from rapid_mlx.tool_parsers import ToolParserManager

from .harness import route

SURFACES = ["chat", "responses", "messages"]
CALL = (
    '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
)
EXPECTED = [["run_shell_command", {"command": "pwd"}]]
PLAIN = "Hello world. This is a streamed answer."


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("reasoning", ["qwen3", "hy_v3", "hy3"])
@pytest.mark.parametrize("tools_mode", ["enabled", "absent", "none"])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="R1 #5277: prompt-aware bounded holdback"
)
def test_first_visible_delta(surface, reasoning, tools_mode):
    result = route(
        reasoning,
        "hermes",
        PLAIN,
        surface,
        True,
        thinking=False,
        tools_mode=tools_mode,
        prompt="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
    assert result["content"] == PLAIN
    assert result["first_content_index"] <= 8
    assert result["content_deltas"] > 1


@pytest.mark.parametrize("tool", ["auto", "generic", "nemotron", "nemotron3", "muse"])
@pytest.mark.parametrize("reasoning", ["qwen3", "hy_v3", "hy3"])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="R2 #5277: EOF content replay"
)
def test_one_content_owner(tool, reasoning):
    result = route(reasoning, tool, PLAIN, "chat", True, thinking=False)
    assert result["content"] == PLAIN


@pytest.mark.parametrize("reasoning", ["qwen3", "hy_v3", "hy3"])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="R2b #5277: EOF tool replay"
)
def test_one_tool_owner(reasoning):
    raw = "<seed:tool_call><function=run_shell_command><parameter=command>pwd</parameter></function></seed:tool_call>"
    result = route(reasoning, "seed_oss", raw, "chat", True, thinking=False)
    assert result["calls"] == EXPECTED


@pytest.mark.parametrize(
    "surface,stream", list(itertools.product(SURFACES, [False, True]))
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="R3 #5273: reasoning-to-tool ownership loses call or answer",
)
def test_promotion(surface, stream):
    result = route(
        "qwen3", "hermes", "<think>Secret." + CALL + "</think>Answer", surface, stream
    )
    assert result["calls"] == EXPECTED
    assert result["content"] == "Answer"
    assert result["reasoning"] == "Secret."


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#5272: tool wire duplicated into buffered reasoning",
)
def test_no_tool_wire_in_reasoning(surface):
    result = route("qwen3", "hermes", CALL, surface, False)
    assert result["calls"] == EXPECTED
    assert result["reasoning"] == ""
    assert result["content"] == ""


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#5274: orphan native argument markup"
)
def test_orphan_arguments(surface):
    raw = "<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>Answer"
    result = route("qwen3", "glm47", raw, surface, True)
    assert result["content"] == "Answer"
    assert result["reasoning"] == ""
    assert result["calls"] == []


@pytest.mark.parametrize("surface", ["responses", "messages"])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="R5 #5274: foreign malformed envelope loses trailing answer",
)
def test_foreign_fallback(surface):
    raw = '<tool_call>{"name":"run_shell_command","arguments":oops}</tool_call>Answer'
    result = route("deepseek_r1", "deepseek_v3", raw, surface, True)
    assert result["content"].endswith("Answer")
    assert result["calls"] == []


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="R6 #5274: preceding prose establishes literal context",
)
def test_literal_incomplete_header(surface):
    result = route(
        "qwen3", "glm47", "Example <tool_call>foo", surface, False, thinking=False
    )
    assert result["content"] == "Example <tool_call>foo"
    assert result["calls"] == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#5268: generated fenced echo policy remains unresolved",
)
def test_generated_echo():
    raw = (
        '```json\n{"name":"run_shell_command","arguments":{"command":"pwd"}}\n```'
        + CALL
    )
    result = route("qwen3", "hermes", raw, "responses", True, thinking=False)
    assert result["calls"] == EXPECTED
    assert result["content"] == ""


def native_cases():
    fixtures = json.loads(Path(__file__).with_name("native.json").read_text())
    fixtures += [
        [
            "glm47",
            "glm",
            "<tool_call>read_file<arg_key>path</arg_key><arg_value>sample.txt</arg_value></tool_call>",
            [["read_file", {"path": "sample.txt"}]],
        ],
        [
            "nemotron",
            "xml",
            "<tool_call><function=read_file><parameter=path>sample.txt</parameter></function></tool_call>",
            [["read_file", {"path": "sample.txt"}]],
        ],
        [
            "xlam",
            "json",
            '[{"name":"read_file","arguments":{"path":"sample.txt"}}]',
            [["read_file", {"path": "sample.txt"}]],
        ],
    ]
    by_class = {}
    for name, _, raw, calls in fixtures:
        by_class.setdefault(ToolParserManager.get_tool_parser(name), (raw, calls))
    return [
        (name, *by_class[ToolParserManager.get_tool_parser(name)])
        for name in ToolParserManager.list_registered()
    ]


@pytest.mark.parametrize("tool,raw,expected", native_cases())
@pytest.mark.parametrize(
    "surface,stream", list(itertools.product(SURFACES, [False, True]))
)
def test_all_native_aliases(tool, raw, expected, surface, stream):
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name, _ in expected
    ]
    result = route(None, tool, raw, surface, stream, tools=tools)
    assert result["calls"] == expected


PROMPT_STATES = json.loads(Path(__file__).with_name("prompt_states.json").read_text())


@pytest.mark.parametrize("prompt_state", PROMPT_STATES, ids=lambda state: state["name"])
@pytest.mark.parametrize("surface", SURFACES)
def test_prompt_state_evidence(prompt_state, surface):
    # These fixtures deliberately retain history separately: later migration
    # must never infer the new turn's channel from the old turn's closing tag.
    initial = prompt_state["channel"]
    raw = "Secret.</think>" + PLAIN if initial == "reasoning" else PLAIN
    result = route(
        "qwen3",
        "hermes",
        raw,
        surface,
        True,
        thinking=initial == "reasoning",
        prompt=prompt_state["history"] + prompt_state["assistant_suffix"],
        tools_mode="absent",
    )
    assert result["content"] == PLAIN
    assert result["calls"] == []
    if initial == "reasoning":
        assert result["reasoning"] == "Secret."
        assert result["first_content_index"] <= len("Secret.</think>") + 8
        assert result["content_deltas"] > 1


@pytest.mark.parametrize(
    "surface,stream", list(itertools.product(SURFACES, [False, True]))
)
@pytest.mark.parametrize("choice", ["auto", "none", "required", "named"])
def test_tool_choice_contract(surface, stream, choice):
    if surface == "messages":
        wire_choice = (
            {"type": "tool", "name": "run_shell_command"}
            if choice == "named"
            else {"type": "any" if choice == "required" else choice}
        )
    else:
        wire_choice = (
            {"type": "function", "function": {"name": "run_shell_command"}}
            if choice == "named" and surface == "chat"
            else {"type": "function", "name": "run_shell_command"}
            if choice == "named"
            else choice
        )
    raw = CALL
    if surface == "chat" and choice in {"required", "named"}:
        from rapid_mlx.routes.chat import _forced_tool_call_prefix

        raw = (
            _forced_tool_call_prefix("hermes", "run_shell_command")
            + '{"command":"pwd"}}</tool_call>'
        )
    result = route(None, "hermes", raw, surface, stream, tool_choice=wire_choice)
    assert result["calls"] == ([] if choice == "none" else EXPECTED)


@pytest.mark.parametrize(
    "surface,stream", list(itertools.product(SURFACES, [False, True]))
)
def test_undeclared_tool_is_not_executable(surface, stream):
    raw = CALL.replace("run_shell_command", "unauthorized")
    result = route(None, "hermes", raw, surface, stream)
    assert result["calls"] == []
