import itertools
import json
from pathlib import Path

import pytest

from rapid_mlx.prompt_parse_context import PromptParseContext
from tests.parser_migration.harness import route

CASES = list(
    itertools.product(
        ["flint", "lfm2.5", "qwen3.5"],
        ["chat", "responses", "messages"],
        [False, True],
        [1, 1000],
        [
            "plain",
            "explicit",
            "tool_in_reasoning",
            "truncated",
            "stray_close",
            "native_tool",
        ],
    )
)


def replay(key):
    model, surface, stream, split, scenario = key
    initial = "reasoning" if model == "lfm2.5" else "content"
    contract = "flint" if model == "flint" else "chatml"
    prompt = (
        "<think>"
        if initial == "reasoning"
        else "<think></think>"
        if model == "qwen3.5"
        else ""
    )
    tool = "glm47" if model == "flint" else "lfm" if model == "lfm2.5" else "hermes"
    call = (
        "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
        if tool == "glm47"
        else '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
    )
    if tool == "lfm":
        call = '[run_shell_command(command="pwd")]'
    native = '<|tool_call_start|>[run_shell_command(command="pwd")]<|tool_call_end|>'
    raw = {
        "stray_close": "unmarked</think>answer",
        "native_tool": ("<think>" if initial == "content" else "")
        + "private</think>"
        + native,
        "plain": "Hello world.",
        "explicit": ("<think>" if initial == "content" else "")
        + "private plan</think>Answer one. Answer two.",
        "tool_in_reasoning": ("<think>" if initial == "content" else "")
        + call
        + "</think>Done.",
        "truncated": ("<think>" if initial == "content" else "")
        + "private unfinished plan",
    }[scenario]
    context = PromptParseContext(prompt, None, initial, False, contract)
    result = route(
        "qwen3",
        tool,
        raw,
        surface,
        stream,
        chunks=[raw[i : i + split] for i in range(0, len(raw), split)],
        thinking=False,
        model=model,
        tools_mode="enabled",
        prompt_context=context,
        finish="length" if scenario == "truncated" else "stop",
    )
    return result


DATA = json.loads(
    Path(__file__).with_name("prompt_owner_differential.json").read_text()
)


@pytest.mark.parametrize("row", DATA["rows"], ids=lambda row: str(row["key"]))
def test_prompt_owner_differential(row):
    assert replay(row["key"]) == row["candidate"]
    assert bool(row["reason"]) == (row["baseline"] != row["candidate"])
    if row["key"][0] == "qwen3.5":
        assert row["candidate"] == row["baseline"]
