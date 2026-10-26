"""Tool parsing preserves v0.16.9 acceptance independently of schema constraints."""

import json

import pytest

from .harness import route

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "pattern": "^pwd$"},
        "count": {"type": "integer", "multipleOf": 10},
    },
    "required": ["command"],
    "additionalProperties": False,
    "patternProperties": {"(a+)+$": {"type": "null"}},
}
ARGUMENTS = {"command": "pwd", "count": -10, "extra": True}
TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_shell_command", "parameters": SCHEMA},
    }
]
RAW = (
    "<tool_call>"
    + json.dumps({"name": "run_shell_command", "arguments": ARGUMENTS})
    + "</tool_call>"
)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_schema_violations_preserve_baseline_call(surface, stream, split):
    actual = route(
        "qwen3",
        "hermes",
        RAW,
        surface,
        stream,
        list(RAW) if split else [RAW],
        thinking=False,
        tools=TOOLS,
    )
    assert actual["calls"] == [["run_shell_command", ARGUMENTS]]
