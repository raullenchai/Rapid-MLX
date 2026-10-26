"""Frozen review observations; intentional changes require a ledger update."""

import json
from pathlib import Path

import pytest

from .harness import route

DATA = json.loads(Path(__file__).with_name("differential.json").read_text())
CALL = (
    '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
)
RAW = {
    "plain": "Hello world. This is a streamed answer.",
    "explicit": "<think>Secret.</think>Answer",
    "implicit": "Secret.</think>Answer",
    "nested": "<think>Secret.<think>Nested</think>Answer</think>Tail",
    "native": CALL,
    "promoted": "<think>Secret." + CALL + "</think>Answer",
    "orphan": "<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>Answer",
    "malformed": '<tool_call>{"name":"run_shell_command","arguments":oops}</tool_call>Answer',
    "literal": "Example <tool_call> and <|im_end|> are literal markers.",
    "fenced": '```json\n{"name":"run_shell_command","arguments":{"command":"pwd"}}\n```'
    + CALL,
    "harmony": "<|start|>assistant<|channel|>analysis<|message|>Secret<|end|><|start|>assistant<|channel|>final<|message|>Answer<|return|>",
    "gemma": "<|channel>thought\nSecret<channel|>Answer",
}


@pytest.mark.parametrize(
    "row",
    [r for r in DATA["rows"] if r["revision"] == "main"],
    ids=lambda r: str(r["key"]),
)
def test_review_head_observations(row):
    reasoning, tool, scenario, thinking, tools_mode, surface, stream, split = row["key"]
    raw = RAW[scenario]
    chunks = (
        list(raw)
        if split == "chars"
        else [raw]
        if split == "whole"
        else [raw[i : i + 7] for i in range(0, len(raw), 7)]
    )
    actual = route(
        reasoning, tool, raw, surface, stream, chunks, thinking, tools_mode=tools_mode
    )
    assert actual == row["result"]


def test_every_baseline_difference_has_a_ledger_entry():
    ledger = json.loads(Path(__file__).with_name("ledger.json").read_text())
    baseline = {
        tuple(row["key"]): row["result"]
        for row in DATA["rows"]
        if row["revision"] == "baseline"
    }
    current = {
        tuple(row["key"]): row["result"]
        for row in DATA["rows"]
        if row["revision"] == "main"
    }
    changed = {key for key in baseline if baseline[key] != current[key]}
    assert changed == {tuple(row["key"]) for row in ledger}
    for row in ledger:
        key = tuple(row["key"])
        assert row["baseline"] == baseline[key]
        assert row["main"] == current[key]
        assert row["reasons"]
