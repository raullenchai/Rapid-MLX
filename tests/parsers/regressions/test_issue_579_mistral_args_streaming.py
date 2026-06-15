# SPDX-License-Identifier: Apache-2.0
"""Regression guard for #579 — Mistral ``[ARGS]`` streaming tool-call leak.

Wire format: ``[TOOL_CALLS]<name>[ARGS]{json}`` (Devstral-Small-2 /
Mistral >= v11). Reported on rapid-mlx 0.7.6, model
``mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit``, served with
``--enable-auto-tool-choice --tool-call-parser mistral``.

Symptom: **non-streaming** parsing is correct, but **streaming** garbles
every tool call. The old ``extract_tool_calls_streaming`` parsed each token
delta in isolation; the ``[ARGS]`` separator (which starts with ``[``) fell
into the "arguments continuation" branch and leaked into the body, while the
name was reconstructed from stray fragments. The assembled stream came out as
``name='"}'``, ``arguments='[ARGS]{"'``, ``id=''`` — so OpenAI clients
received ``arguments: []`` and aborted (e.g. agentic coders:
``Validation failed for tool "read": root: must be object``).

Fix: the streaming path now re-parses the full accumulated text with the
(working) non-streaming ``extract_tool_calls`` and emits each call once its
arguments are complete JSON — guaranteeing stream ↔ non-stream parity.

Convention mirrors the other regression guards in this directory: each case
states the wire format up front and drives the *incremental* streaming entry
point the way ``service/postprocessor.py`` does (per-delta, with the full
accumulated text), not just ``finalize()``.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager


def _assemble_stream(parser, chunks: list[str]) -> list[dict]:
    """Drive ``extract_tool_calls_streaming`` exactly like the postprocessor.

    Feeds ``chunks`` one at a time with (previous_text, current_text,
    delta_text) and merges the emitted ``tool_calls`` deltas on ``index``
    (the OpenAI streaming contract), concatenating ``arguments`` fragments.
    Returns the assembled tool calls as ``{index, id, name, arguments}``.
    """
    parser.reset()
    merged: dict[int, dict] = {}
    previous = ""
    for chunk in chunks:
        current = previous + chunk
        result = parser.extract_tool_calls_streaming(previous, current, chunk)
        previous = current
        if not result or "tool_calls" not in result:
            continue
        for delta in result["tool_calls"]:
            idx = delta["index"]
            slot = merged.setdefault(
                idx, {"index": idx, "id": "", "name": "", "arguments": ""}
            )
            if delta.get("id"):
                slot["id"] = delta["id"]
            fn = delta.get("function", {})
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
    return [merged[i] for i in sorted(merged)]


def _norm(calls: list[dict]) -> list[tuple[str, dict]]:
    return [(c["name"], json.loads(c["arguments"])) for c in calls]


@pytest.fixture
def parser():
    return ToolParserManager.get_tool_parser("mistral")(None)


def test_single_args_call_streams_intact(parser):
    # [ARGS] arrives as its own chunk and the JSON body is split — the exact
    # boundary the old per-delta parser mangled.
    chunks = [
        "[TOOL_CALLS]",
        "read",
        "[ARGS]",
        '{"file_path":',
        ' "/tmp/foo',
        '.txt"}',
    ]
    assert _norm(_assemble_stream(parser, chunks)) == [
        ("read", {"file_path": "/tmp/foo.txt"})
    ]


def test_args_token_never_leaks_into_name_or_arguments(parser):
    calls = _assemble_stream(
        parser,
        ["[TOOL_CALLS]", "get_weather", "[ARGS]", '{"city": "Milan"}'],
    )
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert "[ARGS]" not in calls[0]["name"]
    assert "[ARGS]" not in calls[0]["arguments"]
    assert json.loads(calls[0]["arguments"]) == {"city": "Milan"}


def test_multiple_args_calls_stream_intact(parser):
    chunks = [
        "[TOOL_CALLS]",
        "read",
        "[ARGS]",
        '{"file_path": "a.txt"}',
        "[TOOL_CALLS]",
        "read",
        "[ARGS]",
        '{"file_path": ',
        '"b.txt"}',
    ]
    assert _norm(_assemble_stream(parser, chunks)) == [
        ("read", {"file_path": "a.txt"}),
        ("read", {"file_path": "b.txt"}),
    ]


def test_leading_content_emitted_before_tool_call(parser):
    parser.reset()
    contents, previous = [], ""
    for chunk in ["Sure!", "[TOOL_CALLS]", "read", "[ARGS]", '{"file_path": "x"}']:
        current = previous + chunk
        result = parser.extract_tool_calls_streaming(previous, current, chunk)
        previous = current
        if result and result.get("content"):
            contents.append(result["content"])
    assert "".join(contents) == "Sure!"


def test_stream_matches_nonstream(parser):
    """The parity invariant this bug violated: streamed == non-streamed."""
    full = '[TOOL_CALLS]search[ARGS]{"q": "mlx", "k": 5}'
    # The tokenizer emits [TOOL_CALLS]/[ARGS] as atomic special tokens; feed
    # them whole and split only the JSON body across deltas.
    stream = _norm(
        _assemble_stream(
            parser, ["[TOOL_CALLS]", "search", "[ARGS]", '{"q": "mlx", ', '"k": 5}']
        )
    )
    ns = parser.extract_tool_calls(full)
    nonstream = [(tc["name"], json.loads(tc["arguments"])) for tc in ns.tool_calls]
    assert ns.tools_called is True
    assert stream == nonstream == [("search", {"q": "mlx", "k": 5})]
