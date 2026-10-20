# SPDX-License-Identifier: Apache-2.0
"""Buffered suffix cleanup with unchanged streaming behavior."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.anthropic import router as messages_router
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.routes.responses import router as responses_router

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]
TRUNCATED = [
    "<tool_call>",
    "<tool_call>\n<function=get_weather",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
]
COMPLETE = "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, text, finish_reason="length", chunks=None):
        self.text = text
        self.finish_reason = finish_reason
        self.chunks = chunks or [text]

    def build_prompt(self, messages, **kwargs):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=self.text,
            prompt_tokens=4,
            completion_tokens=4,
            finished=True,
            finish_reason=self.finish_reason,
        )

    async def stream_chat(self, messages, **kwargs):
        accumulated = ""
        for index, chunk in enumerate(self.chunks):
            accumulated += chunk
            last = index == len(self.chunks) - 1
            yield GenerationOutput(
                text=accumulated,
                new_text=chunk,
                prompt_tokens=4,
                completion_tokens=index + 1,
                finished=last,
                finish_reason=self.finish_reason if last else None,
            )


@pytest.fixture
def client():
    cfg = reset_config()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    for router in (chat_router, messages_router, responses_router):
        app.include_router(router)
    try:
        with TestClient(app) as http:
            yield http, cfg
    finally:
        reset_config()


def _reply(
    client,
    route,
    text,
    *,
    stream=False,
    finish_reason="length",
    choice="auto",
    tools=TOOLS,
    chunks=None,
    budget=16,
    with_reasoning=False,
):
    http, cfg = client
    cfg.engine = _Engine(text, finish_reason, chunks)
    body = {
        "model": "test-model",
        "stream": stream,
        "max_tokens": budget,
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": tools,
        "tool_choice": choice,
    }
    if route == "messages":
        body["tools"] = [
            {"name": t["function"]["name"], "input_schema": t["function"]["parameters"]}
            for t in tools
        ]
        body["tool_choice"] = (
            {"type": "tool", "name": choice["function"]["name"]}
            if isinstance(choice, dict)
            else {"type": choice}
        )
    elif route == "responses":
        if isinstance(choice, dict):
            body["tool_choice"] = {
                "type": "function",
                "name": choice["function"]["name"],
            }
        body["input"] = "Weather?"
        body["max_output_tokens"] = body.pop("max_tokens")
        body.pop("messages")
        body["tools"] = [{"type": "function", **t["function"]} for t in tools]
    path = {
        "chat": "chat/completions",
        "messages": "messages",
        "responses": "responses",
    }[route]
    response = http.post("/v1/" + path, json=body)
    assert response.status_code == 200, response.text
    if stream:
        if route != "chat":
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            if route == "messages":
                return (
                    "".join(e.get("delta", {}).get("text", "") for e in events),
                    [
                        e["content_block"]
                        for e in events
                        if e.get("content_block", {}).get("type") == "tool_use"
                    ],
                    next(
                        e["delta"]["stop_reason"]
                        for e in events
                        if e["type"] == "message_delta"
                    ),
                )
            return (
                "".join(
                    e["delta"]
                    for e in events
                    if e["type"] == "response.output_text.delta"
                ),
                [
                    e["item"]
                    for e in events
                    if e.get("item", {}).get("type") == "function_call"
                ],
                events[-1]["response"]["status"],
            )
        choices = [
            c
            for line in response.text.splitlines()
            if line.startswith("data: {")
            for c in json.loads(line[6:])["choices"]
        ]
        return (
            "".join(c["delta"].get("content") or "" for c in choices),
            [c["delta"]["tool_calls"] for c in choices if c["delta"].get("tool_calls")],
            choices[-1]["finish_reason"],
        )
    data = response.json()
    if with_reasoning:
        return data
    if route == "chat":
        choice = data["choices"][0]
        return (
            choice["message"].get("content") or "",
            choice["message"].get("tool_calls"),
            choice["finish_reason"],
        )
    if route == "messages":
        return (
            "".join(b["text"] for b in data["content"] if b["type"] == "text"),
            [b for b in data["content"] if b["type"] == "tool_use"],
            data["stop_reason"],
        )
    return (
        "".join(
            c["text"]
            for b in data["output"]
            if b["type"] == "message"
            for c in b["content"]
            if c["type"] == "output_text"
        ),
        [b for b in data["output"] if b["type"] == "function_call"],
        data["status"],
    )


@pytest.mark.parametrize("parser", ["qwen3_coder_xml", "hermes"])
@pytest.mark.parametrize("chunk_size", [1, 7, 200])
@pytest.mark.parametrize("stream", [False, True])
def test_forced_named_truncated_json_4663(client, stream, parser, chunk_size):
    client[1].tool_call_parser = parser
    tools = [
        {
            "type": "function",
            "function": {
                "name": "release_probe",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    raw = '<tool_call> {"name": "release_probe", "'
    content, calls, _ = _reply(
        client,
        "chat",
        raw,
        chunks=[raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)],
        stream=stream,
        choice={"type": "function", "function": {"name": "release_probe"}},
        tools=tools,
    )
    assert calls
    assert content == ""
    if stream:
        names = "".join(
            c.get("function", {}).get("name") or "" for chunk in calls for c in chunk
        )
        arguments = "".join(
            c.get("function", {}).get("arguments") or ""
            for chunk in calls
            for c in chunk
        )
    else:
        assert len(calls) == 1
        names = calls[0]["function"]["name"]
        arguments = calls[0]["function"]["arguments"]
    assert names == "release_probe"
    assert json.loads(arguments) == {}


def _clean_content(text, choice=None):
    from rapid_mlx.service.helpers import _parse_tool_calls_with_parser

    class Request:
        tools = TOOLS

        def model_dump(self):
            return {"tools": self.tools, "tool_choice": choice}

    return _parse_tool_calls_with_parser(text, Request(), finish_reason="length")[0]


PARSERS = ["hermes", "qwen3_coder_xml"]
ROUTES = ["chat", "messages", "responses"]
PARTIAL = TRUNCATED + [
    "<tool_call>{",
    "<tool_call>{ \n\t",
    '<tool_call>{"name":"get_weather","arguments":{"city":"</tool_call>',
    r'<tool_call>{"name":"get_weather","arguments":{"city":"escaped \" </tool_call>',
    "<tool_call>\n<function",
    '<tool_call>\n{"name',
    '<tool_call>\n{  "',
    '<tool_call>\n{ "n',
    '<tool_call>\n{ "na',
    '<tool_call>\n{ "nam',
    '<tool_call>\n{ "name"',
    '<tool_call>\n{ "name": "get_weather", "arguments":',
]


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("text", PARTIAL)
@pytest.mark.parametrize("narration", ["", "Let me check.\n"])
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_buffered_truncated(client, parser, route, text, narration, finish_reason):
    client[1].tool_call_parser = parser
    content, calls, finish = _reply(
        client, route, narration + text, budget=4, finish_reason=finish_reason
    )
    assert content == (
        narration.strip() if finish_reason == "length" else (narration + text).strip()
    )
    assert not calls
    assert (
        finish
        == {
            "length": {
                "chat": "length",
                "messages": "max_tokens",
                "responses": "incomplete",
            },
            "stop": {"chat": "stop", "messages": "end_turn", "responses": "completed"},
        }[finish_reason][route]
    )


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_buffered_finish_reason_gate(client, parser, route, reasoning, finish_reason):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = client[1]
    cfg.tool_call_parser = parser
    if reasoning:
        cfg.reasoning_parser_name = "qwen3"
        cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    narration = "Here is an unclosed example.\n"
    raw = narration + '<tool_call>\n{"name": "get_wea'
    content, calls, finish = _reply(client, route, raw, finish_reason=finish_reason)
    assert content == (narration.strip() if finish_reason == "length" else raw)
    assert not calls
    assert (
        finish
        == {
            "stop": {"chat": "stop", "messages": "end_turn", "responses": "completed"},
            "length": {
                "chat": "length",
                "messages": "max_tokens",
                "responses": "incomplete",
            },
        }[finish_reason][route]
    )


@pytest.mark.parametrize("parser", PARSERS)
def test_helper_finish_reason_gate(client, parser):
    from rapid_mlx.service.helpers import _strip_trailing_tool_envelope

    client[1].tool_call_parser = parser
    narration = "Here is an unclosed example.\n"
    raw = narration + '<tool_call>\n{"name": "get_wea'
    assert _strip_trailing_tool_envelope(raw) == raw
    assert _strip_trailing_tool_envelope(raw, finish_reason=None) == raw
    assert _strip_trailing_tool_envelope(raw, finish_reason="stop") == raw
    assert _strip_trailing_tool_envelope(raw, finish_reason="length") == narration


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "text",
    [
        "Use <tool_call> {json} to describe the format.",
        "Use <tool_call> to introduce a call.",
        'Use <tool_call> {"name":"get_weather"} as a template.',
        "Use `code\n<tool_call>\n<function=get_weather\n` here.",
        "<tool_call> then more prose",
        '<tool_call> {"example": "not a call"',
        "Use `<tool_call>` here",
        "Use ``<tool_call>\n<function`` here",
        "```xml\n<tool_call>\n<function=get_weather\n```",
        '~~~xml\n<tool_call>\n{"name\n~~~',
        "<tool_",
    ],
)
def test_prose_and_code_preserved(client, parser, route, text):
    client[1].tool_call_parser = parser
    assert _reply(client, route, text)[0] == text


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("indent", ["    ", "        ", "\t", "  \t"])
@pytest.mark.parametrize("fence", ["~~~", "````"])
def test_nested_fenced_example_preserved(client, parser, route, indent, fence):
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    client[1].tool_call_parser = parser
    text = (
        f"Example:\n\n- XML tool format:\n\n{indent}{fence}xml\n"
        f"{indent}<tool_call>\n{indent}<function=get_weather\n{indent}{fence}\n"
    )
    content, calls, _ = _reply(client, route, text)
    # Routes already trim outer whitespace; cleanup itself must preserve it.
    assert content == text.strip()
    cls = ToolParserManager.get_tool_parser(parser)
    assert cls.strip_trailing_tool_envelope(text, {"tools": TOOLS}) == text
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("indent", ["    ", "        ", "\t", "  \t"])
@pytest.mark.parametrize("preceding", ["", "Example:\n\n", "Example:\n\n    code\n"])
def test_indented_example_preserved(client, parser, route, indent, preceding):
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    client[1].tool_call_parser = parser
    text = preceding + indent + "<tool_call>\n" + indent + "<function=get_weather"
    content, calls, _ = _reply(client, route, text)
    assert content == text.strip()
    cls = ToolParserManager.get_tool_parser(parser)
    assert cls.strip_trailing_tool_envelope(text, {"tools": TOOLS}) == text
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("closing", ["~~~", "~~~~~", "```", "~~", "~~~ trailing"])
def test_nested_fence_closing_rules(client, parser, route, closing):
    client[1].tool_call_parser = parser
    prefix = f"Example:\n\n    ~~~xml\n    code\n    {closing}\n"
    text = prefix + "<tool_call>\n<function=get_weather"
    content, calls, _ = _reply(client, route, text)
    assert content == (prefix.strip() if closing in ("~~~", "~~~~~") else text)
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("prefix", ["Prose\n", "Example:\n\n    code\nProse\n"])
def test_indentation_after_prose_is_not_code(client, parser, route, prefix):
    client[1].tool_call_parser = parser
    content, calls, _ = _reply(
        client, route, prefix + "    <tool_call>\n    <function=get_weather"
    )
    assert content == prefix.strip()
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "tail", ["", "\n<tool_call>", "\n<tool_call>\n<function", '\n<tool_call>\n{"name']
)
def test_complete_call_then_truncated(client, parser, route, tail):
    client[1].tool_call_parser = parser
    content, calls, _ = _reply(client, route, COMPLETE + tail)
    assert content == ""
    assert len(calls) == 1
    call = calls[0]
    assert (call.get("function", {}).get("name") or call.get("name")) == "get_weather"
    arguments = call.get("function", {}).get(
        "arguments", call.get("arguments", call.get("input"))
    )
    assert (json.loads(arguments) if isinstance(arguments, str) else arguments) == {
        "city": "Paris"
    }


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("text", ["<tool_call>\n<function", '<tool_call>\n{"name'])
def test_reasoning_fallback(client, parser, route, text):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = client[1]
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    assert _reply(client, route, text)[0] == ""


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup(client, parser, record_property):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    timings = []
    for count in (800, 6400):
        raw = "Use <tool_call> for a call.\n" * count
        samples = []
        for _ in range(3):
            start = perf_counter()
            cleaned = cls.strip_trailing_tool_envelope(raw)
            samples.append(perf_counter() - start)
            assert cleaned == raw
        timings.append(min(samples))
    record_property("cleanup_n_seconds", timings[0])
    record_property("cleanup_8n_seconds", timings[1])
    # Allow scheduling noise while rejecting quadratic (64x) growth.
    assert timings[1] < 40 * timings[0]


@pytest.mark.parametrize("parser", PARSERS)
def test_undeclared_xml_policy(client, parser):
    client[1].tool_call_parser = parser
    raw = "<tool_call>\n<function=unknown>\n<parameter=city>Paris"
    expected = raw if parser == "qwen3_coder_xml" else ""
    assert _reply(client, "chat", raw)[0] == expected
    assert _reply(client, "chat", raw, stream=True)[0] == expected


@pytest.mark.parametrize("parser", PARSERS)
def test_cleanup_guards(client, parser):
    client[1].tool_call_parser = parser
    assert _clean_content("<tool_call>", choice="none") == "<tool_call>"
    assert _reply(client, "chat", "<tool_call>", tools=[])[0] == "<tool_call>"
    assert _clean_content("plain text") == "plain text"
    assert _clean_content("<tool_call>{}</tool_call>") == "<tool_call>{}</tool_call>"
    assert _clean_content("```xml\n<tool_call>") == "```xml\n<tool_call>"
    assert (
        _clean_content("```xml\n<tool_call>\n```\n<tool_call>")
        == "```xml\n<tool_call>\n```\n"
    )


@pytest.mark.parametrize(
    "parser,enabled",
    [("hermes", False), ("missing-parser", True), (None, True), ("qwen", True)],
)
def test_configuration_guard(client, parser, enabled):
    client[1].tool_call_parser = parser
    client[1].enable_auto_tool_choice = enabled
    assert _clean_content("<tool_call>") == "<tool_call>"


def test_structured_content_authoritative(client):
    from rapid_mlx.service.helpers import _parse_tool_calls_with_parser

    class Request:
        tools = TOOLS

        def model_dump(self):
            return {"tools": self.tools}

    content, calls = _parse_tool_calls_with_parser(
        "<tool_call>",
        Request(),
        structured_tool_calls=[{"name": "get_weather", "arguments": "{}"}],
    )
    assert content == "<tool_call>"
    assert len(calls) == 1


def _stream_snapshots():
    """Review reproducers at the shared streaming seam, including EOF events."""
    from dataclasses import asdict

    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.service.postprocessor import StreamingPostProcessor
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager
    from tests.test_postprocessor import _make_cfg, _make_output

    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k != "id"}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value

    reproducers = [
        '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>\n<tool_call>\n',
        "Use <tool_call> {json} to describe the format.",
        "Use the <tool_call> tag.\nLet me check.\n<tool_call>\n<function=get_weather",
        "<think>Let me think.</think><tool_call>\n<function=get_weather",
        "<think>I should use <tool_call>\n<function=get_weather> to look this up.</think>Final answer.",
        "Let me check.\n<tool_call>\n<function=get_weather>\n",
        "Hello\n<tool_call>\n<function=get_weather>\n<parameter=city>P",
        "<tool_call>\n<function",
        '<tool_call>\n{"name',
        "<tool_call>\n<function=unknown>\n<parameter=city>Paris",
        "Use `code\n<tool_call>\n<function=get_weather\n` here.",
        'Use <tool_call> {"name":"get_weather"} as a template.',
        'Use <tool_call> {json} for examples.\nHe is 6" tall.\nLet me check.\n<tool_call>\n<function=get_weather',
        '<tool_call>{"name":"get_weather","arguments":{"city":"</tool_call>',
        '<tool_call>{"name":"get_weather","arguments":{"city":"<tool_call>',
        "<tool_call>{",
        "<think>I should use <tool_call>\n<function=get_weather> to look this up.</think>Let me check.\n<tool_call>\n<function=get_weather",
        "Use `code` here.\n<tool_call>\n<function=get_weather",
        "<tool_call>\n<function=get_weather></function>",
    ]
    rows = []
    for parser_name in PARSERS:
        for raw in reproducers:
            # Fragmented XML leakage for the Qwen parser is a separate issue.
            sizes = [len(raw)]
            if parser_name == "hermes" and raw in (
                reproducers[0],
                reproducers[5],
                reproducers[6],
                reproducers[9],
            ):
                sizes.append(1)
            for size in sizes:
                parser = ToolParserManager.get_tool_parser(parser_name)(None)
                processor = StreamingPostProcessor(
                    _make_cfg(
                        enable_auto_tool_choice=True,
                        tool_call_parser=parser_name,
                        tool_parser_instance=parser,
                        reasoning_parser=Qwen3ReasoningParser(None)
                        if "<think>" in raw
                        else None,
                    ),
                    tools_requested=True,
                    request={"tools": TOOLS, "tool_choice": "auto"},
                )
                processor.reset()
                events = []
                for i in range(0, len(raw), size):
                    events.extend(
                        processor.process_chunk(_make_output(raw[i : i + size]))
                    )
                events.extend(processor.finalize())
                rows.append(stable([asdict(event) for event in events]))
    return rows


def test_streaming_unchanged_against_base():
    """Unrelated streaming definitions stay identical to origin/main."""
    import ast
    import os
    import subprocess
    import sys
    from pathlib import Path

    import rapid_mlx

    root = Path(rapid_mlx.__file__).resolve().parents[1]
    for name in ("hermes_tool_parser", "qwen3coder_tool_parser"):
        relative = f"rapid_mlx/tool_parsers/{name}.py"
        try:
            base = subprocess.check_output(
                ["git", "show", f"origin/main:{relative}"],
                cwd=root,
                text=True,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            pytest.skip(f"Git unavailable for streaming base comparison: {exc}")
        except subprocess.CalledProcessError as exc:
            pytest.skip(
                f"Cannot resolve origin/main:{relative} for streaming base comparison "
                f"(requires a Git checkout with the base ref/path): {exc.stderr.strip()}"
            )

        def existing_definitions(source):
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    node.body = [
                        method
                        for method in node.body
                        if not (
                            isinstance(method, ast.FunctionDef)
                            and method.name
                            in {
                                "strip_trailing_tool_envelope",
                                # #4714 intentionally changes these scans;
                                # dedicated fence tests pin their output.
                                "_first_opener_pos",
                                "_new_opener_position",
                                "_function_start_positions",
                                "_content_without_admitted_calls",
                                "__init__",
                                "_reset_streaming_state",
                                "inside_fence",
                                "skip_fence_span",
                                "_fence_state",
                                "_restore_fence_state",
                                "_feed_fence",
                                "_feed_fence_plain",
                                "_finish_fence_line",
                                "_fence_result",
                                "mask_markdown_fenced_regions",
                                "flush_held_content",
                                "_streaming_step_once",
                                # #4788 treats CR and CRLF as line boundaries.
                                "_feed",
                                "inside",
                            }
                        )
                    ]
            # The opt-in buffered Markdown query leaves the original streaming
            # methods unchanged. Normalize its new dispatch before comparison.
            for node in tree.body:
                if (
                    isinstance(node, ast.ClassDef)
                    and node.name == "_MarkdownCodeTracker"
                ):
                    node.body = [
                        m
                        for m in node.body
                        if not (
                            isinstance(m, ast.FunctionDef)
                            and m.name == "_inside_buffered"
                        )
                    ]
                    for method in node.body:
                        if (
                            isinstance(method, ast.FunctionDef)
                            and method.name == "inside"
                            and method.args.kwonlyargs
                        ):
                            method.args.kwonlyargs = []
                            method.args.kw_defaults = []
                            method.body = method.body[1:]
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "_strip_trailing_tool_envelope"
                )
            ]
            return ast.dump(tree, include_attributes=False)

        assert existing_definitions(
            (root / relative).read_text()
        ) == existing_definitions(base)

    # Load the same test helper with base production modules on PYTHONPATH.
    base_root = os.environ.get("TRUNCATED_TOOL_BASE")
    if base_root is None:
        return  # Definitions prove parity in CI; opt in to cross-worktree events.
    script = "import json,runpy,sys; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['_stream_snapshots']()))"
    output = subprocess.check_output(
        [sys.executable, "-c", script, str(Path(__file__).resolve())],
        cwd=base_root,
        env={**os.environ, "PYTHONPATH": base_root, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
    )
    assert _stream_snapshots() == json.loads(output)


def test_xml_without_declared_tools(client):
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    assert (
        Qwen3CoderToolParser.strip_trailing_tool_envelope("<tool_call>", {"tools": []})
        == "<tool_call>"
    )


@pytest.mark.parametrize("route", ROUTES)
def test_hermes_complete_json(client, route):
    client[1].tool_call_parser = "hermes"
    raw = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
    content, calls, status = _reply(client, route, raw, finish_reason="stop")
    assert content == ""
    assert len(calls) == 1
    assert (
        status
        == {"chat": "tool_calls", "messages": "tool_use", "responses": "completed"}[
            route
        ]
    )


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "payload,hidden",
    [
        ('{"name" means the function identifier', False),
        ('{ \n "name" means the function identifier', False),
        ('{"name":', True),
        ('{ \n "name" \t :', True),
        ('{"name"', True),
        ('{ \n "name" \t', True),
        ("{", True),
        ("{ \n", True),
        ('{ "', True),
        ('{ "n', True),
        ('{ "na', True),
        ('{ "nam', True),
        ('{ "name', True),
    ],
)
def test_json_member_prefix_boundary(client, parser, route, payload, hidden):
    client[1].tool_call_parser = parser
    prefix = "Let me check.\n"
    raw = prefix + "<tool_call>" + payload
    content, calls, _ = _reply(client, route, raw)
    assert content == (prefix.strip() if hidden else raw)
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup_closed_wrappers(client, parser, record_property):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    unit = "long narration " * 6 + "<tool_call>{}</tool_call>"
    timings = []
    for count in (2150, 8600):
        raw = unit * count + "<tool_call>"
        expected = raw[: -len("<tool_call>")]
        samples = []
        for _ in range(3):
            start = perf_counter()
            cleaned = cls.strip_trailing_tool_envelope(raw)
            samples.append(perf_counter() - start)
            assert cleaned == expected
        timings.append(min(samples))
    record_property("closed_wrappers_n_seconds", timings[0])
    record_property("closed_wrappers_4n_seconds", timings[1])
    assert timings[1] < 10 * timings[0]
    assert timings[1] < 1.0


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
def test_prose_opener_before_real_call(client, parser, route):
    client[1].tool_call_parser = parser
    narration = "Use the <tool_call> tag.\nLet me check.\n"
    raw = narration + "<tool_call>\n<function=get_weather"
    expected = _reply(client, route, narration)[0]
    assert _reply(client, route, raw)[0] == expected


@pytest.mark.parametrize("parser", PARSERS)
def test_exact_whitespace_streaming_parity(client, parser):
    client[1].tool_call_parser = parser
    narration = "Let me check.\n"
    raw = narration + "<tool_call>\n<function=get_weather"
    buffered = _reply(client, "chat", raw)[0]
    streamed = _reply(client, "chat", raw, stream=True)[0]
    # Buffered routes normalize ordinary text too. Check exact parser parity
    # before that normalization, and the route's ordinary-text policy after it.
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cleaned = ToolParserManager.get_tool_parser(parser).strip_trailing_tool_envelope(
        raw
    )
    if parser == "qwen3_coder_xml":
        assert cleaned == streamed == narration
    else:
        assert (
            cleaned
            == _reply(client, "chat", raw, stream=True, chunks=list(raw))[0]
            == narration
        )
    assert buffered == _reply(client, "chat", narration)[0]


@pytest.mark.parametrize("content", [None, "", "  \n"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_reasoning_cleanup_preserves_empty_content(client, content, finish_reason):
    from rapid_mlx.service.helpers import _clean_final_content_tool_envelope

    assert (
        _clean_final_content_tool_envelope(content, finish_reason=finish_reason)
        == content
    )


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("truncated", [False, True])
def test_reasoning_cleanup_preserves_whitespace(
    client, parser, route, finish_reason, truncated, monkeypatch
):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    cfg = client[1]
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser(None)
    visible = "  answer  \n"
    if truncated:
        visible += "<tool_call>\n<function=get_weather"
    # Isolate the finalized visible lane from the earlier, pre-existing
    # sanitizer normalization. Cleanup must preserve whatever reaches it.
    module = {"chat": "chat", "messages": "anthropic", "responses": "responses"}[route]
    monkeypatch.setattr(
        f"rapid_mlx.routes.{module}._apply_reasoning_cutoff_notice",
        lambda *args, **kwargs: visible,
    )
    content, calls, _ = _reply(client, route, visible, finish_reason=finish_reason)
    expected = "  answer" if truncated and finish_reason == "length" else visible
    assert content == expected
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("inside_reasoning", [False, True, "combined"])
@pytest.mark.parametrize("rescue_enabled", [False, True])
def test_reasoning_boundaries(
    client, parser, route, inside_reasoning, rescue_enabled, monkeypatch
):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

    monkeypatch.setenv("RAPID_MLX_REASONING_RESCUE", "on" if rescue_enabled else "off")
    cfg = client[1]
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    raw = (
        "<think>I should use <tool_call>\n"
        "<function=get_weather> to look this up.</think>Final answer."
        if inside_reasoning
        else "<think>Let me think.</think><tool_call>\n<function=get_weather"
    )
    if inside_reasoning == "combined":
        raw = raw.replace(
            "Final answer.", "Let me check.\n<tool_call>\n<function=get_weather"
        )
    content = _reply(client, route, raw)[0]
    if inside_reasoning == "combined":
        assert content.endswith("Let me check.")
        assert not content.endswith("<function=get_weather")
    else:
        assert content.endswith("Final answer.") if inside_reasoning else content == ""
    data = _reply(client, route, raw, with_reasoning=True)
    if route == "chat":
        reasoning = data["choices"][0]["message"].get("reasoning_content")
    elif route == "messages":
        reasoning = "".join(
            b["thinking"] for b in data["content"] if b["type"] == "thinking"
        )
    else:
        reasoning = "".join(
            c["text"]
            for b in data["output"]
            if b["type"] == "reasoning"
            for c in b["summary"]
        )
    if not inside_reasoning:
        assert reasoning == "Let me think."
    if route == "chat" and not inside_reasoning and not rescue_enabled:
        assert content == _reply(client, route, raw, stream=True)[0]


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup_prose_then_real_call(client, parser):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    narration = "Use <tool_call> for a call.\n" * 20000
    raw = narration + "<tool_call>\n<function=get_weather"
    start = perf_counter()
    assert cls.strip_trailing_tool_envelope(raw) == narration
    assert perf_counter() - start < 0.2


@pytest.mark.parametrize("parser", PARSERS)
def test_linear_cleanup_large_json_string(client, parser):
    from time import perf_counter

    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    narration = "Use <tool_call> for a call.\n" * 20000
    raw = (
        narration
        + '<tool_call>{"name": "write_file", "arguments": {"content": "'
        + "x" * 500000
    )
    start = perf_counter()
    assert cls.strip_trailing_tool_envelope(raw) == narration
    assert perf_counter() - start < 0.2


@pytest.mark.parametrize("parser", PARSERS)
def test_reasoning_boundary_guard_without_source(client, parser):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.service.helpers import _strip_trailing_tool_envelope

    client[1].tool_call_parser = parser
    client[1].reasoning_parser = Qwen3ReasoningParser(None)
    raw = "<tool_call>\n<function=get_weather</think>Final answer."
    assert _strip_trailing_tool_envelope(raw, finish_reason="length") == raw
    assert (
        _strip_trailing_tool_envelope(
            "<tool_call>\n<function=get_weather", finish_reason="length"
        )
        == ""
    )


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
def test_inches_after_json_example(client, parser, route):
    client[1].tool_call_parser = parser
    prefix = 'Use <tool_call> {json} for examples.\nHe is 6" tall.\nLet me check.\n'
    raw = prefix + "<tool_call>\n<function=get_weather"
    assert _reply(client, route, raw)[0] == _reply(client, route, prefix)[0]


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("reasoning", [False, True])
def test_literal_opener_in_truncated_json(client, parser, route, reasoning):
    client[1].tool_call_parser = parser
    if reasoning:
        from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        client[1].reasoning_parser_name = "qwen3"
        client[1].reasoning_parser = Qwen3ReasoningParser(None)
    raw = 'Narration.\n<tool_call>{"name":"get_weather","arguments":{"city":"see <tool_call>'
    assert _reply(client, route, raw)[0] == "Narration."


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize(
    "raw,hidden",
    [
        ('<tool_call>{"name":"get_weather"} as a template.', False),
        ('<tool_call>{"name":"get_weather"</tool_call>', False),
        ('<tool_call>{"name":"get_weather","arguments":{"city":"a}b', True),
        ('<tool_call>{"name":"get_weather","arguments":{"city":"a\\"b', True),
        ("<tool_call>\n<function=get_weather></function>", True),
        ("<tool_call>\n<function=get_weather></function></tool_call>", False),
    ],
)
def test_last_payload_visibility(client, parser, raw, hidden):
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cls = ToolParserManager.get_tool_parser(parser)
    assert cls.strip_trailing_tool_envelope(raw) == ("" if hidden else raw)


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize(
    "prefix,kept",
    [
        ("Use `code` here.\n", False),
        ("Use ``code\nspanning lines`` here.\n", False),
        ("Use ``code` still open\n", True),
        ("```xml\n```", True),
        ("```xml\n```\n", False),
    ],
)
def test_code_boundary_before_last_opener(client, parser, prefix, kept):
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    raw = prefix + "<tool_call>\n<function=get_weather"
    cls = ToolParserManager.get_tool_parser(parser)
    assert cls.strip_trailing_tool_envelope(raw) == (raw if kept else prefix)


@pytest.mark.parametrize("stream", [False, True])
def test_closed_inline_code_before_rejected_call(client, stream):
    # A closed code span must not protect a later real undeclared call.
    client[1].tool_call_parser = "qwen3_coder_xml"
    prefix = "Use `code` here. "
    raw = prefix + "<tool_call>\n<function=unknown></function>\n</tool_call>"
    content, calls, _ = _reply(client, "chat", raw, stream=stream)
    assert content.strip() == prefix.strip()
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "suffix",
    [
        "<tool_call>\n<function=write_file>\n<parameter=path>README",
        '<tool_call>{"name":"write_file","arguments":{"path":"README',
    ],
    ids=["xml", "hermes-json"],
)
@pytest.mark.parametrize(
    "example,kept",
    [
        (r"\`", False),
        (r"\\\`", False),
        (r"\\`", True),
        (r"\\\\`", True),
        (r"`code\`", False),
        (r"``code\``", False),
    ],
    ids=[
        "one-slash",
        "three-slashes",
        "two-slashes",
        "four-slashes",
        "literal-slash-in-span",
        "literal-slash-in-double-span",
    ],
)
def test_escaped_backtick_before_truncated_call(
    client, parser, route, suffix, example, kept
):
    client[1].tool_call_parser = parser
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    prefix = f"Escape a backtick as {example} in Markdown.\n"
    prefix += "I will update the documentation.\n"
    raw = prefix + suffix
    content, calls, _ = _reply(client, route, raw, tools=tools)
    assert content == (raw if kept else prefix.rstrip())
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("reasoning", [False, True])
def test_tool_choice_none_preserves_trailing_literal(client, parser, route, reasoning):
    from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    cfg = client[1]
    cfg.tool_call_parser = parser
    visible = "Literal example: <tool_call>"
    cls = ToolParserManager.get_tool_parser(parser)
    assert (
        cls.strip_trailing_tool_envelope(
            visible, {"tools": TOOLS, "tool_choice": "none"}
        )
        == visible
    )
    raw = visible
    if reasoning:
        cfg.reasoning_parser_name = "qwen3"
        cfg.reasoning_parser = Qwen3ReasoningParser(None)
        raw = "<think>Let me think.</think>" + visible
    content, calls, _ = _reply(client, route, raw, choice="none")
    assert content == visible
    assert not calls


@pytest.mark.parametrize("parser", PARSERS)
@pytest.mark.parametrize(
    "prefix,kept",
    [
        ("`x <tool_call>`</tool_call>\n", False),
        ("x <tool_call>`</tool_call>\n", True),
    ],
)
def test_markdown_tracks_closed_wrapper_contents(client, parser, prefix, kept):
    from rapid_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    raw = prefix + "<tool_call>"
    cls = ToolParserManager.get_tool_parser(parser)
    assert cls.strip_trailing_tool_envelope(raw) == (raw if kept else prefix)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_forced_json_stop_4663(client, route, reasoning, stream):
    if reasoning:
        from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

        client[1].reasoning_parser_name = "qwen3"
        client[1].reasoning_parser = Qwen3ReasoningParser(None)
    client[1].tool_call_parser = "hermes"
    raw = '<tool_call> {"name": "get_weather", "'
    content, calls, _ = _reply(
        client,
        route,
        raw,
        finish_reason="stop",
        stream=stream,
        choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert calls
    assert content == ""


@pytest.mark.parametrize("chunk_size", [1, 7, 200])
def test_forced_json_stop_stream_4663(client, chunk_size):
    client[1].tool_call_parser = "hermes"
    raw = '<tool_call> {"name": "get_weather", "'
    content, calls, finish = _reply(
        client,
        "chat",
        raw,
        stream=True,
        finish_reason="stop",
        chunks=[raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)],
        choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert calls
    assert finish == "tool_calls"
    assert content == ""


@pytest.mark.parametrize(
    "raw",
    [
        '<tool_call> {"name": "get_weather", "',
        'Narration.\n<tool_call>{"name":"get_wea',
        '<tool_call>{"name":"get_weather","arguments":{"text":"see <tool_call>',
    ],
)
def test_forced_json_cleanup_evidence_4663(raw):
    from rapid_mlx.routes.chat import _strip_truncated_forced_xml_call

    assert _strip_truncated_forced_xml_call(raw, {"get_weather"}) == (
        "Narration." if raw.startswith("Narration.") else ""
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain prose", "plain prose"),
        (
            'Use <tool_call> in prose.\n<tool_call>{"name":"get_weather","arguments":{"x":"a\\\\b\\" <tool_call>',
            "Use <tool_call> in prose.\n",
        ),
        (
            '<tool_call>{"name":"get_weather"}\n<tool_call>{"name":"get_weather","arguments":',
            '<tool_call>{"name":"get_weather"}\n',
        ),
        (
            '<tool_call>{"name":"first",\n<tool_call>{"name":"get_weather","arguments":',
            '<tool_call>{"name":"first",\n',
        ),
        (
            '<tool_call>{"name":"get_weather"</tool_call>\n<tool_call>',
            '<tool_call>{"name":"get_weather"</tool_call>\n',
        ),
        ('<tool_call>{"name":"get_weather","arguments":{"x":"<tool_call>"}}', None),
    ],
)
def test_json_opener_state(raw, expected):
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import (
        _strip_trailing_tool_envelope,
    )

    assert _strip_trailing_tool_envelope(raw) == (raw if expected is None else expected)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "Narration.",
        "<tool_call>",
        "Use <tool_call> in prose.",
        '<tool_call>{"name":"other","arguments":',
        '<tool_call>{"name":"get_weather_extra","arguments":',
        '<tool_call>{"name":"","arguments":',
        '<tool_call>{"name":"other',
        '<tool_call>{"name":"get_weather"}',
        '<tool_call>{"name":"get_weather"</tool_call>',
        '```json\n<tool_call>{"name":"get_weather","arguments":\n```',
        'Example: `<tool_call>{"name":"get_weather","arguments":`',
    ],
)
def test_forced_json_cleanup_preserves_nonmatching_4663(raw):
    from rapid_mlx.routes.chat import _strip_truncated_forced_xml_call

    assert _strip_truncated_forced_xml_call(raw, {"get_weather"}) == (raw or "")


@pytest.mark.parametrize("terminal_lane", ["finish", "finalize"])
@pytest.mark.parametrize("payload", ['{"name":"get_weather", "', '{"name":"get_wea'])
def test_forced_terminal_json_replay_4663(client, monkeypatch, terminal_lane, payload):
    from dataclasses import replace

    from rapid_mlx.domain.events import StreamEvent
    from rapid_mlx.service.postprocessor import StreamingPostProcessor

    raw = "<tool_call>" + payload
    original_process = StreamingPostProcessor.process_chunk
    original_finalize = StreamingPostProcessor.finalize

    def process(self, output):
        for event in original_process(self, output):
            if terminal_lane == "finish" and event.type == "finish":
                event = replace(event, content=raw)
            yield event

    def finalize(self):
        yield from original_finalize(self)
        if terminal_lane == "finalize":
            yield StreamEvent(type="content", content=raw)

    # Exercise terminal held content independently of parser chunk buffering.
    monkeypatch.setattr(StreamingPostProcessor, "process_chunk", process)
    monkeypatch.setattr(StreamingPostProcessor, "finalize", finalize)
    client[1].tool_call_parser = "hermes"
    content, calls, finish = _reply(
        client,
        "chat",
        "",
        stream=True,
        finish_reason="stop",
        choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert calls
    assert finish == "tool_calls"
    assert content == ""
