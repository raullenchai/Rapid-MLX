# SPDX-License-Identifier: Apache-2.0
"""Implicit Flint reasoning stays private across API surfaces and tool turns."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from rapid_mlx.routes import chat, responses
from rapid_mlx.tool_parsers.glm47_tool_parser import Glm47ToolParser
from tests.test_rejected_tool_diagnostics import TOOLS
from tests.test_rejected_tool_diagnostics import Engine as BaseEngine

TEMPLATE = (Path(__file__).parent / "fixtures/flint_chat_template.jinja").read_text()
CALL = "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"


class Engine(BaseEngine):
    step = 1

    async def stream_chat(self, **kwargs):
        for i in range(0, len(self.raw), self.step):
            yield GenerationOutput(
                text=self.raw[: i + self.step],
                new_text=self.raw[i : i + self.step],
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=False,
            )
        yield GenerationOutput(
            text=self.raw,
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=len(self.raw),
        )


@pytest.mark.parametrize("step", [1, 7, 10000])
@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("later", [False, True])
@pytest.mark.parametrize(
    "raw",
    [
        CALL,
        "Private thought.</think>" + CALL[:-12],
        "Private thought.</think>Done.",
        "Private thought.</think>Literal <arg_",
        "Private thought.</think>" + CALL,
        "Private thought.<tool_call>run_shell_command</think><arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
        "Private thought.</think></arg_value><arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
    ],
)
def test_flint_routes(surface, stream, later, raw, step):
    cfg = reset_config()
    cfg.model_name = "flint-40b-a2.5b-4bit"
    cfg.no_thinking = False
    cfg.reasoning_parser_name = "qwen3"
    cfg.reasoning_parser = Qwen3ReasoningParser()
    cfg.tool_call_parser = "glm47"
    cfg.enable_auto_tool_choice = True
    cfg.engine = Engine(raw)
    cfg.engine.step = step
    cfg.engine.tokenizer = SimpleNamespace(chat_template=TEMPLATE)
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(responses.router)
    history = [{"role": "user", "content": "Build life.py and run it."}]
    if later:
        history += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "run_shell_command", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": "ok"},
        ]
    body = dict(
        model=cfg.model_name,
        stream=stream,
        tools=TOOLS,
        chat_template_kwargs={"enable_thinking": True},
    )
    if surface == "chat":
        body["messages"] = history
        url = "/v1/chat/completions"
    else:
        body["input"] = history[:1]
        if later:
            body["input"] += [
                {
                    "type": "function_call",
                    "call_id": "call_old",
                    "name": "run_shell_command",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_old", "output": "ok"},
            ]
        body["tools"] = [{"type": "function", **TOOLS[0]["function"]}]
        url = "/v1/responses"
    try:
        response = TestClient(app).post(url, json=body)
        assert response.status_code == 200, response.text
        packets = (
            [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: {")
            ]
            if stream
            else [response.json()]
        )
        if surface == "chat":
            messages = [
                p["choices"][0].get("delta", p["choices"][0].get("message", {}))
                for p in packets
                if p.get("choices")
            ]
            visible = "".join(m.get("content") or "" for m in messages)
            calls = [c for m in messages for c in m.get("tool_calls", [])]
        elif stream:
            visible = "".join(
                p.get("delta", "")
                for p in packets
                if p.get("type") == "response.output_text.delta"
            )
            calls = [
                p["item"]
                for p in packets
                if p.get("type") == "response.output_item.done"
                and p.get("item", {}).get("type") == "function_call"
            ]
        else:
            visible = "".join(
                c["text"]
                for i in packets[0]["output"]
                if i["type"] == "message"
                for c in i["content"]
            )
            calls = [i for i in packets[0]["output"] if i["type"] == "function_call"]
        assert "Private thought" not in visible
        assert not any(
            tag in visible
            for tag in [
                "</think>",
                "<tool_call>",
                "<arg_key>",
                "<arg_value>",
                "</tool_call>",
            ]
        )
        if "<tool_call>" in raw and raw.endswith("</tool_call>"):
            assert calls
        elif raw.endswith("Done."):
            assert visible == "Done."
        elif raw.endswith("Literal <arg_"):
            assert visible == "Literal <arg_"
    finally:
        reset_config()


@pytest.mark.parametrize(
    "text", ["Secret.</think>Answer", "<think>Secret.</think>Answer"]
)
def test_qwen3_implicit_explicit_parity_at_every_split(text):
    assert Qwen3ReasoningParser().extract_reasoning(text) == ("Secret.", "Answer")
    for split in range(1, len(text)):
        parser = Qwen3ReasoningParser()
        messages = [
            parser.extract_reasoning_streaming("", text[:split], text[:split]),
            parser.extract_reasoning_streaming(text[:split], text, text[split:]),
        ]
        assert "".join(m.content or "" for m in messages if m) == "Answer"
        assert "Secret." in "".join(m.reasoning or "" for m in messages if m)


def test_qwen3_thinking_off_plain_answer_unchanged():
    assert Qwen3ReasoningParser().extract_reasoning(
        "Answer", enable_thinking=False
    ) == (None, "Answer")


@pytest.mark.parametrize(
    "raw",
    [
        "<tool_call><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
        "</arg_value><arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
        "<arg_key>x</arg_key><arg_value>1",
        "<tool_call>unknown</tool_call>",
    ],
)
def test_glm_malformed_markup_is_not_content(raw):
    parser = Glm47ToolParser()
    result = parser.extract_tool_calls(raw, {"tools": TOOLS})
    assert not result.tools_called
    assert not result.content
    for split in range(1, len(raw)):
        parser.reset()
        messages = [
            parser.extract_tool_calls_streaming(
                "", raw[:split], raw[:split], request={"tools": TOOLS}
            ),
            parser.extract_tool_calls_streaming(
                raw[:split], raw, raw[split:], request={"tools": TOOLS}
            ),
        ]
        visible = "".join(m.get("content", "") for m in messages if m)
        visible += parser.flush_held_content(raw)
        assert not visible


def test_responses_finalize_preserves_plain_text():
    test_flint_routes("responses", True, False, "Done.", 1)


def test_glm_interrupted_name_requires_arguments():
    parser = Glm47ToolParser()
    raw = "<tool_call>run_shell_command</think></tool_call>"
    assert not parser.extract_tool_calls(raw, {"tools": TOOLS}).tools_called
    assert parser.flush_held_content("") == ""


@pytest.mark.parametrize(
    "raw,expected",
    [("Literal <arg_", "Literal <arg_"), ("<arg_key>x", ""), ("Plain <", "Plain <")],
)
def test_glm_stream_flush(raw, expected):
    parser = Glm47ToolParser()
    parser.reset()
    chunks = []
    for i, char in enumerate(raw):
        message = parser.extract_tool_calls_streaming(raw[:i], raw[: i + 1], char)
        if message:
            chunks.append(message.get("content", ""))
    chunks.append(parser.flush_held_content(raw))
    assert "".join(chunks) == expected
    assert parser.flush_held_content(raw) == ""


def test_responses_finalize_flushes_literal_prefix():
    test_flint_routes("responses", True, False, "Literal <arg_", 1)


@pytest.mark.parametrize("name", ["run_shell_command", "unknown"])
def test_glm_repair_preserves_implicit_reasoning_boundary(name):
    raw = f"Private thought.<tool_call>{name}</think><arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
    result = Glm47ToolParser().extract_tool_calls(raw, {"tools": TOOLS})
    assert not result.content
    assert result.tools_called == (name == "run_shell_command")
