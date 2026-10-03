# SPDX-License-Identifier: Apache-2.0
"""Cross-protocol streaming contract for chat, Responses and Anthropic routes.

``/v1/chat/completions``, ``/v1/responses`` and ``/v1/messages`` each keep their
own streaming state machine for splitting reasoning, visible text and tool
calls out of the engine's token stream. They are maintained separately, so a
fix to one can silently miss the others.

This test feeds the same scripted ``stream_chat`` output through all three
routes and asserts they agree on what the client sees: the visible text, the
reasoning text, the parsed tool calls, and exactly one terminal event with the
protocol's own finish reason. Every surface is also pinned to the expected
values, so a regression shared by all three still fails.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser

SURFACES = ("chat", "responses", "anthropic")

_WEATHER_PARAMS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


@dataclass(frozen=True)
class _Scenario:
    deltas: tuple[str, ...]
    engine_finish: str
    content: str
    reasoning: str
    calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    tools: bool = False


def _call(city: str) -> str:
    return (
        '<tool_call>{"name":"get_weather","arguments":{"city":"' + city + '"}}'
        "</tool_call>"
    )


SCENARIOS = {
    "text": _Scenario(
        deltas=("<think>", "Plan the ", "answer.", "</think>", "Hello ", "world."),
        engine_finish="stop",
        content="Hello world.",
        reasoning="Plan the answer.",
    ),
    "no_reasoning": _Scenario(
        deltas=("Just ", "an answer."),
        engine_finish="stop",
        content="Just an answer.",
        reasoning="",
    ),
    # Tags split across chunk boundaries must not leak partial markup.
    "split_tags": _Scenario(
        deltas=("<thi", "nk>Plan", " it.</th", "ink>Hel", "lo."),
        engine_finish="stop",
        content="Hello.",
        reasoning="Plan it.",
    ),
    "length": _Scenario(
        deltas=("<think>", "Short.", "</think>", "Cut off"),
        engine_finish="length",
        content="Cut off",
        reasoning="Short.",
    ),
    "tool": _Scenario(
        deltas=(
            "<think>",
            "Need weather.",
            "</think>",
            '<tool_call>{"name":"get_weather",',
            '"arguments":{"city":"Tokyo"}}</tool_call>',
        ),
        engine_finish="stop",
        content="",
        reasoning="Need weather.",
        calls=(("get_weather", {"city": "Tokyo"}),),
        tools=True,
    ),
    "two_tools": _Scenario(
        deltas=("<think>x</think>", _call("Tokyo"), _call("Paris")),
        engine_finish="stop",
        content="",
        reasoning="x",
        calls=(
            ("get_weather", {"city": "Tokyo"}),
            ("get_weather", {"city": "Paris"}),
        ),
        tools=True,
    ),
    "text_then_tool": _Scenario(
        deltas=(
            "<think>x</think>",
            "Checking now. ",
            '<tool_call>{"name":"get_weather",',
            '"arguments":{"city":"Oslo"}}</tool_call>',
        ),
        engine_finish="stop",
        content="Checking now. ",
        reasoning="x",
        calls=(("get_weather", {"city": "Oslo"}),),
        tools=True,
    ),
}

# Each protocol's terminal signal for an engine finish (stop / tool call / length).
# Responses always ends with ``response.completed`` (Codex treats a missing one
# as a hard failure); a length cutoff is reported through ``status`` instead.
_TERMINAL = {
    "chat": {"stop": "stop", "tool": "tool_calls", "length": "length"},
    "responses": {
        "stop": ("response.completed", "completed", None),
        "tool": ("response.completed", "completed", None),
        "length": ("response.completed", "incomplete", "max_output_tokens"),
    },
    "anthropic": {"stop": "end_turn", "tool": "tool_use", "length": "max_tokens"},
}

_MARKUP = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<thi", "</th")


class _ScriptedEngine:
    preserve_native_tool_format = False
    supports_guided_generation = False
    tokenizer = None
    is_mllm = False

    def __init__(self, scenario: _Scenario) -> None:
        self.scenario = scenario

    def build_prompt(self, messages, **_kwargs):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        deltas = self.scenario.deltas
        text = ""
        for i, delta in enumerate(deltas):
            text += delta
            last = i == len(deltas) - 1
            yield GenerationOutput(
                text=text,
                new_text=delta,
                tokens=[i],
                prompt_tokens=3,
                completion_tokens=i + 1,
                finished=last,
                finish_reason=self.scenario.engine_finish if last else None,
            )


@pytest.fixture(autouse=True)
def _restore_config() -> Iterator[None]:
    yield
    reset_config()


def _client(surface: str, scenario: _Scenario) -> TestClient:
    # Mirrors ``serve --reasoning-parser qwen3 --enable-auto-tool-choice
    # --tool-call-parser hermes``.
    cfg = reset_config()
    cfg.engine = _ScriptedEngine(scenario)
    cfg.model_name = "stream-contract"
    cfg.model_path = "stream-contract"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.reasoning_parser = Qwen3ReasoningParser(tokenizer=None)
    cfg.reasoning_parser_name = "qwen3"
    cfg.tool_call_parser = "hermes"
    cfg.enable_auto_tool_choice = True

    if surface == "chat":
        from rapid_mlx.routes.chat import router
    elif surface == "responses":
        from rapid_mlx.routes.responses import router
    else:
        from rapid_mlx.routes.anthropic import router

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    return TestClient(app)


def _request(surface: str, *, tools: bool) -> tuple[str, dict[str, Any]]:
    if surface == "chat":
        payload: dict[str, Any] = {
            "model": "stream-contract",
            "stream": True,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather?"}],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": _WEATHER_PARAMS},
                }
            ]
        return "/v1/chat/completions", payload
    if surface == "responses":
        payload = {
            "model": "stream-contract",
            "stream": True,
            "max_output_tokens": 64,
            "input": "weather?",
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": _WEATHER_PARAMS,
                }
            ]
        return "/v1/responses", payload
    payload = {
        "model": "stream-contract",
        "stream": True,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "weather?"}],
    }
    if tools:
        payload["tools"] = [{"name": "get_weather", "input_schema": _WEATHER_PARAMS}]
    return "/v1/messages", payload


def _sse_data(lines: list[str]) -> list[Any]:
    events: list[Any] = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        body = line[len("data: ") :]
        events.append(body if body == "[DONE]" else json.loads(body))
    return events


@dataclass
class _Observed:
    content: str
    reasoning: str
    calls: list[tuple[str, dict[str, Any]]]
    terminals: list[Any]


def _observe_chat(events: list[Any]) -> _Observed:
    content = reasoning = ""
    tools: dict[int, dict[str, str]] = {}
    terminals: list[Any] = []
    done = 0
    for event in events:
        if event == "[DONE]":
            done += 1
            continue
        for choice in event["choices"]:
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                slot = tools.setdefault(call["index"], {"name": "", "arguments": ""})
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                terminals.append(choice["finish_reason"])
    assert done == 1, "chat stream must end with exactly one [DONE]"
    calls = [(t["name"], json.loads(t["arguments"])) for _, t in sorted(tools.items())]
    return _Observed(content, reasoning, calls, terminals)


def _observe_responses(events: list[Any]) -> _Observed:
    content = reasoning = ""
    calls: list[tuple[str, dict[str, Any]]] = []
    terminals: list[Any] = []
    for event in events:
        kind = event["type"]
        if kind == "response.output_text.delta":
            content += event["delta"]
        elif kind == "response.reasoning_summary_text.delta":
            reasoning += event["delta"]
        elif (
            kind == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        ):
            item = event["item"]
            calls.append((item["name"], json.loads(item["arguments"])))
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            response = event["response"]
            details = response.get("incomplete_details") or {}
            terminals.append((kind, response["status"], details.get("reason")))
    return _Observed(content, reasoning, calls, terminals)


def _observe_anthropic(events: list[Any]) -> _Observed:
    content = reasoning = ""
    blocks: dict[int, dict[str, Any]] = {}
    terminals: list[Any] = []
    stops = 0
    for event in events:
        kind = event["type"]
        if kind == "content_block_start":
            blocks[event["index"]] = {**event["content_block"], "_json": ""}
        elif kind == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "text_delta":
                content += delta["text"]
            elif delta["type"] == "thinking_delta":
                reasoning += delta["thinking"]
            elif delta["type"] == "input_json_delta":
                blocks[event["index"]]["_json"] += delta["partial_json"]
        elif kind == "message_delta":
            terminals.append(event["delta"].get("stop_reason"))
        elif kind == "message_stop":
            stops += 1
    assert stops == 1, "anthropic stream must end with exactly one message_stop"
    calls = [
        (block["name"], json.loads(block["_json"] or "{}"))
        for _, block in sorted(blocks.items())
        if block["type"] == "tool_use"
    ]
    return _Observed(content, reasoning, calls, terminals)


_OBSERVERS = {
    "chat": _observe_chat,
    "responses": _observe_responses,
    "anthropic": _observe_anthropic,
}


def _stream(surface: str, scenario: _Scenario) -> _Observed:
    client = _client(surface, scenario)
    path, payload = _request(surface, tools=scenario.tools)
    with client.stream("POST", path, json=payload) as response:
        assert response.status_code == 200, response.read()
        lines = list(response.iter_lines())
    return _OBSERVERS[surface](_sse_data(lines))


def _terminal_key(scenario: _Scenario) -> str:
    if scenario.calls:
        return "tool"
    return "length" if scenario.engine_finish == "length" else "stop"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_streaming_routes_agree_on_client_visible_output(name: str) -> None:
    scenario = SCENARIOS[name]
    observed = {surface: _stream(surface, scenario) for surface in SURFACES}

    for surface, seen in observed.items():
        assert seen.content == scenario.content, surface
        assert seen.reasoning == scenario.reasoning, surface
        assert seen.calls == list(scenario.calls), surface
        for markup in _MARKUP:
            assert markup not in seen.content, (surface, markup)
            assert markup not in seen.reasoning, (surface, markup)


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("surface", SURFACES)
def test_streaming_routes_emit_one_mapped_terminal_event(
    surface: str, name: str
) -> None:
    scenario = SCENARIOS[name]
    seen = _stream(surface, scenario)

    assert seen.terminals == [_TERMINAL[surface][_terminal_key(scenario)]]
