# SPDX-License-Identifier: Apache-2.0
"""Paired-surface smoke: /v1/chat/completions and /v1/messages must
agree token-for-token on the same engine stream.

Closes the surface-divergence gap that produced #288 and #289: the two
routes share the engine and the reasoning parser config but diverge at
the streaming-think router. A reasoning-parser PR could land green on
OpenAI smoke + break Anthropic silently, or vice versa. The paired
test makes that class of regression visible at PR time.

The pattern: a single mock ``_StreamingEngine`` is wired to BOTH
routers in one FastAPI app, the same request payload is POSTed twice
(once per route), each SSE stream is reduced to a normalized dict of
``{text, reasoning, tool_calls}``, and the two dicts must be equal.

Issue #320, item M3.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.routes.anthropic import router as anthropic_router
from vllm_mlx.routes.chat import router as chat_router


class _ThinkingTemplateTokenizer:
    chat_template = "{% if add_generation_prompt %}<think>{% endif %}"


class _StreamingEngine:
    """Replays a fixed list of ``new_text`` deltas. Identical engine on
    both routes so any output divergence is route-layer drift, not
    engine non-determinism."""

    # Route surface contract (kept in sync with vllm_mlx/routes/*.py):
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = _ThinkingTemplateTokenizer()

    def __init__(self, deltas: list[str]):
        self._deltas = deltas
        self.calls: list[dict] = []

    async def stream_chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        n = len(self._deltas)
        for i, text in enumerate(self._deltas, start=1):
            last = i == n
            yield GenerationOutput(
                text=text,
                new_text=text,
                prompt_tokens=5,
                completion_tokens=i,
                finished=last,
                finish_reason="stop" if last else None,
                channel=None,
            )

    async def chat(self, messages, **kwargs):
        """Non-streaming path: concatenate all deltas."""
        self.calls.append({"messages": messages, "kwargs": kwargs})
        full = "".join(self._deltas)
        return GenerationOutput(
            text=full,
            new_text=full,
            prompt_tokens=5,
            completion_tokens=len(self._deltas),
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_paired_client(
    deltas: list[str],
    *,
    reasoning_parser: str | None = None,
    no_thinking: bool = True,
) -> tuple[TestClient, _StreamingEngine]:
    """One FastAPI app mounting both routers + shared mock engine."""
    engine = _StreamingEngine(deltas)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.no_thinking = no_thinking
    cfg.reasoning_parser_name = reasoning_parser
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(anthropic_router)
    return TestClient(app), engine


def _parse_sse(response_text: str) -> list[dict]:
    events = []
    for raw_event in response_text.split("\n\n"):
        for line in raw_event.splitlines():
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    continue
                events.append(json.loads(data))
                break
    return events


def _normalize_openai_stream(events: list[dict]) -> dict:
    """Reduce OpenAI chat.completion.chunk stream to {text, reasoning, tool_calls}."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {}) or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(
                    idx, {"name": "", "arguments": "", "id": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
    }


def _normalize_anthropic_stream(events: list[dict]) -> dict:
    """Reduce Anthropic SSE event stream to {text, reasoning, tool_calls}.

    Anthropic streams alternate content-block kinds: ``text``,
    ``thinking``, ``tool_use``. Each block opens with a
    ``content_block_start``, accumulates ``content_block_delta`` events
    (``text_delta`` / ``thinking_delta`` / ``input_json_delta``), and
    closes with ``content_block_stop``.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    open_blocks: dict[int, dict] = {}

    for event in events:
        etype = event.get("type")
        if etype == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {}) or {}
            open_blocks[idx] = block
            if block.get("type") == "tool_use":
                tool_calls[idx] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
        elif etype == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {}) or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif dtype == "thinking_delta":
                reasoning_parts.append(delta.get("thinking", ""))
            elif dtype == "input_json_delta":
                if idx in tool_calls:
                    tool_calls[idx]["arguments"] += delta.get("partial_json", "")
    return {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
    }


def _request_pair(
    client: TestClient,
    *,
    messages: list[dict],
    stream: bool,
    tools: list[dict] | None = None,
    extra_openai: dict | None = None,
    extra_anthropic: dict | None = None,
) -> tuple[dict, dict]:
    """POST the same logical request to both routes; return (openai_norm, anthropic_norm)."""
    openai_payload: dict = {
        "model": "test-model",
        "max_tokens": 64,
        "stream": stream,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        openai_payload["tools"] = tools
    if extra_openai:
        openai_payload.update(extra_openai)

    anthropic_payload: dict = {
        "model": "test-model",
        "max_tokens": 64,
        "stream": stream,
        "messages": messages,
        "temperature": 0,
    }
    if tools:
        # Anthropic tool schema: name + description + input_schema (not function/parameters)
        anthropic_payload["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {}),
            }
            for t in tools
        ]
    if extra_anthropic:
        anthropic_payload.update(extra_anthropic)

    r_openai = client.post("/v1/chat/completions", json=openai_payload)
    r_anthropic = client.post("/v1/messages", json=anthropic_payload)

    assert r_openai.status_code == 200, (
        f"OpenAI route failed: {r_openai.status_code} {r_openai.text[:300]}"
    )
    assert r_anthropic.status_code == 200, (
        f"Anthropic route failed: {r_anthropic.status_code} {r_anthropic.text[:300]}"
    )

    if stream:
        return (
            _normalize_openai_stream(_parse_sse(r_openai.text)),
            _normalize_anthropic_stream(_parse_sse(r_anthropic.text)),
        )

    # Non-streaming: synthesize the same normalized dict from the
    # single response payload of each surface.
    o = r_openai.json()
    msg = o["choices"][0]["message"]
    openai_norm = {
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": [
            {
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments", ""),
            }
            for tc in (msg.get("tool_calls") or [])
        ],
    }
    a = r_anthropic.json()
    text_parts, reasoning_parts, tcs = [], [], []
    for block in a.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tcs.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(
                        block.get("input", {}), separators=(",", ":")
                    ),
                }
            )
    anthropic_norm = {
        "text": "".join(text_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tcs,
    }
    return openai_norm, anthropic_norm


@pytest.fixture(autouse=True)
def _reset_server_config():
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------- cases


@pytest.mark.parametrize("stream", [True, False])
def test_plain_text_agrees(stream):
    """Plain text reply with no reasoning parser, no tools. Sanity floor."""
    client, _ = _make_paired_client(["Hello ", "world!"])
    openai, anthropic = _request_pair(
        client,
        messages=[{"role": "user", "content": "say hi"}],
        stream=stream,
    )
    assert openai["text"] == anthropic["text"] == "Hello world!"
    assert openai["reasoning"] == anthropic["reasoning"] == ""
    assert openai["tool_calls"] == anthropic["tool_calls"] == []


@pytest.mark.parametrize("stream", [True, False])
def test_reasoning_extracted_consistently(stream):
    """A delta stream containing ``<think>...</think>answer`` must
    decompose identically through both routes' reasoning parser
    (qwen3 → reasoning_content for OpenAI, thinking_delta for Anthropic).

    This is the exact class #288/#289 hit: the parser routed correctly
    on one surface and silently leaked tags to the other.
    """
    # With server-default thinking on, qwen3 parser routes pre-</think>
    # text to reasoning, post-</think> text to content.
    client, _ = _make_paired_client(
        ["<think>", "Let me think.", "</think>", "Final answer."],
        reasoning_parser="qwen3",
        no_thinking=False,
    )
    openai, anthropic = _request_pair(
        client,
        messages=[{"role": "user", "content": "reason then answer"}],
        stream=stream,
    )
    # Both surfaces must agree on the split; we don't pin the exact
    # split rule (that's the parser's contract, tested elsewhere) —
    # only that the two surfaces produce the SAME split.
    assert openai["text"] == anthropic["text"], (
        f"text divergence: openai={openai['text']!r} vs anthropic={anthropic['text']!r}"
    )
    assert openai["reasoning"] == anthropic["reasoning"], (
        f"reasoning divergence: openai={openai['reasoning']!r} vs anthropic={anthropic['reasoning']!r}"
    )


@pytest.mark.parametrize("stream", [True, False])
def test_no_thinking_keeps_text_as_text(stream):
    """``no_thinking=True`` (server) + reasoning parser configured:
    direct answers must NOT leak into reasoning on either surface.
    Closes #223 specifically — the qwen3 parser's implicit-think
    heuristic used to route everything-without-think-tag as reasoning,
    which the Anthropic streaming route then put in thinking_delta
    blocks. The fix must apply symmetrically to both surfaces."""
    client, _ = _make_paired_client(
        ["Direct ", "answer."],
        reasoning_parser="qwen3",
        no_thinking=True,
    )
    openai, anthropic = _request_pair(
        client,
        messages=[{"role": "user", "content": "answer directly"}],
        stream=stream,
    )
    assert openai["text"] == anthropic["text"] == "Direct answer."
    assert openai["reasoning"] == anthropic["reasoning"] == ""


@pytest.mark.parametrize("stream", [True, False])
def test_chunking_boundaries_dont_drift(stream):
    """Same logical output, different SSE chunk boundaries. Both routes
    must reassemble to the same text regardless of where the boundaries
    fall. Catches buffering / partial-utf8 / look-ahead divergences."""
    deltas_fine = list("Hello, world!")  # 13 single-char deltas
    client_fine, _ = _make_paired_client(deltas_fine)
    o_fine, a_fine = _request_pair(
        client_fine,
        messages=[{"role": "user", "content": "say hi"}],
        stream=stream,
    )

    client_coarse, _ = _make_paired_client(["Hello, ", "world!"])
    o_coarse, a_coarse = _request_pair(
        client_coarse,
        messages=[{"role": "user", "content": "say hi"}],
        stream=stream,
    )

    assert o_fine["text"] == a_fine["text"] == "Hello, world!"
    assert o_coarse["text"] == a_coarse["text"] == "Hello, world!"
