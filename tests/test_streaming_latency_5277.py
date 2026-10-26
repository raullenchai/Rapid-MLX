# SPDX-License-Identifier: Apache-2.0
"""Exercise streaming delivery at the ASGI wire, without a buffering HTTP client."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from transformers.utils.chat_template_utils import _compile_jinja_template

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning.qwen3_parser import Qwen3ReasoningParser
from rapid_mlx.routes import anthropic, chat, responses
from tests.test_rejected_tool_diagnostics import Engine as BaseEngine

ANSWER = "A useful answer arrives steadily, one token at a time."


class Engine(BaseEngine):
    def __init__(self, template, thinking):
        self.thinking = thinking
        self.tokenizer = SimpleNamespace(chat_template=template)
        self.raw = ("Private thought.</think>" if thinking else "") + ANSWER
        self.tokens = 0

    def build_prompt(
        self,
        messages,
        tools=None,
        enable_thinking=None,
        add_generation_prompt=True,
        chat_template_kwargs=None,
    ):
        return _compile_jinja_template(self.tokenizer.chat_template).render(
            messages=messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
            bos_token="",
            eos_token="",
        )

    async def stream_chat(self, **kwargs):
        for i, char in enumerate(self.raw):
            await asyncio.sleep(0.001)
            self.tokens = i + 1
            yield GenerationOutput(
                text=self.raw[: i + 1],
                new_text=char,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=False,
                finish_reason=None,
            )
        yield GenerationOutput(
            text=self.raw,
            new_text="",
            finished=True,
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=len(self.raw),
        )


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("family", ["qwen35", "flint"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_streaming_delivery(surface, family, thinking, tools, explicit):
    async def run():
        cfg = reset_config()
        cfg.model_name = (
            "qwen3.5-4b-4bit" if family == "qwen35" else "flint-40b-a2.5b-4bit"
        )
        cfg.no_thinking = False
        cfg.reasoning_parser_name = "qwen3"
        cfg.reasoning_parser = Qwen3ReasoningParser()
        template = (
            Path(__file__).parent
            / "fixtures"
            / (
                "qwen35_chat_template.jinja"
                if family == "qwen35"
                else "flint_chat_template.jinja"
            )
        ).read_text()
        engine = cfg.engine = Engine(template, thinking)
        if explicit:
            engine.raw = "<think>Private thought.</think>" + ANSWER
        cfg.tool_call_parser = "glm47" if family == "flint" else "qwen3_coder"
        cfg.enable_auto_tool_choice = True
        app = FastAPI()
        for router in (chat.router, responses.router, anthropic.router):
            app.include_router(router)
        history = [{"role": "user", "content": "Give a useful answer."}]
        body = {
            "model": cfg.model_name,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if surface == "responses":
            body["input"] = history
            path = "/v1/responses"
        else:
            body["messages"] = history
            path = "/v1/chat/completions"
            if surface == "messages":
                path = "/v1/messages"
                body.pop("chat_template_kwargs")
                body.update(
                    max_tokens=256,
                    thinking={"type": "enabled" if thinking else "disabled"},
                )
        if tools:
            fn = {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
            if surface == "messages":
                body["tools"] = [{"name": "lookup", "input_schema": fn["parameters"]}]
            elif surface == "responses":
                body["tools"] = [{"type": "function", **fn}]
            else:
                body["tools"] = [{"type": "function", "function": fn}]
        sent = False
        wire = ""
        content = []
        reasoning = []

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": json.dumps(body).encode()}
            await asyncio.Event().wait()

        async def send(event):
            nonlocal wire
            if event["type"] == "http.response.start":
                assert event["status"] == 200
            if event["type"] != "http.response.body":
                return
            wire += event.get("body", b"").decode()
            while "\n\n" in wire:
                packet, wire = wire.split("\n\n", 1)
                for line in packet.splitlines():
                    if not line.startswith("data:") or not line[5:].lstrip().startswith(
                        "{"
                    ):
                        continue
                    p = json.loads(line[5:])
                    if surface == "chat":
                        d = (p.get("choices") or [{}])[0].get("delta", {})
                        c, r = d.get("content"), d.get("reasoning_content")
                    elif surface == "responses":
                        c = (
                            p.get("delta")
                            if p.get("type") == "response.output_text.delta"
                            else None
                        )
                        r = (
                            p.get("delta")
                            if p.get("type") == "response.reasoning_summary_text.delta"
                            else None
                        )
                    else:
                        d = p.get("delta", {})
                        c, r = d.get("text"), d.get("thinking")
                    if c:
                        content.append((engine.tokens, c))
                    if r:
                        reasoning.append((engine.tokens, r))

        try:
            await app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                    "client": ("127.0.0.1", 1234),
                    "server": ("test", 80),
                },
                receive,
                send,
            )
            assert "".join(c for _, c in content) == ANSWER
            answer_start = len(engine.raw) - len(ANSWER)
            assert content[0][0] - answer_start <= 8, content
            assert len(content) > 8, content
            assert all("<" not in c and ">" not in c for _, c in content + reasoning)
            if thinking and surface != "responses":
                assert reasoning[0][0] <= 8 + (len("<think>") if explicit else 0)
                assert len(reasoning) > 1
        finally:
            reset_config()

    asyncio.run(run())


@pytest.mark.parametrize(
    "prefix,flag,raw,visible",
    [
        ("<think></think>", True, ANSWER, ANSWER),
        ("<think>", False, "Secret</think>" + ANSWER, ANSWER),
        ("<|assistant|>", False, "<think>Secret</think>" + ANSWER, ANSWER),
        ("<|assistant|>", False, ANSWER + "<think>Secret</think>", ANSWER),
        ("<|assistant|>", True, "Secret</think>" + ANSWER, ANSWER),
    ],
)
def test_prompt_lane_and_marker_splits(prefix, flag, raw, visible):
    for split in range(1, len(raw)):
        parser = Qwen3ReasoningParser()
        parser.configure_request(enable_thinking=flag)
        parser.configure_prompt(prefix, enable_thinking=flag)
        chunks = [
            parser.extract_reasoning_streaming("", raw[:split], raw[:split]),
            parser.extract_reasoning_streaming(raw[:split], raw, raw[split:]),
            parser.finalize_streaming(raw),
            parser.finish_stream(),
        ]
        assert "".join(c.content or "" for c in chunks if c) == visible
        assert "".join(c.reasoning or "" for c in chunks if c) in ("", "Secret")


@pytest.mark.parametrize(
    "mode",
    [
        "engine",
        "tokenizer",
        "missing",
        "broken",
        "nonstring",
        "different",
        "empty",
        "noparser",
    ],
)
def test_rendered_prompt_configuration(mode):
    from rapid_mlx.service.helpers import configure_reasoning_from_prompt

    parser = Qwen3ReasoningParser()
    parser.configure_request(enable_thinking=False)

    def render(messages, add_generation_prompt, **kwargs):
        if mode == "broken":
            raise ValueError("unsupported template")
        if mode == "nonstring":
            return [1, 2]
        if mode == "different":
            return "A" if add_generation_prompt else "B"
        return "history" + (
            "<think></think>" if add_generation_prompt and mode != "empty" else ""
        )

    engine = SimpleNamespace()
    if mode in ("tokenizer", "broken"):
        engine.tokenizer = SimpleNamespace(apply_chat_template=render)
    elif mode != "missing":
        engine.build_prompt = render
    configure_reasoning_from_prompt(
        None if mode == "noparser" else parser, engine, [], {"enable_thinking": False}
    )
    delta = parser.extract_reasoning_streaming("", "Hello", "Hello")
    if mode in ("engine", "tokenizer"):
        assert delta.content == "Hello"
    else:
        assert delta is None


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("tail", ["<", "<thi", "</thi"])
def test_prompt_lane_flushes_only_literal_content_tail(active, tail):
    parser = Qwen3ReasoningParser()
    parser.configure_request(enable_thinking=active)
    parser.configure_prompt(
        "<think>" if active else "<think></think>", enable_thinking=active
    )
    text = "Some text" + tail
    delta = parser.extract_reasoning_streaming("", text, text)
    assert (delta.reasoning if active else delta.content) == "Some text"
    final = parser.finalize_streaming(text)
    assert final is None if active else final.content == tail
    assert parser.finish_stream() is None
