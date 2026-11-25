# SPDX-License-Identifier: Apache-2.0
"""Route-level regressions for reasoning delimiter boundaries."""

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning import get_parser
from rapid_mlx.routes import anthropic, chat, responses
from tests.test_flint_parser_routing import Engine as BaseEngine

TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_shell_command", "parameters": {"type": "object"}},
    }
]


class Engine(BaseEngine):
    chunks = None
    finish = "stop"
    matched = None

    async def chat(self, **kw):
        return GenerationOutput(
            text=self.raw,
            raw_text=self.raw,
            prompt_tokens=4,
            completion_tokens=len(self.raw),
            finished=True,
            finish_reason=self.finish,
            matched_stop=self.matched,
        )

    async def stream_chat(self, **kw):
        s = ""
        for c in self.chunks:
            s += c
            yield GenerationOutput(
                text=s,
                new_text=c,
                prompt_tokens=4,
                completion_tokens=len(s),
                finished=False,
            )
        yield GenerationOutput(
            text=s,
            new_text="",
            finished=True,
            finish_reason=self.finish,
            matched_stop=self.matched,
            prompt_tokens=4,
            completion_tokens=len(s),
        )


def norm(packets, surface, stream):
    c = r = ""
    calls = []
    if surface == "chat":
        joined = {}
        for p in packets:
            for ch in p.get("choices", []):
                m = ch.get("delta", ch.get("message", {}))
                c += m.get("content") or ""
                r += m.get("reasoning_content") or ""
                for i, tc in enumerate(m.get("tool_calls") or []):
                    key = tc.get("index", i)
                    f = tc.get("function", {})
                    old = joined.setdefault(key, {"name": "", "arguments": ""})
                    for k in old:
                        old[k] += f.get(k) or ""
        calls = list(joined.values())
    elif surface == "responses":
        if stream:
            for p in packets:
                typ = p.get("type", "")
                if typ == "response.output_text.delta":
                    c += p.get("delta", "")
                if "reasoning" in typ and typ.endswith(".delta"):
                    r += p.get("delta", "")
                if (
                    typ == "response.output_item.done"
                    and p.get("item", {}).get("type") == "function_call"
                ):
                    calls.append(p["item"])
        else:
            for i in packets[0].get("output", []):
                if i["type"] == "message":
                    c += "".join(x.get("text", "") for x in i["content"])
                if i["type"] == "reasoning":
                    r += "".join(
                        x.get("text", "")
                        for x in i.get("summary", []) + i.get("content", [])
                    )
                if i["type"] == "function_call":
                    calls.append(i)
    else:
        joined = {}
        for p in packets:
            if not stream:
                for b in p.get("content", []):
                    if b["type"] == "text":
                        c += b["text"]
                    elif b["type"] == "thinking":
                        r += b["thinking"]
                    elif b["type"] == "tool_use":
                        calls.append({"name": b["name"], "arguments": b["input"]})
            elif p.get("type") == "content_block_start":
                b = p["content_block"]
                idx = p["index"]
                if b["type"] == "text":
                    c += b.get("text", "")
                if b["type"] == "thinking":
                    r += b.get("thinking", "")
                if b["type"] == "tool_use":
                    joined[idx] = {"name": b["name"], "arguments": ""}
            elif p.get("type") == "content_block_delta":
                d = p["delta"]
                c += d.get("text", "")
                r += d.get("thinking", "")
                if "partial_json" in d:
                    joined[p["index"]]["arguments"] += d["partial_json"]
        calls += list(joined.values())
    normcalls = []
    for call in calls:
        a = call.get("arguments", "")
        try:
            a = json.loads(a) if isinstance(a, str) else a
        except Exception:
            pass
        normcalls.append([call.get("name"), a])
    return {"content": c, "reasoning": r, "calls": normcalls}


def route(
    r,
    t,
    raw,
    surface,
    stream,
    chunks=None,
    thinking=True,
    later=False,
    finish="stop",
    matched=None,
    template=None,
    model="audit-model",
    tools_mode="enabled",
    capture=False,
):
    cfg = reset_config()
    cfg.model_name = model
    cfg.no_thinking = not thinking
    cfg.reasoning_parser_name = r
    cfg.reasoning_parser = get_parser(r)() if r else None
    cfg.tool_call_parser = t
    cfg.enable_auto_tool_choice = True
    cfg.engine = Engine(raw)
    cfg.engine.chunks = chunks or list(raw)
    cfg.engine.finish = finish
    cfg.engine.matched = matched
    if template:
        cfg.engine.tokenizer = SimpleNamespace(chat_template=template)
    app = FastAPI()
    app.include_router(
        {"chat": chat, "responses": responses, "messages": anthropic}[surface].router
    )
    b = {"model": model, "stream": stream, "max_tokens": 4096}
    hist = [{"role": "user", "content": "Use a tool if needed; answer."}]
    if later:
        hist += [
            {"role": "assistant", "content": "Previous answer."},
            {"role": "user", "content": "Continue."},
        ]
    if surface == "chat":
        b.update(
            messages=hist,
            tools=TOOLS,
            chat_template_kwargs={"enable_thinking": thinking},
        )
        url = "/v1/chat/completions"
    elif surface == "responses":
        b.update(
            input=hist,
            tools=[{"type": "function", **x["function"]} for x in TOOLS],
            chat_template_kwargs={"enable_thinking": thinking},
        )
        url = "/v1/responses"
    else:
        b.update(
            thinking={"type": "enabled" if thinking else "disabled"},
            messages=hist,
            tools=[
                {"name": x["function"]["name"], "input_schema": {"type": "object"}}
                for x in TOOLS
            ],
        )
        url = "/v1/messages"
    if tools_mode == "none":
        b["tool_choice"] = "none"
    if tools_mode == "absent":
        b.pop("tools", None)
    if matched:
        b["stop"] = [matched]
    try:
        with TestClient(app) as client:
            resp = client.post(url, json=b)
        if resp.status_code != 200:
            return {"error": resp.status_code, "body": resp.text[:500]}
        packets = (
            [
                json.loads(line[6:])
                for line in resp.text.splitlines()
                if line.startswith("data: {")
            ]
            if stream
            else [resp.json()]
        )
        result = norm(packets, surface, stream)
        if capture:
            result["packets"] = packets
        return result
    except Exception as e:
        return {"error": type(e).__name__, "body": str(e)[:400]}
    finally:
        reset_config()


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize(
    "raw,reason,answer",
    [
        ("<think>Secret.</think>Answer", "Secret.", "Answer"),
        (
            "<think>Secret.<think>Nested</think>Answer</think>Tail",
            "Secret.Nested",
            "AnswerTail",
        ),
        ("<think>Secret.</think></think>Answer", "Secret.", "Answer"),
    ],
)
def test_reasoning_delimiters_every_split(surface, thinking, raw, reason, answer):
    baseline = route("qwen3", "hermes", raw, surface, False, thinking=thinking)
    assert baseline == {"content": answer, "reasoning": reason, "calls": []}
    chunks = [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]
    for parts in chunks:
        actual = route("qwen3", "hermes", raw, surface, True, parts, thinking=thinking)
        assert actual == baseline, (parts, actual)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize(
    "raw,reasoning,content",
    [
        ("\n<think>R</think>A", "R", "A"),
        ("preamble<think>R</think>A", "", "preamble<think>RA"),
        (
            "I will list files.\n```bash\nls\n```\n</think>\nRunning it now.",
            "",
            "I will list files.\n```bash\nls\n```\n\nRunning it now.",
        ),
    ],
)
def test_thinking_off_stray_closer_keeps_answer(
    surface, raw, reasoning, content, monkeypatch
):
    from rapid_mlx.api.utils import clean_output_text

    original_chat = Engine.chat

    async def cleaned_chat(self, **kwargs):
        output = await original_chat(self, **kwargs)
        output.text = clean_output_text(
            output.text, **kwargs.get("_output_cleaning_kwargs", {})
        )
        return output

    monkeypatch.setattr(Engine, "chat", cleaned_chat)
    buffered = route("qwen3", "hermes", raw, surface, False, thinking=False)
    assert buffered == {"content": content, "reasoning": reasoning, "calls": []}
    for chunks in [[raw], list(raw)]:
        streamed = route("qwen3", "hermes", raw, surface, True, chunks, thinking=False)
        assert streamed == buffered


def test_nested_block_preserves_later_reasoning_and_answer():
    parser = get_parser("qwen3")()
    assert parser.extract_reasoning(
        "<think>S<think>N</think>A</think>T<think>R</think>Z"
    ) == ("SNR", "ATZ")


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tail", ["", "<"])
def test_thinking_off_tool_argument_opener_after_stray_closer(surface, stream, tail):
    raw = (
        "I will list files.</think><function=run_shell_command>"
        '<parameter=command>echo "<think>"</parameter></function>'
    ) + tail
    result = route("qwen3", "qwen3_coder_xml", raw, surface, stream, thinking=False)
    assert not result["reasoning"]
    assert "I will list files." in result["content"]
    assert result["calls"] == [["run_shell_command", {"command": 'echo "<think>"'}]]


@pytest.mark.parametrize(
    "raw",
    [
        "\n<think>R</think>A",
        "preamble<think>R</think>A",
        "I will list files.</think>Running it now.",
        "plain answer",
        "preamble<think>R",
        "  answer</think>tail</think><",
        "\n\t",
        "<thi",
        'I will list files.</think><function=run_shell_command><parameter=command>echo "<think>"</parameter></function>',
    ],
)
def test_thinking_off_every_split(raw):
    parser = get_parser("qwen3")()
    buffered = parser.extract_reasoning(raw, enable_thinking=False)
    expected = (
        ("R", "A")
        if raw.lstrip().startswith("<think>")
        else (None, raw.replace("</think>", ""))
    )
    assert buffered == expected
    for chunks in [[raw], list(raw)] + [[raw[:i], raw[i:]] for i in range(1, len(raw))]:
        parser.configure_request(enable_thinking=False)
        previous = reasoning = content = ""
        for chunk in chunks:
            delta = parser.extract_reasoning_streaming(
                previous, previous + chunk, chunk
            )
            previous += chunk
            if delta:
                reasoning += delta.reasoning or ""
                content += delta.content or ""
        final = parser.finish_stream()
        if final:
            reasoning += final.reasoning or ""
            content += final.content or ""
        assert (reasoning, content) == tuple(value or "" for value in buffered), chunks
