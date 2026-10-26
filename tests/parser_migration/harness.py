"""CPU-only route harness; normalized values retain duplicate calls and text."""

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning import get_parser
from rapid_mlx.routes import anthropic, chat, responses

TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_shell_command", "parameters": {"type": "object"}},
    }
]


class Engine:
    preserve_native_tool_format = True
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None
    index = 0

    def __init__(self, raw):
        self.raw = raw

    def build_prompt(self, *a, **kw):
        return self.prompt

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
        for idx, c in enumerate(self.chunks, 1):
            await asyncio.sleep(0)
            Engine.index = idx
            s += c
            yield GenerationOutput(
                text=s,
                new_text=c,
                prompt_tokens=4,
                completion_tokens=idx,
                finished=False,
            )
        Engine.index = len(self.chunks) + 1
        yield GenerationOutput(
            text=s,
            new_text="",
            prompt_tokens=4,
            completion_tokens=len(self.chunks),
            finished=True,
            finish_reason=self.finish,
            matched_stop=self.matched,
        )


class Trace:
    events = []

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def wrapped(message):
            if message["type"] == "http.response.body":
                for line in message.get("body", b"").decode().splitlines():
                    if line.startswith("data: {"):
                        p = json.loads(line[6:])
                        d = None
                        if "choices" in p:
                            d = (
                                p["choices"][0].get("delta", {}).get("content")
                                if p["choices"]
                                else None
                            )
                        elif p.get("type") == "response.output_text.delta":
                            d = p.get("delta")
                        elif p.get("type") == "content_block_delta":
                            d = p.get("delta", {}).get("text")
                        if d:
                            Trace.events.append([Engine.index, d])
            await send(message)

        await self.app(scope, receive, wrapped)


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
            joined = {}
            for p in packets:
                typ = p.get("type", "")
                if typ == "response.output_text.delta":
                    c += p.get("delta", "")
                if "reasoning" in typ and typ.endswith(".delta"):
                    r += p.get("delta", "")
                if (
                    typ == "response.output_item.added"
                    and p.get("item", {}).get("type") == "function_call"
                ):
                    old = joined.setdefault(
                        p["output_index"], {"name": "", "arguments": ""}
                    )
                    old["name"] += p["item"].get("name", "")
                    old["arguments"] += p["item"].get("arguments", "")
                if typ == "response.function_call_arguments.delta":
                    joined[p["output_index"]]["arguments"] += p.get("delta", "")
            calls = list(joined.values())
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
                    old = joined.setdefault(idx, {"name": "", "arguments": ""})
                    old["name"] += b["name"]
                    if b.get("input"):
                        old["arguments"] += json.dumps(b["input"])
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


CLIENTS = {}


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
    prompt="prompt",
    tools=None,
    tool_choice=None,
):
    cfg = reset_config()
    cfg.model_name = model
    cfg.no_thinking = not thinking
    cfg.reasoning_parser_name = r
    cfg.reasoning_parser = get_parser(r)() if r else None
    cfg.tool_call_parser = t
    cfg.enable_auto_tool_choice = True
    cfg.engine = Engine(raw)
    cfg.engine.prompt = prompt
    declared_tools = TOOLS if tools is None else tools
    cfg.engine.chunks = chunks or list(raw)
    cfg.engine.finish = finish
    cfg.engine.matched = matched
    if template:
        cfg.engine.tokenizer = SimpleNamespace(chat_template=template)
    Trace.events = []
    Engine.index = 0
    if surface not in CLIENTS:
        app = FastAPI()
        app.include_router(
            {"chat": chat, "responses": responses, "messages": anthropic}[
                surface
            ].router
        )
        CLIENTS[surface] = TestClient(Trace(app))
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
            tools=declared_tools,
            chat_template_kwargs={"enable_thinking": thinking},
        )
        url = "/v1/chat/completions"
    elif surface == "responses":
        b.update(
            input=hist,
            tools=[{"type": "function", **x["function"]} for x in declared_tools],
            chat_template_kwargs={"enable_thinking": thinking},
        )
        url = "/v1/responses"
    else:
        b.update(
            thinking={"type": "enabled" if thinking else "disabled"},
            messages=hist,
            tools=[
                {"name": x["function"]["name"], "input_schema": {"type": "object"}}
                for x in declared_tools
            ],
        )
        url = "/v1/messages"
    if tools_mode == "none":
        b["tool_choice"] = {"type": "none"} if surface == "messages" else "none"
    if tools_mode == "absent":
        b.pop("tools", None)
    if tool_choice is not None:
        b["tool_choice"] = tool_choice
    if matched:
        b["stop"] = [matched]
    try:
        resp = CLIENTS[surface].post(url, json=b)
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
        result["first_content_index"] = Trace.events[0][0] if Trace.events else None
        result["content_deltas"] = len(Trace.events)
        result["input_chunks"] = len(cfg.engine.chunks)
        if capture:
            result["packets"] = packets
        return result
    except Exception as e:
        return {"error": type(e).__name__, "body": str(e)[:400]}
    finally:
        reset_config()
