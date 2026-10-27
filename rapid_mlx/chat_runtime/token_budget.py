"""Token accounting, independent of transport/output byte quotas."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache

import httpx
from pydantic_ai._instrumentation import get_instructions


@lru_cache(maxsize=4096)
def estimate_text(text: str) -> int:
    """Conservative tokenizer-free estimate, cached per immutable message part.

    ASCII is budgeted at three characters/token, CJK at two tokens/codepoint,
    other Unicode at three. This is an estimate, never labelled an exact count.
    """
    ascii_count = sum(ord(c) < 128 for c in text)
    cjk = sum(0x2E80 <= ord(c) <= 0x9FFF for c in text)
    return math.ceil(ascii_count / 3) + 2 * cjk + 3 * (len(text) - ascii_count - cjk)


def text_value(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )


def part_tokens(part):
    cost = 8 + estimate_text(
        text_value(getattr(part, "content", getattr(part, "args", "")))
    )
    if getattr(part, "part_kind", "") in {"tool-call", "tool-return"}:
        cost += 8 + estimate_text(part.tool_name) + estimate_text(part.tool_call_id)
    return cost


def context_tokens(messages):
    return sum(
        8 + sum(part_tokens(part) for part in message.parts) for message in messages
    )


def count_payload(messages, parameters=None, model=""):
    """Translate the same text/tool conversation to the server counting adapter."""
    system, rows = [], []
    instructions = get_instructions(messages, parameters)
    if instructions:
        system.append(instructions)
    for message in messages:
        blocks = []
        role = "assistant" if message.kind == "response" else "user"
        for part in message.parts:
            kind = part.part_kind
            if kind == "system-prompt":
                system.append(part.content)
            elif kind in {"text", "user-prompt"}:
                blocks.append({"type": "text", "text": text_value(part.content)})
            elif kind == "tool-call":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "input": part.args_as_dict(),
                    }
                )
            elif kind == "tool-return":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": part.tool_call_id,
                        "content": part.model_response_str(),
                    }
                )
        if blocks:
            if rows and rows[-1]["role"] == role:
                rows[-1]["content"].extend(blocks)
            else:
                rows.append({"role": role, "content": blocks})
    tools = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.parameters_json_schema,
        }
        for t in (parameters.function_tools if parameters else [])
    ]
    return {
        "model": model,
        "system": "\n\n".join(system),
        "messages": rows
        or [{"role": "user", "content": [{"type": "text", "text": ""}]}],
        **({"tools": tools} if tools else {}),
    }


class TokenCounter:
    def __init__(self, clients=None, request=None):
        self.clients, self.request = clients, request
        self.cache = {}
        self.available = clients is not None
        self.correction = 0
        self.last_count = 0

    async def measure(self, messages, parameters=None):
        body = count_payload(
            messages, parameters, self.request.model if self.request else ""
        )
        prefix = (
            estimate_text(body.get("system", ""))
            + estimate_text(text_value(body.get("tools", [])))
            + 16
        )
        conversation = tool_output = 0
        for message in messages:
            conversation += 8  # Role/template boundary, independent of part payload.
            for part in message.parts:
                cost = part_tokens(part)
                if part.part_kind == "tool-return":
                    tool_output += cost
                elif part.part_kind != "system-prompt":
                    conversation += cost
        estimated_total = prefix + conversation + tool_output
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        exact = self.cache.get(key)
        if exact is None and self.available:
            try:
                response = await self.clients._http.post(
                    self.request.inference_url + "/messages/count_tokens",
                    json=body,
                    headers={
                        "Authorization": "Bearer "
                        + (
                            self.request.inference_api_key.get_secret_value()
                            if self.request.inference_api_key
                            else "rapid-local"
                        )
                    },
                    timeout=5,
                )
                response.raise_for_status()
                value = response.json()["input_tokens"]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("invalid token count")
                exact = value
                if len(self.cache) >= 256:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = exact
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                self.available = False
        used = max(
            0, (exact if exact is not None else estimated_total) + self.correction
        )
        self.last_count = used
        # Full-prompt count is exact; category attribution is proportional because
        # chat templates can join categories into shared tokens.
        system_tools = round(used * prefix / max(1, estimated_total))
        output = round(used * tool_output / max(1, estimated_total))
        return {
            "used_tokens": used,
            "system_tools_tokens": system_tools,
            "conversation_tokens": max(0, used - system_tools - output),
            "tool_output_tokens": output,
            "estimated": exact is None,
        }

    def calibrate(self, prompt_tokens):
        if prompt_tokens and self.last_count:
            self.correction = max(
                self.correction, prompt_tokens - self.last_count + self.correction
            )
