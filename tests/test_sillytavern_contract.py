# SPDX-License-Identifier: Apache-2.0
"""SillyTavern wire contract: chat + text completion, streaming and not.

SillyTavern connects through "Chat Completion → Custom (OpenAI-compatible)"
or "Text Completion" and sends its full sampler payload on every request,
mostly at "off" values. These tests drive the real chat / completions /
models routes with a scripted engine and pin:

* the response bodies (streaming and non-streaming, with stop strings) are
  byte-identical, after id/timestamp normalisation, to origin/main for the
  same stock payload (``tests/fixtures/sillytavern_golden.json`` was
  captured with this harness at origin/main 9221ff773);
* ``/v1/models`` lists the served model for the model picker;
* the samplers we implement reach the engine (DRY, repetition range);
* the samplers we do not implement are refused with a 400 naming the field
  as soon as they are switched on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.middleware.exception_handlers import install_exception_handlers
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.routes.completions import router as completions_router
from rapid_mlx.routes.models import router as models_router

GOLDEN = Path(__file__).parent / "fixtures" / "sillytavern_golden.json"

# What SillyTavern sends with stock settings (Custom OpenAI-compatible chat
# and generic text completion): enabled basics plus many samplers at "off".
STOCK_SAMPLERS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "typical_p": 1,
    "tfs": 1,
    "top_a": 0,
    "mirostat_mode": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.1,
    "xtc_threshold": 0.1,
    "xtc_probability": 0,
    "smoothing_factor": 0,
    "dry_multiplier": 0,
    "max_tokens": 64,
}


class _ScriptedEngine:
    """Engine stand-in: fixed deltas, records the kwargs each call received."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.deltas = ["*smiles* ", "Hello", ", traveler."]

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    def _outputs(self):
        text = ""
        for i, delta in enumerate(self.deltas):
            text += delta
            last = i == len(self.deltas) - 1
            yield GenerationOutput(
                text=text,
                new_text=delta,
                prompt_tokens=12,
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
                channel=None,
            )

    async def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return list(self._outputs())[-1]

    async def stream_chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        for output in self._outputs():
            yield output

    async def generate(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return list(self._outputs())[-1]

    async def stream_generate(self, prompt, **kwargs):
        self.calls.append(kwargs)
        for output in self._outputs():
            yield output


@pytest.fixture
def client():
    engine = _ScriptedEngine()
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "my-rp-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(completions_router)
    app.include_router(models_router)
    test_client = TestClient(app)
    test_client.engine = engine
    return test_client


_VOLATILE = {"id", "created", "system_fingerprint"}


def _normalise(value):
    if isinstance(value, dict):
        return {
            k: _normalise(v) for k, v in sorted(value.items()) if k not in _VOLATILE
        }
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _sse(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line.removeprefix("data:").strip()
            events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


CHAT = {
    "model": "my-rp-model",
    "messages": [
        {"role": "system", "content": "You are Aria, a wandering bard."},
        {"role": "user", "content": "Hi there!"},
    ],
    "stop": ["\nUser:", "</s>"],
    **STOCK_SAMPLERS,
}
TEXT = {
    "model": "my-rp-model",
    "prompt": "Aria is a wandering bard.\nUser: Hi there!\nAria:",
    "stop": ["\nUser:"],
    **STOCK_SAMPLERS,
}


def _transcript(client) -> dict:
    out = {}
    response = client.post("/v1/chat/completions", json=CHAT)
    out["chat"] = [response.status_code, _normalise(response.json())]
    response = client.post("/v1/chat/completions", json={**CHAT, "stream": True})
    out["chat_stream"] = [response.status_code, _normalise(_sse(response.text))]
    response = client.post("/v1/completions", json=TEXT)
    out["text"] = [response.status_code, _normalise(response.json())]
    response = client.post("/v1/completions", json={**TEXT, "stream": True})
    out["text_stream"] = [response.status_code, _normalise(_sse(response.text))]
    response = client.get("/v1/models")
    out["models"] = [
        response.status_code,
        [entry["id"] for entry in response.json()["data"]],
    ]
    return out


def test_stock_sillytavern_payloads_match_origin_main(client):
    assert _transcript(client) == json.loads(GOLDEN.read_text())


def test_stop_strings_and_basic_samplers_reach_the_engine(client):
    client.post("/v1/chat/completions", json=CHAT)
    client.post("/v1/completions", json=TEXT)
    chat, text = client.engine.calls
    assert chat["stop"] == ["\nUser:", "</s>"]
    assert text["stop"] == ["\nUser:"]
    for call in (chat, text):
        assert call["top_k"] == 40
        assert call["min_p"] == pytest.approx(0.05)
        assert call["repetition_penalty"] == pytest.approx(1.1)
        assert "dry_logits_processor" not in call  # multiplier 0 = off
        assert "repetition_context_size" not in call


@pytest.mark.parametrize(
    "route,body", [("/v1/chat/completions", CHAT), ("/v1/completions", TEXT)]
)
def test_dry_and_repetition_range_reach_the_engine(client, route, body):
    payload = {
        **body,
        "dry_multiplier": 0.8,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": 0,
        "dry_sequence_breakers": '["\\n", ":", "\\"", "*"]',
        "rep_pen_range": 1024,
    }
    assert client.post(route, json=payload).status_code == 200
    (call,) = client.engine.calls
    processor = call["dry_logits_processor"]
    assert processor.multiplier == pytest.approx(0.8)
    assert processor.allowed_length == 2
    assert call["repetition_context_size"] == 1024


@pytest.mark.parametrize(
    "field,value",
    [
        ("xtc_probability", 0.5),
        ("typical_p", 0.9),
        ("mirostat_mode", 2),
        ("top_a", 0.2),
        ("smoothing_factor", 0.3),
    ],
)
@pytest.mark.parametrize(
    "route,body", [("/v1/chat/completions", CHAT), ("/v1/completions", TEXT)]
)
def test_enabled_unsupported_samplers_are_refused_by_name(
    client, route, body, field, value
):
    response = client.post(route, json={**body, field: value})
    assert response.status_code == 400
    assert field in json.dumps(response.json())
    assert client.engine.calls == []
