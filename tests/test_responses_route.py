# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for the OpenAI-compatible /v1/responses route.

Same lightweight-engine harness shape as ``test_anthropic_route_auth.py``
(no MLX import) so the tests stay fast and CI-portable.
"""

import json
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _Tokenizer:
    chat_template = ""

    def __init__(self):
        self.calls = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return list(range(len(text)))


class _BaseEngine:
    pass


@dataclass
class _GenerationOutput:
    text: str
    raw_text: str = ""
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None
    tool_calls: list | None = None


class _Engine:
    preserve_native_tool_format = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.stream_calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="hello world",
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        """Emit a tiny synthetic stream: three text chunks then EOS."""
        self.stream_calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        chunks = ["Hello", " from", " rapid"]
        for i, c in enumerate(chunks):
            yield _GenerationOutput(
                text="".join(chunks[: i + 1]),
                new_text=c,
                prompt_tokens=3 if i == 0 else 0,
                completion_tokens=i + 1,
                finish_reason=None if i < len(chunks) - 1 else "stop",
            )


def _install_lightweight_engine_modules(monkeypatch):
    engine_pkg = types.ModuleType("vllm_mlx.engine")
    engine_pkg.BaseEngine = _BaseEngine
    engine_pkg.GenerationOutput = _GenerationOutput

    base_mod = types.ModuleType("vllm_mlx.engine.base")
    base_mod.BaseEngine = _BaseEngine
    base_mod.GenerationOutput = _GenerationOutput

    monkeypatch.setitem(sys.modules, "vllm_mlx.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm_mlx.engine.base", base_mod)


_IMPORTED_UNDER_LIGHTWEIGHT_ENGINE = (
    "vllm_mlx.config",
    "vllm_mlx.config.server_config",
    "vllm_mlx.engine",
    "vllm_mlx.engine.base",
    "vllm_mlx.middleware.auth",
    "vllm_mlx.service.helpers",
    "vllm_mlx.routes.responses",
)
_PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "responses"),
)
_MISSING = object()


@pytest.fixture
def responses_client(monkeypatch):
    previous_modules = {
        name: sys.modules.get(name, _MISSING)
        for name in _IMPORTED_UNDER_LIGHTWEIGHT_ENGINE
    }
    previous_attrs = {}
    for module_name, attr in _PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE:
        module = sys.modules.get(module_name)
        previous_attrs[(module_name, attr)] = (
            getattr(module, attr, _MISSING) if module is not None else _MISSING
        )

    _install_lightweight_engine_modules(monkeypatch)

    from vllm_mlx.config import reset_config
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.routes.responses import router

    cfg = reset_config()
    cfg.api_key = "test-secret"
    cfg.engine = _Engine()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(
        client=TestClient(app),
        engine=cfg.engine,
        rate_limiter=rate_limiter,
        reset_config=reset_config,
    )

    reset_config()
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    for name, previous in previous_modules.items():
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    for (module_name, attr), previous in previous_attrs.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if previous is _MISSING:
            if hasattr(module, attr):
                delattr(module, attr)
        else:
            setattr(module, attr, previous)


def _payload(**overrides) -> dict:
    base = {
        "model": "test-model",
        "input": "Hello, world",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Auth — Bearer token; rapid-mlx Responses route is OpenAI-shape
# ---------------------------------------------------------------------------


class TestResponsesAuth:
    def test_requires_api_key(self, responses_client):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post("/v1/responses", json=_payload())

        assert response.status_code == 401
        assert response.json()["detail"] == "API key required"
        assert engine.calls == []

    def test_rejects_invalid_bearer(self, responses_client):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(),
            headers={"Authorization": "Bearer wrong-secret"},
        )

        assert response.status_code == 401
        assert engine.calls == []

    def test_accepts_valid_bearer(self, responses_client):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200, response.text
        assert len(engine.calls) == 1


# ---------------------------------------------------------------------------
# Non-stream happy path
# ---------------------------------------------------------------------------


class TestResponsesNonStream:
    def test_response_shape_matches_codex_expectation(self, responses_client):
        client = responses_client.client

        response = client.post(
            "/v1/responses",
            json=_payload(),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        # One message item; assistant role; output_text content.
        assert len(body["output"]) == 1
        item = body["output"][0]
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["content"][0]["type"] == "output_text"
        assert item["content"][0]["text"] == "hello world"
        # Usage is populated.
        assert body["usage"]["input_tokens"] == 3
        assert body["usage"]["output_tokens"] == 2
        assert body["usage"]["total_tokens"] == 5

    def test_response_carries_loaded_model_not_request_alias(self, responses_client):
        """The response.model field must be the loaded engine's model
        name, not whatever the client typed — same convention as #557."""
        client = responses_client.client

        response = client.post(
            "/v1/responses",
            json=_payload(model="gpt-5"),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200
        # Loaded model = "test-model" (set in fixture).
        assert response.json()["model"] == "test-model"

    def test_instructions_become_system_message(self, responses_client):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(instructions="You are Codex."),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200
        sent = engine.calls[-1].messages
        # extract_multimodal_content turns Message → dict on its way to the engine.
        first = sent[0]
        first_role = first.role if hasattr(first, "role") else first["role"]
        first_content = first.content if hasattr(first, "content") else first["content"]
        assert first_role == "system"
        assert first_content == "You are Codex."


# ---------------------------------------------------------------------------
# Codex model-name bypass (parallel to #557 claude-* bypass)
# ---------------------------------------------------------------------------


class TestCodexModelBypass:
    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-5",
            "gpt-5-codex",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "claude-opus-4-5",
        ],
    )
    def test_codex_model_names_route_to_loaded_engine(
        self, responses_client, model_name
    ):
        """Codex CLI sends ``gpt-5`` / ``gpt-5-codex`` etc. The route
        must route these to the loaded engine without 404'ing on the
        unknown name — same bypass as #557 for the Anthropic route."""
        client = responses_client.client

        response = client.post(
            "/v1/responses",
            json=_payload(model=model_name),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200, (
            f"Expected 200 for model={model_name!r}, got {response.status_code}: "
            f"{response.text}"
        )
        # Response model is the loaded one, regardless of what we asked for.
        assert response.json()["model"] == "test-model"


# ---------------------------------------------------------------------------
# Statelessness — previous_response_id must 400
# ---------------------------------------------------------------------------


class TestStatelessGate:
    def test_previous_response_id_returns_400(self, responses_client):
        """The shim has no response store. A client that sends
        ``previous_response_id`` would experience silent prompt loss
        on retries, so 400 loudly with an actionable error message."""
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(previous_response_id="resp_abc123"),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 400
        assert "previous_response_id" in response.json()["detail"]
        # Engine must not have been called.
        assert engine.calls == []


# ---------------------------------------------------------------------------
# Codex-style multi-item input replay
# ---------------------------------------------------------------------------


class TestCodexInputReplay:
    def test_function_call_and_output_replay_lands_as_assistant_then_tool(
        self, responses_client
    ):
        """A Codex turn carries a prior function_call + function_call_output
        in input[]. The adapter must produce assistant(tool_calls=...) then
        a tool message wired by tool_call_id."""
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(
                model="gpt-5-codex",
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "ls -la"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "run_shell",
                        "arguments": '{"cmd":"ls -la"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "total 8\ndrwxr-xr-x ...",
                    },
                ],
            ),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200, response.text
        sent = engine.calls[-1].messages

        def _role(m):
            return m.role if hasattr(m, "role") else m["role"]

        def _content(m):
            return m.content if hasattr(m, "content") else m["content"]

        # extract_multimodal_content (preserve_native_format=False — the
        # path most local models take) rewrites assistant tool_calls into
        # "[Calling tool: name(args)]" text and rewrites tool results
        # into role="user" with a "[Tool Result (call_id)]: ..." prefix.
        # So we verify the structural invariant — three turns, with the
        # tool name + tool result text reaching the engine — rather
        # than the native tool message shape (which is verified in
        # test_responses_adapter.py).
        assert len(sent) == 3
        assert _role(sent[0]) == "user"
        assert _role(sent[1]) == "assistant"
        # Assistant content carries the rewritten tool_call signature.
        assistant_text = _content(sent[1])
        assert "run_shell" in assistant_text
        assert "ls -la" in assistant_text
        # Tool result content surfaces (either as role=tool or as
        # rewritten user with the [Tool Result (call_id)] prefix).
        last_role = _role(sent[2])
        last_content = _content(sent[2])
        assert last_role in ("tool", "user")
        if last_role == "user":
            assert "call_1" in last_content  # the rewrite carries the call_id
        assert "total 8" in last_content


# ---------------------------------------------------------------------------
# Streaming path — SSE event names Codex CLI parses
# ---------------------------------------------------------------------------


def _parse_sse(body_text: str) -> list[tuple[str, dict]]:
    """Parse SSE response body into [(event, data_obj), ...] pairs."""
    events: list[tuple[str, dict]] = []
    blocks = body_text.split("\n\n")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_text = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_text = line[len("data:") :].strip()
        if event_name and data_text is not None:
            events.append((event_name, json.loads(data_text)))
    return events


class TestResponsesStream:
    def test_stream_emits_codex_required_events(self, responses_client):
        """Codex CLI hard-requires ``response.created`` first and
        ``response.completed`` last; ``response.output_text.delta``
        events carry the assistant text."""
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        event_names = [e[0] for e in events]

        # First event must be response.created — Codex sets state from it.
        assert event_names[0] == "response.created"
        # Last event must be response.completed — Codex treats absence
        # as a hard failure ("stream closed before response.completed").
        assert event_names[-1] == "response.completed"
        # At least one text delta in between.
        assert "response.output_text.delta" in event_names
        # Message item opened and closed.
        assert "response.output_item.added" in event_names
        assert "response.output_item.done" in event_names

    def test_stream_text_deltas_concatenate_to_full_reply(self, responses_client):
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        deltas = [
            d["delta"] for (name, d) in events if name == "response.output_text.delta"
        ]
        # The mock engine emits "Hello", " from", " rapid".
        assert "".join(deltas) == "Hello from rapid"

    def test_stream_completed_event_carries_usage(self, responses_client):
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        completed = [d for (name, d) in events if name == "response.completed"]
        assert len(completed) == 1
        usage = completed[0]["response"]["usage"]
        # Mock engine returns prompt=3, completion=3 (last chunk).
        assert usage["input_tokens"] == 3
        assert usage["output_tokens"] == 3
        assert usage["total_tokens"] == 6
