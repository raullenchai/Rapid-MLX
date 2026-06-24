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
    is_mllm = False

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
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
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
    # H-17: mirror production wiring — the route relies on the global
    # ``pydantic.ValidationError`` handler to map bad bodies to the
    # sanitized 400 envelope. The earlier fixture omitted handler
    # install and the route's now-removed per-route ``try/except`` was
    # producing the 400 instead, which masked the leak this fixture is
    # meant to gate against.
    install_exception_handlers(app)
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
        # H-17: fixture now installs the global exception handlers
        # (mirror production). Auth ``HTTPException(detail="...")`` is
        # wrapped in the canonical OpenAI envelope by
        # ``_http_error_response`` — assert the new shape, not the raw
        # FastAPI default ``{"detail": "..."}`` that fixture-less tests
        # would have seen.
        assert response.json()["error"]["message"] == "API key required"
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

    def test_input_image_reaches_mllm_engine_as_multimodal_content(
        self, responses_client
    ):
        client = responses_client.client
        engine = responses_client.engine
        engine.is_mllm = True

        response = client.post(
            "/v1/responses",
            json=_payload(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,abc",
                            },
                        ],
                    }
                ],
            ),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200, response.text
        sent = engine.calls[-1].messages
        content = sent[0]["content"]
        assert isinstance(content, list)
        assert [part["type"] for part in content] == ["text", "image_url"]
        assert content[1]["image_url"]["url"] == "data:image/png;base64,abc"

    def test_mllm_context_precheck_counts_text_without_image_payload(
        self, responses_client, monkeypatch
    ):
        from vllm_mlx.routes import responses as responses_route

        client = responses_client.client
        engine = responses_client.engine
        engine.is_mllm = True
        captured = {}

        def _capture_context_messages(_engine, messages, **_kwargs):
            captured["messages"] = messages

        monkeypatch.setattr(
            responses_route,
            "enforce_context_length_for_messages",
            _capture_context_messages,
        )

        response = client.post(
            "/v1/responses",
            json=_payload(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,abc",
                            },
                        ],
                    }
                ],
            ),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 200, response.text
        assert captured["messages"] == [{"role": "user", "content": "Describe this"}]
        sent = engine.calls[-1].messages
        assert sent[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,abc"

    def test_context_precheck_unexpected_error_is_not_swallowed(
        self, responses_client, monkeypatch
    ):
        from vllm_mlx.routes import responses as responses_route

        client = responses_client.client

        def _raise_unexpected(_engine, _openai_request):
            raise RuntimeError("context precheck bug")

        monkeypatch.setattr(
            responses_route,
            "_prepare_messages_for_context_check",
            _raise_unexpected,
        )

        with pytest.raises(RuntimeError, match="context precheck bug"):
            client.post(
                "/v1/responses",
                json=_payload(),
                headers={"Authorization": "Bearer test-secret"},
            )

    def test_mllm_message_prepare_accepts_normalized_object_style_messages(self):
        """Responses MLLM path accepts Chat-normalized object-style messages."""
        from vllm_mlx.routes.responses import _prepare_messages_for_engine

        msg = SimpleNamespace(
            role="user",
            content=[
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        )
        request = SimpleNamespace(messages=[msg])
        engine = SimpleNamespace(is_mllm=True)

        sent = _prepare_messages_for_engine(engine, request)

        assert sent == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]

    def test_mllm_message_prepare_rejects_raw_responses_content_blocks(self):
        from vllm_mlx.routes.responses import _prepare_messages_for_engine

        msg = SimpleNamespace(
            role="user",
            content=[
                {"type": "input_text", "text": "Describe this"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc"},
            ],
        )
        request = SimpleNamespace(messages=[msg])
        engine = SimpleNamespace(is_mllm=True)

        with pytest.raises(ValueError, match="must be normalized"):
            _prepare_messages_for_engine(engine, request)

    def test_message_prepare_defaults_missing_native_tool_flag(self):
        from vllm_mlx.routes.responses import _prepare_messages_for_engine

        request = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])

        assert _prepare_messages_for_engine(
            SimpleNamespace(is_mllm=False), request
        ) == [{"role": "user", "content": "hi"}]

    def test_input_image_rejected_on_text_only_engine(self, responses_client):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,abc",
                            },
                        ],
                    }
                ],
            ),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 400, response.text
        body = response.json()
        msg = body.get("detail") or body.get("error", {}).get("message", "")
        assert "image inputs" in msg
        assert engine.calls == []

    @pytest.mark.parametrize(
        ("content_part", "expected"),
        [
            ({"type": "input_text"}, "input_text.text is required"),
            (
                {"type": "input_text", "text": ""},
                "input_text.text must be a non-empty string",
            ),
            ({"type": "output_text"}, "output_text.text is required"),
            (
                {"type": "input_audio", "input_audio": {"data": "base64data"}},
                "input_audio content blocks are not supported",
            ),
        ],
    )
    def test_malformed_text_content_block_returns_400_not_empty_prompt(
        self, responses_client, content_part, expected
    ):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [content_part],
                    }
                ],
            ),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 400, response.text
        body = response.json()
        msg = body.get("detail") or body.get("error", {}).get("message", "")
        assert expected in msg
        assert engine.calls == []

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (None, "Responses message content is required"),
            ([], "Responses message content must not be empty"),
        ],
    )
    def test_empty_message_content_returns_400_not_empty_prompt(
        self, responses_client, content, expected
    ):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": content,
                    }
                ],
            ),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 400, response.text
        body = response.json()
        msg = body.get("detail") or body.get("error", {}).get("message", "")
        assert expected in msg
        assert engine.calls == []


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
        # H-17: fixture installs the global exception handlers (mirror
        # production), so the route's ``HTTPException`` is wrapped in
        # the canonical OpenAI envelope by ``_http_error_response``.
        assert "previous_response_id" in response.json()["error"]["message"]
        # Engine must not have been called.
        assert engine.calls == []


class TestResponsesPydanticValidation:
    """Codex bundled-review finding on the v0.7.32 bundle: #685 added a
    strict ``reasoning_max_tokens`` ``model_validator(mode="before")``
    that raises ``ValidationError`` on bad input (``0``, ``true``,
    ``"100"``). Since ``create_response()`` constructs
    ``ResponsesRequest(**body)`` manually outside of FastAPI's body
    binding, an uncaught ``ValidationError`` would surface as 500. The
    route wraps construction in try/except and re-raises as 400 —
    matches the same pattern in ``routes/anthropic.py``.
    """

    @pytest.mark.parametrize(
        "bad_value",
        [0, -1, True, False, "100"],
        ids=["zero", "negative", "bool-true", "bool-false", "string"],
    )
    def test_invalid_reasoning_max_tokens_returns_400(
        self, responses_client, bad_value
    ):
        client = responses_client.client
        engine = responses_client.engine

        response = client.post(
            "/v1/responses",
            json=_payload(reasoning_max_tokens=bad_value),
            headers={"Authorization": "Bearer test-secret"},
        )

        assert response.status_code == 400, (
            f"reasoning_max_tokens={bad_value!r} must surface as 400, "
            f"got {response.status_code}: {response.text}"
        )
        assert "reasoning_max_tokens" in response.text
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


# ---------------------------------------------------------------------------
# R10-C3 — streaming spec-conformance regression guard.
# Sven r10-R1 caught 0.8.11 emitting zero ``response.output_text.delta``
# events and a ``response.completed`` payload with no ``output`` field; the
# openai-python SDK then crashed on stream consumption. These tests pin the
# SSE event sequence and final payload shape so that regression cannot
# silently come back.
# ---------------------------------------------------------------------------


class TestResponsesStreamR10C3:
    def test_stream_emits_in_progress_after_created(self, responses_client):
        """``response.in_progress`` must appear immediately after
        ``response.created`` — the openai-python SDK transitions internal
        state on it and skipping leaves the parser half-initialized."""
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        names = [e[0] for e in events]
        assert names[0] == "response.created"
        assert names[1] == "response.in_progress"

    def test_stream_emits_content_part_added_before_first_delta(self, responses_client):
        """``response.content_part.added`` must land between
        ``response.output_item.added`` and the first
        ``response.output_text.delta`` — required by the OpenAI Responses
        SSE spec for the SDK to materialize the output_text part."""
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        names = [e[0] for e in _parse_sse(body)]
        item_added = names.index("response.output_item.added")
        first_delta = names.index("response.output_text.delta")
        content_added = names.index("response.content_part.added")
        assert item_added < content_added < first_delta

    def test_stream_emits_content_part_done_and_text_done(self, responses_client):
        """``response.output_text.done`` and ``response.content_part.done``
        must precede the message ``response.output_item.done`` event."""
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        names = [e[0] for e in events]
        text_done = names.index("response.output_text.done")
        part_done = names.index("response.content_part.done")
        item_done = names.index("response.output_item.done")
        assert text_done < part_done < item_done

        # ``output_text.done`` must carry the full concatenated text.
        text_done_payload = next(
            d for (n, d) in events if n == "response.output_text.done"
        )
        assert text_done_payload["text"] == "Hello from rapid"

        # ``content_part.done`` must carry the finalized output_text block.
        part_done_payload = next(
            d for (n, d) in events if n == "response.content_part.done"
        )
        assert part_done_payload["part"]["type"] == "output_text"
        assert part_done_payload["part"]["text"] == "Hello from rapid"

    def test_completed_payload_carries_output_array(self, responses_client):
        """``response.completed.response.output`` must be a non-empty list
        carrying the assistant message with the concatenated delta text —
        Sven r10-R1 saw this field MISSING entirely on 0.8.11 and the
        openai-python SDK raised on Response.output validation.

        R12-M3 (Mira r12 R-4): the OpenAI Responses spec requires the
        ``reasoning`` item to land BEFORE the ``message`` item on the
        wire, so the assistant message item now sits at ``output[1]`` with
        the leading reasoning item at ``output[0]``.
        """
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        completed = next(d for (n, d) in events if n == "response.completed")
        response_obj = completed["response"]
        # The R10-C3 regression: ``output`` was missing entirely.
        assert "output" in response_obj, (
            "response.completed payload must carry the `output` array — "
            "Sven r10-R1 caught 0.8.11 dropping it, breaking openai-python SDK"
        )
        output = response_obj["output"]
        assert isinstance(output, list)
        assert len(output) >= 2  # reasoning + message (R12-M3)
        # R12-M3: leading reasoning item at index 0.
        assert output[0]["type"] == "reasoning"
        # Message item now at index 1, preserving its prior shape.
        message_item = next(item for item in output if item["type"] == "message")
        assert message_item["status"] == "completed"
        assert message_item["role"] == "assistant"
        assert message_item["content"][0]["type"] == "output_text"
        assert message_item["content"][0]["text"] == "Hello from rapid"

    def test_completed_output_text_equals_concatenated_deltas(self, responses_client):
        """The concatenated ``response.output_text.delta`` payloads MUST
        equal the assistant message's ``content[0].text`` in
        ``response.completed.response.output[]``. This pins the
        streaming-vs-completed invariant — the streaming payload
        reconstructs the same final response object the non-streaming
        path returns.

        R12-M3 (Mira r12 R-4): the message item is no longer guaranteed
        to sit at ``output[0]`` (a leading ``reasoning`` item lands first
        per spec), so the lookup is by ``type=="message"``.
        """
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        deltas = "".join(
            d["delta"] for (n, d) in events if n == "response.output_text.delta"
        )
        completed = next(d for (n, d) in events if n == "response.completed")
        message_item = next(
            item
            for item in completed["response"]["output"]
            if item["type"] == "message"
        )
        final_text = message_item["content"][0]["text"]
        assert deltas == final_text == "Hello from rapid"

    def test_stream_event_payloads_validate_against_sdk_event_models(
        self, responses_client
    ):
        """Each SSE event payload must validate against the corresponding
        openai-python event model. Pre-fix, the SDK's stream consumer
        crashed on terminal events because event payload models marked
        fields like ``output``, ``part``, ``content_index`` required and
        our emitter dropped them.
        """
        sdk_streaming = pytest.importorskip("openai.types.responses")

        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)

        # Spot-check the three previously-broken or newly-added events.
        # If the SDK schemas accept these, the AsyncResponseStream loop
        # can walk the whole stream end to end.
        from openai.types.responses import (
            Response,
            ResponseCompletedEvent,
            ResponseContentPartAddedEvent,
            ResponseContentPartDoneEvent,
            ResponseTextDeltaEvent,
            ResponseTextDoneEvent,
        )

        completed = next(d for (n, d) in events if n == "response.completed")
        ResponseCompletedEvent.model_validate(completed)
        Response.model_validate(completed["response"])

        first_delta = next(d for (n, d) in events if n == "response.output_text.delta")
        ResponseTextDeltaEvent.model_validate(first_delta)

        part_added = next(d for (n, d) in events if n == "response.content_part.added")
        ResponseContentPartAddedEvent.model_validate(part_added)

        part_done = next(d for (n, d) in events if n == "response.content_part.done")
        ResponseContentPartDoneEvent.model_validate(part_done)

        text_done = next(d for (n, d) in events if n == "response.output_text.done")
        ResponseTextDoneEvent.model_validate(text_done)

    def test_stream_consumable_by_openai_python_sdk(self, responses_client):
        """End-to-end guard: the openai-python SDK MUST be able to walk
        our SSE stream without raising. Sven r10-R1 caught 0.8.11 making
        ``openai.AsyncOpenAI(...).responses.create(stream=True)`` raise
        on terminal-event consumption because ``response.completed.output``
        was missing (a required field on ``openai.types.responses.Response``).

        The SDK is an optional dev dep; skip cleanly if absent so this
        suite stays portable on CI without an ``openai`` install.
        """
        openai_types = pytest.importorskip("openai.types.responses")
        Response = openai_types.Response

        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        completed = next(d for (n, d) in events if n == "response.completed")
        # The SDK validates the terminal payload against this model. If
        # ``output`` is missing or malformed it raises ValidationError.
        parsed = Response.model_validate(completed["response"])
        assert parsed.output, "SDK rejected completed payload — output empty"
        assert parsed.status == "completed"

    def test_completed_carries_required_top_level_fields(self, responses_client):
        """``response.completed.response`` must include id / created_at /
        status / model / output / usage so the openai-python SDK can
        construct the terminal Response object without a ValidationError."""
        client = responses_client.client

        with client.stream(
            "POST",
            "/v1/responses",
            json=_payload(stream=True),
            headers={"Authorization": "Bearer test-secret"},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse(body)
        completed = next(d for (n, d) in events if n == "response.completed")
        resp_obj = completed["response"]
        for fld in ("id", "created_at", "status", "model", "output", "usage"):
            assert fld in resp_obj, f"completed.response is missing `{fld}`"
        assert resp_obj["status"] == "completed"
