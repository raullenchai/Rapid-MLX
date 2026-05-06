# SPDX-License-Identifier: Apache-2.0
"""Auth regressions for Anthropic-compatible HTTP routes."""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI
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
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = "stop"
    new_text: str = ""
    finished: bool = True
    logprobs: Any = None
    channel: str | None = None


class _Engine:
    preserve_native_tool_format = False

    def __init__(self):
        self.calls = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text="hello",
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
        )


def _install_lightweight_engine_modules(monkeypatch):
    """Avoid importing MLX-backed engine package in route-level HTTP tests."""
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
    "vllm_mlx.routes.anthropic",
)
_PARENT_ATTRS_UNDER_LIGHTWEIGHT_ENGINE = (
    ("vllm_mlx", "config"),
    ("vllm_mlx", "engine"),
    ("vllm_mlx.config", "server_config"),
    ("vllm_mlx.engine", "base"),
    ("vllm_mlx.middleware", "auth"),
    ("vllm_mlx.service", "helpers"),
    ("vllm_mlx.routes", "anthropic"),
)
_MISSING = object()


@pytest.fixture
def anthropic_client(monkeypatch):
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
    from vllm_mlx.routes.anthropic import router

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


def _messages_payload() -> dict:
    return {
        "model": "test-model",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_anthropic_messages_requires_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post("/v1/messages", json=_messages_payload())

    assert response.status_code == 401
    assert response.json()["detail"] == "API key required"
    assert engine.calls == []


def test_anthropic_messages_rejects_invalid_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.calls == []


def test_anthropic_messages_rejects_invalid_x_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"x-api-key": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.calls == []


def test_anthropic_messages_accepts_valid_x_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"x-api-key": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]
    assert len(engine.calls) == 1


def test_anthropic_messages_accepts_valid_bearer_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"Authorization": "Bearer test-secret"},
    )

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]
    assert len(engine.calls) == 1


def test_anthropic_messages_rejects_mixed_invalid_credentials(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={
            "Authorization": "Bearer wrong-secret",
            "x-api-key": "test-secret",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.calls == []


def test_anthropic_messages_rejects_mixed_invalid_x_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={
            "Authorization": "Bearer test-secret",
            "x-api-key": "wrong-secret",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.calls == []


def test_anthropic_messages_respects_rate_limit(anthropic_client):
    client = anthropic_client.client
    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    headers = {"x-api-key": "test-secret"}
    first = client.post("/v1/messages", json=_messages_payload(), headers=headers)
    second = client.post("/v1/messages", json=_messages_payload(), headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_anthropic_messages_rate_limit_uses_same_identity_for_valid_headers(
    anthropic_client,
):
    client = anthropic_client.client
    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    first = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"x-api-key": "test-secret"},
    )
    second = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={
            "Authorization": "Bearer test-secret",
            "x-api-key": "test-secret",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_anthropic_messages_rate_limit_treats_x_api_key_and_bearer_as_same_key(
    anthropic_client,
):
    client = anthropic_client.client
    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    first = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"x-api-key": "test-secret"},
    )
    second = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"Authorization": "Bearer test-secret"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_anthropic_messages_rate_limit_treats_bearer_and_x_api_key_as_same_key(
    anthropic_client,
):
    client = anthropic_client.client
    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    first = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"Authorization": "Bearer test-secret"},
    )
    second = client.post(
        "/v1/messages",
        json=_messages_payload(),
        headers={"x-api-key": "test-secret"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_anthropic_count_tokens_requires_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "API key required"
    assert engine.tokenizer.calls == []


def test_anthropic_count_tokens_rejects_invalid_bearer_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.tokenizer.calls == []


def test_anthropic_count_tokens_rejects_invalid_x_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"x-api-key": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.tokenizer.calls == []


def test_anthropic_count_tokens_rejects_mixed_invalid_bearer(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={
            "Authorization": "Bearer wrong-secret",
            "x-api-key": "test-secret",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.tokenizer.calls == []


def test_anthropic_count_tokens_rejects_mixed_invalid_x_api_key(anthropic_client):
    client = anthropic_client.client
    engine = anthropic_client.engine

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={
            "Authorization": "Bearer test-secret",
            "x-api-key": "wrong-secret",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"
    assert engine.tokenizer.calls == []


def test_anthropic_count_tokens_respects_rate_limit(anthropic_client):
    client = anthropic_client.client
    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    payload = {"messages": [{"role": "user", "content": "hello"}]}
    headers = {"x-api-key": "test-secret"}
    first = client.post("/v1/messages/count_tokens", json=payload, headers=headers)
    second = client.post("/v1/messages/count_tokens", json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_anthropic_count_tokens_rate_limit_treats_header_forms_as_same_key(
    anthropic_client,
):
    client = anthropic_client.client
    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    payload = {"messages": [{"role": "user", "content": "hello"}]}
    first = client.post(
        "/v1/messages/count_tokens",
        json=payload,
        headers={"x-api-key": "test-secret"},
    )
    second = client.post(
        "/v1/messages/count_tokens",
        json=payload,
        headers={"Authorization": "Bearer test-secret"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_shared_rate_limit_ignores_x_api_key_for_non_anthropic_routes(
    anthropic_client,
):
    from vllm_mlx.middleware.auth import check_rate_limit

    anthropic_client.rate_limiter.enabled = True
    anthropic_client.rate_limiter.requests_per_minute = 1

    app = FastAPI()

    @app.get("/test", dependencies=[Depends(check_rate_limit)])
    async def test_endpoint():
        return {"ok": True}

    client = TestClient(app)
    first = client.get("/test", headers={"x-api-key": "one"})
    second = client.get("/test", headers={"x-api-key": "two"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_configure_rate_limiter_updates_shared_anthropic_dependency(
    anthropic_client,
):
    from vllm_mlx.middleware.auth import configure_rate_limiter

    configured = configure_rate_limiter(requests_per_minute=1, enabled=True)

    assert configured is anthropic_client.rate_limiter

    payload = {"messages": [{"role": "user", "content": "hello"}]}
    headers = {"x-api-key": "test-secret"}
    first = anthropic_client.client.post(
        "/v1/messages/count_tokens", json=payload, headers=headers
    )
    second = anthropic_client.client.post(
        "/v1/messages/count_tokens", json=payload, headers=headers
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"].startswith("Rate limit exceeded.")


def test_server_startup_configures_shared_rate_limiter():
    server_source = Path("vllm_mlx/server.py").read_text()
    cli_source = Path("vllm_mlx/cli.py").read_text()

    assert "configure_rate_limiter(args.rate_limit" in server_source
    assert "configure_rate_limiter(args.rate_limit" in cli_source
    assert "_rate_limiter = RateLimiter(requests_per_minute=args.rate_limit" not in (
        server_source + cli_source
    )


def test_anthropic_count_tokens_accepts_valid_bearer_api_key(anthropic_client):
    client = anthropic_client.client

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 5}


def test_anthropic_count_tokens_accepts_valid_x_api_key(anthropic_client):
    client = anthropic_client.client

    response = client.post(
        "/v1/messages/count_tokens",
        json={"messages": [{"role": "user", "content": "hello"}]},
        headers={"x-api-key": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 5}
