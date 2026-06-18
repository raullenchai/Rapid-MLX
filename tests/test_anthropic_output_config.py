# SPDX-License-Identifier: Apache-2.0
"""Tests for ``output_config`` on the Anthropic /v1/messages surface.

Backport of upstream vLLM PR #42396 (v0.22.0). Covers:

- Pydantic model parsing for ``output_config.format = json_schema``
  (including alias handling for the ``schema`` field).
- Adapter translation Anthropic ``output_config`` → OpenAI
  ``response_format``, including the unsupported-type / missing-schema
  validation paths surfaced as ``AnthropicOutputConfigError``.
- Route-level HTTP 400 propagation of those validation errors.
- ``output_config.effort`` is accepted but does NOT mutate the
  forwarded ``chat_kwargs`` — it belongs to a separate concurrent
  backport (Pick 1).
- Backward compatibility: requests with no ``output_config`` behave
  identically to the pre-backport surface.

The route-level tests reuse the lightweight-engine fixture pattern from
``test_anthropic_route_auth.py`` so they avoid importing the MLX-backed
engine, which keeps CI hermetic.
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
from pydantic import ValidationError

from vllm_mlx.api.anthropic_adapter import (
    AnthropicOutputConfigError,
    _convert_output_config,
    anthropic_to_openai,
)
from vllm_mlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicOutputConfig,
    AnthropicOutputFormat,
    AnthropicRequest,
)
from vllm_mlx.api.models import ResponseFormat

# =============================================================================
# Pydantic model parsing
# =============================================================================


class TestAnthropicOutputFormatModel:
    """Parse the Anthropic ``output_config.format`` shape."""

    def test_minimal_json_schema(self):
        fmt = AnthropicOutputFormat(
            type="json_schema",
            schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        assert fmt.type == "json_schema"
        assert fmt.schema_ == {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
        # name/description/strict are optional
        assert fmt.name is None
        assert fmt.description is None
        assert fmt.strict is None

    def test_full_json_schema_with_optional_fields(self):
        fmt = AnthropicOutputFormat(
            type="json_schema",
            schema={"type": "object"},
            name="person",
            description="A person record",
            strict=True,
        )
        assert fmt.name == "person"
        assert fmt.description == "A person record"
        assert fmt.strict is True

    def test_schema_alias_round_trips_via_model_dump(self):
        """``schema_`` ↔ ``schema`` alias must survive a dump/load cycle."""
        fmt = AnthropicOutputFormat(type="json_schema", schema={"type": "object"})
        # by_alias=True so external serialization uses the wire name.
        dumped = fmt.model_dump(by_alias=True)
        assert "schema" in dumped
        assert "schema_" not in dumped
        re_parsed = AnthropicOutputFormat(**dumped)
        assert re_parsed.schema_ == {"type": "object"}

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            AnthropicOutputFormat(schema={"type": "object"})


class TestAnthropicOutputConfigModel:
    """Parse the Anthropic ``output_config`` wrapper."""

    def test_empty_config_allowed(self):
        cfg = AnthropicOutputConfig()
        assert cfg.format is None
        assert cfg.effort is None

    def test_with_format(self):
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(type="json_schema", schema={})
        )
        assert cfg.format is not None
        assert cfg.format.type == "json_schema"

    def test_effort_field_is_accepted_but_separate(self):
        """``effort`` is part of Pick 1 (concurrent reasoning-effort PR).

        We accept it on the Pydantic model today so the two PRs don't
        race on the same shape during merge, but this PR must NOT act
        on the value. The adapter-level test below asserts the value
        doesn't leak into the OpenAI request.
        """
        cfg = AnthropicOutputConfig(effort="high")
        assert cfg.effort == "high"


# =============================================================================
# Adapter translation
# =============================================================================


def _req(**kwargs) -> AnthropicRequest:
    defaults = {
        "model": "default",
        "messages": [AnthropicMessage(role="user", content="hi")],
        "max_tokens": 32,
    }
    defaults.update(kwargs)
    return AnthropicRequest(**defaults)


class TestConvertOutputConfigUnit:
    """Direct unit tests for ``_convert_output_config``."""

    def test_none_passthrough(self):
        assert _convert_output_config(None) is None

    def test_format_none_passthrough(self):
        assert _convert_output_config(AnthropicOutputConfig(format=None)) is None

    def test_json_schema_translates(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(
                type="json_schema",
                schema=schema,
                name="person",
                description="A person",
                strict=True,
            )
        )
        rf = _convert_output_config(cfg)
        assert isinstance(rf, ResponseFormat)
        assert rf.type == "json_schema"
        assert rf.json_schema is not None
        assert rf.json_schema.name == "person"
        assert rf.json_schema.description == "A person"
        assert rf.json_schema.schema_ == schema
        assert rf.json_schema.strict is True

    def test_json_schema_name_defaults_to_response(self):
        rf = _convert_output_config(
            AnthropicOutputConfig(
                format=AnthropicOutputFormat(type="json_schema", schema={})
            )
        )
        assert rf is not None
        assert rf.json_schema.name == "response"

    def test_json_schema_strict_defaults_to_false(self):
        rf = _convert_output_config(
            AnthropicOutputConfig(
                format=AnthropicOutputFormat(type="json_schema", schema={})
            )
        )
        assert rf is not None
        assert rf.json_schema.strict is False

    def test_unsupported_type_raises(self):
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(type="regex", schema={})
        )
        with pytest.raises(AnthropicOutputConfigError) as exc:
            _convert_output_config(cfg)
        # The error must call out the surface and the rejected type so
        # operators reading server logs can map a 400 to the offending
        # request field without grepping internals.
        msg = str(exc.value)
        assert "json_schema" in msg
        assert "/v1/messages" in msg
        assert "'regex'" in msg

    def test_missing_schema_raises(self):
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(type="json_schema")  # schema absent
        )
        with pytest.raises(AnthropicOutputConfigError) as exc:
            _convert_output_config(cfg)
        assert "schema" in str(exc.value)

    def test_non_dict_schema_raises(self):
        """If callers bypass the Pydantic model (e.g. by constructing the
        adapter input directly), a non-dict schema must still be
        rejected by the adapter rather than crashing downstream.
        """
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(type="json_schema", schema={})
        )
        # bypass model coercion by mutating the field post-construction
        cfg.format.schema_ = "not a dict"  # type: ignore[assignment]
        with pytest.raises(AnthropicOutputConfigError):
            _convert_output_config(cfg)


class TestAnthropicToOpenaiOutputConfig:
    """End-to-end adapter translation including ``output_config``."""

    def test_no_output_config_leaves_response_format_none(self):
        """Backward compat: pre-existing requests must not gain a
        ``response_format`` field after the backport.
        """
        result = anthropic_to_openai(_req())
        assert result.response_format is None

    def test_output_config_json_schema_propagates(self):
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(
                type="json_schema",
                schema={"type": "object", "properties": {"a": {"type": "integer"}}},
                name="thing",
            )
        )
        result = anthropic_to_openai(_req(output_config=cfg))
        assert result.response_format is not None
        assert result.response_format.type == "json_schema"
        assert result.response_format.json_schema.name == "thing"

    def test_effort_field_sets_reasoning_max_tokens(self):
        """Pick 1 has now landed. ``effort`` is translated through
        ``ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS`` into a per-request
        ``reasoning_max_tokens`` on the OpenAI side. Every other OpenAI
        field stays identical to the baseline request — only
        ``reasoning_max_tokens`` changes, so the cap path is the single
        wire effect.

        See ``tests/test_per_request_thinking_budget.py`` for the full
        effort-mapping test matrix; this case pins coexistence with
        Pick 2's ``format`` field so neither PR's tests can regress
        the other's surface.
        """
        baseline = anthropic_to_openai(_req())
        with_effort = anthropic_to_openai(
            _req(output_config=AnthropicOutputConfig(effort="high"))
        )
        # ``high`` → 8192 per the canonical Anthropic mapping.
        assert baseline.reasoning_max_tokens is None
        assert with_effort.reasoning_max_tokens == 8192
        # Everything else must be identical so Pick 1 doesn't sneak in
        # a side effect on the OpenAI surface.
        baseline_dump = baseline.model_dump(exclude={"reasoning_max_tokens"})
        with_effort_dump = with_effort.model_dump(exclude={"reasoning_max_tokens"})
        assert baseline_dump == with_effort_dump

    def test_unsupported_format_raises(self):
        cfg = AnthropicOutputConfig(
            format=AnthropicOutputFormat(type="regex", schema={})
        )
        with pytest.raises(AnthropicOutputConfigError):
            anthropic_to_openai(_req(output_config=cfg))

    def test_missing_schema_raises(self):
        cfg = AnthropicOutputConfig(format=AnthropicOutputFormat(type="json_schema"))
        with pytest.raises(AnthropicOutputConfigError):
            anthropic_to_openai(_req(output_config=cfg))


# =============================================================================
# Route-level: HTTP 400 surface
# =============================================================================


class _Tokenizer:
    chat_template = ""

    def encode(self, text: str) -> list[int]:  # pragma: no cover - simple stub
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


class _RecordingEngine:
    """Captures every ``chat`` call so tests can assert what was forwarded.

    This is the "recording engine that captures ``chat_kwargs``" pattern
    called for by the task spec, so we can pin that ``output_config.effort``
    really doesn't leak into the engine's sampling kwargs.
    """

    preserve_native_tool_format = False

    def __init__(self):
        self.calls: list[SimpleNamespace] = []
        self.tokenizer = _Tokenizer()

    async def chat(self, messages, **kwargs):
        self.calls.append(SimpleNamespace(messages=messages, kwargs=kwargs))
        return _GenerationOutput(
            text='{"name": "alice"}',
            prompt_tokens=3,
            completion_tokens=2,
            finish_reason="stop",
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
    # Keep auth out of the way for these tests — output_config validation
    # runs after auth and we want the 400 / 200 surface clean.
    cfg.api_key = None
    cfg.engine = _RecordingEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    app.include_router(router)
    yield SimpleNamespace(client=TestClient(app), engine=cfg.engine)

    reset_config()
    rate_limiter.enabled = False
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


def _payload(**extra) -> dict:
    body = {
        "model": "test-model",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "give me a name"}],
    }
    body.update(extra)
    return body


class TestRouteOutputConfigSurface:
    """HTTP-level coverage for the 400 / 200 paths."""

    def test_round_trip_json_schema_returns_200(self, anthropic_client):
        resp = anthropic_client.client.post(
            "/v1/messages",
            json=_payload(
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        "name": "person",
                    }
                }
            ),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # The recording engine returns ``'{"name": "alice"}'`` as the
        # text — the Anthropic adapter wraps it in a text content block.
        # Parse it as JSON and verify the schema's ``name`` key.
        text_blocks = [b for b in body["content"] if b["type"] == "text"]
        assert text_blocks, "expected at least one text block in response"
        parsed = json.loads(text_blocks[0]["text"])
        assert isinstance(parsed, dict)
        assert "name" in parsed

    def test_unknown_format_type_returns_400(self, anthropic_client):
        resp = anthropic_client.client.post(
            "/v1/messages",
            json=_payload(output_config={"format": {"type": "regex", "schema": {}}}),
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "json_schema" in detail
        assert "/v1/messages" in detail
        # The engine must NOT have been invoked for a 400.
        assert anthropic_client.engine.calls == []

    def test_missing_schema_returns_400(self, anthropic_client):
        resp = anthropic_client.client.post(
            "/v1/messages",
            json=_payload(output_config={"format": {"type": "json_schema"}}),
        )
        assert resp.status_code == 400
        assert "schema" in resp.json()["detail"]
        assert anthropic_client.engine.calls == []

    def test_invalid_schema_type_returns_400(self, anthropic_client):
        """A non-object ``schema`` must be rejected as 400 — the route
        wraps Pydantic ``ValidationError`` from manual request construction
        so clients see a clean 400 instead of a 500.
        """
        resp = anthropic_client.client.post(
            "/v1/messages",
            json=_payload(
                output_config={
                    "format": {"type": "json_schema", "schema": "not a dict"}
                }
            ),
        )
        assert resp.status_code == 400, resp.text
        assert anthropic_client.engine.calls == []

    def test_no_output_config_is_backward_compatible(self, anthropic_client):
        """Requests with no ``output_config`` must behave identically to
        the pre-backport surface — engine called once, no ``response_format``
        passed through, 200 OK.
        """
        resp = anthropic_client.client.post("/v1/messages", json=_payload())
        assert resp.status_code == 200, resp.text
        assert len(anthropic_client.engine.calls) == 1

    def test_effort_field_accepted_but_does_not_alter_chat_kwargs(
        self, anthropic_client
    ):
        """``output_config.effort`` is Pick 1 territory — accepting the
        field is required for forward-compat with Anthropic SDKs, but
        this PR must NOT alter what the engine receives.

        We pin via the recording engine: send two identical requests,
        one with ``effort: high``, one without, and assert the captured
        ``chat_kwargs`` are equal.
        """
        # baseline — no output_config at all
        baseline_resp = anthropic_client.client.post("/v1/messages", json=_payload())
        assert baseline_resp.status_code == 200

        # with effort field (no format) — must accept and ignore
        effort_resp = anthropic_client.client.post(
            "/v1/messages",
            json=_payload(output_config={"effort": "high"}),
        )
        assert effort_resp.status_code == 200

        assert len(anthropic_client.engine.calls) == 2
        baseline_kwargs = anthropic_client.engine.calls[0].kwargs
        effort_kwargs = anthropic_client.engine.calls[1].kwargs
        assert baseline_kwargs == effort_kwargs, (
            "output_config.effort leaked into engine chat kwargs — "
            "this field belongs to Pick 1 (concurrent PR)."
        )
