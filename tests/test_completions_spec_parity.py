# SPDX-License-Identifier: Apache-2.0
"""Spec parity for legacy ``/v1/completions`` (F-152 + F-153).

Pre-fix the route silently accepted ``n``, ``best_of``, ``echo``, and
``logprobs:true`` while doing nothing with them — a silent-compat lie
that broke OpenAI SDK clients (wrong billing math, missing logprobs in
eval harnesses). The pydantic schema also typed ``logprobs`` as
``bool``, so the canonical ``logprobs=5`` SDK form bounced with a
422 ``bool_parsing`` error that never mentioned the actual mismatch.

Each test below pins one piece of the new contract so a future
refactor cannot silently regress the spec parity. Tests run as
isolated FastAPI test-clients with a ``MagicMock`` engine; no real
model load, no GPU, no port — they live in the same file family as
``test_api_validation_bundle.py``.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def patched_config():
    """Mirror of ``test_api_validation_bundle.patched_config``.

    Patches select fields on the global cfg singleton and restores
    them on test exit so each test sees a clean config.
    """
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


def _build_completions_app(patch_cfg, monkeypatch, *, engine_factory=None):
    """Wire a stub completions app with a MagicMock engine.

    ``engine_factory`` lets a test customise the engine (e.g. wire a
    streaming async generator into ``stream_generate``). When omitted,
    a MagicMock with sensible defaults is used.
    """
    from vllm_mlx.routes import completions as comp_route

    app = FastAPI()
    app.include_router(comp_route.router)

    engine = engine_factory() if engine_factory else MagicMock()
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)
    # ``enforce_context_length_for_prompt`` calls into the engine's
    # tokenizer; short-circuit so the schema-validation tests don't
    # depend on a tokenizer.
    monkeypatch.setattr(
        comp_route, "enforce_context_length_for_prompt", lambda *_a, **_kw: None
    )
    return TestClient(app, raise_server_exceptions=False), engine


# ---------------------------------------------------------------------------
# F-152: n / best_of / echo behaviour
# ---------------------------------------------------------------------------


class TestN:
    """``n>1`` must 400 (mirroring the chat-completions route)."""

    def test_n_above_one_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "n": 3},
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "n > 1" in detail or "n>1" in detail.replace(" ", "")

    def test_n_one_is_accepted(self, patched_config, monkeypatch):
        """``n: 1`` is the OpenAI default — must not trip the new guard."""

        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "n": 1, "max_tokens": 4},
        )
        assert r.status_code == 200

    def test_n_omitted_is_accepted(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "max_tokens": 4},
        )
        assert r.status_code == 200


class TestBestOf:
    """``best_of>1`` must 400 (no server-side reranker)."""

    def test_best_of_above_one_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "best_of": 5},
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "best_of" in detail

    def test_best_of_one_is_accepted(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput()

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "best_of": 1},
        )
        assert r.status_code == 200


class TestEcho:
    """``echo: true`` must prepend the prompt to ``choices[0].text``."""

    def test_echo_true_prepends_prompt(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput(text=" world!")

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "Hello",
                "max_tokens": 4,
                "echo": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"] == "Hello world!"

    def test_echo_false_does_not_prepend(self, patched_config, monkeypatch):
        async def _fake_generate(*_a, **_kw):
            return _StubGenerationOutput(text=" world!")

        def _factory():
            e = MagicMock()
            e.generate = _fake_generate
            return e

        client, _ = _build_completions_app(
            patched_config, monkeypatch, engine_factory=_factory
        )
        r = client.post(
            "/v1/completions",
            json={
                "model": "stub-model",
                "prompt": "Hello",
                "max_tokens": 4,
                "echo": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"] == " world!"


# ---------------------------------------------------------------------------
# F-153: logprobs schema + range
# ---------------------------------------------------------------------------


class TestLogprobsSchema:
    """``logprobs`` is an integer 0..5 on legacy completions, not a bool."""

    def test_logprobs_bool_rejected_with_422(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "logprobs": True},
        )
        # Pydantic before-mode validator returns 422 (not 400) for
        # the wire-form schema mismatch — a clear "this is the wrong
        # shape" signal distinct from the route-level 400s ranges
        # use.
        assert r.status_code == 422
        body = r.json()
        # Detail must mention the integer expectation so SDK clients
        # see the actual mismatch (pre-fix they got bool_parsing
        # which never mentioned the schema).
        msg = str(body)
        assert "integer" in msg.lower()
        assert "bool" in msg.lower() or "boolean" in msg.lower()

    def test_logprobs_above_five_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "logprobs": 6},
        )
        assert r.status_code == 400
        detail = (r.json().get("error") or {}).get("message") or r.json().get(
            "detail", ""
        )
        assert "0 and 5" in detail or "logprobs" in detail.lower()

    def test_logprobs_negative_rejected_with_400(self, patched_config, monkeypatch):
        client, _ = _build_completions_app(patched_config, monkeypatch)
        r = client.post(
            "/v1/completions",
            json={"model": "stub-model", "prompt": "hi", "logprobs": -1},
        )
        assert r.status_code == 400

    def test_logprobs_int_accepted_by_schema(self):
        """Direct pydantic-level test: ``logprobs: 5`` parses cleanly."""
        from vllm_mlx.api.models import CompletionRequest

        req = CompletionRequest(model="x", prompt="y", logprobs=5)
        assert req.logprobs == 5

    def test_logprobs_bool_rejected_by_schema(self):
        """Direct pydantic-level test: ``logprobs: True`` raises."""
        from pydantic import ValidationError

        from vllm_mlx.api.models import CompletionRequest

        with pytest.raises(ValidationError) as ei:
            CompletionRequest(model="x", prompt="y", logprobs=True)
        # The error must explain the integer expectation, NOT just
        # ``bool_parsing`` (the pre-fix opaque message).
        msg = str(ei.value)
        assert "integer" in msg.lower()


# ---------------------------------------------------------------------------
# Field declarations — pydantic must NOT silently drop these
# ---------------------------------------------------------------------------


class TestFieldDeclarations:
    """The pre-fix schema silently dropped ``n``, ``best_of``, ``echo``
    on parse — equivalent to the silent-compat lie F-152 closes."""

    def test_all_four_fields_declared(self):
        from vllm_mlx.api.models import CompletionRequest

        fields = CompletionRequest.model_fields
        assert "n" in fields
        assert "best_of" in fields
        assert "echo" in fields
        assert "logprobs" in fields
        # ``logprobs`` annotation must be the integer form (not bool).
        # Pydantic stores this as ``int | None`` — checking the
        # representation is robust against future ``Annotated[int, ...]``
        # tightening.
        ann = str(fields["logprobs"].annotation)
        assert "int" in ann and "bool" not in ann


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubGenerationOutput:
    """Minimal ``GenerationOutput`` stand-in for non-streaming tests."""

    def __init__(self, text: str = " world", finish_reason: str = "stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.completion_tokens = 4
        self.prompt_tokens = 1
        self.cached_tokens = 0
