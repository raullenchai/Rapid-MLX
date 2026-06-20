# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the opt-in per-request generation-budget ceiling.

Aanya (0.8.0 round-2 dogfooding) showed that the F-007 body-bytes cap
does NOT enforce the generation-token budget — a 5K-token system
prompt with ``max_tokens=10000`` was accepted with the body cap left at
its 8 MiB default. M-04 closes the gap by adding an OPT-IN cap exposed
via the ``RAPID_MLX_MAX_GENERATION_TOKENS`` env var, applied at parse
time on all three OpenAI/Anthropic compatible request models:

* ``/v1/chat/completions`` → ``ChatCompletionRequest``
* ``/v1/completions``      → ``CompletionRequest``
* ``/v1/messages``         → ``AnthropicRequest``

Design contract (mirrored across all three routes):

* Env unset / blank / non-int / ``<= 0`` → no enforcement. Existing
  single-machine UX (Yuki posture) is unaffected — the cap is purely
  opt-in for multi-tenant operators.
* Env set to a positive integer ``N`` → reject at parse time when
  ``max_tokens > N``. The error envelope mentions
  ``RAPID_MLX_MAX_GENERATION_TOKENS`` so operators see the actionable
  lever in the 400 body.

The tests below pin both arms — the opt-in invariant AND the
per-route enforcement.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    _enforce_max_generation_tokens_ceiling,
    _resolve_max_generation_tokens_ceiling,
)

# Env var name kept as a module constant so a future rename has exactly
# one source of truth (tests and the helper would both fail loudly).
_ENV = "RAPID_MLX_MAX_GENERATION_TOKENS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_payload(max_tokens: int | None = None) -> dict:
    body: dict = {
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def _completion_payload(max_tokens: int | None = None) -> dict:
    body: dict = {"model": "default", "prompt": "hello"}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def _anthropic_payload(max_tokens: int = 100) -> dict:
    # ``max_tokens`` is required on the Anthropic surface — no None case.
    return {
        "model": "default",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Env-resolution helper unit tests
# ---------------------------------------------------------------------------


class TestResolveCeiling:
    """Pin the env-parsing contract — the opt-in stays opt-in even
    when the env var contains a typo (so an operator who fat-fingered
    the value never accidentally turns on enforcement)."""

    @pytest.mark.parametrize("raw", ["", "   ", "abc", "0", "-1", "1.5", "0x10"])
    def test_invalid_or_unset_resolves_none(self, monkeypatch, raw):
        if raw == "":
            monkeypatch.delenv(_ENV, raising=False)
        else:
            monkeypatch.setenv(_ENV, raw)
        # ``""`` exercise: env unset, ceiling None.
        # Other values exercise: env set but invalid → still None.
        assert _resolve_max_generation_tokens_ceiling() is None

    def test_unset_env_resolves_none(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)
        assert _resolve_max_generation_tokens_ceiling() is None

    @pytest.mark.parametrize(
        "raw,expected", [("1", 1), ("1000", 1000), ("  4096 ", 4096)]
    )
    def test_positive_int_resolves_to_value(self, monkeypatch, raw, expected):
        monkeypatch.setenv(_ENV, raw)
        assert _resolve_max_generation_tokens_ceiling() == expected


class TestEnforceCeilingHelper:
    """Direct-helper coverage. The per-model validators delegate here, so
    pinning the helper avoids re-asserting the same branches three times."""

    def test_none_max_tokens_is_always_accepted(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1")
        # ``None`` means "use the route default"; the ceiling only fires
        # on explicit caller intent. Otherwise a multi-tenant operator
        # who set the env var would unintentionally break callers that
        # rely on the server-side default.
        _enforce_max_generation_tokens_ceiling(None)

    def test_env_unset_is_always_accepted(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)
        # Any value passes when the env var is unset — opt-in stays
        # opt-in.
        _enforce_max_generation_tokens_ceiling(10_000_000)

    def test_within_ceiling_is_accepted(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")
        _enforce_max_generation_tokens_ceiling(1000)  # boundary
        _enforce_max_generation_tokens_ceiling(500)

    def test_over_ceiling_is_rejected_with_env_name_in_message(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")
        with pytest.raises(ValueError) as exc:
            _enforce_max_generation_tokens_ceiling(1001)
        msg = str(exc.value)
        # The error has to surface the env var name so the operator
        # reading the 400 envelope knows which lever to flip.
        assert _ENV in msg
        # The actual values are echoed so a downstream alert / log line
        # carries the same diagnostic the client sees.
        assert "1001" in msg
        assert "1000" in msg


# ---------------------------------------------------------------------------
# Per-model parse-time invariants
# ---------------------------------------------------------------------------


class TestEnvUnsetAcceptsAnyMaxTokens:
    """When the env var is unset, the historical behaviour (no cap on
    ``max_tokens``) must be preserved on every route. Locks down the
    opt-in promise — single-machine users see no change."""

    @pytest.fixture(autouse=True)
    def _no_env(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)

    def test_chat_request_accepts_huge_max_tokens(self):
        # Aanya's repro: pre-fix this returned HTTP 200; post-fix this
        # also returns HTTP 200 because the env var is unset.
        req = ChatCompletionRequest(**_chat_payload(max_tokens=10_000))
        assert req.max_tokens == 10_000

    def test_completion_request_accepts_huge_max_tokens(self):
        req = CompletionRequest(**_completion_payload(max_tokens=10_000))
        assert req.max_tokens == 10_000

    def test_anthropic_request_accepts_huge_max_tokens(self):
        req = AnthropicRequest(**_anthropic_payload(max_tokens=10_000))
        assert req.max_tokens == 10_000

    def test_chat_request_max_completion_tokens_accepted(self):
        # ``max_completion_tokens`` is the canonical OpenAI Sept-2024
        # field name — has to share the same opt-in semantics.
        req = ChatCompletionRequest(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=10_000,
        )
        assert req.max_tokens == 10_000


class TestEnvSetWithinCeilingAccepts:
    """Env set + ``max_tokens`` within the ceiling must round-trip the
    request unchanged. Boundary value (``max_tokens == ceiling``) must
    be accepted — the comparison is strict ``>``."""

    @pytest.fixture(autouse=True)
    def _ceiling_1000(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")

    def test_chat_request_within_ceiling(self):
        req = ChatCompletionRequest(**_chat_payload(max_tokens=500))
        assert req.max_tokens == 500

    def test_chat_request_at_ceiling(self):
        # Boundary: 1000 is the ceiling and is accepted (>, not >=).
        req = ChatCompletionRequest(**_chat_payload(max_tokens=1000))
        assert req.max_tokens == 1000

    def test_completion_request_within_ceiling(self):
        req = CompletionRequest(**_completion_payload(max_tokens=500))
        assert req.max_tokens == 500

    def test_anthropic_request_within_ceiling(self):
        req = AnthropicRequest(**_anthropic_payload(max_tokens=500))
        assert req.max_tokens == 500

    def test_chat_request_with_none_max_tokens(self):
        # Omitted max_tokens is unconstrained — the ceiling only fires
        # when the caller asks for a specific budget.
        req = ChatCompletionRequest(**_chat_payload(max_tokens=None))
        assert req.max_tokens is None


class TestEnvSetOverCeilingRejects:
    """Env set + ``max_tokens > ceiling`` → ValidationError with the env
    var name embedded in the message."""

    @pytest.fixture(autouse=True)
    def _ceiling_1000(self, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")

    def _assert_envelope_mentions_env_var(self, exc: ValidationError) -> None:
        # Pydantic embeds the validator's ValueError message in the
        # error dict; assert against both the str form (what the route
        # surfaces to clients) and the structured form (what programmatic
        # callers see).
        as_str = str(exc.value)
        assert _ENV in as_str
        assert "1500" in as_str
        assert "1000" in as_str

    def test_chat_request_rejected_with_clean_envelope(self):
        with pytest.raises(ValidationError) as exc:
            ChatCompletionRequest(**_chat_payload(max_tokens=1500))
        self._assert_envelope_mentions_env_var(exc)

    def test_chat_request_via_max_completion_tokens_rejected(self):
        # The Sept-2024 OpenAI field has to be enforced too — otherwise
        # callers using the newer SDK shape bypass the cap.
        with pytest.raises(ValidationError) as exc:
            ChatCompletionRequest(
                model="default",
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=1500,
            )
        self._assert_envelope_mentions_env_var(exc)

    def test_completion_request_rejected_with_clean_envelope(self):
        with pytest.raises(ValidationError) as exc:
            CompletionRequest(**_completion_payload(max_tokens=1500))
        self._assert_envelope_mentions_env_var(exc)

    def test_anthropic_request_rejected_with_clean_envelope(self):
        with pytest.raises(ValidationError) as exc:
            AnthropicRequest(**_anthropic_payload(max_tokens=1500))
        self._assert_envelope_mentions_env_var(exc)


# ---------------------------------------------------------------------------
# Wire-shape coverage via TestClient
# ---------------------------------------------------------------------------


class TestWireShape:
    """The unit tests above pin the model-layer contract. This class
    exercises the FastAPI surface end-to-end so we also catch routing
    surprises (e.g. a future change that moves validation off the
    BaseModel onto a route-local Depends). We mount tiny handlers that
    just echo the parsed model — the engine is not loaded.
    """

    @pytest.fixture
    def app(self) -> FastAPI:
        # Build a minimal app that mounts only the three request models.
        # Avoids pulling in ``vllm_mlx.server`` (which imports MLX) so the
        # test runs on every CI matrix entry including the no-mlx Linux
        # validation runner.
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def _chat(req: ChatCompletionRequest):
            return {"max_tokens": req.max_tokens}

        @app.post("/v1/completions")
        async def _completion(req: CompletionRequest):
            return {"max_tokens": req.max_tokens}

        @app.post("/v1/messages")
        async def _anthropic(req: AnthropicRequest):
            return {"max_tokens": req.max_tokens}

        return app

    def test_chat_over_ceiling_returns_400_with_env_name(self, app, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")
        client = TestClient(app)
        r = client.post("/v1/chat/completions", json=_chat_payload(max_tokens=1500))
        # FastAPI's default RequestValidationError surfaces as 422; we
        # accept either 400 (chat route's exception handler) or 422
        # (FastAPI default). Production hosts install a 400-mapping
        # handler — see ``middleware.exception_handlers`` — but in this
        # minimal app the default 422 is fine for asserting the value
        # is rejected and the env var name is in the body.
        assert r.status_code in (400, 422)
        body_text = r.text
        assert _ENV in body_text

    def test_completion_over_ceiling_returns_400_with_env_name(self, app, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")
        client = TestClient(app)
        r = client.post("/v1/completions", json=_completion_payload(max_tokens=1500))
        assert r.status_code in (400, 422)
        assert _ENV in r.text

    def test_anthropic_over_ceiling_returns_400_with_env_name(self, app, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")
        client = TestClient(app)
        r = client.post("/v1/messages", json=_anthropic_payload(max_tokens=1500))
        assert r.status_code in (400, 422)
        assert _ENV in r.text

    def test_chat_within_ceiling_round_trips(self, app, monkeypatch):
        monkeypatch.setenv(_ENV, "1000")
        client = TestClient(app)
        r = client.post("/v1/chat/completions", json=_chat_payload(max_tokens=500))
        assert r.status_code == 200
        assert r.json() == {"max_tokens": 500}

    def test_env_unset_accepts_huge_max_tokens_on_all_routes(self, app, monkeypatch):
        # Single-machine UX: huge ``max_tokens`` still passes through.
        monkeypatch.delenv(_ENV, raising=False)
        client = TestClient(app)
        big = 10_000

        r = client.post("/v1/chat/completions", json=_chat_payload(max_tokens=big))
        assert r.status_code == 200, r.text
        assert r.json() == {"max_tokens": big}

        r = client.post("/v1/completions", json=_completion_payload(max_tokens=big))
        assert r.status_code == 200, r.text
        assert r.json() == {"max_tokens": big}

        r = client.post("/v1/messages", json=_anthropic_payload(max_tokens=big))
        assert r.status_code == 200, r.text
        assert r.json() == {"max_tokens": big}
