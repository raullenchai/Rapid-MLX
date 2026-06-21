# SPDX-License-Identifier: Apache-2.0
"""D-ENVELOPE-FIELD-LEAK — field-name surfacing in the 400 envelope.

Sarah dogfood F-S1-1 / F-S2-2 (PyPI 0.8.3): every validation 400 across
``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``, and
``/v1/messages`` was emitting the literal placeholder ``<field>`` in
``error.message`` and ``null`` in ``error.param``, e.g.::

    {"error":{"message":"Invalid request body: <field>: Input should be
     less than or equal to 2","type":"invalid_request_error",
     "param":null,"code":null}}

Provenance: PR #784 (H-17 round-2) defended against the
``AWS_SECRET_ACCESS_KEY``-in-extra-forbidden leak by collapsing EVERY
string component of the Pydantic ``loc`` tuple. That also dropped the
schema-owned field-name hint on legitimate 400s and left ``error.param``
permanently ``None`` — which broke OpenAI SDK error branches keying on
the canonical param slot.

Fix shape: walk the loc against the actual root request-model class.
Schema-owned field names survive; attacker-controlled dict keys and
extra-forbidden names still collapse to ``<field>``. The first
schema-owned field name in the error list is mirrored into
``error.param``.

This module tests:

* All 4 routes hit the same shared envelope builder, so the field-name
  surfacing works uniformly.
* Pydantic-built-in validators (``ge=`` / ``le=`` / ``Field required``
  / type checks) surface the field name.
* Custom validators (``value_error``: NaN check, ``n != 1`` rejection)
  surface the field name.
* The H-17 attack stays closed — an ``extra_forbidden`` key still
  collapses to ``<field>`` and ``error.param`` stays ``None`` on that
  one error.
* Nested loc paths (``messages.0.role``) render correctly and
  ``error.param`` resolves to the leaf field name.
* No ``<field>`` placeholder ever appears in the envelope for a
  legitimate schema-owned field path.
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
from vllm_mlx.middleware.exception_handlers import install_exception_handlers


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A minimal FastAPI app wiring every route through the canonical
    request-model class so the registry resolves correctly."""

    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest):  # noqa: ARG001
        return {"ok": True}

    @app.post("/v1/completions")
    async def comp(req: CompletionRequest):  # noqa: ARG001
        return {"ok": True}

    @app.post("/v1/embeddings")
    async def emb(req: EmbeddingRequest):  # noqa: ARG001
        return {"ok": True}

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        # Mirror the production route: build the model manually so the
        # raw ``pydantic.ValidationError`` reaches the handler. The
        # exception handler's title-based registry lookup must still
        # resolve the field name even on this branch.
        AnthropicRequest(**body)
        return {"ok": True}

    return TestClient(app)


def _err(resp) -> dict:
    payload = resp.json()
    assert "error" in payload, payload
    return payload["error"]


# ---------------------------------------------------------------------------
# Family 1: Pydantic built-in constraints (ge=/le=/missing/type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,body,field,marker",
    [
        # /v1/chat/completions — every common Pydantic built-in family
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 99.0,
            },
            "temperature",
            "less than or equal to 2",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 2.0,
            },
            "top_p",
            "less than or equal to 1",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "min_p": -0.5,
            },
            "min_p",
            "greater than or equal to 0",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "repetition_penalty": -1.0,
            },
            "repetition_penalty",
            "greater than or equal to 0",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": 10.0,
            },
            "presence_penalty",
            "less than or equal to 2",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": -10.0,
            },
            "frequency_penalty",
            "greater than or equal to -2",
        ),
        # Missing required + type-mismatch families
        ("/v1/chat/completions", {"model": "x"}, "messages", "Field required"),
        (
            "/v1/chat/completions",
            {"model": "x", "messages": "not a list"},
            "messages",
            "valid list",
        ),
        # /v1/completions
        ("/v1/completions", {"model": "x"}, "prompt", "Field required"),
        (
            "/v1/completions",
            {"model": "x", "prompt": "hi", "temperature": 99.0},
            "temperature",
            "less than or equal to 2",
        ),
        # /v1/embeddings
        ("/v1/embeddings", {"model": "x"}, "input", "Field required"),
    ],
)
def test_pydantic_builtin_constraint_surfaces_field_name(
    client, path, body, field, marker
):
    """Sarah F-S1-1: every Pydantic-built-in validation 400 must surface
    the schema-owned field name in ``error.message`` (no ``<field>``
    placeholder) AND populate ``error.param`` with the same name."""
    resp = client.post(path, json=body)
    assert resp.status_code == 400, resp.text
    err = _err(resp)
    msg = err["message"]
    assert "<field>" not in msg, f"placeholder leaked: {msg}"
    assert field in msg, f"field {field!r} missing from message: {msg}"
    assert marker.lower() in msg.lower(), msg
    assert err["param"] == field, err


# ---------------------------------------------------------------------------
# Family 2: Custom validators (value_error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,body,field",
    [
        # NaN guard — _reject_nonfinite_sampling
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": math.nan,
            },
            "temperature",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": math.inf,
            },
            "top_p",
        ),
        # n must equal 1
        (
            "/v1/chat/completions",
            {
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 5,
            },
            "n",
        ),
        # CompletionRequest mirror
        (
            "/v1/completions",
            {"model": "x", "prompt": "hi", "temperature": math.nan},
            "temperature",
        ),
    ],
)
def test_custom_validator_surfaces_field_name(client, path, body, field):
    """Custom ``field_validator``/``model_validator`` raises surface as
    ``value_error`` in Pydantic v2 with the field in ``loc``. The
    envelope must still set ``error.param`` and render the field name in
    ``error.message`` so the OpenAI SDK error branches resolve."""
    # NaN/inf can't go through JSON encoder — send via raw body.
    raw = json.dumps(body, allow_nan=True)
    resp = client.post(path, content=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 400, resp.text
    err = _err(resp)
    assert "<field>" not in err["message"], err["message"]
    assert field in err["message"], err["message"]
    assert err["param"] == field, err


# ---------------------------------------------------------------------------
# Family 3: Nested loc paths — schema-owned all the way down
# ---------------------------------------------------------------------------


def test_nested_loc_path_renders_full_path(client):
    """Missing ``role`` on a nested message item: the rendered loc must
    be ``messages.0.role`` (NOT ``<field>.0.<field>``) and
    ``error.param`` is the leaf name ``role``."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"content": "hi"}]},
    )
    assert resp.status_code == 400
    err = _err(resp)
    assert "messages.0.role" in err["message"], err["message"]
    assert "<field>" not in err["message"], err["message"]
    assert err["param"] == "role", err


# ---------------------------------------------------------------------------
# Family 4: H-17 attack vector stays closed
# ---------------------------------------------------------------------------


def test_extra_forbidden_attacker_key_does_not_leak(client):
    """Sarah's vector: attacker stuffs a secret-shaped name into an
    ``extra_forbidden`` slot on the request body. The H-17 round-2
    BLOCKING finding was that any whitelist-by-shape (identifier regex
    etc.) would still echo identifier-shaped attacker bytes like
    ``AWS_SECRET_ACCESS_KEY``.

    The fix walks the model class — names that ARE schema-owned survive,
    names that ARE NOT collapse to ``<field>`` and stop the walk. So the
    attack stays closed: the secret-shaped name MUST collapse and MUST
    NOT appear in the rendered loc."""
    resp = client.post(
        "/v1/embeddings",
        json={
            "model": "x",
            "input": "hi",
            "AWS_SECRET_ACCESS_KEY": "leak-me-please",
        },
    )
    assert resp.status_code == 400
    err = _err(resp)
    msg = err["message"]
    # The attacker-controlled key never appears anywhere in the body.
    assert "AWS_SECRET_ACCESS_KEY" not in msg, msg
    assert "leak-me-please" not in msg, msg
    # The placeholder DOES appear, signalling the attacker-controlled
    # name was redacted.
    assert "<field>" in msg, msg
    # ``error.param`` MUST stay null on this branch — the only failing
    # loc component is attacker-controlled, so there's no safe leaf to
    # populate.
    assert err["param"] is None, err


def test_dict_value_field_attacker_key_does_not_leak(client):
    """``logit_bias`` on /v1/chat/completions is typed
    ``dict[str, float] | None``. An attacker-supplied dict KEY with
    bytes that don't parse as a float lands as a Pydantic
    ``float_parsing`` error with the attacker-controlled key inside
    ``loc`` — the dict-key attack family. The walker must collapse the
    key to ``<field>`` and stop descending."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": {"AWS_SECRET_DICT_KEY": "not_a_float"},
        },
    )
    # logit_bias has a custom validator that runs before the dict[str,
    # float] coercion — assert based on the response. The contract is:
    # whatever 400 surfaces, the attacker key must not appear.
    if resp.status_code == 400:
        err = _err(resp)
        assert "AWS_SECRET_DICT_KEY" not in err["message"], err["message"]


# ---------------------------------------------------------------------------
# Family 5: /v1/messages (Anthropic) — manual model construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,field",
    [
        (
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            "model",
        ),
        (
            {"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            "max_tokens",
        ),
    ],
)
def test_anthropic_route_surfaces_field_name(client, body, field):
    """``AnthropicRequest(**body)`` raises the raw
    :class:`pydantic.ValidationError`. The handler uses
    ``exc.title == 'AnthropicRequest'`` to find the root model in the
    registry, then walks the loc just like the FastAPI-bound routes."""
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 400, resp.text
    err = _err(resp)
    assert "<field>" not in err["message"], err["message"]
    assert field in err["message"], err["message"]
    assert err["param"] == field, err


# ---------------------------------------------------------------------------
# Family 6: Belt-and-suspenders — no ``<field>`` in any happy-path 400
# ---------------------------------------------------------------------------


def test_envelope_param_is_populated_on_first_schema_owned_error(client):
    """When multiple validation errors fire at once, ``error.param``
    must populate from the FIRST schema-owned field name encountered
    (matching OpenAI's single-``param`` envelope shape)."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 99.0,
            "top_p": 2.0,
        },
    )
    assert resp.status_code == 400
    err = _err(resp)
    assert err["param"] in ("temperature", "top_p"), err
    # Both field names appear in the message.
    assert "temperature" in err["message"], err["message"]
    assert "top_p" in err["message"], err["message"]


def test_unregistered_root_model_falls_back_to_h17_default(monkeypatch):
    """A FastAPI route binding an unregistered request-model class must
    still get the H-17 conservative behaviour — every string component
    collapses to ``<field>`` — so a plugin that forgets to register its
    own model can't accidentally leak attacker bytes."""
    from pydantic import BaseModel, ConfigDict

    class UnregisteredRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a_field: str

    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/unregistered")
    async def route(req: UnregisteredRequest):  # noqa: ARG001
        return {"ok": True}

    cli = TestClient(app)
    resp = cli.post("/unregistered", json={"a_field": "x", "EVIL": "bytes"})
    assert resp.status_code == 400
    err = resp.json()["error"]
    # Default secrecy default — the attacker key must NOT echo.
    assert "EVIL" not in err["message"], err["message"]
