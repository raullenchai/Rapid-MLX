# SPDX-License-Identifier: Apache-2.0
"""VAL-2361 — /v1/models/load validation errors name the real field.

Issue #2361: invalid residency ``performance`` settings returned the
literal placeholder ``<field>`` in ``error.message`` and ``null`` in
``error.param``, e.g.::

    {"error":{"message":"Invalid request body: <field>: Value error, KV
     dtype and TurboQuant are mutually exclusive","param":null}}

Root cause: ``ModelLoadRequest`` was not registered with the safe
error-location contract (D-ENVELOPE-FIELD-LEAK), so every string ``loc``
component collapsed to ``<field>`` and no schema-owned field reached
``error.param``.

Fix: :mod:`vllm_mlx.routes.residency` registers ``ModelLoadRequest`` and
binds ``/v1/models/load`` at import time (plugin-style, keeping the
middleware module import-light). The envelope now surfaces the real
schema-owned field path — matching how the chat endpoints name fields —
for all three cases in the issue:

1. mutually exclusive ``kv_cache_dtype`` + ``kv_cache_turboquant`` ->
   ``performance`` (the owning setting group);
2. ``estimated_size_gb=0`` (``gt=0`` violation) -> ``estimated_size_gb``;
3. invalid ``replace_group`` (pattern violation) -> ``replace_group``.

The H-17 default-deny contract is preserved: the walker only echoes names
that live on ``ModelLoadRequest``'s ``model_fields``, so attacker-supplied
keys still collapse to ``<field>``.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Importing the residency route module runs its module-scope registration
# of ModelLoadRequest + the /v1/models/load path binding (same as production).
from vllm_mlx.middleware.exception_handlers import install_exception_handlers

# Defines the route's request model AND triggers the registry registration.
from vllm_mlx.routes import residency  # noqa: F401
from vllm_mlx.routes.residency import ModelLoadRequest


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Minimal app wiring /v1/models/load through ModelLoadRequest, exactly
    as the production residency router does, so the FastAPI-bound validation
    error reaches the shared envelope handler."""

    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/v1/models/load")
    async def load(req: ModelLoadRequest):  # noqa: ARG001
        return {"ok": True}

    return TestClient(app)


def _err(resp) -> dict:
    payload = resp.json()
    assert "error" in payload, payload
    return payload["error"]


@pytest.mark.parametrize(
    "body,field,marker",
    [
        # 1. Mutually exclusive cache modes — value_error on the owning
        #    `performance` setting group.
        (
            {
                "model": "qwen3.5-9b-4bit",
                "model_path": "<cached-snapshot>",
                "reload_if_changed": True,
                "performance": {
                    "kv_cache_dtype": "int8",
                    "kv_cache_turboquant": "v4",
                },
            },
            "performance",
            "mutually exclusive",
        ),
        # 2. estimated_size_gb=0 (gt=0 violation).
        (
            {
                "model": "qwen3.5-9b-4bit",
                "model_path": "<cached-snapshot>",
                "estimated_size_gb": 0,
            },
            "estimated_size_gb",
            "greater than 0",
        ),
        # 3. invalid replace_group (pattern violation).
        (
            {
                "model": "qwen3.5-9b-4bit",
                "model_path": "<cached-snapshot>",
                "replace_group": "not-assistant",
            },
            "replace_group",
            "pattern",
        ),
    ],
)
def test_load_validation_names_the_field(client, body, field, marker) -> None:
    """A schema-owned residency-load setting failure reports the real field
    path in the message and mirrors it into error.param."""

    resp = client.post("/v1/models/load", json=body)
    assert resp.status_code == 400
    err = _err(resp)
    # The real schema-owned field name (not `<field>`) is surfaced.
    assert field in err["message"]
    assert marker.lower() in err["message"].lower()
    assert "<field>" not in err["message"]
    # error.param carries the schema-owned field so SDK error branches key.
    assert err["param"] == field
    assert err["type"] == "invalid_request_error"


def test_load_validation_message_is_issue_shaped(client) -> None:
    """Regression-lock the exact envelope shape from issue #2361: the message
    now identifies `performance` instead of leaking the `<field>` placeholder."""

    resp = client.post(
        "/v1/models/load",
        json={
            "model": "qwen3.5-9b-4bit",
            "model_path": "<cached-snapshot>",
            "reload_if_changed": True,
            "performance": {
                "kv_cache_dtype": "int8",
                "kv_cache_turboquant": "v4",
            },
        },
    )
    body = resp.json()
    assert resp.status_code == 400
    assert body["error"]["message"] == (
        "Invalid request body: performance: Value error, "
        "KV dtype and TurboQuant are mutually exclusive"
    )
    assert body["error"]["param"] == "performance"
    assert "<field>" not in json.dumps(body)
