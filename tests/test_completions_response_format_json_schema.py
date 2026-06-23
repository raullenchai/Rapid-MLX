# SPDX-License-Identifier: Apache-2.0
"""r10-J — codex r10-G/H round-2 HIGH #1 (json_schema on /v1/completions).

The r10-H4 ``response_format`` wiring on ``/v1/completions`` accepted
``{"type": "json_schema", "json_schema": {...}}`` and post-processed the
generated text through ``extract_json_from_response`` — which strips
markdown fences but does NOT enforce the schema. A caller setting
``json_schema.strict=true`` therefore got HTTP 200 back with JSON that
could violate the schema (silent corruption of the OpenAI strict
contract).

The chat lane (``routes/chat.py``) already owns the strict path via
``is_strict_json_schema`` / ``extract_json_schema_for_guided`` plus
guided-generation. The legacy completions lane never had FSM
constraints; wiring guided generation there is a separate refactor.

The conservative spec-correct fix is to reject ``json_schema`` on
``/v1/completions`` with a 400 envelope that points callers at chat.
``json_object`` keeps working — fence-strip alone is sufficient for
loose JSON, no schema to validate — so this is a surgical close of the
strict gap.

The three tests below pin the contract:

1. ``stream=false`` + ``response_format=json_schema`` (any strictness) → 400
2. ``stream=true``  + ``response_format=json_schema`` (any strictness) → 400
3. ``stream=false`` + ``response_format=json_object``                  → 200
   (regression test — ``json_object`` must keep working post-r10-J,
   else we over-corrected and broke loose-JSON callers).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.completions import router as completions_router

_FENCED_TEXT = 'Just the JSON.\n```json\n{"answer": 42}\n```'
_CLEAN_TEXT = '{"answer": 42}'

_JSON_SCHEMA_STRICT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_envelope",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

_JSON_SCHEMA_LOOSE = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_envelope",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
        },
    },
}


class _FencedCompletionEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    async def generate(self, prompt, **kwargs):
        return GenerationOutput(
            text=_FENCED_TEXT,
            prompt_tokens=8,
            completion_tokens=12,
            finished=True,
            finish_reason="stop",
        )

    async def stream_generate(self, prompt, **kwargs):
        yield GenerationOutput(
            text=_FENCED_TEXT,
            new_text=_FENCED_TEXT,
            prompt_tokens=8,
            completion_tokens=12,
            finished=True,
            finish_reason="stop",
        )


def _make_client() -> TestClient:
    cfg = reset_config()
    cfg.engine = _FencedCompletionEngine()
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(completions_router)
    return TestClient(app)


def _assert_400_envelope(resp) -> None:
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "unsupported_response_format"
    assert err["param"] == "response_format"
    assert "json_schema" in err["message"]
    assert "/v1/chat/completions" in err["message"]


def test_sync_json_schema_strict_rejected_with_400():
    """Strict path: pre-fix the route returned 200 with the
    fence-stripped JSON but never validated it against the schema —
    silent ``strict=true`` violation. Post-fix: 400 invalid_request."""
    client = _make_client()
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "response_format": _JSON_SCHEMA_STRICT,
        },
    )
    _assert_400_envelope(resp)


def test_sync_json_schema_loose_also_rejected_with_400():
    """Defense-in-depth: even ``strict`` omitted, ``json_schema``
    semantics imply schema enforcement; this route doesn't do that,
    so reject the whole ``type=json_schema`` shape — don't try to
    guess intent. Callers needing loose JSON use ``json_object``."""
    client = _make_client()
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "response_format": _JSON_SCHEMA_LOOSE,
        },
    )
    _assert_400_envelope(resp)


def test_stream_json_schema_rejected_with_400():
    """Streaming path: the reject runs BEFORE the SSE
    StreamingResponse commits, so the client sees a 400 envelope
    instead of an EventSource that emits unconstrained text."""
    client = _make_client()
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "stream": True,
            "response_format": _JSON_SCHEMA_STRICT,
        },
    )
    _assert_400_envelope(resp)


def test_sync_json_object_still_works_after_r10_j_rejection():
    """Regression guard for over-correction: ``json_object`` (loose,
    no schema) must keep working post-r10-J. The wire ``text`` should
    be fence-stripped per r10-H4 chat-lane parity."""
    client = _make_client()
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["text"] == _CLEAN_TEXT


def test_json_schema_plus_logprobs_400_still_fires_for_combination_rule():
    """Cross-check: ``response_format + logprobs`` was already
    rejected by the r10-G/H r1 fix. Make sure r10-J's earlier-in-the-
    pipeline reject of ``json_schema`` doesn't accidentally mask
    that combination error — the matrix has two orthogonal 400
    rules, and either firing is acceptable as long as the request
    is rejected."""
    client = _make_client()
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 16,
            "response_format": _JSON_SCHEMA_STRICT,
            "logprobs": 2,
        },
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "response_format"
    # Either rule fired — both are correct 400s for this matrix cell.
    assert err["code"] in {
        "unsupported_response_format",
        "unsupported_combination",
    }
