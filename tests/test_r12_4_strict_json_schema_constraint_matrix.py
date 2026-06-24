# SPDX-License-Identifier: Apache-2.0
"""R12-4 — strict ``response_format.json_schema`` 16-constraint matrix.

Pins the systemic R12-4 fix:

* Pre-R12-4, ``strict=true`` without the ``[guided]`` extra installed
  returned HTTP 400 ``guided_extra_required``. Astrid r3 surfaced that
  this broke ``pydantic-ai`` end-to-end — every SDK retry hit the same
  deterministic empty-args synthetic ``final_result`` tool_call and
  the client exhausted ``max_retries`` against a server-side blocker
  the SDK could not circumvent.

* Post-R12-4, ``strict=true`` runs the engine UNCONSTRAINED, validates
  the output against the schema after generation, attempts a single
  repair retry with a system-prompt-injected hint naming the failing
  path, and returns 422 with a structured envelope ONLY if the repair
  also fails. The disable flag ``RAPID_MLX_STRICT_JSON_SCHEMA=off``
  short-circuits the gate for operators who need the legacy behavior.

This file specifically pins the **16 constraint families** the design
review surfaced as silently-passing pre-R12-4:

    1.  ``additionalProperties: false`` — extra property
    2.  ``required``                    — missing property
    3.  ``type``                        — wrong type at top level
    4.  ``enum``                        — value outside enum set
    5.  ``pattern``                     — string that doesn't match regex
    6.  ``minLength``                   — string too short
    7.  ``maxLength``                   — string too long
    8.  ``minimum``                     — integer below floor
    9.  ``maximum``                     — integer above ceiling
    10. ``multipleOf``                  — not divisible
    11. ``minItems``                    — array too short
    12. ``maxItems``                    — array too long
    13. ``uniqueItems``                 — duplicate item
    14. ``format``                      — string fails format check (email)
    15. ``type`` coercion — number      — string where number expected
    16. ``type`` coercion — boolean     — string where boolean expected

Each row trips the post-generate validator AND surfaces 422 with a
structured envelope. Operators reading dashboards see one
``strict_violations_total`` tick per row.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import response_format_metrics
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.chat import router as chat_router


class _StubEngine:
    """Deterministic engine that returns a fixed body for each chat call.

    The repair retry would normally fire after the first invalid
    response, so we configure the engine to return the SAME invalid
    body on every call. This guarantees the route always reaches the
    422 surface (no flaky pass-through if the repair coincidentally
    produced valid output).
    """

    preserve_native_tool_format = False
    is_mllm = False
    tokenizer = None
    supports_guided_generation = False

    def __init__(self, *, body: str):
        self._body = body
        self.chat_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._body,
            new_text=self._body,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):  # pragma: no cover
        yield GenerationOutput(
            text=self._body,
            new_text=self._body,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    response_format_metrics.reset_for_tests()
    yield
    response_format_metrics.reset_for_tests()


def _client(body: str) -> tuple[TestClient, _StubEngine]:
    engine = _StubEngine(body=body)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app), engine


def _payload(*, schema: dict) -> dict:
    """Strict json_schema payload for the chat-completions route."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "produce something"}],
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "CMatrix",
                "schema": schema,
                "strict": True,
            },
        },
    }


# Each row: (constraint_family_label, schema, violating_body,
#            expected_validator_substring).
#
# The bodies are deliberately chosen so json.loads succeeds but the
# specific JSON-Schema constraint fails — that way every row exercises
# the validator's ``iter_errors`` path (not the json-parse arm).
#
# ``expected_validator_substring`` is a fragment that MUST appear in
# the returned ``details.expected`` field. This pins each row to the
# CORRECT validator (codex r1 #2) — e.g. the ``format_email`` row
# must surface ``"format"`` in ``expected``, NOT ``"type"``, so a
# regression that disables the FormatChecker is caught immediately.
_CONSTRAINT_MATRIX: list[tuple[str, dict, str, str]] = [
    (
        "additionalProperties_false",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        json.dumps({"name": "ok", "extra": "should not be here"}),
        "additionalProperties",
    ),
    (
        "required",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
        json.dumps({"name": "missing age"}),
        "required",
    ),
    (
        "type_top_level",
        # Expect an object, model returns an array.
        {"type": "object", "properties": {"x": {"type": "integer"}}},
        json.dumps([1, 2, 3]),
        "type",
    ),
    (
        "enum",
        {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "green"]}},
            "required": ["color"],
        },
        json.dumps({"color": "purple"}),
        "enum",
    ),
    (
        "pattern",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": r"^[A-Z]{3}$"},
            },
            "required": ["code"],
        },
        json.dumps({"code": "abcde"}),
        "pattern",
    ),
    (
        "minLength",
        {
            "type": "object",
            "properties": {"slug": {"type": "string", "minLength": 5}},
            "required": ["slug"],
        },
        json.dumps({"slug": "hi"}),
        "minLength",
    ),
    (
        "maxLength",
        {
            "type": "object",
            "properties": {"slug": {"type": "string", "maxLength": 3}},
            "required": ["slug"],
        },
        json.dumps({"slug": "way-too-long"}),
        "maxLength",
    ),
    (
        "minimum",
        {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 18}},
            "required": ["age"],
        },
        json.dumps({"age": 5}),
        "minimum",
    ),
    (
        "maximum",
        {
            "type": "object",
            "properties": {"score": {"type": "integer", "maximum": 100}},
            "required": ["score"],
        },
        json.dumps({"score": 9001}),
        "maximum",
    ),
    (
        "multipleOf",
        {
            "type": "object",
            "properties": {"step": {"type": "integer", "multipleOf": 5}},
            "required": ["step"],
        },
        json.dumps({"step": 7}),
        "multipleOf",
    ),
    (
        "minItems",
        {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            },
            "required": ["tags"],
        },
        json.dumps({"tags": ["only-one"]}),
        "minItems",
    ),
    (
        "maxItems",
        {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 1},
            },
            "required": ["tags"],
        },
        json.dumps({"tags": ["a", "b", "c"]}),
        "maxItems",
    ),
    (
        "uniqueItems",
        {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "uniqueItems": True,
                },
            },
            "required": ["ids"],
        },
        json.dumps({"ids": [1, 1, 2]}),
        "uniqueItems",
    ),
    (
        "format_email",
        {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
            },
            "required": ["email"],
        },
        # Codex r1 #2: the value MUST be a syntactically valid string
        # so it passes ``type:"string"`` and ONLY the ``format`` check
        # can be the failing keyword. Pre-fix the row used ``12345``
        # which tripped ``type`` instead — false-passing the
        # ``format`` enforcement contract this row is supposed to pin.
        json.dumps({"email": "not-an-email"}),
        "format",
    ),
    (
        "type_coercion_number",
        # Model returns a string where the schema demands a number;
        # JSON-Schema does NOT coerce, so this is a hard violation.
        {
            "type": "object",
            "properties": {"price": {"type": "number"}},
            "required": ["price"],
        },
        json.dumps({"price": "12.50"}),
        "type",
    ),
    (
        "type_coercion_boolean",
        # Model returns a string where the schema demands a boolean.
        {
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
            "required": ["active"],
        },
        json.dumps({"active": "true"}),
        "type",
    ),
]


@pytest.mark.parametrize(
    "label,schema,violating_body,expected_validator_substring",
    _CONSTRAINT_MATRIX,
    ids=[row[0] for row in _CONSTRAINT_MATRIX],
)
def test_strict_constraint_family_trips_422(
    label, schema, violating_body, expected_validator_substring
):
    """Every constraint family in the 16-row matrix must trip a 422
    response with the structured ``json_schema_violation`` envelope,
    AND the failing validator must match the constraint the row is
    claiming to exercise.

    The engine returns the SAME violating body on every chat call,
    so the route MUST:
        1. validate-and-fail on the initial output
        2. attempt the repair retry
        3. validate-and-fail again
        4. surface 422 with the structured envelope

    The ``expected_validator_substring`` assertion (codex r1 #2)
    pins each row to the CORRECT validator — a regression that
    disables the FormatChecker, swaps draft versions, or otherwise
    causes a different keyword to be the first failing one is
    caught immediately.
    """
    client, engine = _client(violating_body)

    resp = client.post("/v1/chat/completions", json=_payload(schema=schema))
    assert resp.status_code == 422, f"{label}: {resp.text}"
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation"
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["param"] == "response_format.json_schema"
    details = body["error"]["details"]
    assert details["attempts"] == 2  # initial + single repair retry
    assert "reason" in details
    # Codex r1 #2: pin the failing validator. ``expected`` is shaped
    # ``"<validator_name>: <repr(validator_value)>"`` (e.g.
    # ``"minimum: 18"``, ``"format: 'email'"``), so the row's
    # constraint name MUST appear as the validator portion before
    # the colon.
    assert details.get("reason") == "schema_violation", (
        f"{label}: expected schema_violation, got {details}"
    )
    assert details["expected"].startswith(expected_validator_substring + ":"), (
        f"{label}: expected validator `{expected_validator_substring}` "
        f"as the failing keyword but got `{details['expected']}`. "
        "If you are seeing `type` for a non-type row, the FormatChecker "
        "may have regressed (codex r1 #1)."
    )
    # Both the initial AND repair retry must have hit the engine —
    # otherwise the route is short-circuiting somewhere it shouldn't.
    assert len(engine.chat_calls) == 2, f"{label}: chat_calls = {engine.chat_calls}"
    # Counters must reflect the request + the final violation.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1
    assert snap["strict_repairs_attempted_total"] == 1
    assert snap["strict_repairs_succeeded_total"] == 0


def test_strict_constraint_matrix_disable_flag_returns_200_for_all(monkeypatch):
    """With ``RAPID_MLX_STRICT_JSON_SCHEMA=off`` every row of the
    matrix returns 200 (legacy silent-pass-through). Pins the
    escape hatch operators can flip if R12-4 enforcement is too
    aggressive for their workload."""
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA", "off")
    for label, schema, violating_body, _ in _CONSTRAINT_MATRIX:
        client, engine = _client(violating_body)
        resp = client.post("/v1/chat/completions", json=_payload(schema=schema))
        # Legacy behavior: 200 even with schema-violating body.
        assert resp.status_code == 200, (
            f"{label} should fall through under disable flag: {resp.text}"
        )
        # Only the initial chat call should fire; no repair retry path.
        assert len(engine.chat_calls) == 1, (
            f"{label}: repair retry must NOT fire under disable flag"
        )
