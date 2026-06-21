# SPDX-License-Identifier: Apache-2.0
"""H-06 — ``response_format.json_schema.strict=true`` enforcement contract.

Pins the systemic fix for v0.8 H-06:

* Pre-fix, ``strict=true`` was suggestion-only. The route injected the
  schema into the system prompt and let the engine emit unconstrained
  tokens. Any client built around ``chat.completions.parsed`` (the
  OpenAI SDK structured-output helper) silently received schema
  violations.

* Post-fix, ``strict=true`` activates outlines-backed constrained
  decoding via ``engine.generate_with_schema`` (the same path
  ``stream=true`` already used for ``json_schema`` requests). When the
  ``[guided]`` extra is missing, the route 400s with a canonical OpenAI-
  shape envelope that names the install hint instead of degrading to
  unconstrained output.

* The route also bumps two Prometheus counters surfaced via
  ``/metrics``: ``rapid_mlx_response_format_strict_total`` for every
  admitted strict request, and ``rapid_mlx_response_format_strict_violations_total``
  for any post-decode ``jsonschema.validate`` rejection (smoke alarm —
  outlines should make this unreachable).

The contract surfaces:

  * ``strict=true`` + guided available → constrained decoding
  * ``strict=true`` + guided absent     → 400 with H-17 envelope shape
  * ``strict=false`` (or absent)        → prompt-injection fallthrough
  * Streaming + non-streaming           → same gating + counter behaviour
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.api import response_format_metrics
from vllm_mlx.api.tool_calling import (
    is_strict_json_schema,
    validate_output_against_schema,
)
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.chat import router as chat_router
from vllm_mlx.routes.metrics import router as metrics_router

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


_VALID_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["value"],
    "additionalProperties": False,
}


_VALID_PAYLOAD = json.dumps({"value": 42})
_INVALID_PAYLOAD_PROSE = "Sure! Here you go: " + json.dumps({"value": 42})
_INVALID_PAYLOAD_WRONG_KEY = json.dumps({"number": 42})
_INVALID_PAYLOAD_OUT_OF_RANGE = json.dumps({"value": 200})


class _Engine:
    """Mock engine that returns a fixed buffered response.

    Tracks calls so tests can assert whether the route dispatched to
    ``generate_with_schema`` (constrained) or ``chat``/``stream_chat``
    (unconstrained).
    """

    preserve_native_tool_format = False
    is_mllm = False
    tokenizer = None

    def __init__(
        self,
        *,
        supports_guided: bool = True,
        guided_text: str = _VALID_PAYLOAD,
        chat_text: str = _VALID_PAYLOAD,
    ):
        # Property assignment is required because BaseEngine declares
        # ``supports_guided_generation`` as a property — overriding on
        # an instance is fine for a duck-typed mock.
        self.supports_guided_generation = supports_guided
        self._guided_text = guided_text
        self._chat_text = chat_text
        self.guided_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append(
            {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
        )
        return GenerationOutput(
            text=self._guided_text,
            new_text=self._guided_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def chat(self, *, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return GenerationOutput(
            text=self._chat_text,
            new_text=self._chat_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        yield GenerationOutput(
            text=self._chat_text,
            new_text=self._chat_text,
            prompt_tokens=4,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _make_client(engine: _Engine) -> TestClient:
    """Build a TestClient mounting both the chat router and /metrics."""
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(metrics_router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_metrics_between_tests():
    """Counters are process-local — tests would otherwise pollute each other."""
    response_format_metrics.reset_for_tests()
    yield
    response_format_metrics.reset_for_tests()


# ---------------------------------------------------------------------------
# Unit-level: is_strict_json_schema + validate_output_against_schema
# ---------------------------------------------------------------------------


def test_is_strict_json_schema_recognizes_dict_payloads():
    """Pure-helper unit: the strict flag must be detected on dict and
    typed-model shapes alike. ``extract_json_schema_for_guided`` already
    handles both arms; this helper must agree."""
    rf_strict = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": _VALID_SCHEMA, "strict": True},
    }
    rf_non_strict = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": _VALID_SCHEMA, "strict": False},
    }
    rf_no_strict_field = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": _VALID_SCHEMA},
    }
    assert is_strict_json_schema(rf_strict) is True
    assert is_strict_json_schema(rf_non_strict) is False
    assert is_strict_json_schema(rf_no_strict_field) is False
    assert is_strict_json_schema({"type": "json_object"}) is False
    assert is_strict_json_schema({"type": "text"}) is False
    assert is_strict_json_schema(None) is False


def test_validate_output_against_schema_accepts_valid_json():
    """Post-decode helper must accept output that parses + validates."""
    ok, err = validate_output_against_schema(_VALID_PAYLOAD, _VALID_SCHEMA)
    assert ok is True
    assert err is None


def test_validate_output_against_schema_rejects_prose_wrapped_json():
    """Prose-wrapped JSON must fail (no JSON parser fallback)."""
    ok, err = validate_output_against_schema(_INVALID_PAYLOAD_PROSE, _VALID_SCHEMA)
    assert ok is False
    assert err is not None
    assert "invalid JSON" in err


def test_validate_output_against_schema_rejects_schema_violation():
    """Schema violations must be flagged separately from JSON parse errors."""
    ok, err = validate_output_against_schema(
        _INVALID_PAYLOAD_WRONG_KEY, _VALID_SCHEMA
    )
    assert ok is False
    assert err is not None
    assert "schema violation" in err


def test_validate_output_against_schema_rejects_out_of_range_integer():
    """Numeric bounds violations must surface as schema violations."""
    ok, err = validate_output_against_schema(
        _INVALID_PAYLOAD_OUT_OF_RANGE, _VALID_SCHEMA
    )
    assert ok is False
    assert "schema violation" in (err or "")


def test_validate_output_against_schema_rejects_empty_output():
    """Empty text must NOT be silently passed through as valid."""
    ok, err = validate_output_against_schema("", _VALID_SCHEMA)
    assert ok is False
    assert err == "empty output"


# ---------------------------------------------------------------------------
# Route-level: strict=true + guided available
# ---------------------------------------------------------------------------


def _payload(*, strict: bool, stream: bool = False) -> dict:
    return {
        "model": "test-model",
        "stream": stream,
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Pick a number between 1 and 100"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "NumberOnly",
                "schema": _VALID_SCHEMA,
                "strict": strict,
            },
        },
    }


def test_strict_true_guided_available_non_streaming_routes_to_constrained():
    """Non-stream strict=true + outlines available → guided path used."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text

    # The guided path must have been hit; the unconstrained chat path
    # must not be the primary dispatch when outlines is available.
    assert len(engine.guided_calls) == 1, (
        f"expected 1 guided call, got {len(engine.guided_calls)} "
        f"(chat_calls={len(engine.chat_calls)})"
    )
    # The schema passed to outlines must be the raw user schema —
    # un-wrapped from ``response_format`` so outlines can interpret
    # ``$defs``/``$ref``/``additionalProperties:false`` natively (PR #419).
    assert engine.guided_calls[0]["json_schema"] == _VALID_SCHEMA

    # The strict counter must have ticked exactly once.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    # No violation expected for a valid response.
    assert snap["strict_violations_total"] == 0


def test_strict_true_guided_available_streaming_routes_to_constrained():
    """Stream strict=true + outlines available → guided streaming path."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True, stream=True))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")

    assert len(engine.guided_calls) == 1
    assert engine.stream_calls == []
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0


@pytest.mark.parametrize(
    "valid_payload",
    [
        json.dumps({"value": v})
        for v in [1, 2, 7, 13, 42, 50, 73, 88, 99, 100]
    ],
)
def test_strict_true_responses_validate_for_10_distinct_valid_payloads(valid_payload):
    """10-prompt sweep: every response the route admits MUST validate
    against the schema when outlines is in play.

    This is the contract-level pin: H-06 fix means even when the model
    would normally drift, the guided path produces output that
    ``jsonschema.validate`` accepts. The mock engine here returns
    schema-valid text; the real-server reproducer in the PR description
    runs the same shape against a live MLX model.
    """
    engine = _Engine(supports_guided=True, guided_text=valid_payload)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    parsed = json.loads(text)
    from jsonschema import validate

    validate(instance=parsed, schema=_VALID_SCHEMA)


# ---------------------------------------------------------------------------
# Route-level: strict=true + guided UNavailable → canonical 400
# ---------------------------------------------------------------------------


def test_strict_true_guided_unavailable_returns_canonical_400_non_streaming():
    """``[guided]`` missing must surface as 400 with the H-17 envelope shape."""
    engine = _Engine(supports_guided=False)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 400, resp.text
    body = resp.json()

    # H-17 canonical envelope shape: top-level "error" key, with
    # message / type / code / param subkeys.
    assert "error" in body
    err = body["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "guided_extra_required"
    assert "rapid-mlx[guided]" in err["message"]
    assert "pip install" in err["message"]
    # ``param`` must point at the strict flag specifically — clients
    # parsing the envelope use this to surface a targeted SDK error.
    assert "strict" in (err.get("param") or "")

    # No fallback to unconstrained generation: the 400 short-circuits
    # before either guided or chat is hit.
    assert engine.guided_calls == []
    assert engine.chat_calls == []

    # Counter still ticks: operators tracking strict traffic want to
    # see the rate even on installs that 400 them.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1


def test_strict_true_guided_unavailable_returns_canonical_400_streaming():
    """Streaming strict=true + no guided → same 400 BEFORE SSE begins."""
    engine = _Engine(supports_guided=False)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True, stream=True))
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "guided_extra_required"
    assert engine.stream_calls == []


# ---------------------------------------------------------------------------
# Route-level: strict=false → fallthrough to prompt-injection
# ---------------------------------------------------------------------------


def test_strict_false_guided_unavailable_falls_through_to_prompt_injection():
    """``strict=false`` without outlines must NOT 400.

    The existing prompt-injection contract is suggestion-only; clients
    that send ``strict=false`` are opting INTO that contract. Only
    strict=true triggers the enforcement gate.
    """
    engine = _Engine(supports_guided=False)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=False))
    assert resp.status_code == 200, resp.text
    # Strict counter must NOT have ticked — only strict=true requests count.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 0
    # Unconstrained chat path serves the response.
    assert len(engine.chat_calls) == 1


def test_strict_false_guided_available_routes_to_guided():
    """``strict=false`` + outlines available preserves the existing
    guided-routing behavior — the route opportunistically constrains
    whenever it can, regardless of the strict flag. Only the 400 gate
    is gated on ``strict=true``."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=False))
    assert resp.status_code == 200, resp.text
    assert len(engine.guided_calls) == 1
    # Strict counter must NOT have ticked — only strict=true requests count.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 0


# ---------------------------------------------------------------------------
# Belt-and-braces: post-decode violation counter
# ---------------------------------------------------------------------------


def test_post_decode_violation_bumps_counter_non_streaming():
    """If guided decoding silently produces invalid JSON, the violation
    counter must tick — outlines should make this unreachable, so a
    non-zero rate is the smoke alarm operators alert on.

    Simulated here by handing the mock engine a string that would
    *normally* fail validation: prose around the JSON body. The
    response still returns 200 (clients in strict-mode validate
    themselves) but the counter records the breach.
    """
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_PROSE)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text

    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_post_decode_violation_bumps_counter_streaming():
    """Streaming path must run the same belt-and-braces check."""
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_PROSE)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions", json=_payload(strict=True, stream=True)
    )
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_post_decode_schema_violation_bumps_counter():
    """Schema-violation (vs JSON-parse-failure) also bumps the counter."""
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_WRONG_KEY)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_violations_total"] == 1


# ---------------------------------------------------------------------------
# Faked-absent outlines (monkeypatch the import) — 400 envelope on missing extra
# ---------------------------------------------------------------------------


def test_strict_true_with_outlines_faked_absent_returns_canonical_400(monkeypatch):
    """Monkeypatch HAS_OUTLINES=False to simulate a base install
    without the ``[guided]`` extra. The route must surface the
    canonical 400, NOT silently degrade.

    The bug surface this gates: if a future engine refactor
    forwards ``supports_guided_generation=True`` while outlines is
    actually absent, the post-decode validator would catch the
    mismatch — but the operator-actionable signal needs to be the
    400 with the install hint, not a quiet 200 + violation counter
    increment. This test reaches all the way down to ``HAS_OUTLINES``
    so a refactor of ``supports_guided_generation`` to compose with
    ``HAS_OUTLINES`` keeps the 400 envelope intact.
    """
    from vllm_mlx.api import guided as guided_mod

    monkeypatch.setattr(guided_mod, "HAS_OUTLINES", False)

    # An engine that honestly reports supports_guided_generation=False
    # is the production shape on a base install. The route must 400.
    engine = _Engine(supports_guided=False)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "guided_extra_required"
    assert "pip install" in body["error"]["message"]
    assert "rapid-mlx[guided]" in body["error"]["message"]


# ---------------------------------------------------------------------------
# /metrics surfaces the new counters
# ---------------------------------------------------------------------------


def test_metrics_exposes_strict_counters_at_zero_on_clean_process():
    """Both counters must be present in /metrics output even when zero —
    dashboards prefer a flat line to a missing series."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "rapid_mlx_response_format_strict_total" in body
    assert "rapid_mlx_response_format_strict_violations_total" in body


def test_metrics_reflects_strict_request_count_after_traffic():
    """After three strict requests, the counter must read 3 in /metrics."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)
    for _ in range(3):
        resp = client.post("/v1/chat/completions", json=_payload(strict=True))
        assert resp.status_code == 200, resp.text
    resp = client.get("/metrics")
    body = resp.text
    # Look for the exact sample line for the counter.
    for line in body.splitlines():
        if line.startswith("rapid_mlx_response_format_strict_total ") and not line.startswith("#"):
            assert line.endswith(" 3"), line
            break
    else:
        raise AssertionError(
            "rapid_mlx_response_format_strict_total sample line missing"
        )
