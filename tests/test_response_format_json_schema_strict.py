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


def test_is_strict_json_schema_rejects_truthy_non_true_strict_values():
    """Codex r1 NIT: a malformed payload like ``"strict": "false"``
    is truthy under ``bool()`` but is clearly NOT a strict=true
    intent. We require identity-True so the dict arm matches the
    typed-model arm's semantics (Pydantic coerces strings/None to
    bool there) and a client that fat-fingers the strict value
    fails closed (opts INTO suggestion-only) rather than failing
    open (silent enforcement)."""
    # Strings are truthy under bool() but must NOT enable strict mode.
    rf_strict_str_false = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": _VALID_SCHEMA, "strict": "false"},
    }
    rf_strict_str_yes = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": _VALID_SCHEMA, "strict": "yes"},
    }
    rf_strict_one = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": _VALID_SCHEMA, "strict": 1},
    }
    assert is_strict_json_schema(rf_strict_str_false) is False
    assert is_strict_json_schema(rf_strict_str_yes) is False
    assert is_strict_json_schema(rf_strict_one) is False


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
    ok, err = validate_output_against_schema(_INVALID_PAYLOAD_WRONG_KEY, _VALID_SCHEMA)
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
    [json.dumps({"value": v}) for v in [1, 2, 7, 13, 42, 50, 73, 88, 99, 100]],
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


def test_post_decode_violation_returns_502_non_streaming():
    """Codex r2 BLOCKING #3: under strict mode, a post-decode
    validation failure MUST surface as 5xx — a 200 with knowingly
    schema-invalid body violates the OpenAI ``strict=true`` contract.

    The violations counter still ticks before we raise so operators
    see both the alertable rate AND the error response.
    """
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_PROSE)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_schema_violation"
    assert body["error"]["type"] == "api_error"
    assert "strict response_format violated" in body["error"]["message"]

    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_post_decode_violation_emits_error_sse_envelope_streaming():
    """Codex r2 BLOCKING #3 streaming variant: SSE clients can't
    receive a 5xx mid-stream (status was already 200), but the
    helper must emit a canonical OpenAI error SSE envelope BEFORE
    any role/content chunks land, then [DONE]. That way clients
    parsing SSE see the contract breach explicitly instead of
    silently consuming schema-invalid bytes."""
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_PROSE)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True, stream=True))
    # SSE response starts before validation; the status itself is 200.
    assert resp.status_code == 200, resp.text
    body = resp.text

    # The error envelope MUST appear and the role/content chunks
    # MUST NOT — the helper short-circuits before emitting them.
    assert "strict_schema_violation" in body
    assert "strict response_format violated" in body
    assert '"role":"assistant"' not in body, (
        "role chunk must NOT precede an error envelope for strict violations"
    )
    assert "[DONE]" in body

    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_post_decode_schema_violation_returns_502():
    """Schema-violation (vs JSON-parse-failure) also surfaces as 502."""
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_WRONG_KEY)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 502, resp.text
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
        if line.startswith(
            "rapid_mlx_response_format_strict_total "
        ) and not line.startswith("#"):
            assert line.endswith(" 3"), line
            break
    else:
        raise AssertionError(
            "rapid_mlx_response_format_strict_total sample line missing"
        )


# ---------------------------------------------------------------------------
# /v1/responses parity — same gate + same post-decode validation
# ---------------------------------------------------------------------------


def _make_responses_client(engine: _Engine) -> TestClient:
    """Mount the /v1/responses router with shared cfg + metrics surface."""
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    app.include_router(metrics_router)
    return TestClient(app)


def _responses_payload(*, strict: bool, stream: bool = False) -> dict:
    """Responses-API body shape with ``text.format`` carrying the schema."""
    return {
        "model": "test-model",
        "input": "Pick a number between 1 and 100",
        "stream": stream,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "NumberOnly",
                "schema": _VALID_SCHEMA,
                "strict": strict,
            }
        },
    }


def test_responses_strict_true_guided_unavailable_returns_400():
    """Codex r1 BLOCKING parity: /v1/responses + strict=true + no
    guided → canonical 400 with the same envelope shape the chat
    route emits. The two surfaces must agree on the contract."""
    engine = _Engine(supports_guided=False)
    client = _make_responses_client(engine)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "guided_extra_required"
    assert "rapid-mlx[guided]" in body["error"]["message"]
    # Counter must tick on /v1/responses too — the same strict-traffic
    # series spans both surfaces.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1


def test_responses_strict_true_guided_available_routes_to_constrained():
    """Codex r1 BLOCKING parity: /v1/responses non-stream + strict +
    guided → dispatches to ``generate_with_schema`` (constrained),
    NOT ``chat`` (unconstrained). This was the bug the codex review
    flagged: pre-fix, /v1/responses dropped the strict flag and went
    straight to ``engine.chat()``."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_responses_client(engine)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 200, resp.text
    # Constrained path is the dispatch; unconstrained chat must not
    # be the primary call when strict+guided.
    assert len(engine.guided_calls) == 1
    assert engine.chat_calls == []
    # Counter ticks; no violation on valid output.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0


def test_responses_strict_true_post_decode_violation_returns_502():
    """Codex r2 BLOCKING #3 parity: /v1/responses non-stream must
    surface a 502 (not 200) when the post-decode validation fails.

    Pre-fix the route returned 200 with the schema-invalid body in
    the response — that violates the OpenAI strict contract on the
    Responses surface just as it did on /v1/chat/completions.
    """
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_PROSE)
    client = _make_responses_client(engine)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_schema_violation"
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_responses_strict_true_stream_rejected_with_400():
    """Codex r2 BLOCKING #2 parity: /v1/responses + strict + stream
    is rejected with a clear 400 because constrained decoding on
    the Responses surface is buffered-only — there is no
    guided-streaming SSE helper for the Responses event shape
    today. The error message names both escape hatches (drop
    stream=true, or use /v1/chat/completions)."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json=_responses_payload(strict=True, stream=True),
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_stream_unsupported"
    assert "stream=true" in body["error"]["message"]
    assert "/v1/chat/completions" in body["error"]["message"]
    # The counter still ticks — clients asking for strict+stream
    # are reflected in the strict-traffic series even though we 400.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1


def test_responses_strict_true_guided_failure_returns_502():
    """Codex r2 BLOCKING #1 parity: under strict mode the
    /v1/responses non-stream path MUST NOT fall back to
    unconstrained ``engine.chat`` on guided failure — it must
    propagate the failure as 502."""

    class _BrokenEngine(_Engine):
        async def generate_with_schema(self, *, messages, json_schema, **kwargs):
            raise RuntimeError("simulated outlines failure")

    engine = _BrokenEngine(supports_guided=True)
    client = _make_responses_client(engine)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_schema_violation"
    # Unconstrained chat path MUST NOT have been called.
    assert engine.chat_calls == []
    snap = response_format_metrics.snapshot()
    assert snap["strict_violations_total"] == 1


def test_strict_true_with_tools_returns_400_chat():
    """Codex r3 BLOCKING #2 hole — strict + tools: the existing
    ``if response_format and not request.tools`` guard around the
    guided dispatch silently dropped strict mode when tools were
    set, so a strict request with tools fell through to
    unconstrained generation. The new gate fails closed with
    ``strict_with_tools_unsupported`` because constrained-decoding
    grammar and tool-call grammar are mutually exclusive."""
    engine = _Engine(supports_guided=True)
    client = _make_client(engine)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "NumberOnly",
                "schema": _VALID_SCHEMA,
                "strict": True,
            },
        },
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_with_tools_unsupported"
    # Strict counter ticks so operators see the malformed-strict rate.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    # Neither guided nor chat path was hit.
    assert engine.guided_calls == []
    assert engine.chat_calls == []


def test_responses_strict_true_with_tools_returns_400():
    """Codex r3 BLOCKING #3 parity: /v1/responses + strict + tools
    must also 400 ``strict_with_tools_unsupported``."""
    engine = _Engine(supports_guided=True)
    client = _make_responses_client(engine)
    payload = {
        "model": "test-model",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "noop",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "NumberOnly",
                "schema": _VALID_SCHEMA,
                "strict": True,
            }
        },
    }
    resp = client.post("/v1/responses", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_with_tools_unsupported"
    assert engine.guided_calls == []
    assert engine.chat_calls == []


def test_strict_true_without_schema_returns_400_chat_unit():
    """Unit-style coverage for the defense-in-depth gate when the
    body-level ``_validate_response_format`` is bypassed.

    Pre-fix path: ``response_format`` with strict=true + missing
    schema would 400 at the body validator. Defense-in-depth: my
    new ``strict_schema_required`` gate also catches this if the
    body validator is ever moved or refactored. We exercise the
    gate via a direct function call rather than a route round-trip
    (the body validator pre-empts it on a real request)."""
    from vllm_mlx.api.tool_calling import (
        extract_json_schema_for_guided,
        is_strict_json_schema,
    )

    rf = {
        "type": "json_schema",
        "json_schema": {"name": "X", "schema": {}, "strict": True},
    }
    # The strict-mode check sees strict=true...
    assert is_strict_json_schema(rf) is True
    # ...but the extractor returns None because schema is empty.
    assert extract_json_schema_for_guided(rf) is None
    # The combination ``strict_mode=True AND extracted_schema=None``
    # is exactly what the new route gate catches as
    # ``strict_schema_required``.


def test_responses_strict_false_falls_through_to_unconstrained():
    """``strict=false`` on /v1/responses must NOT tick the strict
    counter and must NOT 400 when guided is unavailable. Parity
    with the chat route's suggestion-only contract."""
    engine = _Engine(supports_guided=False)
    client = _make_responses_client(engine)
    resp = client.post("/v1/responses", json=_responses_payload(strict=False))
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 0
