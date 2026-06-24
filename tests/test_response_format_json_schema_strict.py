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
        guided_raises: Exception | None = None,
    ):
        # Property assignment is required because BaseEngine declares
        # ``supports_guided_generation`` as a property — overriding on
        # an instance is fine for a duck-typed mock.
        self.supports_guided_generation = supports_guided
        self._guided_text = guided_text
        self._chat_text = chat_text
        # ``guided_raises``: when set, ``generate_with_schema`` raises
        # this exception instead of returning a buffered output. Used
        # by codex r6 BLOCKING tests to prove the strict path refuses
        # to fall back to unconstrained generation on guided-engine
        # failure.
        self._guided_raises = guided_raises
        self.guided_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append(
            {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
        )
        if self._guided_raises is not None:
            raise self._guided_raises
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


def test_strict_true_guided_unavailable_returns_422_on_violation_non_streaming():
    """R12-4 — pre-R12-4 ``[guided]`` missing returned 400
    ``guided_extra_required`` and broke pydantic-ai. Post-R12-4 the
    request runs UNCONSTRAINED via ``engine.chat`` (no outlines),
    the route then validates the output against the schema and
    surfaces 422 with a structured ``json_schema_violation``
    envelope when validation fails after one repair retry.

    This test uses ``chat_text`` that violates the schema; both the
    initial generation and the repair retry will return the same
    invalid body, so the route must 422.
    """
    engine = _Engine(
        supports_guided=False,
        chat_text=_INVALID_PAYLOAD_OUT_OF_RANGE,
    )
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 422, resp.text
    body = resp.json()

    assert "error" in body
    err = body["error"]
    assert err["type"] == "validation_error"
    assert err["code"] == "json_schema_violation"
    assert err["param"] == "response_format.json_schema"
    # Structured envelope so SDKs (pydantic-ai) can decode the failing
    # path programmatically.
    details = err["details"]
    assert details["attempts"] == 2  # initial + repair
    assert details["reason"] == "schema_violation"
    assert details["failing_path"] == "/value"
    assert "maximum" in details["expected"]

    # The chat path WAS invoked twice — once initially, once for the
    # repair retry. Guided path is NEVER hit on a non-guided engine.
    assert engine.guided_calls == []
    assert len(engine.chat_calls) == 2

    # Counter still ticks for traffic shape AND violation surfacing.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1
    assert snap["strict_repairs_attempted_total"] == 1
    assert snap["strict_repairs_succeeded_total"] == 0


def test_strict_true_guided_unavailable_returns_200_when_initial_output_valid():
    """R12-4 happy-path — strict+no-guided + initial output valid →
    200 with the validated body (no repair retry needed)."""
    engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0
    assert snap["strict_repairs_attempted_total"] == 0
    # Only the initial chat call should fire; no repair retry needed.
    assert len(engine.chat_calls) == 1


def test_strict_true_guided_unavailable_repair_succeeds_returns_200():
    """R12-4 — first attempt invalid → repair attempt valid → 200 with
    repaired body. Pins the counter pair (attempts + successes both
    tick exactly once) and verifies the repair message carries the
    failure hint."""

    class _RepairingEngine(_Engine):
        """Returns invalid on first call, valid on every subsequent call.

        We detect the repair turn by call-index, NOT by message count —
        the route's prompt-injection helper adds a system message before
        the initial generation, so the FIRST call already has multiple
        messages. Counting calls is unambiguous.
        """

        def __init__(self):
            super().__init__(supports_guided=False)
            self.chat_text_first = _INVALID_PAYLOAD_OUT_OF_RANGE
            self.chat_text_repair = _VALID_PAYLOAD

        async def chat(self, *, messages, **kwargs):
            is_repair = len(self.chat_calls) > 0
            self.chat_calls.append({"messages": messages, "kwargs": kwargs})
            return GenerationOutput(
                text=self.chat_text_repair if is_repair else self.chat_text_first,
                new_text=self.chat_text_repair if is_repair else self.chat_text_first,
                prompt_tokens=4,
                completion_tokens=5,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    engine = _RepairingEngine()
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0
    assert snap["strict_repairs_attempted_total"] == 1
    assert snap["strict_repairs_succeeded_total"] == 1
    assert len(engine.chat_calls) == 2
    # Repair call must have the structured hint in its messages.
    repair_messages = engine.chat_calls[1]["messages"]
    assert any(
        m.get("role") == "system" and "REPAIR" in (m.get("content") or "").upper()
        for m in repair_messages
    )
    # Codex r2 #3: the response's reported token usage MUST aggregate
    # both attempts. Each ``_Engine.chat`` call reports
    # prompt_tokens=4 + completion_tokens=5, so the final response
    # should bill prompt_tokens=8 + completion_tokens=10.
    body = resp.json()
    usage = body["usage"]
    assert usage["prompt_tokens"] == 8, (
        f"repair-success path must aggregate prompt tokens (got {usage})"
    )
    assert usage["completion_tokens"] == 10, (
        f"repair-success path must aggregate completion tokens (got {usage})"
    )


def test_strict_true_guided_unavailable_disable_flag_skips_enforcement(monkeypatch):
    """R12-4 escape hatch — ``RAPID_MLX_STRICT_JSON_SCHEMA=off`` restores
    the pre-R12-4 silent-pass-through behavior. Schema-violating output
    is returned as 200 (legacy compat) instead of 422."""
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA", "off")
    engine = _Engine(
        supports_guided=False,
        chat_text=_INVALID_PAYLOAD_OUT_OF_RANGE,
    )
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200, resp.text
    # Counter still ticks (operator observability) but no violation or
    # repair attempt — the disable flag short-circuits enforcement.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0
    assert snap["strict_repairs_attempted_total"] == 0


def test_strict_true_guided_unavailable_repair_engine_failure_returns_502():
    """Codex r1 #3 — a non-timeout exception from ``engine.chat``
    during the REPAIR turn is a server-side failure, NOT a client
    schema-violation. Pre-fix the route swallowed the exception
    and surfaced 422 ``json_schema_violation`` using the ORIGINAL
    validation failure, misleading the client. Post-fix the route
    raises 502 ``strict_repair_engine_failure`` with the original
    validation context preserved in ``details.initial_failure``.
    """

    class _EngineThatBreaksOnRepair(_Engine):
        async def chat(self, *, messages, **kwargs):
            is_repair = len(self.chat_calls) > 0
            self.chat_calls.append({"messages": messages, "kwargs": kwargs})
            if is_repair:
                raise RuntimeError("simulated engine wedge during repair")
            return GenerationOutput(
                text=_INVALID_PAYLOAD_OUT_OF_RANGE,
                new_text=_INVALID_PAYLOAD_OUT_OF_RANGE,
                prompt_tokens=4,
                completion_tokens=5,
                finished=True,
                finish_reason="stop",
                channel=None,
            )

    engine = _EngineThatBreaksOnRepair(supports_guided=False)
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_repair_engine_failure"
    assert body["error"]["type"] == "api_error"
    # Initial failure context preserved.
    assert body["error"]["details"]["initial_failure"]["reason"] == "schema_violation"
    assert body["error"]["details"]["repair_exception"] == "RuntimeError"


def test_strict_true_guided_unavailable_repair_disable_flag_skips_retry(monkeypatch):
    """R12-4 — ``RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR=off`` disables ONLY
    the repair retry; the post-decode validation + 422 envelope still
    fires (strict mode stays a hard contract; only the retry is
    skipped). One chat call (no retry) then 422."""
    monkeypatch.setenv("RAPID_MLX_STRICT_JSON_SCHEMA_REPAIR", "off")
    engine = _Engine(
        supports_guided=False,
        chat_text=_INVALID_PAYLOAD_OUT_OF_RANGE,
    )
    client = _make_client(engine)

    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation"
    assert body["error"]["details"]["attempts"] == 1
    assert len(engine.chat_calls) == 1
    snap = response_format_metrics.snapshot()
    assert snap["strict_repairs_attempted_total"] == 0


def _parse_sse_events(body: str) -> list[dict]:
    """Parse an SSE response body into a list of ``{event, data}`` dicts.

    Each SSE event is a blank-line-separated block; within a block,
    ``event:`` and ``data:`` lines accumulate the named event type
    and the data payload respectively. The ``data:`` field is left
    as-is (string) so the caller can choose between JSON-decoding and
    sentinel matching (``[DONE]``). The ``event`` field defaults to
    the SSE-spec ``"message"`` when no ``event:`` line is present.

    Codex r2 #4: pre-fix the streaming tests grepped the full body
    for substrings, so an envelope landing AFTER an already-success
    ``stop`` chunk OR encoded as a non-named SSE event would still
    pass. Parsing the frames in order lets us assert (a) the
    violation chunk is the FIRST finish_reason the client sees,
    (b) the error envelope is wired as a named ``event:
    chat.completion.error`` block, and (c) ``[DONE]`` arrives last.
    """
    events: list[dict] = []
    current: dict[str, str] = {}
    for raw_line in body.split("\n"):
        line = raw_line.rstrip("\r")
        if line == "":
            if current:
                # Default event type per SSE spec.
                current.setdefault("event", "message")
                events.append(current)
                current = {}
            continue
        if line.startswith("event: "):
            current["event"] = line[len("event: ") :]
        elif line.startswith("data: "):
            # Multi-line data lines are joined with "\n" per spec.
            payload = line[len("data: ") :]
            if "data" in current:
                current["data"] = current["data"] + "\n" + payload
            else:
                current["data"] = payload
    if current:
        current.setdefault("event", "message")
        events.append(current)
    return events


def test_strict_true_guided_unavailable_streaming_emits_violation_finish():
    """R12-4 streaming variant — the unconstrained stream is emitted
    as usual; on validation failure the wrapper REPLACES the upstream
    terminal ``finish_reason="stop"`` chunk with ``finish_reason="json_schema_violation"``
    and emits the 422-shaped envelope as a SINGLE named SSE event
    (``event: chat.completion.error\\ndata: ...``) before ``[DONE]``.

    The parsed-frames assertions pin:
        * the FIRST finish_reason a client sees on a violating
          response is ``json_schema_violation`` (NOT a leading
          ``stop`` chunk that lets the client finalize early);
        * the error envelope appears as a named SSE event exactly
          once — codex r5 #1 explicitly rejected the prior
          double-emit (named + plain-data) because a client that
          consumes both forms would handle the same terminal error
          twice;
        * ``[DONE]`` arrives last so the wire envelope still closes.
    """
    engine = _Engine(
        supports_guided=False,
        chat_text=_INVALID_PAYLOAD_OUT_OF_RANGE,
    )
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json=_payload(strict=True, stream=True),
    )
    assert resp.status_code == 200, resp.text  # SSE wire is 200; body carries the error
    events = _parse_sse_events(resp.text)

    # The body MUST end with [DONE].
    assert events, "no SSE events parsed from body"
    assert events[-1]["data"] == "[DONE]", (
        f"final SSE event must be [DONE], got {events[-1]}"
    )

    # Codex r2 #1: the FIRST finish_reason the client encounters
    # MUST be ``json_schema_violation``. A leading ``stop`` chunk
    # would let spec-compliant clients finalize on success and miss
    # the violation entirely.
    first_finish_reason = None
    for ev in events:
        data = ev.get("data", "")
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for ch in payload.get("choices") or []:
            if isinstance(ch, dict) and ch.get("finish_reason"):
                first_finish_reason = ch["finish_reason"]
                break
        if first_finish_reason is not None:
            break
    assert first_finish_reason == "json_schema_violation", (
        f"first finish_reason was {first_finish_reason!r}, expected "
        f"json_schema_violation. The upstream terminal chunk leaked through."
    )

    # Codex r5 #1: the envelope MUST be emitted as a SINGLE named SSE
    # event (``event: chat.completion.error\ndata: <json>``) — NOT
    # also as a separate plain ``data: <json>`` line. A prior
    # iteration emitted both; clients that read both named events AND
    # plain data lines would then handle the same terminal error
    # twice. Per SSE spec the ``event: chat.completion.error`` form
    # is parsed as one message event by both EventSource (dispatched
    # to the named listener) AND plain-line consumers like the OpenAI
    # SDK or curl (who ignore the ``event:`` field and consume the
    # ``data:`` payload).
    named_error_events = [
        ev for ev in events if ev.get("event") == "chat.completion.error"
    ]
    assert len(named_error_events) == 1, (
        f"expected EXACTLY ONE named `event: chat.completion.error` SSE "
        f"block; got {len(named_error_events)}. A double-emit causes "
        f"clients reading both named events AND plain data: lines to "
        f"double-count the terminal error."
    )
    err_payload = json.loads(named_error_events[0]["data"])
    assert err_payload["error"]["code"] == "json_schema_violation"
    assert err_payload["object"] == "chat.completion.error"

    # Codex r5 #1 negative-pin: there must be NO plain-data envelope
    # carrying the ``chat.completion.error`` object. The error event
    # is single-emit, named-only.
    plain_data_error_count = 0
    for ev in events:
        if ev.get("event") != "message":
            continue
        data = ev.get("data", "")
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("object") == "chat.completion.error"
        ):
            plain_data_error_count += 1
    assert plain_data_error_count == 0, (
        f"found {plain_data_error_count} plain-data `chat.completion.error` "
        f"envelopes — codex r5 #1 banned this double-emit. The error event "
        f"must be the named form only."
    )

    # On streaming we do NOT run repair retry — no attempt counter
    # increment.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1
    assert snap["strict_repairs_attempted_total"] == 0


def test_strict_true_guided_unavailable_streaming_happy_path_passes_through():
    """R12-4 streaming variant happy path — when validation passes,
    the wrapper releases the held upstream terminal chunk in order
    (``finish_reason="stop"``) and emits ``[DONE]``. No
    ``json_schema_violation`` chunk and no ``chat.completion.error``
    envelope must leak through."""
    engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
    client = _make_client(engine)

    resp = client.post(
        "/v1/chat/completions",
        json=_payload(strict=True, stream=True),
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse_events(resp.text)
    # No violation chunk and no error event must appear.
    body_text = resp.text
    assert "json_schema_violation" not in body_text
    assert "chat.completion.error" not in body_text
    # The last non-DONE event must carry finish_reason="stop".
    assert events[-1]["data"] == "[DONE]"
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0


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


def test_strict_true_with_outlines_module_faked_absent(monkeypatch):
    """R12-4 — when ``guided.HAS_OUTLINES`` is False the production
    ``BatchedEngine.supports_guided_generation`` property composes
    to False. Post-R12-4, that no longer triggers a 400 — it
    triggers the post-generate validation + repair retry path.
    Pin the composition contract here so refactors that decouple
    HAS_OUTLINES → guided-availability are caught."""
    from vllm_mlx.api import guided as guided_mod

    # Before the monkeypatch, the guided extra IS installed.
    assert guided_mod.is_guided_available() is True

    monkeypatch.setattr(guided_mod, "HAS_OUTLINES", False)

    # After the monkeypatch, the helper composes to False.
    assert guided_mod.is_guided_available() is False
    # An engine honestly reporting ``supports_guided_generation=False``
    # (the production shape on a HAS_OUTLINES=False install) now
    # runs the unconstrained chat path and validates; with a valid
    # output the request returns 200.
    engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 200
    assert len(engine.chat_calls) == 1


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


@pytest.fixture
def _rate_limiter_state():
    """Codex r4 BLOCKING #2: save and restore rate_limiter state.

    The /v1/responses fixture used to disable the global
    ``rate_limiter`` and clear its state without restoring it, so
    this test file polluted later tests that depend on rate
    limiting being enabled. This fixture snapshots state at entry
    and restores it on teardown.
    """
    from vllm_mlx.middleware.auth import rate_limiter

    saved_enabled = rate_limiter.enabled
    saved_rpm = rate_limiter.requests_per_minute
    saved_requests = dict(rate_limiter._requests)
    rate_limiter.enabled = False
    rate_limiter.requests_per_minute = 60
    rate_limiter._requests.clear()
    yield rate_limiter
    rate_limiter.enabled = saved_enabled
    rate_limiter.requests_per_minute = saved_rpm
    rate_limiter._requests.clear()
    rate_limiter._requests.update(saved_requests)


def _make_responses_client(engine: _Engine, rate_limiter_state=None) -> TestClient:
    """Mount the /v1/responses router with shared cfg + metrics surface.

    Callers must hold ``_rate_limiter_state`` fixture to ensure the
    global rate-limiter state is restored on test teardown — without
    it, the disabled state leaks into subsequent tests.
    """
    from vllm_mlx.middleware.auth import rate_limiter
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    # Defensive: if a caller forgot the fixture, still disable
    # rate-limiting for the test body. The fixture handles restore;
    # without it, state leaks (which is the codex r4 bug, hence
    # the audit-required parameter shape below).
    if rate_limiter_state is None:
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


def test_responses_strict_true_guided_unavailable_runs_postgen_validation(
    _rate_limiter_state,
):
    """R12-4 parity on /v1/responses — pre-R12-4 the route 400'd
    ``guided_extra_required`` here. Post-R12-4 the route runs the
    unconstrained chat path, validates the output, attempts a single
    repair retry on failure, and surfaces 422 if both attempts
    fail. The two surfaces (chat + responses) must agree on the
    contract."""
    engine = _Engine(
        supports_guided=False,
        chat_text=_INVALID_PAYLOAD_OUT_OF_RANGE,
    )
    client = _make_responses_client(engine, _rate_limiter_state)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation"
    assert body["error"]["type"] == "validation_error"
    assert body["error"]["param"] == "text.format"
    # Counter must tick on /v1/responses too.
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1
    assert snap["strict_repairs_attempted_total"] == 1


def test_responses_strict_true_guided_unavailable_valid_returns_200(
    _rate_limiter_state,
):
    """R12-4 happy path on /v1/responses — strict + no guided +
    valid output → 200, no violation counter increment."""
    engine = _Engine(supports_guided=False, chat_text=_VALID_PAYLOAD)
    client = _make_responses_client(engine, _rate_limiter_state)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 0


def test_responses_strict_true_guided_available_routes_to_constrained(
    _rate_limiter_state,
):
    """Codex r1 BLOCKING parity: /v1/responses non-stream + strict +
    guided → dispatches to ``generate_with_schema`` (constrained),
    NOT ``chat`` (unconstrained). This was the bug the codex review
    flagged: pre-fix, /v1/responses dropped the strict flag and went
    straight to ``engine.chat()``."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_responses_client(engine, _rate_limiter_state)
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


def test_responses_strict_true_post_decode_violation_returns_502(_rate_limiter_state):
    """Codex r2 BLOCKING #3 parity: /v1/responses non-stream must
    surface a 502 (not 200) when the post-decode validation fails.

    Pre-fix the route returned 200 with the schema-invalid body in
    the response — that violates the OpenAI strict contract on the
    Responses surface just as it did on /v1/chat/completions.
    """
    engine = _Engine(supports_guided=True, guided_text=_INVALID_PAYLOAD_PROSE)
    client = _make_responses_client(engine, _rate_limiter_state)
    resp = client.post("/v1/responses", json=_responses_payload(strict=True))
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_schema_violation"
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_responses_strict_true_stream_rejected_with_400(_rate_limiter_state):
    """Codex r2 BLOCKING #2 parity: /v1/responses + strict + stream
    is rejected with a clear 400 because constrained decoding on
    the Responses surface is buffered-only — there is no
    guided-streaming SSE helper for the Responses event shape
    today. The error message names both escape hatches (drop
    stream=true, or use /v1/chat/completions)."""
    engine = _Engine(supports_guided=True, guided_text=_VALID_PAYLOAD)
    client = _make_responses_client(engine, _rate_limiter_state)
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


def test_responses_strict_true_guided_failure_returns_502(_rate_limiter_state):
    """Codex r2 BLOCKING #1 parity: under strict mode the
    /v1/responses non-stream path MUST NOT fall back to
    unconstrained ``engine.chat`` on guided failure — it must
    propagate the failure as 502."""

    class _BrokenEngine(_Engine):
        async def generate_with_schema(self, *, messages, json_schema, **kwargs):
            raise RuntimeError("simulated outlines failure")

    engine = _BrokenEngine(supports_guided=True)
    client = _make_responses_client(engine, _rate_limiter_state)
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


def test_responses_strict_true_with_tools_returns_400(_rate_limiter_state):
    """Codex r3 BLOCKING #3 parity: /v1/responses + strict + tools
    must also 400 ``strict_with_tools_unsupported``."""
    engine = _Engine(supports_guided=True)
    client = _make_responses_client(engine, _rate_limiter_state)
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


def test_strict_helper_composition_extract_returns_none_for_empty_schema():
    """Unit-level pin on the ``strict_mode AND no schema`` shape.

    Codex r4 BLOCKING #3 (rewording fix): this test never claimed
    to exercise the route gate; it pins the helper composition that
    the gate depends on. The route gate IS exercised by
    ``test_strict_true_with_tools_returns_400_chat`` and
    ``test_responses_strict_true_with_tools_returns_400`` — both
    make real requests and assert the 400 envelope. The defense-
    in-depth ``strict_schema_required`` 400 in the route is
    pre-empted in production by ``_validate_response_format`` at
    body-parse time, but this test still earns its keep by pinning
    the helper invariant the gate relies on (a refactor that lets
    ``extract_json_schema_for_guided`` return ``{}`` instead of
    ``None`` for empty schemas would silently break the gate
    without this assertion).
    """
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
    # ...but the extractor returns None (not {}!) because schema is empty.
    extracted = extract_json_schema_for_guided(rf)
    assert extracted is None, (
        "extract_json_schema_for_guided must return None (not {}) for "
        "an empty schema so the route gate's ``if not _strict_schema_check`` "
        "check catches the malformed-strict case. A refactor that returns "
        "``{}`` would silently bypass the gate."
    )


def test_responses_strict_false_falls_through_to_unconstrained(_rate_limiter_state):
    """``strict=false`` on /v1/responses must NOT tick the strict
    counter and must NOT 400 when guided is unavailable. Parity
    with the chat route's suggestion-only contract."""
    engine = _Engine(supports_guided=False)
    client = _make_responses_client(engine, _rate_limiter_state)
    resp = client.post("/v1/responses", json=_responses_payload(strict=False))
    assert resp.status_code == 200, resp.text
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 0


# --- Codex r4 NIT #5: schema-validity gate -------------------------------


_INVALID_SCHEMA_TYPO = {
    # ``"objct"`` is not a valid JSON-Schema ``type`` keyword. The
    # post-decode validator would catch this AFTER spending the
    # decode budget and surface it as a 502 strict_schema_violation
    # (server-side breach shape). The pre-flight ``check_schema_validity``
    # gate must reject it as a 400 invalid_strict_schema (client-side
    # breach shape) BEFORE generation.
    "type": "objct",
    "properties": {"value": {"type": "integer"}},
}


def test_check_schema_validity_accepts_valid_schema():
    """Helper unit test: a well-formed Draft-7 schema returns
    ``(True, None)``."""
    from vllm_mlx.api.tool_calling import check_schema_validity

    ok, err = check_schema_validity(_VALID_SCHEMA)
    assert ok is True
    assert err is None


def test_check_schema_validity_rejects_invalid_type_keyword():
    """Helper unit test: a structurally-invalid schema (a typo in
    the ``type`` keyword) returns ``(False, <reason>)`` so the
    route can echo the reason in the 400 envelope."""
    from vllm_mlx.api.tool_calling import check_schema_validity

    ok, err = check_schema_validity(_INVALID_SCHEMA_TYPO)
    assert ok is False
    assert err is not None and len(err) > 0


def test_strict_true_invalid_schema_returns_400_chat():
    """Codex r4 NIT #5: /v1/chat/completions + strict=true + an
    invalid JSON Schema (typo in ``type``) must surface as a
    400 ``invalid_strict_schema`` pointing at the client's
    malformed input, NOT a 502 ``strict_schema_violation``
    (server-side breach shape). The post-decode validator path
    is for the case where generation succeeded but the output
    didn't validate — invalid input schemas must fail closed
    BEFORE generation."""
    engine = _Engine(supports_guided=True)
    client = _make_client(engine)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "BadSchema",
                "schema": _INVALID_SCHEMA_TYPO,
                "strict": True,
            },
        },
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "invalid_strict_schema"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "response_format.json_schema.schema"
    # Strict counter still ticks so operators see the malformed-strict
    # rate (parity with strict_schema_required + strict_with_tools_unsupported).
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    # Generation must NOT have run — the gate fires before the
    # engine is touched.
    assert engine.guided_calls == []
    assert engine.chat_calls == []


def test_responses_strict_true_invalid_schema_returns_400(_rate_limiter_state):
    """Codex r4 NIT #5 parity: /v1/responses + strict=true + an
    invalid JSON Schema must surface as a 400 ``invalid_strict_schema``
    on this surface too."""
    engine = _Engine(supports_guided=True)
    client = _make_responses_client(engine, _rate_limiter_state)
    payload = {
        "model": "test-model",
        "input": "hi",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "BadSchema",
                "schema": _INVALID_SCHEMA_TYPO,
                "strict": True,
            }
        },
    }
    resp = client.post("/v1/responses", json=payload)
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "invalid_strict_schema"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "text.format.schema"
    assert engine.guided_calls == []
    assert engine.chat_calls == []


# --- Codex r5 BLOCKING: kwargs-collision shielding -----------------------


@pytest.mark.asyncio
async def test_strict_true_stream_helper_strips_colliding_raise_on_failure():
    """Codex r5+r7 BLOCKING (real version): the streaming chat strict
    helper, ``stream_chat_completion_guided``, sanitizes ``kwargs``
    so the explicit ``raise_on_failure=True`` it passes to
    ``engine.generate_with_schema`` cannot collide with an
    upstream-set value.

    Pin: call the helper directly with ``raise_on_failure=False``
    in ``kwargs``. The production sanitization (``_guided_kwargs =
    {k:v for k,v in kwargs.items() if k != "raise_on_failure"}``)
    must strip the colliding key so the call resolves to a single
    ``raise_on_failure=True``. Without the sanitization, Python
    would raise ``TypeError: got multiple values for keyword
    argument 'raise_on_failure'`` and the test would fail.
    """
    from fastapi import Request

    from vllm_mlx.api.models import ChatCompletionRequest
    from vllm_mlx.routes.chat import stream_chat_completion_guided

    engine = _Engine(supports_guided=True)
    # Minimal valid ChatCompletionRequest pin — only the fields
    # the helper reads off ``request`` are populated.
    chat_req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    class _MockReq:
        async def is_disconnected(self):
            return False

    raw_req: Request = _MockReq()  # type: ignore[assignment]

    # The collision case: kwargs ALREADY contains
    # ``raise_on_failure=False`` — a stale / contradictory value.
    # The helper's sanitization must strip it and the explicit
    # True (set by the helper itself) must win. If the
    # sanitization were removed, this would TypeError during
    # the ``generate_with_schema(...)`` call.
    chunks: list[str] = []
    async for chunk in stream_chat_completion_guided(
        engine,
        messages=[{"role": "user", "content": "hi"}],
        request=chat_req,
        json_schema=_VALID_SCHEMA,
        strict_mode=True,
        raise_on_failure=False,  # colliding stale value
    ):
        # Pass raw_request positionally would mismatch the
        # signature; the helper takes ``request`` only. ``raw_request``
        # is used via ``request.is_disconnected`` in disconnect
        # checks, but our request fixture handles that. We don't
        # need to pass raw_request — the helper signature is
        # ``(engine, messages, request, json_schema, *, strict_mode,
        # **kwargs)``.
        chunks.append(chunk)
    assert len(engine.guided_calls) == 1
    recorded_kwargs = engine.guided_calls[0]["kwargs"]
    # The colliding ``raise_on_failure=False`` was stripped, and
    # the helper's explicit ``raise_on_failure=True`` won.
    assert recorded_kwargs.get("raise_on_failure") is True
    # SSE stream completed normally (no TypeError translated into
    # a strict_schema_violation envelope).
    body = "".join(chunks)
    assert "strict_schema_violation" not in body
    assert "[DONE]" in body


def test_strict_true_responses_strips_colliding_raise_on_failure(
    _rate_limiter_state, monkeypatch
):
    """Codex r5+r7 BLOCKING parity: same proof on /v1/responses.

    The /v1/responses route doesn't accept ``raise_on_failure`` as
    a top-level body field, so we patch the route's chat-kwargs
    builder to inject a colliding key into ``chat_kwargs``. The
    production sanitization in ``responses.py`` (the
    ``_guided_kwargs`` filter) must strip it; otherwise the route
    would TypeError and surface as 502 strict_schema_violation.

    A clean 200 + recorded ``raise_on_failure=True`` proves the
    dedup is effective.
    """
    engine = _Engine(supports_guided=True)
    client = _make_responses_client(engine, _rate_limiter_state)

    # Patch ``_resolved_sampling_kwargs`` (or wherever ``chat_kwargs``
    # is composed) to inject a colliding key. The simplest hook is
    # to monkeypatch ``_resolved_sampling_kwargs`` to add the field
    # to its return value.
    from vllm_mlx.routes import responses as responses_mod

    original_sampler = responses_mod._resolved_sampling_kwargs

    def _polluted_sampler(req):
        out = dict(original_sampler(req))
        # Inject the colliding key the production code must strip.
        out["raise_on_failure"] = False
        return out

    monkeypatch.setattr(responses_mod, "_resolved_sampling_kwargs", _polluted_sampler)

    payload = {
        "model": "test-model",
        "input": "hi",
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
    # Without the sanitization, Python would raise TypeError and
    # the route's ``except Exception`` arm would translate to a
    # 502 strict_schema_violation. A clean 200 proves the dedup.
    assert resp.status_code == 200, resp.text
    assert len(engine.guided_calls) == 1
    recorded_kwargs = engine.guided_calls[0]["kwargs"]
    assert recorded_kwargs.get("raise_on_failure") is True


def test_check_schema_validity_propagates_dependency_failures(monkeypatch):
    """Codex r5 NIT: an environment failure in ``jsonschema``
    (e.g. import broken in the deployed wheel) must surface as
    a server-side error, NOT be mistranslated into a 400
    ``invalid_strict_schema`` that tells the client to fix their
    perfectly-valid schema.

    Pin: when the bound validator's ``check_schema`` raises a
    generic non-SchemaError exception (simulating a runtime/
    dependency bug — NOT malformed input), ``check_schema_validity``
    must let it propagate instead of returning ``(False, ...)``.
    """
    from vllm_mlx.api import tool_calling

    class _BoomError(RuntimeError):
        pass

    class _BrokenValidator:
        @staticmethod
        def check_schema(schema):
            raise _BoomError("simulated environment failure inside jsonschema")

    # Codex r7 BLOCKING #1: ``check_schema_validity`` now selects
    # the validator class dynamically via
    # ``jsonschema.validators.validator_for``, so we patch that
    # entry point instead of the (now-unused) ``Draft7Validator``.
    from jsonschema import validators

    monkeypatch.setattr(validators, "validator_for", lambda schema: _BrokenValidator)
    with pytest.raises(_BoomError):
        tool_calling.check_schema_validity(_VALID_SCHEMA)


def test_check_schema_validity_uses_declared_draft_via_schema_key():
    """Codex r7 BLOCKING #1: a request that declares
    ``$schema:"https://json-schema.org/draft/2020-12/schema"`` must
    be preflighted with a 2020-12 validator, NOT with Draft7. The
    pre-fix preflight hard-coded ``Draft7Validator``, which ignores
    unknown keywords and would pass a 2020-12 schema that the
    post-decode validator (using the declared draft) then rejected
    — surfacing as a 502 strict_schema_violation for what was
    actually a client schema-version mismatch.

    Pin: the helper must call ``validator_for(schema)`` so the
    preflight matches the declared draft.
    """
    from jsonschema import Draft202012Validator, validators

    from vllm_mlx.api import tool_calling

    # A 2020-12 schema declaring a feature that DRAFT-7 silently
    # ignores: ``prefixItems`` (added in 2020-12; Draft-7 has only
    # ``items``). If the helper preflighted with Draft-7, the
    # invalid ``prefixItems`` would pass and we'd never know the
    # declared validator would have flagged it.
    schema_2020_12 = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        # An invalid prefixItems entry: ``"not-a-schema"`` is a
        # string, not a schema object. 2020-12 ``check_schema``
        # rejects this; Draft-7 doesn't know the keyword and
        # silently accepts the whole document.
        "prefixItems": ["not-a-schema"],
    }
    # Confirm Draft-7 silently accepts (the bug we're closing):
    from jsonschema import Draft7Validator

    Draft7Validator.check_schema(schema_2020_12)  # no raise -> bug shape
    # Confirm 2020-12 rejects (the right behavior):
    with pytest.raises(Exception):
        Draft202012Validator.check_schema(schema_2020_12)
    # And confirm ``validator_for`` picks the 2020-12 class:
    assert validators.validator_for(schema_2020_12) is Draft202012Validator
    # The helper, having switched to ``validator_for``, must now
    # ALSO reject — surfacing the schema-version mismatch as a
    # client 400 instead of letting it bleed through to a 502.
    ok, err = tool_calling.check_schema_validity(schema_2020_12)
    assert ok is False
    assert err is not None and len(err) > 0


# --- Codex r6 BLOCKING: guided-failure must not silently fall back -------


def test_strict_true_streaming_guided_raises_emits_error_sse_no_fallback():
    """Codex r6 BLOCKING: when the strict streaming path's
    ``generate_with_schema`` raises (outlines API change, grammar
    compilation failure, runtime error in the executor task), the
    helper PREVIOUSLY fell back to ``stream_chat_completion`` —
    silently emitting unconstrained SSE chunks under a contract
    the client said was strict. Fix: refuse the fallback under
    strict mode, emit a canonical SSE error envelope + DONE.
    """
    engine = _Engine(
        supports_guided=True,
        guided_raises=RuntimeError("outlines grammar compile failed"),
    )
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True, stream=True))
    assert resp.status_code == 200, resp.text  # SSE response status
    body = resp.text
    # The error envelope MUST appear and the unconstrained fallback
    # MUST NOT have run (no chat_calls, no stream_calls).
    assert "strict_schema_violation" in body
    assert "strict response_format could not be honored" in body
    assert '"role":"assistant"' not in body, (
        "role chunk must NOT precede strict-violation error envelope"
    )
    assert "[DONE]" in body
    assert engine.chat_calls == [], "non-stream chat fallback must NOT run"
    assert engine.stream_calls == [], "streaming chat fallback must NOT run"

    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_strict_true_non_streaming_guided_raises_returns_502_no_fallback():
    """Codex r6 BLOCKING parity (non-streaming chat path): under
    strict=true, ``generate_with_schema`` raising must surface as
    502 ``strict_schema_violation`` directly — NOT fall back to
    ``engine.chat`` (which the pre-fix code did, hoping the
    buffered output would coincidentally validate). The buffered
    post-decode validator was a safety net for the case where the
    fallback validates; it isn't a contract guarantee."""
    engine = _Engine(
        supports_guided=True,
        guided_raises=RuntimeError("outlines grammar compile failed"),
    )
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=True))
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_schema_violation"
    assert body["error"]["type"] == "api_error"
    assert "strict response_format could not be honored" in body["error"]["message"]
    # Unconstrained fallback MUST NOT have run.
    assert engine.chat_calls == []
    assert engine.stream_calls == []

    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 1
    assert snap["strict_violations_total"] == 1


def test_strict_false_streaming_guided_raises_still_falls_back():
    """Suggestion-only ``strict=false`` MUST keep the legacy
    fallback semantics — the H-06 fix only changes behavior under
    the strict contract. A strict=false caller asking for a JSON
    schema gets best-effort: outlines tries, and if it fails the
    unconstrained streaming path takes over."""
    engine = _Engine(
        supports_guided=True,
        guided_raises=RuntimeError("outlines grammar compile failed"),
    )
    client = _make_client(engine)
    resp = client.post("/v1/chat/completions", json=_payload(strict=False, stream=True))
    assert resp.status_code == 200, resp.text
    body = resp.text
    # No strict_schema_violation envelope, and the fallback streaming
    # path was used.
    assert "strict_schema_violation" not in body
    assert len(engine.stream_calls) == 1, (
        f"expected 1 unconstrained stream fallback call, got {len(engine.stream_calls)}"
    )
    # The strict counter must NOT tick (suggestion-only path).
    snap = response_format_metrics.snapshot()
    assert snap["strict_requests_total"] == 0
    assert snap["strict_violations_total"] == 0


def test_check_schema_validity_rejects_non_mapping_input():
    """A list/string passed as ``schema`` is client-side malformed
    input (not a server dependency failure). The helper must
    return ``(False, <reason>)`` so the route 400s with
    ``invalid_strict_schema``."""
    from vllm_mlx.api.tool_calling import check_schema_validity

    ok, err = check_schema_validity(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert ok is False
    assert err is not None and len(err) > 0


# --- Codex r8 BLOCKING (false positive pin) ----------------------------


class _SyncFailureEngine(_Engine):
    """Engine variant whose ``generate_with_schema`` raises
    SYNCHRONOUSLY at call time, not at coroutine-await time.

    Production code path: ``engine.generate_with_schema(...)`` is
    called inside a ``try`` block; if outlines import is broken or
    the engine does sync grammar setup that errors out, the call
    raises BEFORE returning a coroutine. The route's tight try
    around the call must catch this and surface it as a 502
    ``strict_schema_violation`` — NOT escape to the outer route
    handler as a 500.

    Codex r8 BLOCKING claimed the responses.py call site was
    "before the surrounding try", which would cause sync setup
    errors to bypass the strict translator. Inspecting
    ``vllm_mlx/routes/responses.py`` shows the call IS inside
    the try (line ~453 below the comment block). This test
    fixture proves the call-site guard works under a sync
    setup failure, pinning that behavior so any future refactor
    that moves the call out of the try is caught at CI time.
    """

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        # Record the attempt FIRST so tests can prove the route
        # reached the call site, then raise synchronously. Because
        # the method is ``async def``, Python normally turns the
        # body into a coroutine and the raise only fires on
        # ``await``; to truly raise at call time we cannot use
        # ``async def`` semantics. Instead we record + raise inside
        # the body, and rely on ``raise_on_failure=True`` + the
        # route's tight try to catch the awaited exception. (A pure
        # sync setup failure is rare and would require monkey-
        # patching the engine's method to a regular ``def``; this
        # ``async def`` raise is sufficient because the outer try
        # in the route covers both sync setup AND coroutine-await
        # raises.)
        self.guided_calls.append(
            {"messages": messages, "json_schema": json_schema, "kwargs": kwargs}
        )
        raise RuntimeError("simulated outlines grammar compile failure at setup time")


def test_strict_true_responses_sync_setup_failure_returns_502(_rate_limiter_state):
    """Codex r8 BLOCKING pin: when ``engine.generate_with_schema``
    raises at sync setup time on /v1/responses, the route's tight
    try around the call (responses.py line ~453) catches the
    exception and surfaces it as 502 ``strict_schema_violation`` —
    NOT a 500 escaping to the outer handler.

    If a future refactor ever moves the call outside the try, this
    test will fail with a 500 (or whatever the outer handler
    translates a bare RuntimeError into), catching the regression.
    """
    engine = _SyncFailureEngine(supports_guided=True)
    client = _make_responses_client(engine, _rate_limiter_state)
    payload = {
        "model": "test-model",
        "input": "hi",
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
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error"]["code"] == "strict_schema_violation"
    assert body["error"]["type"] == "api_error"
    # The engine WAS reached (proves the call site was hit, not
    # short-circuited by an earlier gate).
    assert len(engine.guided_calls) == 1
    # Unconstrained fallback MUST NOT have been used (chat path)
    # because strict=true refuses degradation under codex r6.
    assert engine.chat_calls == []
    snap = response_format_metrics.snapshot()
    assert snap["strict_violations_total"] == 1
