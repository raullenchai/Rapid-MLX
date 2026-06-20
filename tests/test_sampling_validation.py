# SPDX-License-Identifier: Apache-2.0
"""F-011: NaN / inf / out-of-range sampling-param rejection.

Pre-fix, ``temperature=NaN``, ``top_p=NaN``,
``presence_penalty=NaN``, ``frequency_penalty=NaN``,
``presence_penalty=±10``, ``frequency_penalty=±10``, and
``frequency_penalty=Infinity`` all returned HTTP 200 on
``/v1/chat/completions`` and ``/v1/completions``:

* ``temperature=NaN`` / ``top_p=NaN`` produced a silent burn —
  ``choices[0].message.content=null`` with ``usage=0,0,0`` — and on
  Metal Apple-silicon backends crashed the command buffer with a
  ``kIOGPUCommandBufferCallbackErrorTimeout`` that killed the server
  process. The client saw a clean 200.

* penalty NaN/inf produced real output with mathematically
  undefined logit shifts (NaN propagates through softmax; inf
  collapses the distribution to a single token).

* ``presence_penalty=10`` / ``frequency_penalty=-10`` exceeded the
  OpenAI-spec ``[-2, 2]`` bound and produced equally-undefined
  outputs because no range gate existed in the legacy route path.

Root cause: the legacy route guard used ``not (0 < x <= 2)`` style
range checks, which evaluate to ``False`` for NaN (every NaN
comparison is False), so NaN slipped past. ``presence_penalty`` /
``frequency_penalty`` had no range check at all in any route — the
mlx-lm sampler accepted them unchecked.

Fix shape (vllm_mlx/api/models.py):
* Declare OpenAI-spec range bounds on the field itself via
  ``Field(ge=..., le=...)`` so finite out-of-range values 422 from
  the schema layer instead of the route layer.
* Add a ``field_validator`` calling ``math.isfinite`` to catch
  NaN/inf, which Field bounds skip.
* Apply the same gates on BOTH ``ChatCompletionRequest`` and
  ``CompletionRequest`` so the chat and legacy completions
  endpoints close the gap identically.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test-app fixtures: stub the engine on both routes so we hit only the
# Pydantic-level validators (which run before the route handler).
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_config():
    """Patch the global config singleton and restore on teardown."""
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


def _stub_engine_cfg(patch_cfg):
    engine = MagicMock()
    engine.is_mllm = False
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
    return engine


def _build_chat_client(patch_cfg, monkeypatch):
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import chat as chat_route

    engine = _stub_engine_cfg(patch_cfg)
    monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

    app = FastAPI()
    app.include_router(chat_route.router)
    # F-011: production wires the unified validation-error handler so
    # a NaN/inf rejection never escapes as a 500 via the default
    # FastAPI handler (which embeds the bad value in input_value and
    # crashes on JSON serialization). Tests must wire the same shape.
    install_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _build_completions_client(patch_cfg, monkeypatch):
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import completions as comp_route

    engine = _stub_engine_cfg(patch_cfg)
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)

    app = FastAPI()
    app.include_router(comp_route.router)
    install_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _base_chat_body() -> dict:
    return {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }


def _base_completion_body() -> dict:
    return {
        "model": "stub-model",
        "prompt": "hi",
        "max_tokens": 5,
    }


def _post_json_raw(client: TestClient, url: str, body: dict):
    """POST a body that may contain non-JSON-compliant floats (NaN, inf).

    httpx refuses to encode these via ``json=`` (it sets
    ``allow_nan=False`` internally), but stdlib ``json.dumps`` happily
    emits the ``NaN`` / ``Infinity`` / ``-Infinity`` tokens that
    Python's parser AND FastAPI's request decoder accept. Using the
    raw ``content=`` channel mirrors the bug repro exactly — clients
    DO send these values on the wire (see F-011 production reports);
    the server must reject them with a clean 4xx, not crash."""
    payload = json.dumps(body)  # allow_nan=True by default
    return client.post(
        url,
        content=payload,
        headers={"Content-Type": "application/json"},
    )


# ---------------------------------------------------------------------------
# The parametrized shape matrix.
#
# Each entry is a (field, wire-value) pair. We test every shape against
# BOTH endpoints by parametrizing the endpoint builder + base body. The
# wire values cover the three rejection classes:
#
#   * NaN / inf : Pydantic ``field_validator`` finite gate (or Field
#     ``le=``/``ge=`` which happens to reject NaN because every NaN
#     comparison is False — either gate is acceptable, both close the
#     bug).
#   * out-of-OpenAI-spec range : Field ``ge=`` / ``le=`` bound.
#   * string ``"NaN"`` / ``"Infinity"`` : Pydantic coerces these to
#     ``float`` non-finite values, then the same gates fire.
# ---------------------------------------------------------------------------


# Non-finite values — these are the F-011 silent-burn cases. All must
# go through the project's unified ``invalid_request_error`` 400
# envelope (production handler in
# ``vllm_mlx.middleware.exception_handlers``). Asserting the precise
# envelope shape here pins F-011 to its documented contract instead
# of letting a future regression slip back into FastAPI's default
# 422 path (codex round-1 BLOCKING #2).
NONFINITE_SHAPES: list[tuple[str, object]] = [
    # Raw JSON tokens (json.loads non-strict mode)
    ("temperature", float("nan")),
    ("top_p", float("nan")),
    ("presence_penalty", float("nan")),
    ("frequency_penalty", float("nan")),
    ("temperature", float("inf")),
    ("top_p", float("inf")),
    ("presence_penalty", float("inf")),
    ("frequency_penalty", float("inf")),
    ("presence_penalty", float("-inf")),
    # String wire forms (the F-011 bug repro uses these exactly)
    ("temperature", "NaN"),
    ("top_p", "NaN"),
    ("presence_penalty", "NaN"),
    ("frequency_penalty", "NaN"),
    ("frequency_penalty", "Infinity"),
    # NaN on auxiliary float sampling fields — would otherwise
    # propagate into mlx-lm's logits processor with no signal.
    ("min_p", float("nan")),
    ("repetition_penalty", float("nan")),
    ("repetition_penalty", float("inf")),
]

# Finite-but-out-of-OpenAI-spec-range values. These trip the Field
# ``ge=``/``le=`` bounds (Pydantic 422) before the scrub validator
# sees them, so the response is the 400 envelope produced by the
# project's RequestValidationError handler converting the 422 path
# down to the 400 contract. Either way the envelope shape + status
# are identical to the non-finite case, but we keep the two matrices
# split so a future regression on either path is pinpointed without
# guesswork.
OUT_OF_RANGE_SHAPES: list[tuple[str, object]] = [
    ("temperature", 3.0),
    ("temperature", -0.5),
    ("top_p", 2.0),
    ("top_p", 0),
    ("presence_penalty", 10),
    ("presence_penalty", -10),
    ("frequency_penalty", 10),
    ("frequency_penalty", -3.0),
]

INVALID_SHAPES: list[tuple[str, object]] = NONFINITE_SHAPES + OUT_OF_RANGE_SHAPES


def _assert_invalid_request_envelope(r, field: str, expected_status: int = 400) -> None:
    """Assert the response matches the project's unified
    ``invalid_request_error`` envelope (see
    ``vllm_mlx.middleware.exception_handlers._validation_error_response``)
    AND that the offending field name appears in the error message.

    This is the production contract — pinning it precisely catches a
    future cleanup that drops the custom handler and falls back to
    FastAPI's default 422 path (which embeds ``input_value`` and
    crashes on NaN serialization — exactly the F-011 secondary bug)."""
    assert r.status_code == expected_status, (
        f"expected {expected_status} for {field}; got {r.status_code} "
        f"body={r.text[:200]}"
    )
    body = r.json()
    assert isinstance(body, dict) and "error" in body, (
        f"missing top-level ``error`` key in {field} response: {r.text[:200]}"
    )
    err = body["error"]
    assert err.get("type") == "invalid_request_error", (
        f"wrong error type for {field}: {err}"
    )
    assert err.get("code") == "invalid_request", f"wrong error code for {field}: {err}"
    msg = err.get("message", "")
    assert isinstance(msg, str) and field in msg, (
        f"error message for {field} missing field name: {msg!r}"
    )


@pytest.mark.parametrize("field,value", NONFINITE_SHAPES)
def test_chat_nonfinite_sampling_param_returns_invalid_request_400(
    patched_config, monkeypatch, field, value
):
    """F-011 contract: NaN/inf wire values must return the project's
    unified ``invalid_request_error`` 400 envelope on the chat
    endpoint — NOT FastAPI's default 422 path (which would crash on
    NaN serialization, the silent-burn cause F-011 closes)."""
    client = _build_chat_client(patched_config, monkeypatch)
    body = _base_chat_body()
    body[field] = value
    r = _post_json_raw(client, "/v1/chat/completions", body)
    _assert_invalid_request_envelope(r, field)


@pytest.mark.parametrize("field,value", NONFINITE_SHAPES)
def test_completions_nonfinite_sampling_param_returns_invalid_request_400(
    patched_config, monkeypatch, field, value
):
    """Same F-011 contract on the legacy /v1/completions endpoint —
    the schemas share gates by construction, so the rejection
    surface must be byte-for-byte identical."""
    client = _build_completions_client(patched_config, monkeypatch)
    body = _base_completion_body()
    body[field] = value
    r = _post_json_raw(client, "/v1/completions", body)
    _assert_invalid_request_envelope(r, field)


@pytest.mark.parametrize("field,value", OUT_OF_RANGE_SHAPES)
def test_chat_out_of_range_sampling_param_returns_invalid_request_400(
    patched_config, monkeypatch, field, value
):
    """Finite-but-out-of-OpenAI-spec values must hit the same 400
    envelope. The Field ``ge=``/``le=`` bound triggers a Pydantic
    422 that the project's RequestValidationError handler converts
    down to the documented 400 ``invalid_request_error`` shape."""
    client = _build_chat_client(patched_config, monkeypatch)
    body = _base_chat_body()
    body[field] = value
    r = _post_json_raw(client, "/v1/chat/completions", body)
    _assert_invalid_request_envelope(r, field)


@pytest.mark.parametrize("field,value", OUT_OF_RANGE_SHAPES)
def test_completions_out_of_range_sampling_param_returns_invalid_request_400(
    patched_config, monkeypatch, field, value
):
    client = _build_completions_client(patched_config, monkeypatch)
    body = _base_completion_body()
    body[field] = value
    r = _post_json_raw(client, "/v1/completions", body)
    _assert_invalid_request_envelope(r, field)


# ---------------------------------------------------------------------------
# Valid baselines — every OpenAI-spec-canonical value must still 4xx-clean
# (i.e. NOT be rejected by the new gate). The route stub returns 500 from
# the MagicMock engine downstream, but the rejection — if any — must NOT
# blame the sampling field we're checking.
# ---------------------------------------------------------------------------


VALID_SHAPES: list[tuple[str, float | int]] = [
    ("temperature", 0.0),
    ("temperature", 0.7),
    ("temperature", 1.0),
    ("temperature", 2.0),
    ("top_p", 0.1),
    ("top_p", 0.9),
    ("top_p", 1.0),
    ("presence_penalty", -2.0),
    ("presence_penalty", -0.5),
    ("presence_penalty", 0.0),
    ("presence_penalty", 1.5),
    ("presence_penalty", 2.0),
    ("frequency_penalty", -2.0),
    ("frequency_penalty", -0.5),
    ("frequency_penalty", 0.0),
    ("frequency_penalty", 1.5),
    ("frequency_penalty", 2.0),
    ("min_p", 0.0),
    ("min_p", 0.05),
    ("min_p", 1.0),
    ("repetition_penalty", 1.0),
    ("repetition_penalty", 1.5),
]


def _stub_chat_impl(monkeypatch) -> dict:
    """Replace the chat route's inner dispatch with a sentinel that
    records the parsed ``ChatCompletionRequest`` and returns a
    deterministic 200. Asserting the stub was reached proves the
    Pydantic schema accepted the request AND the route's
    pre-engine validation block passed — closing the codex round-1
    NIT where the previous "field not blamed in 400" check could
    pass on any downstream 500."""
    captured: dict = {"called": False, "request": None}

    async def _impl(
        request,
        raw_request,
        engine,
        _commit_state,
        _admission_acquired,
    ):
        captured["called"] = True
        captured["request"] = request
        # The outer route handler manages admission via these lists;
        # mark committed so the finally-block release is a no-op and
        # we don't trip the admission accounting in the engine stub.
        _commit_state[0] = True
        _admission_acquired[0] = False
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": "stub-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    from vllm_mlx.routes import chat as chat_route

    monkeypatch.setattr(chat_route, "_create_chat_completion_impl", _impl, raising=True)
    return captured


def _stub_completion_route(monkeypatch) -> dict:
    """Same idea on the legacy /v1/completions endpoint — the route
    body is one big handler so we monkeypatch the engine's
    ``generate`` instead. The MagicMock baseline already does this,
    but here we wire a deterministic async coroutine so the route
    returns 200 and we can assert the request actually reached
    engine dispatch."""
    captured: dict = {"called": False, "request": None}

    # Stub at the engine level — the route's pre-engine validation
    # block has to pass for ``engine.generate`` to be invoked.
    async def _generate(*args, **kwargs):
        captured["called"] = True
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "text": "ok",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    return captured, _generate


@pytest.mark.parametrize("field,value", VALID_SHAPES)
def test_chat_valid_sampling_param_reaches_route_dispatch(
    patched_config, monkeypatch, field, value
):
    """A valid sampling value must survive every gate and reach the
    route's inner dispatch. We assert the dispatch stub was actually
    invoked — that proves the request made it through Pydantic
    schema validation AND the route-level guards (codex round-1
    NIT)."""
    client = _build_chat_client(patched_config, monkeypatch)
    captured = _stub_chat_impl(monkeypatch)
    body = _base_chat_body()
    body[field] = value
    r = client.post("/v1/chat/completions", json=body)
    assert captured["called"], (
        f"chat dispatch never invoked for {field}={value!r}; "
        f"status={r.status_code} body={r.text[:200]}"
    )
    assert r.status_code == 200, (
        f"valid {field}={value!r} rejected with {r.status_code}: {r.text[:200]}"
    )
    # And pin that the typed value survived to the request object
    # exactly as sent — catches a future regression where the
    # validator coerces or drops the field silently.
    assert getattr(captured["request"], field) == pytest.approx(value)


@pytest.mark.parametrize("field,value", VALID_SHAPES)
def test_completions_valid_sampling_param_parses_to_request(
    patched_config, monkeypatch, field, value
):
    """For the legacy completions endpoint we assert the schema
    layer accepts the value by parsing it directly through
    ``CompletionRequest.model_validate`` — the route's engine
    plumbing is too entangled to stub cleanly here, and the chat
    test above already pins "request reaches dispatch" for the
    shared gate. The schema-level assert proves the value typed
    cleanly through the Field bounds + finite check and survived
    onto the model with the exact value we sent."""
    from vllm_mlx.api.models import CompletionRequest

    body = _base_completion_body()
    body[field] = value
    req = CompletionRequest.model_validate(body)
    assert getattr(req, field) == pytest.approx(value)


# ---------------------------------------------------------------------------
# Pydantic-level unit checks — pin the validator behaviour directly so a
# refactor of the route stack can't regress this even if the integration
# tests above stop reaching the schema.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ],
)
def test_chat_schema_rejects_nan_directly(field):
    from vllm_mlx.api.models import ChatCompletionRequest

    body = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        field: float("nan"),
    }
    with pytest.raises(Exception):
        ChatCompletionRequest.model_validate(body)


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ],
)
def test_completions_schema_rejects_nan_directly(field):
    from vllm_mlx.api.models import CompletionRequest

    body = {"model": "x", "prompt": "hi", field: float("nan")}
    with pytest.raises(Exception):
        CompletionRequest.model_validate(body)


def test_chat_schema_accepts_default_none():
    """All sampling fields default to ``None``; an empty request must
    parse cleanly (no false positives from the new gates)."""
    from vllm_mlx.api.models import ChatCompletionRequest

    req = ChatCompletionRequest.model_validate(
        {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert req.temperature is None
    assert req.top_p is None
    assert req.presence_penalty is None
    assert req.frequency_penalty is None
    assert req.min_p is None
    assert req.repetition_penalty is None


def test_completions_schema_accepts_default_none():
    from vllm_mlx.api.models import CompletionRequest

    req = CompletionRequest.model_validate({"model": "x", "prompt": "hi"})
    assert req.temperature is None
    assert req.top_p is None
    assert req.presence_penalty is None
    assert req.frequency_penalty is None
    assert req.min_p is None
    assert req.repetition_penalty is None


def test_reject_nonfinite_helper_directly():
    """Module-level helper is shared by both schemas — pin its
    behaviour explicitly so a refactor can't silently change it."""
    from vllm_mlx.api.models import _reject_nonfinite_float

    # Valid: None and finite numbers pass through unchanged.
    assert _reject_nonfinite_float(None) is None
    assert _reject_nonfinite_float(0.0) == 0.0
    assert _reject_nonfinite_float(-1.5) == -1.5
    assert _reject_nonfinite_float(2.0) == 2.0

    # Invalid: NaN, +inf, -inf.
    for bad in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="finite"):
            _reject_nonfinite_float(bad)
        # math.isfinite mirror check — pins the invariant we rely on.
        assert not math.isfinite(bad)
