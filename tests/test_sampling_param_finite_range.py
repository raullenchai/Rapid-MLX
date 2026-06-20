# SPDX-License-Identifier: Apache-2.0
"""H-10: finite-range validation sweep on every sampling param.

Provenance: F-011 (#740) closed the NaN / inf / out-of-range gap on
``temperature`` + ``top_p`` and the ``presence_penalty`` /
``frequency_penalty`` Field bounds on the OpenAI routes. The H-10 bug
report showed the same hole was wide open on the other sampling
params: ``repetition_penalty=-1.0`` slipped past the schema, reached
``mlx_lm/sample_utils.py:298`` (``make_repetition_penalty`` raises on
``penalty < 0``), the un-caught ``ValueError`` escaped the request
scheduler, and uvicorn went down — port dead. Same silent-burn class
as F-011, just on a param F-011 didn't cover.

H-10's mandate: don't whack-a-mole the one repro. Sweep every
sampling param across every route, define the legal range, wire one
shared helper. This test pins the contract that emerged:

Sweep matrix per param × per route × per shape:

  Routes: /v1/chat/completions, /v1/completions, /v1/messages
  Shapes: NaN, +inf, -inf, below-min, above-max,
          boundary-low, boundary-high, valid-mid

Params covered (route gates each one with a Field bound + shared
``_validate_finite_in_range`` / ``_validate_nonnegative_int`` helper):

  * temperature
  * top_p
  * top_k
  * min_p              (chat + completions only)
  * repetition_penalty (chat + completions only — H-10 root bug)
  * presence_penalty   (chat + completions only)
  * frequency_penalty  (chat + completions only)
  * logit_bias values  (chat only — Anthropic surface has no
                       equivalent; legacy completions never declared it)

The test asserts every bad shape returns the project's unified
``invalid_request_error`` 400 envelope (production handler in
``vllm_mlx.middleware.exception_handlers``) AND that the server stays
alive across the full burst (no port death — the H-10 symptom this
PR closes).
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Per-test app fixtures — stub the engine so we hit only the Pydantic +
# route-layer validators, never the real sampler.
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


def _build_anthropic_client(patch_cfg, monkeypatch):
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import anthropic as anthropic_route

    engine = _stub_engine_cfg(patch_cfg)
    monkeypatch.setattr(
        anthropic_route, "get_engine", lambda *_a, **_kw: engine, raising=False
    )

    app = FastAPI()
    app.include_router(anthropic_route.router)
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


def _base_anthropic_body() -> dict:
    return {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }


def _post_json_raw(client: TestClient, url: str, body: dict):
    """POST a body that may contain non-JSON-compliant floats.

    Mirrors the F-011 test scaffolding — httpx's ``json=`` channel
    refuses to emit ``NaN`` / ``Infinity`` tokens (it sets
    ``allow_nan=False``), but stdlib ``json.dumps`` does so happily
    and FastAPI's body decoder accepts them. The H-10 repro path is
    exactly this wire form: clients sending real Python ``float('nan')``
    end up with these tokens on the wire."""
    payload = json.dumps(body)  # allow_nan=True by default
    return client.post(
        url,
        content=payload,
        headers={"Content-Type": "application/json"},
    )


# ---------------------------------------------------------------------------
# Legal ranges per route — one source of truth for the test matrix. The
# server-side validators MUST agree with these; mismatches surface as
# clearly-failing parametrized tests rather than as "tests look right
# but the contract drifted".
# ---------------------------------------------------------------------------


# Float sampling params: (field, min, max, min_inclusive, max_inclusive,
#                         routes-that-accept-it).
# ``None`` for min/max means "unbounded on that side".
FLOAT_PARAM_SPEC = [
    # OpenAI surfaces — chat + legacy completions share Field bounds.
    ("temperature", 0.0, 2.0, True, True, ("chat", "completions")),
    ("top_p", 0.0, 1.0, False, True, ("chat", "completions")),
    ("min_p", 0.0, 1.0, True, True, ("chat", "completions")),
    # H-10 ROOT: pre-fix this had no Field bound on the OpenAI routes
    # → ``repetition_penalty=-1.0`` slipped through to mlx-lm and
    # crashed uvicorn. Range matches mlx-lm's "non-negative" contract
    # + a safety upper cap (above 2 the distribution degenerates).
    ("repetition_penalty", 0.0, 2.0, True, True, ("chat", "completions")),
    ("presence_penalty", -2.0, 2.0, True, True, ("chat", "completions")),
    ("frequency_penalty", -2.0, 2.0, True, True, ("chat", "completions")),
    # Anthropic surfaces (per https://docs.anthropic.com/en/api/messages):
    # temperature is narrower than OpenAI ([0, 1] not [0, 2]).
    ("temperature_anthropic", 0.0, 1.0, True, True, ("anthropic",)),
    ("top_p_anthropic", 0.0, 1.0, False, True, ("anthropic",)),
]


INT_PARAM_SPEC = [
    # ``top_k`` is range-checked >= 0 on every route.
    # 0 == "disabled" per mlx-lm. Negative is rejected (M-14 also
    # noted the silent-ignore on the OpenAI chat path).
    ("top_k", ("chat", "completions", "anthropic")),
]


def _logical_to_wire(field: str) -> str:
    """Map ``"temperature_anthropic"`` → ``"temperature"`` on the wire.

    The logical names in FLOAT_PARAM_SPEC distinguish the
    OpenAI-spec'd ``temperature`` range ``[0, 2]`` from the
    Anthropic-spec'd ``[0, 1]`` range. The wire field is the same
    name in both cases."""
    return field.replace("_anthropic", "")


# ---------------------------------------------------------------------------
# Shape enumeration. Each shape labels its expected outcome (reject) so
# the parametrized test name reads naturally in pytest's verbose output.
# ---------------------------------------------------------------------------


def _bad_float_shapes(
    field: str,
    min_value: float | None,
    max_value: float | None,
    min_inclusive: bool,
    max_inclusive: bool,
):
    """Yield bad float wire values per the H-10 sweep matrix.

    Each value is paired with a short label used in the test id.
    """
    yield "nan", float("nan")
    yield "plus_inf", float("inf")
    yield "minus_inf", float("-inf")

    if min_value is not None:
        # below the min boundary
        yield "below_min", min_value - 0.5
        if not min_inclusive:
            # the boundary itself is illegal in the exclusive case
            yield "at_excl_min", min_value
    if max_value is not None:
        yield "above_max", max_value + 0.5
        if not max_inclusive:
            yield "at_excl_max", max_value


def _good_float_shapes(
    field: str,
    min_value: float | None,
    max_value: float | None,
    min_inclusive: bool,
    max_inclusive: bool,
):
    """Yield legal float values: boundary-low, boundary-high, valid-mid."""
    if min_value is not None and min_inclusive:
        yield "boundary_low", min_value
    elif min_value is not None and not min_inclusive:
        # tiny step past the exclusive bound — gives us a "boundary"
        # value to pin without tripping the exclusive gate
        yield "just_above_min", min_value + 1e-3
    if max_value is not None and max_inclusive:
        yield "boundary_high", max_value
    # Pick a midpoint that's well inside the range
    if min_value is not None and max_value is not None:
        mid = (min_value + max_value) / 2.0
        yield "valid_mid", mid
    elif min_value is not None:
        yield "valid_mid", min_value + 1.0
    elif max_value is not None:
        yield "valid_mid", max_value - 1.0


# ---------------------------------------------------------------------------
# Helpers to drive the right client per route name.
# ---------------------------------------------------------------------------


def _client_and_url(
    route: str, patched_config, monkeypatch
) -> tuple[TestClient, str, dict]:
    if route == "chat":
        return (
            _build_chat_client(patched_config, monkeypatch),
            "/v1/chat/completions",
            _base_chat_body(),
        )
    if route == "completions":
        return (
            _build_completions_client(patched_config, monkeypatch),
            "/v1/completions",
            _base_completion_body(),
        )
    if route == "anthropic":
        return (
            _build_anthropic_client(patched_config, monkeypatch),
            "/v1/messages",
            _base_anthropic_body(),
        )
    raise AssertionError(f"unknown route {route!r}")


def _assert_invalid_request_envelope(r, wire_field: str) -> None:
    """The 400 ``invalid_request_error`` envelope plus the wire field
    name in the error message.

    Envelope shape consistency across the three routes is M-10's
    concern, not H-10's: the OpenAI surfaces route Pydantic
    ``RequestValidationError`` through
    ``_validation_error_response`` (``code="invalid_request"``),
    while the Anthropic route catches ``ValidationError`` inside
    the handler and re-raises ``HTTPException(400, str(e))`` which
    ``_http_error_response`` renders with ``code=None``. The H-10
    contract is "400 + invalid_request_error + field-name-in-message"
    — that gate IS the systemic survival proof regardless of which
    handler shaped the body.

    The 400 status matters as much as the body: a 422 would mean
    the request escaped to FastAPI's default validation handler,
    which embeds the bad value in ``input_value`` and crashes on
    NaN serialization — that's the F-011 silent-500 cause this
    PR closes for every sampling param at once."""
    assert r.status_code == 400, (
        f"expected 400 for {wire_field}; got {r.status_code} body={r.text[:200]}"
    )
    body = r.json()
    assert isinstance(body, dict) and "error" in body, (
        f"missing top-level ``error`` key for {wire_field}: {r.text[:200]}"
    )
    err = body["error"]
    assert err.get("type") == "invalid_request_error", (
        f"wrong error type for {wire_field}: {err}"
    )
    # Code is either ``"invalid_request"`` (OpenAI route validation
    # handler) or ``None`` (Anthropic route's catch-and-rethrow path).
    # Tighten this when M-10 unifies envelopes across routes.
    assert err.get("code") in ("invalid_request", None), (
        f"unexpected code for {wire_field}: {err}"
    )
    msg = err.get("message", "")
    assert isinstance(msg, str) and wire_field in msg, (
        f"error message for {wire_field} missing field name: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Bad-shape sweep — every (param × shape × route) combo must 4xx-clean.
# ---------------------------------------------------------------------------


def _bad_float_matrix():
    """Build the full (field, shape_label, value, route) sweep."""
    cases: list[tuple[str, str, str, object, str]] = []
    for (
        logical_field,
        min_v,
        max_v,
        min_incl,
        max_incl,
        routes,
    ) in FLOAT_PARAM_SPEC:
        wire_field = _logical_to_wire(logical_field)
        for label, value in _bad_float_shapes(
            logical_field, min_v, max_v, min_incl, max_incl
        ):
            for route in routes:
                cases.append((logical_field, wire_field, label, value, route))
    return cases


def _good_float_matrix():
    cases: list[tuple[str, str, str, float, str]] = []
    for (
        logical_field,
        min_v,
        max_v,
        min_incl,
        max_incl,
        routes,
    ) in FLOAT_PARAM_SPEC:
        wire_field = _logical_to_wire(logical_field)
        for label, value in _good_float_shapes(
            logical_field, min_v, max_v, min_incl, max_incl
        ):
            for route in routes:
                cases.append((logical_field, wire_field, label, value, route))
    return cases


BAD_FLOAT_CASES = _bad_float_matrix()
GOOD_FLOAT_CASES = _good_float_matrix()


@pytest.mark.parametrize(
    "logical_field,wire_field,shape_label,value,route",
    BAD_FLOAT_CASES,
    ids=lambda v: str(v)[:24],
)
def test_bad_float_sampling_param_rejected(
    patched_config,
    monkeypatch,
    logical_field,
    wire_field,
    shape_label,
    value,
    route,
):
    """Every illegal float wire shape on every sampling param on every
    route must return the unified ``invalid_request_error`` 400
    envelope. This is the systemic H-10 contract — no param-specific
    whack-a-mole branches in the validators, one shared
    ``_validate_finite_in_range`` helper does the work."""
    client, url, body = _client_and_url(route, patched_config, monkeypatch)
    body[wire_field] = value
    r = _post_json_raw(client, url, body)
    _assert_invalid_request_envelope(r, wire_field)


@pytest.mark.parametrize(
    "logical_field,wire_field,shape_label,value,route",
    GOOD_FLOAT_CASES,
    ids=lambda v: str(v)[:24],
)
def test_good_float_sampling_param_accepted_by_schema(
    logical_field,
    wire_field,
    shape_label,
    value,
    route,
):
    """All boundary/mid-range legal values parse cleanly through the
    schema. We assert at the Pydantic layer directly (not the full
    route) — the route's downstream engine plumbing is irrelevant to
    the H-10 contract, and the F-011 ``test_chat_valid_sampling_param_reaches_route_dispatch``
    already pins ``valid → route dispatch reached`` for the chat
    surface."""
    from vllm_mlx.api.anthropic_models import AnthropicRequest
    from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest

    if route == "chat":
        body = _base_chat_body()
        body[wire_field] = value
        req = ChatCompletionRequest.model_validate(body)
    elif route == "completions":
        body = _base_completion_body()
        body[wire_field] = value
        req = CompletionRequest.model_validate(body)
    elif route == "anthropic":
        body = _base_anthropic_body()
        body[wire_field] = value
        req = AnthropicRequest.model_validate(body)
    else:  # pragma: no cover — guarded by the parametrize fixture
        raise AssertionError(f"unknown route {route!r}")

    assert getattr(req, wire_field) == pytest.approx(value)


# ---------------------------------------------------------------------------
# Integer sampling params — currently only ``top_k``. Mirrors the float
# matrix shape so adding e.g. ``best_of`` later is one entry, not a
# new test file.
# ---------------------------------------------------------------------------


BAD_INT_SHAPES = [
    ("negative_small", -1),
    ("negative_large", -1000),
    # 0 is legal on top_k (means "disabled") so it is NOT in this list.
]
GOOD_INT_SHAPES = [
    ("zero_disabled", 0),
    ("typical", 40),
    ("large", 1000),
]


@pytest.mark.parametrize(
    "field,routes",
    INT_PARAM_SPEC,
)
@pytest.mark.parametrize("shape_label,value", BAD_INT_SHAPES)
def test_bad_int_sampling_param_rejected(
    patched_config, monkeypatch, field, routes, shape_label, value
):
    for route in routes:
        client, url, body = _client_and_url(route, patched_config, monkeypatch)
        body[field] = value
        r = _post_json_raw(client, url, body)
        _assert_invalid_request_envelope(r, field)


@pytest.mark.parametrize("field,routes", INT_PARAM_SPEC)
@pytest.mark.parametrize("shape_label,value", GOOD_INT_SHAPES)
def test_good_int_sampling_param_accepted(field, routes, shape_label, value):
    from vllm_mlx.api.anthropic_models import AnthropicRequest
    from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest

    for route in routes:
        if route == "chat":
            body = _base_chat_body()
            body[field] = value
            req = ChatCompletionRequest.model_validate(body)
        elif route == "completions":
            body = _base_completion_body()
            body[field] = value
            req = CompletionRequest.model_validate(body)
        elif route == "anthropic":
            body = _base_anthropic_body()
            body[field] = value
            req = AnthropicRequest.model_validate(body)
        else:  # pragma: no cover
            raise AssertionError(f"unknown route {route!r}")
        assert getattr(req, field) == value


# ---------------------------------------------------------------------------
# logit_bias finite-value sweep (chat only — Anthropic doesn't accept it;
# the legacy completions schema never declared it).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        {"42": float("nan")},
        {"42": float("inf")},
        {"42": float("-inf")},
        {"hello": float("nan")},
    ],
    ids=lambda v: str(v)[:30],
)
def test_logit_bias_nonfinite_value_rejected(patched_config, monkeypatch, value):
    """A defensively crafted ``logit_bias = {"42": NaN}`` previously
    survived the Pydantic parse — the route's existing "non-empty
    logit_bias not supported" 400 caught the case in practice, but
    if a downstream PR ever wires the field through to a real
    logits processor the NaN payload would land in the Metal kernel.
    H-10's defensive schema-layer check closes the gap up front."""
    client = _build_chat_client(patched_config, monkeypatch)
    body = _base_chat_body()
    body["logit_bias"] = value
    r = _post_json_raw(client, "/v1/chat/completions", body)
    _assert_invalid_request_envelope(r, "logit_bias")


def test_logit_bias_empty_accepted():
    """The OpenAI spec accepts ``logit_bias: {}`` as "no bias" — must
    still parse cleanly post-fix."""
    from vllm_mlx.api.models import ChatCompletionRequest

    req = ChatCompletionRequest.model_validate(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": {},
        }
    )
    assert req.logit_bias == {}


# ---------------------------------------------------------------------------
# H-10 repro — exact wire payload from the bug report. This is the
# canary: if it ever passes pre-validation again the server will die.
# ---------------------------------------------------------------------------


def test_h10_repro_repetition_penalty_negative_rejected(patched_config, monkeypatch):
    """The exact H-10 production repro:
    ``chat.completions.create(..., extra_body={"repetition_penalty": -1.0})``
    must surface as a clean 400 (not the silent 500 + uvicorn death
    the pre-fix path produced)."""
    client = _build_chat_client(patched_config, monkeypatch)
    body = _base_chat_body()
    body["repetition_penalty"] = -1.0
    r = _post_json_raw(client, "/v1/chat/completions", body)
    _assert_invalid_request_envelope(r, "repetition_penalty")


def test_h10_repro_repetition_penalty_negative_rejected_on_completions(
    patched_config, monkeypatch
):
    """Same repro on the legacy completions surface — both OpenAI
    routes share the schema by construction, so the gate must close
    both surfaces simultaneously."""
    client = _build_completions_client(patched_config, monkeypatch)
    body = _base_completion_body()
    body["repetition_penalty"] = -1.0
    r = _post_json_raw(client, "/v1/completions", body)
    _assert_invalid_request_envelope(r, "repetition_penalty")


# ---------------------------------------------------------------------------
# Server survival across a 50-shot burst — the H-10 symptom was uvicorn
# DEATH (port goes dead), not "wrong error code". Even if the envelope
# regressed silently, the burst would catch the un-caught-exception
# path because TestClient would surface ``ConnectionError`` after the
# first crash. ``raise_server_exceptions=False`` keeps subsequent
# requests flowing so the burst really does exercise N independent
# bad shapes in sequence.
# ---------------------------------------------------------------------------


def _burst_bad_payloads() -> list[tuple[str, str, dict]]:
    """Build a 50+ bad-payload sequence drawn from the full matrix."""
    seq: list[tuple[str, str, dict]] = []
    for logical_field, wire_field, _label, value, route in BAD_FLOAT_CASES[:50]:
        if route == "chat":
            body = _base_chat_body()
        elif route == "completions":
            body = _base_completion_body()
        elif route == "anthropic":
            body = _base_anthropic_body()
        else:  # pragma: no cover
            continue
        body[wire_field] = value
        seq.append((route, wire_field, body))
    return seq


def test_server_survives_50_bad_payloads_back_to_back(patched_config, monkeypatch):
    """H-10 root symptom: a single ``repetition_penalty=-1.0`` killed
    uvicorn (port dead). Even after the schema gate lands, a regression
    in the error path could re-open the silent-burn surface. Smashing
    the server with 50 different bad shapes in sequence — and asserting
    each one returns a 400 envelope WITHOUT the TestClient dropping the
    connection — is the systemic shape-survival proof."""
    chat = _build_chat_client(patched_config, monkeypatch)
    comp = _build_completions_client(patched_config, monkeypatch)
    anth = _build_anthropic_client(patched_config, monkeypatch)
    by_route = {
        "chat": (chat, "/v1/chat/completions"),
        "completions": (comp, "/v1/completions"),
        "anthropic": (anth, "/v1/messages"),
    }

    payloads = _burst_bad_payloads()
    assert len(payloads) >= 50, (
        f"sweep matrix shrunk below the 50-shot floor "
        f"(now {len(payloads)}) — the burst test loses its teeth."
    )

    for i, (route, wire_field, body) in enumerate(payloads, start=1):
        client, url = by_route[route]
        r = _post_json_raw(client, url, body)
        # We deliberately use ``status_code`` rather than the full
        # envelope assert: the burst is about SURVIVAL, not envelope
        # shape (which is pinned by the parametric tests above).
        assert r.status_code == 400, (
            f"survival burst failed at {i}/{len(payloads)}: "
            f"{route} {wire_field} got {r.status_code} body={r.text[:120]}"
        )

    # One last bad-shape probe AFTER the 50-shot burst to prove the
    # validator pipeline is still serving. If the un-caught-exception
    # path had taken uvicorn down on the real server the TestClient
    # would surface ``ConnectionError`` (or this final assertion would
    # fail). We deliberately send a malformed payload (not a valid
    # one) so we don't dip into the stubbed engine — the engine is
    # ``MagicMock`` and not coroutine-aware, which would fail the
    # downstream ``asyncio.ensure_future`` path with a TypeError that
    # has nothing to do with H-10. The 400 envelope is the proof.
    final = _post_json_raw(
        chat,
        "/v1/chat/completions",
        {**_base_chat_body(), "repetition_penalty": -2.0},
    )
    assert final.status_code == 400, (
        f"server stopped honoring requests after burst — "
        f"escape-the-scheduler regression? "
        f"status={final.status_code} body={final.text[:200]}"
    )


# ---------------------------------------------------------------------------
# Direct unit checks on the shared helpers — pin the helper contract so a
# refactor that drops a callsite can't silently regress.
# ---------------------------------------------------------------------------


def test_validate_finite_in_range_passes_through_none():
    from vllm_mlx.api.models import _validate_finite_in_range

    assert _validate_finite_in_range(None, min_value=0, max_value=1) is None


def test_validate_finite_in_range_rejects_nan_inf():
    from vllm_mlx.api.models import _validate_finite_in_range

    for bad in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="finite"):
            _validate_finite_in_range(
                bad, min_value=0.0, max_value=2.0, field_name="temperature"
            )
        assert not math.isfinite(bad)


def test_validate_finite_in_range_enforces_inclusive_bounds():
    from vllm_mlx.api.models import _validate_finite_in_range

    # Inclusive default: boundaries OK
    assert _validate_finite_in_range(0.0, min_value=0.0, max_value=2.0) == 0.0
    assert _validate_finite_in_range(2.0, min_value=0.0, max_value=2.0) == 2.0
    # Out of range
    with pytest.raises(ValueError, match=">="):
        _validate_finite_in_range(-0.1, min_value=0.0, max_value=2.0)
    with pytest.raises(ValueError, match="<="):
        _validate_finite_in_range(2.1, min_value=0.0, max_value=2.0)


def test_validate_finite_in_range_enforces_exclusive_min():
    from vllm_mlx.api.models import _validate_finite_in_range

    # exclusive min: 0.0 must be rejected
    with pytest.raises(ValueError, match=">"):
        _validate_finite_in_range(
            0.0, min_value=0.0, max_value=1.0, min_inclusive=False
        )
    # but a tiny step is fine
    assert (
        _validate_finite_in_range(
            0.001, min_value=0.0, max_value=1.0, min_inclusive=False
        )
        == 0.001
    )


def test_validate_nonnegative_int_rejects_bools():
    from vllm_mlx.api.models import _validate_nonnegative_int

    for bad in (True, False):
        with pytest.raises(ValueError, match="bool"):
            _validate_nonnegative_int(bad, field_name="top_k")


def test_validate_nonnegative_int_accepts_integer_valued_float():
    """Pydantic v2 lax coercion turns ``top_k: 64.0`` into ``top_k=64``.
    We mirror that contract so this H-10 gate is purely additive (only
    rejects shapes the legacy path also rejected, plus the
    specifically-H-10 bad shapes). Anything else is a regression risk."""
    from vllm_mlx.api.models import _validate_nonnegative_int

    out = _validate_nonnegative_int(64.0, field_name="top_k")
    assert out == 64 and isinstance(out, int)


def test_validate_nonnegative_int_rejects_nonint_float():
    from vllm_mlx.api.models import _validate_nonnegative_int

    with pytest.raises(ValueError, match="integer"):
        _validate_nonnegative_int(64.5, field_name="top_k")


def test_validate_nonnegative_int_rejects_negative():
    from vllm_mlx.api.models import _validate_nonnegative_int

    with pytest.raises(ValueError, match=">= 0"):
        _validate_nonnegative_int(-1, field_name="top_k")


def test_validate_logit_bias_finite_passes_through_none_and_empty():
    from vllm_mlx.api.models import _validate_logit_bias_finite

    assert _validate_logit_bias_finite(None) is None
    assert _validate_logit_bias_finite({}) == {}


def test_validate_logit_bias_finite_rejects_nan_value():
    from vllm_mlx.api.models import _validate_logit_bias_finite

    with pytest.raises(ValueError, match="finite"):
        _validate_logit_bias_finite({"42": float("nan")})


def test_validate_logit_bias_finite_rejects_inf_value():
    from vllm_mlx.api.models import _validate_logit_bias_finite

    with pytest.raises(ValueError, match="finite"):
        _validate_logit_bias_finite({"42": float("inf")})


def test_validate_logit_bias_finite_rejects_bool_value():
    """``logit_bias`` values must be numeric; ``True`` is an int subclass
    but a wire-bool value here is almost certainly a serialization
    bug, not "bias of +1.0". H-10's logit_bias gate is defensive
    only — but defensively rejecting bool keeps the surface tight
    and matches the convention on ``_reject_non_one_n`` /
    ``_validate_nonnegative_int``."""
    from vllm_mlx.api.models import _validate_logit_bias_finite

    with pytest.raises(ValueError, match="bool"):
        _validate_logit_bias_finite({"42": True})
