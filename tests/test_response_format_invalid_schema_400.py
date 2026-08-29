# SPDX-License-Identifier: Apache-2.0
"""0.10.16 dogfood P1-③ — an invalid ``response_format`` schema must NOT
silently degrade to unconstrained generation with an HTTP 200.

Repro (pre-fix): a request with
``response_format={"type":"json_schema","json_schema":{"name":"x",
"schema":{"type":"notatype"}}}`` returned **HTTP 200 with non-conforming
free-form text**. The server logged the llguidance compile error
(``Invalid type: notatype``) and then ``WARNING: Guided generation failed,
falling back to regular generation`` — but the caller received ZERO signal
that the constraint had been dropped. That is worse than a 400: the client
believes its output is schema-constrained when it is not.

Root cause: ``GuidedGenerator._decode_constrained`` returned ``None`` on a
grammar-compile error, indistinguishable from the benign
guided-unavailable / truncated-parse ``None``. ``BatchedEngine.
generate_with_schema`` swallowed that ``None`` into a silent
``self.chat(...)`` fallback (HTTP 200, unconstrained).

Fix: structural validity of a caller schema is settled ONCE, at the ROUTE
BOUNDARY — a single shared validator (``nonstrict_json_schema_boundary_error``
for non-strict json_schema; the strict ``check_schema_validity`` pre-flight for
strict) returns a clean HTTP 400 BEFORE any engine dispatch, on chat and
responses alike, streaming or not. ``/v1/completions`` rejects json_schema
up-front as unsupported. Downstream, the guided layer treats EVERY llguidance
failure as OPERATIONAL — it degrades to ``None`` (→ strict 502, non-strict
best-effort 200), never a 400 — so there is no schema-specific propagation path
to a handler anymore (the vestigial engine/route/handler machinery for that was
removed once the boundary became the single structural-validation point).

These tests pin the contract with a mock engine (no model / llguidance needed):
an invalid schema is rejected at the boundary (engine never dispatched); a valid
schema that fails OPERATIONALLY (``RuntimeError``) is a strict 502 / non-strict
best-effort 200. A few tests exercise the guided layer directly; llguidance is a
CORE dependency (promoted out of the ``[guided]`` extra in 0.10.15), so they run
UNCONDITIONALLY and fail loudly if it is unavailable rather than silently skip.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_mlx
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import from the DEPENDENCY-FREE errors module — and assert it is the SAME
# class the guided module re-exports, so the whole chain agrees on identity.
from vllm_mlx.api.errors import (
    CHAT_RESPONSE_FORMAT_PARAM,
    RESPONSES_TEXT_FORMAT_PARAM,
    GuidedSchemaCompileError,
    guided_schema_compile_error_detail,
)
from vllm_mlx.api.guided import GuidedSchemaCompileError as _GuidedErrFromGuided
from vllm_mlx.config import reset_config
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.middleware.exception_handlers import install_exception_handlers
from vllm_mlx.routes.chat import router as chat_router

assert _GuidedErrFromGuided is GuidedSchemaCompileError  # re-export identity

_VALID_PAYLOAD = '{"answer": 42}'
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_INVALID_SCHEMA = {"type": "notatype"}
_COMPILE_ERR = GuidedSchemaCompileError("Invalid type: notatype")


class _Engine:
    """Mock engine.

    ``guided_raises`` selects what ``generate_with_schema`` does:
      * ``None``                    → return the fixed valid payload,
      * a ``GuidedSchemaCompileError`` → simulate an invalid caller schema,
      * any other exception         → simulate a transient guided failure.
    """

    preserve_native_tool_format = False
    is_mllm = False
    tokenizer = None

    def __init__(
        self,
        *,
        supports_guided: bool = True,
        guided_text: str = _VALID_PAYLOAD,
        chat_text: str = "FALLBACK unconstrained text",
        guided_raises: Exception | None = None,
    ):
        self.supports_guided_generation = supports_guided
        self._guided_text = guided_text
        self._chat_text = chat_text
        self._guided_raises = guided_raises
        self.guided_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def generate_with_schema(self, *, messages, json_schema, **kwargs):
        self.guided_calls.append({"json_schema": json_schema, "kwargs": kwargs})
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
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app)


def _response_format(schema: dict, *, strict: bool | None = None) -> dict:
    js: dict = {"name": "x", "schema": schema}
    if strict is not None:
        js["strict"] = strict
    return {"type": "json_schema", "json_schema": js}


def _parse_sse_events(text: str) -> tuple[list[dict], bool]:
    events: list[dict] = []
    saw_done = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events, saw_done


def _assert_invalid_schema_400(resp) -> None:
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert err["param"] == "response_format.json_schema.schema"
    assert "is not a valid JSON schema" in err["message"]
    assert "notatype" in err["message"]


# ---------------------------------------------------------------------------
# Dependency-free errors module (Round-3 NIT #4): the envelope builder lives in
# ``vllm_mlx.api.errors`` with NO mlx/llguidance import, so the exception
# handlers and routes can build the 400 body without triggering engine init.
# ---------------------------------------------------------------------------


def test_errors_module_is_dependency_free():
    """``vllm_mlx.api.errors`` must not drag in the heavy engine stack — that
    is the whole point of splitting it out of ``api.guided`` (which imports
    mlx/llguidance). If either becomes importable-through-errors, importing the
    module for the envelope builder would boot the engine on every handler.

    Runs in a FRESH subprocess: an in-process ``importlib.reload`` would mint a
    NEW ``GuidedSchemaCompileError`` class object and poison the ``isinstance``
    identity the rest of this suite (and the live handlers) depend on. A clean
    interpreter both avoids that and proves the import graph from a cold start,
    which is what a real handler process sees.
    """
    import subprocess
    import sys

    code = (
        "import sys; import vllm_mlx.api.errors as e;"
        "assert hasattr(e, 'GuidedSchemaCompileError');"
        "assert hasattr(e, 'guided_schema_compile_error_detail');"
        "assert 'mlx.core' not in sys.modules, sorted(m for m in sys.modules "
        "if m.startswith('mlx'));"
        "assert 'llguidance' not in sys.modules;"
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK", proc.stdout


def test_guided_compile_error_detail_envelope_and_param_override():
    """The envelope builder produces the canonical OpenAI-style body and honors
    the ``param`` override the ``/v1/responses`` route uses
    (``text.format.schema`` vs the chat default)."""
    exc = GuidedSchemaCompileError("Invalid type: notatype")

    default = guided_schema_compile_error_detail(exc)["error"]
    assert default["type"] == "invalid_request_error"
    assert default["code"] == "invalid_response_format_schema"
    assert default["param"] == "response_format.json_schema.schema"
    assert "is not a valid JSON schema" in default["message"]
    assert "notatype" in default["message"]

    overridden = guided_schema_compile_error_detail(exc, param="text.format.schema")[
        "error"
    ]
    assert overridden["param"] == "text.format.schema"


def test_nonstrict_json_schema_boundary_error_helper():
    """Round-5 centralization — the SINGLE shared route-boundary structural
    validator returns the 400 detail for a structurally-invalid NON-strict
    json_schema, ``None`` for valid / strict / non-json_schema, and honors the
    per-surface ``param``. Both chat and responses call THIS one function."""
    from vllm_mlx.api.tool_calling import nonstrict_json_schema_boundary_error

    rf_invalid = {
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": _INVALID_SCHEMA},
    }
    err = nonstrict_json_schema_boundary_error(rf_invalid, CHAT_RESPONSE_FORMAT_PARAM)
    assert err is not None
    assert err["error"]["code"] == "invalid_response_format_schema"
    assert err["error"]["param"] == CHAT_RESPONSE_FORMAT_PARAM
    assert "notatype" in err["error"]["message"]

    # Per-surface param override (responses).
    err_resp = nonstrict_json_schema_boundary_error(
        rf_invalid, RESPONSES_TEXT_FORMAT_PARAM
    )
    assert err_resp["error"]["param"] == RESPONSES_TEXT_FORMAT_PARAM

    # Valid non-strict schema → None.
    rf_valid = {"type": "json_schema", "json_schema": {"name": "x", "schema": _SCHEMA}}
    assert (
        nonstrict_json_schema_boundary_error(rf_valid, CHAT_RESPONSE_FORMAT_PARAM)
        is None
    )

    # STRICT is skipped here (owned by the more specific invalid_strict_schema
    # pre-flight) even when the schema is invalid.
    rf_strict = {
        "type": "json_schema",
        "json_schema": {"name": "x", "strict": True, "schema": _INVALID_SCHEMA},
    }
    assert (
        nonstrict_json_schema_boundary_error(rf_strict, CHAT_RESPONSE_FORMAT_PARAM)
        is None
    )

    # Nothing to validate → None.
    assert (
        nonstrict_json_schema_boundary_error(None, CHAT_RESPONSE_FORMAT_PARAM) is None
    )
    assert (
        nonstrict_json_schema_boundary_error(
            {"type": "json_object"}, CHAT_RESPONSE_FORMAT_PARAM
        )
        is None
    )


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def test_sync_invalid_schema_returns_400_not_silent_200():
    """The core P1-③ repro: a NON-strict invalid schema → 400 at the ROUTE
    BOUNDARY, NOT a silent 200 with unconstrained text. The route rejects the
    malformed schema BEFORE any engine dispatch, so neither the guided path nor
    the unconstrained ``chat`` fallback runs."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA),
        },
    )
    _assert_invalid_schema_400(resp)
    assert engine.guided_calls == [], (
        "route-boundary validation must reject before engine dispatch"
    )
    assert engine.chat_calls == [], (
        "invalid schema must NOT silently fall back to unconstrained chat"
    )


def test_sync_invalid_schema_unsupported_guided_engine_returns_400_not_200():
    """Round-4 #1 regression — the exact hole the route-boundary check closes:
    an engine that CANNOT guide (``supports_guided_generation=False``) never
    calls ``generate_json`` on a non-strict request, so pre-fix an invalid
    schema slipped through to unconstrained HTTP 200. The boundary check now
    returns 400 regardless of engine capability."""
    engine = _Engine(supports_guided=False)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA),
        },
    )
    _assert_invalid_schema_400(resp)
    assert engine.guided_calls == []
    assert engine.chat_calls == [], (
        "guided-unsupported engine must STILL 400 an invalid schema, not fall "
        "back to unconstrained generation (the P1-③ silent-degrade)"
    )


def test_sync_invalid_schema_strict_returns_400():
    """Under ``strict=true`` a structurally-INVALID schema is rejected 400 by
    the strict pre-flight (``invalid_strict_schema``), before any generation —
    the schema itself is the fault. (Only a validator-INVALID schema is a 400;
    the validator-valid-but-compiler-rejected case is the 502 path — see
    ``test_sync_valid_schema_strict_operational_failure_returns_502``.)"""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA, strict=True),
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_strict_schema"
    assert engine.guided_calls == []
    assert engine.chat_calls == []


def test_sync_valid_schema_strict_operational_failure_returns_502():
    """Corrected strict contract (codex Round-4 #3): a validator-VALID schema
    that the guided path fails to honor OPERATIONALLY (llguidance reject /
    runtime failure → ``None`` → ``raise_on_failure``) is the strict-failure
    path — a sanitized 502 ``strict_schema_violation`` (server could not honor
    the constraint), NOT a 400 (the schema is not the fault) and NOT a silent
    unconstrained 200. This is the case a previous test WRONGLY asserted as a
    400 by fabricating a compile error from a valid schema; production returns
    None → 502 here."""
    engine = _Engine(guided_raises=RuntimeError("operational llguidance failure"))
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA, strict=True),
        },
    )
    assert resp.status_code == 502, resp.text
    err = resp.json()["error"]
    assert err["code"] == "strict_schema_violation"
    assert engine.chat_calls == [], (
        "strict mode must refuse the unconstrained fallback on an operational "
        "guided failure"
    )


def test_sync_valid_schema_still_returns_200():
    """Regression guard: a VALID schema must still return 200 with the
    constrained content — the fix must not turn valid schemas into 400s."""
    engine = _Engine(guided_text=_VALID_PAYLOAD)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == _VALID_PAYLOAD
    assert len(engine.guided_calls) == 1
    assert engine.chat_calls == []


def test_sync_plain_chat_without_response_format_still_works():
    """A plain request with no ``response_format`` must be unaffected — it
    routes through ``chat`` and returns 200, never touching guided."""
    engine = _Engine()
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert engine.guided_calls == []
    assert len(engine.chat_calls) == 1


def test_sync_generic_guided_failure_still_falls_back_non_strict():
    """A GENERIC (non-compile) guided failure under ``strict=false`` must
    PRESERVE today's best-effort fallback to unconstrained ``chat`` (200).
    Only a ``GuidedSchemaCompileError`` (invalid schema) is a hard 400 —
    this pins that the fix narrowly special-cases compile errors and does
    not turn every transient guided hiccup into a 400."""
    engine = _Engine(guided_raises=RuntimeError("transient llguidance blip"))
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_SCHEMA, strict=False),
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(engine.guided_calls) == 1
    assert len(engine.chat_calls) == 1, (
        "a transient (non-compile) guided failure must keep its best-effort "
        "unconstrained fallback"
    )


# ---------------------------------------------------------------------------
# Streaming. The route-boundary schema validation runs BEFORE the stream/non-
# stream split, so an invalid schema is rejected as a clean HTTP 400 BEFORE the
# SSE response opens — better than a 200-status SSE error envelope, and it can
# never silently fall back to unconstrained streaming.
# ---------------------------------------------------------------------------


def test_stream_invalid_schema_returns_400_at_boundary():
    """Streaming, ``strict=false``: an invalid schema is rejected 400 at the
    route boundary, BEFORE the SSE stream opens — never a silent unconstrained
    stream. The engine's streaming helper is never reached."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA, strict=False),
        },
    )
    _assert_invalid_schema_400(resp)
    assert engine.stream_calls == [], (
        "invalid schema must NOT reach the unconstrained streaming helper"
    )
    assert engine.guided_calls == []


def test_stream_invalid_schema_strict_returns_400_at_boundary():
    """Streaming, ``strict=true``: an invalid schema is rejected 400
    (``invalid_strict_schema``) by the strict pre-flight, BEFORE the SSE stream
    opens. Only a validator-INVALID schema is a 400; validator-valid +
    operational failure is the 502 path (non-streaming strict)."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_client(engine)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": _response_format(_INVALID_SCHEMA, strict=True),
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_strict_schema"
    assert engine.stream_calls == []
    assert engine.guided_calls == []


# ---------------------------------------------------------------------------
# Guided layer. llguidance is a CORE dependency (promoted out of the [guided]
# extra in 0.10.15), so ``is_guided_available()`` is True in CI and these run.
# They depend on NO developer-local HF tokenizer cache — every llguidance
# interaction is stubbed deterministically, so a regression (e.g. reverting
# _decode_constrained's raise to ``return None``) fails the suite in clean CI
# rather than silently skipping.
# ---------------------------------------------------------------------------


class _StubMatcher:
    """Minimal stand-in for ``llguidance.mlx.LLMatcher``. ``get_error()``
    returns the configured string, driving ``_decode_constrained``'s
    grammar-rejection branch without any real grammar/tokenizer/model."""

    def __init__(self, error: str):
        self._error = error

    def get_error(self) -> str:
        return self._error


def test_decode_constrained_raises_on_matcher_get_error(monkeypatch):
    """Round-4 #2 — DETERMINISTIC coverage of the load-bearing
    ``matcher.get_error()`` raise, with NO HF-cache dependency.

    ``_decode_constrained`` builds an ``LLMatcher`` and, when ``get_error()``
    is non-empty, MUST raise ``GuidedSchemaCompileError`` (which ``generate_json``
    then catches → ``None`` → operational path). We stub ``LLMatcher`` so its
    ``get_error()`` returns an error string and stub ``_get_lltokenizer`` so the
    method reaches the matcher check; the raise happens before any real
    tokenizer/model use, so ``model``/``tokenizer`` are bare objects.

    Runs UNCONDITIONALLY: llguidance is a CORE dependency (0.10.15+), so this
    load-bearing guided-decoder regression must never silently skip — reverting
    the raise to ``return None`` (or a broken llguidance import) turns it red.
    """
    from vllm_mlx.api import guided as guided_mod

    monkeypatch.setattr(
        guided_mod, "LLMatcher", lambda _lltok, _grammar: _StubMatcher("Invalid type")
    )
    gen = guided_mod.GuidedGenerator(model=object(), tokenizer=object())
    monkeypatch.setattr(gen, "_get_lltokenizer", lambda: object())
    with pytest.raises(GuidedSchemaCompileError) as excinfo:
        gen._decode_constrained(
            grammar="<grammar>", prompt="hi", max_tokens=8, temperature=0.0
        )
    assert "Invalid type" in str(excinfo.value)


def test_generate_json_has_no_inengine_structural_validation():
    """Round-5 (validate-once): structural schema validation is centralized at
    the ROUTE BOUNDARY and runs exactly once. The former in-generator up-front
    ``_schema_invalid_reason`` structural check is REMOVED, so ``generate_json``
    does NOT re-validate the schema (no duplicate work, nothing on the
    executor/event-loop thread twice)."""
    from vllm_mlx.api import guided as guided_mod

    assert not hasattr(guided_mod, "_schema_invalid_reason"), (
        "the in-generator structural validation must be removed — the route "
        "boundary is the single structural-validation point"
    )


def test_generate_json_treats_any_llguidance_reject_as_operational_none(monkeypatch):
    """Round-5 — with structural validation owned by the route boundary,
    ``generate_json`` no longer discriminates schema validity: EVERY llguidance
    rejection is OPERATIONAL and degrades to ``None`` (→ strict 502 / best-effort
    200), NEVER a raised 400. It must NOT raise even for a structurally-INVALID
    schema (which the boundary normally rejects first) — proving the in-generator
    layer does no structural gate."""
    from vllm_mlx.api import guided as guided_mod

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(lambda *_a, **_k: "<grammar>"),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)

    def _decode_raises(**_k):
        raise GuidedSchemaCompileError("llguidance rejected at matcher construction")

    monkeypatch.setattr(gen, "_decode_constrained", _decode_raises)
    # Neither a valid NOR an invalid schema raises — both degrade to None.
    assert gen.generate_json(prompt="hi", json_schema=_SCHEMA, max_tokens=8) is None
    assert (
        gen.generate_json(prompt="hi", json_schema=_INVALID_SCHEMA, max_tokens=8)
        is None
    )


def test_generate_json_operational_runtime_error_degrades_to_none(monkeypatch):
    """Operational arm: an INTERNAL failure (e.g. a ``RuntimeError`` /  eager
    ``ValueError`` from ``grammar_from_json_schema``) degrades to ``None`` (the
    operational path), never a 400."""
    from vllm_mlx.api import guided as guided_mod

    def _internal_boom(*_a, **_k):
        raise RuntimeError("internal resource failure / OOM")

    monkeypatch.setattr(
        guided_mod.LLMatcher,
        "grammar_from_json_schema",
        staticmethod(_internal_boom),
    )
    gen = guided_mod.GuidedGenerator(model=None, tokenizer=None)
    assert gen.generate_json(prompt="hi", json_schema=_SCHEMA, max_tokens=8) is None


# ---------------------------------------------------------------------------
# Engine path: with structural validation owned by the route boundary, the REAL
# BatchedEngine treats EVERY guided-layer failure as operational and degrades it
# to ``None`` (→ strict 502 / non-strict best-effort). It no longer discriminates
# a compile error, so there is no schema-specific propagation to assert; this
# pins that the sync guided worker degrades a raised failure to ``None``.
# ---------------------------------------------------------------------------


def _bare_engine(monkeypatch, gen_exc: Exception):
    """A minimally-wired REAL ``BatchedEngine`` whose ``GuidedGenerator``
    raises ``gen_exc`` from ``generate_json``. Constructed via ``__new__`` so
    no model load happens; only the attributes the guided path reads are set.
    """
    from vllm_mlx.engine import batched as batched_mod

    class _FakeGuidedGenerator:
        def __init__(self, model, tokenizer):
            pass

        def generate_json(self, **_kw):
            raise gen_exc

    monkeypatch.setattr(batched_mod, "GuidedGenerator", _FakeGuidedGenerator)
    eng = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    eng._model = object()
    eng._tokenizer = object()
    eng._is_mllm = False
    return eng, batched_mod


def test_run_guided_generation_degrades_failure_to_none(monkeypatch):
    """The sync guided worker degrades ANY guided-layer failure to a graceful
    ``None`` (the operational path — strict 502 / non-strict best-effort). It no
    longer re-raises a compile error: structural validity is settled at the route
    boundary, so nothing schema-specific propagates out of the engine."""
    eng, _ = _bare_engine(monkeypatch, RuntimeError("transient blip"))
    assert (
        eng._run_guided_generation(
            prompt="p", json_schema=_SCHEMA, max_tokens=8, temperature=0.0
        )
        is None
    )


async def test_generate_with_schema_operational_none_raises_no_chat_fallback(
    monkeypatch,
):
    """The strict-502 guarantee now rests ENTIRELY on the REAL
    ``BatchedEngine.generate_with_schema(..., raise_on_failure=True)`` raising when
    guided decoding returns ``None`` — once the compile-error propagation machinery
    was deleted, ``None`` is the ONLY operational-failure signal. This exercises
    the real wrapper (NOT a mock that raises directly, which would stay green even
    if the wrapper silently fell back): stub ``_run_guided_generation`` to return
    ``None``, then assert the wrapper RAISES (the route maps this to 502) and NEVER
    calls unconstrained ``chat()``. Without ``raise_on_failure`` this same ``None``
    is the best-effort fallback; with it (strict), the silent degrade is refused."""
    from vllm_mlx.engine import batched as batched_mod

    eng = batched_mod.BatchedEngine.__new__(batched_mod.BatchedEngine)
    eng._model = object()
    eng._tokenizer = object()
    eng._is_mllm = False  # → supports_guided_generation is True (llguidance core)
    eng._loaded = True
    eng._model_name = "test-model"
    # None → the wrapper uses asyncio.to_thread; our stub touches no model/GPU.
    eng._model_load_executor = None
    # Guided decoding degrades to None (operational failure — the only signal now
    # that structural validity is settled at the route boundary).
    monkeypatch.setattr(eng, "_run_guided_generation", lambda **_kw: None)
    monkeypatch.setattr(
        batched_mod, "shared_apply_chat_template", lambda *a, **k: "PROMPT"
    )

    chat_calls: list = []

    async def _spy_chat(**kwargs):
        chat_calls.append(kwargs)
        return GenerationOutput(
            text="FALLBACK", new_text="FALLBACK", finish_reason="stop"
        )

    eng.chat = _spy_chat

    with pytest.raises(RuntimeError):
        await eng.generate_with_schema(
            messages=[{"role": "user", "content": "hi"}],
            json_schema=_SCHEMA,
            raise_on_failure=True,
        )
    assert chat_calls == [], (
        "strict (raise_on_failure=True) must NOT silently fall back to "
        "unconstrained chat() when guided decoding returns None"
    )


# ---------------------------------------------------------------------------
# Non-chat endpoint coverage (Issue 1): /v1/responses must ALSO reject an invalid
# schema at the ROUTE BOUNDARY (HTTP 400, not the strict 502, not a silent 200).
# ---------------------------------------------------------------------------


@pytest.fixture
def _rate_limiter_state():
    """Save/restore the global rate-limiter so disabling it for the responses
    route does not leak into other tests."""
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


def _make_responses_client(engine: _Engine) -> TestClient:
    from vllm_mlx.routes.responses import router as responses_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    return TestClient(app)


def test_responses_nonstrict_invalid_schema_returns_400(_rate_limiter_state):
    """/v1/responses, NON-strict json_schema with a structurally-invalid schema
    → HTTP 400 at the ROUTE BOUNDARY (param ``text.format.schema``), NOT a
    silent 200. ``/v1/responses`` only guides STRICT json_schema, so without the
    boundary check a non-strict invalid schema would skip validation entirely
    and degrade to unconstrained output. Engine is never dispatched."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x",
                    "schema": _INVALID_SCHEMA,
                }
            },
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_response_format_schema"
    assert err["param"] == "text.format.schema"
    assert "is not a valid JSON schema" in err["message"]
    assert engine.guided_calls == []
    assert engine.chat_calls == [], "invalid schema must NOT fall back to chat()"


def test_responses_strict_invalid_schema_returns_400(_rate_limiter_state):
    """/v1/responses, STRICT json_schema with a structurally-invalid schema →
    400 ``invalid_strict_schema`` from the strict pre-flight, before any
    generation. Only a validator-INVALID schema is a 400."""
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x",
                    "schema": _INVALID_SCHEMA,
                    "strict": True,
                }
            },
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "invalid_strict_schema"
    assert err["param"] == "text.format.schema"
    assert engine.guided_calls == []
    assert engine.chat_calls == []


def test_responses_strict_valid_schema_operational_failure_returns_502(
    _rate_limiter_state,
):
    """Corrected strict contract (codex Round-4 #3) on /v1/responses: a
    validator-VALID schema that the guided path fails OPERATIONALLY (runtime
    failure) is the strict-failure path → 502 ``strict_schema_violation`` (param
    ``text.format.strict``), NOT a 400. Uses a valid-SHAPED schema so it clears
    the strict pre-flight and reaches ``generate_with_schema``, where the (mock)
    engine reports a non-compile operational failure."""
    engine = _Engine(guided_raises=RuntimeError("operational llguidance failure"))
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x",
                    "schema": _SCHEMA,
                    "strict": True,
                }
            },
        },
    )
    assert resp.status_code == 502, resp.text
    err = resp.json()["error"]
    assert err["code"] == "strict_schema_violation"
    assert engine.chat_calls == [], "strict mode must not fall back to chat()"


# ---------------------------------------------------------------------------
# /v1/completions (Round-5 #1): codex's premise was that the boundary check
# was missing on completions and an invalid json_schema could reach generation.
# VERDICT: the legacy completions lane rejects ANY json_schema response_format
# outright (it never routes through guided generation), so there is no silent
# 200 to guard against — this test pins that rejection so the surface can never
# regress into a silent unconstrained 200 for an invalid schema.
# ---------------------------------------------------------------------------


def _make_completions_client(engine: _Engine) -> TestClient:
    from vllm_mlx.routes.completions import router as completions_router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(completions_router)
    return TestClient(app)


def test_completions_json_schema_rejected_never_silent_200():
    """/v1/completions rejects an (invalid) json_schema response_format up-front
    with 400 ``unsupported_response_format`` — pointing callers at the chat lane
    — and NEVER a silent unconstrained 200. Uses a guided-UNSUPPORTED engine to
    prove the rejection is capability-independent and reaches no generation."""
    engine = _Engine(supports_guided=False)
    client = _make_completions_client(engine)
    resp = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hi",
            "max_tokens": 8,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": _INVALID_SCHEMA},
            },
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["code"] == "unsupported_response_format"
    assert err["type"] == "invalid_request_error"
    assert engine.guided_calls == []
    assert engine.chat_calls == []


def test_responses_strict_stream_rejected_before_generation(_rate_limiter_state):
    """Round-3 #3 pin: ``/v1/responses`` NEVER has the chat streaming's silent
    gap because a strict schema with ``stream=true`` is rejected UP-FRONT with
    a 400 (``strict_stream_unsupported``) — constrained decoding on this surface
    is buffered-only, so ``_stream_responses`` never calls
    ``generate_with_schema`` and no compile error can surface mid-SSE. This is
    the structural reason the compile-error 400 for ``/v1/responses`` lives only
    on the non-stream path (``test_responses_nonstrict_invalid_schema_returns_400``).

    Guarding this here means a future change that lets strict schemas stream on
    ``/v1/responses`` (re-opening the silent-degrade hole) turns this test red.
    ``guided_raises`` is set so that IF generation were (wrongly) reached, the
    engine would blow up — but it must not be reached at all.
    """
    engine = _Engine(guided_raises=_COMPILE_ERR)
    client = _make_responses_client(engine)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "stream": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x",
                    "schema": _SCHEMA,
                    "strict": True,
                }
            },
        },
    )
    assert resp.status_code == 400, resp.text
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "strict_stream_unsupported"
    assert engine.guided_calls == [], (
        "strict+stream must be rejected before any generation is attempted"
    )
    assert engine.chat_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
