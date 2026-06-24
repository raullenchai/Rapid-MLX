# SPDX-License-Identifier: Apache-2.0
"""H-06 #267b — strict json_schema repair-retry context-length guard.

Codex review on PR #878 surfaced that the R12-4 strict-mode repair
retry built a strictly LARGER prompt than the initial request
(prepended repair instructions, repeated schema, up to 4 KiB of
failed output) but **never re-checked the context-length gate
before re-calling the engine**. A request that passed the initial
``enforce_context_length_for_messages`` could blow context only on
the repair attempt and surface as ``502 strict_repair_engine_failure``
rather than a deterministic ``422 json_schema_violation``.

These tests pin the systemic fix:

    1. Initial fits AND repair fits  → repair runs as before.
    2. Initial fits, repair does NOT fit → SKIP repair, return the
       ORIGINAL 422 envelope (NOT 502). The skip counter ticks.
    3. Edge: repair is exactly ONE token over the model's context
       window → skip cleanly with the same 422 surface and counter.

The same gate is applied at both call sites (chat.py:2806 and
responses.py:899) via the centralized
``service.helpers.repair_messages_fit_context`` helper so the
behavior cannot drift between surfaces — a single bug class to
guard against rather than two.
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
from vllm_mlx.routes.responses import router as responses_router

# ---------------------------------------------------------------------------
# Stubs — modelled after the constraint-matrix file but with a real
# tokenizer + a tunable max-context window so we can hit the H-06
# boundary deterministically.
# ---------------------------------------------------------------------------


class _StubArgs:
    """Surface for ``model.args.max_position_embeddings`` resolution."""

    def __init__(self, max_position_embeddings: int):
        self.max_position_embeddings = max_position_embeddings


class _StubModel:
    def __init__(self, max_position_embeddings: int):
        self.args = _StubArgs(max_position_embeddings)


class _StubTokenizer:
    """Deterministic tokenizer: 1 token per 4 characters.

    The H-06 fix relies on ``count_prompt_tokens`` returning a real
    integer — the helper short-circuits on tokenizer-returned-zero,
    so we cannot stub the count to 0. We use a fixed chars-per-token
    ratio so the test can place a tiny budget at the byte boundary
    where the repair prompt exceeds context but the initial prompt
    does not.
    """

    bos_token = None

    def __init__(self, chars_per_token: int = 4):
        self._cpt = chars_per_token

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        return [0] * max(1, len(text) // self._cpt)


class _StubEngine:
    """Deterministic engine that returns a fixed body for each chat
    call AND exposes a configurable context window via ``_model``
    so the H-06 repair-fit gate can be exercised end-to-end."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False

    def __init__(self, *, body: str, max_position_embeddings: int):
        self._body = body
        self._model = _StubModel(max_position_embeddings)
        self._tokenizer = _StubTokenizer()
        self.chat_calls: list[dict] = []

    @property
    def tokenizer(self):
        return self._tokenizer

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        # Render messages as concatenated text so the prompt length
        # scales with the repair-hint + schema + failed-output
        # payload. The exact format doesn't matter as long as more
        # messages → more characters → more tokens.
        parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            else:
                # Multipart content blocks — keep the text parts.
                for p in content or []:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(str(p.get("text", "")))
        return "\n".join(parts)

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


def _client(*, body: str, max_position_embeddings: int):
    engine = _StubEngine(body=body, max_position_embeddings=max_position_embeddings)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(chat_router)
    return TestClient(app), engine


def _responses_client(*, body: str, max_position_embeddings: int):
    """Same fixture as ``_client`` but mounts the /v1/responses
    router. Both routes call into the SAME centralized helper
    (``service.helpers.repair_messages_fit_context``), so we exercise
    the responses surface end-to-end to make sure the wiring at
    ``responses.py:963`` is identical to ``chat.py``.
    """
    from vllm_mlx.middleware.auth import rate_limiter

    engine = _StubEngine(body=body, max_position_embeddings=max_position_embeddings)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True

    # /v1/responses is gated by the global rate-limiter; the strict
    # repair-overflow test file pins counter behaviour and must not
    # be perturbed by 429s.
    rate_limiter.enabled = False
    rate_limiter._requests.clear()

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(responses_router)
    return TestClient(app), engine


# ---------------------------------------------------------------------------
# Boundary fixtures
# ---------------------------------------------------------------------------

# Schema that triggers a deterministic ``minimum`` violation — the
# constraint-matrix file already pins that every constraint family
# tripping ``422`` works, so we reuse the simplest row.
_SCHEMA_MINIMUM = {
    "type": "object",
    "properties": {"age": {"type": "integer", "minimum": 18}},
    "required": ["age"],
}
_VIOLATING_BODY = json.dumps({"age": 5})


def _payload(*, messages: list[dict] | None = None) -> dict:
    return {
        "model": "test-model",
        "messages": messages or [{"role": "user", "content": "produce something"}],
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "AgeCheck",
                "schema": _SCHEMA_MINIMUM,
                "strict": True,
            },
        },
    }


def _responses_payload() -> dict:
    """``/v1/responses`` shape — schema carried under ``text.format``
    instead of ``response_format``. Mirrors the test in
    ``test_response_format_json_schema_strict.py``.
    """
    return {
        "model": "test-model",
        "input": "produce something",
        "max_output_tokens": 64,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "AgeCheck",
                "schema": _SCHEMA_MINIMUM,
                "strict": True,
            }
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — initial fits AND repair fits → repair runs.
# ---------------------------------------------------------------------------


def test_repair_runs_when_both_initial_and_repair_fit_context():
    """Generous context window: both the initial prompt and the
    repair prompt fit. Behaviour MUST be identical to the
    pre-#267b path — engine.chat called twice, 422 returned with
    ``attempts=2``, ``strict_repairs_attempted_total`` ticks, and
    the skip-counter does NOT tick.
    """
    # 1 048 576 tokens — comfortably larger than the repair prompt
    # (instructions + schema + ~50 char failed output ≈ a few
    # hundred tokens at 4 chars/token).
    client, engine = _client(body=_VIOLATING_BODY, max_position_embeddings=1_048_576)

    resp = client.post("/v1/chat/completions", json=_payload())
    assert resp.status_code == 422, resp.text
    body = resp.json()
    details = body["error"]["details"]
    # Repair retry must have fired.
    assert details["attempts"] == 2
    assert len(engine.chat_calls) == 2

    snap = response_format_metrics.snapshot()
    assert snap["strict_repairs_attempted_total"] == 1
    # Skip counter does NOT tick on the happy path.
    assert snap["strict_repairs_skipped_context_overflow_total"] == 0


# ---------------------------------------------------------------------------
# Test 2 — initial fits, repair does NOT fit → skip repair, return
# the original 422 (NOT 502), skip counter ticks.
# ---------------------------------------------------------------------------


def test_repair_skipped_when_repair_prompt_exceeds_context():
    """Tight context window: initial prompt (small user message)
    fits, but the repair prompt — which prepends instructions,
    repeats the schema, and quotes the failed output — exceeds the
    cap. Pre-#267b this surfaced as ``502
    strict_repair_engine_failure``. Post-fix, the route must SKIP
    the retry and return the ORIGINAL ``422 json_schema_violation``
    envelope so the client sees a deterministic outcome.

    Concretely:
      * status_code == 422 (NOT 502)
      * envelope code == "json_schema_violation"
      * attempts == 1 (single generation, no retry)
      * engine.chat called exactly once (the initial attempt)
      * strict_repairs_skipped_context_overflow_total ticks by 1
      * strict_repairs_attempted_total does NOT tick (no retry
        was attempted)
    """
    # Sizing (4 chars/token stub tokenizer):
    #   * INITIAL prompt (injected JSON system + user msg) = ~116 tokens.
    #     With ``max_tokens=64`` → requested_total = 180 ≤ 256 → fits.
    #   * REPAIR prompt (repair instructions + schema dup + failed
    #     output quote + user retry) = ~302 tokens. With
    #     ``max_tokens=64`` → requested_total = 366 > 256 → fails.
    # 256 is the cleanest "initial passes, repair fails" cap.
    client, engine = _client(body=_VIOLATING_BODY, max_position_embeddings=256)

    resp = client.post("/v1/chat/completions", json=_payload())
    # Critical: NOT 502.
    assert resp.status_code == 422, (
        f"expected 422 (deterministic validation outcome), got {resp.status_code}: "
        f"{resp.text}"
    )
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation"
    assert body["error"]["type"] == "validation_error"
    details = body["error"]["details"]
    # Only ONE generation attempt because the repair was skipped.
    assert details["attempts"] == 1
    # The original validation failure (NOT the made-up
    # ``initial_failure`` payload from the 502 path) is preserved.
    assert details.get("reason") == "schema_violation"

    # Engine MUST NOT have been called a second time.
    assert len(engine.chat_calls) == 1, (
        f"repair was skipped → expected 1 chat call, got {len(engine.chat_calls)}"
    )

    snap = response_format_metrics.snapshot()
    # Skip counter ticked exactly once.
    assert snap["strict_repairs_skipped_context_overflow_total"] == 1, snap
    # Attempt counter did NOT tick — we never called the engine for the retry.
    assert snap["strict_repairs_attempted_total"] == 0, snap
    # The violation still ticks because the client sees a 422.
    assert snap["strict_violations_total"] == 1, snap


# ---------------------------------------------------------------------------
# Test 3 — edge: initial barely fits, repair is exactly one token
# over → skip cleanly with the same 422 surface and counter.
# ---------------------------------------------------------------------------


def test_repair_skipped_at_exact_one_token_overflow():
    """Edge of the gate: pick a context window such that the
    initial prompt fits (with room for ``max_tokens=64``
    completion) but the repair prompt — at ``count_prompt_tokens``
    + completion budget — is exactly one token over the cap.

    The boundary is checked with ``<=`` (request <= max_context),
    so "exactly one over" must fail cleanly. We do NOT need to
    compute the exact token count — we ratchet the cap up to a
    value that we KNOW the initial passes but the repair fails on,
    and we assert the same observable contract as Test 2:

      * 422 (NOT 502)
      * single chat call
      * skip counter == 1, attempt counter == 0

    This is the "one token over" boundary because we already
    proved (Test 1) that a much-larger cap allows the repair, and
    we now sit at the smallest cap that still excludes the repair
    prompt. Concretely: 128 tokens is tight enough that the
    several-hundred-character repair prompt (instructions +
    schema + failed-output quote) at 4 chars/token cannot fit.
    """
    # Sizing math (4 chars/token stub tokenizer):
    #   * REPAIR prompt tokens (concat of all repair messages) = 302.
    #   * ``max_tokens=64`` completion budget.
    #   * ``requested_total = 302 + 64 = 366``.
    # The gate's check is ``requested_total <= max_context``. Setting
    # ``max_position_embeddings=365`` puts the repair exactly one
    # token over the cap — boundary case. The initial prompt
    # (~116+64=180 tokens) still fits trivially.
    client, engine = _client(body=_VIOLATING_BODY, max_position_embeddings=365)

    resp = client.post("/v1/chat/completions", json=_payload())
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation"
    details = body["error"]["details"]
    assert details["attempts"] == 1
    assert len(engine.chat_calls) == 1

    snap = response_format_metrics.snapshot()
    assert snap["strict_repairs_skipped_context_overflow_total"] == 1, snap
    assert snap["strict_repairs_attempted_total"] == 0, snap


# ---------------------------------------------------------------------------
# Bonus — pin the helper's permissive-skip contract.
# ---------------------------------------------------------------------------


def test_repair_fits_helper_returns_true_when_engine_is_mllm():
    """``repair_messages_fit_context`` is intentionally permissive
    on the MLLM path — same as
    ``enforce_context_length_for_messages``. The MLLM scheduler
    handles its own context accounting (multimodal token costs are
    not visible to the text-only tokenizer), so the helper returns
    ``True`` and lets the engine surface its own error if the prompt
    really is too large.

    This pins that contract so a future "tighten the MLLM path"
    refactor doesn't accidentally also tighten the repair gate and
    introduce a regression on the multimodal surface.
    """
    from vllm_mlx.service.helpers import repair_messages_fit_context

    class _MLLM:
        is_mllm = True

    assert repair_messages_fit_context(_MLLM(), [{"role": "user", "content": "x"}])


def test_repair_fits_helper_returns_true_when_build_prompt_missing():
    """If the engine doesn't expose ``build_prompt`` (route stub,
    half-loaded engine) the helper cannot compute the prompt size
    and falls through to ``True`` — exact mirror of the
    permissive-skip path in
    ``enforce_context_length_for_messages``. The downstream
    scheduler's own validation still applies.
    """
    from vllm_mlx.service.helpers import repair_messages_fit_context

    class _NoBuildPrompt:
        is_mllm = False

    assert repair_messages_fit_context(
        _NoBuildPrompt(), [{"role": "user", "content": "x"}]
    )


# ---------------------------------------------------------------------------
# /v1/responses parity — codex r1 #2: the chat suite above proves the
# centralized helper is wired into chat.py, but the /v1/responses
# call site (responses.py:963) has its own envelope, its own request-
# disconnect path, and its own ``max_output_tokens`` plumbing. Pin
# the SAME overflow contract end-to-end on the responses surface so a
# future refactor that only edits the chat route cannot regress
# /v1/responses without an obvious test failure.
# ---------------------------------------------------------------------------


def test_responses_repair_skipped_when_repair_prompt_exceeds_context():
    """``/v1/responses`` parity for Test 2 (chat-side).

    Tight context window: the initial prompt fits but the post-build
    repair prompt does not. Pre-#267b /v1/responses would have
    surfaced this as ``502 strict_repair_engine_failure``. Post-fix
    the route MUST short-circuit, leave ``attempts == 1``, and
    surface the original ``422 json_schema_violation`` envelope.

    Pinned observable contract on the responses surface:
      * status_code == 422 (NOT 502)
      * error.code == "json_schema_violation"
      * error.param == "text.format" (responses-specific envelope)
      * engine.chat called exactly once
      * strict_repairs_skipped_context_overflow_total ticks by 1
      * strict_repairs_attempted_total does NOT tick
    """
    # Sizing on /v1/responses (4 chars/token stub tokenizer):
    #   * Unlike /v1/chat/completions, the responses route does NOT
    #     inject a ``build_json_system_prompt`` instruction into the
    #     initial messages (chat.py:1919 has no responses-side
    #     parallel). The initial prompt is therefore just the user
    #     "produce something" message → a few tokens; trivially fits.
    #   * The REPAIR prompt (instructions + schema + failed-output
    #     quote + retry hint) ≈ 188 tokens. With ``max_output_tokens=64``
    #     → 252 total. Setting the cap to 200 puts the repair 52
    #     tokens over the limit so the gate MUST fire while the
    #     initial pass still succeeds.
    client, engine = _responses_client(
        body=_VIOLATING_BODY, max_position_embeddings=200
    )

    resp = client.post("/v1/responses", json=_responses_payload())
    # The critical assertion: NOT 502 (no opaque server-side error).
    assert resp.status_code == 422, (
        f"/v1/responses expected 422 (deterministic validation outcome), "
        f"got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation", body
    assert body["error"]["type"] == "validation_error", body
    # Responses surface carries ``text.format`` in the envelope —
    # distinct from chat's ``response_format`` param.
    assert body["error"].get("param") == "text.format", body

    # Engine MUST NOT have been called a second time.
    assert len(engine.chat_calls) == 1, (
        f"/v1/responses repair was skipped → expected 1 chat call, "
        f"got {len(engine.chat_calls)}"
    )

    snap = response_format_metrics.snapshot()
    # Skip counter ticked exactly once — same counter as chat.py.
    assert snap["strict_repairs_skipped_context_overflow_total"] == 1, snap
    # Attempt counter did NOT tick — repair was skipped, not run.
    assert snap["strict_repairs_attempted_total"] == 0, snap
    # The violation still ticks because the client sees a 422.
    assert snap["strict_violations_total"] == 1, snap


def test_responses_repair_runs_when_both_initial_and_repair_fit_context():
    """``/v1/responses`` parity for Test 1 (chat-side).

    Sanity-check the responses surface on the HAPPY path: with a
    generous context window the repair runs, ``attempts == 2``, and
    the skip counter does NOT tick. Without this companion test, a
    regression that wedged the responses skip path "always-skip"
    could pass the overflow test above on its own. Pinning the
    happy path here ensures the gate only fires when it should.
    """
    client, engine = _responses_client(
        body=_VIOLATING_BODY, max_position_embeddings=1_048_576
    )

    resp = client.post("/v1/responses", json=_responses_payload())
    # Repair fires, validation still fails → 422 with attempts=2.
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "json_schema_violation", body
    # Engine called twice — initial + repair retry.
    assert len(engine.chat_calls) == 2, (
        f"/v1/responses repair should have run → expected 2 chat calls, "
        f"got {len(engine.chat_calls)}"
    )

    snap = response_format_metrics.snapshot()
    # Repair was attempted; skip counter did NOT tick.
    assert snap["strict_repairs_attempted_total"] == 1, snap
    assert snap["strict_repairs_skipped_context_overflow_total"] == 0, snap
