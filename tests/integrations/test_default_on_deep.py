# SPDX-License-Identifier: Apache-2.0
"""Stage-2 deep cells for the #558 default-on constrained-tool-calling gate.

The always-on agent×family *smoke* matrix (``test_agents_matrix.py`` /
``test_frameworks_matrix.py``) proves every agent wire still SPEAKS on the
running server. This file adds the DEEP coverage the default-on flip needs:

* **Multi-turn tool loop** — issue a tool call, feed a synthetic tool result
  back as a ``role="tool"`` turn, and assert the model produces a coherent
  final answer that consumes the tool output. A single-turn smoke can pass
  even if the multi-turn tool-result plumbing regresses.
* **Varied JSON schemas** — enum / nested-object / required-fields /
  ``additionalProperties:false``. These are exactly the schema shapes a
  grammar constraint has to honour; the default-on flip must not regress any
  of them.
* **Negative control (the constraint actually FIRES)** — the same schema is
  requested twice against the *live* guided-decode path
  (``response_format={"type":"json_schema","strict":true,...}`` → llguidance,
  see ``vllm_mlx/api/guided.py`` + ``vllm_mlx/engine/batched.py``): once
  UN-constrained (plain ``json_object`` / no schema — the model is free to
  emit an off-schema / hallucinated field) and once CONSTRAINED (strict
  schema — llguidance masks the off-schema token). The control PASSES only
  when the constrained run is on-schema; if a "constrained" run can still
  emit an off-schema key, the constraint is not firing and the cell fails.

Design notes
------------
* These cells reuse the SAME conftest fixtures as the smoke matrix
  (``rapid_mlx_server``, ``family_alias``, the family-guard autouse fixture,
  the strict-xfail collection hook). They are ``family_alias``-parametrized,
  so a single-family server boot runs only that family's deep cells and the
  family-guard skips the rest — identical semantics to the smoke matrix.
* Constraint MODE is a SERVER-env toggle applied when the per-arm server is
  booted: the off arm boots with ``RAPID_MLX_CONSTRAIN_TOOLS=0`` (free-form
  base) and the on arm boots default-on (env unset — #558 PR-5 flipped the
  default to ON). Both arms keep ``tool_choice="auto"`` so the REAL PR-5
  auto-path is exercised; the only independent variable is the server-side
  constraint. ``RAPID_MLX_TOOL_CONSTRAINT`` (optionally set on the pytest
  process) now only labels the per-cell latency breadcrumb with its mode.
  Before PR-5 landed, ``on`` was a per-request
  ``tool_choice="auto"->"required"`` proxy in ``_apply_constraint_mode``; that
  seam is now a no-op that pins ``tool_choice="auto"`` in both arms. The
  negative-control cell does NOT depend on the toggle: it drives the
  ALREADY-LIVE ``response_format`` guided path, so it proves the underlying
  llguidance masking works on any ref, today.
* Every cell degrades to skip on a missing server / SDK unless
  ``RAPID_MLX_MATRIX_STRICT=1`` (shared ``strict_skip_or_fail`` semantics),
  so a naive ``pytest tests/integrations`` on a clean box stays green.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

# ``jsonschema`` is a hard dep of the [dev] extra (used across the suite), so we
# import it at module scope: the varied-schema cell validates EVERY emitted
# argument object against the COMPLETE tool parameter schema (codex #558-PR5),
# which requires it. If it were somehow absent the deep cells cannot prove the
# constraint, so failing to import here is correct (not a silent skip).
import jsonschema
import pytest

from tests.integrations.conftest import (
    FamilyAlias,
    assert_content_nonempty,
    assert_no_analysis_channel_leak,
    assert_no_think_tag_leak,
    strict_skip_or_fail,
)

# --------------------------------------------------------------------------- #
# Constraint mode (forward-compatible knob)
# --------------------------------------------------------------------------- #


def constraint_mode() -> str:
    """Return the requested constraint mode: ``"off"`` (default) or ``"on"``.

    Read from ``RAPID_MLX_TOOL_CONSTRAINT`` (optionally set on the pytest
    process) so the SAME cells can be run once per mode and a caller can diff
    the two per-cell result sets (baseline-vs-constrained). Defaults to ``off``
    when unset, so an ordinary ``pytest`` run needs no external orchestration.
    """
    val = os.environ.get("RAPID_MLX_TOOL_CONSTRAINT", "off").strip().lower()
    return "on" if val in ("1", "on", "true", "yes") else "off"


def _apply_constraint_mode(payload: dict[str, Any]) -> dict[str, Any]:
    """Pin ``tool_choice="auto"`` in BOTH arms — the seam PR-5 turned into a no-op.

    #558 PR-5 landed the real default-on knob, so the off/on distinction is now
    driven by the SERVER the gate driver boots, NOT by a per-request
    ``tool_choice`` swap:

    * **off** — driver boots the server with ``RAPID_MLX_CONSTRAIN_TOOLS=0``
      (constrained tool-calling opted OUT -> legacy free-form base).
    * **on**  — driver boots the server default-on (env unset/ON), so
      ``tool_choice="auto"`` exercises the REAL PR-5 auto-path optional-call
      grammar (the model MAY emit a structurally-correct call or plain text and
      is never forced).

    Both arms therefore issue the identical ``tool_choice="auto"`` request and
    the server-side constraint is the only independent variable — that is what
    makes the off-vs-on comparison a true free-form-vs-constrained parity test.
    Before PR-5 this seam forced ``tool_choice="required"`` on the on arm as a
    pre-landing proxy, which would have tested forced-emission, not the auto
    path. ``constraint_mode()`` is still read to label the latency breadcrumb.
    """
    payload = dict(payload)
    payload["tool_choice"] = "auto"
    return payload


# --------------------------------------------------------------------------- #
# Shared tool + schema fixtures
# --------------------------------------------------------------------------- #


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}

# Varied schemas the default-on flip must not regress. Each is a self-contained
# JSON Schema exercised through the /v1/chat/completions ``tools`` path.
_VARIED_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "enum": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city", "unit"],
        "additionalProperties": False,
    },
    "nested_object": {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                },
                "required": ["city"],
            }
        },
        "required": ["location"],
    },
    "required_fields": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "day": {"type": "string"},
        },
        "required": ["city", "day"],
    },
    "no_additional_props": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}


def _openai_client_and_errors(base_url: str):
    """Lazy openai import + typed wire-error tuple (mirrors the smoke matrix)."""
    try:
        from openai import (
            APIStatusError,
            BadRequestError,
            NotFoundError,
            OpenAI,
        )
    except ImportError:
        pytest.skip("openai package not installed — deep cells skipped")
    client = OpenAI(base_url=base_url, api_key="not-needed")
    return client, (BadRequestError, NotFoundError, APIStatusError)


# --------------------------------------------------------------------------- #
# Stage-2 deep cell: multi-turn tool loop
# --------------------------------------------------------------------------- #


class TestMultiTurnToolLoop:
    """Two-turn tool loop: call → synthetic tool result → grounded final answer.

    Proves the multi-turn tool-result plumbing (``role="tool"`` turn keyed by
    ``tool_call_id``) survives the default-on flip. A single-turn smoke passes
    even if this regresses.
    """

    def test_two_turn(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
        model_id = rapid_mlx_server["model_id"]
        ctx = f"multiturn/{family_alias.family}"

        first_payload = _apply_constraint_mode(
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo? Use get_weather.",
                    }
                ],
                "tools": [_WEATHER_TOOL],
                "temperature": 0.0,
                "max_tokens": 384,
            }
        )
        t0 = time.perf_counter()
        try:
            first = client.chat.completions.create(**first_payload)
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected turn-1 tool request: {exc}")
        msg = first.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            # Small alias may answer inline — assert wire cleanliness, then
            # treat the missing call as a strict-mode regression signal.
            content = msg.content or ""
            assert_content_nonempty(content, ctx=ctx)
            assert_no_think_tag_leak(content)
            assert_no_analysis_channel_leak(content)
            strict_skip_or_fail(
                f"{ctx}: turn-1 produced no tool_calls (content={content[:120]!r})"
            )
            return

        tc = tool_calls[0]
        assert tc.function.name == "get_weather", tc.function.name
        # ``arguments`` must be a JSON-parseable object. We do NOT hard-require
        # the ``city`` key here: under the constrained-mode proxy
        # (``tool_choice="required"``) a small model's FORCED call can come back
        # with empty ``{}`` args — that is a known forced-emission behavior on
        # tiny aliases, not a wire regression to gate on. When it IS populated
        # we assert it names Tokyo; either way we feed a fixed Tokyo tool result
        # in turn 2 so the grounded-answer assertion stays deterministic.
        args = json.loads(tc.function.arguments)
        assert isinstance(args, dict), f"{ctx}: tool args not an object: {args!r}"
        if "city" in args and args["city"]:
            assert "tokyo" in str(args["city"]).lower(), (
                f"{ctx}: forced call named wrong city: {args!r}"
            )

        # Turn 2 — feed a synthetic tool result back and ask for a final answer.
        second_payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo? Use get_weather.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(
                        {"city": "Tokyo", "temp_c": 21, "sky": "sunny"}
                    ),
                },
            ],
            "tools": [_WEATHER_TOOL],
            "temperature": 0.0,
            "max_tokens": 384,
        }
        try:
            second = client.chat.completions.create(**second_payload)
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected turn-2 tool-result: {exc}")
        latency_s = time.perf_counter() - t0

        final = second.choices[0].message.content or ""
        assert_content_nonempty(final, ctx=ctx)
        assert_no_think_tag_leak(final)
        assert_no_analysis_channel_leak(final)
        # The final answer must consume the tool RESULT — grounded on the
        # values only the tool returned (21°C / sunny), NOT the echoed city.
        # ``Tokyo`` is already in the user prompt, so accepting it would let an
        # answer that ignored the tool output pass (codex #558-PR5 finding 4).
        # We require result-only evidence: the temperature or the sky the tool
        # reported, which the model could only have obtained from the fed-back
        # ``role="tool"`` turn.
        low = final.lower()
        assert ("21" in final) or ("sunny" in low), (
            f"{ctx}: final answer {final[:200]!r} did not consume the tool "
            "result — expected the tool-returned 21°C / sunny, not the echoed "
            "city (which the user prompt already contained)"
        )
        # Perf breadcrumb for the gate's per-cell latency record.
        print(f"[deep-latency] {ctx} mode={constraint_mode()} {latency_s:.2f}s")


# --------------------------------------------------------------------------- #
# Stage-2 deep cell: varied tool schemas
# --------------------------------------------------------------------------- #


class TestVariedSchemas:
    """Tool call against enum / nested / required / additionalProperties:false.

    Parametrized over the four schema shapes a grammar constraint must honour.
    Each cell asserts a well-formed call whose ``arguments`` JSON-parse and
    respect the shape's key constraints.
    """

    @pytest.mark.parametrize("schema_key", sorted(_VARIED_TOOL_SCHEMAS))
    def test_schema_shape(
        self,
        rapid_mlx_server: dict[str, Any],
        family_alias: FamilyAlias,
        schema_key: str,
    ) -> None:
        client, wire_errors = _openai_client_and_errors(rapid_mlx_server["base_url"])
        model_id = rapid_mlx_server["model_id"]
        ctx = f"schema-{schema_key}/{family_alias.family}"
        schema = _VARIED_TOOL_SCHEMAS[schema_key]

        tool = {
            "type": "function",
            "function": {
                "name": "record_query",
                "description": "Record a structured weather query.",
                "parameters": schema,
            },
        }
        payload = _apply_constraint_mode(
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Record a weather query for Tokyo, Japan, on "
                            "Monday, in celsius. Call record_query with all "
                            "the fields the schema requires."
                        ),
                    }
                ],
                "tools": [tool],
                "temperature": 0.0,
                "max_tokens": 384,
            }
        )
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**payload)
        except wire_errors as exc:
            strict_skip_or_fail(f"{ctx}: server rejected schema request: {exc}")
        latency_s = time.perf_counter() - t0

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            content = msg.content or ""
            assert_content_nonempty(content, ctx=ctx)
            assert_no_think_tag_leak(content)
            assert_no_analysis_channel_leak(content)
            strict_skip_or_fail(f"{ctx}: no tool_calls (content={content[:120]!r})")
            return

        # Validate EVERY emitted tool_call, not just the first (codex #558-PR5
        # finding 5): a model that emits a correct first call plus a malformed
        # additional call must fail. For each call we assert it names the tool we
        # offered and that its ``arguments`` JSON-parse and satisfy the COMPLETE
        # parameter schema. ``jsonschema.validate`` enforces the whole contract:
        # ``additionalProperties: false`` rejects any forbidden extra key, and a
        # missing ``required`` field fails — real proof the constrained arm
        # honours the schema, where the earlier per-key spot checks (only on
        # ``tool_calls[0]``) could pass a call that snuck in an extra key, dropped
        # a required one, or a bad sibling call entirely.
        for idx, call in enumerate(tool_calls):
            called = call.function.name
            assert called == "record_query", (
                f"{ctx}: tool_calls[{idx}] wrong function called: {called!r} "
                "(expected record_query)"
            )
            args = json.loads(call.function.arguments)
            assert isinstance(args, dict), (
                f"{ctx}: tool_calls[{idx}] args not an object: {args!r}"
            )
            try:
                jsonschema.validate(instance=args, schema=schema)
            except jsonschema.ValidationError as exc:
                pytest.fail(
                    f"{ctx}: tool_calls[{idx}] args violate the parameter "
                    f"schema: {args!r} — {exc.message}"
                )
        print(f"[deep-latency] {ctx} mode={constraint_mode()} {latency_s:.2f}s")


# --------------------------------------------------------------------------- #
# Stage-2 deep cell: NEGATIVE CONTROL — the constraint actually fires
#
# DETERMINISTIC offline token-mask probe (codex #558-PR5). The prior version
# drove a LIVE model through ``response_format=json_schema`` and only asserted
# the CONSTRAINED output was on-schema — so it passed TRIVIALLY whenever the
# model happened to emit an in-schema value on its own, proving nothing about
# masking. A real negative control must show the constraint CHANGES the outcome:
# the SAME off-schema token that is ACCEPTED without guidance is REJECTED with
# guidance. We do that offline (no model, no sampling → zero flake) with the
# llguidance ``LLMatcher.consume_tokens`` idiom from
# ``tests/test_tool_grammar_558.py``: feed a fixed off-schema JSON token stream
# to (a) a PERMISSIVE grammar (no guidance) and (b) the strict-schema grammar.
# --------------------------------------------------------------------------- #

# The pinned single-special-token tokenizer the #558 enforcement proofs use
# (mirrors ``tests/test_tool_grammar_558.py``); an IMMUTABLE revision so the
# probe exercises a fixed artifact.
_NEGCTRL_TOKENIZER = "mlx-community/Qwen3.5-4B-MLX-4bit"
_NEGCTRL_REVISION = "32f3e8ecf65426fc3306969496342d504bfa13f3"

# Strict schema: ``answer`` is pinned to the enum ["yes","no"]. "maybe" is
# trivially off-schema and cannot be reassembled — a clean masking target.
_NEGCTRL_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}
_NEGCTRL_OFFSCHEMA = '{"answer": "maybe"}'  # "maybe" ∉ {"yes","no"}
_NEGCTRL_ONSCHEMA = '{"answer": "yes"}'  # positive control


def _negctrl_offline_skip_types() -> tuple[type[BaseException], ...]:
    """Only genuine offline/cache-miss errors are a sanctioned skip.

    Mirrors ``test_tool_grammar_558._offline_skip_exc_types``: a corrupt/absent
    tokenizer artifact must FAIL the probe, not silently skip. We skip ONLY on
    the specific huggingface_hub offline / cache-miss signals (or a raw
    connection error); every other exception propagates.
    """
    types: list[type[BaseException]] = []
    try:
        from huggingface_hub.errors import (
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
        )

        types += [LocalEntryNotFoundError, OfflineModeIsEnabled]
    except Exception:  # pragma: no cover - old hub without these names
        pass
    try:
        from requests.exceptions import ConnectionError as _ReqConnErr

        types.append(_ReqConnErr)
    except Exception:  # pragma: no cover - requests not present
        pass
    return tuple(types)


def _is_negctrl_offline_cache_miss(exc: BaseException) -> bool:
    """Recognize a genuine HF offline miss through transformers wrappers.

    Transformers 5.15 wraps ``LocalEntryNotFoundError`` in a generic
    ``OSError``. Match the typed cause anywhere in the exception chain rather
    than treating every disk-related ``OSError`` as skippable; corrupt local
    tokenizer artifacts must remain real failures.
    """
    offline_types = _negctrl_offline_skip_types()
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, offline_types):
            return True
        current = current.__cause__ or current.__context__
    return False


def test_negctrl_offline_cache_miss_detection_is_narrow():
    """Wrapped HF misses skip, but unrelated OSErrors still fail closed."""
    assert not _is_negctrl_offline_cache_miss(OSError("corrupt tokenizer.json"))
    hub_errors = pytest.importorskip("huggingface_hub.errors")
    wrapped = OSError("transformers could not resolve the tokenizer")
    wrapped.__cause__ = hub_errors.LocalEntryNotFoundError("not cached")
    assert _is_negctrl_offline_cache_miss(wrapped)


def _load_negctrl_tokenizers():
    """Return ``(tok, lltok)`` for the offline probe, or ``pytest.skip``.

    Skips ONLY on genuine unavailability (llguidance extra absent, tokenizer
    neither cached nor reachable, or a slow-only tokenizer). A real
    ``from_tokenizer`` regression is surfaced, not swallowed.
    """
    import importlib.util

    if importlib.util.find_spec("llguidance") is None:
        pytest.skip("llguidance ([guided] extra) not installed — negctrl needs it")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            _NEGCTRL_TOKENIZER, revision=_NEGCTRL_REVISION
        )
    except Exception as exc:  # noqa: BLE001 - re-raised unless typed offline miss
        if not _is_negctrl_offline_cache_miss(exc):
            raise
        pytest.skip(
            f"tokenizer {_NEGCTRL_TOKENIZER}@{_NEGCTRL_REVISION[:8]} not cached "
            "and no network — negative control requires it"
        )
    import llguidance.hf as llg_hf

    candidates = []
    inner = getattr(tok, "_tokenizer", None)
    if inner is not None:
        candidates.append(inner)
    candidates.append(tok)
    fast = [c for c in candidates if getattr(c, "is_fast", True) is not False]
    if not fast:
        pytest.skip("tokenizer is not a fast tokenizer — llguidance needs one")
    last_exc = None
    for cand in fast:
        try:
            return tok, llg_hf.from_tokenizer(cand)
        except Exception as exc:  # noqa: BLE001 - re-raised below if all fail
            last_exc = exc
    raise AssertionError(
        f"llguidance could not build an LLTokenizer from any fast candidate: "
        f"{last_exc!r}"
    )


def _negctrl_consume(tok, lltok, grammar, text: str) -> tuple[int, int, bool]:
    """Advance ``grammar`` over ``text``'s tokens; return (accepted, total, terminal).

    Uses ``LLMatcher.consume_tokens`` (advances real matcher state) so a full
    accept also proves the string is a COMPLETE derivation, not merely a
    prefix — identical to the ``_consume`` idiom in test_tool_grammar_558.
    """
    from llguidance.mlx import LLMatcher

    ids = tok.encode(text, add_special_tokens=False)
    matcher = LLMatcher(lltok, grammar)
    assert not matcher.get_error(), matcher.get_error()
    accepted = 0
    for tid in ids:
        if not matcher.consume_tokens([tid]):
            break
        accepted += 1
    return accepted, len(ids), matcher.is_accepting()


class TestConstraintNegativeControl:
    """Prove llguidance masking actually fires — not a no-op passthrough.

    DETERMINISTIC, OFFLINE, model-free. Builds the strict-``["yes","no"]`` JSON
    schema grammar and a PERMISSIVE (accept-any-bytes) grammar, then feeds the
    SAME off-schema token stream (``{"answer": "maybe"}``) to both:

    * WITHOUT guidance (permissive grammar): the off-schema value is ACCEPTED in
      full — establishing that the token stream is otherwise unobjectionable, so
      any later rejection is attributable to the constraint alone.
    * WITH guidance (strict schema grammar): the off-schema ``"maybe"`` token is
      REJECTED mid-stream — the constraint demonstrably masks it.

    If the guided run ALSO accepted the off-schema value the constraint would be
    a no-op and the cell FAILS. A positive control (on-schema ``"yes"`` accepted
    + terminal under the same strict grammar) rules out a grammar that rejects
    everything. This is the load-bearing proof for raullen's acceptance bar and,
    being offline, produces a real signal on ANY git ref with zero flake.
    """

    def test_offschema_token_accepted_without_guidance_rejected_with(self) -> None:
        from llguidance.mlx import LLMatcher

        tok, lltok = _load_negctrl_tokenizers()

        constrained = LLMatcher.grammar_from_lark(
            "%llguidance {}\nstart: %json " + json.dumps(_NEGCTRL_SCHEMA) + "\n"
        )
        permissive = LLMatcher.grammar_from_lark(
            "%llguidance {}\nstart: TAG_TEXT\nTAG_TEXT: /(.|\\n)*/\n"
        )

        # (a) WITHOUT guidance: the off-schema stream is fully accepted + terminal
        # — a truly unconstrained baseline (else the negative control is vacuous).
        u_acc, u_total, u_term = _negctrl_consume(
            tok, lltok, permissive, _NEGCTRL_OFFSCHEMA
        )
        assert u_acc == u_total and u_term, (
            f"WITHOUT guidance the off-schema stream {_NEGCTRL_OFFSCHEMA!r} was "
            f"not fully accepted ({u_acc}/{u_total}, terminal={u_term}) — the "
            "unconstrained baseline is not truly permissive, so the probe would "
            "be vacuous"
        )

        # (b) WITH guidance: the SAME off-schema stream is REJECTED mid-stream —
        # the constraint masks the off-enum token. A no-op constraint would
        # accept it (u_acc == c_acc), which fails here.
        c_acc, c_total, c_term = _negctrl_consume(
            tok, lltok, constrained, _NEGCTRL_OFFSCHEMA
        )
        assert c_acc < c_total, (
            f"CONSTRAINT NO-OP — off-schema stream {_NEGCTRL_OFFSCHEMA!r} was "
            f"ACCEPTED in full under the strict schema ({c_acc}/{c_total}). "
            "llguidance masking is not firing."
        )
        assert not c_term, (
            f"off-schema stream reached a terminal state under the strict schema "
            f"({c_acc}/{c_total}) — the constraint did not actually forbid it"
        )

        # Positive control: an ON-schema value IS accepted + terminal under the
        # SAME strict grammar — proving (b)'s rejection is the enum constraint,
        # not a grammar that rejects everything.
        p_acc, p_total, p_term = _negctrl_consume(
            tok, lltok, constrained, _NEGCTRL_ONSCHEMA
        )
        assert p_acc == p_total and p_term, (
            f"strict schema rejected the ON-schema value {_NEGCTRL_ONSCHEMA!r} "
            f"({p_acc}/{p_total}, terminal={p_term}) — the grammar is over-"
            "constraining, so (b)'s rejection is not a clean enum-mask signal"
        )
        print(
            f"[negctrl] offline mask proof: off-schema accepted-without-guidance="
            f"{u_acc}/{u_total} rejected-with-guidance={c_acc}/{c_total} "
            f"on-schema={p_acc}/{p_total}"
        )
