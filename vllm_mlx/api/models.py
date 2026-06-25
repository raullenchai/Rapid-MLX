# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

import math
import re
import time
import uuid
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

# =============================================================================
# Shared sampling-parameter validators (F-011)
# =============================================================================
#
# Range validators of the form ``not (0 < x <= 2)`` are False for NaN
# because every comparison involving NaN returns False — so NaN slips
# past as "valid" and is then handed to the sampler / Metal kernels.
# Symptoms observed pre-fix on /v1/chat/completions and /v1/completions:
#
#   * ``temperature=NaN`` / ``top_p=NaN`` → HTTP 200 with
#     ``choices[0].message.content=null`` + ``usage=0,0,0``; the Metal
#     backend then aborts the command buffer with a GPU Timeout and the
#     server process dies. Silent burn from the client's POV.
#   * ``presence_penalty=±10`` / ``frequency_penalty=±10`` → HTTP 200
#     with mathematically undefined logit shifts (OpenAI spec caps both
#     at [-2, 2]).
#   * ``presence_penalty=inf`` / ``frequency_penalty=inf`` → HTTP 200,
#     same undefined-logit hazard.
#
# Fix shape: declare the OpenAI-spec range on the field itself (Field
# ge=/le=) so Pydantic emits a 422 for finite out-of-range values, and
# add a single ``field_validator`` that rejects NaN/inf. The
# field-level Field bounds skip NaN (same comparison semantics as the
# legacy in-route guard), so the finite check has to run separately —
# Pydantic v2 invokes both gates per field, and either one failing
# returns 422 with a clear "Input should be a finite number" / "Input
# should be less than or equal to N" message.
#
# Range is OpenAI-spec-faithful:
#   * temperature       : [0, 2]
#   * top_p             : (0, 1]   — 0 disables sampling (illegal)
#   * presence_penalty  : [-2, 2]
#   * frequency_penalty : [-2, 2]
#
# Defined as module-level helpers so both ChatCompletionRequest and
# CompletionRequest share the exact same logic — neither schema can
# drift away from the other.


def _reject_non_one_n(v: int | None) -> int | None:
    """Reject ``n`` values other than ``1`` (or omitted ``None``) on
    chat/completion requests (F-155).

    Rapid-MLX only generates one completion per request. Pre-fix the
    route layer enforced ``n > 1`` → 400 but silently accepted
    ``n == 0`` and ``n == -1`` (HTTP 200 with one choice). Both
    forms are almost always client-side bugs — ``n=0`` is a typo for
    ``n=1`` and ``n=-1`` is a serialization mistake (e.g. SDK
    sentinel for "use server default"). Accepting them as 200 hid
    the bug; rejecting at parse time surfaces it to the client with
    the same error shape as ``n > 1``.

    Returns ``None`` unchanged so the field's optional contract
    is preserved (no value still means "1 choice"). Booleans are
    rejected explicitly because Python's ``bool`` is an ``int``
    subclass and Pydantic would otherwise coerce ``True`` → 1 and
    ``False`` → 0 silently.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("n must be an integer equal to 1 (not bool)")
    if v != 1:
        raise ValueError(
            "n must equal 1 (multi-choice is not supported; omit the field or pass n=1)"
        )
    return v


def _reject_nonfinite_float(v: float | None) -> float | None:
    """Reject NaN / ±inf on a sampling-parameter float field.

    Returns ``None`` unchanged so the field's default-None semantics
    are preserved. Raises ``ValueError`` (→ Pydantic 422) on any
    non-finite value. Integer wire values that arrive as ``float``
    (Pydantic coerces ``1`` → ``1.0`` on a ``float | None`` field) are
    fine — ``math.isfinite`` handles them.
    """
    if v is None:
        return None
    if not math.isfinite(v):
        raise ValueError("must be a finite number (not NaN or inf)")
    return v


# =============================================================================
# Shared finite-range guard (H-10) — one source of truth
# =============================================================================
#
# H-10 generalized F-011: ``repetition_penalty=-1.0`` slipped past every
# gate, reached ``mlx_lm/sample_utils.py:298`` (``make_repetition_penalty``
# raises ``ValueError`` on ``penalty < 0``), and the un-caught exception
# escaped the request scheduler and crashed uvicorn — port goes dead.
# Same shape as the F-011 silent-burn cases, just on a different param
# that F-011 didn't have a Field bound on.
#
# The systemic fix is one tiny helper used by every sampling-param
# validator on every route. Future contributors who add a new sampling
# param have a single place to wire it through (Field bound + finite
# check + scrub), so the next "new param leaks NaN to Metal kernel"
# bug class never reopens.
#
# Legal ranges (OpenAI spec + Anthropic spec + mlx-lm contract):
#
#   * temperature        OpenAI: [0, 2]   Anthropic: [0, 1]
#   * top_p              OpenAI: (0, 1]   Anthropic: (0, 1]
#   * min_p              OpenAI: [0, 1]
#   * top_k              OpenAI: >= 0     Anthropic: >= 0
#                        (mlx-lm: 0 == "disabled", > 0 keeps top-k tokens;
#                         negative is a no-op silent-ignore which M-14
#                         calls out, but the H-10 mandate is finite-range
#                         sweep, so we 4xx it here)
#   * repetition_penalty [0, 2]
#                        (mlx-lm requires >= 0; bug repro: -1.0 ValueError
#                         → server death. Upper bound is a safety cap that
#                         matches the most common convention across HF
#                         libraries — values much above 2 collapse the
#                         distribution to a single token, mathematically
#                         degenerate.)
#   * presence_penalty   OpenAI: [-2, 2]
#   * frequency_penalty  OpenAI: [-2, 2]
#   * logit_bias values  finite floats (no spec-stated range, but NaN/inf
#                        propagates through softmax and re-creates the
#                        F-011 silent-burn surface — H-10 closes it
#                        defensively even though the chat route already
#                        4xx's non-empty logit_bias).
def _validate_finite_in_range(
    v: float | None,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    field_name: str = "value",
) -> float | None:
    """One-stop sampling-param value guard (H-10).

    Pre-H-10 each Pydantic field declared its own ``Field(ge=, le=)``
    + ``field_validator`` finite check. Half the routes (Anthropic in
    particular) had neither, so a NaN ``temperature`` HTTP-200'd into a
    Metal kernel. Centralising the rule into a single helper makes the
    contract enforceable at code-review time: every sampling-param
    validator MUST call this with explicit bounds.

    Contract:
      * ``None`` passes through unchanged (preserves default-None
        semantics so untouched fields don't trip the gate).
      * ``NaN`` / ``±inf`` → ``ValueError`` (Pydantic 422 → 400).
      * Out-of-``[min, max]`` → ``ValueError`` (same envelope).
      * Inside the range → returned unchanged.

    The error message uses ``field_name`` so the project's unified
    validation-error handler can render a message that names the
    offending wire field exactly, matching the F-011 contract pinned
    by ``test_sampling_validation.py``.

    Booleans deliberately do NOT short-circuit here — Pydantic v2
    rejects ``bool`` on ``float`` fields when ``strict=True`` and
    coerces ``True`` → ``1.0`` otherwise, which is the historical
    contract preserved through F-011. The scrub layer
    (``_scrub_nonfinite_sampling_raw``) also skips bools, so this
    helper only ever sees post-coercion floats.
    """
    if v is None:
        return None
    if not math.isfinite(v):
        raise ValueError(f"{field_name} must be a finite number (not NaN or inf)")
    if min_value is not None:
        ok = v >= min_value if min_inclusive else v > min_value
        if not ok:
            comp = ">=" if min_inclusive else ">"
            raise ValueError(f"{field_name} must be {comp} {min_value} (got {v})")
    if max_value is not None:
        ok = v <= max_value if max_inclusive else v < max_value
        if not ok:
            comp = "<=" if max_inclusive else "<"
            raise ValueError(f"{field_name} must be {comp} {max_value} (got {v})")
    return v


# r5-E B-7 / F-DGF-V080-C-4: ``top_k`` upper-bound cap. Pre-fix,
# ``top_k=999999999`` and ``top_k=2**63-1`` slid through with HTTP 200
# even though no vocabulary in the catalog has more than ~256K entries
# (Gemma 3 sits at 262144; Qwen3 at 151936; Llama 3 at 128256). The
# request-time gate cannot consult the resolved model's vocab_size
# because route validation runs before model selection in the engine
# (and ``model`` is a free-form alias that may not even be loaded yet).
# We instead use a generous sentinel cap of 2**20 (~1 M tokens) — wider
# than any vocab we anticipate shipping while still rejecting the
# pathological 999M / 2^63 forms that produce silent-overflow inside
# mlx-lm's top-k kernel. The backend kernel does its own ``min(top_k,
# vocab_size)`` clamp at sample time, so a request with ``top_k=500_000``
# on a model with a 128K vocab still works exactly like ``top_k=128_000``;
# the cap here is purely a defence-in-depth ceiling against the obviously-
# invalid forms. ``top_k=0`` is documented as "disabled" by mlx-lm and
# is preserved as a legal value (the v0.8.2 validator's own error string
# read "use 0 to disable" — the by-design escape hatch).
_TOP_K_SENTINEL_CAP: int = 1 << 20  # 1_048_576


def _validate_nonnegative_int(
    v,
    *,
    max_value: int | None = None,
    field_name: str = "value",
):
    """Integer counterpart to ``_validate_finite_in_range`` (H-10).

    Used by ``top_k`` on every route. Integers don't carry NaN, so the
    finite check is moot — but the range check matters because mlx-lm
    silently ignores ``top_k < 0`` (M-14 also notes this) and
    Pydantic v2 would happily coerce ``top_k: -5`` onto an ``int | None``
    slot without complaint. H-10's sweep mandate says: every sampling
    param needs a range gate, so we 4xx the negative form rather than
    silent-accept it.

    Compatibility with Pydantic v2's lax int coercion:
      * ``True``/``False`` are explicitly rejected (``bool`` is an
        ``int`` subclass in Python; without this, ``top_k: true``
        would silently become ``top_k=1``). Mirrors
        ``_reject_non_one_n`` + ``_validate_thinking_budget``.
      * Integer-valued floats (e.g. ``64.0``) are accepted and
        coerced to ``int`` — Pydantic v2 lax mode also accepts these,
        and a pre-H-10 client sending ``{"top_k": 64.0}`` would have
        worked. Preserving that contract keeps H-10 a pure tightening
        (only rejects shapes the legacy path also rejected OR shapes
        F-011 / H-10 specifically targets), no new regressions.
      * NaN / ±inf floats are rejected for the same reason
        ``_reject_nonfinite_float`` rejects them on float fields:
        they propagate into the engine with mathematically-undefined
        behavior (kept out of the legal-int domain by the
        ``math.isfinite`` check).
      * Genuine non-numeric types (str, list, dict, ...) → 422.

    The mode="before" attachment lets this validator decide the
    type-coercion contract per field instead of inheriting Pydantic's
    default int-coercion (which silently turns ``True`` into ``1``).
    """
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer when set (got bool)")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{field_name} must be a finite integer (not NaN or inf)")
        if not v.is_integer():
            raise ValueError(f"{field_name} must be an integer when set (got {v})")
        v = int(v)
    elif not isinstance(v, int):
        # Defensive — Pydantic v2's int field already 422s on
        # arbitrary types, but mode="before" paths see the raw value.
        raise ValueError(
            f"{field_name} must be an integer when set (got {type(v).__name__})"
        )
    if v < 0:
        raise ValueError(f"{field_name} must be >= 0 (got {v}); use 0 to disable.")
    if max_value is not None and v > max_value:
        raise ValueError(f"{field_name} must be <= {max_value} (got {v})")
    return v


def _validate_positive_int(
    v,
    *,
    max_value: int | None = None,
    field_name: str = "value",
):
    """Integer with a strict ``>= 1`` lower bound (R7-M3 systemic gate).

    Used by the generation-budget fields — ``max_tokens`` (chat /
    completions / messages) and ``max_output_tokens`` (responses) — so
    every OpenAI-compat surface 4xx's negative / zero / bool / float
    wire forms at the schema layer instead of letting them slip
    through to the route. Pre-R7-M3 the validator coverage was
    asymmetric: the chat route's hand-rolled
    ``max_tokens must be at least 1`` check was the ONLY gate, so
    legacy ``/v1/completions``, Anthropic ``/v1/messages``, and
    OpenAI ``/v1/responses`` silently accepted ``max_tokens=-5`` (HTTP
    200 with one token / ``status="incomplete"``) — a silent-correctness
    hazard the same shape as ``seed=-1``.

    The validator's contract mirrors ``_validate_nonnegative_int``
    (same bool / float-integer-coercion / NaN-inf handling) but with
    a strict ``>= 1`` floor so ``max_tokens=0`` is rejected too —
    pointing at the right OpenAI escape hatch (``stream=true`` with
    no limit + finish_reason=stop is how clients ask for "unlimited
    tokens"; ``max_tokens=0`` has no spec meaning and was previously
    a silent no-token request).
    """
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer when set (got bool)")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{field_name} must be a finite integer (not NaN or inf)")
        if not v.is_integer():
            raise ValueError(f"{field_name} must be an integer when set (got {v})")
        v = int(v)
    elif not isinstance(v, int):
        raise ValueError(
            f"{field_name} must be an integer when set (got {type(v).__name__})"
        )
    if v < 1:
        raise ValueError(
            f"{field_name} must be >= 1 when set (got {v}); omit the field to "
            "use the server default."
        )
    if max_value is not None and v > max_value:
        raise ValueError(f"{field_name} must be <= {max_value} (got {v})")
    return v


def _validate_logit_bias_finite(
    v: dict | None, *, field_name: str = "logit_bias"
) -> dict | None:
    """Reject NaN/±inf inside ``logit_bias`` values (H-10).

    The chat route already 4xx's non-empty ``logit_bias`` (the
    server doesn't wire it through to the sampler yet), but the
    schema layer never inspected the *values* — so a defensively
    crafted ``{"42": NaN}`` would survive the Pydantic parse, get
    embedded in the ``ValidationError.input_value`` if a downstream
    validator failed, and recreate the F-011 NaN-in-error-body
    silent-500. H-10's sweep mandate is "every sampling param is
    finite-checked at the schema layer" — covering ``logit_bias``
    here means the same gate applies even if a future PR routes
    the field through to a real logits processor.

    Returns ``None`` unchanged (defaults preserved). An empty dict
    is fine — it's the spec's "no bias applied" form.
    """
    if v is None:
        return None
    if not isinstance(v, dict):
        # Pydantic's type-check normally catches this first, but
        # this validator may run in ``mode="before"`` paths so the
        # explicit guard keeps the error message coherent.
        raise ValueError(f"{field_name} must be a mapping of token-id (str) → float.")
    for _k, vv in v.items():
        # D-ENVELOPE-FIELD-LEAK sibling: the original error messages
        # embedded ``{k!r}`` — the dict key the caller supplied —
        # straight into the ValueError text. Since the canonical
        # envelope drops ``input`` but DOES surface ``msg``, that
        # reflected attacker-controlled bytes (``{"AWS_SECRET":
        # NaN}`` → ``logit_bias['AWS_SECRET'] must be a finite
        # number``). The schema-walker in
        # :mod:`vllm_mlx.middleware.exception_handlers` only sanitizes
        # the ``loc`` path; bytes a validator chooses to embed in
        # ``msg`` aren't covered. So we drop the key from the message
        # entirely — the field name + value type is sufficient context
        # for the caller to fix their request.
        if isinstance(vv, bool):
            raise ValueError(f"{field_name} values must be finite numbers (got bool).")
        if not isinstance(vv, (int, float)):
            raise ValueError(
                f"{field_name} values must be finite numbers (got {type(vv).__name__})."
            )
        if not math.isfinite(vv):
            raise ValueError(
                f"{field_name} values must be finite numbers (not NaN or inf)."
            )
    return v


def _validate_seed(v) -> int | None:
    """Reject bool / non-integer / negative values on ``seed`` (H-11 + r5-E B-8).

    OpenAI's spec documents ``seed`` as an integer; their Python SDK
    ships ``seed: Optional[int]`` with no documented range, but every
    real-world usage in their docs and examples uses non-negative
    integers (the canonical example is ``seed=12345``). The mlx-core
    backend key is a ``uint32``: a negative seed has no natural
    meaning in that space and ``-1`` is almost always a sentinel from a
    misbehaving SDK (Python ``random.randint(0, sys.maxsize)`` returning
    ``-1`` on overflow, ``np.int32(-1)`` from a probe). Accepting it
    silently and then bit-folding via ``& 0xFFFFFFFF`` maps ``-1`` →
    ``0xFFFFFFFF`` (a perfectly valid backend key) — which means the
    user thinks they passed a sentinel but actually pinned a fixed PRNG
    state. That mismatch is the exact silent-correctness hazard
    r5-E B-8 / cycle-DGF-V080-V2 surfaced; we close it with a 400.

    What we reject at the request layer:

      * ``bool`` — Pydantic v2 silently coerces ``True`` → 1 / ``False``
        → 0 on an ``int | None`` field, but ``seed=True`` is almost
        always a serialization mistake (same family as the ``n: true``
        footgun ``_reject_non_one_n`` already closes). A
        ``bool``-as-seed silent acceptance would mask client bugs.
      * Non-integer types — strings, floats, dicts. The OpenAI spec
        is unambiguous that ``seed`` is an integer; a float seed is
        a serialization bug, not a legitimate request.
      * Negative integers — see the rationale block above. ``seed=0``
        is preserved as a legal value (a fixed PRNG key, common in
        evaluation harnesses).

    Upper bound is unbounded by design: 64-bit / very-large seeds are
    accepted and folded to the backend's uint32 PRNG key range in
    ``make_seeded_sampler`` via a deterministic ``seed & 0xFFFFFFFF``
    bit-fold. Same input value always maps to the same backend key, so
    reproducibility is preserved within rapid-mlx (OpenAI's spec only
    promises within-engine determinism — "we cannot guarantee
    determinism across model versions or backends"). Codex round-6
    BLOCKING fix on the original ``le=0xFFFFFFFF`` Field bound that
    422'd valid OpenAI seed values.

    Returns ``None`` unchanged so the field's optional contract is
    preserved (no value → no per-request seed → process-global
    behaviour, matching the OpenAI spec where omitted ``seed`` means
    "no determinism guarantee").
    """
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("seed must be an integer (not bool)")
    if not isinstance(v, int):
        raise ValueError("seed must be an integer")
    # r5-E B-8: reject negative seeds at the request layer. The
    # downstream ``& 0xFFFFFFFF`` fold maps ``-1`` to ``0xFFFFFFFF``
    # (a valid uint32 key), which silently pins a fixed PRNG state
    # for a caller that thought they were passing a sentinel.
    # ``seed=0`` is a legitimate PRNG key (used by every eval
    # harness) and is preserved.
    if v < 0:
        raise ValueError(
            f"seed must be >= 0 (got {v}); use a non-negative integer "
            "or omit the field for no determinism guarantee."
        )
    return v


# R10-H5 (R9-H3 carry) — closed set of values accepted by OpenAI's
# ``reasoning_effort`` parameter. Pre-fix BOTH ``/v1/chat/completions``
# and ``/v1/responses`` silently accepted any value (string, int, list,
# null, garbage like ``"banana"``) with HTTP 200 — the field was
# undeclared on the request schema, so Pydantic dropped it before the
# adapter or postprocessor saw it (sven r10-R1 + vlad r10-R1). That
# means clients passing ``reasoning_effort="none"`` to suppress thinking
# (IDE assistants on low-token budgets) got the silent no-op + 32-token
# reasoning_tokens regression Sven called out as R10-CRIT3. Declaring
# the field with this closed-set validator now means:
#
#   * valid value → flows through to the request body for downstream
#     consumers (today still a no-op at the engine layer; future work
#     translates it into ``reasoning_max_tokens`` headroom);
#   * invalid value → clean 400 with the supported set listed.
#
# The set mirrors OpenAI's public docs (gpt-5/o-series spec): ``minimal``
# was added Aug 2025 alongside ``low``/``medium``/``high``; ``none`` is
# the rapid-mlx-specific "suppress thinking entirely" alias that the
# desktop UI surfaces, kept here so SDK clients that learned it don't
# 400. Anthropic Claude's ``thinking.type`` enum is intentionally NOT
# included — that's a different surface (the Anthropic adapter handles
# its own translation).
_VALID_REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
)


def _validate_reasoning_effort(v):
    """Pin ``reasoning_effort`` to the OpenAI-spec closed set.

    Accepts ``None`` (the field's default — no preference signalled)
    unchanged. Any other shape (int, list, dict, ``True``/``False``,
    case-variant strings, garbage strings) raises ``ValueError`` so
    Pydantic surfaces a 400. The route layer is free to translate the
    accepted string into a ``reasoning_max_tokens`` cap or a system-
    prompt hint — that translation lives outside the schema layer.

    Codex-pattern note: bool is a subclass of int, so the ``isinstance(v,
    bool)`` check has to run BEFORE the string check (no risk here
    because the string check is first in the chain). The error message
    enumerates the legal set so SDK consumers see exactly what to send.
    """
    if v is None:
        return None
    if isinstance(v, str) and v in _VALID_REASONING_EFFORTS:
        return v
    raise ValueError(
        "reasoning_effort must be one of "
        f"{list(_VALID_REASONING_EFFORTS)} or null "
        f"(got {type(v).__name__}={v!r})."
    )


# Fields that must reject NaN / ±inf BEFORE pydantic coerces them onto a
# typed ``float | None`` slot. Pydantic v2's default ``ValidationError``
# embeds ``input_value`` in the error dict; when the bad value is
# ``float('nan')`` the downstream ``starlette.JSONResponse`` (which uses
# stdlib ``json.dumps`` with ``allow_nan=False``) crashes mid-serialize
# with ``ValueError: Out of range float values are not JSON compliant``
# — meaning the client gets a 500 instead of a 422. We MUST sanitize
# the raw dict before Pydantic ever sees the float so the error path
# never carries a NaN payload. See F-011.
_FINITE_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)


_NONFINITE_PLACEHOLDER = "<non-finite>"


def _scrub_nonfinite_sampling_raw(data):
    """Replace NaN / ±inf sampling-param values with a JSON-safe
    placeholder in the raw request dict, then raise ``ValueError``
    so Pydantic emits a clean ``ValidationError``.

    The mutate-then-raise dance matters because Pydantic captures the
    raw ``data`` reference as ``input_value`` on the resulting
    ``ValidationError`` — if NaN remains in the dict the downstream
    ``starlette.JSONResponse`` (which uses ``json.dumps`` with
    ``allow_nan=False`` by default) crashes serializing the error
    body and the client sees a 500 instead of a 422. Replacing the
    bad value with a string sentinel keeps the captured error body
    JSON-safe AND preserves enough context for an operator reading
    the log to see which field was bad. The production
    ``_validation_error_response`` handler strips ``input`` from the
    rendered body anyway (F-094/F-104), but this codepath also has
    to survive a future cleanup that re-enables FastAPI's default
    422 handler — codex round-1 BLOCKING #1.

    Wire forms covered:
      * raw JSON token ``NaN`` / ``Infinity`` / ``-Infinity`` (parsed
        by Python's stdlib json decoder, which is non-strict by
        default — every popular OpenAI client SDK plus ``requests``
        emits these on the wire when the caller passes
        ``float('nan')``).
      * String form ``"NaN"`` / ``"Infinity"`` / ``"-Infinity"``
        (clients that defensively pre-stringify floats — the bug
        repro under F-011 uses this form).

    The string form is matched case-insensitively against the JSON
    spec tokens with an optional sign prefix; any other string flows
    through untouched so Pydantic still emits its native
    ``float_parsing`` 422 for ``temperature: "hot"``.
    """
    if not isinstance(data, dict):
        return data
    bad_field: str | None = None
    for field in _FINITE_SAMPLING_FIELDS:
        if field not in data:
            continue
        v = data[field]
        if v is None:
            continue
        if isinstance(v, bool):
            # bool is an int subclass — let Pydantic's native rules
            # decide (it currently coerces True/False to 1.0/0.0 onto
            # ``float | None``; that's the historical contract and
            # outside F-011's scope).
            continue
        if isinstance(v, (int, float)):
            if not math.isfinite(v):
                data[field] = _NONFINITE_PLACEHOLDER
                bad_field = bad_field or field
            continue
        if isinstance(v, str):
            stripped = v.strip().lower().lstrip("+-")
            if stripped in ("nan", "inf", "infinity"):
                data[field] = _NONFINITE_PLACEHOLDER
                bad_field = bad_field or field
    if bad_field is not None:
        raise ValueError(f"{bad_field} must be a finite number (not NaN or inf)")
    return data


# =============================================================================
# Optional generation-budget ceiling (M-04 — opt-in)
# =============================================================================
#
# Aanya (round-2 dogfooding) demonstrated that the F-007 body-bytes cap
# does not constrain ``max_tokens``: a small JSON body (e.g. 5K-token
# system prompt + ``max_tokens=10000``) sails through the body-size
# middleware because the wire bytes are well under the 8 MiB cap, yet
# the multi-tenant cost surface is determined by the *generation budget*
# the request claims, not the body size. A request asking for 10K
# output tokens on a model that costs $X/1K tokens is a cost vector
# regardless of how few bytes the request body weighs.
#
# This validator adds an OPT-IN ceiling on ``max_tokens`` so multi-tenant
# operators can cap the per-request generation budget without affecting
# single-machine ("Yuki posture") users who never set the env var.
#
# Contract:
#   * ``RAPID_MLX_MAX_GENERATION_TOKENS`` unset / blank / non-positive
#     → no enforcement. Identical behaviour to pre-fix; the body-bytes
#     cap remains the only generic DoS gate.
#   * ``RAPID_MLX_MAX_GENERATION_TOKENS=N`` with positive int N → reject
#     at parse time when ``max_tokens`` (after the
#     ``max_completion_tokens``/``max_tokens`` normalization on the
#     OpenAI side) exceeds N.
#
# Enforcement is intentionally narrow:
#   * Does NOT add a ``prompt_tokens + max_tokens <= context_window``
#     early-reject. Context window varies per model and that check
#     belongs in the engine (where the tokenizer is loaded). The cap
#     here is purely a *budget ceiling* — a hard policy lever an
#     operator can set without coupling to model state.
#   * Reads the env var at validator time (not module-import time) so
#     a test/operator can flip the ceiling without restarting the
#     process. Cost is one ``os.environ.get`` per request — negligible.


_MAX_GEN_TOKENS_ENV = "RAPID_MLX_MAX_GENERATION_TOKENS"


def _resolve_max_generation_tokens_ceiling() -> int | None:
    """Return the configured ceiling, or ``None`` when unset / invalid.

    Read at validator time so tests can monkeypatch the env var without
    re-importing the module. Invalid values (non-int, ``0``, negative)
    are treated the same as ``unset`` — the opt-in stays opt-in even
    when an operator types a typo, and we never raise from here (the
    request validator stays the only error surface).
    """
    import os

    raw = os.environ.get(_MAX_GEN_TOKENS_ENV)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        ceiling = int(raw)
    except (TypeError, ValueError):
        return None
    if ceiling <= 0:
        return None
    return ceiling


def _enforce_max_generation_tokens_ceiling(max_tokens: int | None) -> None:
    """Raise ``ValueError`` if ``max_tokens`` exceeds the opt-in ceiling.

    ``None`` is always accepted — an absent ``max_tokens`` falls back to
    a route-level default that is itself bounded by the loaded model's
    ``max_tokens`` (see ``load_model`` in ``server.py``). The ceiling
    only applies when the caller *explicitly* asks for a large budget.
    """
    if max_tokens is None:
        return
    ceiling = _resolve_max_generation_tokens_ceiling()
    if ceiling is None:
        return
    if max_tokens > ceiling:
        raise ValueError(
            f"max_tokens ({max_tokens}) exceeds the per-request generation "
            f"budget ceiling of {ceiling} configured via the "
            f"{_MAX_GEN_TOKENS_ENV} environment variable. Lower max_tokens "
            f"or raise the ceiling on the server."
        )


# =============================================================================
# Content Types (for multimodal messages)
# =============================================================================


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""

    url: str
    detail: str | None = None


class VideoUrl(BaseModel):
    """Video URL."""

    url: str


class AudioUrl(BaseModel):
    """Audio URL for audio content."""

    url: str


class ContentPart(BaseModel):
    """
    A part of a multimodal message content.

    Supports:
    - text: Plain text content
    - image_url: Image from URL or base64
    - video: Video from local path
    - video_url: Video from URL or base64
    - audio_url: Audio from URL or base64
    """

    type: str  # "text", "image_url", "video", "video_url", "audio_url"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None
    audio: str | None = None
    audio_url: AudioUrl | dict | str | None = None
    input_audio: dict | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text_type(cls, value):  # noqa: ANN001, ANN206
        """Reject non-string ``text`` on a content part with a clean,
        field-named error message (H-15).

        ``text`` is declared ``str | None`` so direct ContentPart
        construction already 422s on a non-string value — but Pydantic's
        default ``string_type`` message buries ``text`` under a nested
        loc trail. Run a ``mode="before"`` field validator so the error
        surfaces a single actionable line naming ``content[].text``.

        The dict-fallback escape (where ``Message.content`` falls back
        to ``list[dict]`` because the ``list[ContentPart]`` arm rejected)
        is pinned separately at the parent ``Message`` validator — see
        ``Message._validate_media_url_types`` below. That's the path
        that actually escaped to ``_join_text_parts`` pre-H-15 (and
        500'd with raw ``TypeError``); the validator here keeps direct
        ``ContentPart(...)`` construction (engine tests, the gradio
        app, the speculative server) covered as well.
        """
        if value is None or isinstance(value, str):
            return value
        raise ValueError(
            f"content[].text must be a string (got {type(value).__name__})"
        )

    @model_validator(mode="after")
    def _validate_media_url_types(self) -> "ContentPart":
        """Reject non-string ``url`` inside multimodal-content dicts (F-066)
        AND reject the bare-string ``image_url`` shorthand when the part
        ``type`` advertises a media payload (F-065).

        F-066 covered: ``image_url: dict`` with a non-string ``url`` slot
        used to fall through the schema layer and crash inside
        ``process_image_input`` with ``'int' object has no attribute
        'startswith'``. The dict-arm check below pins that.

        F-065 covered: the wire form
        ``{"type":"image_url","image_url":"data:image/png;base64,..."}``
        (bare string instead of the OpenAI-spec
        ``{"url":"data:image/png;base64,..."}`` object) used to pass the
        schema layer because the ``image_url`` union allowed a plain
        ``str``. Downstream multimodal preprocessors then unwrapped the
        dict-shape via ``image["url"]`` but had no fallback for the
        bare-string form — on a text-only model the request 400'd at
        preprocess time with a vague "model does not support image"
        message, and on a multimodal model the image was silently
        dropped (model received only the text and hallucinated
        "image is blank"). Both surfaces are silent-correctness bugs
        OpenAI-compat clients (the official SDK + LangChain serialize
        the spec-shape, but a hand-rolled payload OR an SDK that
        flattens the object to a string before posting hits this).
        Reject at the schema layer with a clean 422 so the client
        sees the actual mismatch.

        Mirror the same rule on ``video_url`` and ``audio_url`` which
        share the same OpenAI-spec object shape — same silent-drop
        hazard if the bare-string shorthand were allowed on those
        downstream parsers as well.
        """
        for field, label in (
            ("image_url", "image_url"),
            ("video_url", "video_url"),
            ("audio_url", "audio_url"),
        ):
            value = getattr(self, field, None)
            if value is None:
                continue
            # F-065: bare-string shorthand is NOT the OpenAI-spec shape.
            # Gate it on ``type`` so a content part with
            # ``type:"text"`` that happens to carry an unrelated
            # ``image_url:"..."`` slot (legacy / hand-rolled clients)
            # is not collaterally broken — the validator only fires
            # when the part is ACTUALLY advertising itself as that
            # media type. Without this gate, code that flattens all
            # parts through a generic ``ContentPart(**raw)`` could see
            # surprise rejections.
            if isinstance(value, str) and self.type == field:
                raise ValueError(
                    f"{label} must be an object with required field "
                    f"'url' (got bare string; OpenAI-spec shape is "
                    f'`{label}: {{"url": "..."}}`)'
                )
            # F-066: dict-arm non-string ``url`` slot.
            if isinstance(value, dict) and "url" in value:
                url = value["url"]
                if url is not None and not isinstance(url, str):
                    raise ValueError(f"{label}.url must be a string")
        return self


# =============================================================================
# Messages
# =============================================================================


class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Multimodal messages (role + content list with text/images/videos)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """

    role: str
    content: str | list[ContentPart] | list[dict] | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _validate_content_not_null(self) -> "Message":
        """Reject ``content: null`` on roles where the OpenAI spec
        requires content (R15 #175 item 3 / fuzz a9c828).

        OpenAI chat-completions spec: ``content`` is REQUIRED for
        ``role:"user" | "system" | "developer"``. Only ``role:"assistant"``
        may carry ``content=null`` — and only when ``tool_calls`` is
        present (the model's response is the call, not text).
        ``role:"tool"`` is policed by the route-level F-111 validator
        which already accepts ``None`` and normalises to ``""``
        downstream; we preserve that contract here.

        Pre-fix the schema accepted any of {None, str, list[ContentPart],
        list[dict]} for every role. A user message with ``content=null``
        returned HTTP 200 with garbled output (the chat-template
        renderer flattened ``None`` to the literal string ``"None"`` or
        skipped the turn outright, depending on the model). Adversarial
        fuzz a9c828 reproduced it on Qwen3-0.6B; same shape silently
        passed on every other model.

        Empty list ``[]`` and empty string ``""`` are both accepted —
        multimodal user turns can legitimately send ``content=[]`` (e.g.
        the only relevant part is sourced via a tool the model called
        on the previous turn). The engine decides how to render an
        empty-list turn; the schema layer only rejects the ``null``
        shape that the fuzz exposed.
        """
        if self.content is not None:
            return self
        # ``content is None`` past this point.
        #
        # Allow assistant ``content=null`` regardless of ``tool_calls``:
        # the OpenAI spec technically requires ``tool_calls`` to license
        # the null, but many existing clients (and the response shape
        # of THIS server when it emits AssistantMessage back into a
        # follow-up request) send assistant turns with null content
        # without tool_calls. Tightening that path is out of scope for
        # the #175 fix — the fuzz only exposed the user-role case.
        if self.role == "assistant":
            return self
        # ``role:"tool"`` content = None is normalised by the route's
        # F-111 validator to an empty string. Preserve that path so we
        # don't double-reject what an existing layer already cleans up.
        if self.role == "tool":
            return self
        raise ValueError(
            "content must be a string or a content-parts array, not null "
            f"(role={self.role!r}); send '' for an empty turn or [] "
            "for an empty multimodal turn"
        )

    @model_validator(mode="after")
    def _validate_media_url_types(self) -> "Message":
        """Reject non-string ``url`` inside multimodal-content payloads
        (F-066).

        The ``content`` union is ``list[ContentPart] | list[dict]`` —
        Pydantic falls back to ``list[dict]`` when ContentPart-level
        validation raises (e.g. ``image_url.url=123`` makes the inner
        ``ImageUrl`` model reject, the ``dict`` arm accepts the raw
        shape, and ``list[ContentPart]`` then fails the whole list so
        Pydantic picks ``list[dict]``). The malformed payload used to
        slip past the schema layer and crash deep inside
        ``process_image_input`` with ``'int' object has no attribute
        'startswith'`` — raw Python type-error text leaked verbatim in
        the HTTP 400 body. Run the same check at the parent Message so
        the dict-fallback path is also covered (the inner ContentPart
        validator handles the ``list[ContentPart]`` arm — kept there so
        direct ``ContentPart(...)`` construction in tests is also
        protected).
        """
        if not isinstance(self.content, list):
            return self
        for item in self.content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            # H-15: dict-arm non-string ``text`` slot. The
            # ``list[ContentPart] | list[dict]`` union falls back to
            # ``list[dict]`` when the ContentPart arm rejects (e.g. a
            # nested-list or dict ``text`` value fails the
            # ``str | None`` declaration on ``ContentPart.text``), so
            # the malformed payload used to slip past the schema layer
            # and crash inside ``_join_text_parts`` with raw
            # ``TypeError: sequence item 0: expected str instance, X
            # found`` (HTTP 500). Pin it at this dict-fallback path so
            # the client sees a clean 400 naming ``content[].text``.
            if "text" in item:
                text_value = item["text"]
                if text_value is not None and not isinstance(text_value, str):
                    raise ValueError(
                        "content[].text must be a string "
                        f"(got {type(text_value).__name__})"
                    )
            for field in ("image_url", "video_url", "audio_url"):
                value = item.get(field)
                if value is None:
                    continue
                # F-065: bare-string ``image_url`` (etc.) on a part
                # whose ``type`` advertises that media slot. Gated by
                # ``type`` for the same back-compat rationale as the
                # ContentPart-level validator.
                if isinstance(value, str) and item_type == field:
                    raise ValueError(
                        f"{field} must be an object with required "
                        f"field 'url' (got bare string; OpenAI-spec "
                        f'shape is `{field}: {{"url": "..."}}`)'
                    )
                # F-066: dict-arm non-string ``url`` slot.
                if isinstance(value, dict) and "url" in value:
                    url = value["url"]
                    if url is not None and not isinstance(url, str):
                        raise ValueError(f"{field}.url must be a string")
        return self


# =============================================================================
# Tool Calling
# =============================================================================


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str
    type: str = "function"
    function: FunctionCall


# F-035 / F-146: OpenAI function-name spec — non-empty, ≤64 chars,
# ASCII alphanumerics + ``_`` + ``-``. Same regex Anthropic and OpenAI
# publish in their tool-definition schemas. Defined at module scope so
# the request-level validator below and any future direct
# ``ToolDefinition`` consumers share one source of truth. A single
# constraint covers the whole space the F-035 / F-146 fuzz exposed
# (empty / emoji / 10000-char / shell-metachar names all rejected).
_FUNCTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# R10-H6 (R9-H4 carry) — accepted ``type`` aliases for the Computer-Use
# tool surface on the chat lane. Mirrors the Responses-lane alias map
# (``vllm_mlx/api/responses_adapter.py``) so the SDK-shorthand
# ``tools=[{"type":"computer_use"}]`` works on /v1/chat/completions
# too — pre-fix only /v1/responses normalised the alias and chat
# 400'd with ``tools.0.function: Field required`` (Mira r10-R1
# R10-HIGH3, vlad r10-R1 R10-HIGH3). The canonical Computer-Use shape
# is a synthetic ``function`` tool named ``computer`` so the UI-TARS
# tool parser (which always emits ``function.name == "computer"``)
# sees a matching entry; ``_synthesize_computer_use_function`` runs
# the same translation the Responses adapter applies at
# ``_convert_tools``.
_COMPUTER_USE_TYPE_ALIASES: frozenset[str] = frozenset(
    {
        "computer_use",
        "computer_use_preview",
        "computer_20251022",
    }
)


def _synthesize_computer_use_function(tool_data: dict) -> dict:
    """Build the canonical ``function`` dict for a Computer-Use shorthand
    tool entry. Mirror of the Responses-lane translation in
    ``responses_adapter._convert_tools`` so downstream chat-route code
    sees ONE shape regardless of which alias the client sent.

    ``display_width`` / ``display_height`` / ``environment`` hints are
    placed under ``_computer_use`` on the parameters dict — the chat
    template / system-prompt builder picks them up to ground the model
    on the actual screen geometry, matching the Responses lane.
    """
    geometry: dict = {
        "type": "object",
        "properties": {
            "display_width": {"type": "integer"},
            "display_height": {"type": "integer"},
            "environment": {"type": "string"},
        },
        "_computer_use": {
            "display_width": tool_data.get("display_width"),
            "display_height": tool_data.get("display_height"),
            "environment": tool_data.get("environment"),
        },
    }
    return {
        "name": "computer",
        "description": "Computer-Use (UI-TARS) GUI action tool",
        "parameters": geometry,
    }


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    # R10-H6: ``function`` is optional at the wire level so the
    # Computer-Use shorthand (``{"type":"computer_use_preview"}``)
    # parses. The ``_normalize_computer_use_shorthand`` model_validator
    # below synthesises the canonical function dict when ``type`` is
    # a Computer-Use alias, so downstream code (``ToolDefinition.function``,
    # parser, chat-route) reads exactly one shape. For the canonical
    # ``type == "function"`` path, the function field is still required —
    # the ``_validate_function_name`` validator below raises when it's
    # missing/empty (preserving the F-035 contract).
    function: dict | None = None

    # R10-H6 (R9-H4 carry) — Computer-Use shorthand normalisation.
    # Runs BEFORE ``_validate_function_name`` so the synthetic
    # ``function`` dict the F-035 regex check sees is the canonical
    # ``computer`` shape, not the missing-field shape the client sent.
    # ``mode="before"`` would require returning a dict; the
    # ``model_validator(mode="after")`` style would re-run after
    # ``_validate_function_name`` 422'd. The cleanest split is a
    # ``mode="before"`` raw-dict normaliser PLUS the canonical-path
    # name validator below — see test_tool_definition_computer_use_shorthand
    # in tests/test_chat_route_tool_choice_enforcement.py for the
    # parity matrix.
    @model_validator(mode="before")
    @classmethod
    def _normalize_computer_use_shorthand(cls, data):
        if not isinstance(data, dict):
            return data
        ttype = data.get("type")
        if (
            isinstance(ttype, str)
            and ttype in _COMPUTER_USE_TYPE_ALIASES
            and not data.get("function")
        ):
            # Canonicalise the type so downstream readers (engine
            # plumbing, postprocessor, finalize path) see exactly
            # ``"function"`` — mirrors the Responses adapter's
            # ``_canonicalize_tool_type`` + ``_convert_tools`` pair.
            data = dict(data)
            data["type"] = "function"
            data["function"] = _synthesize_computer_use_function(data)
        return data

    # F-035 / F-146: reject malformed ``function.name`` at the schema
    # layer. Pre-fix:
    #   * ``name=""`` (F-035) - request 200'd; on hermes-parser models
    #     the model emitted ``<tool_call>{"name":"",...}</tool_call>``
    #     literally into ``content`` because the tool-call detector
    #     keyed off a non-empty name field.
    #   * ``name`` containing shell metacharacters / newlines (F-146)
    #     - silently passed to the chat template; downstream parsers
    #     either dropped the call or routed garbage into the tool-name
    #     slot.
    #   * ``name`` ≥10 KB (F-146) - accepted; ballooned the prompt and
    #     wasted the context window.
    # ``function`` is typed ``dict`` (legacy shape; the field accepts
    # arbitrary OpenAI extensions) so the Pydantic v2 ``pattern=`` lever
    # on ``Field`` isn't directly available - validate manually here.
    @model_validator(mode="after")
    def _validate_function_name(self) -> "ToolDefinition":
        name = self.function.get("name") if isinstance(self.function, dict) else None
        if (
            name is None
            or not isinstance(name, str)
            or not _FUNCTION_NAME_PATTERN.match(name)
        ):
            # D-ENVELOPE-FIELD-LEAK sibling: do NOT embed ``{name!r}``
            # — the regex pattern accepts identifier-shaped bytes
            # (``AWS_SECRET_ACCESS_KEY`` matches ``^[a-zA-Z0-9_-]{1,64}$``)
            # which is the exact H-17 round-2 attack vector for a
            # rejected-name reflection. The error message gives the
            # rule; the operator log captures the literal value if
            # needed.
            raise ValueError(
                "function.name must be a non-empty string of 1-64 characters "
                "matching ^[a-zA-Z0-9_-]{1,64}$ (per OpenAI spec)."
            )
        # D-TOOL-RECUR: reject a ``function.parameters`` (JSON Schema)
        # whose structural nesting exceeds ``RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH``
        # (default 64). Pre-fix, a client could ship a ~1000-deep
        # nested ``parameters`` schema (~10 KB JSON) and crash the
        # worker with HTTP 500 inside
        # ``vllm_mlx.utils.chat_template._sanitize_tools_for_template._walk``
        # — an unauthenticated DoS surface confirmed against five
        # tool-call parsers (qwen / hermes / phi / deepseek / glm47).
        # The iterative walk in ``_sanitize_tools_for_template`` is the
        # structural fix for the crash; this validator is the upstream
        # rejection so the canonical 400 envelope (no stack trace, no
        # leaked tokenizer state) is what the client actually sees,
        # and so the iterative walk doesn't even have to traverse an
        # adversarial-sized tree.
        from ..utils.json_depth import (
            json_nesting_depth_exceeds,
            resolve_max_tool_schema_depth,
        )

        params = (
            self.function.get("parameters") if isinstance(self.function, dict) else None
        )
        if isinstance(params, (dict, list)):
            cap = resolve_max_tool_schema_depth()
            if cap > 0 and json_nesting_depth_exceeds(params, cap):
                raise ValueError(
                    f"function.parameters JSON nesting depth exceeds the "
                    f"{cap}-level server cap (set via "
                    "RAPID_MLX_MAX_TOOL_SCHEMA_DEPTH)."
                )
        return self


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    # R7-B codex MED follow-up: ``StrictBool`` so non-bool wire forms
    # like ``"strict":"true"`` are rejected at parse time instead of
    # being coerced or silently treated as falsey by downstream
    # ``is_strict_json_schema``. Paired with the explicit dict-arm
    # check in ``_validate_response_format_raw`` so the bare-dict
    # union arm can't bypass either.
    strict: StrictBool | None = False

    class Config:
        populate_by_name = True


# F-103: legal ``response_format.type`` enum, kept in sync with the
# route-layer ``_VALID_RESPONSE_FORMAT_TYPES`` in
# ``vllm_mlx/service/helpers.py``. Defined at module scope so both
# the typed ``ResponseFormat`` Pydantic model and the request-level
# raw-dict guard share one source of truth.
_VALID_RESPONSE_FORMAT_TYPES: tuple[str, ...] = (
    "text",
    "json_object",
    "json_schema",
)


def _validate_response_format_raw(value):
    """Reject malformed ``response_format`` dict payloads BEFORE the
    typed ``ResponseFormat`` Pydantic arm is reached (F-103).

    ``ChatCompletionRequest.response_format`` is declared as
    ``ResponseFormat | dict | None`` so OpenAI-compat clients can send
    the spec-shape directly without our typed model rejecting unknown
    extension keys. The downside is Pydantic's Union coercion picks
    the FIRST arm that parses — so when the typed arm rejects (e.g.
    ``json_schema.schema=42`` because ``ResponseFormatJsonSchema``
    declares ``schema_: dict``), the bare-``dict`` arm silently
    swallows the same payload. That was the F-103 silent-200 hazard:

      * ``{"type":"json_schema","json_schema":{"name":"x",
         "schema":42}}`` (int) → fell through to the dict arm, then
         ``extract_json_schema_for_guided`` bailed at
         ``if not schema: return None`` (HTTP 200 with no constraint).
      * Same with ``"schema":"hello"`` (string truthy → schema was
         coerced into a string literal in the system prompt;
         downstream JSON parsing produced garbage).

    The route-layer ``_validate_response_format`` helper closes the
    ``type``-enum and ``json_schema``-required arms; this validator
    pins the inner ``json_schema.schema`` shape so the silent-200
    arm becomes a clean 400 at parse time. Kept in models.py (not the
    route helper) so the gate fires before any FastAPI dependency runs
    — the resulting ``ValidationError`` surfaces as a Pydantic 400
    with a structured ``detail`` body the existing
    ``_validation_error_response`` handler already strips of
    ``input``.
    """
    # ``None`` flows through (default — no structure enforcement).
    if value is None:
        return value
    # If the value already parses as the typed model, the typed arm's
    # own validators cover everything we need — pass through.
    if isinstance(value, ResponseFormat):
        return value
    # Non-dict wire forms (string / list / int) — reject with a clean
    # message; Pydantic's default union-fallback would otherwise
    # produce a misleading error pointing at the wrong arm.
    if not isinstance(value, dict):
        raise ValueError("response_format must be an object with a 'type' field")
    # From here we're on the dict arm. Mirror the route-layer enum
    # rules + add the missing inner-``schema`` shape check.
    if "type" not in value:
        raise ValueError("response_format.type is required")
    rf_type = value.get("type")
    if rf_type not in _VALID_RESPONSE_FORMAT_TYPES:
        raise ValueError(
            "response_format.type must be 'text', 'json_object', or 'json_schema'"
        )
    # R7-B codex MED follow-up: ``strict`` is a strict boolean on BOTH
    # nesting positions. Pre-fix, ``{"type":"json_schema","strict":"true",
    # "json_schema":{...}}`` and the nested form
    # ``{"json_schema":{"strict":"true",...}}`` fell through the
    # bare-dict union arm with no strict check — ``is_strict_json_schema``
    # then treated the truthy string as falsey and the ``[guided]`` gate
    # silently downgraded the request to unconstrained generation. The
    # typed arm uses ``StrictBool`` so the same wire forms parse-fail
    # there; mirror the same wire-form rejection on the dict arm so the
    # ``ResponseFormat | dict`` union has no escape hatch.
    if "strict" in value and not isinstance(value["strict"], bool):
        raise ValueError(
            "response_format.strict must be a boolean "
            f"(got {type(value['strict']).__name__})"
        )
    if rf_type == "json_schema":
        json_schema_field = value.get("json_schema")
        if not json_schema_field:
            raise ValueError(
                "response_format.type='json_schema' requires "
                "non-empty 'json_schema' field"
            )
        if not isinstance(json_schema_field, dict):
            raise ValueError("response_format.json_schema must be an object")
        if "strict" in json_schema_field and not isinstance(
            json_schema_field["strict"], bool
        ):
            raise ValueError(
                "response_format.json_schema.strict must be a boolean "
                f"(got {type(json_schema_field['strict']).__name__})"
            )
        # Mirror the typed ``ResponseFormatJsonSchema.name: str``
        # required field on the dict arm — codex round-1 BLOCKING
        # follow-up. The bare-dict union arm would otherwise swallow
        # ``{"json_schema":{"schema":{...}}}`` (no ``name``) and the
        # downstream guided-generation extractor skips its lookup
        # silently, producing an unconstrained 200.
        name_field = json_schema_field.get("name")
        if not name_field or not isinstance(name_field, str):
            raise ValueError(
                "response_format.json_schema.name is required and must be a string"
            )
        inner_schema = json_schema_field.get("schema")
        if inner_schema is None:
            # Missing inner ``schema`` member; the route-layer helper
            # already covers this, but mirroring here keeps the schema
            # arm self-contained.
            raise ValueError(
                "response_format.type='json_schema' requires "
                "'json_schema.schema' to be a non-empty object"
            )
        if not isinstance(inner_schema, dict):
            # F-103 silent-200 closer: ``schema:42`` / ``schema:"hello"``
            # / ``schema:[1,2]`` previously fell through the bare-dict
            # union arm and produced HTTP 200 with no JSON-Schema
            # enforcement. Now a clean Pydantic ``ValidationError``
            # naming the actual type the client sent.
            raise ValueError(
                "response_format.json_schema.schema must be an object "
                f"(got {type(inner_schema).__name__})"
            )
        if not inner_schema:
            # Empty dict — also unconstrained, same hazard class.
            raise ValueError(
                "response_format.type='json_schema' requires "
                "'json_schema.schema' to be a non-empty object"
            )
    return value


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """

    # F-103: ``Literal`` so the typed arm of the
    # ``ResponseFormat | dict`` union rejects unknown ``type`` values
    # (``"xml"``, ``""``, etc.) at parse time. The dict arm is
    # separately guarded by ``_validate_response_format_raw`` on the
    # request model.
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: ResponseFormatJsonSchema | None = None
    # R7-H5 (Vlad r7 — 0.8.8 sweep): the OpenAI Responses API uses a
    # ``text.format = {"type":"json_schema","strict":true,"schema":{...},
    # "name":"..."}`` shape where ``strict`` is a SIBLING of ``type``
    # (not nested inside ``json_schema``). Clients writing against the
    # Responses surface and then pointing at the legacy chat endpoint
    # routinely keep the outer-level nesting — pre-R7 the unknown
    # field was silently dropped by Pydantic, so the chat path saw
    # a non-strict request, skipped the ``[guided]``-required gate,
    # and HTTP-200'd with no constraint enforcement. Declaring the
    # field here means BOTH nesting positions reach
    # ``is_strict_json_schema`` and the centralized gate fires
    # regardless of which nesting position the client used. The chat
    # route's existing strict-mode handling sees the same boolean
    # whichever wire form was sent. ``None`` is preserved as the
    # absent default; the dict-arm validator mirrors this so an
    # outer-level ``"strict":"true"`` (string) is rejected with a
    # clean 422 too.
    # R7-B codex MED follow-up: ``StrictBool`` rejects non-bool wire
    # forms at parse time on the typed arm; the dict arm is closed by
    # the explicit ``strict``-shape check in
    # ``_validate_response_format_raw`` below.
    strict: StrictBool | None = None


# =============================================================================
# Logprobs
# =============================================================================


class TopLogProb(BaseModel):
    """A top log probability for a token."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogProb(BaseModel):
    """Log probability information for a single token."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogProb] = []


class ChoiceLogProbs(BaseModel):
    """Log probability information for a choice."""

    content: list[TokenLogProb] | None = None


class LegacyCompletionLogProbs(BaseModel):
    """Log probability information for a legacy ``/v1/completions`` choice.

    The OpenAI legacy completions schema (pre-chat) is *different* from
    the chat-completions ``ChoiceLogProbs`` shape: it carries four
    parallel arrays — ``tokens``, ``token_logprobs``, ``top_logprobs``
    (a list of ``{token: logprob}`` dicts), and ``text_offset`` — keyed
    positionally per generated token. This split is required by SDK
    clients (the ``openai`` Python SDK, ``langchain``'s legacy
    ``OpenAI`` LLM wrapper, eval harnesses like ``lm-evaluation-harness``)
    that pre-1.0 OpenAI API never unified.
    """

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    text_offset: list[int]


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    # r5-E B-9 / F-DGF-V080-V1: strict bool, no coercion. Pre-fix,
    # Pydantic v2's default ``bool`` field happily coerced
    # ``include_usage:"yes"`` / ``"true"`` / ``1`` to ``True`` (lax-mode
    # coercion). The OpenAI spec is unambiguous that the wire shape is
    # a JSON ``true``/``false`` literal — a truthy string is a
    # client-side serialization bug and silently rewriting it to
    # ``True`` masks the bug AND diverges from real OpenAI which
    # 400's the same payload. Use ``StrictBool`` so a non-bool wire
    # value 422s at the schema layer; the ``_validate_include_usage``
    # mode="before" validator below flattens the Pydantic union-error
    # noise into a single human-readable message naming the field.
    include_usage: StrictBool = False  # Include usage stats in final chunk

    @field_validator("include_usage", mode="before")
    @classmethod
    def _validate_include_usage(cls, v):
        """Reject non-bool wire values BEFORE Pydantic's StrictBool gate.

        Without this, Pydantic v2's default ``StrictBool`` error reads
        ``"Input should be a valid boolean"`` (loc=include_usage),
        which is correct but doesn't pin the spec hint that callers
        most need (the wire shape MUST be a JSON literal, not a
        truthy string like ``"yes"`` / ``"on"`` / ``"1"``).
        Running the type check ourselves with ``mode="before"`` lets
        us return a message a client can act on AND keeps the
        contract identical to Pydantic's strict-mode behavior
        (booleans pass through unchanged; everything else 4xx's).
        """
        if v is None:
            # ``None`` is not a valid value for StrictBool; let the
            # field's default-False semantics handle the absent case
            # via field_validator returning None which Pydantic then
            # rejects with a clean "missing"/"none" error. Callers
            # that want the default should omit the field entirely.
            raise ValueError(
                "stream_options.include_usage must be a boolean "
                "(true or false); got null. Omit the field for the "
                "default (false) instead."
            )
        if not isinstance(v, bool):
            raise ValueError(
                "stream_options.include_usage must be a boolean "
                "(JSON true or false); got "
                f"{type(v).__name__}={v!r}. Truthy strings such as "
                "'yes'/'on'/'1' are NOT accepted per OpenAI spec."
            )
        return v


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str = "default"
    # D-ANTHRO-VALIDATION F11 (Anthropic sibling fix on OpenAI surface):
    # OpenAI's real ``/v1/chat/completions`` requires ``messages`` to be
    # a non-empty array (a request without prompts is structurally
    # uninterpretable). Pre-fix this server accepted ``messages=[]`` and
    # downstream code raised an unhandled exception → 500. Enforce at
    # the schema layer so the 4xx fires before any handler logic.
    messages: list[Message] = Field(..., min_length=1)
    # F-011: range bounds match OpenAI spec; the field-level Field
    # constraints cover finite out-of-range values (Pydantic 422). NaN
    # slips past Field bounds (every NaN comparison is False) so the
    # ``_reject_nonfinite_sampling`` validator below catches it
    # separately. Same gate applied to top_p / presence_penalty /
    # frequency_penalty + their CompletionRequest twins.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = None
    # OpenAI-canonical token cap since Sept 2024 (preferred over max_tokens for
    # reasoning models; newer SDKs >=1.45 send only this field). Normalized to
    # max_tokens by a model_validator so all downstream code keeps reading the
    # single max_tokens field.
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = (
        None  # Streaming options (include_usage, etc.)
    )
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters. Without these declared,
    # Pydantic drops them on parse (#355). top_k / min_p flow through to the
    # mlx-lm sampler; repetition_penalty / presence_penalty / frequency_penalty
    # flow through to mlx-lm's make_logits_processors().
    # H-10: ``top_k`` is range-checked by ``_validate_top_k`` below (mlx-lm
    # silently ignores negative values; we 4xx them).
    top_k: int | None = None
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # H-10 root: ``repetition_penalty=-1.0`` previously slipped past every
    # gate, reached ``mlx_lm/sample_utils.py:298``
    # (``make_repetition_penalty`` raises ``ValueError`` on negative
    # values), and the uncaught exception escaped the request scheduler
    # and crashed uvicorn. Field bound catches finite out-of-range;
    # the ``_reject_nonfinite_sampling`` validator below catches NaN/inf
    # (which Field bounds skip — every NaN comparison is False).
    # Upper bound is a safety cap (values >> 2 collapse the distribution
    # to a single token; the upstream HF/vLLM convention also caps here).
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    # F-011: presence_penalty / frequency_penalty are bounded [-2, 2] by
    # OpenAI spec. Field bounds catch finite out-of-range values (e.g.
    # ``presence_penalty=10`` previously HTTP 200'd as silent
    # mathematically-undefined logit shifts); the
    # ``_reject_nonfinite_sampling`` validator below catches NaN/inf,
    # which Field bounds skip (NaN comparisons always return False).
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # OpenAI extended spec — declared so Pydantic stops silently dropping it.
    # When set to False, the route caps the parsed ``tool_calls`` list at
    # length 1 in the response. Default None == True (model may emit
    # multiple). Cannot rely on decoder-level enforcement; this is a
    # post-generation truncation (the only reliable lever absent FSM
    # constraints — see PR #132 / #442 for the decoder-level path).
    parallel_tool_calls: bool | None = None
    # Legacy OpenAI tool-calling shape (pre-1.0 SDK + LangChain compat layers).
    # When set and the modern ``tools``/``tool_choice`` slots are empty, the
    # post-init validator below normalizes them to the modern equivalent so
    # downstream code keeps reading a single shape. Declared so Pydantic
    # stops silently dropping them (same blind-spot family as #355 /
    # #459 / #464). If a client supplies BOTH shapes, modern wins —
    # OpenAI's documented deprecation behavior — and the legacy slots are
    # ignored.
    functions: list[dict] | None = None
    function_call: str | dict | None = None
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # Logprobs
    logprobs: bool | None = None
    top_logprobs: int | None = None  # 0-20, per OpenAI spec
    # OpenAI extended spec — declared so Pydantic stops silently dropping
    # it. Currently rejected with 400 in routes/chat.py if non-empty;
    # mapping to mlx-lm's logits processor is tracked separately.
    logit_bias: dict[str, float] | None = None
    # MLLM-specific parameters
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # Thinking/reasoning control (Qwen3 style).  None = server default.
    enable_thinking: bool | None = None
    # OpenAI extended spec: arbitrary kwargs forwarded to the chat template.
    # We currently honor the ``enable_thinking`` key here; other keys are
    # accepted (no Pydantic drop) but not yet forwarded — see
    # ``_resolve_enable_thinking`` in service/helpers.py for precedence.
    chat_template_kwargs: dict | None = None
    # Per-request cap on reasoning tokens (upstream vLLM PR #20859 backport).
    # Semantic: force-close the ``<think>`` channel after N reasoning tokens
    # are emitted; subsequent model output is routed to the final/content
    # channel. DIFFERENT from the global ``thinking_token_budget`` which
    # additively extends ``max_tokens`` headroom for reasoning models — this
    # is a SUBTRACTIVE cap that gates how long the model is allowed to
    # think before answering. Distinct from ``max_tokens`` (which caps the
    # overall completion length). ``None`` = no cap, model decides.
    reasoning_max_tokens: int | None = None
    # R10-H5 (R9-H3 carry) — OpenAI ``reasoning_effort`` knob. Declared
    # so Pydantic stops silently dropping it (sven r10-R1 + vlad r10-R1
    # both confirmed every value 200'd because the field was undeclared).
    # Validated against ``_VALID_REASONING_EFFORTS`` via the
    # ``_validate_reasoning_effort_field`` validator below so int / list
    # / null / case-variant / garbage strings produce a clean 400.
    # Today the value is accepted-but-not-yet-translated at the engine
    # layer (a follow-up wires ``"none"`` through to ``enable_thinking
    # =False`` and ``low/medium/high`` to ``reasoning_max_tokens``
    # tiers); the schema-layer validation is the hard surface so SDK
    # clients see the contract round-trip.
    reasoning_effort: str | None = None
    # Number of completions (only n=1 supported). F-155: the route used
    # to reject ``n > 1`` only, so ``n=0`` / ``n=-1`` slipped through
    # and HTTP 200'd with a single choice — asymmetric with the
    # ``n > 1`` 400. The ``_validate_n`` field_validator below pins
    # the contract: omitted (``None``) or ``1`` is the only legal
    # surface; anything else (negative, zero, or > 1) → 422.
    n: int | None = None
    # H-11: OpenAI ``seed`` — per-request PRNG seed. When set, the
    # scheduler builds a fresh per-request sampler closure that maintains
    # its own ``mx.random.key(seed)`` state and splits it per step, so
    # two calls with the same ``(seed, temperature, top_p, prompt)`` pair
    # produce the same token stream. Without this field declaration
    # Pydantic silently dropped the OpenAI ``seed`` parameter and the
    # wire-claim was false (Tomek r3 — five calls with ``seed=42``
    # produced five different outputs).
    #
    # The field is intentionally UNBOUNDED at the request layer to
    # match the OpenAI spec surface — their SDK ships
    # ``seed: Optional[int]`` with no range and clients routinely pass
    # 64-bit integer seeds. The backend-specific narrowing to mlx-core's
    # ``uint32`` ``mx.random.key`` is applied in ``make_seeded_sampler``
    # via a deterministic ``seed & 0xFFFFFFFF`` bit-fold: same input
    # value always maps to the same backend key, so reproducibility is
    # preserved within rapid-mlx. (OpenAI's spec only promises within-
    # engine determinism — "we cannot guarantee determinism across
    # model versions or backends".) Codex round-6 BLOCKING fix on the
    # original ``Field(ge=0, le=0xFFFFFFFF)`` bound that 422'd valid
    # OpenAI seed values and broke compatibility for clients using the
    # broader documented integer range.
    seed: int | None = None

    # F-011: NaN/inf scrub on the raw dict, BEFORE Pydantic coerces a
    # bad value onto the typed float slot. Needed because (a) Field
    # range bounds skip NaN (every NaN comparison is False) so they
    # don't close the gap, and (b) even if we caught NaN at the
    # field_validator layer the resulting Pydantic ValidationError
    # would embed ``input_value=nan`` in the error dict — and
    # starlette's ``JSONResponse`` then crashes serializing it with
    # ``allow_nan=False``, turning a 422 into a 500. Sanitizing the
    # raw dict avoids both pitfalls.
    @model_validator(mode="before")
    @classmethod
    def _scrub_nonfinite_sampling(cls, data):
        return _scrub_nonfinite_sampling_raw(data)

    # F-103: tighten ``response_format`` dict-arm validation so the
    # silent-200 hazard (``json_schema.schema=42`` / ``"hello"``
    # falling through the union to bare-dict) is rejected with a
    # clean 422 at parse time. Runs on the raw value BEFORE the
    # ``ResponseFormat | dict`` union resolves so the dict arm
    # cannot swallow shapes the typed arm would reject.
    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format(cls, v):
        return _validate_response_format_raw(v)

    # F-155: enforce ``n == 1`` at parse time. The route already
    # rejected ``n > 1`` (#733), but ``n=0`` and ``n=-1`` slipped
    # through as HTTP 200 with one choice — asymmetric with the
    # ``> 1`` 400, and indistinguishable from a typo'd ``n: 0``
    # meant as ``n: 1``. Mirror the same rule on
    # ``CompletionRequest`` so the legacy completions surface is
    # consistent.
    @field_validator("n", mode="before")
    @classmethod
    def _validate_n(cls, v) -> int | None:
        # ``mode="before"`` so Python's ``bool`` (an ``int`` subclass)
        # is caught BEFORE Pydantic v2 coerces ``True`` → 1. Without
        # this, ``n: true`` would silently coerce to ``n=1`` and the
        # codex round-2 BLOCKING gap stays open.
        return _reject_non_one_n(v)

    # H-16 (Pavel r3): mirror M-03 (PR #766) onto the OpenAI surface.
    # The OpenAI spec accepts these ``tool_choice`` shapes:
    #
    #   * string: ``"none"`` / ``"auto"`` / ``"required"``
    #   * object: ``{"type": "function", "function": {"name": "<X>"}}``
    #
    # Plus the deprecated bare-string ``"function"`` literal that
    # some pre-2024 OpenAI SDKs still send to mean "force any
    # function call" (codex r9 NIT #1 on #551, pinned by
    # test_diffusion_engine.py::
    # test_engine_opts_out_blocks_legacy_function_literal_tool_choice).
    # The route's ``_forced_tool_choice`` check (routes/chat.py
    # L1212) honors it; the schema must let it through so the
    # route layer can apply its 422.
    #
    # Without this validator, a typo'd object like
    # ``tool_choice={"foo":"bar"}`` (no ``type`` field) or
    # ``tool_choice={"type":"banana"}`` (unknown ``type``) silently
    # falls through the typed ``str | dict`` union onto the dict arm,
    # the chat-route ``type=='function'`` guard (routes/chat.py L756)
    # doesn't match, and the request HTTP 200s as a free-form chat
    # completion — masking the client bug and diverging from the
    # Anthropic surface that PR #766 closed. Run BEFORE the typed
    # union resolves so the union's individual arms cannot swallow
    # shapes either arm alone would reject; mirror the M-03
    # envelope (``invalid_request_error`` with the field name) by
    # raising ``ValueError`` here — the global validation handler
    # (middleware/exception_handlers.py) maps the resulting
    # ``RequestValidationError`` to the canonical 400.
    @field_validator("tool_choice", mode="before")
    @classmethod
    def _validate_tool_choice(cls, v):
        # ``None`` / absent is the default — server picks the
        # default policy (``"auto"`` when ``tools`` is set, else
        # ``"none"``). Don't tighten beyond the M-03 wording.
        if v is None:
            return v
        # String form: closed-set. Spec values plus the deprecated
        # ``"function"`` legacy literal (kept because the route's
        # forced-tool-choice gate at L1212 still honors it; codex
        # r9 NIT #1 on #551). Reject unknown strings (``"banana"``,
        # ``"any"`` / ``"tool"`` (Anthropic's words), case
        # variants like ``"AUTO"``) so a cross-API confusion 400s
        # at parse instead of silently degrading. Pydantic v2
        # treats ``bool`` as an ``int`` subclass but ``isinstance(v,
        # str)`` is False for bools, so no boolean-leak concern
        # here.
        allowed_strings = ("none", "auto", "required", "function")
        if isinstance(v, str):
            if v not in allowed_strings:
                raise ValueError(
                    "tool_choice string must be one of "
                    f"{list(allowed_strings)} (got {v!r}). "
                    "See https://platform.openai.com/docs/api-reference/"
                    "chat/create#chat-create-tool_choice."
                )
            return v
        # Object form: must carry ``type=="function"``. Anything
        # else (no ``type`` field, unknown ``type`` value) is
        # outside the spec. The route-level guard at chat.py L756
        # already covers ``type=='function'`` without ``function.name``
        # with a more descriptive 400 (including the F-145
        # case-insensitive name hint), so we defer that arm to the
        # route and only reject the shapes the route silently
        # accepts (no ``type`` key, unknown ``type`` value).
        if isinstance(v, dict):
            if "type" not in v:
                raise ValueError(
                    "tool_choice object must have a 'type' field. Legal "
                    "shapes: a string from "
                    f"{list(allowed_strings)} or "
                    "{'type': 'function', 'function': {'name': '<X>'}}."
                )
            choice_type = v["type"]
            if choice_type != "function":
                raise ValueError(
                    "tool_choice.type must be 'function' for the object "
                    f"form (got {choice_type!r}). Legal shapes: a string "
                    f"from {list(allowed_strings)} or "
                    "{'type': 'function', 'function': {'name': '<X>'}}."
                )
            return v
        # Neither string nor dict — e.g. ``tool_choice=42``,
        # ``tool_choice=[1,2]``, ``tool_choice=true``. Pydantic's
        # union dispatch would surface a multi-line error blaming
        # both arms ("tool_choice.str: Input should be a valid
        # string; tool_choice.dict[any,any]: ..."); flatten it to
        # a single human-readable message that names the field.
        raise ValueError(
            "tool_choice must be a string from "
            f"{list(allowed_strings)} or an object "
            "{'type': 'function', 'function': {'name': '<X>'}} "
            f"(got {type(v).__name__})."
        )

    # H-11: ``mode="before"`` so Python's ``bool`` (an ``int`` subclass)
    # is caught BEFORE Pydantic v2 coerces ``True`` → 1. Same family as
    # ``_validate_n``: a seed of ``True`` is a serialization mistake, not
    # a legitimate request for seed=1.
    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed_field(cls, v) -> int | None:
        return _validate_seed(v)

    # R10-H5 (R9-H3 carry) — closed-set gate on ``reasoning_effort``.
    # ``mode="before"`` so non-string wire forms (int, list, dict,
    # bool, null) are rejected with a clear envelope BEFORE Pydantic's
    # union dispatch surfaces a generic "Input should be a valid
    # string" error pointing at the wrong arm. Mirrored on the
    # Responses surface (``ResponsesRequest._validate_reasoning_effort``)
    # so /v1/chat/completions and /v1/responses share one contract.
    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _validate_reasoning_effort_field(cls, v):
        return _validate_reasoning_effort(v)

    # Belt-and-braces: catches non-finite values that bypass the
    # raw-dict path (e.g. ``ChatCompletionRequest(temperature=nan)``
    # in-process). The Field range bounds also reject NaN as a side
    # effect (NaN comparisons are False), but we wire the explicit
    # finite check anyway so the field_validator-only path stays
    # sound even if a future cleanup drops the Field bound.
    @field_validator(
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    )
    @classmethod
    def _reject_nonfinite_sampling(cls, v: float | None) -> float | None:
        return _reject_nonfinite_float(v)

    # H-10: ``top_k`` had no range gate pre-fix — ``top_k=-5`` and
    # ``top_k=0`` (acceptable on mlx-lm as "disabled") slid through.
    # M-14 noted the silent-ignore on negatives; H-10's finite-range
    # sweep mandate is to 4xx instead. ``mode="before"`` so booleans
    # (``True``/``False`` are int subclasses in Python) are caught
    # before Pydantic coerces them to 1/0 — same pattern as
    # ``_validate_n``.
    @field_validator("top_k", mode="before")
    @classmethod
    def _validate_top_k(cls, v) -> int | None:
        return _validate_nonnegative_int(
            v, max_value=_TOP_K_SENTINEL_CAP, field_name="top_k"
        )

    # H-10: defensive finite-check on ``logit_bias`` values. The chat
    # route already 4xx's non-empty ``logit_bias`` (see ``routes/chat.py``
    # L699), but if a downstream PR ever wires it through, NaN/inf in
    # the values would re-create the F-011 silent-burn surface. Cheap
    # gate, large defense-in-depth payoff.
    @field_validator("logit_bias", mode="before")
    @classmethod
    def _validate_logit_bias(cls, v):
        return _validate_logit_bias_finite(v, field_name="logit_bias")

    # R7-M3: shared ``>= 1`` gate on the generation-budget fields so the
    # whole OpenAI-compat surface (chat / completions / messages /
    # responses) rejects ``max_tokens <= 0`` at the schema layer instead
    # of relying on the chat route's hand-rolled check. The chat route
    # used to be the only path with a ``max_tokens < 1`` guard — Vlad's
    # r7 sweep surfaced ``max_output_tokens=-5`` on /v1/responses,
    # ``max_tokens=-5`` on /v1/completions and /v1/messages all
    # returning 200. Schema-layer validator means the contract is
    # uniform regardless of which route handler runs.
    @field_validator("max_tokens", mode="before")
    @classmethod
    def _validate_max_tokens(cls, v) -> int | None:
        return _validate_positive_int(v, field_name="max_tokens")

    @field_validator("max_completion_tokens", mode="before")
    @classmethod
    def _validate_max_completion_tokens(cls, v) -> int | None:
        return _validate_positive_int(v, field_name="max_completion_tokens")

    @model_validator(mode="after")
    def _normalize_max_completion_tokens(self) -> "ChatCompletionRequest":
        if self.max_completion_tokens is not None:
            if (
                self.max_tokens is not None
                and self.max_tokens != self.max_completion_tokens
            ):
                raise ValueError(
                    "Cannot specify both max_tokens and max_completion_tokens with "
                    "different values; use max_completion_tokens only."
                )
            self.max_tokens = self.max_completion_tokens
        return self

    # M-04: opt-in per-request generation-budget ceiling. Runs AFTER
    # ``_normalize_max_completion_tokens`` so the canonical
    # ``self.max_tokens`` is what gets checked (callers that send
    # ``max_completion_tokens=N`` get the same enforcement as callers
    # that send ``max_tokens=N``). See the module-level rationale block
    # for why this is opt-in and why we deliberately don't add a
    # ``prompt_tokens + max_tokens <= context_window`` check here.
    @model_validator(mode="after")
    def _enforce_generation_budget_ceiling(self) -> "ChatCompletionRequest":
        _enforce_max_generation_tokens_ceiling(self.max_tokens)
        return self

    @model_validator(mode="before")
    @classmethod
    def _validate_reasoning_max_tokens_raw(cls, data):
        """Strict type-and-range check on ``reasoning_max_tokens``
        BEFORE Pydantic coercion (codex round-3 NIT #4).

        Without ``mode="before"`` plus an explicit type guard, Pydantic
        v2 silently coerces JSON-string ints (``"100"``) and booleans
        (``True`` → 1) onto the field — same wire-value hazard the
        Anthropic ``thinking.budget_tokens`` validator already covers,
        so this surface must match. ``StrictInt`` was rejected because
        it also rejects perfectly-fine wire ints (Pydantic strict-mode
        chokes on ``int`` vs ``StrictInt`` cross-pollination from
        nested models). A manual mode=before validator hits the same
        contract without touching the typed field declaration.

        Rules (mirror ``AnthropicRequest._validate_thinking_budget``):
        * Absent / ``None`` → no cap (default).
        * ``int`` with value ``>= 1`` → accepted.
        * Anything else (str, float, bool, list, dict) → 422.
        """
        if not isinstance(data, dict):
            return data
        # Pydantic v2 also accepts the field alias; the JSON wire name
        # IS ``reasoning_max_tokens`` (no alias), so only one lookup.
        if "reasoning_max_tokens" not in data:
            return data
        v = data["reasoning_max_tokens"]
        if v is None:
            return data
        # Booleans are an int subclass in Python — reject explicitly so
        # ``True``/``False`` don't slip in as 1/0.
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                "reasoning_max_tokens must be an integer when set "
                f"(got {type(v).__name__})."
            )
        if v < 1:
            raise ValueError(
                "reasoning_max_tokens must be >= 1 when set; pass "
                "enable_thinking=false to disable reasoning entirely."
            )
        return data

    @model_validator(mode="after")
    def _normalize_legacy_functions(self) -> "ChatCompletionRequest":
        """Translate the pre-1.0 ``functions``/``function_call`` shape into
        the modern ``tools``/``tool_choice`` slots so the route never has
        to know about the legacy form. Modern fields take precedence when
        a client supplies both — matches OpenAI's deprecation behavior."""
        if self.functions and self.tools is None:
            self.tools = [
                ToolDefinition(type="function", function=fn) for fn in self.functions
            ]
        if self.function_call is not None and self.tool_choice is None:
            fc = self.function_call
            if isinstance(fc, str):
                # "auto" / "none" map 1:1; anything else passes through and
                # the existing tool_choice handler will 400 on it.
                self.tool_choice = fc
            elif isinstance(fc, dict) and "name" in fc:
                self.tool_choice = {
                    "type": "function",
                    "function": {"name": fc["name"]},
                }
        return self

    # F-034: ``tool_choice="required"`` with no ``tools`` array (or with
    # an explicit ``tools: []``) is a malformed request per the OpenAI
    # spec — "required" means the model MUST emit a tool_call, which is
    # impossible to satisfy when the tools surface is empty. Without this
    # gate the request silently degrades to a plain chat completion (200),
    # masking a client bug. Fires AFTER ``_normalize_legacy_functions`` so
    # a legacy ``functions=[...]`` payload that has already been promoted
    # to ``tools`` is exempt. The ``{type:"function",function:{name:X}}``
    # named-tool shape is intentionally NOT mirrored here — the chat-route
    # validator (``vllm_mlx/routes/chat.py`` ~L748) already 400s on that
    # case with a more informative error referencing the missing function
    # name; duplicating that check at the schema layer would just produce
    # a less-helpful message.
    @model_validator(mode="after")
    def _validate_tool_choice_against_tools(self) -> "ChatCompletionRequest":
        if self.tool_choice == "required" and not self.tools:
            raise ValueError(
                "tool_choice='required' requires a non-empty 'tools' array; "
                "got tools=None or tools=[]."
            )
        return self


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = (
        None  # Reasoning/thinking content (when --reasoning-parser is used)
    )
    tool_calls: list[ToolCall] | None = None

    # R12-FIX-V2 (Vlad r12 MED-2): centralize special-token sanitization
    # on the assistant message envelope. Pre-fix the chat route called
    # ``sanitize_output`` on ``final_content`` only — ``reasoning_content``
    # was passed through to ``AssistantMessage`` untouched, and
    # ``<|im_start|>`` leaked verbatim on the ``tool_choice="required"``
    # branch for qwen3-0.6b-4bit. Anthropic + Responses adapters then
    # propagated the leaked token into ``thinking`` / ``reasoning``
    # output items.
    #
    # The field validator approach is the systematic fix: every code path
    # that constructs an ``AssistantMessage`` (chat route, Responses
    # adapter input, Anthropic adapter input) automatically gets
    # sanitized fields. New call sites cannot reopen the leak by
    # forgetting to sanitize — the contract is enforced at the type
    # boundary.
    @field_validator("content", "reasoning_content", mode="after")
    @classmethod
    def _sanitize_user_visible_strings(cls, v: str | None) -> str | None:
        from .utils import sanitize_reasoning_content

        return sanitize_reasoning_content(v)

    def model_post_init(self, __context) -> None:
        """Reserved hook (previously seeded a ``reasoning`` alias)."""
        pass

    @model_serializer(mode="wrap")
    def _serialize_assistant_message(self, handler):
        """Always emit ``content`` on the wire.

        Per OpenAI's ``chat.completion`` schema, ``message.content`` is a
        REQUIRED field — never absent. When a reasoning model truncates
        inside ``<think>`` or extract-tool-calls drains the visible text
        away the parser yields ``content=None`` and our callers serialize
        the response with ``model_dump_json(exclude_none=True)``; pydantic
        then drops the ``content`` key entirely and strongly-typed
        clients crash:

        * **Swift Codable**: ``decoder.container(keyedBy:)`` fails on
          missing required keys.
        * **Python pydantic** with ``extra="forbid"`` or ``content: str``
          on a strict response model.
        * **Rust serde**: a non-``Option`` field decode is a hard error.
        * **openai-python SDK**: ``resp.choices[0].message.content`` walks
          a Python dataclass that expects the key to be addressable.

        This wrap-mode serializer runs AFTER ``exclude_none`` pruning so
        we can put the field back regardless of how the parent was
        dumped.

        D-MISSING-CONTENT-KEY (r12-7, 0.8.14): switched the
        fill-in value from ``None`` (→ JSON ``null``) to the empty
        string ``""``. Both are spec-legal per the OpenAI canonical
        shape (``content: "" | string | null | array``), but ``""`` is
        what OpenAI's production API actually returns for the
        finish_reason="stop" + empty-completion case AND it preserves
        the wire-level ``string`` type discriminator so Swift Codable
        decodes cleanly into ``String`` (a ``null`` would force
        callers to model the field as ``String?``). Tool-call-only
        assistant messages now also serialize as
        ``{"role":"assistant","tool_calls":[...],"content":""}`` —
        same canonical shape OpenAI emits for tool-call turns.

        r10-B R10-C2 — emit ONLY ``reasoning_content``. r7-A R7-H2
        had additionally surfaced a duplicate ``reasoning`` alias as a
        one-release deprecation window; that window is now closed.
        The duplicate was the byte-for-byte root cause of R9-CRIT3
        (``openai-agents`` ``Runner.run_streamed`` doubling every
        text_delta because the SDK walks both keys). The OpenAI
        o1-style spec uses ``reasoning_content`` only — there is no
        ``reasoning`` key on chat-completion messages.
        """
        d = handler(self)
        # OpenAI contract: ``content`` is always present. Prefer ``""``
        # over ``null`` per D-MISSING-CONTENT-KEY (matches OpenAI's
        # production envelope; null is spec-legal but less common and
        # forces strongly-typed clients into an Optional/Nullable shape).
        if "content" not in d:
            d["content"] = ""
        return d

    def model_dump(self, **kwargs) -> dict:
        """Emit the standard OpenAI ``assistant`` message shape.

        Kept for callers that invoke ``.model_dump()`` directly (rather
        than via a parent ``model_dump_json``). The wrap-mode
        ``@model_serializer`` above already handles both paths, but this
        override remains a defensive belt-and-braces for the
        always-present ``content`` invariant.
        """
        d = super().model_dump(**kwargs)
        # Belt-and-braces: ensure ``content`` is always present, matching
        # the OpenAI-spec invariant enforced by the wrap-mode serializer.
        # D-MISSING-CONTENT-KEY: prefer ``""`` over ``None`` so the
        # dict view matches the wire view (string type discriminator).
        if "content" not in d:
            d["content"] = ""
        return d


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"
    logprobs: ChoiceLogProbs | None = None


class CompletionTokensDetails(BaseModel):
    """Breakdown of completion token usage (OpenAI-compatible)."""

    reasoning_tokens: int = 0


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt token usage (OpenAI-compatible)."""

    cached_tokens: int = 0


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str = "default"
    prompt: str | list[str]
    # F-011: shared range bounds + NaN/inf reject — see
    # ChatCompletionRequest for the rationale block. Field bounds
    # mirror OpenAI spec; the ``_reject_nonfinite_sampling`` validator
    # below catches NaN/inf, which Field bounds skip.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = None
    stream: bool = False
    # OpenAI streaming-spec parity (D-SSE-USAGE): legacy ``/v1/completions``
    # also accepts ``stream_options.include_usage`` to opt in to a
    # dedicated trailing usage chunk on SSE responses. Pre-fix this
    # field was silently dropped, so the only behavior was the
    # non-compliant "usage attached to finish chunk" that breaks
    # LangChain / AI-SDK accumulators (same root cause as the
    # chat-route bug). Default ``None`` ⇒ no usage on the wire.
    stream_options: StreamOptions | None = None
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters — see #355 + the
    # matching block on ChatCompletionRequest for wiring + caveats.
    # H-10: ``top_k`` / ``repetition_penalty`` finite-range gates
    # mirror the chat schema so both OpenAI routes share one contract.
    top_k: int | None = None
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    # H-10: ``repetition_penalty=-1.0`` previously slipped past the
    # schema, reached ``mlx_lm.sample_utils.make_repetition_penalty``,
    # raised ``ValueError``, escaped the request scheduler, and
    # crashed uvicorn. Field bound catches finite negatives;
    # ``_reject_nonfinite_sampling`` catches NaN/inf. Same rationale +
    # safety upper bound as the chat schema.
    repetition_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Logprobs — per the *legacy* OpenAI completions schema, this is an
    # *integer* (0..5) specifying the number of top alternative tokens
    # to return alongside each generated token, NOT a boolean (that is
    # the chat-completions shape). Wire-form ``bool`` is rejected by
    # the validator below with a clear 422 — pre-fix, Pydantic parsed
    # the field as ``bool`` and the canonical SDK call
    # ``logprobs=5`` (every official OpenAI client does this on the
    # legacy route) bounced with ``bool_parsing`` instead of being
    # served. F-153.
    logprobs: int | None = None
    # Echo the prompt back as the prefix of the response text (legacy
    # OpenAI behaviour — used by eval harnesses like
    # ``lm-evaluation-harness`` to compute prompt-conditioned token
    # log-probabilities). Pre-fix this was silently dropped (the
    # Pydantic schema didn't declare it) so clients depending on the
    # prompt prefix saw truncated output and proceeded with
    # garbage-in-garbage-out. F-152.
    echo: bool | None = None
    # Number of completions per prompt (legacy OpenAI). Rapid-MLX only
    # generates one completion per request — declared so Pydantic
    # stops silently dropping it; rejected with 400 in
    # ``routes/completions.py`` when ``> 1`` (mirroring the chat-route
    # behaviour). F-152.
    n: int | None = None
    # Server-side top-k re-rank knob from legacy OpenAI completions.
    # Not implementable on local MLX inference without a reranker;
    # declared so Pydantic stops silently dropping it and rejected
    # with 400 when ``> 1``. F-152.
    best_of: int | None = None
    # OpenAI FIM (fill-in-the-middle) suffix. Declared so Pydantic stops
    # silently dropping it; rejected with 400 in routes/completions.py
    # when non-empty since no MLX engine implements FIM yet (and silently
    # ignoring it produces wrong completions on code-completion clients).
    suffix: str | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # H-11: OpenAI ``seed`` — mirror of ChatCompletionRequest.seed. See
    # the matching declaration on the chat schema for the full rationale
    # block and the plumbing path through SamplingParams → scheduler.
    # Unbounded at the request layer (codex round-6); backend uint32
    # narrowing happens in ``make_seeded_sampler``.
    seed: int | None = None
    # R10-H4 (R9-H2 carry) — ``response_format`` on legacy completions.
    # Vlad r10-R1 + Bo r10-R1: ``/v1/completions`` with
    # ``response_format={"type":"json_object"}`` silently accepted the
    # field then ignored it (HTTP 200 with markdown ``` ```json `` fences
    # in the ``text`` reply); PR #844's streaming-fence path only ran on
    # ``/v1/chat/completions``. Declaring the field here means Pydantic
    # stops dropping it AND the field_validator below catches malformed
    # shapes via the same ``_validate_response_format_raw`` helper the
    # chat lane uses, so /v1/completions and /v1/chat/completions share
    # one schema contract. The route layer (``routes/completions.py``)
    # consumes this field and runs the chat-style fence-strip /
    # JSON-extraction on both the sync and streaming output paths.
    response_format: ResponseFormat | dict | None = None

    # F-011: NaN/inf scrub + finite belt-and-braces, exactly mirroring
    # ChatCompletionRequest. See the rationale block at the top of
    # this module + the matching validators on the chat schema.
    @model_validator(mode="before")
    @classmethod
    def _scrub_nonfinite_sampling(cls, data):
        return _scrub_nonfinite_sampling_raw(data)

    # R10-H4: response_format validation shares one helper with chat.
    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format(cls, v):
        return _validate_response_format_raw(v)

    @field_validator(
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    )
    @classmethod
    def _reject_nonfinite_sampling(cls, v: float | None) -> float | None:
        return _reject_nonfinite_float(v)

    # H-10: ``top_k`` range gate — mirror the chat schema (4xx for
    # negative; 0 means "disabled" and is fine).
    @field_validator("top_k", mode="before")
    @classmethod
    def _validate_top_k(cls, v) -> int | None:
        return _validate_nonnegative_int(
            v, max_value=_TOP_K_SENTINEL_CAP, field_name="top_k"
        )

    # R7-M3: shared ``>= 1`` gate on ``max_tokens`` — mirror of the
    # chat surface. Pre-R7-M3 the legacy completions surface silently
    # accepted ``max_tokens=-5`` (HTTP 200 with a single token)
    # because no route-level gate ran; the schema layer now closes
    # the bypass for every OpenAI-compat surface.
    @field_validator("max_tokens", mode="before")
    @classmethod
    def _validate_max_tokens(cls, v) -> int | None:
        return _validate_positive_int(v, field_name="max_tokens")

    # F-155: enforce ``n == 1`` at parse time, mirroring the chat
    # surface. The route already 400's ``n > 1``; the schema layer
    # now also rejects ``n=0`` / ``n=-1`` (silent-200 pre-fix).
    @field_validator("n", mode="before")
    @classmethod
    def _validate_n(cls, v) -> int | None:
        # ``mode="before"`` so Python's ``bool`` (an ``int`` subclass)
        # is caught BEFORE Pydantic v2 coerces ``True`` → 1. Without
        # this, ``n: true`` would silently coerce to ``n=1`` and the
        # codex round-2 BLOCKING gap stays open.
        return _reject_non_one_n(v)

    # H-11: mirror of ChatCompletionRequest._validate_seed_field.
    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed_field(cls, v) -> int | None:
        return _validate_seed(v)

    @model_validator(mode="before")
    @classmethod
    def _reject_bool_logprobs_raw(cls, data):
        """Reject ``logprobs: bool`` wire form on legacy completions.

        Same shape as ``ChatCompletionRequest._validate_reasoning_max_tokens_raw``:
        Pydantic v2 coerces ``True`` → 1 / ``False`` → 0 silently when
        the field is typed ``int | None``, but that coercion would be
        a footgun — a client that learned the chat-completions
        ``logprobs: bool`` shape would have its ``logprobs: true``
        silently turned into ``top_k=1`` and ``logprobs: false`` into
        ``top_k=0`` (no logprobs payload at all). Both surface as
        wrong-shaped responses with no error, exactly the silent-compat
        lie F-152 / F-153 closes. So a bool wire value gets a clear
        422 telling the client to send the canonical integer instead.
        """
        if not isinstance(data, dict):
            return data
        if "logprobs" not in data:
            return data
        v = data["logprobs"]
        # ``bool`` is an int subclass — check it BEFORE the integer
        # acceptance branch so ``True``/``False`` are rejected even
        # though they'd otherwise coerce to ``1``/``0``.
        if isinstance(v, bool):
            raise ValueError(
                "`logprobs` on /v1/completions must be an integer "
                "0-5 (number of top tokens to return), not a bool. "
                "The chat-completions endpoint uses `logprobs: bool + "
                "top_logprobs: int`; the legacy completions endpoint "
                "merges both into a single integer field. "
                "Pass `logprobs: <int>` instead."
            )
        return data

    # M-04: opt-in per-request generation-budget ceiling. Mirrors the
    # ``ChatCompletionRequest._enforce_generation_budget_ceiling``
    # validator so both OpenAI surfaces share the same env-var hook.
    # See the module-level rationale block for the opt-in design.
    @model_validator(mode="after")
    def _enforce_generation_budget_ceiling(self) -> "CompletionRequest":
        _enforce_max_generation_tokens_ceiling(self.max_tokens)
        return self


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    # Legacy completions uses the 4-parallel-array shape
    # (``LegacyCompletionLogProbs``); chat-completions uses the
    # ``ChoiceLogProbs`` shape — both are accepted here so the
    # downstream serializer can pick the spec-correct one per route.
    logprobs: LegacyCompletionLogProbs | ChoiceLogProbs | None = None


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model.

    The first four fields (`id`, `object`, `created`, `owned_by`) are
    the OpenAI-canonical shape. The trailing fields are Rapid-MLX
    vendor extensions surfaced so OpenAI-compatible clients that
    *also* want per-alias profile info (e.g. the rapid-desktop app
    auto-applying curated sampling defaults) don't need a separate
    private endpoint. OpenAI-only clients ignore unknown fields per
    spec, so the extension is additive — the OpenAI baseline
    contract is unchanged.

    Extension fields are populated from ``AliasProfile`` when an
    alias is known to the registry; absent fields stay ``None`` and
    serialize as JSON ``null`` on the wire (FastAPI default — we do
    not set ``exclude_none`` so the wire shape is stable and
    predictable). OpenAI-only clients ignore the unknown keys per
    spec whether they appear as ``null`` or are omitted, so the
    baseline contract is unchanged either way.
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rapid-mlx"

    # ---- Rapid-MLX vendor extensions (additive; OpenAI clients
    # ignore unknown fields) ------------------------------------
    # Curated sampling defaults that out-perform the model's
    # ``generation_config.json`` baseline on the canonical eval suite.
    # Shape: ``{key: value}`` where ``key`` is one of
    # {temperature, top_p, top_k, min_p, repetition_penalty,
    # presence_penalty, frequency_penalty} and ``value`` is a number.
    # rapid-desktop applies this on first alias load when the user
    # hasn't manually overridden the sliders. ``None`` means the
    # alias has no curated profile — the desktop should fall back
    # to the model's ``generation_config.json`` (already handled
    # server-side, no desktop change needed for that path).
    recommended_sampling: dict[str, float] | None = None
    # Hybrid-thinking architecture flag (Qwen 3 / 3.5 / 3.6, GLM 4.7,
    # Qwopus). When ``True`` the desktop surfaces the "Show reasoning"
    # toggle in Settings → Sampling AND defaults the toggle to OFF
    # (PR #154 default; see also the parser fix #570). When ``False``
    # the toggle stays hidden — no reason to show a UI knob that
    # the model's chat template silently ignores.
    is_hybrid: bool | None = None
    # MoE / sparse-expert architecture. Informational only — the
    # desktop uses this for the "this is an MoE alias" info row
    # in Settings → Models. Not load-bearing for sampling defaults.
    is_moe: bool | None = None
    # Parser pair — informational, useful for diagnostics rows in
    # the desktop's Settings → Models tab so an operator can see
    # which parser is doing the routing without grepping the
    # server logs.
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    # Inference modality. Desktop's ``ModelInfoCatalog`` already
    # dispatches on this — populating from the server lets us drop
    # the desktop-side hard-coded modality map in a future release.
    modality: str | None = None
    # Max prompt-token context window the loaded engine advertises
    # for this id. Populated only when the entry maps to an actively
    # loaded engine (single-model serve, or the matching ModelEntry
    # in multi-model mode); otherwise ``None``. rapid-desktop reads
    # this on first chat to auto-scale the ``max_tokens`` slider's
    # upper bound — without it, the desktop had to fall back to a
    # per-family hard-coded heuristic that drifted out of sync with
    # every new long-context release (Qwen3.6-A3B 1M, DeepSeek-Coder
    # 160K). Sourced from ``service.helpers.get_model_max_context``
    # which probes the same chain the engine-side context-length
    # guard uses, so the value the client sees lines up with the
    # cap the server will actually enforce. Issue #363.
    context_window: int | None = None
    # Capability tags advertised on the wire. Currently used to mark
    # the configured embedding model with ``"embedding"`` so clients
    # can discover which entry is wired to ``/v1/embeddings`` without
    # heuristics on the model id. H-09 sub-fix: when no embedding
    # model is configured at startup, NO entry carries ``"embedding"``
    # — the server used to silently fall back to chat-model hidden
    # states, so listing the chat model as embedding-capable was a
    # lie the desktop happily believed. Empty list (rather than
    # ``None``) means "we've thought about it; nothing applies".
    capabilities: list[str] = Field(default_factory=list)
    # F-K-CAPABILITIES-OMIT-AUDIO: per-lane audio backend status
    # surfaced when the deep audio probe has been run
    # (``RAPID_MLX_AUDIO_DEEP_PROBE=1``). ``None`` means the deep
    # probe never ran on this server, so we can't make any claim
    # about lane health beyond what ``capabilities`` already says.
    # ``{"stt": "ok", "tts": "degraded", ...}`` callers can read to
    # surface a "backend degraded" UX before sending a real audio
    # request. ``"degraded"`` means the lane imported cleanly but
    # the dry-run inference raised (F-K-WHISPER-500-shape failure —
    # a Whisper model whose processor never attached, a Kokoro
    # install missing ``misaki``, etc.). Values:
    # ``"ok"``, ``"degraded"``, ``"missing"``, ``"unknown"``.
    audio_lanes: dict[str, str] | None = None


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================


class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""

    tools: list[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    state: str
    transport: str
    tools_count: int
    error: str | None = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Audio (STT/TTS)
# =============================================================================


class AudioTranscriptionRequest(BaseModel):
    """Request for audio transcription (STT)."""

    model: str = "whisper-large-v3"
    language: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    timestamp_granularities: list[str] | None = None


class AudioTranscriptionResponse(BaseModel):
    """Response from audio transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


# R8-H5 (Bo 0.8.9 dogfood): the set of TTS ``response_format`` values
# the route can actually produce. Centralised here so the Pydantic
# validator AND the runtime encoder share one source of truth — any
# format added to :data:`vllm_mlx.routes.audio._TTS_CONTENT_TYPES`
# MUST be added here, otherwise the request would 400 before reaching
# the encoder. Pre-fix the route accepted any string verbatim and
# silently returned RIFF/WAV bytes labelled as the requested type;
# rejecting unknown values up front means clients get an actionable
# 400 with the supported set instead of a mislabeled body.
_TTS_ALLOWED_RESPONSE_FORMATS: tuple[str, ...] = (
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "pcm",
    # Note: ``aac`` is intentionally absent — libsndfile does not ship
    # an AAC encoder in any wheel we depend on. The route's encoder
    # also rejects ``aac`` (raises ``UnsupportedAudioFormatError``);
    # listing it here too would let a request through that then 500s
    # at the encoder boundary.
)


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech (OpenAI ``/v1/audio/speech`` compatible).

    R7-M8 (Bo 0.8.8 dogfood): ``input`` must reject empty / whitespace-
    only strings BEFORE we reach the synthesis engine. Pre-fix the
    route declared ``input: str = ""`` as a bare query parameter, so
    JSON bodies were silently dropped and an empty/missing ``input``
    collapsed into the generic 500 ``No audio generated`` envelope
    (the engine looped over an empty phoneme list and rapid-mlx's
    chunk-collector raised). Both shapes are CLIENT errors and should
    surface a 400 with ``param="input"`` so callers can fix their
    request payload without an opaque 500 round-trip.

    ``input`` carries an explicit ``min_length=1``; the bound model is
    registered with the envelope handler so the Pydantic validation
    error surfaces as ``{"error": {"type": "invalid_request_error",
    "param": "input", ...}}`` instead of the FastAPI 422 default.

    R8-M4 / R8-H5 (Bo 0.8.9 dogfood): ``voice`` and ``response_format``
    are validated up front against the known set so an invalid value
    surfaces as a 400 ``invalid_request_error`` with the relevant
    ``param=`` BEFORE the engine loads weights. Pre-fix ``voice``
    fell through to ``mlx_audio.load_safetensors`` which 500'd on the
    missing voice file, and ``response_format`` was accepted as any
    string then silently mislabeled WAV bytes.

    R11-B-F2 (Bo 0.8.12 dogfood): accept the legacy ``format`` spelling
    as an alias for ``response_format`` so OpenAI SDKs that still emit
    the old key (early ``openai-python`` < 1.0, Anthropic sample code,
    drop-in clients copied from pre-OAI-spec tutorials) actually get
    the codec they asked for. Pre-fix ``{"format":"mp3"}`` returned
    HTTP 200 with ``Content-Type: audio/wav`` and RIFF/WAVE bytes —
    the legacy key was silently dropped and ``response_format`` fell
    back to its ``"wav"`` default. The alias runs in a
    ``model_validator(mode="before")`` so the existing
    ``response_format`` field-validator still enforces the allowed-set
    contract once the alias has been folded in. The explicit caller
    (``{"response_format":"mp3"}``) wins on conflict — never the
    silent override Pre-fix did.
    """

    # ``extra="ignore"`` (the Pydantic default) is intentionally kept so
    # forward-compatible clients sending future OpenAI fields don't
    # 422 here. The before-validator below handles the ONE legacy key
    # we know about (``format``); everything else still passes through
    # without rejection so a benign extra ``"user":"xyz"`` doesn't break
    # callers.
    model: str = "kokoro"
    # min_length=1 catches the ``input=""`` shape Bo reported. The
    # ``model_validator`` below catches the whitespace-only shape that
    # slips past Pydantic's length check (``" "`` is one char) — both
    # produce empty phoneme lists downstream so both must 400.
    input: str = Field(..., min_length=1)
    voice: str = "af_heart"
    # OpenAI bounds: 0.25..4.0. Out-of-range values silently no-op
    # inside mlx_audio in some lanes; reject up front so the envelope
    # matches the documented contract.
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: str = "wav"

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_format_to_response_format(cls, data):
        """R11-B-F2: fold the legacy ``format`` key into ``response_format``.

        Some OpenAI-compatible clients (early ``openai-python`` releases,
        Anthropic sample code, drop-in tutorials copied before the spec
        was tightened) send ``{"format":"mp3"}`` instead of
        ``{"response_format":"mp3"}``. Pre-fix the key was silently
        dropped and ``response_format`` defaulted to ``"wav"`` —
        callers got HTTP 200 with RIFF/WAVE bytes mislabeled as the
        requested codec.

        Resolution rules (in order):

        * If ``response_format`` is explicitly set by the caller →
          keep it; the legacy ``format`` key is ignored (with a debug
          log) so the explicit, spec-correct field always wins. Never
          a silent override of caller intent.
        * Else if ``format`` is present and a string → fold it into
          ``response_format`` so the downstream field-validator
          enforces the allowed-set contract on the same value.
        * Else → leave the payload untouched (``response_format``
          falls back to its ``"wav"`` default).

        We do NOT remove the ``format`` key from the payload — Pydantic's
        ``extra="ignore"`` (the model default) drops it before field
        binding, so leaving it in place is harmless and keeps this
        validator side-effect-free if a future refactor changes the
        ConfigDict.
        """
        if not isinstance(data, dict):
            # Pydantic passes the raw payload here; non-dict shapes
            # (already a BaseModel instance, list, ...) skip the alias
            # — the field-level validators still fire.
            return data
        if "format" not in data:
            return data
        legacy = data["format"]
        if legacy is None:
            return data
        # Only fold the alias when the spec-correct field is unset.
        # ``None``/missing both count as unset; an explicit empty string
        # is the caller's choice and surfaces through the field
        # validator as a 400 like any other invalid value.
        if "response_format" not in data or data.get("response_format") is None:
            # Codex r1: fold EVERY non-None legacy value into
            # ``response_format`` — including non-strings like
            # ``{"format": 123}``. Pre-codex this guarded the assign
            # behind ``isinstance(legacy, str)`` so non-string aliases
            # silently fell through to ``response_format="wav"`` (the
            # exact silent-downgrade shape the field exists to prevent;
            # see codex review #1 on PR review-20260315-103736). Pydantic's
            # field validator runs next and surfaces an
            # ``invalid_request_error`` with ``param="response_format"``
            # so the caller learns the field is unhappy with the type
            # they sent — the same envelope an explicit
            # ``{"response_format": 123}`` would get.
            data["response_format"] = legacy
        return data

    @field_validator("input")
    @classmethod
    def _input_must_be_non_blank(cls, v: str) -> str:
        # ``min_length=1`` only checks character count — ``"   "`` is
        # length 3 and still passes, but it produces an empty phoneme
        # list downstream and trips the same 500 ``No audio generated``
        # the empty-string case did. Reject here too so the wire
        # contract is "non-blank text" and the param= envelope stays
        # consistent. We raise ``ValueError`` so Pydantic packages it as
        # the same ``value_error`` shape the field handler emits.
        if not v.strip():
            raise ValueError("input must be a non-empty, non-blank string")
        return v

    @field_validator("voice")
    @classmethod
    def _voice_must_be_non_blank(cls, v: str) -> str:
        # R8-M4 (Bo 0.8.9 dogfood): pre-fix ``voice=""`` fell through
        # to ``mlx_audio.load_safetensors("voices/.safetensors")`` which
        # 500'd with a stack trace. The model-aware list check happens
        # in the route handler (we don't know which model family is in
        # use here); the blank-string rejection is a cheap structural
        # check that catches the most common client bug — sending an
        # empty default — before we hit the model-aware validator.
        if not v or not v.strip():
            raise ValueError("voice must be a non-empty, non-blank string")
        return v

    @field_validator("response_format")
    @classmethod
    def _response_format_must_be_known(cls, v: str) -> str:
        # R8-H5 (Bo 0.8.9 dogfood): pre-fix every string was accepted
        # and ``Content-Type: audio/{format}`` echoed back, masking
        # the fact that the encoder produced WAV bytes regardless.
        # Reject unknown values here so a typo ("wave" → "wav") OR an
        # unsupported codec ("aac") returns a clean 400 envelope
        # before the engine loads weights. The allowed set mirrors the
        # route's ``_TTS_CONTENT_TYPES`` table (one source of truth);
        # adding a format requires editing both, which a unit test
        # pins.
        if v is None:
            return "wav"
        lower = v.lower()
        if lower not in _TTS_ALLOWED_RESPONSE_FORMATS:
            supported = ", ".join(_TTS_ALLOWED_RESPONSE_FORMATS)
            raise ValueError(f"response_format must be one of: {supported}; got {v!r}")
        return lower


class AudioSeparationRequest(BaseModel):
    """Request for audio source separation."""

    model: str = "htdemucs"
    stems: list[str] = Field(default_factory=lambda: ["vocals", "accompaniment"])


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request for text embeddings (OpenAI compatible)."""

    # extra="forbid" turns silent-drop into a 422 with a clear field name.
    # Without it, fields like `dimensions` or `encoding_format` typos pass
    # through and the user only notices when the response shape is wrong.
    # protected_namespaces=() suppresses the Pydantic v2 warning about
    # the `model` field colliding with the reserved `model_` prefix; a
    # future Pydantic point release could otherwise promote that warning
    # to an error and 500 every embeddings request.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # OpenAI spec lists 4 input shapes: ``str``, ``list[str]``,
    # ``list[int]`` (single pre-tokenized input), and
    # ``list[list[int]]`` (batch of pre-tokenized inputs). Production
    # pipelines that pre-tokenize with a shared HF tokenizer send the
    # latter two forms — refusing them broke LangChain / LlamaIndex
    # integrations that hard-code the spec shape (R10 sweep H6).
    #
    # ``StrictInt`` / ``StrictStr`` so Pydantic does NOT silently
    # coerce ``"123"`` → 123 (would be treated as token id 123, a
    # different embedding from the word "123") or ``True`` → 1
    # (Python ``bool`` is an ``int`` subclass; without ``StrictInt``
    # a JSON ``true`` would pass as token id 1).
    input: StrictStr | list[StrictStr] | list[StrictInt] | list[list[StrictInt]]
    model: str
    # Literal so an unknown value (typo like "base65" or "BASE64") 422s
    # at parse time rather than silently falling back to float — that
    # silent fallback is the same class of bug this PR exists to close.
    encoding_format: Literal["float", "base64"] | None = "float"
    # OpenAI spec: per-vector truncation. Common for MRL-style models
    # (text-embedding-3-large, nomic-embed-text-v1.5). Implemented in
    # the route as a post-embed slice + L2 renormalization (required
    # for the truncated vector to remain a valid embedding for cosine
    # similarity per the OpenAI cookbook).
    dimensions: int | None = None
    # OpenAI abuse-tracking field. Accepted (not validated) so clients
    # using the upstream SDK don't see a 422 on unknown field.
    user: str | None = None


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    index: int
    # `list[float]` for encoding_format="float"; base64-encoded float32
    # little-endian bytes (as ASCII string) for encoding_format="base64".
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    """Token usage for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """Response for embeddings endpoint (OpenAI compatible)."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None

    # R12-FIX-V2 (Vlad r12 MED-2): streaming-side parity for the
    # non-stream ``AssistantMessage`` sanitizer. Terminal SSE chunks that
    # carry a final ``reasoning_content`` must not leak special tokens
    # either — the same Vlad-r12 ``tool_choice="required"`` leak repro
    # but on the streaming surface. The ``_fast_sse_chunk`` per-delta
    # path is sanitized separately at the emit call site (see
    # ``routes/chat.py`` ``_fast_sse_chunk``) because it bypasses
    # pydantic serialization.
    #
    # Codex R3 [P2] (R12-FIX-V2 follow-up): when ``logprobs`` is
    # enabled, per-delta content/reasoning chunks ARE serialized through
    # this validator (see ``routes/chat.py`` streaming loop — the
    # logprobs branch builds a ``ChatCompletionChunk`` via pydantic
    # instead of the ``_fast_sse_chunk`` fast path). The same cross-delta
    # whitespace concatenation contract applies: trimming leading /
    # trailing whitespace from an individual delta after marker removal
    # corrupts the client's concatenated view. Use the
    # whitespace-preserving stream variant instead of the final-string
    # ``sanitize_reasoning_content`` (which calls ``.strip()``) so the
    # logprobs streaming path produces the same on-wire bytes as the
    # non-logprobs ``_fast_sse_chunk`` path.
    @field_validator("content", "reasoning_content", mode="after")
    @classmethod
    def _sanitize_user_visible_strings(cls, v: str | None) -> str | None:
        from .utils import sanitize_reasoning_for_stream

        if v is None:
            return None
        sanitized = sanitize_reasoning_for_stream(v)
        # Mirror the field's nullable contract: an empty post-sanitization
        # result collapses to ``None`` so ``exclude_none`` drops the
        # field cleanly on the wire when the delta was entirely markup.
        return sanitized if sanitized != "" else None

    @model_serializer(mode="wrap")
    def _serialize_chunk_delta(self, handler):
        """Ensure ``content`` is present on reasoning-only / terminal deltas.

        Callers serialize streaming chunks with
        ``model_dump_json(exclude_none=True)`` so per-token deltas stay
        terse (most deltas carry exactly one of ``content`` /
        ``reasoning_content`` / ``tool_calls``). When a generation
        terminates mid-``<think>`` reasoning, the terminal delta carries
        only ``reasoning_content`` plus a ``finish_reason`` on the
        parent choice — and the missing ``content`` key crashes any
        client that does ``chunk.choices[0].delta.content`` on the
        terminal chunk (the standard OpenAI SDK pattern; see the
        non-stream counterpart on ``AssistantMessage``).

        Mirror the OpenAI on-the-wire shape: surface ``content: ""``
        on any delta that carries ``reasoning_content`` (or
        ``tool_calls``) but no visible content, so the field is
        addressable on every reasoning-bearing delta — including the
        final one. Normal pure-content / pure-role / empty deltas keep
        their current minimal shape, so the per-token streaming budget
        is unchanged for non-reasoning paths.

        D-MISSING-CONTENT-KEY (r12-7, 0.8.14): the fill-in value was
        flipped from ``None`` to ``""`` for parity with the
        non-streaming ``AssistantMessage`` serializer. When the
        OpenAI streaming SDK reduces deltas into a final-state
        message (the canonical ``ChatCompletion`` aggregator pattern),
        the resulting ``message.content`` is a string, not a
        sometimes-None sometimes-string union — so strongly-typed
        Swift / Rust clients decoding the aggregated message keep
        their happy path.

        r10-B R10-C2 — emit ONLY ``reasoning_content`` on the wire.
        r7-A R7-H2 had also emitted a duplicate ``reasoning`` alias
        as a one-release deprecation window; that window is now
        closed. The duplicate was the byte-for-byte root cause of
        R9-CRIT3 (``openai-agents`` ``Runner.run_streamed`` emitting
        every text_delta twice because the SDK walks both keys).
        The OpenAI o1-style streaming spec uses ``reasoning_content``
        only — there is no ``reasoning`` key on chat-completion deltas.
        """
        d = handler(self)
        if "content" not in d and ("reasoning_content" in d or "tool_calls" in d):
            d["content"] = ""
        return d


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: ChoiceLogProbs | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Included when stream_options.include_usage=true
