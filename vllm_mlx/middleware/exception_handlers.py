# SPDX-License-Identifier: Apache-2.0
"""Unified FastAPI exception handlers for rapid-mlx.

The shapes here are the single source of truth — both
:mod:`vllm_mlx.server` (production) and the route-level test apps under
``tests/`` call :func:`install_exception_handlers` to wire them in.

The module intentionally has **no heavy imports** (no ``.engine``, no
``mlx``) so isolated route tests can import it without pulling the
whole engine stack into the fixture.

Fixes covered:

* F-161 / F-162 — malformed JSON bodies on ``/v1/messages``,
  ``/v1/messages/count_tokens``, and ``/v1/responses`` were producing
  HTTP 500 because ``await request.json()`` raises
  :class:`json.JSONDecodeError` and the only catch-all was the global
  ``Exception`` handler.
* F-094 / F-104 mitigation — the default FastAPI 422 echoes the
  offending value verbatim in ``detail[*].input``. We collapse to a
  400 with no value echo and strip the pydantic.dev help URL (F-163).
* H-17 — ``/v1/messages`` and ``/v1/responses`` construct their
  Pydantic request models manually (``AnthropicRequest(**body)`` /
  ``ResponsesRequest(**body)``) instead of binding them as FastAPI
  body parameters, so the resulting :class:`pydantic.ValidationError`
  never reached ``RequestValidationError``. The previous per-route
  ``raise HTTPException(status_code=400, detail=str(e))`` patches
  echoed the full Pydantic message — leaking the model class name,
  the pinned pydantic version (``errors.pydantic.dev/2.13/...``),
  and attacker-controlled ``input_value`` blobs. The dedicated
  ``pydantic.ValidationError`` handler below routes both routes
  through the same sanitized 400 envelope used by ``/v1/chat/
  completions`` and ``/v1/completions``.
"""

from __future__ import annotations

import json as _json
import logging
import typing as _t
from typing import get_args, get_origin

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

logger = logging.getLogger("rapid_mlx.exception_handlers")


# ---------------------------------------------------------------------------
# Trusted root-model registry (D-ENVELOPE-FIELD-LEAK, layered on top of
# the D-ANTHRO-VALIDATION F1 closed allowlist below)
# ---------------------------------------------------------------------------
#
# D-ANTHRO-VALIDATION (PR #811) shipped a CLOSED ALLOWLIST of schema-
# owned field names (``_SCHEMA_OWNED_FIELD_NAMES`` below) — names in
# the set echo verbatim, names outside collapse to ``<field>``. That
# fixes Sergei's F1 dogfood case and Sarah F-S1-1 alike.
#
# D-ENVELOPE-FIELD-LEAK layers a STRICTER per-error walk ON TOP:
# whenever the failing root model can be identified (via
# ``exc.title`` or the registry probe below), we walk the loc against
# THAT class's ``model_fields`` instead of trusting the global
# allowlist. Two wins over the allowlist alone:
#
#   1. ``error.param`` gets populated — the OpenAI SDK error branches
#      key on this slot, and the allowlist path leaves it ``None``.
#   2. The H-17 round-2 attack (``AWS_SECRET_ACCESS_KEY`` stuffed into
#      a ``dict[str, T]`` field's KEY) stays closed even if a future
#      contributor adds ``AWS_SECRET_ACCESS_KEY`` as a request-model
#      field somewhere — the walker only echoes names that live on
#      the SPECIFIC class for the current loc, not the global union.
#
# Both gates fire belt-and-suspenders: schema-owned names must pass
# BOTH the walker (when a root resolves) AND the allowlist (when it
# doesn't). The allowlist remains the safety floor for the FastAPI-
# bound body cases where the wrapped ``RequestValidationError`` carries
# only ``loc=("body",)`` — i.e. no string component to disambiguate
# the root.
_REQUEST_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_request_model(model_cls: type[BaseModel]) -> None:
    """Register a root request-model class for envelope field-name resolution.

    Idempotent — re-registering the same class is a no-op. Called from
    :func:`install_exception_handlers` (where the canonical OpenAI /
    Anthropic / Responses / Embeddings request models live) so tests
    that import the middleware in isolation get the same registry as
    production.

    Third-party plugins that add their own routes can call this at
    import time to opt into safe field-name surfacing for their own
    request models. Classes not registered fall through to the
    D-ANTHRO closed-allowlist behaviour, which is the safe default.
    """
    _REQUEST_MODEL_REGISTRY[model_cls.__name__] = model_cls


def _unwrap_optional(tp: _t.Any) -> _t.Any:
    """Strip ``Optional[X]`` / ``X | None`` / ``Union[X, None]`` to ``X``."""
    origin = get_origin(tp)
    if origin is None:
        return tp
    args = get_args(tp)
    if not args:
        return tp
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == len(args):
        return tp
    if len(non_none) == 1:
        return non_none[0]
    return non_none[0] if non_none else tp


def _union_arms(tp: _t.Any) -> tuple[_t.Any, ...]:
    """Return the non-``None`` arms of ``tp`` if it's a union, else ``(tp,)``.

    Handles both ``typing.Union[...]`` and the PEP 604 ``X | Y`` form —
    they share the same ``get_origin`` semantics in Python 3.10+
    (``typing.Union``), so ``get_args`` returns the arm tuple.
    """
    import types as _types

    origin = get_origin(tp)
    if origin is _t.Union or origin is getattr(_types, "UnionType", None):
        return tuple(a for a in get_args(tp) if a is not type(None))
    return (tp,)


def _descend_field(tp: _t.Any, hint: str | None = None) -> _t.Any:
    """Pick the inner type for ``list[X]`` / ``tuple[X, ...]`` / ``dict[K, V]``.

    Returns ``None`` when the inner type can't be determined.

    Codex r3 BLOCKING #2: when ``tp`` is a non-``Optional`` union such
    as ``str | list[ResponseInputItem]``, the caller may also provide a
    ``hint`` — the next loc string component we're about to step into.
    We pick the arm whose ``model_fields`` contain that hint (so the
    walker can keep descending instead of bailing to ``<field>`` on the
    first attacker-controlled-looking step).
    """
    inner = _unwrap_optional(tp)
    arms = _union_arms(inner)
    # If we have a hint, prefer a union arm that schema-owns the hint.
    if hint is not None and len(arms) > 1:
        for arm in arms:
            candidate = _descend_field(arm, hint=None)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseModel)
                and hint in candidate.model_fields
            ):
                return candidate
            if (
                isinstance(arm, type)
                and issubclass(arm, BaseModel)
                and hint in arm.model_fields
            ):
                return arm
    # Single-arm path (or hint-less): fall through to container peeling.
    target = arms[0] if arms else inner
    origin = get_origin(target)
    if origin is None:
        return target
    args = get_args(target)
    if not args:
        return None
    if origin in (list, tuple, set, frozenset):
        return args[0]
    if origin is dict:
        return args[1] if len(args) >= 2 else None
    return None


# Request-path → root model class. Wired up at install time so the
# FastAPI-wrapped ``RequestValidationError`` path (where ``exc.title``
# is unavailable) can disambiguate request models that share a field
# name (codex r3 BLOCKING #1: ``input`` is a field on both
# ``EmbeddingRequest`` and ``ResponsesRequest`` — the naive
# first-match registry probe would mis-resolve and redact legitimate
# nested locs as ``<field>``). Keyed by URL path prefix because
# FastAPI routes can mount under arbitrary prefixes.
_REQUEST_PATH_TO_ROOT: list[tuple[str, type[BaseModel]]] = []


def register_request_path(path_prefix: str, model_cls: type[BaseModel]) -> None:
    """Register a request-path → root-model-class mapping.

    Used by the FastAPI-wrapped ``RequestValidationError`` path to
    pick the correct root model when the loc starts with ``"body"``
    only (no string component to disambiguate via the field-name
    registry probe). ``path_prefix`` is matched against
    ``request.url.path`` with the same shape as
    :func:`_is_anthropic_path` (exact match OR strict sub-path).

    Idempotent: re-registering the same ``(path_prefix, model_cls)``
    pair is a no-op. Registering a NEW class for an existing prefix
    overwrites the prior binding so a plugin that swaps the request
    model can update the mapping at startup.
    """
    for i, (existing_prefix, _) in enumerate(_REQUEST_PATH_TO_ROOT):
        if existing_prefix == path_prefix:
            _REQUEST_PATH_TO_ROOT[i] = (path_prefix, model_cls)
            return
    _REQUEST_PATH_TO_ROOT.append((path_prefix, model_cls))


def _path_matches_canonical_prefix(path: str, prefix: str) -> bool:
    """Return True if ``path`` matches ``prefix`` as a canonical segment.

    Matching shape (codex r4 BLOCKING):

    * Exact match — ``/v1/embeddings`` == ``/v1/embeddings``.
    * Strict sub-path — ``/v1/embeddings/anything`` matches
      ``/v1/embeddings`` but ``/v1/embeddings-foo`` does not.
    * Mounted-prefix match — ``/api/v1/embeddings`` matches
      ``/v1/embeddings`` because the canonical prefix appears at a
      ``/``-boundary AND the path ends there (or continues with a
      ``/``).  Substring attacks like ``/v1/embeddings-foo`` /
      ``/foo/v1/embeddingsbar`` are rejected because the segment must
      align on both sides.

    The implementation does the cheap exact/startswith case first and
    only falls back to the segment scan for the mounted-deployment
    case so the hot path stays O(1).
    """
    if path == prefix or path.startswith(prefix + "/"):
        return True
    # Mounted-prefix: ``/<anything>/<prefix>`` exact or with trailing
    # ``/<rest>``. The canonical prefix already starts with a leading
    # ``/`` so the left boundary IS that leading slash — we don't need
    # to require a second ``/`` before it. We just require that idx
    # itself is > 0 (so there's a non-empty mount segment) AND that
    # the right side closes at a ``/`` or end-of-string so
    # ``/v1/embeddings-foo`` doesn't false-match.
    needle = prefix
    idx = path.find(needle, 1)  # start at 1 — idx=0 was the unmounted case
    while idx != -1:
        end = idx + len(needle)
        # Right side must align on a ``/`` boundary or end of string.
        right_ok = end == len(path) or path[end] == "/"
        if right_ok:
            return True
        idx = path.find(needle, idx + 1)
    return False


def _resolve_root_model(
    exc: object,
    loc: tuple,
    request: Request | None = None,
) -> type[BaseModel] | None:
    """Pick the root request-model class for ``loc`` from the registry.

    Three probes, in order of precedence:

    1. ``exc.title`` — Pydantic v2 sets this on the raw
       ``ValidationError``. The route's ``AnthropicRequest(**body)`` /
       ``ResponsesRequest(**body)`` constructions land here. Unambiguous.
    2. Request-path probe — for FastAPI-bound bodies the wrapped
       ``RequestValidationError`` doesn't carry a title, but the
       ``Request`` object's path tells us which route received the
       body. Path-based lookup is the disambiguator for request
       models that share a field name (codex r3 BLOCKING #1:
       ``input`` is on both ``EmbeddingRequest`` and
       ``ResponsesRequest``).
    3. Field-name probe (legacy fallback) — try each registered
       model and pick the first whose ``model_fields`` contain the
       FIRST non-``"body"`` string component of ``loc``. Still a
       safe fallback because the field-name probe only fires when
       the path probe has nothing to say, and even an ambiguous
       resolution falls back to ``<field>`` on a per-loc-component
       basis.

    Returns ``None`` if nothing matches — the caller falls back to
    the D-ANTHRO closed-allowlist path.
    """
    title = getattr(exc, "title", None)
    if isinstance(title, str):
        root = _REQUEST_MODEL_REGISTRY.get(title)
        if root is not None:
            return root
    # Path probe: matches by canonical-segment suffix so the same
    # registry entry covers both unmounted routes (``/v1/embeddings``)
    # AND deployments behind a FastAPI ``APIRouter`` mount that prepends
    # an arbitrary prefix (e.g. ``/api/v1/embeddings`` or
    # ``/proxy/foo/v1/embeddings``). Codex r4 BLOCKING: the prior
    # ``startswith`` check left mounted deployments resolving by the
    # ambiguous field-name probe again. We still reject
    # ``/v1/messages-foo`` (would-be substring attack) because the
    # match is on path SEGMENTS, not raw bytes — the canonical prefix
    # must align with a ``/``-boundary on both ends.
    if request is not None:
        try:
            path = request.url.path
        except Exception:  # pragma: no cover — defensive
            path = None
        if path:
            for prefix, cls in _REQUEST_PATH_TO_ROOT:
                if _path_matches_canonical_prefix(path, prefix):
                    return cls
    # Field-name probe (legacy fallback).
    first_str: str | None = None
    for raw in loc:
        if raw == "body":
            continue
        if isinstance(raw, str):
            first_str = raw
            break
        break
    if first_str is None:
        return None
    for cls in _REQUEST_MODEL_REGISTRY.values():
        if first_str in cls.model_fields:
            return cls
    return None


def _walk_loc_with_root(
    loc: tuple,
    root_cls: type[BaseModel] | None,
) -> tuple[list[str], str | None]:
    """Walk ``loc`` against ``root_cls`` and return ``(parts, last_field)``.

    Schema-owned field names (in ``current_class.model_fields``) pass
    through unchanged; attacker-controlled string components collapse
    to ``<field>`` and stop the descent. Integer indices pass through
    unchanged. ``last_field`` is the last schema-owned field name we
    crossed — used to populate ``error.param``.
    """
    parts: list[str] = []
    last_field: str | None = None
    current: _t.Any = root_cls
    loc_list = list(loc)
    for idx, raw in enumerate(loc_list):
        if raw == "body":
            continue
        if isinstance(raw, int):
            parts.append(str(raw))
            if current is not None:
                # Peek the next string loc segment as a hint so
                # _descend_field can pick the right union arm
                # (codex r3 BLOCKING #2).
                hint = _peek_next_field_hint(loc_list, idx)
                current = _descend_field(current, hint=hint)
            continue
        if isinstance(raw, str) and _is_union_arm_discriminator(raw):
            continue
        # When ``current`` is a non-Optional union (e.g. ResponsesRequest.input
        # is ``str | list[ResponseInputItem]``), try to resolve through the
        # union arm whose model_fields contain ``raw`` before deciding the
        # token is attacker-controlled (codex r3 BLOCKING #2).
        if current is not None and not (
            isinstance(current, type) and issubclass(current, BaseModel)
        ):
            current = _descend_field(current, hint=raw)
        is_schema_owned = (
            isinstance(current, type)
            and issubclass(current, BaseModel)
            and raw in current.model_fields
        )
        if is_schema_owned:
            parts.append(raw)
            last_field = raw
            field_info = current.model_fields[raw]  # type: ignore[union-attr]
            current = _unwrap_optional(field_info.annotation)
        else:
            parts.append("<field>")
            current = None
    return parts, last_field


def _peek_next_field_hint(loc_list: list, idx: int) -> str | None:
    """Return the next string component of ``loc_list`` after ``idx``.

    Skips the union-arm discriminator marker and integer indices —
    those are not field names. Used by :func:`_walk_loc_with_root` to
    give :func:`_descend_field` a hint when picking the right arm of a
    non-``Optional`` union.
    """
    for nxt in loc_list[idx + 1 :]:
        if (
            isinstance(nxt, str)
            and nxt != "body"
            and not _is_union_arm_discriminator(nxt)
        ):
            return nxt
    return None


def _extract_field_from_value_error_msg(
    msg: str,
    root_cls: type[BaseModel] | None,
) -> str | None:
    """Extract a schema-owned field name from a model-level ``value_error``.

    The NaN/inf scrubber runs ``mode='before'`` so it can mutate the
    raw dict before Pydantic's float-coercion fires. The ``ValueError``
    reaches the envelope with ``loc=()`` (raw ``PydanticValidationError``)
    or ``loc=("body",)`` (FastAPI-wrapped ``RequestValidationError``) —
    the field name is only in the message string.

    Recovery is gated on three layers of schema membership to keep the
    H-17 round-2 secrecy default intact even on this empty-loc path
    (pr_validate codex r2 BLOCKING — explicit demonstration that the
    fallback path actually populates ``error.param``):

    1. If ``root_cls`` was resolved, the token must be a field of
       THAT class (strictest gate).
    2. Else, the token must be a field of SOME registered request
       model — covers the FastAPI-wrapped case where ``loc`` only
       has ``"body"`` so ``_resolve_root_model`` can't disambiguate.
    3. Else, the token must be a member of the D-ANTHRO closed
       allowlist — same safety floor every other code path uses.

    Each layer is a closed schema set, so attacker-controlled bytes
    can never reflect even when steps 1-2 produce no anchor.
    """
    if not msg:
        return None
    stripped = msg
    prefix = "Value error, "
    if stripped.startswith(prefix):
        stripped = stripped[len(prefix) :]
    first = stripped.split(None, 1)[0] if stripped else ""
    while first and not (first[-1].isalnum() or first[-1] == "_"):
        first = first[:-1]
    if not first:
        return None
    # Layer 1: strict per-class membership when a root resolved.
    if root_cls is not None:
        return first if first in root_cls.model_fields else None
    # Layer 2: registry-wide schema membership (covers the FastAPI-
    # wrapped ``loc=("body",)`` case — _resolve_root_model can't
    # disambiguate without a string loc component, but the field
    # name in the message is still gated on the closed union of
    # registered request models).
    for cls in _REQUEST_MODEL_REGISTRY.values():
        if first in cls.model_fields:
            return first
    # Layer 3: D-ANTHRO closed allowlist (final safety floor — used
    # when nothing is registered yet, e.g. in an isolated test
    # fixture that didn't call install_exception_handlers).
    return first if first in _SCHEMA_OWNED_FIELD_NAMES else None


# D-ANTHRO-VALIDATION F1 — closed allowlist of schema-owned field names
# the loc sanitizer is allowed to ECHO instead of collapsing to
# ``<field>``. Pre-fix, every string ``loc`` component (even safe
# schema-owned ones like ``temperature`` / ``messages``) collapsed to
# the placeholder, producing a user-facing 400 like
# ``<field>: Field required`` that names nothing actionable. Sergei's
# F1 dogfood (Anthropic /v1/messages with ``temperature="hot"``) shows
# the leak: ``message: "Invalid request body: <field>: Input should
# be a valid number, ..."``.
#
# The H-17 round-2 finding (codex BLOCKING #1) is preserved by keeping
# the default-deny: only names that appear in this allowlist are
# echoed; everything else (attacker-supplied dict keys, extra-forbid
# field names, identifiers we don't recognize) still collapses to
# ``<field>``. The set is built from the public request-model surfaces
# (AnthropicRequest, ChatCompletionRequest, CompletionRequest,
# ResponsesRequest) plus the nested content-block models — every name
# below is schema-determined public-API surface, so echoing it leaks
# no attacker bytes.
#
# IMPORTANT: this list is a *closed* allowlist — adding a new field to
# a request model also requires adding it here if you want the
# validation error to name it. The default-deny means a forgotten
# entry just produces a less-informative ``<field>`` placeholder; it
# never opens a leak vector. The H-17 round-2 attacker shapes
# (``AWS_SECRET_ACCESS_KEY``, ``X-Forwarded-For``, ``../../etc/passwd``,
# 256-char identifiers) are NOT in this set and remain collapsed.
_SCHEMA_OWNED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        # AnthropicRequest
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "output_config",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
        # ChatCompletionRequest (additional)
        "chat_template_kwargs",
        "enable_thinking",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "min_p",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_max_tokens",
        "repetition_penalty",
        "response_format",
        "seed",
        "stop",
        "stream_options",
        "timeout",
        "top_logprobs",
        "video_fps",
        "video_max_frames",
        # CompletionRequest (additional)
        "best_of",
        "echo",
        "prompt",
        "suffix",
        # ResponsesRequest (additional)
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "service_tier",
        "store",
        "text",
        # Nested Anthropic content-block / message fields
        # (``text``, ``input`` already declared at top-level above)
        "role",
        "content",
        "type",
        "source",
        "id",
        "name",
        "tool_use_id",
        "is_error",
        # Nested OpenAI Message / ContentPart fields
        "tool_call_id",
        "tool_calls",
        "audio_url",
        "image_url",
        "video",
        "video_url",
        # Nested output_config / reasoning / text.format fields
        "format",
        "effort",
        "schema",
        "description",
        "strict",
        "budget_tokens",
        # tool_choice nested
        "function",
        # Nested image source
        "media_type",
        "data",
        "url",
        # tool definitions
        "input_schema",
        "parameters",
    }
)


def _is_union_arm_discriminator(raw: str) -> bool:
    """Identify Pydantic v2 union-arm loc components.

    On a union field, Pydantic v2 appends the failing arm's *type
    descriptor* to ``loc`` so the validator can disambiguate which arm
    rejected the input. The descriptors are not user-controlled bytes
    (they're synthesised from the schema) but they're noisy — bare
    primitive names like ``"str"``, ``"int"``, ``"bool"``, ``"dict"``,
    or wrapped names like ``"list[function-after[...]]"`` /
    ``"nullable[...]"``. Filter them out entirely so the surfaced
    ``loc`` path stays readable: instead of
    ``messages.0.content.<field>: Input should be a valid string`` the
    user sees ``messages.0.content: Input should be a valid string``.

    The set covers Pydantic's primitive-arm names AND the bracketed
    composite-arm shapes (detected by structural prefix because the
    inner schema text varies per model).
    """
    if raw in {"str", "int", "float", "bool", "dict", "list", "bytes", "tuple"}:
        return True
    # Composite-arm shapes: ``list[...]``, ``dict[...]``, ``tuple[...]``,
    # ``nullable[...]``, ``function-after[...]``, ``function-before[...]``,
    # ``union[...]``. The bracket is the structural tell.
    if "[" in raw and raw.endswith("]"):
        return True
    return False


def _sanitize_loc(loc: tuple) -> str:
    """Collapse a Pydantic ``loc`` tuple to a safe dotted path.

    Drops the synthetic ``"body"`` prefix FastAPI prepends. Keeps
    positional indices (``int``) as-is — they come from list/sequence
    positions and the attacker can't inject arbitrary bytes there.
    Drops union-arm discriminator components (see
    :func:`_is_union_arm_discriminator`) so the user-visible path
    matches the request body shape, not Pydantic's internal dispatch
    metadata.

    For string components, applies a *closed allowlist* of schema-owned
    field names (see :data:`_SCHEMA_OWNED_FIELD_NAMES`):

    * Names in the allowlist are echoed verbatim — they're public API
      surface (declared on the request Pydantic models) and naming
      them in the error message gives clients an actionable hint.
    * Names NOT in the allowlist collapse to ``<field>`` so attacker-
      controlled bytes (dict keys on ``dict[str, T]`` fields,
      extra-forbid field names) are never reflected. This is the H-17
      round-2 safety contract preserved unchanged for unknown names.

    Pre-D-ANTHRO-VALIDATION (F1), EVERY string collapsed — so
    ``temperature="hot"`` produced ``<field>: Input should be a valid
    number`` and the client had no idea which field broke. The closed
    allowlist closes that informational gap without re-opening the
    H-17 leak vector (any name an attacker could choose is NOT in the
    allowlist and still collapses).
    """
    parts: list[str] = []
    for raw in loc:
        if raw == "body":
            continue
        if isinstance(raw, int):
            parts.append(str(raw))
            continue
        # Drop Pydantic union-arm discriminator metadata entirely —
        # it's noisy and not a real path the user can act on.
        if isinstance(raw, str) and _is_union_arm_discriminator(raw):
            continue
        # String component — echo if it's a known schema-owned field
        # name; otherwise collapse to the H-17 placeholder.
        if isinstance(raw, str) and raw in _SCHEMA_OWNED_FIELD_NAMES:
            parts.append(raw)
        else:
            parts.append("<field>")
    return ".".join(parts)


def _render_loc_for_envelope(
    exc: object,
    loc: tuple,
    request: Request | None = None,
) -> tuple[str, str | None]:
    """Render ``loc`` for the user-facing 400 envelope (D-ENVELOPE-FIELD-LEAK).

    Returns ``(rendered, param)``. ``rendered`` is the dotted-path
    string the envelope ``message`` field uses; ``param`` is the
    last schema-owned field name on the path (used to populate the
    OpenAI envelope's ``error.param`` slot — Sarah F-S1-1 / F-S2-2).

    Two-stage resolution:

    1. Resolve the root request-model class via
       :func:`_resolve_root_model` (uses ``exc.title`` for
       ``PydanticValidationError`` + a registry probe for the
       FastAPI-wrapped ``RequestValidationError``). If a root
       resolves, walk it against ``model_fields`` for the strict
       per-class membership check.
    2. If no root resolves (an unregistered request model, or the
       FastAPI-bound case where ``loc`` is just ``("body",)``), fall
       back to :func:`_sanitize_loc` which applies the D-ANTHRO closed
       allowlist. ``param`` is recovered from the last allowlist-hit
       component on that branch.

    Either way, the rendered path matches the per-route SCHEMA — the
    H-17 round-2 attack (attacker-controlled dict keys / extra-
    forbidden names) stays closed.
    """
    root_cls = _resolve_root_model(exc, loc, request)
    if root_cls is not None:
        parts, last_field = _walk_loc_with_root(loc, root_cls)
        return ".".join(parts), last_field
    # Fallback: closed-allowlist sanitisation. Recover param by
    # scanning the rendered path for an allowlist-hit token (no
    # attacker bytes survived the allowlist, so this is safe).
    rendered = _sanitize_loc(loc)
    param: str | None = None
    for part in rendered.split("."):
        if part and part in _SCHEMA_OWNED_FIELD_NAMES:
            param = part  # last match wins — matches the walker's contract
    return rendered, param


# D-ANTHRO-VALIDATION F1 — Anthropic /v1/messages routes use a
# different top-level error envelope than the OpenAI surfaces:
# ``{"type":"error","error":{...}}`` (with an explicit ``type`` key on
# the outer object) versus ``{"error":{...}}``. The Anthropic SDK
# routes errors by ``response.type == "error"``; without the wrapper a
# 400 looks like an unstructured response and the SDK falls back to a
# generic ``APIStatusError`` with no typed Anthropic error class.
#
# Detect Anthropic surfaces by request path so a single set of
# handlers covers every error path (validation, HTTPException, JSON
# decode, recursion, generic 500). Path matching is strict — an exact
# match on the root path OR a strict sub-path match. Codex round-1
# NIT: a bare ``startswith("/v1/messages")`` would also classify
# unrelated paths like ``/v1/messages-foo`` or ``/v1/messagesevil`` as
# Anthropic surfaces, so an attacker who can probe arbitrary paths
# would receive the Anthropic envelope on 404/405s. The explicit
# ``path == ROOT or path.startswith(ROOT + "/")`` shape rejects those
# while still covering the legitimate ``/v1/messages/count_tokens``
# sub-route.
_ANTHROPIC_ROOT_PATHS: tuple[str, ...] = ("/v1/messages",)


def _is_anthropic_path(request: Request | None) -> bool:
    """Return True if ``request`` targets an Anthropic-compat route."""
    if request is None:
        return False
    try:
        path = request.url.path
    except Exception:
        return False
    for root in _ANTHROPIC_ROOT_PATHS:
        if path == root or path.startswith(root + "/"):
            return True
    return False


def _wrap_for_anthropic(response: JSONResponse) -> JSONResponse:
    """Rewrap an OpenAI-shaped error envelope to the Anthropic shape.

    Input:  ``{"error":{...}}``
    Output: ``{"type":"error","error":{...}}``

    Idempotent: if the body already carries a top-level ``type=="error"``
    (e.g. a route opted into the Anthropic envelope explicitly via
    ``HTTPException(detail={"type":"error","error":{...}})``), the
    response is returned unchanged. Non-error bodies are also returned
    unchanged so non-error JSON responses can't accidentally pick up an
    ``error`` field.

    Headers from the original response are preserved EXCEPT for
    ``content-length`` / ``content-type`` (Starlette recomputes these
    when constructing the new ``JSONResponse`` — copying them verbatim
    would emit an ``h11._util.LocalProtocolError: Too much data for
    declared Content-Length`` when the wrapped body is longer than the
    original).
    """
    raw = getattr(response, "body", None)
    if raw is None:
        return response
    try:
        body = _json.loads(raw)
    except (_json.JSONDecodeError, TypeError):
        return response
    if not isinstance(body, dict):
        return response
    if body.get("type") == "error" and isinstance(body.get("error"), dict):
        return response
    if "error" not in body:
        return response
    wrapped = {"type": "error", "error": body["error"]}
    # Preserve any non-error sibling keys (none expected today but
    # forward-compatible) by surfacing them at the top level.
    for k, v in body.items():
        if k == "error":
            continue
        wrapped[k] = v
    # Carry over auth / rate-limit headers but skip length+type which
    # Starlette regenerates from the new body. h11's protocol checker
    # rejects a stale Content-Length when the wrapped body is longer
    # than the original (D-ANTHRO-VALIDATION first-pass repro).
    preserved_headers: dict[str, str] | None = None
    if response.headers:
        preserved_headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-length", "content-type")
        }
        if not preserved_headers:
            preserved_headers = None
    return JSONResponse(
        status_code=response.status_code,
        content=wrapped,
        headers=preserved_headers,
    )


def _decode_error_response(exc: _json.JSONDecodeError) -> JSONResponse:
    """Build the 400 envelope for a malformed-JSON request body.

    The message includes the structural reason from ``exc.msg``
    (e.g. ``Expecting value``, ``Expecting property name``) so clients
    can fix the bug — these strings are short and stable across Python
    versions, no secret leakage risk.
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid JSON in request body: {exc.msg}",
                "type": "invalid_request_error",
                "code": "invalid_json",
                "param": None,
            }
        },
    )


def _validation_error_response(
    exc: RequestValidationError | PydanticValidationError,
    request: Request | None = None,
) -> JSONResponse:
    """Build the 400 envelope for Pydantic body-validation failures.

    Strips ``detail[*].input`` (F-094 / F-104 secret-bounce vector)
    and drops the pydantic.dev help URL (F-163). Surfaces only the
    location + message so clients still get an actionable hint.

    Accepts both the FastAPI-wrapped :class:`RequestValidationError`
    (raised when a Pydantic model is bound as a FastAPI body parameter)
    and the raw :class:`pydantic.ValidationError` (raised when a route
    constructs the request model manually — e.g. ``/v1/messages`` and
    ``/v1/responses``). Both expose the same ``.errors()`` shape, so a
    single sanitizer covers both code paths (H-17).

    The ``loc`` is run through :func:`_render_loc_for_envelope` which
    layers a strict per-class registry walk (D-ENVELOPE-FIELD-LEAK) on
    top of the D-ANTHRO closed allowlist. Attacker-controlled dict keys
    / extra-forbidden names (H-17 round-2) still collapse to ``<field>``.

    ``error.param`` is populated from the LAST schema-owned field on
    the FIRST error in the list — matches the OpenAI single-``param``
    envelope shape so the SDK error branches (Sarah F-S1-1 / F-S2-2)
    finally have something to key on.
    """
    details: list[str] = []
    param: str | None = None
    for err in exc.errors():
        raw_loc = tuple(err.get("loc", ()))
        loc, last_field = _render_loc_for_envelope(exc, raw_loc, request)
        msg = err.get("msg", "validation error")
        # Model-level ``value_error`` (e.g. the NaN/inf scrubber that
        # runs ``mode='before'``) lands with ``loc=()`` — the field
        # name is only in the message string. Pull it back out IFF
        # the leading token matches a schema-owned field, so we never
        # surface attacker-controlled bytes.
        if last_field is None and not loc and err.get("type") == "value_error":
            root_cls = _resolve_root_model(exc, raw_loc, request)
            recovered = _extract_field_from_value_error_msg(msg, root_cls)
            if recovered is not None:
                last_field = recovered
        details.append(f"{loc}: {msg}" if loc else msg)
        if param is None and last_field is not None:
            param = last_field
    summary = "; ".join(details) or "Invalid request body"
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid request body: {summary}",
                "type": "invalid_request_error",
                "code": "invalid_request",
                "param": param,
            }
        },
    )


_HTTP_ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    405: "invalid_request_error",
    409: "conflict_error",
    429: "rate_limit_error",
}


def _http_error_response(exc: StarletteHTTPException) -> JSONResponse:
    """Build the OpenAI-shaped envelope for a Starlette ``HTTPException``.

    Routes can opt in to a fully custom envelope by raising
    ``HTTPException(detail={"error": {...}})`` — the structured detail
    is passed through unchanged. Bare-string detail is wrapped in the
    legacy envelope so existing callers keep working.
    """
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=detail,
            headers=getattr(exc, "headers", None),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": _HTTP_ERROR_TYPE_MAP.get(exc.status_code, "api_error"),
                "code": None,
                "param": None,
            }
        },
        headers=getattr(exc, "headers", None),
    )


def _generic_error_response() -> JSONResponse:
    """The unmodified-secret 500 envelope used for unhandled errors.

    Exception message / traceback go to the log; the client sees a
    generic message so we don't leak filesystem paths, model paths, or
    environment values to a probing attacker.
    """
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error"}},
    )


def _recursion_error_response() -> JSONResponse:
    """The sanitized envelope used when a ``RecursionError`` reaches
    the framework boundary (D-TOOL-RECUR / D-DEEP-JSON defense-in-depth).

    The primary defense for both bugs is structural — an iterative
    chat-template walk (see
    :func:`vllm_mlx.utils.chat_template._walk_tools_iter`) plus a body-
    depth guard middleware (see
    :mod:`vllm_mlx.middleware.body_depth`) plus a per-tool-schema
    depth validator (see :class:`vllm_mlx.api.models.ToolDefinition`).
    None of those should let a ``RecursionError`` propagate. But:

    * The body-depth gate path-scopes to JSON content types only,
      so a future route that accepts JSON via a different content
      type could bypass it.
    * The per-tool-schema validator runs at request-model construction,
      so a code path that builds a ``ChatCompletionRequest`` from a
      programmatically-constructed dict (engine tests, internal
      adapters) skips it.
    * Pydantic / FastAPI / Starlette internals may add new recursive
      paths in a future release that we haven't audited.

    Surfacing a ``RecursionError`` as HTTP 500 with a stack trace
    fragment (the pre-fix shape) is both a DoS signal AND an info-leak
    — the pre-fix trace named ``_sanitize_tools_for_template._walk``
    on every parser, so an attacker could identify the function and
    the line. This handler returns the SAME shape as
    :func:`_generic_error_response` (HTTP 500 ``Internal server
    error``) so:

    * No stack trace ever reaches the client (info-leak closed).
    * We DON'T claim "request body too deep" when the cause might
      actually be an unrelated recursion bug elsewhere in the server
      (codex r4 BLOCKING — a misleading 400 on a server-side bug
      would mask the real failure mode and the client would retry).
      The user-facing message stays neutral so an SDK keying on
      ``error.message == "Internal server error"`` handles it the
      same as any other unhandled server-side fault.
    * The body-depth gate middleware still emits its own
      ``request_body_too_deep`` 400 from the depth-cap rejection
      path — clients DO see that more-actionable error when the
      cause was actually a deep body.

    We log the trace at WARNING level so an operator can spot a new
    recursion site we should put a structural fix on, regardless of
    whether the cause was body-depth-related or somewhere else.
    """
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error"}},
    )


def _register_canonical_request_models() -> None:
    """Pre-populate the request-model registry with the canonical roots.

    Lazy on first ``install_exception_handlers`` call (not at module-
    import time) so the middleware module stays import-light — the API
    model modules pull in pydantic + a fair amount of route metadata,
    and H-17 deliberately kept the middleware importable in isolated
    route tests without dragging that in. Idempotent.
    """
    try:
        from ..api.anthropic_models import AnthropicRequest
        from ..api.models import (
            ChatCompletionRequest,
            CompletionRequest,
            EmbeddingRequest,
        )
        from ..api.responses_models import ResponsesRequest
    except Exception:  # pragma: no cover — defensive
        logger.debug(
            "Failed to import canonical request models for the envelope "
            "registry — falling back to the D-ANTHRO closed allowlist only.",
            exc_info=True,
        )
        return
    for cls in (
        ChatCompletionRequest,
        CompletionRequest,
        EmbeddingRequest,
        AnthropicRequest,
        ResponsesRequest,
    ):
        register_request_model(cls)
    # Route-path → root-model bindings. Walked in declaration order, so
    # more-specific prefixes must come before less-specific ones. These
    # break ties when the same field name (e.g. ``input``, ``messages``,
    # ``model``) appears on more than one canonical request model — the
    # path probe wins over the field-name fallback so ``/v1/responses``
    # resolves to ``ResponsesRequest``, never ``EmbeddingRequest``.
    register_request_path("/v1/chat/completions", ChatCompletionRequest)
    register_request_path("/v1/completions", CompletionRequest)
    register_request_path("/v1/embeddings", EmbeddingRequest)
    register_request_path("/v1/messages", AnthropicRequest)
    register_request_path("/v1/responses", ResponsesRequest)


def install_exception_handlers(app: FastAPI) -> None:
    """Register the rapid-mlx exception handlers on ``app``.

    Wiring is idempotent — re-registering the same exception class
    just overwrites the previous binding (FastAPI / Starlette behaviour).
    Tests and production both call this exactly once.

    Also pre-populates the request-model registry that
    :func:`_render_loc_for_envelope` uses to walk the loc against the
    actual root request model class (D-ENVELOPE-FIELD-LEAK, layered on
    top of D-ANTHRO).
    """
    _register_canonical_request_models()

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(
        request: Request,
        exc: StarletteHTTPException,
    ):
        response = _http_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(_json.JSONDecodeError)
    async def _decode_handler(
        request: Request,
        exc: _json.JSONDecodeError,
    ):
        response = _decode_error_response(exc)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        response = _validation_error_response(exc, request)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(PydanticValidationError)
    async def _pydantic_validation_handler(
        request: Request,
        exc: PydanticValidationError,
    ):
        # H-17: routes that build a Pydantic model manually
        # (``AnthropicRequest(**body)`` on /v1/messages,
        # ``ResponsesRequest(**body)`` on /v1/responses, plus the
        # adapter-layer ``ChatCompletionRequest`` constructions inside
        # those routes) raise the raw ``pydantic.ValidationError``.
        # Route it through the same sanitized envelope as the FastAPI-
        # bound bodies so the model class name, the pinned pydantic
        # version (``errors.pydantic.dev/2.13/...``), and the attacker-
        # supplied ``input_value`` stay out of the response body.
        #
        # Codex H-17 round-2 NIT #4: a global handler converts every
        # internal Pydantic bug into a client 400, which can mask
        # server-side defects. Log at WARNING with sanitized metadata
        # so operators can spot "this 400 actually came from a server-
        # side response-model construction failure".
        #
        # Codex H-17 round-3 BLOCKING: do NOT pass the raw exception
        # (``exc_info=exc`` or ``str(exc)``) — its string form embeds
        # ``input_value=...`` which can carry attacker-supplied
        # secrets, and this PR is explicitly preventing that
        # reflection. Operator log lines must be sanitized too;
        # otherwise an attacker can stuff secrets into a body field
        # and pivot them into the operator's log pipeline. We surface
        # only the per-error ``type`` codes (``missing``,
        # ``int_parsing``, ``extra_forbidden``, …) and the sanitized
        # ``loc`` path — both schema-determined.
        sanitized = [
            {
                "type": err.get("type", "validation_error"),
                "loc": _sanitize_loc(tuple(err.get("loc", ()))),
            }
            for err in exc.errors()
        ]
        logger.warning(
            "pydantic.ValidationError on %s %s — %d sanitized error(s): %s",
            request.method,
            request.url.path,
            len(sanitized),
            sanitized,
        )
        response = _validation_error_response(exc, request)
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(RecursionError)
    async def _recursion_handler(
        request: Request,
        exc: RecursionError,  # noqa: ARG001
    ):
        # D-TOOL-RECUR / D-DEEP-JSON defense-in-depth — see
        # :func:`_recursion_error_response`. Log the trace at WARNING
        # so an operator can spot the new recursion site and add a
        # structural fix (the iterative walk + the depth guards are
        # the primary defenses; this handler should be unreachable in
        # production). Path + method only — no request-body bytes are
        # logged, mirroring the H-17 round-3 sanitization rule.
        logger.warning(
            "RecursionError on %s %s — caught at framework boundary, "
            "returning sanitized 500 (Internal server error). Add a "
            "structural fix (iterative walk / depth guard) for the new "
            "recursion site.",
            request.method,
            request.url.path,
            exc_info=True,
        )
        response = _recursion_error_response()
        if _is_anthropic_path(request):
            response = _wrap_for_anthropic(response)
        return response

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception):
        # Re-route the specific subclasses in case a TaskGroup /
        # thread boundary dispatches them here instead of through the
        # dedicated handlers above (FastAPI/Starlette occasionally
        # falls back to the generic handler on cancellation paths).
        anthropic = _is_anthropic_path(request)
        if isinstance(exc, _json.JSONDecodeError):
            response = _decode_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, RequestValidationError):
            response = _validation_error_response(exc, request)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, PydanticValidationError):
            response = _validation_error_response(exc, request)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, StarletteHTTPException):
            response = _http_error_response(exc)
            return _wrap_for_anthropic(response) if anthropic else response
        if isinstance(exc, RecursionError):
            # ``isinstance(RecursionError) before isinstance(Exception)``:
            # the dedicated handler above SHOULD catch this first, but
            # FastAPI's fallback chain occasionally lands here (same
            # rationale as the other ``isinstance`` rerouting above).
            logger.warning(
                "RecursionError on %s %s (via generic handler) — "
                "returning sanitized 500 (Internal server error).",
                request.method,
                request.url.path,
                exc_info=True,
            )
            response = _recursion_error_response()
            return _wrap_for_anthropic(response) if anthropic else response
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        response = _generic_error_response()
        return _wrap_for_anthropic(response) if anthropic else response
