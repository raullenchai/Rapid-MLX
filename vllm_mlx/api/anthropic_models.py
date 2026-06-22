# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Anthropic Messages API.

These models define the request and response schemas for the
Anthropic-compatible /v1/messages endpoint, enabling clients like
Claude Code to communicate with rapid-mlx.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import (
    _TOP_K_SENTINEL_CAP,
    _enforce_max_generation_tokens_ceiling,
    _scrub_nonfinite_sampling_raw,
    _validate_finite_in_range,
    _validate_nonnegative_int,
)

# =============================================================================
# Request Models
# =============================================================================


# D-ANTHRO-VALIDATION F4 — Known Anthropic content-block ``type``
# values. Used by ``AnthropicContentBlock._validate_block_shape`` to
# reject unknown types at the schema layer (pre-fix, an unknown type
# like ``{"type":"weirdblock", "data":1}`` slipped past the loose
# ``type: str`` declaration and Pydantic accepted the request — the
# model then received empty content and returned 200 with garbage
# output). The Anthropic spec accepts exactly these on the request
# surface; assistant-only ``thinking`` is included so request-side
# echoes of prior assistant turns parse (cross-role compat is enforced
# separately on ``AnthropicMessage``).
_ANTHROPIC_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {"text", "image", "tool_use", "tool_result", "thinking", "document"}
)


# D-ANTHRO-VALIDATION F4 — per-type required field maps. Each tuple
# names the fields that MUST carry a non-None value for a block of
# that type to be well-formed. Pre-fix:
#
# * ``{type:"text"}`` (no ``text``) returned 200 with the model
#   running on empty content (Sergei F4 evidence).
# * ``{type:"tool_use"}`` (no ``id``/``name``/``input``) likewise.
# * ``{type:"tool_result"}`` (no ``tool_use_id``/``content``) likewise.
#
# Anthropic's real backend 400s every one of these with a typed
# ``invalid_request_error``; mirror that here at the schema layer so
# the contract is enforced uniformly across non-stream / stream / and
# /v1/responses parallel routes.
_ANTHROPIC_BLOCK_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "text": ("text",),
    "tool_use": ("id", "name", "input"),
    "tool_result": ("tool_use_id", "content"),
    "thinking": ("thinking",),
    "image": ("source",),
    "document": ("source",),
}


class AnthropicContentBlock(BaseModel):
    """A content block in an Anthropic message."""

    type: str  # see ``_ANTHROPIC_CONTENT_BLOCK_TYPES``
    # text block
    text: str | None = None
    # thinking block (echoed assistant block; assistant-only on inputs)
    thinking: str | None = None
    # tool_use block
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # tool_result block
    tool_use_id: str | None = None
    content: str | list[Any] | None = None
    is_error: bool | None = None
    # image / document block
    source: dict | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _reject_non_string_text(cls, value: Any) -> Any:
        """Reject non-string ``text`` with a clean, field-named error
        message (H-15).

        ``text: str | None`` already 422s on a non-string value, but
        Pydantic's default message buries ``text`` under a
        ``messages.0.content.str: Input should be a valid string`` loc
        trail because the parent ``AnthropicMessage.content`` is a
        ``str | list[AnthropicContentBlock]`` union (the str-arm fails
        first, then the list-arm fails on ``text`` — two confusing
        union errors). Run an explicit before-validator so the client
        sees a single actionable message naming ``content[].text``
        directly.
        """
        if value is None or isinstance(value, str):
            return value
        raise ValueError(
            f"content[].text must be a string (got {type(value).__name__})"
        )

    @model_validator(mode="after")
    def _validate_block_shape(self) -> "AnthropicContentBlock":
        """D-ANTHRO-VALIDATION F4 — reject unknown / underspecified
        content blocks at the schema layer.

        Pre-fix, ``AnthropicContentBlock`` accepted ANY ``type`` string
        and treated every payload field as optional, so:

        * ``{"type":"text"}`` (no ``text``) → 200 OK, model ran on
          empty content.
        * ``{"type":"weirdblock"}`` → 200 OK, unknown block silently
          dropped.
        * ``{"type":"tool_use"}`` (no ``id``/``name``/``input``) → 200
          OK, malformed tool call sent to the engine.

        The spec is explicit: every Anthropic content block has a known
        ``type`` discriminator and a per-type set of required fields.
        Reject unknown types (so a typo or an SDK-version mismatch
        produces a clear 400 with a list of allowed types) and reject
        missing required fields (so a buggy client during the
        OpenAI→Anthropic migration sees the validation gap instead of
        a 200 with garbage output).

        Implemented as a single ``model_validator(mode="after")`` so
        every type-check fires once per block — discriminated-union
        Pydantic would split the model into per-type classes but that
        would also force every existing call-site (anthropic_adapter,
        the streaming router, the tests) to deal with a Union[...]
        narrowing surface. A single validator on the unified shape is
        less churn and produces the same client-visible 400.
        """
        block_type = self.type
        if block_type not in _ANTHROPIC_CONTENT_BLOCK_TYPES:
            allowed = sorted(_ANTHROPIC_CONTENT_BLOCK_TYPES)
            raise ValueError(
                f"content[].type {block_type!r} is not a recognized "
                f"Anthropic content block type. Allowed types: {allowed}."
            )
        required = _ANTHROPIC_BLOCK_REQUIRED_FIELDS.get(block_type, ())
        missing = [field for field in required if getattr(self, field) is None]
        if missing:
            field_list = ", ".join(missing)
            raise ValueError(
                f"content[].type={block_type!r} is missing required "
                f"field(s): {field_list}."
            )
        return self

    @model_validator(mode="after")
    def _validate_image_source_type(self) -> "AnthropicContentBlock":
        """Reject non-string ``source.data`` / ``source.url`` (H-15
        sibling).

        Anthropic image blocks ship the payload inside ``source`` with
        either ``{"type":"base64","media_type":"…","data":"…"}`` or
        ``{"type":"url","url":"…"}``. ``source`` is declared ``dict``
        (no inner schema) so a non-string ``data`` / ``url`` value
        (e.g. a nested list — the same H-15 shape) used to fall through
        the schema layer and surface as an uninformative downstream
        error. Pin a string-typed check here for parity with the
        OpenAI-side ``image_url.url`` rule (F-066) so the failure
        names the field cleanly at the schema layer.
        """
        if self.type == "image" and isinstance(self.source, dict):
            for key in ("data", "url"):
                if key in self.source:
                    val = self.source[key]
                    if val is not None and not isinstance(val, str):
                        raise ValueError(
                            f"image source.{key} must be a string "
                            f"(got {type(val).__name__})"
                        )
        return self


# D-ANTHRO-VALIDATION F10 — role-block compatibility matrix.
#
# The Anthropic spec defines which content-block types can appear in
# which role. Pre-fix every block type was accepted in every role, so:
#
# * A user-role message could carry a ``thinking`` block (assistant-
#   only on the spec), and the request went through with 200 OK.
#   Adversarial clients could "smuggle" synthesized thinking into the
#   conversation; well-meaning buggy clients silently sent garbage.
# * An assistant-role message could carry a ``tool_result`` block
#   (user-only on the spec) — same 200-with-garbage outcome.
#
# The matrix below pins the per-role allowed types. Enforced on
# ``AnthropicMessage._validate_role_block_compat`` (a single
# model_validator on the message shape so both routes — /v1/messages
# and any future Anthropic-shaped surface — inherit the contract).
_ANTHROPIC_ROLE_ALLOWED_BLOCK_TYPES: dict[str, frozenset[str]] = {
    "user": frozenset({"text", "image", "tool_result", "document"}),
    "assistant": frozenset({"text", "tool_use", "thinking"}),
    # System-role messages are conventionally a string on the request
    # surface (the ``AnthropicRequest.system`` field), but a few client
    # libraries pass them as a message with role="system" and
    # content=[{type:"text",...}]. Allow text-only there.
    "system": frozenset({"text"}),
}


class AnthropicMessage(BaseModel):
    """A message in an Anthropic conversation."""

    role: str  # "user" | "assistant" | (rarely) "system"
    content: str | list[AnthropicContentBlock]

    @model_validator(mode="after")
    def _validate_role_block_compat(self) -> "AnthropicMessage":
        """D-ANTHRO-VALIDATION F10 — enforce role-block compatibility.

        Reject blocks that don't belong to the message's role (e.g.
        ``thinking`` on user, ``tool_result`` on assistant). Unknown
        roles get a 400 too (the upstream chat-route rejects them but
        the Anthropic surface previously accepted any role string).

        Codex round-1 BLOCKING: the unknown-role gate has to fire
        BEFORE the string-content early return, otherwise
        ``{"role":"wizard","content":"hi"}`` slips through. The
        per-block-type loop is what's bypassed on string content (no
        blocks to iterate), but the role itself still needs to be one
        of the recognized Anthropic roles.
        """
        allowed = _ANTHROPIC_ROLE_ALLOWED_BLOCK_TYPES.get(self.role)
        if allowed is None:
            raise ValueError(
                f"role {self.role!r} is not recognized. Allowed roles: "
                f"{sorted(_ANTHROPIC_ROLE_ALLOWED_BLOCK_TYPES.keys())}."
            )
        if isinstance(self.content, str):
            return self
        for block in self.content:
            block_type = block.type
            if block_type not in allowed:
                raise ValueError(
                    f"content[].type={block_type!r} is not allowed in a "
                    f"{self.role!r}-role message. Allowed block types for "
                    f"role={self.role!r}: {sorted(allowed)}."
                )
        return self


class AnthropicToolDef(BaseModel):
    """Definition of a tool in Anthropic format."""

    name: str
    description: str | None = None
    input_schema: dict | None = None


class AnthropicOutputFormat(BaseModel):
    """Output format spec inside ``output_config``.

    Upstream vLLM PR #42396 (shipped v0.22.0) added native structured
    output to the Anthropic Messages surface via
    ``output_config.format = json_schema``. This mirrors the OpenAI
    ``response_format.json_schema`` shape so the existing guided-decode
    pipeline (see ``api/guided.py`` + outlines) can drive constrained
    JSON output on ``/v1/messages`` clients (e.g. Claude SDKs) without
    a separate code path.

    ``type`` is the only Pydantic-required field. Today only
    ``"json_schema"`` is accepted; any other value is rejected with
    HTTP 400 inside ``anthropic_to_openai`` (with a clear error message
    pointing at this surface), and the adapter additionally enforces
    that ``schema`` is a dict for ``type == "json_schema"``.
    """

    type: str  # only "json_schema" is supported on this surface today
    # JSON Schema dict. Required when ``type == "json_schema"``; the
    # adapter rejects requests where it is missing or not an object
    # (400). Declared as Optional/dict here so the Pydantic parse
    # surfaces validation in the adapter's domain-specific error
    # message rather than as a generic "field required" 422.
    schema_: dict | None = Field(default=None, alias="schema")
    name: str | None = None
    description: str | None = None
    strict: bool | None = None

    class Config:
        populate_by_name = True


class AnthropicOutputConfig(BaseModel):
    """Output-side configuration for an Anthropic Messages request.

    Backport of upstream vLLM PR #42396 (v0.22.0). Two fields are wired:

    * ``format`` — native structured output (Pick 2, PR #683).
      ``format = json_schema`` is translated to OpenAI ``response_format``
      by the adapter so the existing guided-decode pipeline applies.
    * ``effort`` — coarse-grained reasoning-token budget (Pick 1, this PR
      — upstream vLLM PR #20859 + #42396 backport). Translated to a
      concrete ``reasoning_max_tokens`` value on the OpenAI side via the
      ``ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS`` mapping below.
      ``max`` means "no cap" (uncapped — Anthropic default).

    Tightening note on ``effort``: Pick 2 originally landed this as a
    plain ``str | None`` accept-but-ignore field; Pick 1 narrows it to a
    ``Literal`` so a typo like ``"hgih"`` 422s at parse time instead of
    being silently dropped through to the no-cap path.

    Codex round-6 NIT: this IS an intentional API tightening — a
    future Anthropic SDK version that adds a new effort value would
    422 against this server until the ``Literal`` is widened AND a
    corresponding ``ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS`` entry
    is added. The trade-off is favorable: silently accepting unknown
    values today (Pick 2's permissive path) means a client requesting
    a brand-new ``"ultra"`` budget would get an uncapped response
    that bills the user differently from what they asked for. Failing
    loud + fast at parse time forces clients to surface the version
    mismatch. The cost of widening is two lines (one ``Literal``
    member + one mapping entry), so the maintenance cost is trivial.
    """

    format: AnthropicOutputFormat | None = None
    # Pick 1 (this PR) — wired into the reasoning-cap pipeline via
    # ``ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS`` + the adapter helper
    # ``_resolve_reasoning_max_tokens``. Literal-typed so unknown
    # values are rejected at parse time.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


# Per-Anthropic-spec mapping from ``effort`` to a concrete reasoning
# token budget (upstream vLLM PR #42396 / Anthropic SDK v0.22). Kept
# module-scoped so tests can import + assert against the same mapping
# the adapter uses. ``None`` means "no cap" (the default Anthropic
# behavior when the client omits ``output_config`` entirely).
ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS: dict[str, int | None] = {
    "low": 512,
    "medium": 2048,
    "high": 8192,
    "xhigh": 24000,
    "max": None,
}


class AnthropicRequest(BaseModel):
    """Request for Anthropic Messages API."""

    model: str
    # D-ANTHRO-VALIDATION F11 — ``messages`` must be a non-empty array.
    # Pre-fix, ``messages=[]`` fell through to the route handler which
    # then dereferenced ``messages[0]`` deep inside the adapter and
    # raised an unhandled exception → 500 ``Internal server error``.
    # Anthropic's real backend (and our own ``/v1/messages/count_tokens``
    # sub-route) return 400 ``invalid_request_error`` for the same
    # input. Enforce at the schema layer so streaming + non-streaming
    # both inherit the contract.
    messages: list[AnthropicMessage] = Field(..., min_length=1)
    system: str | list[dict] | None = None
    max_tokens: int  # Required in Anthropic API
    # H-10: Anthropic spec narrows ``temperature`` to ``[0, 1]`` (the
    # OpenAI ``[0, 2]`` range is a different surface). Pre-H-10 this
    # field had no Field bound AND no finite check — a NaN ``temperature``
    # HTTP-200'd into the Metal kernel and crashed the server, same
    # silent-burn class as F-011 on the OpenAI route. The ``ge``/``le``
    # bound catches finite out-of-range; the ``_reject_nonfinite_sampling``
    # validator below catches NaN/inf (Field bounds skip NaN).
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    # H-10: same finite-range gate on ``top_p`` (Anthropic spec: ``(0, 1]``).
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[AnthropicToolDef] | None = None
    tool_choice: dict | None = None
    metadata: dict | None = None
    # H-10: ``top_k`` range gate — the ``_validate_top_k`` validator
    # below 4xx's negative values (mlx-lm would otherwise silently
    # ignore them, same family as M-14).
    top_k: int | None = None
    # Upstream vLLM PR #42396 (v0.22.0) — native structured output on
    # /v1/messages via ``output_config.format = json_schema`` AND
    # reasoning budget via ``output_config.effort`` (Pick 1, this PR;
    # upstream PR #20859 + #42396 backport). Optional; absence preserves
    # the pre-existing free-form-text + no-cap path so existing SDK
    # callers see no behavior change.
    output_config: AnthropicOutputConfig | None = None
    # Legacy Anthropic ``thinking`` field (v0.20+) — mirrors the same
    # idea but as a ``{"type": "enabled", "budget_tokens": N}`` shape.
    # The adapter consults ``thinking.budget_tokens`` only when
    # ``output_config.effort`` is unset (newer surface wins).
    thinking: dict | None = None

    # H-10: NaN/inf scrub BEFORE Pydantic coerces a non-finite value
    # onto the typed ``float | None`` slot. Mirrors the
    # ``ChatCompletionRequest`` / ``CompletionRequest`` block — without
    # this, ``temperature=NaN`` would survive into the
    # ``ValidationError.input_value`` and starlette's JSONResponse
    # would crash serializing the error body (``allow_nan=False``),
    # turning the intended 422 into a silent 500 (or worse — uvicorn
    # death on the engine path). One source of truth: the helper
    # lives in ``models.py`` so every API surface scrubs identically.
    @model_validator(mode="before")
    @classmethod
    def _scrub_nonfinite_sampling(cls, data):
        return _scrub_nonfinite_sampling_raw(data)

    # H-10: belt-and-braces finite + range check on the typed slots.
    # The Field ``ge``/``le`` bounds already 422 finite out-of-range,
    # but the field-level call lets us pin the exact spec range
    # (Anthropic ``[0, 1]`` for temperature) in one shared helper and
    # keeps the contract sound even if a future cleanup drops the
    # Field bound. ``_validate_finite_in_range`` emits a message that
    # names the field, so the unified validation-error handler
    # renders something a client can act on.
    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float | None) -> float | None:
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=1.0, field_name="temperature"
        )

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, v: float | None) -> float | None:
        return _validate_finite_in_range(
            v,
            min_value=0.0,
            max_value=1.0,
            min_inclusive=False,
            field_name="top_p",
        )

    # H-10: ``top_k`` range gate — mirrors the OpenAI surfaces.
    # r5-E B-7: upper-bound sentinel cap (see ``models._TOP_K_SENTINEL_CAP``).
    @field_validator("top_k", mode="before")
    @classmethod
    def _validate_top_k(cls, v) -> int | None:
        return _validate_nonnegative_int(
            v, max_value=_TOP_K_SENTINEL_CAP, field_name="top_k"
        )

    # M-03 (#742 follow-up): the Anthropic Messages spec only accepts
    # four ``tool_choice.type`` values — ``auto``, ``any``, ``tool``,
    # ``none``. Without parse-time validation, unknown types like
    # ``{"type": "banana"}`` silently fall through ``_convert_tool_choice``'s
    # final ``return "auto"`` (anthropic_adapter.py L452) and the
    # request HTTP 200s with plain text instead of the 400 the OpenAI
    # route surfaces. The validator below mirrors the strict-Literal
    # discipline ``AnthropicOutputConfig.effort`` already uses (codex
    # round-6 NIT precedent): fail loud + fast at the schema boundary
    # so a client typo can't silently degrade tool-forcing semantics.
    #
    # Pre-Pydantic-coercion so the ``dict`` type slot doesn't strip
    # the keys we need to inspect, and so the typed ``tool_choice``
    # field remains ``dict`` for the downstream adapter (which
    # already calls ``.get("type")`` + ``.get("name")``). Keeping the
    # field type unchanged means zero churn on the adapter and on
    # the downstream chat route.
    @field_validator("tool_choice", mode="before")
    @classmethod
    def _validate_tool_choice(cls, v):
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError(
                "tool_choice must be an object with a 'type' field "
                f"(got {type(v).__name__})."
            )
        # Match the Anthropic public spec — see
        # https://docs.anthropic.com/en/api/messages#body-tool-choice.
        # Includes ``none`` because the adapter (anthropic_adapter.py
        # L449) already maps it through to OpenAI's ``"none"``; the
        # existing TestConvertToolChoice.test_none_type test pins this.
        # An entirely missing ``type`` key (``tool_choice={}``) is
        # preserved as a no-op by the adapter (defaults to ``"auto"``,
        # TestConvertToolChoice.test_missing_type_defaults_to_auto);
        # only EXPLICITLY-set unknown values trip the gate so we
        # don't tighten beyond M-03's wording.
        allowed = ("auto", "any", "tool", "none")
        if "type" not in v:
            return v
        choice_type = v["type"]
        if choice_type not in allowed:
            raise ValueError(
                "tool_choice.type must be one of "
                f"{list(allowed)} (got {choice_type!r}). "
                "See https://docs.anthropic.com/en/api/messages."
            )
        # Anthropic's spec requires ``name`` on the forced-tool form.
        # The Anthropic SDK enforces this client-side but a raw HTTP
        # client can omit it; without this guard the adapter builds
        # an OpenAI ``{"type":"function","function":{"name":""}}`` and
        # the chat-route ``tool_choice with type='function' requires
        # function.name`` 400 fires deep into the routing stack, with
        # a less Anthropic-shape error message. Surface the contract
        # at parse time so the message points at the right field.
        if choice_type == "tool":
            name = v.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "tool_choice with type='tool' requires a non-empty "
                    "string 'name' field."
                )
        return v

    @model_validator(mode="after")
    def _validate_thinking_budget(self) -> "AnthropicRequest":
        """Reject malformed ``thinking.budget_tokens`` when the field is
        present. Mirrors the OpenAI-side validation on
        ``ChatCompletionRequest.reasoning_max_tokens`` so the same 400
        shape surfaces whether the client uses the OpenAI or Anthropic
        surface (upstream vLLM PR #43402).

        Codex round-1 BLOCKING #2: an earlier draft only rejected
        non-positive INTS — wire values like ``"0"`` or ``"100"`` (string
        coercion mistakes from JSON-typed clients) were silently
        accepted and then ignored by ``_resolve_reasoning_max_tokens``,
        turning a requested cap into no cap. Now reject any non-int
        type AND any int < 1 so the contract is symmetrical with the
        OpenAI-side Literal-checked ``reasoning_max_tokens`` validator.
        Booleans are an int subclass in Python — reject explicitly
        because ``True`` would otherwise count as 1.
        """
        if isinstance(self.thinking, dict):
            budget = self.thinking.get("budget_tokens")
            if budget is None:
                return self
            if not isinstance(budget, int) or isinstance(budget, bool):
                raise ValueError(
                    "thinking.budget_tokens must be an integer when set "
                    f"(got {type(budget).__name__})."
                )
            if budget < 1:
                raise ValueError("thinking.budget_tokens must be >= 1 when set.")
        return self

    # M-04: opt-in per-request generation-budget ceiling, mirroring the
    # OpenAI surfaces (``ChatCompletionRequest`` /
    # ``CompletionRequest``). Anthropic ``max_tokens`` is required (not
    # ``int | None``), so the ceiling check fires on every request when
    # the env var is set. See ``models._enforce_max_generation_tokens_ceiling``
    # for the opt-in contract — the cap is only applied when
    # ``RAPID_MLX_MAX_GENERATION_TOKENS`` is set to a positive integer.
    @model_validator(mode="after")
    def _enforce_generation_budget_ceiling(self) -> "AnthropicRequest":
        _enforce_max_generation_tokens_ceiling(self.max_tokens)
        return self


# =============================================================================
# Response Models
# =============================================================================


class AnthropicUsage(BaseModel):
    """Token usage for Anthropic response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicResponseContentBlock(BaseModel):
    """A content block in the Anthropic response."""

    type: str  # "text", "thinking", or "tool_use"
    text: str | None = None
    # thinking-block field (Anthropic extended-thinking surface)
    thinking: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: Any | None = None


class AnthropicResponse(BaseModel):
    """Response for Anthropic Messages API."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    model: str
    content: list[AnthropicResponseContentBlock]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)
