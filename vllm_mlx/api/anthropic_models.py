# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Anthropic Messages API.

These models define the request and response schemas for the
Anthropic-compatible /v1/messages endpoint, enabling clients like
Claude Code to communicate with rapid-mlx.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .models import _enforce_max_generation_tokens_ceiling

# =============================================================================
# Request Models
# =============================================================================


class AnthropicContentBlock(BaseModel):
    """A content block in an Anthropic message."""

    type: str  # "text", "image", "tool_use", "tool_result"
    # text block
    text: str | None = None
    # tool_use block
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # tool_result block
    tool_use_id: str | None = None
    content: str | list[Any] | None = None
    is_error: bool | None = None
    # image block
    source: dict | None = None


class AnthropicMessage(BaseModel):
    """A message in an Anthropic conversation."""

    role: str  # "user" | "assistant"
    content: str | list[AnthropicContentBlock]


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
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    max_tokens: int  # Required in Anthropic API
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[AnthropicToolDef] | None = None
    tool_choice: dict | None = None
    metadata: dict | None = None
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
