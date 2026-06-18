# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Anthropic Messages API.

These models define the request and response schemas for the
Anthropic-compatible /v1/messages endpoint, enabling clients like
Claude Code to communicate with rapid-mlx.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field

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

    Backport of upstream vLLM PR #42396 (v0.22.0). Mirrors the Anthropic
    SDK's ``output_config`` shape so SDKs can request structured output
    via ``output_config.format = json_schema`` without falling back to
    tool-call-emulated structured output.

    ``effort`` is accepted but intentionally NOT acted on here — it is
    part of a separate concurrent backport (Pick 1, reasoning-effort).
    Declaring the field today prevents the two PRs from racing on the
    same model shape during merge; the Pick 1 PR wires the value into
    the sampling cascade. Until then, any value the client supplies
    (low / medium / high / xhigh / max) is silently ignored.
    """

    format: AnthropicOutputFormat | None = None
    # Accepted-but-ignored: see class docstring. Pick 1 (separate PR)
    # will wire this into the sampling cascade.
    effort: str | None = None


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
    # /v1/messages via ``output_config.format = json_schema``. Optional;
    # absence preserves the pre-existing free-form text path so existing
    # SDK callers see no behavior change.
    output_config: AnthropicOutputConfig | None = None


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
