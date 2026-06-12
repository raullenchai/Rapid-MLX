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
