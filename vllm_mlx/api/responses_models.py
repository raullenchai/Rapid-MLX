# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI Responses API.

These models define the request and response schemas for the
OpenAI-compatible /v1/responses endpoint, enabling the official Codex CLI
(and other Responses-API clients) to talk to rapid-mlx as a local backend.

This is a stateless shim: ``previous_response_id`` is not supported and
the route returns 400 if set. Codex CLI re-sends the full conversation
history in ``input`` each turn, so statelessness is sufficient
(openai/codex#3841 confirms ``previous_response_id`` is not used by the
client).
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Request Models
# =============================================================================


class ResponsesContentItem(BaseModel):
    """A content item inside a Responses-API input message.

    Codex sends ``input_text`` for user/system turns and ``output_text``
    when echoing back prior assistant turns; ``input_image`` is the
    vision shape we mirror for future MLLM support.
    """

    type: str  # "input_text" | "output_text" | "input_image"
    text: str | None = None
    image_url: str | None = None


class ResponsesInputItem(BaseModel):
    """A single item in the Responses-API ``input`` array.

    The Responses API unifies user/assistant messages, function calls,
    function call outputs, and reasoning blocks into one polymorphic
    list. Codex CLI replays this full list each turn (no
    ``previous_response_id``).
    """

    type: (
        str  # "message" | "function_call" | "function_call_output" | "reasoning" | ...
    )
    # message
    role: str | None = None
    content: list[ResponsesContentItem] | str | None = None
    # function_call
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # function_call_output — Codex sometimes sends a structured shape,
    # sometimes a bare string. Both are coerced to str downstream.
    output: str | dict | list | None = None
    # reasoning — Codex emits these as ``encrypted_content`` blobs we
    # cannot decode; the adapter drops them entirely.
    summary: list[dict] | None = None
    encrypted_content: str | None = None


class ResponsesRequest(BaseModel):
    """Request body for ``POST /v1/responses``.

    Fields beyond ``model`` / ``input`` are declared so Pydantic does not
    silently drop them when Codex sends them. ``previous_response_id`` /
    ``store`` / ``include`` / ``service_tier`` / ``prompt_cache_key`` /
    ``metadata`` are accepted-but-ignored — same shape Anthropic compat
    uses for fields we know about but don't act on.
    """

    model: str
    # The Responses API allows either a bare prompt string OR an array
    # of polymorphic ``ResponsesInputItem`` blocks. Codex CLI sends the
    # array form with the full conversation history each turn.
    input: str | list[ResponsesInputItem]
    instructions: str | None = None  # rendered as system message
    tools: list[dict] | None = None  # Responses-FLAT shape
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict | None = None  # {"effort": "low|medium|high", "summary": ...}
    stream: bool = False
    store: bool | None = None
    include: list[str] | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    text: dict | None = None  # {"format": {...}, "verbosity": ...}
    metadata: dict | None = None
    previous_response_id: str | None = None  # 400 if set; this shim is stateless
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Per-request cap on reasoning tokens — see ``ChatCompletionRequest``
    # for the full semantic. ``None`` = no cap. Validated >= 1 by the
    # post-init validator below; the Responses route forwards this to
    # the underlying ChatCompletionRequest so the streaming SSE pipeline
    # and the non-streaming finalize path apply the same enforcement
    # (upstream vLLM PRs #20859 / #42396 / #43402 backport).
    reasoning_max_tokens: int | None = None

    @model_validator(mode="after")
    def _validate_reasoning_max_tokens(self) -> "ResponsesRequest":
        if self.reasoning_max_tokens is not None and self.reasoning_max_tokens < 1:
            raise ValueError(
                "reasoning_max_tokens must be >= 1 when set; pass "
                "enable_thinking=false to disable reasoning entirely."
            )
        return self


# =============================================================================
# Response Models
# =============================================================================


class ResponsesUsage(BaseModel):
    """Token usage block for a Responses-API response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Optional details for prompt cache + reasoning tokens. Codex parses
    # these fields and they're 1:1 with the OpenAI public spec.
    input_tokens_details: dict[str, int] | None = None
    output_tokens_details: dict[str, int] | None = None


class ResponsesOutputContent(BaseModel):
    """A content item inside an output ``message`` block."""

    type: str = "output_text"
    text: str = ""
    annotations: list[Any] = Field(default_factory=list)


class ResponsesOutputItem(BaseModel):
    """An item in the ``output`` array of a non-streaming response.

    Two shapes the shim emits:
    - ``message`` — assistant text reply, content array of output_text
    - ``function_call`` — one per tool call the model produced
    """

    type: str  # "message" | "function_call"
    id: str
    status: str = "completed"
    # message
    role: str | None = None
    content: list[ResponsesOutputContent] | None = None
    # function_call
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None


class ResponsesResponse(BaseModel):
    """Non-streaming response from ``POST /v1/responses``."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:24]}")
    object: str = "response"
    created_at: int = 0  # epoch seconds, populated by route
    model: str
    status: str = "completed"  # "completed" | "failed" | "incomplete"
    output: list[ResponsesOutputItem]
    usage: ResponsesUsage = Field(default_factory=ResponsesUsage)
    parallel_tool_calls: bool = False
    tool_choice: str | dict = "auto"
    tools: list[dict] = Field(default_factory=list)
    # Echoed back when client supplied them; ignored by Codex but on-spec.
    metadata: dict | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
