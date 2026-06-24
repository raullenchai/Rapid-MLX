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
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import (
    _TOP_K_SENTINEL_CAP,
    _VALID_REASONING_EFFORTS,
    StreamOptions,
    _validate_nonnegative_int,
    _validate_positive_int,
    _validate_reasoning_effort,
    _validate_seed,
)

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
    # R7-H4: OpenAI streaming-spec parity — Responses API also accepts
    # ``stream_options.include_usage`` per OpenAI SDK ``stream_options``
    # shape. Pre-R7-H4 the field was undeclared on this surface, so
    # Pydantic silently dropped any value the client sent; that gave
    # ``stream_options={"include_usage":"yes"}`` an HTTP-200 free pass
    # while the chat / completions surfaces (which DO declare the field
    # via ``StreamOptions`` with a ``StrictBool``) correctly 422'd.
    # Declaring it here with the SAME shared ``StreamOptions`` model
    # routes the strict-bool gate through the same validator so the
    # contract is uniform across all four routes (chat / completions /
    # messages / responses). The Responses route does not currently
    # emit a trailing-usage SSE chunk — the field is accepted-but-
    # ignored on this surface (parity with ``previous_response_id`` /
    # ``store`` / ``include`` etc.); the strict-bool gate is the
    # load-bearing piece for the r7 sweep.
    stream_options: StreamOptions | None = None
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
    # Yuki R6 (0.8.5 dogfood): OpenAI Responses spec defines
    # ``truncation`` as ``"auto" | "disabled"``. rapid-mlx accepts and
    # echoes the requested value back on the response envelope; the
    # engine-level truncation behaviour is a no-op in this release (the
    # context-length gate already rejects oversized prompts upstream).
    # NOTE: implement actual auto-truncation in a follow-up — operator
    # preference (0.8 dogfood r4) is to echo + no-op so migrating
    # clients don't see a silent drop while the implementation lands.
    #
    # Codex r3 NIT (PR #817): ``Literal[...]`` so typos like
    # ``"enabled"`` produce a Pydantic 400 instead of silently
    # round-tripping as if they were valid.
    truncation: Literal["auto", "disabled"] | None = None
    # Per-request cap on reasoning tokens — see ``ChatCompletionRequest``
    # for the full semantic. ``None`` = no cap. Validated >= 1 by the
    # post-init validator below; the Responses route forwards this to
    # the underlying ChatCompletionRequest so the streaming SSE pipeline
    # and the non-streaming finalize path apply the same enforcement
    # (upstream vLLM PRs #20859 / #42396 / #43402 backport).
    reasoning_max_tokens: int | None = None
    # H-11: OpenAI Responses API exposes ``seed`` on its own surface —
    # without declaring it here Pydantic drops it before the adapter
    # converts to ``ChatCompletionRequest``.
    #
    # Codex round-4 BLOCKING fix: apply the SAME ``mode="before"``
    # bool/non-int guard the chat schema uses, because the conversion
    # path (``ResponsesRequest.seed: True`` → Pydantic coerces to ``1``
    # → ``responses_to_openai`` passes ``1`` to ChatCompletionRequest →
    # ChatCompletionRequest sees a legitimate ``int=1``) silently
    # swallows the bool. Validating AT THIS LAYER closes the bypass
    # so the contract is enforced regardless of which surface the
    # client hit. See ``api/models.py::_validate_seed`` for the
    # rationale block.
    #
    # Codex round-6 BLOCKING fix: removed the ``Field(ge=0,
    # le=0xFFFFFFFF)`` bound so the Responses surface accepts the full
    # OpenAI-documented integer range and uint32 narrowing happens
    # downstream in ``make_seeded_sampler`` (parity with the chat /
    # legacy completion surfaces).
    seed: int | None = None
    # r6-A R6-H8: ``top_k`` upper-bound gate on the Responses surface.
    # r5-E B-7 (PR #824) landed the shared ``_validate_nonnegative_int``
    # validator + ``_TOP_K_SENTINEL_CAP = 1 << 20`` ceiling on
    # ChatCompletionRequest and CompletionRequest, but Responses never
    # declared the field at all — Pydantic silently dropped any
    # ``top_k`` the client sent (e.g. ``999999999``) and the route then
    # generated with the engine's default sampler. From the SDK
    # consumer's perspective the value was silently accepted (HTTP 200
    # with no validation error), which is the exact silent-correctness
    # hazard r5-E exists to close. Mirroring the chat/completion schema
    # here (single shared validator, single shared ceiling) keeps the
    # three OpenAI-surface lanes (chat / responses / legacy completions)
    # under one contract — no copy-pasted thresholds to drift.
    top_k: int | None = None
    # R10-H5 (R9-H3 carry) — OpenAI Responses spec exposes
    # ``reasoning.effort`` (nested under the ``reasoning`` dict). Some
    # SDK clients also send the chat-completions-shape top-level
    # ``reasoning_effort`` field; declare it here so Pydantic stops
    # silently dropping it. Both surfaces are validated by the
    # ``_validate_reasoning_effort_*`` validators below — the top-level
    # field via field_validator (closed-set string), and the nested
    # ``reasoning.effort`` via a model_validator (mode="before") that
    # runs the same check on the dict member. Sven r10-R1 + vlad
    # r10-R1: every value (int / list / null / case-variant / garbage
    # string) 200'd on this surface pre-fix.
    reasoning_effort: str | None = None
    # R12-M2 (Mira r12 / finding R-1) — surface parity with
    # ``ChatCompletionRequest`` for the two thinking-control knobs.
    # Pre-fix /v1/responses had no declared field for either, so a
    # client sending ``chat_template_kwargs={"enable_thinking":false}``
    # (the OpenAI-extension shape Qwen / DeepSeek-R1 honor) had the key
    # silently dropped by Pydantic, then the route's
    # ``_resolve_enable_thinking`` consult returned ``None`` (template
    # default = True for the Qwen3 family) and the model burned the
    # whole token budget inside ``<think>`` before emitting JSON. The
    # strict-json_schema path then 422'd with ``reason:"invalid_json"``
    # on a perfectly valid happy-path prompt — a soft-broken state
    # for SDK consumers because the envelope incorrectly suggests
    # their schema is at fault. Declaring the fields here makes the
    # passthrough first-class on this surface (parity with chat) and
    # ``responses_to_openai`` forwards them onto the materialized
    # ``ChatCompletionRequest`` so the existing
    # ``_resolve_enable_thinking`` helper resolves them identically
    # to /v1/chat/completions.
    #
    # ``chat_template_kwargs`` is intentionally typed as a dict — the
    # only key rapid-mlx consumes today is ``enable_thinking`` (other
    # keys round-trip but no-op), and the chat surface uses the same
    # unconstrained ``dict`` shape (api/models.py line 1511) so the
    # two routes share one contract.
    chat_template_kwargs: dict | None = None
    # ``enable_thinking`` is the rapid-mlx convenience top-level knob;
    # the OpenAI-extension shape ``chat_template_kwargs.enable_thinking``
    # wins when both are set (see service/helpers.py
    # ``_extract_thinking_from_request`` precedence).
    enable_thinking: bool | None = None

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed_field(cls, v) -> int | None:
        return _validate_seed(v)

    # R10-H5 (R9-H3 carry) — mirror of
    # ``ChatCompletionRequest._validate_reasoning_effort_field``. Closed
    # set; nulls flow through.
    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _validate_reasoning_effort_field(cls, v):
        return _validate_reasoning_effort(v)

    # R10-H5 (R9-H3 carry) — canonical Responses-spec surface validates
    # ``reasoning.effort`` (nested). Runs in ``mode="before"`` so the
    # raw payload is gated BEFORE the ``reasoning`` dict is coerced
    # onto the schema. Closed-set ``_VALID_REASONING_EFFORTS`` matches
    # the top-level shorthand so both wire forms accept the same set.
    @model_validator(mode="before")
    @classmethod
    def _validate_reasoning_dict_effort(cls, data):
        if not isinstance(data, dict):
            return data
        reasoning = data.get("reasoning")
        if not isinstance(reasoning, dict):
            return data
        if "effort" not in reasoning:
            return data
        effort = reasoning["effort"]
        if effort is None:
            return data
        if isinstance(effort, str) and effort in _VALID_REASONING_EFFORTS:
            return data
        raise ValueError(
            "reasoning.effort must be one of "
            f"{list(_VALID_REASONING_EFFORTS)} or null "
            f"(got {type(effort).__name__}={effort!r})."
        )

    @field_validator("top_k", mode="before")
    @classmethod
    def _validate_top_k(cls, v) -> int | None:
        return _validate_nonnegative_int(
            v, max_value=_TOP_K_SENTINEL_CAP, field_name="top_k"
        )

    # R7-M3: shared ``>= 1`` gate on ``max_output_tokens``. Pre-R7-M3
    # ``max_output_tokens=-5`` HTTP-200'd on /v1/responses with
    # ``status="incomplete"`` — silent-correctness hazard same shape
    # as ``seed=-1`` (which DOES 400). The chat route had its own
    # hand-rolled ``max_tokens < 1`` check; the schema-level
    # validator means /v1/responses / /v1/completions / /v1/messages
    # all share one contract now.
    @field_validator("max_output_tokens", mode="before")
    @classmethod
    def _validate_max_output_tokens(cls, v) -> int | None:
        return _validate_positive_int(v, field_name="max_output_tokens")

    @model_validator(mode="before")
    @classmethod
    def _validate_reasoning_max_tokens_raw(cls, data):
        """Strict type-and-range check on ``reasoning_max_tokens``
        BEFORE Pydantic coercion. Mirror of the same validator on
        ``ChatCompletionRequest`` so the three API surfaces
        (/v1/chat/completions, /v1/responses, /v1/messages) share one
        contract — codex round-3 NIT #5. See the ChatCompletionRequest
        validator for the full rationale.
        """
        if not isinstance(data, dict):
            return data
        if "reasoning_max_tokens" not in data:
            return data
        v = data["reasoning_max_tokens"]
        if v is None:
            return data
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
    def _validate_input_nonempty(self) -> "ResponsesRequest":
        """D-ANTHRO-VALIDATION F11 sibling — reject an empty ``input``.

        ``input=[]`` (and ``input=""``) pre-fix slipped past the schema
        and the downstream adapter then crashed dereferencing an empty
        list / running a no-token prompt through the engine. Anthropic-
        parity surface: same shape rejected at the schema layer with a
        clear 400 instead of a 500.
        """
        if isinstance(self.input, str):
            if self.input == "":
                raise ValueError(
                    "`input` must be a non-empty string or a non-empty "
                    "list of input items."
                )
        elif isinstance(self.input, list):
            if len(self.input) == 0:
                raise ValueError("`input` must be a non-empty list of input items.")
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

    Four shapes the shim emits:
    - ``message`` — assistant text reply, content array of output_text
    - ``function_call`` — one per tool call the model produced
    - ``reasoning`` — top-level reasoning summary (Yuki F4 / R10);
      emitted alongside ``message`` when the model produced reasoning
      (cross-lane parity with /v1/chat/completions ``message.reasoning_content``).
    - ``computer_call`` — UI-TARS / Computer-Use action call (Ana C-06);
      emitted instead of ``function_call`` when the request supplied
      ``tools=[{type:"computer_20251022", ...}]`` and the underlying
      parser surfaced a ``computer``-tool call.
    """

    type: str  # "message" | "function_call" | "reasoning" | "computer_call"
    id: str
    status: str = "completed"
    # message
    role: str | None = None
    content: list[ResponsesOutputContent] | None = None
    # function_call
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # reasoning — OpenAI Responses spec, output[i].type=="reasoning":
    #   {"type":"reasoning","id":"rs_...","summary":[{"type":"summary_text","text":"..."}]}
    # ``encrypted_content`` is omitted unless include=["reasoning.encrypted_content"]
    # is requested AND the backend produces one. rapid-mlx is stateless,
    # so the field is always absent today.
    summary: list[dict] | None = None
    encrypted_content: str | None = None
    # computer_call — Computer-Use action shape, populated by translating
    # the UI-TARS tool_call (``name="computer"``, JSON arguments) into the
    # OpenAI ``computer_call`` envelope.
    action: dict | None = None
    pending_safety_checks: list[dict] | None = None


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
    # Yuki R6 / R7 (0.8.5 dogfood): the OpenAI Responses spec exposes
    # ``truncation`` and ``service_tier`` as response-envelope fields.
    # ``truncation`` is echoed (today no-op'd at the engine level — see
    # ``ResponsesRequest`` docstring), ``service_tier`` is echoed as
    # the requested value so clients see the contract round-trip. Both
    # default to ``None`` so non-strict SDKs that ignore them keep
    # working. ``truncation`` is ``Literal`` so the request-side
    # validator's contract carries over to the response shape too.
    truncation: Literal["auto", "disabled"] | None = None
    service_tier: str | None = None
    # R11-B (R11-M-F1): structured truncation block. When ``status ==
    # "incomplete"`` because the engine reported ``finish_reason="length"``,
    # this carries ``{"reason": "max_output_tokens"}`` so SDK consumers
    # (Codex CLI, openai-python) can distinguish a budget-exhaust
    # truncation from a stop-sequence / EOS completion. Mirrors the
    # OpenAI Responses spec ``Response.incomplete_details`` field;
    # left ``None`` for ``completed`` and ``failed`` envelopes.
    incomplete_details: dict | None = None
