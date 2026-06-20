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
import time
import uuid
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
    audio_url: AudioUrl | dict | str | None = None

    @model_validator(mode="after")
    def _validate_media_url_types(self) -> "ContentPart":
        """Reject non-string ``url`` inside multimodal-content dicts (F-066).

        The Union ``ImageUrl | dict | str | None`` falls back to ``dict``
        when ``url`` is not a string (Pydantic can't coerce e.g. ``int``
        into the strict-typed ``ImageUrl.url: str`` slot), so the
        malformed payload used to slip past the schema layer and crash
        deep inside ``process_image_input`` with
        ``'int' object has no attribute 'startswith'`` — Python type-
        error text leaked verbatim in the HTTP 400 body.

        Catching it at the schema layer means the client sees a clean
        ``image_url.url must be a string`` (mirrors the message pydantic
        itself would produce if the union allowed only ``ImageUrl |
        str``) — and the same check applies to the sister
        ``video_url`` / ``audio_url`` slots whose URL parsers share the
        same ``startswith`` hazard.
        """
        for field, label in (
            ("image_url", "image_url.url"),
            ("video_url", "video_url.url"),
            ("audio_url", "audio_url.url"),
        ):
            value = getattr(self, field, None)
            # Only the dict-fallback branch can carry a non-string ``url``
            # — the Pydantic-typed ``ImageUrl`` / ``VideoUrl`` / ``AudioUrl``
            # variants reject it at parse time, and the plain-``str``
            # variant is trivially well-typed.
            if isinstance(value, dict) and "url" in value:
                url = value["url"]
                if url is not None and not isinstance(url, str):
                    raise ValueError(f"{label} must be a string")
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
            for field, label in (
                ("image_url", "image_url.url"),
                ("video_url", "video_url.url"),
                ("audio_url", "audio_url.url"),
            ):
                value = item.get(field)
                if isinstance(value, dict) and "url" in value:
                    url = value["url"]
                    if url is not None and not isinstance(url, str):
                        raise ValueError(f"{label} must be a string")
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


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: bool | None = False

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
    if rf_type == "json_schema":
        json_schema_field = value.get("json_schema")
        if not json_schema_field:
            raise ValueError(
                "response_format.type='json_schema' requires "
                "non-empty 'json_schema' field"
            )
        if not isinstance(json_schema_field, dict):
            raise ValueError("response_format.json_schema must be an object")
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

    include_usage: bool = False  # Include usage stats in final chunk


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str = "default"
    messages: list[Message]
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
    top_k: int | None = None
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = None
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
    # Number of completions (only n=1 supported). F-155: the route used
    # to reject ``n > 1`` only, so ``n=0`` / ``n=-1`` slipped through
    # and HTTP 200'd with a single choice — asymmetric with the
    # ``n > 1`` 400. The ``_validate_n`` field_validator below pins
    # the contract: omitted (``None``) or ``1`` is the only legal
    # surface; anything else (negative, zero, or > 1) → 422.
    n: int | None = None

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


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = (
        None  # Reasoning/thinking content (when --reasoning-parser is used)
    )
    tool_calls: list[ToolCall] | None = None

    def model_post_init(self, __context) -> None:
        """Add deprecated 'reasoning' alias for backward compatibility."""
        pass

    @model_serializer(mode="wrap")
    def _serialize_assistant_message(self, handler):
        """Always emit ``content`` (and ``reasoning`` alias) on the wire.

        Per OpenAI's ``chat.completion`` schema, ``message.content`` is a
        REQUIRED field that is ``string`` or ``null`` — never absent. When
        a reasoning model truncates inside ``<think>`` the parser yields
        ``content=None`` and our callers serialize the response with
        ``model_dump_json(exclude_none=True)``; pydantic then drops the
        ``content`` key entirely and clients that read
        ``resp["choices"][0]["message"]["content"]`` (the standard
        OpenAI SDK pattern) crash with ``KeyError: 'content'``. This
        wrap-mode serializer runs AFTER ``exclude_none`` pruning, so we
        can put the field back as an explicit ``None`` (→ JSON ``null``)
        regardless of how the parent was dumped.

        Also forwards the deprecated ``reasoning`` alias for
        backward-compat clients that read either field. (The legacy
        ``model_dump`` override below covered direct ``.model_dump()``
        calls but was bypassed when a parent's ``model_dump_json``
        recursed — pydantic v2 routes JSON serialization through this
        ``@model_serializer`` instead.)
        """
        d = handler(self)
        # OpenAI contract: ``content`` is always present (string|null).
        if "content" not in d:
            d["content"] = None
        if "reasoning_content" in d:
            d["reasoning"] = d["reasoning_content"]
        return d

    def model_dump(self, **kwargs) -> dict:
        """Include 'reasoning' as alias of reasoning_content for clients expecting it.

        Kept for callers that invoke ``.model_dump()`` directly (rather
        than via a parent ``model_dump_json``). The wrap-mode
        ``@model_serializer`` above already handles both paths, but this
        override remains a defensive belt-and-braces for any external
        caller relying on the historical behaviour.
        """
        d = super().model_dump(**kwargs)
        # Belt-and-braces: ensure ``content`` is always present, matching
        # the OpenAI-spec invariant enforced by the wrap-mode serializer.
        if "content" not in d:
            d["content"] = None
        # Add backward-compat alias — clients may read either field
        if "reasoning_content" in d:
            d["reasoning"] = d["reasoning_content"]
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
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters — see #355 + the
    # matching block on ChatCompletionRequest for wiring + caveats.
    top_k: int | None = None
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = None
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

    # F-011: NaN/inf scrub + finite belt-and-braces, exactly mirroring
    # ChatCompletionRequest. See the rationale block at the top of
    # this module + the matching validators on the chat schema.
    @model_validator(mode="before")
    @classmethod
    def _scrub_nonfinite_sampling(cls, data):
        return _scrub_nonfinite_sampling_raw(data)

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


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "wav"


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

        Mirror the OpenAI on-the-wire shape: surface ``content: null``
        on any delta that carries ``reasoning_content`` (or
        ``tool_calls``) but no visible content, so the field is
        addressable on every reasoning-bearing delta — including the
        final one. Normal pure-content / pure-role / empty deltas keep
        their current minimal shape, so the per-token streaming budget
        is unchanged for non-reasoning paths.
        """
        d = handler(self)
        if "content" not in d and ("reasoning_content" in d or "tool_calls" in d):
            d["content"] = None
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
