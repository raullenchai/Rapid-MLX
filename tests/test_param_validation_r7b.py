# SPDX-License-Identifier: Apache-2.0
"""
r7-B cross-route validation hardening (R7-H4 / R7-H5 / R7-M3 / R7-M4).

Vlad r7 (rapid-desktop /tmp/dogfood-088/vlad/vlad-r1.md, vlad-r2.md)
surfaced four validation-gap bugs on v0.8.8. r5-E (PR #824) had
already hardened the top-level chat / completions surfaces but
``/v1/responses`` and ``/v1/messages`` slipped through. The fixes
are systemic — shared validators + shared types — so the four
schema-level tests below exercise the four request models directly
via Pydantic, the same gate every route hits before the handler
runs.

  * **R7-H4** (``stream_options.include_usage`` accepts any value on
    /v1/responses + /v1/messages): the bypass was a missing field
    declaration on ``ResponsesRequest`` / ``AnthropicRequest`` — the
    chat + completions models DECLARED ``StreamOptions`` (whose
    ``StrictBool`` field validator catches the truthy-string form);
    Responses/Anthropic silently dropped the unknown field and
    HTTP-200'd. Fix declares the same shared ``StreamOptions`` type
    on both surfaces so the strict-bool gate runs through one
    validator.

  * **R7-H5** (legacy chat ``response_format`` strict bypass when
    ``strict`` is at the OUTER level): pre-fix
    ``is_strict_json_schema`` only inspected the canonical
    ``response_format.json_schema.strict`` slot. Clients writing
    against the Responses-API ``text.format.strict`` shape and then
    pointing at /v1/chat/completions kept ``strict`` at the OUTER
    level (a sibling of ``type``). Pydantic silently dropped the
    unknown field on the typed ``ResponseFormat`` model, and the
    legacy chat path HTTP-200'd without the [guided]-required gate
    ever firing. Fix declares ``strict`` as an outer-level field on
    ``ResponseFormat`` and extends ``is_strict_json_schema`` to fire
    on either nesting position.

  * **R7-M3** (``max_output_tokens=-5`` silently accepted on
    /v1/responses; ``max_tokens=-5`` on /v1/completions and
    /v1/messages): only the chat route had a hand-rolled
    ``max_tokens < 1`` guard; the schema layer accepted any int.
    Fix consolidates a shared ``_validate_positive_int`` validator
    + ``mode="before"`` field-validator on every numeric token-budget
    field across all four request models so the rejection is
    schema-layer, uniform, and route-independent.

  * **R7-M4** (``response_format={"type":"json_object"}`` wraps body
    in markdown fence on /v1/responses): the chat route's
    ``extract_json_from_response`` fence-strip was never wired into
    the /v1/responses non-stream path. Fix calls the same helper
    after ``sanitize_output`` so the same model+prompt produces a
    consistently-stripped body on both surfaces. The schema layer
    test below covers the strict-mode contract; the fence-strip
    helper itself has direct coverage in
    ``test_api_utils.py``.
"""

import pytest
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponseFormat,
    StreamOptions,
)
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.api.tool_calling import is_strict_json_schema
from vllm_mlx.api.utils import extract_json_from_response


def _user_msg():
    return [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# R7-H4: stream_options.include_usage strict-bool across ALL request surfaces
# ---------------------------------------------------------------------------


class TestStreamOptionsIncludeUsageCrossRoute:
    """The strict-bool gate must fire from every OpenAI-compat surface
    (chat / completions / responses / messages). r5-E (B-9) closed the
    chat + completions arms; r7-B closes the Responses + Anthropic arms
    by declaring the SAME shared ``StreamOptions`` field on those models.
    """

    # The four request surfaces in one matrix — parametrize so a future
    # surface (or a regression on any of these) is one test line, not a
    # new copy-paste class. Each entry: (Model, required-kwargs).
    _SURFACES = [
        (ChatCompletionRequest, {"messages": _user_msg()}),
        (CompletionRequest, {"prompt": "hi"}),
        (ResponsesRequest, {"input": "hi"}),
        (AnthropicRequest, {"messages": _user_msg(), "max_tokens": 5}),
    ]

    @pytest.mark.parametrize("Model,extra", _SURFACES)
    @pytest.mark.parametrize(
        "bad_value",
        ["yes", "true", "no", "false", "1", "0", "on", "off", 1, 0],
    )
    def test_non_bool_include_usage_rejected_every_route(self, Model, extra, bad_value):
        with pytest.raises(ValidationError) as excinfo:
            Model(
                model="x",
                stream_options={"include_usage": bad_value},
                **extra,
            )
        msg = str(excinfo.value)
        # Field name is in either loc or msg — both are fine for a
        # client surfacing the error.
        assert "include_usage" in msg

    @pytest.mark.parametrize("Model,extra", _SURFACES)
    @pytest.mark.parametrize("good_value", [True, False])
    def test_proper_bool_include_usage_accepted_every_route(
        self, Model, extra, good_value
    ):
        req = Model(
            model="x",
            stream_options={"include_usage": good_value},
            **extra,
        )
        assert req.stream_options is not None
        assert req.stream_options.include_usage is good_value

    @pytest.mark.parametrize("Model,extra", _SURFACES)
    def test_stream_options_omitted_accepted_every_route(self, Model, extra):
        """The field is optional — omitting it is the spec default and
        MUST NOT 4xx (no regression on existing clients)."""
        req = Model(model="x", **extra)
        assert req.stream_options is None

    @pytest.mark.parametrize("Model,extra", _SURFACES)
    def test_one_shared_stream_options_type_every_route(self, Model, extra):
        """The validator is consolidated by SHARING the same
        ``StreamOptions`` type across all four surfaces — otherwise the
        next surface added would silently bypass the gate. This pin
        guards against accidental schema duplication on a future
        refactor."""
        req = Model(
            model="x",
            stream_options={"include_usage": True},
            **extra,
        )
        # All routes' stream_options field is the SAME class.
        assert isinstance(req.stream_options, StreamOptions)


# ---------------------------------------------------------------------------
# R7-H5: outer-level strict on legacy chat response_format
# ---------------------------------------------------------------------------


class TestResponseFormatStrictOuterLevel:
    """``response_format = {"type":"json_schema","strict":true,
    "json_schema":{"name":"x","schema":{...}}}`` — strict at the OUTER
    level, the Responses-API ``text.format.strict`` shape clients
    sometimes carry over to /v1/chat/completions. Pre-r7-B Pydantic
    dropped the unknown field and ``is_strict_json_schema`` returned
    False — silent 200 with no [guided]-required gate. Post-fix the
    field is declared on ``ResponseFormat`` AND
    ``is_strict_json_schema`` recognizes it on the typed-model arm
    too.
    """

    _SCHEMA = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }

    def test_typed_response_format_outer_strict_recognized(self):
        rf = ResponseFormat(
            type="json_schema",
            strict=True,
            json_schema={"name": "p", "schema": self._SCHEMA},
        )
        # Pre-r7 this was silently dropped (Pydantic ignored the unknown
        # field). Post-r7 it round-trips as a typed field.
        assert rf.strict is True
        assert is_strict_json_schema(rf) is True

    def test_typed_response_format_inner_strict_recognized(self):
        """Canonical OpenAI nesting — must still work (no regression)."""
        rf = ResponseFormat(
            type="json_schema",
            json_schema={"name": "p", "strict": True, "schema": self._SCHEMA},
        )
        assert is_strict_json_schema(rf) is True

    def test_typed_response_format_no_strict_anywhere_not_strict(self):
        rf = ResponseFormat(
            type="json_schema",
            json_schema={"name": "p", "schema": self._SCHEMA},
        )
        # Neither outer nor inner — non-strict mode.
        assert rf.strict is None
        assert is_strict_json_schema(rf) is False

    def test_dict_response_format_outer_strict_recognized(self):
        """Bare-dict arm of the ``ResponseFormat | dict`` union — same
        contract: outer-level strict fires the gate."""
        rf_dict = {
            "type": "json_schema",
            "strict": True,
            "json_schema": {"name": "p", "schema": self._SCHEMA},
        }
        assert is_strict_json_schema(rf_dict) is True

    def test_dict_response_format_outer_strict_string_not_strict(self):
        """``strict:"true"`` (string, NOT bool) must NOT enable strict
        mode — codex r1 NIT precedent: ``strict is True`` identity
        check, not ``bool(strict)``."""
        rf_dict = {
            "type": "json_schema",
            "strict": "true",  # malformed payload
            "json_schema": {"name": "p", "schema": self._SCHEMA},
        }
        assert is_strict_json_schema(rf_dict) is False

    def test_chat_request_outer_strict_passes_to_strict_detector(self):
        """End-to-end: a /v1/chat/completions request body with
        outer-level strict reaches the detector as strict. Pre-r7 this
        was the silent-bypass path."""
        req = ChatCompletionRequest(
            model="x",
            messages=_user_msg(),
            response_format={
                "type": "json_schema",
                "strict": True,
                "json_schema": {"name": "p", "schema": self._SCHEMA},
            },
        )
        assert is_strict_json_schema(req.response_format) is True

    def test_json_object_outer_strict_not_a_constraint_path(self):
        """``response_format.type == "json_object"`` with outer-level
        ``strict:true`` is nonsensical (json_object has no schema to
        enforce strictly) but must NOT trip the json_schema gate —
        return False so the route doesn't 400 on a degenerate-but-
        harmless payload."""
        rf = ResponseFormat(type="json_object", strict=True)
        assert is_strict_json_schema(rf) is False


# ---------------------------------------------------------------------------
# R7-M3: shared >= 1 gate on max_tokens / max_output_tokens / max_completion_tokens
# ---------------------------------------------------------------------------


class TestPositiveIntGenerationBudget:
    """The token-budget fields (``max_tokens`` on chat / completions /
    messages, ``max_output_tokens`` on /v1/responses,
    ``max_completion_tokens`` on chat) all share one validator now so
    the contract is uniform across surfaces. Pre-r7 only the chat
    route had a route-level guard; the schema layer accepted -5 and
    HTTP-200'd with a single token (or status="incomplete" on
    /v1/responses). Mirrors the seed validator's cross-route shape.
    """

    # Each entry: (Model, required-kwargs, token-field-name).
    _FIELD_MATRIX = [
        (ChatCompletionRequest, {"messages": _user_msg()}, "max_tokens"),
        (
            ChatCompletionRequest,
            {"messages": _user_msg()},
            "max_completion_tokens",
        ),
        (CompletionRequest, {"prompt": "hi"}, "max_tokens"),
        (ResponsesRequest, {"input": "hi"}, "max_output_tokens"),
        (AnthropicRequest, {"messages": _user_msg()}, "max_tokens"),
    ]

    @pytest.mark.parametrize("Model,extra,field", _FIELD_MATRIX)
    @pytest.mark.parametrize("bad_value", [-1, -5, -(2**31), 0])
    def test_non_positive_token_budget_rejected_every_route(
        self, Model, extra, field, bad_value
    ):
        with pytest.raises(ValidationError) as excinfo:
            Model(model="x", **{**extra, field: bad_value})
        msg = str(excinfo.value)
        assert field in msg
        assert ">= 1" in msg or "must be >= 1" in msg

    @pytest.mark.parametrize("Model,extra,field", _FIELD_MATRIX)
    @pytest.mark.parametrize("good_value", [1, 5, 1024])
    def test_positive_token_budget_accepted_every_route(
        self, Model, extra, field, good_value
    ):
        req = Model(model="x", **{**extra, field: good_value})
        assert getattr(req, field) == good_value

    @pytest.mark.parametrize("Model,extra,field", _FIELD_MATRIX)
    def test_token_budget_bool_rejected_every_route(self, Model, extra, field):
        """``True``/``False`` are int subclasses in Python — without
        the ``mode="before"`` gate Pydantic would coerce ``True`` → 1
        and ``False`` → 0 silently. Both must 4xx (bool is a
        serialization mistake on a numeric field, same family as the
        ``n: true`` / ``seed: true`` footguns)."""
        with pytest.raises(ValidationError):
            Model(model="x", **{**extra, field: True})
        with pytest.raises(ValidationError):
            Model(model="x", **{**extra, field: False})

    @pytest.mark.parametrize("Model,extra,field", _FIELD_MATRIX)
    def test_token_budget_string_rejected_every_route(self, Model, extra, field):
        """JSON-string ints (``"100"``) are a wire-form bug — every
        spec lists the field as a plain integer, and the schema must
        4xx rather than lax-coerce."""
        with pytest.raises(ValidationError):
            Model(model="x", **{**extra, field: "100"})

    @pytest.mark.parametrize(
        "Model,extra,field",
        [
            (ChatCompletionRequest, {"messages": _user_msg()}, "max_tokens"),
            (
                ChatCompletionRequest,
                {"messages": _user_msg()},
                "max_completion_tokens",
            ),
            (CompletionRequest, {"prompt": "hi"}, "max_tokens"),
            (ResponsesRequest, {"input": "hi"}, "max_output_tokens"),
        ],
    )
    def test_token_budget_omitted_accepted_every_route(self, Model, extra, field):
        """Token budget is optional on the OpenAI-compat surfaces;
        absent means "server default". Anthropic surface excluded —
        ``max_tokens`` is REQUIRED per the upstream spec, the absence
        case is covered by the type-checker (``int``, not ``int | None``).
        """
        req = Model(model="x", **extra)
        assert getattr(req, field) is None


# ---------------------------------------------------------------------------
# R7-M4: fence-strip helper covers ```json fenced bodies (the systemic
# load-bearing piece). The /v1/responses route now calls the SAME helper
# the chat route already uses — so the cross-route inconsistency is closed.
# ---------------------------------------------------------------------------


class TestFenceStripHelper:
    """``extract_json_from_response`` is the single fence-strip
    helper the chat + responses routes both call when a structured
    response_format was requested. Direct coverage so a future
    refactor that drops the fence-detection branch trips CI before
    it lands."""

    def test_json_fence_peeled(self):
        fenced = '```json\n{"name": "Alice"}\n```'
        assert extract_json_from_response(fenced) == '{"name": "Alice"}'

    def test_plain_fence_peeled(self):
        fenced = '```\n{"name": "Alice"}\n```'
        assert extract_json_from_response(fenced) == '{"name": "Alice"}'

    def test_unfenced_json_passthrough(self):
        unfenced = '{"name": "Alice"}'
        assert extract_json_from_response(unfenced) == unfenced

    def test_no_json_passthrough(self):
        """The helper is a defensive normalizer — when the text has
        no JSON to peel, it returns the input unchanged so a
        non-structured-output response is never mangled."""
        plain = "Hello, world!"
        assert extract_json_from_response(plain) == plain

    def test_empty_input_returns_empty(self):
        assert extract_json_from_response("") == ""
