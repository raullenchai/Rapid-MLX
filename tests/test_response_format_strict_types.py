# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-103 — invalid ``response_format.type`` values
and malformed inner ``json_schema`` payloads used to silently slip
through the request-schema layer and HTTP 200 with no structure
enforcement.

The route-layer ``_validate_response_format`` helper in
``vllm_mlx/service/helpers.py`` already covers the ``type`` enum and
the "missing inner ``json_schema``" arm; this test file pins the
remaining silent-200 surface the helper did NOT cover:

* ``response_format.type`` = ``null`` / array / unknown string
  (also covered by the route helper today; pinned here so a future
  refactor that moves the gate into the schema layer alone keeps
  the contract).
* ``json_schema.schema`` is a non-dict (``42``, ``"hello"``,
  ``[1,2]``). Previously the typed ``ResponseFormatJsonSchema`` arm
  rejected these (schema_: dict), the Pydantic union fell back to
  the bare-``dict`` arm, and the route helper's
  ``if not json_schema_field.get("schema")`` check returned False
  for any truthy value — so the request continued without
  guided-generation enforcement. The desired contract: reject at
  Pydantic-parse time with a clean 422 / 400 naming the actual
  client-side type that was sent.

Validation is wired as a ``@field_validator(mode="before")`` on
``ChatCompletionRequest.response_format`` (and shared with
``_validate_response_format_raw`` in ``vllm_mlx/api/models.py``)
so the gate runs at FastAPI body-parse — never reaches any
downstream code that could leak raw Python type errors.
"""

import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    ResponseFormat,
    _validate_response_format_raw,
)


# ---------------------------------------------------------------------------
# Direct helper-level tests (cheap, no route invocation).
# ---------------------------------------------------------------------------


class TestValidateResponseFormatRaw:
    """``_validate_response_format_raw`` is the shared dict-arm guard.
    Each case pins one previously-silent-200 input shape."""

    def test_none_flows_through(self):
        assert _validate_response_format_raw(None) is None

    def test_typed_instance_flows_through(self):
        rf = ResponseFormat(type="json_object")
        assert _validate_response_format_raw(rf) is rf

    def test_bare_string_rejected(self):
        with pytest.raises(ValueError, match="must be an object"):
            _validate_response_format_raw("json")

    def test_bare_list_rejected(self):
        with pytest.raises(ValueError, match="must be an object"):
            _validate_response_format_raw(["json"])

    def test_missing_type_key_rejected(self):
        with pytest.raises(ValueError, match="type is required"):
            _validate_response_format_raw({})

    def test_type_null_rejected(self):
        with pytest.raises(ValueError, match="must be 'text'"):
            _validate_response_format_raw({"type": None})

    def test_type_unknown_string_rejected(self):
        with pytest.raises(ValueError, match="must be 'text'"):
            _validate_response_format_raw({"type": "xml"})

    def test_type_array_rejected(self):
        with pytest.raises(ValueError, match="must be 'text'"):
            _validate_response_format_raw(
                {"type": ["json_schema", "json_object"]}
            )

    def test_json_schema_type_missing_schema_field_rejected(self):
        with pytest.raises(
            ValueError, match="non-empty 'json_schema' field"
        ):
            _validate_response_format_raw({"type": "json_schema"})

    def test_json_schema_type_empty_dict_field_rejected(self):
        with pytest.raises(
            ValueError, match="non-empty 'json_schema' field"
        ):
            _validate_response_format_raw(
                {"type": "json_schema", "json_schema": {}}
            )

    @pytest.mark.parametrize(
        "bad_schema, type_name",
        [
            (42, "int"),
            ("hello", "str"),
            ([1, 2], "list"),
            (3.14, "float"),
        ],
    )
    def test_json_schema_schema_not_a_dict_rejected(
        self, bad_schema, type_name
    ):
        """F-103 silent-200 closer. ``json_schema.schema`` of any
        non-dict type used to fall through the bare-dict union arm and
        be silently swallowed (downstream
        ``extract_json_schema_for_guided`` bailed at the truthy-check
        for ints/strings; truthy strings produced literal-in-prompt
        garbage). Now → clean 422 naming the actual sent type."""
        with pytest.raises(ValueError) as ei:
            _validate_response_format_raw(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "x", "schema": bad_schema},
                }
            )
        msg = str(ei.value)
        assert "json_schema.schema must be an object" in msg
        assert f"got {type_name}" in msg

    def test_json_schema_empty_inner_schema_dict_rejected(self):
        """``schema:{}`` is technically dict-shaped but unconstrained
        in practice (matches anything) — mirror the typed arm's
        rejection of empty inner schema."""
        with pytest.raises(ValueError, match="non-empty object"):
            _validate_response_format_raw(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "x", "schema": {}},
                }
            )

    def test_valid_json_schema_passes(self):
        value = {
            "type": "json_schema",
            "json_schema": {
                "name": "r",
                "schema": {"type": "object"},
            },
        }
        assert _validate_response_format_raw(value) is value


# ---------------------------------------------------------------------------
# End-to-end through ChatCompletionRequest (the field_validator hookup).
# ---------------------------------------------------------------------------


def _minimal_request(**overrides):
    base = {
        "model": "qwen3-0.6b-8bit",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }
    base.update(overrides)
    return base


class TestChatCompletionRequestResponseFormat:
    """The ``@field_validator("response_format", mode="before")`` hookup
    must reject every silent-200 shape before the Pydantic Union
    resolves — i.e., a ``ValidationError`` must surface at parse
    time, not a successful coercion to the bare-dict arm."""

    @pytest.mark.parametrize(
        "rf",
        [
            {"type": "xml"},
            {"type": None},
            {"type": ["json_schema", "json_object"]},
            {"type": "json_schema"},
            {"type": "json_schema", "json_schema": {}},
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": 42},
            },
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": "hello"},
            },
            {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": [1, 2]},
            },
        ],
    )
    def test_invalid_response_format_rejected(self, rf):
        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest.model_validate(
                _minimal_request(response_format=rf)
            )
        msg = str(ei.value)
        # Pydantic's default error body carries the field name plus our
        # raised ``ValueError`` text — pin one or the other so a future
        # error-shape change still has at least the field-name handle.
        assert "response_format" in msg

    @pytest.mark.parametrize(
        "rf",
        [
            None,
            {"type": "text"},
            {"type": "json_object"},
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "r",
                    "schema": {"type": "object"},
                },
            },
            # Pydantic-typed instance also passes through.
            ResponseFormat(type="text"),
            ResponseFormat(type="json_object"),
        ],
    )
    def test_valid_response_format_accepted(self, rf):
        req = ChatCompletionRequest.model_validate(
            _minimal_request(response_format=rf)
        )
        # Sanity: the field is reachable, either as the typed model
        # (Pydantic coerced the dict into ``ResponseFormat``), as
        # the dict-arm fallback, or as ``None``.
        if rf is None:
            assert req.response_format is None
        elif isinstance(rf, ResponseFormat):
            # Typed input — Pydantic preserves the same instance shape.
            assert isinstance(req.response_format, ResponseFormat)
            assert req.response_format.type == rf.type
        else:
            # Dict input — typed arm wins for well-formed payloads.
            assert isinstance(
                req.response_format, (ResponseFormat, dict)
            )
            # ``type`` is reachable either way.
            rf_type = (
                req.response_format.type
                if isinstance(req.response_format, ResponseFormat)
                else req.response_format.get("type")
            )
            assert rf_type == rf["type"]
