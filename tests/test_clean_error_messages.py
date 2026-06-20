# SPDX-License-Identifier: Apache-2.0
"""Regression tests for F-013 + F-066 — raw Python AttributeError /
TypeError text used to leak into HTTP 400 response bodies for two
distinct malformed-payload classes:

* F-013: ``response_format={"type":"json_schema"}`` (missing inner
  ``json_schema`` field) crashed inside ``build_json_system_prompt`` with
  ``AttributeError: 'NoneType' object has no attribute 'get'`` — text
  surfaced verbatim in the 400 body. Sister silent-200 cases
  (``type:"xml"`` / ``type:""`` / ``{}`` / ``json_schema:{}``) were
  silently accepted as unconstrained text generation.

* F-066: ``image_url.url=123`` (or any non-string) crashed inside
  ``process_image_input`` → ``is_base64_image`` with
  ``AttributeError: 'int' object has no attribute 'startswith'`` — text
  surfaced verbatim in the 400 body.

Both share the same class: provider-side Python type errors leaking
through ``except Exception → HTTPException(detail=str(e))`` paths
without input-shape validation upstream. The fixes pin schema-layer
and route-layer validators so the malformed payload is rejected before
any Python ``getattr``/``startswith`` is attempted.
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# F-013: response_format validation (route-layer)
# ---------------------------------------------------------------------------


class TestResponseFormatValidation:
    """``_validate_response_format`` runs in
    ``_create_chat_completion_impl`` BEFORE any code that might raise
    ``AttributeError`` on a malformed payload. Each test pins one
    F-013 reproduction case."""

    def test_json_schema_type_without_field_raises_clean_400(self):
        """The original raw-leak case. ``type:"json_schema"`` with no
        ``json_schema`` field used to fall through to
        ``build_json_system_prompt`` which called
        ``rf_dict.get("json_schema", {}).get("schema", {})`` — the inner
        ``.get`` raised ``AttributeError: 'NoneType' object has no
        attribute 'get'`` because the field defaults to ``None``, not
        ``{}``. Wrapped by ``except Exception: raise
        HTTPException(detail=str(e))`` and the raw error string
        surfaced in the response body."""
        from vllm_mlx.service.helpers import _validate_response_format
        from vllm_mlx.api.models import ResponseFormat

        with pytest.raises(HTTPException) as ei:
            _validate_response_format(ResponseFormat(type="json_schema"))
        assert ei.value.status_code == 400
        # Must NOT contain the raw Python AttributeError text.
        assert "NoneType" not in ei.value.detail
        assert "attribute" not in ei.value.detail
        # Must contain the actionable hint.
        assert "json_schema" in ei.value.detail
        assert "non-empty" in ei.value.detail

    def test_json_schema_type_with_name_but_no_schema_raises_400(self):
        """Codex r1 BLOCKING follow-up: ``{"type":"json_schema",
        "json_schema":{"name":"r"}}`` has a non-empty outer dict but no
        inner ``schema`` member, so the prior round of the gate let it
        through and ``extract_json_schema_for_guided`` still bailed at
        ``if not schema: return None`` — request proceeded with no
        constraint. Now → 400 naming the missing inner member."""
        from vllm_mlx.service.helpers import _validate_response_format

        with pytest.raises(HTTPException) as ei:
            _validate_response_format(
                {"type": "json_schema", "json_schema": {"name": "r"}}
            )
        assert ei.value.status_code == 400
        assert "json_schema.schema" in ei.value.detail
        assert "non-empty" in ei.value.detail

    def test_json_schema_type_with_empty_schema_member_raises_400(self):
        """Same shape as above but with ``schema:{}`` — an empty inner
        schema is functionally equivalent to no schema at all."""
        from vllm_mlx.service.helpers import _validate_response_format

        with pytest.raises(HTTPException) as ei:
            _validate_response_format(
                {
                    "type": "json_schema",
                    "json_schema": {"name": "r", "schema": {}},
                }
            )
        assert ei.value.status_code == 400
        assert "json_schema.schema" in ei.value.detail

    def test_json_schema_type_with_empty_dict_field_raises_400(self):
        """``response_format={"type":"json_schema","json_schema":{}}``
        used to be silently accepted as HTTP 200 — the empty schema dict
        passed the ``type == "json_schema"`` branch in
        ``extract_json_schema_for_guided`` but its
        ``json_schema_spec.get("schema", {})`` returned ``{}`` so the
        function fell out at ``if not schema: return None`` and the
        request proceeded with no structure enforcement."""
        from vllm_mlx.service.helpers import _validate_response_format

        with pytest.raises(HTTPException) as ei:
            _validate_response_format(
                {"type": "json_schema", "json_schema": {}}
            )
        assert ei.value.status_code == 400
        assert "non-empty" in ei.value.detail

    def test_unknown_type_xml_raises_400(self):
        """The F-013 silent-200 arm. ``type:"xml"`` used to fall
        through ``build_json_system_prompt`` (which returns ``None``
        for any non-listed type) and the request was treated as
        unconstrained text — the client received plain prose with no
        signal that their format choice was unsupported."""
        from vllm_mlx.service.helpers import _validate_response_format

        with pytest.raises(HTTPException) as ei:
            _validate_response_format({"type": "xml"})
        assert ei.value.status_code == 400
        assert "must be" in ei.value.detail
        assert "json_object" in ei.value.detail
        assert "json_schema" in ei.value.detail

    def test_empty_string_type_raises_400(self):
        """``type:""`` is a common client-side bug (env var unset →
        empty string default). Used to silent-200 via the same path
        as ``xml``."""
        from vllm_mlx.service.helpers import _validate_response_format

        with pytest.raises(HTTPException) as ei:
            _validate_response_format({"type": ""})
        assert ei.value.status_code == 400
        assert "must be" in ei.value.detail

    def test_empty_response_format_dict_raises_400(self):
        """``response_format={}`` — no ``type`` key at all. Used to
        silent-200 because ``rf_dict.get("type", "text")`` returned
        ``"text"`` and the route proceeded as unconstrained generation."""
        from vllm_mlx.service.helpers import _validate_response_format

        with pytest.raises(HTTPException) as ei:
            _validate_response_format({})
        assert ei.value.status_code == 400
        assert "type is required" in ei.value.detail

    def test_valid_text_type_passes(self):
        """``type:"text"`` is the documented default and is the
        explicit value Pydantic assigns when no ``type`` is provided
        to the typed path — must not raise."""
        from vllm_mlx.service.helpers import _validate_response_format
        from vllm_mlx.api.models import ResponseFormat

        # Both shapes — dict and Pydantic — must pass.
        _validate_response_format(ResponseFormat(type="text"))
        _validate_response_format({"type": "text"})

    def test_valid_json_object_passes(self):
        """The OpenAI ``json_object`` JSON-mode shorthand."""
        from vllm_mlx.service.helpers import _validate_response_format
        from vllm_mlx.api.models import ResponseFormat

        _validate_response_format(ResponseFormat(type="json_object"))
        _validate_response_format({"type": "json_object"})

    def test_valid_json_schema_with_spec_passes(self):
        """A fully-specified ``json_schema`` request must still work —
        the gate only fires on missing/empty inner ``json_schema``."""
        from vllm_mlx.service.helpers import _validate_response_format

        _validate_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "r",
                    "schema": {"type": "object"},
                },
            }
        )

    def test_none_response_format_passes(self):
        """``response_format`` is optional — ``None`` is the default
        and must remain a no-op."""
        from vllm_mlx.service.helpers import _validate_response_format

        _validate_response_format(None)


# ---------------------------------------------------------------------------
# F-066: image_url.url type validation (schema layer)
# ---------------------------------------------------------------------------


class TestImageUrlTypeValidation:
    """Non-string ``image_url.url`` payloads used to slip past the
    schema layer because the Union ``ImageUrl | dict | str | None``
    falls back to ``dict`` when the inner ``ImageUrl`` model rejects
    a non-string ``url``. The dict shape then crashed inside
    ``process_image_input`` → ``is_base64_image(123)`` with
    ``AttributeError: 'int' object has no attribute 'startswith'``."""

    def test_int_url_rejected_at_schema_layer(self):
        """The F-066 repro: ``image_url.url=123``. Pydantic v2 raises
        ValidationError (HTTP 422 at the FastAPI route layer)."""
        from vllm_mlx.api.models import ChatCompletionRequest

        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest(
                model="x",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": 123}}
                        ],
                    }
                ],
            )
        msg = str(ei.value)
        assert "image_url.url must be a string" in msg
        # Must NOT contain the raw Python AttributeError marker.
        assert "startswith" not in msg
        assert "'int' object" not in msg

    @pytest.mark.parametrize(
        "bad_url", [42, 3.14, [1, 2], {"a": "b"}, True]
    )
    def test_non_string_url_variants_rejected(self, bad_url):
        """Coverage for the full set of JSON-encodable non-string
        types a buggy client could send. All share the same
        ``.startswith`` hazard at the parser layer — caught at the
        schema instead."""
        from vllm_mlx.api.models import ChatCompletionRequest

        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest(
                model="x",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": bad_url},
                            }
                        ],
                    }
                ],
            )
        assert "image_url.url must be a string" in str(ei.value)

    def test_video_url_int_rejected(self):
        """Sister field — ``video_url.url`` has the same parser
        hazard (``process_video_input`` calls ``is_base64_video`` →
        ``startswith``)."""
        from vllm_mlx.api.models import ChatCompletionRequest

        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest(
                model="x",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": 99}}
                        ],
                    }
                ],
            )
        assert "video_url.url must be a string" in str(ei.value)

    def test_audio_url_int_rejected(self):
        """Sister field — ``audio_url.url`` shares the same shape."""
        from vllm_mlx.api.models import ChatCompletionRequest

        with pytest.raises(ValidationError) as ei:
            ChatCompletionRequest(
                model="x",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": 7}}
                        ],
                    }
                ],
            )
        assert "audio_url.url must be a string" in str(ei.value)

    def test_valid_string_url_dict_accepted(self):
        """A well-typed dict-shape ``image_url`` must still parse —
        the validator only rejects non-string ``url``."""
        from vllm_mlx.api.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/x.png"},
                        }
                    ],
                }
            ],
        )
        # Well-typed dict resolves to the Pydantic ContentPart variant
        # (not the dict fallback), so the ``image_url`` field is the
        # parsed ``ImageUrl`` model.
        part = req.messages[0].content[0]
        url = part.image_url.url if hasattr(part, "image_url") else part[
            "image_url"
        ]["url"]
        assert url == "https://example.com/x.png"

    def test_valid_string_url_shorthand_accepted(self):
        """The OpenAI-compat ``image_url`` can also be a plain
        string — that path bypasses the dict branch entirely and
        must still parse."""
        from vllm_mlx.api.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/x.png",
                        }
                    ],
                }
            ],
        )
        # The Pydantic ContentPart variant wins for well-typed input,
        # so the ContentPart was constructed (not the dict-fallback).
        assert req.messages[0].content[0].image_url == (
            "https://example.com/x.png"
        )

    def test_valid_text_content_unaffected(self):
        """The validator only runs on dict items with multimodal
        fields — a pure-text content list must not be affected."""
        from vllm_mlx.api.models import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        )
        assert req.messages[0].content[0].text == "hello"


class TestProcessImageInputDefenseInDepth:
    """Belt-and-suspenders: ``process_image_input`` is also called
    from the Anthropic / Responses adapters and from internal code
    paths where the schema-layer guard doesn't apply. The function's
    own ``isinstance(image, str)`` check raises a clean ValueError
    instead of bubbling the raw ``startswith`` AttributeError."""

    def test_int_image_raises_clean_value_error(self):
        from vllm_mlx.models.mllm import process_image_input

        with pytest.raises(ValueError) as ei:
            process_image_input(123)  # type: ignore[arg-type]
        assert "must be a string" in str(ei.value)
        assert "startswith" not in str(ei.value)

    def test_dict_with_int_url_raises_clean_value_error(self):
        """Dict-shape path: ``{"url": 123}`` unwraps to ``123`` and
        then trips the new isinstance gate."""
        from vllm_mlx.models.mllm import process_image_input

        with pytest.raises(ValueError) as ei:
            process_image_input({"url": 123})  # type: ignore[arg-type]
        assert "must be a string" in str(ei.value)
        assert "startswith" not in str(ei.value)
