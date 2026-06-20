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
        from vllm_mlx.api.models import ResponseFormat
        from vllm_mlx.service.helpers import _validate_response_format

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
            _validate_response_format({"type": "json_schema", "json_schema": {}})
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
        from vllm_mlx.api.models import ResponseFormat
        from vllm_mlx.service.helpers import _validate_response_format

        # Both shapes — dict and Pydantic — must pass.
        _validate_response_format(ResponseFormat(type="text"))
        _validate_response_format({"type": "text"})

    def test_valid_json_object_passes(self):
        """The OpenAI ``json_object`` JSON-mode shorthand."""
        from vllm_mlx.api.models import ResponseFormat
        from vllm_mlx.service.helpers import _validate_response_format

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
                        "content": [{"type": "image_url", "image_url": {"url": 123}}],
                    }
                ],
            )
        msg = str(ei.value)
        assert "image_url.url must be a string" in msg
        # Must NOT contain the raw Python AttributeError marker.
        assert "startswith" not in msg
        assert "'int' object" not in msg

    @pytest.mark.parametrize("bad_url", [42, 3.14, [1, 2], {"a": "b"}, True])
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
                        "content": [{"type": "video_url", "video_url": {"url": 99}}],
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
                        "content": [{"type": "audio_url", "audio_url": {"url": 7}}],
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
        url = (
            part.image_url.url
            if hasattr(part, "image_url")
            else part["image_url"]["url"]
        )
        assert url == "https://example.com/x.png"

    def test_bare_string_image_url_rejected(self):
        """F-065: the bare-string ``image_url`` shorthand was
        previously accepted by the union arm
        (``ImageUrl | dict | str | None``) and then silently
        dropped by the multimodal preprocessor (which unwrapped
        ``image["url"]`` from the dict shape but had no fallback
        for the bare-string form). The model received only the
        text and hallucinated ("the image is blank").

        Per OpenAI spec the ``image_url`` slot MUST be an object
        with a required ``url`` field; reject the bare-string form
        at the schema layer with a clean 422 so the client sees
        the actual mismatch instead of a silent-correctness bug.
        """
        from pydantic import ValidationError

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
                                "image_url": "https://example.com/x.png",
                            }
                        ],
                    }
                ],
            )
        msg = str(ei.value)
        assert "must be an object" in msg
        assert "url" in msg

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
    instead of bubbling the raw ``startswith`` AttributeError.

    Critical preservation contract (codex r2): the existing dict
    unwrapping at the top of ``process_image_input`` (``{"url": "..."}``
    → ``url``) MUST still run before the type guard so valid
    dict-shaped callers keep working — the guard only fires on the
    POST-unwrap value, never on the original dict. Tests below pin
    both branches.
    """

    def test_int_image_raises_clean_value_error(self):
        from vllm_mlx.models.mllm import process_image_input

        with pytest.raises(ValueError) as ei:
            process_image_input(123)  # type: ignore[arg-type]
        assert "must be a string" in str(ei.value)
        # Must surface the type name in the error for client debuggability
        # (codex r2 NIT: prior assertion didn't pin the ``got X`` shape).
        assert "got int" in str(ei.value)
        assert "startswith" not in str(ei.value)

    def test_dict_with_int_url_raises_clean_value_error(self):
        """Dict-shape path: ``{"url": 123}`` is unwrapped at the
        function's top (line ~525) to ``image = 123``, then the
        type guard catches the non-string and raises. The error
        message names the post-unwrap type (``int``), not ``dict``,
        which is the load-bearing test (codex r2 BLOCKING)."""
        from vllm_mlx.models.mllm import process_image_input

        with pytest.raises(ValueError) as ei:
            process_image_input({"url": 123})  # type: ignore[arg-type]
        assert "must be a string" in str(ei.value)
        assert "got int" in str(ei.value)
        assert "startswith" not in str(ei.value)

    def test_dict_with_valid_string_url_still_processed(self):
        """Codex r2 BLOCKING coverage: valid ``{"url": "<str>"}``
        dict shape MUST still unwrap and proceed past the type guard.
        The downstream call hits ``Path.exists()`` and raises
        ``Cannot process image`` because the path is fake — that's
        downstream behavior, not the type guard rejecting the dict."""
        from vllm_mlx.models.mllm import process_image_input

        # Pick a path that's <4096 chars and definitely doesn't exist
        # — proves we got past the type guard into the body of the
        # function.
        with pytest.raises(ValueError) as ei:
            process_image_input({"url": "/nonexistent/__rapid_mlx_test_path__.png"})
        # The error must be the downstream "Cannot process image"
        # marker (path not found), NOT the type guard's "must be a
        # string" — proves the dict was unwrapped and the unwrapped
        # string was accepted by the type guard.
        assert "Cannot process image" in str(ei.value)
        assert "must be a string" not in str(ei.value)

    def test_nested_dict_url_unwrapped_correctly(self):
        """The function also handles the nested ``{"url": {"url":
        "..."}}`` shape (line ~527-528). Pin that the unwrap still
        works through both levels."""
        from vllm_mlx.models.mllm import process_image_input

        with pytest.raises(ValueError) as ei:
            process_image_input({"url": {"url": "/nonexistent/__nested_test__.png"}})
        # Same as above — should fall through to the path-not-found
        # error, not be caught by the type guard.
        assert "Cannot process image" in str(ei.value)
        assert "must be a string" not in str(ei.value)
