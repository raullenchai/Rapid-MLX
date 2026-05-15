# SPDX-License-Identifier: Apache-2.0
"""
Tests for structured output (JSON Schema) functionality.

Tests the JSON parsing, validation, and response_format handling.
"""

import importlib.util
import json

import pytest

from vllm_mlx.api.models import ResponseFormat, ResponseFormatJsonSchema
from vllm_mlx.api.tool_calling import (
    build_json_system_prompt,
    extract_json_from_text,
    parse_json_output,
    validate_json_schema,
)

_MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None


class TestValidateJsonSchema:
    """Tests for validate_json_schema function."""

    def test_valid_object(self):
        """Test validation of a valid object against schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        data = {"name": "Alice", "age": 30}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is True
        assert error is None

    def test_invalid_type(self):
        """Test validation fails for wrong type."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        data = {"age": "not an integer"}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is False
        assert error is not None
        assert "integer" in error.lower() or "type" in error.lower()

    def test_missing_required(self):
        """Test validation fails for missing required field."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {}
        is_valid, error = validate_json_schema(data, schema)
        assert is_valid is False
        assert error is not None

    def test_array_validation(self):
        """Test validation of array types."""
        schema = {
            "type": "object",
            "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
        }
        # Valid
        is_valid, _ = validate_json_schema({"colors": ["red", "blue"]}, schema)
        assert is_valid is True

        # Invalid - number in array
        is_valid, _ = validate_json_schema({"colors": ["red", 123]}, schema)
        assert is_valid is False


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_pure_json(self):
        """Test extraction from pure JSON string."""
        text = '{"name": "test", "value": 42}'
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_markdown(self):
        """Test extraction from markdown code block."""
        text = """Here is the result:
```json
{"name": "test", "value": 42}
```
"""
        result = extract_json_from_text(text)
        assert result == {"name": "test", "value": 42}

    def test_json_in_plain_code_block(self):
        """Test extraction from plain code block without json marker."""
        text = """Result:
```
{"items": [1, 2, 3]}
```
"""
        result = extract_json_from_text(text)
        assert result == {"items": [1, 2, 3]}

    def test_json_embedded_in_text(self):
        """Test extraction from JSON embedded in text."""
        text = 'The answer is: {"result": true} and that concludes the analysis.'
        result = extract_json_from_text(text)
        assert result == {"result": True}

    def test_array_extraction(self):
        """Test extraction of JSON arrays."""
        text = 'Colors: ["red", "green", "blue"]'
        result = extract_json_from_text(text)
        assert result == ["red", "green", "blue"]

    def test_no_json(self):
        """Test returns None when no JSON found."""
        text = "This is just plain text with no JSON."
        result = extract_json_from_text(text)
        assert result is None

    def test_invalid_json(self):
        """Test returns None for invalid JSON."""
        text = '{"broken": json, not valid}'
        result = extract_json_from_text(text)
        assert result is None

    def test_nested_json(self):
        """Test extraction of nested JSON."""
        text = '{"outer": {"inner": {"deep": "value"}}}'
        result = extract_json_from_text(text)
        assert result == {"outer": {"inner": {"deep": "value"}}}


class TestParseJsonOutput:
    """Tests for parse_json_output function."""

    def test_no_response_format(self):
        """Test with no response_format returns original text."""
        text = "Hello, world!"
        cleaned, parsed, is_valid, error = parse_json_output(text, None)
        assert cleaned == text
        assert parsed is None
        assert is_valid is True
        assert error is None

    def test_text_format(self):
        """Test with type='text' returns original text."""
        text = "Hello, world!"
        response_format = {"type": "text"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert cleaned == text
        assert parsed is None
        assert is_valid is True

    def test_json_object_valid(self):
        """Test json_object mode extracts valid JSON."""
        text = '{"name": "test"}'
        response_format = {"type": "json_object"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": "test"}
        assert is_valid is True
        assert error is None

    def test_json_object_invalid(self):
        """Test json_object mode fails for non-JSON."""
        text = "This is not JSON"
        response_format = {"type": "json_object"}
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed is None
        assert is_valid is False
        assert "Failed to extract" in error

    def test_json_schema_valid(self):
        """Test json_schema mode validates against schema."""
        text = '{"name": "Alice", "age": 30}'
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": "Alice", "age": 30}
        assert is_valid is True
        assert error is None

    def test_json_schema_invalid(self):
        """Test json_schema mode fails validation for wrong data."""
        text = '{"name": 123}'  # name should be string
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"name": 123}
        assert is_valid is False
        assert "validation failed" in error.lower()

    def test_response_format_model(self):
        """Test with ResponseFormat Pydantic model."""
        text = '{"result": true}'
        response_format = ResponseFormat(type="json_object")
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"result": True}
        assert is_valid is True

    def test_response_format_with_json_schema_model(self):
        """Test with ResponseFormat and ResponseFormatJsonSchema models."""
        text = '{"colors": ["red", "blue"]}'
        json_schema = ResponseFormatJsonSchema(
            name="colors",
            schema={
                "type": "object",
                "properties": {
                    "colors": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["colors"],
            },
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema)
        cleaned, parsed, is_valid, error = parse_json_output(text, response_format)
        assert parsed == {"colors": ["red", "blue"]}
        assert is_valid is True


class TestBuildJsonSystemPrompt:
    """Tests for build_json_system_prompt function."""

    def test_no_format(self):
        """Test returns None for no format."""
        result = build_json_system_prompt(None)
        assert result is None

    def test_text_format(self):
        """Test returns None for text format."""
        result = build_json_system_prompt({"type": "text"})
        assert result is None

    def test_json_object(self):
        """Test prompt for json_object mode."""
        result = build_json_system_prompt({"type": "json_object"})
        assert result is not None
        assert "valid JSON" in result
        assert "only" in result.lower()

    def test_json_schema(self):
        """Test prompt for json_schema mode."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": "A person object",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "person" in result
        assert "A person object" in result
        assert "JSON Schema" in result

    def test_json_schema_model(self):
        """Test prompt with ResponseFormat model."""
        json_schema = ResponseFormatJsonSchema(
            name="output", description="Output format", schema={"type": "object"}
        )
        response_format = ResponseFormat(type="json_schema", json_schema=json_schema)
        result = build_json_system_prompt(response_format)
        assert result is not None
        assert "output" in result


class TestInjectJsonInstruction:
    """Tests for _inject_json_instruction function in server."""

    def test_inject_new_system_message(self):
        """Test injecting instruction when no system message exists."""
        from vllm_mlx.server import _inject_json_instruction

        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Return JSON only" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_append_to_existing_system(self):
        """Test appending to existing system message."""
        from vllm_mlx.server import _inject_json_instruction

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Return JSON only" in result[0]["content"]

    def test_does_not_modify_original(self):
        """Test that original messages are not modified."""
        from vllm_mlx.server import _inject_json_instruction

        original = [{"role": "user", "content": "Hello"}]
        original_content = original[0]["content"]
        result = _inject_json_instruction(original, "Return JSON only")

        # Original should be unchanged
        assert len(original) == 1
        assert original[0]["content"] == original_content


# Integration test - run only if model available
@pytest.mark.skip(reason="Requires model loaded")
class TestStructuredOutputIntegration:
    """Integration tests for structured output with real model."""

    @pytest.fixture
    def client(self):
        """Create OpenAI client pointing to local server."""
        from openai import OpenAI

        return OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    def test_json_object_mode(self, client):
        """Test json_object mode returns valid JSON."""
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "List 3 colors"}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_json_schema_mode(self, client):
        """Test json_schema mode returns valid structured data."""
        response = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "List 3 colors"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "colors",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "colors": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["colors"],
                    },
                },
            },
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        assert "colors" in data
        assert isinstance(data["colors"], list)


@pytest.mark.skipif(
    not _MLX_AVAILABLE,
    reason="routes.chat transitively imports mlx (skipped on no-MLX CI)",
)
class TestStripBackslashBeforeUnicode:
    """Cherry-picked from upstream waybarrios#525.

    ``lm-format-enforcer``'s grammar permits ``\\`` followed by any
    codepoint as a valid JSON escape, so a model emitting JSON with
    non-ASCII content can produce strings like ``"\\빠\\르\\게"``: valid
    JSON, but the decoded value carries literal backslashes that look
    like corruption to clients. The helper in ``routes/chat.py`` strips
    those spurious backslashes recursively across dicts / lists / strs.
    """

    def test_strips_backslash_before_cjk(self):
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        assert _strip_backslash_before_unicode("\\빠\\르\\게") == "빠르게"

    def test_preserves_valid_ascii_escapes(self):
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        # ``\\n`` decodes to a newline; the helper sees an actual newline
        # (non-ASCII codepoint? no — newline is ASCII), so it must remain
        # untouched.  Same for ``\\\\`` (literal backslash).
        assert _strip_backslash_before_unicode("line1\nline2") == "line1\nline2"
        assert _strip_backslash_before_unicode("path\\\\to") == "path\\\\to"

    def test_recurses_into_dict_and_list(self):
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        nested = {
            "title": "\\안\\녕",
            "items": ["a", "\\🚀", {"name": "\\한\\글"}],
        }
        cleaned = _strip_backslash_before_unicode(nested)
        assert cleaned == {
            "title": "안녕",
            "items": ["a", "🚀", {"name": "한글"}],
        }

    def test_cleans_non_ascii_keys(self):
        """Codex review round 1 finding: keys can also carry spurious
        backslashes (``lm-format-enforcer`` makes no distinction between
        JSON keys and values). The cleaner must strip both."""
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        # Key with backslashes before CJK; value also dirty.
        assert _strip_backslash_before_unicode({"\\제\\목": "\\값"}) == {"제목": "값"}
        # Nested case: the inner dict's key must also be cleaned.
        assert _strip_backslash_before_unicode({"items": [{"\\이\\름": "raul"}]}) == {
            "items": [{"이름": "raul"}]
        }

    def test_non_string_scalars_pass_through(self):
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        assert _strip_backslash_before_unicode(42) == 42
        assert _strip_backslash_before_unicode(True) is True
        assert _strip_backslash_before_unicode(None) is None

    def test_emoji_with_surrogate_pair(self):
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        # Emoji past U+FFFF — the regex matches by codepoint, not by
        # UTF-16 surrogate, so the single backslash before the emoji
        # should still be stripped.
        assert _strip_backslash_before_unicode("hi \\🎉 there") == "hi 🎉 there"

    def test_key_collision_logs_and_keeps_first(self, caplog):
        """Codex review round 2 finding: two dirty keys can collapse to
        the same clean key (``"\\한"`` and ``"한"`` both → ``"한"``).
        Silently dropping one is data loss; we keep the first occurrence
        and log a warning."""
        import logging

        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        with caplog.at_level(logging.WARNING, logger="vllm_mlx.routes.chat"):
            cleaned = _strip_backslash_before_unicode({"\\한": 1, "한": 2})
        assert cleaned == {"한": 1}
        assert any("key collision" in rec.message for rec in caplog.records)

    def test_end_to_end_response_format_cleanup(self):
        """Integration: drive the exact chain ``routes/chat.py`` runs on
        a response_format='json_object' request whose model output
        contains spurious ``\\`` before non-ASCII chars. This is the
        regression that exercises wiring — if a future refactor moves
        the helper to a different module or skips it, this test fails.

        Codex review round 2 asked for this end-to-end coverage so the
        unit test alone isn't the only safeguard against the wiring
        getting accidentally dropped."""
        import json as _json

        from vllm_mlx.api.tool_calling import parse_json_output
        from vllm_mlx.routes.chat import _strip_backslash_before_unicode

        # Simulated model output: looks like JSON, but every CJK char
        # carries a leading backslash (lm-format-enforcer behavior).
        # Use double-escaped backslashes because the model emits the
        # literal characters ``\``, ``빠``, ``\``, ``르``, …
        raw_model_text = (
            '{"message": "안녕 \\\\빠\\\\르\\\\게", "name": "\\\\한\\\\글"}'
        )
        response_format = {"type": "json_object"}

        _, parsed_json, is_valid, _ = parse_json_output(raw_model_text, response_format)
        assert is_valid, "json_object parser should accept this input"
        assert parsed_json is not None

        # The chat route now applies the helper before re-serializing.
        cleaned = _strip_backslash_before_unicode(parsed_json)
        final = _json.dumps(cleaned, ensure_ascii=False)

        # Final body is what the client sees in ``message.content``.
        # No spurious backslashes before the CJK characters.
        assert "\\빠" not in final
        assert "\\한" not in final
        # CJK content survives intact (not double-escaped to \uXXXX).
        assert "빠르게" in final
        assert "한글" in final


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
